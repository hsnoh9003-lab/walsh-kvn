"""Calibrate and validate cylinder force observables on the frozen wake basis.

Force-map calibration uses t*=10..60; t*>60..90 is reserved for force-map
validation. The existing spatial basis and dynamical ROM are kept frozen.
The existing basis was itself constructed using later CFD, so this split is
not a fully prospective end-to-end forecast validation.
"""
import os
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
from pathlib import Path
import argparse
import json
import hashlib
import shutil
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def fit_scaled(A, b):
    scale = np.sqrt(np.mean(A * A, axis=0))
    if np.any(scale == 0):
        raise ValueError("Zero regression column")
    coeff_scaled, _, rank, singular = np.linalg.lstsq(A / scale, b, rcond=None)
    return coeff_scaled / scale, {
        "rank": int(rank),
        "columns": int(A.shape[1]),
        "rms_scaled_condition_number": float(singular[0] / singular[-1]),
        "singular_values": singular.tolist(),
    }


def drag_features(q):
    x, y, z = np.asarray(q).T
    return np.column_stack((z, x*x, x*y, y*y, z*z))


def lift_features(q):
    x, y, z = np.asarray(q).T
    return np.column_stack((x, y, x*z, y*z))


def metrics(ref, pred):
    error = pred - ref
    rmse = np.sqrt(np.mean(error**2))
    rms_ref = np.sqrt(np.mean(ref**2))
    centered = np.sqrt(np.mean((ref - np.mean(ref))**2))
    return {
        "rmse": float(rmse),
        "bias": float(np.mean(error)),
        "max_absolute_error": float(np.max(np.abs(error))),
        "rms_reference": float(rms_ref),
        "nrmse_percent_of_reference_rms": float(100*rmse/rms_ref),
        "nrmse_percent_of_reference_std": float(100*rmse/centered),
        "r_squared": float(1-np.sum(error**2)/np.sum((ref-np.mean(ref))**2)),
    }


def late_cycle_statistics(dense):
    late = dense.loc[dense["time"] >= 60].copy()
    t, cl = late["time"].to_numpy(), late["C_L"].to_numpy()
    indices = np.flatnonzero((cl[:-1] < 0) & (cl[1:] >= 0))
    crossings = t[indices] - cl[indices]*(t[indices+1]-t[indices])/(cl[indices+1]-cl[indices])
    if len(crossings) < 3:
        raise ValueError("Insufficient late lift periods")
    a, b = float(crossings[0]), float(crossings[-1])
    time = np.r_[a, t[(t>a)&(t<b)], b]
    out = {"time_start": a, "time_end": b, "complete_cycles": int(len(crossings)-1),
           "lift_upward_zero_crossings": crossings.tolist(),
           "Strouhal": float((len(crossings)-1)/(b-a))}
    for col in ("C_D", "C_L", "C_D_pressure", "C_D_viscous", "C_L_pressure", "C_L_viscous"):
        v = np.interp(time, dense["time"], dense[col])
        avg = np.trapezoid(v, time)/(b-a)
        out[col] = {"mean": float(avg),
                    "rms": float(np.sqrt(np.trapezoid(v*v,time)/(b-a))),
                    "standard_deviation": float(np.sqrt(np.trapezoid((v-avg)**2,time)/(b-a))),
                    "min": float(np.min(v)), "max": float(np.max(v)),
                    "half_peak_to_peak": float(np.ptp(v)/2)}
    return out


def run(source, cfd, archived_cfd, out):
    out.mkdir(parents=True, exist_ok=True)
    model = np.load(source/"new_model_inputs.npz")
    physical = np.load(source/"physical_model_inputs.npz")
    raw = np.load(archived_cfd)
    replay = np.load(cfd/"snapshots.npz")
    exact = {key: bool(np.array_equal(raw[key], replay[key])) for key in raw.files}
    verification = {
        "archive_snapshot_sha256": hashlib.sha256(archived_cfd.read_bytes()).hexdigest(),
        "all_original_arrays_bitwise_equal": bool(all(exact.values())),
        "per_array_equal": exact,
        "max_velocity_snapshot_difference": float(np.max(np.abs(raw["fields"].astype(float)-replay["fields"].astype(float)))),
    }
    if not verification["all_original_arrays_bitwise_equal"]:
        raise ValueError("Replayed flow does not match the frozen CFD")
    times = model["all_times"]
    if not np.array_equal(times, raw["times"]):
        raise ValueError("Modal/CFD time mismatch")
    phi = model["Phi"]
    # Frozen modes are scaled and are not Euclidean-unit-normalized.
    projected = np.linalg.solve(phi@phi.T, ((raw["fields"].reshape(len(times),-1).astype(float)-model["base"])@phi.T).T).T
    verification["projection_max_difference"] = float(np.max(np.abs(projected-model["all_projected_coordinates"])))
    if verification["projection_max_difference"] > 1e-10:
        raise ValueError("Modal projection convention mismatch")
    R = float(physical["R"])
    c0, c2 = physical["closure"]
    scales = np.array([R,R,c2*R*R])
    q = projected.copy()
    q[:,2] -= c0
    q /= scales
    force = pd.read_csv(cfd/"forces.csv")
    force = force.loc[force["pressure_valid"].astype(str).str.lower().isin(["true","1"])].copy()
    # Original CFD samples and force records share exact solver steps.
    sample_steps = np.rint(times/.02).astype(int)
    ref = force.set_index("step").reindex(sample_steps)
    cd = ref["C_D"].to_numpy()
    cl = ref["C_L"].to_numpy()
    finite = np.isfinite(cd)&np.isfinite(cl)
    train = (times>=10)&(times<=60)&finite
    valid = (times>60)&(times<=90)&finite
    cd0 = json.loads((cfd/"steady_force.json").read_text())["C_D"]
    d, drag_design = fit_scaled(drag_features(q[train]), cd[train]-cd0)
    ell, lift_design = fit_scaled(lift_features(q[train]), cl[train])
    pred_cd = cd0 + drag_features(q)@d
    pred_cl = lift_features(q)@ell
    terms_d = ["1","z","x^2","xy","y^2","z^2"]
    terms_l = ["x","y","xz","yz"]
    pd.DataFrame([
        *({"observable":"C_D","term":t,"coefficient":float(v)} for t,v in zip(terms_d,np.r_[cd0,d])),
        *({"observable":"C_L","term":t,"coefficient":float(v)} for t,v in zip(terms_l,ell)),
    ]).to_csv(out/"force_coefficients.csv", index=False)
    table = pd.DataFrame({"time":times,"x":q[:,0],"y":q[:,1],"z":q[:,2],
                          "C_D_CFD":cd,"C_L_CFD":cl,"C_D_force_model":pred_cd,
                          "C_L_force_model":pred_cl,"split":np.where(train,"force_fit",np.where(valid,"force_holdout","unused"))})
    table.to_csv(out/"force_model_timeseries.csv",index=False)
    force.to_csv(out/"forces_cfd.csv",index=False)
    for name in ["new_model_inputs.npz","physical_model_inputs.npz","model_configuration.json"]:
        shutil.copy2(source/name,out/name)
    shutil.copy2(cfd/"steady_force.json",out/"steady_force.json")
    scores = []
    for tag,mask in [("force_fit_10_to_60",train),("force_holdout_after60_to90",valid)]:
        for observable,y,p in [("C_D",cd,pred_cd),("C_L",cl,pred_cl)]:
            scores.append({"comparison":tag,"observable":observable,"samples":int(mask.sum()),
                           "time_start":float(times[mask][0]),"time_end":float(times[mask][-1]),
                           **metrics(y[mask],p[mask])})
    pd.DataFrame(scores).to_csv(out/"force_model_metrics.csv",index=False)
    # The already-frozen three-state ODE is a separate dynamical comparison.
    C, powers = model["C"], model["polynomial_exponents"]
    i0 = int(np.flatnonzero(times>=29.04-1e-10)[0])
    ode_t = times[i0:]
    def rhs(t,a):
        return np.prod(a[None,:]**powers,axis=1)@C
    sol=solve_ivp(rhs,(float(ode_t[0]),float(ode_t[-1])),projected[i0],
                  t_eval=ode_t,method="DOP853",rtol=1e-11,atol=1e-12)
    if not sol.success:
        raise ValueError(sol.message)
    ode_q=sol.y.T.copy();ode_q[:,2]-=c0;ode_q/=scales
    ode_cd=cd0+drag_features(ode_q)@d
    ode_cl=lift_features(ode_q)@ell
    pd.DataFrame({"time":ode_t,"x_ROM":ode_q[:,0],"y_ROM":ode_q[:,1],"z_ROM":ode_q[:,2],
                  "C_D_ROM":ode_cd,"C_L_ROM":ode_cl,"C_D_CFD":cd[i0:],"C_L_CFD":cl[i0:]}).to_csv(out/"classical_rom_forces.csv",index=False)
    ode_scores=[]
    for tag,mask in [("growth_29p04_to49p68",ode_t<=49.82906260206943),
                     ("late_after60_to90",ode_t>60),
                     ("full_29p04_to90",np.ones(len(ode_t),dtype=bool))]:
        for observable,ref_,pred_ in [("C_D",cd[i0:],ode_cd),("C_L",cl[i0:],ode_cl)]:
            ode_scores.append({"comparison":tag,"observable":observable,"samples":int(mask.sum()),
                               "time_start":float(ode_t[mask][0]),"time_end":float(ode_t[mask][-1]),
                               **metrics(ref_[mask],pred_[mask])})
    pd.DataFrame(ode_scores).to_csv(out/"classical_rom_metrics.csv",index=False)
    stats=late_cycle_statistics(force)
    result = {
        "normalization":{"Re":100,"rho":1,"U_infinity":1,"diameter":1,
                         "time":"t*=t U_infinity / D","C_D":"F_x per unit span / (0.5 rho U_infinity^2 D)",
                         "C_L":"F_y per unit span / (0.5 rho U_infinity^2 D)",
                         "x":"raw_a1/R","y":"raw_a2/R","z":"(raw_a3-c0)/(c2 R^2)",
                         "raw_coordinate_scales":scales.tolist(),"c0":float(c0)},
        "fit":{"fixed_steady_drag":cd0,"drag_terms":terms_d,"drag_coefficients":np.r_[cd0,d].tolist(),
               "lift_terms":terms_l,"lift_coefficients":ell.tolist(),"drag_design":drag_design,"lift_design":lift_design,
               "fit_window":[10,60],"force_holdout_window":[60,90],
               "method":"RMS-column-scaled ordinary least squares; steady drag fixed from independent surface stress integration",
               "representation_note":"Fixed existing modes: oscillatory POD t*=20..80; shift uses late complete cycles through t*=87.334. Holdout evaluates force-map coefficients conditional on this pre-existing basis, not a fully prospective end-to-end forecast."},
        "metrics":scores,
        "classical_rom_metrics":ode_scores,
        "late_complete_cycle_statistics":stats,
        "verification":verification,
        "limitations":["Surface forces are evaluated on the existing coarse FEM discretization; no new transient force mesh convergence study is claimed.",
                       "Force polynomial is calibrated on a single deterministic CFD trajectory; uncertain off-trajectory ensembles can extend outside that calibration support.",
                       "The force map does not change the frozen dynamical ODE or the quantum propagation circuit.",
                       "Perturbed initial t*=0 has no solved pressure; total initial force is not fabricated."],
    }
    (out/"force_results.json").write_text(json.dumps(result,indent=2)+"\n")
    plt.rcParams.update({"font.family":"serif","mathtext.fontset":"stix","font.size":9,
                         "axes.linewidth":0.6,"xtick.direction":"in","ytick.direction":"in",
                         "figure.dpi":150,"savefig.dpi":300,"savefig.bbox":"tight"})
    fig,axes=plt.subplots(2,2,figsize=(7,4.5),layout="constrained",gridspec_kw={"width_ratios":[1.55,1.]})
    blue="#2166ac"
    for i,(label,rf,pr) in enumerate([("C_D",cd,pred_cd),("C_L",cl,pred_cl)]):
        ax=axes[i,0]
        ax.plot(force["time"],force[label],color="#222222",lw=.8,label="CFD surface force")
        mask=times>=10
        ax.plot(times[mask],pr[mask],color=blue,lw=.9,ls="--",label="Three-mode force model")
        ax.axvspan(10,60,color="#eeeeee",alpha=.75,zorder=-5)
        ax.axvline(60,color="#888888",lw=.6,ls=":")
        ax.set(xlim=(0,90),ylabel="$"+label+"$")
        if i==0:
            ax.legend(frameon=False,fontsize=8,loc="upper left")
        ax=axes[i,1]
        ax.plot(force["time"],force[label],color="#222222",lw=.8)
        ax.plot(times[valid],pr[valid],color=blue,lw=.9,ls="--")
        ax.set_xlim(60,78)
        in_zoom=(force["time"]>=60)&(force["time"]<=78)
        pred_zoom=(times>60)&(times<=78)
        zoom_values=np.r_[force.loc[in_zoom,label].to_numpy(),pr[pred_zoom]]
        lo,hi=np.min(zoom_values),np.max(zoom_values)
        pad=.12*(hi-lo)
        ax.set_ylim(lo-pad,hi+pad)
        ax.set_ylabel("$"+label+"$")
    axes[0,0].set_title("Growth and saturation",fontsize=9)
    axes[0,1].set_title("Force-map holdout",fontsize=9)
    for ax in axes[1]: ax.set_xlabel(r"$tU_\infty/D$")
    for letter,ax in zip("abcd",axes.flatten()):
        ax.text(-.13,1.035,letter,transform=ax.transAxes,fontweight="bold",fontsize=10)
    fig.savefig(out/"cylinder_drag_lift.pdf")
    fig.savefig(out/"cylinder_drag_lift.png")
    plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(7,4),sharex=True,layout="constrained")
    for ax,label,pred in zip(axes,["C_D","C_L"],[ode_cd,ode_cl]):
        ax.plot(force["time"],force[label],color="#222222",lw=.8,label="CFD surface force")
        ax.plot(ode_t,pred,color="#b2182b",lw=.9,ls="--",label="Frozen three-state ROM + force map")
        ax.set(xlim=(ode_t[0],90),ylabel="$"+label+"$")
    axes[0].legend(frameon=False,fontsize=8)
    axes[1].set_xlabel(r"$tU_\infty/D$")
    fig.savefig(out/"classical_rom_drag_lift.pdf")
    fig.savefig(out/"classical_rom_drag_lift.png")
    plt.close(fig)
    print(json.dumps({"fit":result["fit"],"metrics":scores,"late_statistics":stats,"verification":verification},indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--cfd",type=Path,required=True)
    p.add_argument("--archived-cfd",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    run(a.source,a.cfd,a.archived_cfd,a.out)
