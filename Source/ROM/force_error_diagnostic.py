"""Read-only decomposition of the archived force errors and map sensitivities.

Inputs use the shared Source_Data resolver; set KVN_SOURCE_DATA to select a dataset.
"""

# Public Source Data mapping (single canonical file per published input).
import sys as _kvn_path_sys
from pathlib import Path as _kvn_path_Path

_kvn_path_sys.path.insert(0, str(_kvn_path_Path(__file__).resolve().parents[2]))
from Source.data_locations import published_path, shared_data
from pathlib import Path
import argparse, csv, json
import numpy as np
from scipy.integrate import solve_ivp


def main(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    read = lambda p: list(csv.DictReader(p.open()))
    coef = read(
        published_path("Figure/F4/data/plot_values/Re100/force_coefficients.csv")
    )
    cd = {r["term"]: float(r["coefficient"]) for r in coef if r["observable"] == "C_D"}
    cl = {r["term"]: float(r["coefficient"]) for r in coef if r["observable"] == "C_L"}
    rows = read(published_path("Data/Force_Sampling/force_815_shots.csv"))
    times = np.array([float(r["time"]) for r in rows])
    fem = np.array([[float(r[f"{k}_FEM"]) for k in ["C_D", "C_L"]] for r in rows])
    qforce = np.array([[float(r[f"{k}_exact"]) for k in ["C_D", "C_L"]] for r in rows])
    noise = np.array(
        [[float(r[f"{k}_standard_error"]) for k in ["C_D", "C_L"]] for r in rows]
    )
    data = shared_data("CFD_ROM", "CFD_ROM", "Re100", "data")
    hist = np.load(data / "direct_quantum_history.npz")
    ref = np.load(data / "unified_reference.npz")
    conf = json.loads((data / "model_configuration.json").read_text())["parameters"]
    R = conf["R"]
    zscale = conf["c2"] * R * R

    def force(x, y, z):
        return np.stack(
            [
                cd["1"]
                + cd["z"] * z
                + cd["x^2"] * x * x
                + cd["xy"] * x * y
                + cd["y^2"] * y * y
                + cd["z^2"] * z * z,
                cl["x"] * x + cl["y"] * y + cl["xz"] * x * z + cl["yz"] * y * z,
            ],
            axis=-1,
        )

    def normalized(a):
        return a[..., 0] / R, a[..., 1] / R, (a[..., 2] - conf["c0"]) / zscale

    stored = np.load(data / "new_model_inputs.npz")
    a0 = np.array(
        [
            np.interp(
                times[0], stored["all_times"], stored["all_projected_coordinates"][:, j]
            )
            for j in range(3)
        ]
    )
    acfd = np.column_stack(
        [
            np.interp(
                times, stored["all_times"], stored["all_projected_coordinates"][:, j]
            )
            for j in range(3)
        ]
    )
    fcfd = force(*normalized(acfd))
    model = np.load(published_path("Data/Model/candidate_K2.npz"))
    C = model["physical_coefficients"]
    powers = model["powers"]

    def rhs(t, a):
        return np.prod(a[None, :] ** powers, axis=1) @ C

    sol = solve_ivp(
        rhs,
        (times[0], times[-1]),
        a0,
        t_eval=times,
        method="DOP853",
        rtol=2e-12,
        atol=2e-14,
    )
    assert sol.success
    fthree = force(*normalized(sol.y.T))
    mu, omega, gamma = [conf[k] for k in ["mu", "omega", "gamma_normalized"]]
    x0, y0, _ = normalized(a0)
    r0 = np.hypot(x0, y0)
    theta0 = np.arctan2(y0, x0)
    elapsed = times - times[0]

    def radial_path(initial):
        initial = np.asarray(initial)
        den = initial[None, :] ** 2 + (1 - initial[None, :] ** 2) * np.exp(
            -2 * mu * elapsed[:, None]
        )
        r = initial[None, :] / np.sqrt(den)
        phase = (
            theta0
            + omega * elapsed[:, None]
            + gamma * (elapsed[:, None] + np.log(den) / (2 * mu))
        )
        return r, phase

    rpoint, theta = radial_path(np.array([r0]))
    fpoint = force(rpoint * np.cos(theta), rpoint * np.sin(theta), rpoint * rpoint)[
        :, 0, :
    ]
    r, theta = radial_path(ref["nodes32"])
    continuum = np.einsum(
        "tno,n->to",
        force(r * np.cos(theta), r * np.sin(theta), r * r),
        ref["weights32"],
    )
    r16, theta16 = radial_path(ref["nodes16"])
    continuum16 = np.einsum(
        "tno,n->to",
        force(r16 * np.cos(theta16), r16 * np.sin(theta16), r16 * r16),
        ref["weights16"],
    )
    quadrature_change = float(np.max(abs(continuum - continuum16)))
    assert quadrature_change < 3e-12
    accumulated = theta - ref["known_phase"][:, None]
    moments = np.column_stack(
        [
            (r * np.cos(accumulated)) @ ref["weights32"],
            (r * np.sin(accumulated)) @ ref["weights32"],
            (r * r) @ ref["weights32"],
        ]
    )
    reference_moment_difference = float(np.max(abs(moments - ref["continuum_moments"])))
    assert reference_moment_difference < 2e-12
    components = np.stack(
        [
            fcfd - fem,
            fthree - fcfd,
            fpoint - fthree,
            continuum - fpoint,
            qforce - continuum,
        ],
        axis=1,
    )
    names = [
        "map_and_projected_reference",
        "three_mode_dynamics",
        "shift_closure_and_angular_average",
        "ensemble_vs_single_initial_state",
        "circuit_vs_continuum_ensemble",
    ]
    residual = qforce - fem
    identity = float(np.max(abs(components.sum(axis=1) - residual)))
    assert identity < 3e-15
    # Exact response of saved circuit expectations to a uniform phase offset.
    state = hist["direct_history"].copy()
    state /= np.linalg.norm(state, axis=(1, 2))[:, None, None]
    radius = hist["r"]
    known = ref["known_phase"][:, None]
    X = np.array([[0, 1], [1, 0]], complex)
    Y = np.array([[0, -1j], [1j, 0]], complex)
    p = np.sum(abs(state) ** 2, axis=1)
    first = (
        np.einsum("tij,ik,tkj->tj", state.conj(), np.kron(np.eye(2), X), state).real
        + 1j
        * np.einsum("tij,ik,tkj->tj", state.conj(), np.kron(Y, np.eye(2)), state).real
    ) * np.exp(1j * known)
    second = 4 * state[:, 0, :].conj() * state[:, 3, :] * np.exp(2j * known)
    A = cd["z"] + (cd["x^2"] + cd["y^2"]) / 2
    B = (cd["x^2"] - cd["y^2"]) / 2
    C2 = cd["xy"] / 2
    dbase = p @ (cd["1"] + A * radius**2 + cd["z^2"] * radius**4)
    dphase = (B * second.real + C2 * second.imag) @ (radius**2)
    lvalue = first.real @ (radius * (cl["x"] + cl["xz"] * radius**2)) + first.imag @ (
        radius * (cl["y"] + cl["yz"] * radius**2)
    )
    np.testing.assert_allclose(
        np.column_stack([dbase + dphase, lvalue]), qforce, rtol=0, atol=3e-13
    )
    ddphase = 2 * (-B * second.imag + C2 * second.real) @ (radius**2)
    dlphase = -first.imag @ (
        radius * (cl["x"] + cl["xz"] * radius**2)
    ) + first.real @ (radius * (cl["y"] + cl["yz"] * radius**2))
    rms = lambda a: np.sqrt(np.mean(np.asarray(a) ** 2, axis=0))
    summary = {}
    for j, key in enumerate(["C_D", "C_L"]):
        norm = float(rms(fem[:, j]))
        bias = float(rms(residual[:, j]))
        sampling = float(rms(noise[:, j]))
        total = float(np.hypot(bias, sampling))
        summary[key] = {
            "FEM_RMS_normalizer": norm,
            "noiseless_absolute_RMS": bias,
            "sampling_absolute_RMS": sampling,
            "combined_absolute_RMS": total,
            "noiseless_relative_RMS_percent": 100 * bias / norm,
            "sampling_relative_RMS_percent": 100 * sampling / norm,
            "combined_relative_RMS_percent": 100 * total / norm,
            "circuit_vs_continuum_relative_RMS_percent_in_FEM_norm": 100
            * float(rms(qforce[:, j] - continuum[:, j]))
            / norm,
            "residual_components": {
                name: {
                    "absolute_RMS": float(rms(components[:, k, j])),
                    "projection_share_of_total_noiseless_MSE": float(
                        np.mean(components[:, k, j] * residual[:, j])
                        / np.mean(residual[:, j] ** 2)
                    ),
                }
                for k, name in enumerate(names)
            },
        }
    report = {
        "source": "saved force maps, joint state, FEM data and frozen models; no refitting",
        "CD_radial_coefficients": {"constant": cd["1"], "r2": A, "r4": cd["z^2"]},
        "CD_phase_coefficients": {"r2_cos2theta": B, "r2_sin2theta": C2},
        "CL_phase_coefficients": {
            "r_costheta": cl["x"],
            "r_sintheta": cl["y"],
            "r3_costheta": cl["xz"],
            "r3_sintheta": cl["yz"],
        },
        "CD_radial_part_RMS": float(rms(dbase)),
        "CD_angular_part_RMS": float(rms(dphase)),
        "uniform_phase_offset_sensitivity_RMS_per_radian": {
            "C_D": float(rms(ddphase)),
            "C_L": float(rms(dlphase)),
        },
        "force_errors": summary,
        "additive_error_identity_max_residual": identity,
        "force_reference_16_vs_32_quadrature_max_change": quadrature_change,
        "archived_continuum_moments_max_difference": reference_moment_difference,
        "interpretation_limits": "The residual-component path is a diagnostic, not a unique causal attribution. The map/reference term includes modal projection, force-fit mismatch and time interpolation. Component RMS values do not add; cross terms may cancel. The uniform-phase sensitivity does not assert that all observed error is a phase error.",
    }
    (output / "force_error_structure.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    outrows = []
    for i, time in enumerate(times):
        row = {"time": float(time)}
        for j, key in enumerate(["C_D", "C_L"]):
            row.update({key + "_FEM": fem[i, j], key + "_exact_circuit": qforce[i, j]})
            for k, name in enumerate(names):
                row[key + "_" + name] = components[i, k, j]
        outrows.append(row)
    with (output / "force_error_components.csv").open("w", newline="") as f:
        cw = csv.DictWriter(f, fieldnames=list(outrows[0]))
        cw.writeheader()
        cw.writerows(outrows)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.output)
