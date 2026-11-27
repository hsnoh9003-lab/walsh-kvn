"""Same-data comparison; no model selection or published-file replacement."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import csv,json
import numpy as np
from scipy.integrate import solve_ivp
from model import load,raw_drift
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results/physical'

def save(name,rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def simulate(model,initial,times):
    sol=solve_ivp(lambda t,a:raw_drift(model,a),[times[0],times[-1]],initial,t_eval=times,
                  method='DOP853',rtol=1e-11,atol=1e-13)
    assert sol.success;return sol.y.T

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    d=np.load(ROOT/'inputs/physical_model_inputs.npz');mask=np.tile(d['fluid_mask'].ravel(),2)
    modes=d['Phi'][:,mask];base=d['base'][mask];truth=d['evaluation_fields'].reshape(-1,len(mask))[:,mask]
    norm=np.linalg.norm(truth,axis=1);times=d['evaluation_times'];x0=d['evaluation_coordinates'][0]
    rows=[];pred={};aggregate=[]
    for name in ['deployed','candidate']:
        model=load(name);pred[name]=simulate(model,x0,times)
        if name=='deployed':
            difference=float(np.max(abs(pred[name]-d['quadratic_coordinates'])))
            assert difference<1e-7,difference
        fields=pred[name]@modes+base;radial=d['cubic_coordinates']@modes+base
        field=100*np.linalg.norm(fields-truth,axis=1)/norm
        closure=100*np.linalg.norm(fields-radial,axis=1)/norm
        for i,t in enumerate(times):
            rows.append(dict(model=name,time=float(t),field_vs_FEM_percent=float(field[i]),
                radial_vs_three_mode_FEM_norm_percent=float(closure[i]),
                a1=float(pred[name][i,0]),a2=float(pred[name][i,1]),a3=float(pred[name][i,2])))
        aggregate.append(dict(model=name,window='Fig2_40_to_49.829',
            terminal_field_vs_FEM_percent=float(field[-1]),
            aggregate_field_vs_FEM_percent=float(100*np.linalg.norm((fields-truth)[1:])/np.linalg.norm(truth[1:])),
            terminal_radial_vs_three_mode_percent=float(closure[-1]),
            max_radial_vs_three_mode_percent=float(closure.max())))
    save('figure2_comparison.csv',rows);save('figure2_metrics.csv',aggregate)
    observed=d['all_coordinates'];observed_times=d['all_coordinate_times']
    windows=[('calibration_reproduction',10.08,79.92),('growth',29.8,49.82906260206943),
             ('late',40.,49.82906260206943),('post_fit',79.92,90.),('extended',20.,90.)]
    diagnostics=[];trajectory=[]
    for label,start,end in windows:
        ts=np.unique(np.r_[start,observed_times[(observed_times>start)&(observed_times<end)],end])
        reference=np.column_stack([np.interp(ts,observed_times,observed[:,j]) for j in range(3)])
        reference_field=reference@modes+base
        for name in ['deployed','candidate']:
            prediction=simulate(load(name),reference[0],ts)
            error=100*np.linalg.norm((prediction-reference)@modes,axis=1)/np.linalg.norm(reference_field,axis=1)
            diagnostics.append(dict(model=name,window=label,start=start,end=end,
                projected_field_RMS_percent=float(100*np.linalg.norm(((prediction-reference)@modes)[1:])/np.linalg.norm(reference_field[1:])),
                terminal_projected_field_error_percent=float(error[-1]),maximum_projected_field_error_percent=float(error.max())))
            for t,point,ref,e in zip(ts,prediction,reference,error):
                trajectory.append(dict(model=name,window=label,time=float(t),projected_field_error_percent=float(e),
                    **{f'a{j+1}':float(point[j]) for j in range(3)},**{f'reference_a{j+1}':float(ref[j]) for j in range(3)}))
    save('window_metrics.csv',diagnostics);save('window_trajectories.csv',trajectory)
    np.savez_compressed(OUT/'figure2_fields.npz',time=times,deployed=pred['deployed'],candidate=pred['candidate'],
        radial=d['cubic_coordinates'],reference=d['evaluation_coordinates'])
    delta=load('candidate').physical_coefficients-load('deployed').physical_coefficients
    report=dict(original_quadratic_coordinate_recovery_max_difference=difference,
        changed_physical_coefficients=int(np.count_nonzero(abs(delta)>1e-12)),
        max_coefficient_change_by_equation=abs(delta).max(axis=0).tolist(),
        figure2=aggregate,window_diagnostics=diagnostics,
        scope='Fig2 has full masked FEM fields. Other windows compare only fixed-basis CFD projections, excluding projection error.')
    (OUT/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
