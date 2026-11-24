"""Replay the selected 12-coefficient model and rebuild Figure 4e source data.

No fitting or quantum-resource calculations are performed. The unchanged
circuit bias and target rows are retained from the historical source table.
"""

# Public Source Data mapping (single canonical file per published input).
import sys as _kvn_path_sys
from pathlib import Path as _kvn_path_Path
_kvn_path_sys.path.insert(0, str(_kvn_path_Path(__file__).resolve().parents[3]))
from Source.data_locations import published_path, figure_data
from pathlib import Path
import csv, json, shutil
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

ROOT=Path(__file__).resolve().parents[3]
HERE=ROOT/'Figure/F4'

def read(path):
    with path.open() as f:return list(csv.DictReader(f))

def main():
    model=np.load(published_path('Data/Model/candidate_K2.npz'))
    C,powers=model['physical_coefficients'],model['powers']
    assert np.count_nonzero(C)==12
    supplied=[r for r in read(published_path('Data/SINDy_Degree/model_comparison/figure4_comparison.csv')) if r['model']=='candidate']
    t=np.array([float(r['time']) for r in supplied])
    states=np.array([[float(r[k]) for k in ['a1','a2','a3']] for r in supplied])
    frozen=np.load(published_path('Data/CFD_ROM/Re100/data/new_model_inputs.npz'))
    snap=np.load(published_path('Data/CFD_ROM/Re100/data/source_snapshots.npz'))
    a0=interp1d(frozen['all_times'],frozen['all_projected_coordinates'],axis=0)(t[0])
    np.testing.assert_allclose(a0,states[0],rtol=1e-11,atol=1e-11)
    def rhs(time,a):return np.prod(a[None,:]**powers,axis=1)@C
    sol=solve_ivp(rhs,(t[0],t[-1]),a0,method='DOP853',t_eval=t,rtol=1e-11,atol=1e-13)
    state_difference=float(np.max(abs(sol.y.T-states)))
    assert state_difference<1e-8
    mask=np.tile(frozen['fluid_mask'].ravel(),2)
    fem=interp1d(snap['times'],snap['fields'].reshape(len(snap['times']),-1),axis=0)(t)
    pred=frozen['base']+sol.y.T@frozen['Phi']
    denominator=np.linalg.norm(fem[:,mask],axis=1)
    field_error=np.linalg.norm((pred-fem)[:,mask],axis=1)/denominator*100
    params=json.loads((published_path('Data/CFD_ROM/Re100/data/model_configuration.json')).read_text())['parameters']
    dt=t-t[0];R=params['R'];u0=np.sum(a0[:2]**2)/R**2
    radial=np.sqrt(u0/(u0+(1-u0)*np.exp(-2*params['mu']*dt)))
    integral=dt+np.log(u0+(1-u0)*np.exp(-2*params['mu']*dt))/(2*params['mu'])
    phase=np.arctan2(a0[1],a0[0])+params['omega']*dt+params['gamma_normalized']*integral
    radial_states=np.column_stack([R*radial*np.cos(phase),R*radial*np.sin(phase),params['c0']+params['c2']*R**2*radial**2])
    radial_field=frozen['base']+radial_states@frozen['Phi']
    radial_error=np.linalg.norm((radial_field-pred)[:,mask],axis=1)/denominator*100
    checks={
        'state_replay_max_difference':state_difference,
        'field_error_replay_max_difference_percent':float(np.max(abs(field_error-np.array([float(r['field_vs_FEM_percent']) for r in supplied])))),
        'radial_error_replay_max_difference_percent':float(np.max(abs(radial_error-np.array([float(r['radial_vs_three_mode_FEM_norm_percent']) for r in supplied])))),
    }
    assert all(x<1e-8 for x in checks.values())
    # Retain original precision of the independently verified source table.
    template=read(figure_data('F4')/'final_panel_e/Figure4e_curves.csv')
    by_time={round(float(r['time']),10):r for r in supplied}
    for row in template:
        if row['series'] in ['3-mode vs FEM','Radial vs 3-mode']:
            source=by_time[round(float(row['time_t_star']),10)]
            key='field_vs_FEM_percent' if row['series']=='3-mode vs FEM' else 'radial_vs_three_mode_FEM_norm_percent'
            row['relative_field_error_pct']=source[key]
    for row in template:
        row['series']={'Walsh vs continuum':'Circuit vs Radial','1% target (vs continuum)':'1% RMS target (vs Radial)'}.get(row['series'],row['series'])
    primary=published_path('Data/Source_Data_Figure_4e.csv')
    with primary.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(template[0]));writer.writeheader();writer.writerows(template)
    for target in [figure_data('F4')/'final_panel_e/Figure4e_curves.csv',figure_data('F4')/'plot_values/Figure4e_curves.csv']:
        shutil.copy2(primary,target)
    report={'selected_coefficients':12,'no_refitting':True,**checks,
        'last_time':float(t[-1]),'three_mode_vs_FEM_percent':float(field_error[-1]),
        'Radial_vs_three_mode_percent':float(radial_error[-1]),
        'maximum_Radial_vs_three_mode_percent':float(radial_error.max()),
        'circuit_bias_and_target_rows_unchanged':True}
    report_file=ROOT/'Build/Figure_4_Selected_Model_Replay.json'
    report_file.parent.mkdir(parents=True,exist_ok=True)
    report_file.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))

if __name__=='__main__':main()
