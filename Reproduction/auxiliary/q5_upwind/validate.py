"""Validate the Q5 control; --rerun also reproduces every CSV row.

Run from the extracted paper package root:
    python Data/Q5_Upwind_Comparison/validate.py --rerun
"""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
from pathlib import Path
import argparse,csv,json,tempfile,hashlib
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import expm_multiply
from scipy.linalg import expm
import reproduce as model
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from Source.data_locations import shared_data

ROOT=shared_data('Radial_refinement', 'Q5_Upwind_Comparison')
SOURCE=shared_data('Radial_refinement', 'Grid_Oscillation_Diagnostics', 'inputs', 'scalar_conservative_cell_masses.csv')
def records(path):
    with path.open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def independent(F,h):
    return diags([-abs(F)/h,
        np.where(F[1:]>0,(F[1:]+F[:-1])/(2*h),0),
        np.where(F[:-1]<0,-(F[:-1]+F[1:])/(2*h),0)],[0,-1,1],format='csr')
def main(rerun=False):
    saved=records(ROOT/'q5_upwind_comparison.csv');rerun_error=None
    auxiliary=records(ROOT/'positive_root_control/q5_upwind_comparison.csv')
    summary=json.loads((ROOT/'summary.json').read_text())
    assert summary['input_amplitude_convention']=='mps'
    assert summary['initial_negative_amplitude_cells']==16
    assert abs(summary['initial_negative_amplitude_probability']-8.483481423107534e-9)<1e-20
    if rerun:
        differences=[];array_differences=[]
        with tempfile.TemporaryDirectory() as folder:
            for convention,old,base in [('mps',saved,ROOT),('positive_root',auxiliary,ROOT/'positive_root_control')]:
                output=Path(folder)/convention
                model.main(str(SOURCE),str(output),convention)
                fresh=records(output/'q5_upwind_comparison.csv')
                assert len(old)==len(fresh)==16
                for a,b in zip(old,fresh):
                    assert a['method']==b['method']
                    for key in a:
                        if key!='method':
                            differences.append(abs(float(a[key])-float(b[key])))
                            assert np.isclose(float(a[key]),float(b[key]),rtol=2e-10,atol=2e-12)
                with np.load(output/'q5_upwind_distributions.npz') as a,np.load(base/'q5_upwind_distributions.npz') as b:
                    assert set(a.files)==set(b.files)
                    for key in a.files:
                        difference=float(np.max(abs(a[key]-b[key])))
                        array_differences.append(difference)
                        assert difference<2e-12
        rerun_error=max(differences)
    reference={int(r['cells']):r for r in records(ROOT/'previous_exact_generator_rows.csv')}
    for r in saved:
        if r['method']=='centred':
            old=reference[int(r['cells'])]
            for key in ['native_cell_TV','aggregated_64_bin_TV','relative_continuum_moment_error']:
                assert abs(float(r[key])-float(old[key]))<1e-9
    source=[r for r in records(SOURCE) if r['id']==model.CASE]
    p64=np.array([float(r['initial_prepared_mass']) for r in source]);checks=[]
    for n in model.GRIDS:
        F=model.drift((np.arange(n)+.5)/n);G=model.upwind_generator(n).tocsr();H=independent(F,1/n)
        D=G-H;operator_error=float(max(abs(D.data),default=0.));assert operator_error<1e-13
        initial=np.repeat(np.sqrt(p64/(n//64)),n//64);state=expm_multiply(G,initial)
        signed=np.repeat(np.real(np.load(ROOT/'original_prepared_amplitude.npz')['signed_mps_amplitude'])/np.sqrt(n//64),n//64)
        signed_state=expm_multiply(G,signed)
        check=dict(cells=n,independent_stencil_max_abs=operator_error,
            positive_root_upwind_min_amplitude=float(state.min()),positive_root_negative_cells=int((state<0).sum()),
            mps_upwind_min_amplitude=float(signed_state.min()),mps_negative_amplitude_probability=float(np.sum(signed_state[signed_state < -1e-9*max(abs(signed_state))]**2)/np.dot(signed_state,signed_state)))
        assert check['positive_root_negative_cells']==0
        record=next(r for r in saved if int(r['cells'])==n and r['method']=='upwind_NJ')
        assert abs(float(record['negative_amplitude_probability'])-check['mps_negative_amplitude_probability'])<2e-16
        if n in [64,512]:
            check['sparse_dense_linf']=float(max(abs(expm(G.toarray())@initial-state)))
            mirror=independent(-F[::-1],1/n);D=mirror-G[::-1,::-1]
            check['mirror_operator_max_abs']=float(max(abs(D.data),default=0.))
            check['mirror_state_linf']=float(max(abs(expm_multiply(mirror,initial[::-1])[::-1]-state)))
            steps=65536;dt=1/steps;x=initial.copy()
            for _ in range(steps):
                k1=G@x;k2=G@(x+dt*k1/2);k3=G@(x+dt*k2/2);k4=G@(x+dt*k3);x+=dt*(k1+2*k2+2*k3+k4)/6
            check['rk4_sparse_linf']=float(max(abs(x-state)))
            assert all(check[k]<2e-13 for k in ['sparse_dense_linf','mirror_operator_max_abs','mirror_state_linf','rk4_sparse_linf'])
        checks.append(check)
    phase=np.load(ROOT/'original_prepared_amplitude.npz');a=phase['signed_mps_amplitude'];positive=phase['positive_root_amplitude']
    assert max(abs(abs(a)**2-positive**2))<2e-14
    # The amplitude snapshot must be the actual archived circuit input when the
    # full release is available; standalone paper copies retain the same snapshot.
    circuit_history=SOURCE.with_name('quintic_multistable__boundary_clipped.npz')
    if circuit_history.exists():
        with np.load(circuit_history) as original:
            np.testing.assert_array_equal(a,original['state_history'][0])
    sensitivity=[]
    for primary,control in zip(saved,auxiliary):
        assert (primary['cells'],primary['method'])==(control['cells'],control['method'])
        row=dict(cells=int(primary['cells']),method=primary['method'])
        for key in ['native_cell_TV','aggregated_64_bin_TV','relative_continuum_moment_error','squared_output_norm']:
            row[key+'_change_percentage_points']=100*(float(primary[key])-float(control[key]))
        row['mps_negative_amplitude_probability']=float(primary['negative_amplitude_probability'])
        row['positive_root_negative_amplitude_probability']=float(control['negative_amplitude_probability'])
        sensitivity.append(row)
    max_change=max(abs(value) for row in sensitivity for key,value in row.items() if key.endswith('_change_percentage_points'))
    assert max_change<2.3e-4
    output=Path(__file__).resolve().parents[3]/'Build/Q5_Upwind_Comparison'
    output.mkdir(parents=True,exist_ok=True)
    with (output/'input_phase_comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(sensitivity[0]));writer.writeheader();writer.writerows(sensitivity)
    report=dict(status='PASS',csv_rows=len(saved),all_csv_rows_rerun=rerun,rerun_max_abs_difference=rerun_error,
        auxiliary_csv_rows=len(auxiliary),primary_input_convention='actual recorded MPS amplitudes',
        auxiliary_input_convention='positive square roots of prepared cell masses',
        primary_and_auxiliary_arrays_rerun_max_abs_difference=max(array_differences) if rerun else None,
        initial_mps_negative_amplitude_probability=float(np.sum(abs(a[np.real(a)<0])**2)),
        maximum_reported_metric_change_percentage_points=max_change,
        previous_centred_metrics_agree_within=1e-9,checks=checks,
        signed_mps_vs_positive_root_amplitude_l2=float(np.linalg.norm(a-positive)),
        prepared_probability_max_abs_difference=float(max(abs(abs(a)**2-positive**2))),
        mps_phase_preserved_in_primary=True,positive_root_is_auxiliary=True,
        sign_diagnostic_is_not_a_density_ripple_certificate=True,
        source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest())
    destination=output/('validation.json' if rerun else 'quick_validation.json')
    destination.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--rerun',action='store_true');main(p.parse_args().rerun)
