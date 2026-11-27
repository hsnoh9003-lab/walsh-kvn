"""Audit/recompute Figure 1 CFD/POD input provenance using the supplied original functions.

This audit driver is newly packaged; it does not pretend to be the historical run log.
The imported fitting and solver functions are the actual archived source.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import argparse,sys,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'data_pipeline/analysis_vendor'))
from original_rom_functions import make_basis,closure,fit_weighted,physical

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshots',type=Path,default=ROOT/'source/Figure1/source/cfd_original/data/snapshots_Re100_original.npz')
    parser.add_argument('--output',type=Path,default=ROOT/'generated/data_audit')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    with np.load(args.snapshots) as z:data={k:z[k] for k in z.files}
    basis,coordinates,meta=make_basis(data)
    mask=(data['times']>=10)&(data['times']<=80)
    cl=closure(coordinates[mask]);model=fit_weighted(coordinates[mask],data['times'][mask],1e-4,0.,True,0.)
    ok,params,eigenvalues=physical(model,cl)
    assert ok
    shape=(2,len(data['grid_y']),len(data['grid_x']))
    values=dict(grid_x=data['grid_x'],grid_y=data['grid_y'],fluid_mask=data['fluid_mask'],
      modes=basis['Phi'].reshape(3,*shape),base=basis['reference'].reshape(shape),Phi=basis['Phi'],closure=cl,
      R=np.array(params['R']),mu=np.array(params['mu']),beta=np.array(params['beta']),omega=np.array(params['omega']),
      gamma_raw=np.array(params['gamma_raw']),gamma=np.array(params['gamma_normalized']),normalized_shift_scale=np.array(cl[1]*params['R']**2))
    np.savez_compressed(args.output/'physical_figure_data_recomputed.npz',**values)
    expected=ROOT/'source/Figure1/source/shared_FEM_field_inputs/physical_figure_data.npz'
    comparison={}
    with np.load(expected) as ref:
      for key,value in values.items():
        a=np.asarray(value);b=ref[key]
        same=np.array_equal(a,b)
        close=bool(np.allclose(a,b,rtol=2e-9,atol=2e-10))
        comparison[key]={'array_equal':same,'allclose':close,'max_abs':float(np.max(np.abs(a.astype(float)-b.astype(float))))}
    csv_path=ROOT/'source/Figure1/source/palette_variants/NCS_Fig1_Fig2_Design_20261030/Fig1/CFD_A_POD_coefficients_t29p04_to49p68.csv'
    archived=np.genfromtxt(csv_path,delimiter=',',names=True)
    target_times=archived['tstar']
    indices=np.array([int(np.argmin(abs(data['times']-t))) for t in target_times])
    np.testing.assert_allclose(data['times'][indices],target_times,rtol=0,atol=1e-12)
    coefficients=coordinates[indices]
    csv_values=np.c_[archived['a1'],archived['a2'],archived['a3']]
    spiral_error=float(np.max(abs(coefficients-csv_values)))
    np.testing.assert_allclose(coefficients,csv_values,rtol=2e-10,atol=2e-10)
    np.savetxt(args.output/'CFD_POD_coefficients_recomputed.csv',np.c_[data['times'][indices],coefficients],delimiter=',',header='tstar,a1,a2,a3',comments='')
    report={'status':'passed' if all(v['allclose'] for v in comparison.values()) else 'failed',
      'coordinate_spiral':{'rows':len(indices),'time_interval':[float(target_times.min()),float(target_times.max())],
        'source_csv_sha256':hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        'projection_max_abs_error':spiral_error,'passed':spiral_error<2e-10},
      'snapshots_sha256':hashlib.sha256(args.snapshots.read_bytes()).hexdigest(),
      'stored_basis_sha256':hashlib.sha256(expected.read_bytes()).hexdigest(),
      'basis_POD_interval':[20,80],'coefficient_fit_interval':[10,80],
      'ridge':1e-4,'threshold':0.,'reflection_parity':True,'early_weight_power':0.,
      'shift':meta,'comparison':comparison,
      'notes':'Uses the original archived modeling functions and original CFD snapshots; does not rerun CFD or the quantum calculation.'}
    (args.output/'INPUT_DATA_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'comparison':comparison}))
    if report['status']!='passed':raise SystemExit(1)
if __name__=='__main__':main()
