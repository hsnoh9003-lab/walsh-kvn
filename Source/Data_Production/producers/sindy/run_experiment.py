"""Nested polynomial-library validation on the original projected CFD data.

The protocol is fixed before fitting: reflection parity, no constant term,
original RMS/column scaling and ridge=1e-4; four contiguous validation blocks;
selection by mean within-basis velocity-field rollout error; final unseen-time
evaluation only after selecting a threshold for each library degree.
"""
from pathlib import Path
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import ast,csv,json,time,sys,hashlib,platform
import numpy as np
from scipy.integrate import solve_ivp
import scipy

ROOT=Path(__file__).resolve().parent
from Source.POD_SINDy import identification as ident
OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
DEGREES=[1,2,3,4,5]
THRESHOLDS=[0.,1e-5,3e-5,1e-4,3e-4,.001,.003,.01,.03,.1,.3,1.,3.]
RIDGE=1e-4

def fit(raw,t,degree,threshold,observed=None,derivative_mask=None):
    if observed is None:observed=np.ones(len(raw),bool)
    scale=np.sqrt(np.mean(raw[observed]**2,axis=0))
    assert np.all(scale>0)
    u,tt,du=ident.differentiate(raw/scale,t)
    if derivative_mask is not None:u,du=u[derivative_mask],du[derivative_mask]
    p=ident.powers(3,degree);A=ident.features(u,p)
    signs=np.array([-1,-1,1]);term_sign=np.prod(signs[None,:]**p,axis=1)
    allowed=(term_sign[:,None]==signs[None,:]);allowed[0]=False
    active=allowed.copy();C=np.zeros_like(active,dtype=float)
    for iteration in range(20):
        C[:]=0
        for j in range(3):
            mask=active[:,j]
            if not mask.any():continue
            a=A[:,mask];s=np.linalg.norm(a,axis=0);s[s==0]=1
            aa=a/s
            C[mask,j]=np.linalg.lstsq(np.vstack([aa,np.sqrt(RIDGE)*np.eye(aa.shape[1])]),
                                  np.r_[du[:,j],np.zeros(aa.shape[1])],rcond=1e-12)[0]/s
        new=active&(np.abs(C)>=threshold)
        if np.array_equal(new,active):break
        active=new
    physical=C*scale[None,:]/np.prod(scale[None,:]**p,axis=1)[:,None]
    used=np.any(C!=0,axis=1)
    maxdegree=int(p[used].sum(axis=1).max()) if used.any() else 0
    higher=int(np.count_nonzero(C[p.sum(axis=1)>2]))
    return dict(p=p,C=C,physical_C=physical,scale=scale,degree=degree,threshold=threshold,
       allowed_terms=int(allowed.sum()),active_terms=int(np.count_nonzero(C)),
       active_degree=maxdegree,active_above_quadratic=higher)

def rollout(model,x0,times):
    p=model['p'];C=model['C'];scale=model['scale'];used=np.any(C!=0,axis=1);p=p[used];C=C[used]
    calls=[0]
    def rhs(t,u):
        calls[0]+=1
        if calls[0]>25000:raise RuntimeError('rhs_evaluation_budget')
        if not np.all(np.isfinite(u)) or np.linalg.norm(u)>100:
            raise RuntimeError('out_of_support_compute_safeguard')
        return np.prod(u[None,:]**p,axis=1)@C
    try:
        with np.errstate(over='raise',invalid='raise'):
            sol=solve_ivp(rhs,[times[0],times[-1]],x0/scale,t_eval=times,
                          method='DOP853',rtol=1e-8,atol=1e-10)
        if not sol.success or sol.y.shape[1]!=len(times):raise RuntimeError('incomplete_integration')
        return sol.y.T*scale,dict(completed=True,nfev=calls[0])
    except (FloatingPointError,RuntimeError,ValueError) as error:
        return None,dict(completed=False,nfev=calls[0],reason=str(error))

def errors(pred,truth,scale,G,c,base2):
    delta=pred[1:]-truth[1:];true=truth[1:]
    field_numerator=float(np.einsum('ni,ij,nj->',delta,G,delta))
    field_denominator=len(true)*base2+2*float(true.sum(axis=0)@c)+float(np.einsum('ni,ij,nj->',true,G,true))
    field=100*np.sqrt(max(0,field_numerator)/field_denominator)
    canonical=100*np.linalg.norm(delta/scale)/np.linalg.norm(true/scale)
    return dict(projected_field_error_percent=float(field),RMS_scaled_coordinate_error_percent=float(canonical))

def write_csv(path,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def main():
    tick=time.monotonic();d=np.load(ROOT/'inputs/baseline_inputs.npz')
    x=d['training_coordinates'];t=d['training_times'];fullx=d['all_projected_coordinates'];fullt=d['all_times']
    assert len(t)==292 and np.allclose(np.diff(t),.24)
    assert np.array_equal(x,fullx[(fullt>=10)&(fullt<=80)])
    phi=d['Phi'][:,np.tile(d['fluid_mask'].ravel(),2)]
    base=d['base'][np.tile(d['fluid_mask'].ravel(),2)]
    G=phi@phi.T;c=phi@base;base2=float(base@base)
    # Recover the archived quadratic coefficients without any changed settings.
    baseline=fit(x,t,2,0.)
    diff=float(np.max(np.abs(baseline['physical_C']-d['C'])))
    assert np.array_equal(baseline['p'],d['polynomial_exponents']) and diff<1e-10,diff
    blocks=np.repeat(np.arange(4),73)
    centre=np.arange(2,len(x)-2)
    folds=[]
    for fold in range(4):
        observed=blocks!=fold
        training=np.array([observed[i-2:i+3].all() for i in centre])
        validation=np.array([(blocks[i-2:i+3]==fold).all() for i in centre])
        hold=np.flatnonzero(blocks==fold)
        folds.append(dict(fold=fold,observed=observed,training=training,validation=validation,hold=hold))
    protocol=dict(training_case='A',training_states=292,training_actual_interval=[float(t[0]),float(t[-1])],
      library_degrees=DEGREES,thresholds=THRESHOLDS,ridge=RIDGE,
      constraints='Original reflection parity; equilibrium at origin; no constant term; no restriction to quadratic monomials.',
      scaling='Training-only coordinate RMS; numerical feature column equilibration, as in original fit.',
      derivative='Original centred fourth-order finite difference; stencil must lie wholly in its train/validation partition.',
      validation='Four contiguous 73-state blocks; each rollout is initialized at the first held-out CFD projected state.',
      selection='Minimum mean validation projected-field rollout error; no degree or term-count penalty and no one-SE rule.',
      final_test='Original case A after the coefficient-fitting interval: rollout from t*=79.92 through 90, scored at later samples only.',
      reference='Projected CFD states in the frozen original POD/shift basis, not trajectories generated by the fitted ODE.',
      scope='Coefficient/degree validation conditional on a frozen spatial basis, which uses later CFD data. This is not independent validation of the full POD-learning pipeline or cases D–F.',
      original_quadratic_max_coefficient_difference=diff,
      folds=[dict(fold=f['fold'],interval=t[f['hold']][[0,-1]].tolist(),
          training_derivative_rows=int(f['training'].sum()),validation_derivative_rows=int(f['validation'].sum())) for f in folds])
    (ROOT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    (OUT/'baseline_recovery.json').write_text(json.dumps(dict(max_absolute_coefficient_difference=diff,active_terms=baseline['active_terms']),indent=2)+'\n')
    fold_rows=[];candidates=[];predictions={}
    for degree in DEGREES:
        for threshold in THRESHOLDS:
            scores=[];coord=[];trainfits=[];validfits=[]
            for f in folds:
                model=fit(x,t,degree,threshold,f['observed'],f['training'])
                u,_,du=ident.differentiate(x/model['scale'],t)
                predicted=ident.features(u,model['p'])@model['C']
                trainerr=100*np.linalg.norm((predicted-du)[f['training']])/np.linalg.norm(du[f['training']])
                validerr=100*np.linalg.norm((predicted-du)[f['validation']])/np.linalg.norm(du[f['validation']])
                hold=f['hold'];pred,audit=rollout(model,x[hold[0]],t[hold])
                metric=errors(pred,x[hold],model['scale'],G,c,base2) if pred is not None else dict(projected_field_error_percent=float('inf'),RMS_scaled_coordinate_error_percent=float('inf'))
                row=dict(library_degree=degree,threshold=threshold,fold=f['fold'],
                    active_terms=model['active_terms'],active_degree=model['active_degree'],
                    active_above_quadratic=model['active_above_quadratic'],
                    training_derivative_error_percent=float(trainerr),validation_derivative_error_percent=float(validerr),
                    **metric,**audit)
                fold_rows.append(row);scores.append(metric['projected_field_error_percent']);coord.append(metric['RMS_scaled_coordinate_error_percent'])
                trainfits.append(trainerr);validfits.append(validerr)
                if pred is not None:predictions[f'K{degree}_t{threshold:g}_f{f["fold"]}']=pred
            candidates.append(dict(library_degree=degree,threshold=threshold,
                mean_validation_field_error_percent=float(np.mean(scores)),
                mean_validation_scaled_coordinate_error_percent=float(np.mean(coord)),
                mean_training_derivative_error_percent=float(np.mean(trainfits)),
                mean_validation_derivative_error_percent=float(np.mean(validfits)),
                completed_folds=int(sum(np.isfinite(scores))),
                fold_active_degrees=';'.join(str(r['active_degree']) for r in fold_rows[-4:]),
                fold_active_above_quadratic=';'.join(str(r['active_above_quadratic']) for r in fold_rows[-4:])))
            print('CANDIDATE',json.dumps(candidates[-1]),flush=True)
        write_csv(OUT/'validation_folds.csv',fold_rows);write_csv(OUT/'validation_candidates.csv',candidates)
    selected=[];coefficients=[];test_predictions={}
    last=np.flatnonzero(fullt<=t[-1]+1e-9)[-1]
    testtimes=fullt[last:];testtruth=fullx[last:]
    for degree in DEGREES:
        rows=[r for r in candidates if r['library_degree']==degree]
        best=min(rows,key=lambda r:(r['mean_validation_field_error_percent'],r['threshold']))
        model=fit(x,t,degree,best['threshold'])
        prediction,audit=rollout(model,testtruth[0],testtimes)
        metric=errors(prediction,testtruth,model['scale'],G,c,base2) if prediction is not None else dict(projected_field_error_percent=float('inf'),RMS_scaled_coordinate_error_percent=float('inf'))
        row=dict(best,allowed_terms=model['allowed_terms'],active_terms=model['active_terms'],active_degree=model['active_degree'],
            active_above_quadratic=model['active_above_quadratic'],test_field_error_percent=metric['projected_field_error_percent'],
            test_scaled_coordinate_error_percent=metric['RMS_scaled_coordinate_error_percent'],test_completed=audit['completed'])
        selected.append(row)
        np.savez_compressed(OUT/f'model_K{degree}.npz',powers=model['p'],canonical_coefficients=model['C'],
            physical_coefficients=model['physical_C'],training_RMS=model['scale'])
        if prediction is not None:test_predictions[f'K{degree}']=prediction
        terms=ident.labels(model['p'],['a1','a2','a3'])
        for i,term in enumerate(terms):
            for axis in range(3):
                if model['C'][i,axis]!=0:
                    coefficients.append(dict(library_degree=degree,threshold=best['threshold'],equation=axis+1,
                        term=term,total_degree=int(model['p'][i].sum()),canonical_coefficient=model['C'][i,axis],physical_coefficient=model['physical_C'][i,axis]))
        print('SELECTED_PER_LIBRARY',json.dumps(row),flush=True)
    winner=min(selected,key=lambda r:(r['mean_validation_field_error_percent'],r['library_degree']))
    original_prediction,audit=rollout(baseline,testtruth[0],testtimes)
    original_test=errors(original_prediction,testtruth,baseline['scale'],G,c,base2)
    write_csv(OUT/'selected_by_library.csv',selected);write_csv(OUT/'selected_coefficients.csv',coefficients)
    np.savez_compressed(OUT/'validation_rollouts.npz',**predictions)
    np.savez_compressed(OUT/'test_rollouts.npz',time=testtimes,reference=testtruth,original_quadratic=original_prediction,**test_predictions)
    report=dict(protocol=protocol,winner=winner,selected_per_library=selected,
        original_quadratic_test=original_test,n_candidates=len(candidates),n_fold_fits=len(fold_rows),
        elapsed_seconds=time.monotonic()-tick,software=dict(Python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__))
    # JSON Infinity is avoided in the report; CSV retains explicit inf failures.
    def finite(v):
        if isinstance(v,dict):return {k:finite(x) for k,x in v.items()}
        if isinstance(v,list):return [finite(x) for x in v]
        if isinstance(v,float) and not np.isfinite(v):return None
        return v
    (OUT/'summary.json').write_text(json.dumps(finite(report),indent=2,allow_nan=False)+'\n')
    print('COMPLETE',json.dumps(winner),'seconds',report['elapsed_seconds'],flush=True)

if __name__=='__main__':main()
