"""Evaluate three fixed case-A SINDy fits on additional CFD trajectories.

No threshold or degree is selected using these B–F scores. Both full-field
and fixed-basis projected-field errors are retained for the two S1 windows.
"""

# Public Source Data mapping (single canonical file per published input).
import sys as _kvn_path_sys
from pathlib import Path as _kvn_path_Path
_kvn_path_sys.path.insert(0, str(_kvn_path_Path(__file__).resolve().parents[2]))
from Source.data_locations import published_path
from pathlib import Path
import csv, hashlib, json
from itertools import product
from math import comb
import numpy as np
from .fit import fit
from .identification import simulate_diagnostic

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'Build/SINDy_Degree/additional_cases'
SPECS=[(2,.03,12,226),(3,.03,18,671),(3,.01,26,1297)]
WINDOWS={'short':(40.,49.82906260206943),'extended':(20.,90.)}

def write_csv(path, rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)

def support_bound(p,C,n=6):
    total=0
    for equation in range(3):
        union=set()
        for powers in p[C[:,equation]!=0]:
            union.update(product(*(range(int(x)+1) for x in powers)))
        bits=[n-1 if j==equation else n for j in range(3)]
        total+=sum(np.prod([comb(bits[j],w[j]) for j in range(3)]) for w in union)
    return int(total)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with np.load(published_path('Data/Model/baseline_inputs.npz')) as f:
        times=f['training_times'];coordinates=f['training_coordinates']
    models=[];coefficients={};frozen=[]
    with np.load(published_path('Data/CFD_ROM/Re100/data/new_model_inputs.npz')) as f:
        phi,base=f['Phi'],f['base']
    for degree,threshold,terms,bound in SPECS:
        m=fit(coordinates,times,degree,threshold)
        assert m['active_terms']==terms
        assert support_bound(m['p'],m['physical_C'])==bound
        key=f'd{degree}_t{threshold:g}'
        model={'p':m['p'],'C':m['physical_C'],'training_rms_scale':m['scale']}
        models.append((key,degree,threshold,terms,bound,model))
        coefficients[key+'_C']=m['physical_C'];coefficients[key+'_p']=m['p']
        frozen.append({'model':key,'library_degree':degree,'threshold':threshold,'terms':terms,
            'walsh_support_bound':bound,'coefficient_sha256':hashlib.sha256(m['physical_C'].tobytes()).hexdigest()})
    # Persist the complete specification before opening any additional case.
    protocol={'fitted_on':'A only, 292 states at 10.08–79.92; original fixed spatial basis',
        'models':frozen,'windows':WINDOWS,'primary_evaluation_cases':['B','C','D','E','F'],
        'selection':'Three prespecified candidates; no selection or refit from the additional-case errors',
        'scope':'Additional evaluation of already published candidates and historical test cases; not a newly blinded test',
        'projected_metric':'Forecast minus CFD projection, normalized by the reconstructed projected CFD field, including the steady background; initialization state excluded from scoring',
        'physical_metric':'Forecast minus full CFD field, normalized by the full CFD field, including the steady background; initialization state excluded from scoring'}
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    np.savez_compressed(OUT/'frozen_candidates.npz',**coefficients)
    rows=[];trajectories={};gram=phi@phi.T;bp=phi@base;bb=base@base
    with np.load(published_path('Data/Physical_Forecast_Audit/audit_inputs.npz')) as data:
        for case in 'BCDEF':
            for window,(lo,hi) in WINDOWS.items():
                t=data[case+'_times'];use=(t>=lo)&(t<=hi+1e-9)
                tt=t[use];observed=data[case+'_coordinates'][use]
                full_den=float(np.sum(data[case+'_field_squared'][use][1:]))
                residual=float(np.sum(data[case+'_projection_squared'][use][1:]))
                norms=bb+2*(observed@bp)+np.einsum('ti,ij,tj->t',observed,gram,observed)
                projected_den=float(np.sum(norms[1:]))
                for key,degree,threshold,terms,bound,model in models:
                    pred,diag=simulate_diagnostic(model,observed[0],tt)
                    row={'case':case,'model':key,'library_degree':degree,'threshold':threshold,
                        'terms':terms,'walsh_support_bound':bound,'window':window,
                        'start':float(tt[0]),'end':float(tt[-1]),'states':len(tt),'scored_states':len(tt)-1,
                        'completed':diag['completed'],'reason':diag['reason'],
                        'projected_field_error_pct':None,'physical_field_error_pct':None,
                        'physical_projection_error_pct':100*np.sqrt(residual/full_den),'nfev':diag['nfev']}
                    if pred is not None:
                        delta=(pred-observed)[1:]
                        extra=max(0.,float(np.einsum('ti,ij,tj->',delta,gram,delta)))
                        row['projected_field_error_pct']=100*np.sqrt(extra/projected_den)
                        row['physical_field_error_pct']=100*np.sqrt((residual+extra)/full_den)
                        trajectories[f'{case}_{window}_{key}']=pred
                    rows.append(row)
    write_csv(OUT/'case_errors.csv',rows)
    # Reproduce the three reported four-block validation means, using the
    # same exclusion of the initialized state as the additional-case scores.
    reference=list(csv.DictReader((published_path('Data/SINDy_Degree/validation_folds.csv')).open()))
    fold_rows=[]
    for key,degree,threshold,terms,bound,_ in models:
        for fold in range(4):
            val=np.zeros(len(times),bool);val[fold*73:(fold+1)*73]=True;train=~val
            valid=np.array([train[i-2:i+3].all() for i in range(2,len(times)-2)])
            m=fit(coordinates,times,degree,threshold,observed=train,derivative_mask=valid)
            pred,diag=simulate_diagnostic({'C':m['physical_C'],'p':m['p']},coordinates[val][0],times[val])
            assert diag['completed']
            observed=coordinates[val];d=(pred-observed)[1:]
            denom=np.sum((bb+2*(observed@bp)+np.einsum('ti,ij,tj->t',observed,gram,observed))[1:])
            score=float(100*np.sqrt(np.einsum('ti,ij,tj->',d,gram,d)/denom))
            expected=next(r for r in reference if int(r['library_degree'])==degree and float(r['threshold'])==threshold and int(r['fold'])==fold)
            difference=abs(score-float(expected['projected_field_error_percent']))
            assert difference<1e-5,(key,fold,difference)
            fold_rows.append({'model':key,'fold':fold,'projected_field_error_percent':score,'difference_from_retained_score':difference})
    write_csv(OUT/'replayed_validation_folds.csv',fold_rows)
    summary=[]
    for window in WINDOWS:
        for key,degree,threshold,terms,bound,model in models:
            chosen=[r for r in rows if r['case'] in 'DEF' and r['window']==window and r['model']==key]
            okay=all(r['completed'] for r in chosen)
            summary.append({'model':key,'library_degree':degree,'threshold':threshold,'terms':terms,
                'walsh_support_bound':bound,'window':window,'cases':'D;E;F','completed_cases':sum(r['completed'] for r in chosen),
                'mean_projected_field_error_pct':float(np.mean([r['projected_field_error_pct'] for r in chosen])) if okay else None,
                'mean_physical_field_error_pct':float(np.mean([r['physical_field_error_pct'] for r in chosen])) if okay else None})
    write_csv(OUT/'three_case_summary.csv',summary)
    # Supplementary Note 1 includes every non-training case. Keep the old D–F summary
    # solely for provenance, not as a selected subset of current evidence.
    all_summary=[]
    for window in WINDOWS:
        for key,degree,threshold,terms,bound,model in models:
            chosen=[r for r in rows if r['window']==window and r['model']==key]
            assert len(chosen)==5 and all(r['completed'] for r in chosen)
            all_summary.append({'model':key,'library_degree':degree,'threshold':threshold,'terms':terms,
                'walsh_support_bound':bound,'window':window,'cases':'B;C;D;E;F','completed_cases':5,
                'mean_projected_field_error_pct':float(np.mean([r['projected_field_error_pct'] for r in chosen])),
                'mean_physical_field_error_pct':float(np.mean([r['physical_field_error_pct'] for r in chosen]))})
    write_csv(OUT/'all_case_summary.csv',all_summary)
    np.savez_compressed(OUT/'forecast_coordinates.npz',**trajectories)
    (OUT/'validation.json').write_text(json.dumps({'all_30_rollouts_completed':all(r['completed'] for r in rows),
        'rollouts':len(rows),'fitted_candidates':len(models),'new_test_based_model_selection':False,
        'all_three_support_bounds_recomputed':True,'all_12_validation_fold_scores_reproduced':True},indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
