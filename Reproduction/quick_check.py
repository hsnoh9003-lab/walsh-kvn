"""Short reproducible numerical checks. Excludes long CFD, complete RE search and full-grid KvN.

Use from the package root: python -m Reproduction.quick_check [--plots]
"""
from __future__ import annotations

# Public Source Data mapping (single canonical file per published input).
import sys as _kvn_path_sys
from pathlib import Path as _kvn_path_Path
_kvn_path_sys.path.insert(0, str(_kvn_path_Path(__file__).resolve().parents[1]))
from Source.data_locations import published_path
import argparse, json, subprocess, sys, time
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]

def execute(label, command, timeout=60):
    start=time.perf_counter()
    try:
        result=subprocess.run([sys.executable,*command],cwd=ROOT,text=True,
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=timeout)
        status='PASS' if result.returncode==0 else 'FAIL'
        tail=result.stdout[-1200:]
    except subprocess.TimeoutExpired as e:
        status='TIMEOUT';tail=str(e)
    elapsed=round(time.perf_counter()-start,2)
    print(f'{label}: {status} ({elapsed:.2f} s)',flush=True)
    if status!='PASS':print(tail,flush=True)
    return {'test':label,'status':status,'seconds':elapsed,'log_tail':tail if status!='PASS' else ''}

def verify_fig3():
    source=pd.read_csv(published_path('Data/RE/v38_reference_selected.csv'))
    data=pd.read_csv(published_path('Figure/F3/data/panel_b_selected_designs.csv'))
    wanted=source[source.model.isin(['direct','generator','nj_lchs']) & source.selection.isin(['minimum_qubits','fastest_uncapped'])].copy()
    assert len(data)==len(wanted)==48
    keys=['model','selection','p_phys']
    got=data.set_index(keys).sort_index();target=wanted.set_index(keys).sort_index()
    assert got.index.equals(target.index)
    for col in ['physical_qubits','d_core','seconds_per_attempt','complete_task_seconds','N_T','T_count']:
        import numpy as np
        np.testing.assert_allclose(got[col],target[col],rtol=5e-11,atol=3e-10,equal_nan=True,err_msg=col)
    return {'rows':len(data),'columns_verified':6}

def main():
    arg=argparse.ArgumentParser()
    arg.add_argument('--plots',action='store_true',help='also render public, reproducible plots F3, F5 and S08–S10')
    arg.add_argument('--pod-basis',action='store_true',help='also rebuild the frozen Re=100 POD basis and selected model from recovered CFD snapshots')
    arg.add_argument('--full-mps-preparation',action='store_true',help='also replay both full-size native MPS preparation circuits')
    a=arg.parse_args()
    tasks=[
        ('RE 384 saved-design cost evaluations',['-m','Source.RE.verify'],45),
        ('POD-SINDy/ROM/KvN/CFD small-grid numerical regression',['-m','Tests.verify_science'],45),
        ('Regenerate Fig 3 RE-selected CSV tables',['-m','Reproduction.generate_fig3_data'],30),
        ('Q5 actual MPS input and positive-root auxiliary control',['Reproduction/auxiliary/q5_upwind/validate.py'],45),
        ('S1 six-case physical forecast CSV generation',
            ['-m','Source.CFD.physical_audit','--output','Build/Replayed_Physical_Audit','--check'],45),
        ('Degree fits: support bounds, validation folds and additional CFD cases',
            ['-m','Source.POD_SINDy.evaluate_degree_cases'],45),
        ('Conditional BE normalization sensitivity',
            ['-m','Source.RE.normalization_sensitivity'],30),
        ('815-shot drag/lift sampling and second-phase coherences',
            ['-m','Tests.verify_force_sampling'],45),
        ('Fixed-design 1%-to-5% terminal-target sensitivity',
            ['-m','Source.ROM.accuracy_target_sensitivity'],30),
        ('Cultivation trial counts and selected preparation inputs',
            ['-m','Reproduction.verify_cultivation_shots'],30),
    ]
    if a.pod_basis:
        tasks.append(('Rebuild frozen POD and selected SINDy model from CFD snapshots',
            ['-m','Source.POD_SINDy.replay_frozen_basis','--out-dir','Build/Replayed_POD'],180))
    if a.full_mps_preparation:
        tasks.append(('Full 17/21-qubit elementary-gate MPS preparation',
            ['-m','Source.KvN.verify_three_coordinate_mps','both','--preparation-only','--output','Build/Replayed_MPS'],120))
    if a.plots:
        for kind,name in [('figure','F3'),('figure','F5'),('supply','S08'),('supply','S09'),('supply','S10')]:
            tasks.append((f'{kind}/{name} plot from CSV',['-m','Reproduction.run',kind,name,'--mode','plot'],45))
    results=[execute(label,args,timeout) for label,args,timeout in tasks]
    if results[2]['status']=='PASS':
        try:
            meta=verify_fig3();results.append({'test':'Fig 3 regenerated table matches RE reference','status':'PASS',**meta})
            print('F3 source-to-table validation: PASS (48 rows)',flush=True)
        except Exception as e:
            results.append({'test':'Fig 3 regenerated table matches RE reference','status':'FAIL','error':str(e)})
            print('F3 source-to-table validation: FAIL',str(e),flush=True)
    output=ROOT/'Build/Quick_Reproduction_Results.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps({'tests':results},indent=2,ensure_ascii=False)+'\n')
    print('Wrote',output,flush=True)
    return 0 if all(x['status']=='PASS' for x in results) else 1
if __name__=='__main__':sys.exit(main())
