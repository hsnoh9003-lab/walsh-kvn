"""Reoptimize both three-coordinate circuits under the shared hardware model.

Run: python -m Reproduction.reestimate_one_clean_three_coordinate --rates 2 3 4 5 6 7 8 9
Uses the same code distances, synthesis range, factory levels and preparation
tables as the other benchmark circuits. All supplied field/shot budgets stay
fixed because only exact index arithmetic and work allocation change.
"""

# Public Source Data mapping (single canonical file per published input).
import sys as _kvn_path_sys
from pathlib import Path as _kvn_path_Path
_kvn_path_sys.path.insert(0, str(_kvn_path_Path(__file__).resolve().parents[1]))
from Source.data_locations import published_path
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse,json,math,os,time
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from Source.RE import preparation as S, optimization as O, uncapped as U

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'Build/One_Clean_Three_Coordinate/resources'

def write_csv(path,rows):S.write_csv(path,rows)

def rate(ip,models=('restart40','long298')):
    tick=time.monotonic();p=ip/10000
    works=S.load_workloads(published_path('Data/RE/workloads_v38.json'))
    selected=[];capped=[];statistics={};certificates={}
    for name in models:
        w=works[name];families=[]
        for block in ('compact','intermediate','fast'):
            rows,geoms,candidates,stats=O.run_group(name,p,block,published_path(f'Data/RE/prep_p{ip}.csv'),w,
                output_dir=None,nt_min=45,nt_max=200,d_min=13,d_max=49,q_max=1_000_000)
            families.extend(rows);capped.extend(rows);statistics[name+'_'+block]=stats
            print(json.dumps({'rate':ip,'model':name,'block':block,'geometries':stats['valid_geometries'],'seconds':round(stats['seconds'],2)}),flush=True)
        minimum=dict(min(families,key=lambda r:(r['physical_qubits'],r['seconds_per_attempt'])),selection='minimum_qubits')
        incumbent=min((r for r in families if r['selection']=='fastest_under_cap'),key=U.fastest_key)
        fastest,certificate=U.find_fastest(name,w,incumbent,published_path(f'Data/RE/prep_p{ip}.csv'))
        selected.extend((minimum,fastest));certificates[name]=certificate
    for r in selected+capped:
        w=works[r['model']]
        r.update(Clifford=w['Clifford'],seconds_per_accepted_shot=r['seconds_per_attempt']/w['success'],
            complete_task_hours=r['complete_task_seconds']/3600,
            complete_task_days=r['complete_task_seconds']/86400,
            complete_task_years=r['complete_task_seconds']/(86400*365.25),
            shot_basis=w['shot_basis'],workload_scope=w['workload_scope'])
    audit={'p_phys':p,'statistics':statistics,'uncapped_certificates':certificates,'elapsed_seconds':time.monotonic()-tick,
        'logical_widths':{'restart40':20,'long298':24},'code_distance_search':[13,49,2],
        'synthesis_T_per_rotation_search':[45,200,1],'distillation_levels_max':2,
        'capped_search_physical_qubits':1_000_000,'QEC_round_seconds':1e-6,'reaction_time_seconds':10e-6,
        'per_attempt_error_budget':1e-4,'field_target':0.01,
        'unchanged_shots':{'restart40':104,'long298':234},'uncapped_fastest_search':True}
    OUT.mkdir(parents=True,exist_ok=True)
    for filename,rows in [(f'selected_p{ip}.csv',selected),(f'capped_selected_p{ip}.csv',capped)]:
        old=OUT/filename
        if old.exists():
            previous=[r for r in S.read_csv(old) if r['model'] not in models]
            rows[:0]=previous
    prior_audit=OUT/f'audit_p{ip}.json'
    if prior_audit.exists():
        prior=json.loads(prior_audit.read_text())
        audit['statistics']={**prior.get('statistics',{}),**audit['statistics']}
        audit['uncapped_certificates']={**prior.get('uncapped_certificates',{}),**audit['uncapped_certificates']}
    audit['logical_widths']['2d_interior']=11
    write_csv(OUT/f'selected_p{ip}.csv',selected);write_csv(OUT/f'capped_selected_p{ip}.csv',capped)
    (OUT/f'audit_p{ip}.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({'rate':ip,'status':'complete','elapsed_seconds':round(audit['elapsed_seconds'],2),'selected':[{k:r[k] for k in ['model','selection','physical_qubits','complete_task_hours','N_T','d_core','L']} for r in selected]}),flush=True)
    return ip

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--rates',nargs='+',type=int,default=list(range(2,10)));ap.add_argument('--processes',type=int,default=4)
    ap.add_argument('--models',nargs='+',default=['restart40','long298','2d_interior'])
    a=ap.parse_args()
    if a.processes==1:
        for ip in a.rates:rate(ip,tuple(a.models))
    else:
        with ProcessPoolExecutor(max_workers=a.processes) as pool:
            futures={pool.submit(rate,ip,tuple(a.models)):ip for ip in a.rates}
            for f in as_completed(futures):f.result()
    selected=[];capped=[]
    for p in sorted(OUT.glob('selected_p*.csv')):selected+=S.read_csv(p)
    for p in sorted(OUT.glob('capped_selected_p*.csv')):capped+=S.read_csv(p)
    write_csv(OUT/'selected.csv',selected);write_csv(OUT/'capped_selected.csv',capped)

if __name__=='__main__':main()
