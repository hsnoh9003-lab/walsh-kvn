
# Public Source Data mapping (single canonical file per published input).
import sys as _kvn_path_sys
from pathlib import Path as _kvn_path_Path
_kvn_path_sys.path.insert(0, str(_kvn_path_Path(__file__).resolve().parents[1]))
from Source.data_locations import published_path, figure_data
from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'Build/Regenerated/Supplementary'
from Source.RE import sensitivity as SUP
CAP=SUP
from Source.RE.optimization import run_group
from Source.RE.uncapped import select_fastest_with_tolerance
from Source.RE.preparation import load_workloads
from functools import lru_cache
from Reproduction.sweep_output import staged_sweep
VALUES=figure_data('S09')/'plot_values'
native_values=sorted(pd.read_csv(VALUES/'native_T_sensitivity.csv').native_T.unique().astype(int))
@lru_cache(None)
def geometries(k):
    work=load_workloads(published_path('Data/RE/workloads_v38.json'))['direct']
    frames=[]
    for block in SUP.BLOCKS:
        _,gs,_,_=run_group('direct',k/10000,block,published_path(f'Data/RE/prep_p{k}.csv'),work)
        frames.append(pd.DataFrame(gs))
    return pd.concat(frames,ignore_index=True)
def evaluate(g,native=7392,eta=1e-4):
 costs=np.arange(45,201);core=g.log_core_survival_per_T.to_numpy();magic=g.log_magic_survival_per_T.to_numpy()
 j=SUP.first_feasible_cost_indices(core,magic,native,4803,costs,eta);valid=j>=0;a=g.loc[valid].copy();ids=np.flatnonzero(valid);n=costs[j[valid]];t=native+4803*n
 ec=-np.expm1(core[ids]*t);em=-np.expm1(magic[ids]*t);es=4803*np.exp2(-n/3.)
 a['N_T']=n;a['synthesis_T_per_RZ']=n;a['T_count']=t;a['T_core']=t;a['native_T']=native;a['RZ']=4803
 a['E_core']=ec;a['E_MSF']=em;a['E_synth']=es;a['E_total']=ec+em-ec*em+es
 a['seconds_per_attempt']=t*a.service_period_s;a['complete_task_seconds']=815*a.seconds_per_attempt
 a['mean_supply_wait_per_T_s']=a.service_period_s-a.core_period_s
 return a
def policies(g,native):
 a=evaluate(g,native);out=[]
 for dc in sorted(a.d_core.unique()):
  rows=a[a.d_core.eq(dc)]
  s=SUP.pick(rows,'supply_enforced');r=SUP.pick(rows,'delay_allowed')
  d={'model':'walsh_A','p_phys':g.p_phys.iloc[0],'d_core':int(dc)}
  for tag,v in [('strict',s),('one',r)]:
   for key,src in [('q','physical_qubits'),('t','seconds_per_attempt'),('u','n_units'),('L','L'),('dmsf','distances'),('E','E_total'),('NT','N_T'),('block','block')]:d[tag+'_'+key]=v[src] if v is not None else np.nan
  out.append(d)
 s=SUP.pick(a,'supply_enforced');r=SUP.pick(a,'delay_allowed')
 return out,s,r
def worker(k):
 g=geometries(k);rows=[]
 for native in native_values:
  s,r=SUP.scan_pair_fast(g,int(native))
  assert s is not None and r is not None
  rows.append(SUP.compare(s,r))
 distances,s,r=policies(g,7392);comparison=SUP.compare(s,r)
 opt={'p_phys':k/10000,'strict_opt_d':s['d_core'],'one_opt_d':r['d_core'],'strict_opt_Q':s['physical_qubits'],'one_opt_Q':r['physical_qubits'],'strict_opt_time':s['seconds_per_attempt'],'one_opt_time':r['seconds_per_attempt'],'physical_qubit_change_pct':comparison['qubit_change_pct'],'runtime_change_pct':comparison['runtime_change_pct'],'strict_units':s['n_units'],'one_units':r['n_units'],'strict_opt_L':s['L'],'one_opt_L':r['L']}
 capacity=[];fastest33=[]
 for cap in [100000,1000000]:
  for block in SUP.BLOCKS:
   q=g[g.block.eq(block)&g.physical_qubits.le(cap)].copy().reset_index(drop=True)
   result=CAP.joint_capacity(q,'A',45,125,86400/815,4803,143,7392,1.0)
   assert result is not None
   capacity.append({**result,'Q_budget':cap,'days':1,'p_phys':k/10000,'block':block,'K':result['max_steps'],'N_T':result['N_T'],'t_step_s':result['t_step_seconds'],'p_good':1,'shots':815,**{'verified_next_step_error':result['E_total_next_step'],'verified_next_step_time':result['next_step_runtime_seconds']}})
 for scheme,native in [('A',7392),('B',12936)]:
  for block in SUP.BLOCKS:
   q=g[g.block.eq(block)&g.physical_qubits.le(1000000)].copy().reset_index(drop=True)
   a=select_fastest_with_tolerance(evaluate(q,native).to_dict('records'))
   a.update(scheme=scheme,Q_budget=1000000,selection='fastest_under_cap',model='walsh_'+scheme,successful_shots=815,success_probability=1.0);fastest33.append(a)
 dT=r['T_core']-s['T_core'];Tr=r['T_core']
 decomp={'p_phys':k/10000,'T_count_contribution_s':dT*s['service_period_s'],'core_period_contribution_s':Tr*(r['core_period_s']-s['core_period_s']),'wait_contribution_s':Tr*(r['mean_supply_wait_per_T_s']-s['mean_supply_wait_per_T_s']),'actual_runtime_difference_s':r['seconds_per_attempt']-s['seconds_per_attempt']}
 decomp['identity_residual_s']=decomp['actual_runtime_difference_s']-sum(decomp[x] for x in ['T_count_contribution_s','core_period_contribution_s','wait_contribution_s']);assert abs(decomp['identity_residual_s'])<1e-9
 print('supplement rate',k,'done',flush=True)
 return k,rows,distances,opt,capacity,fastest33,comparison,decomp,[dict(s,policy='supply_enforced'),dict(r,policy='delay_allowed')]
def generate(rates, output):
 results=[worker(k) for k in rates]
 flatten=lambda i:[row for result in results for row in result[i]]
 pd.DataFrame(flatten(1)).to_csv(output/'native_T_sensitivity.csv',index=False)
 pd.DataFrame(flatten(2)).to_csv(output/'supply_distances.csv',index=False)
 pd.DataFrame([r[3] for r in results]).to_csv(output/'supply_optima.csv',index=False)
 pd.DataFrame(flatten(4)).to_csv(output/'capacity.csv',index=False)
 pd.DataFrame(flatten(4)).to_csv(output/'capacity_selected_designs.csv',index=False)
 pd.DataFrame(flatten(5)).to_csv(output/'capacity_fastest33.csv',index=False)
 pd.DataFrame([r[6] for r in results]).to_csv(output/'baseline_policy_comparison.csv',index=False)
 pd.DataFrame([r[7] for r in results]).to_csv(output/'runtime_decomposition.csv',index=False)
 pd.DataFrame(flatten(8)).to_csv(output/'baseline_policy_optima.csv',index=False)
 scan=pd.DataFrame(flatten(1))
 controls=scan[scan.native_T.gt(0)].sort_values('runtime_change_pct').groupby('p_phys').tail(1).copy();controls['search_scope']='maximum over the declared inventory grid';controls['maximization_scope']='original_range';controls.to_csv(output/'refined_runtime_controls.csv',index=False)
 representative=((3,7392),(3,3210771),(4,510000),(5,200000),(6,1140000))
 selected=[scan[(abs(scan.p_phys-k/10000)<1e-12)&scan.native_T.eq(t)].iloc[0].to_dict() for k,t in representative if k in rates]
 pd.DataFrame(selected, columns=scan.columns).to_csv(output/'representative_workloads.csv',index=False)
 if 5 in rates:
  curves,s,r=policies(geometries(5),200000);pd.DataFrame(curves).to_csv(output/'control_distance_curves.csv',index=False)
  comparison=SUP.compare(s,r)
  pd.DataFrame([dict(p_phys=.0005,circuit_T=200000,strict_opt_d=s['d_core'],one_opt_d=r['d_core'],strict_opt_Q=s['physical_qubits'],one_opt_Q=r['physical_qubits'],strict_opt_time=s['seconds_per_attempt'],one_opt_time=r['seconds_per_attempt'],physical_qubit_change_pct=comparison['qubit_change_pct'],runtime_change_pct=comparison['runtime_change_pct'])]).to_csv(output/'control_distance_optima.csv',index=False)
  original=pd.read_csv(published_path('Supply/S08/data/plot_values/implementation_budget_sensitivity.csv'));sensitivity=[];g=geometries(5)
  for old in original.to_dict('records'):
   candidates=evaluate(g,7392,float(old['implementation_allowance']))
   if old['selection']=='fastest_under_cap':
    candidates=candidates.loc[candidates.physical_qubits.le(1_000_000)]
    row=select_fastest_with_tolerance(candidates.to_dict('records'))
   else:
    row=candidates.sort_values(['physical_qubits','seconds_per_attempt']).iloc[0].to_dict()
   new={**old,**row}
   new['selection']=old['selection']
   new['successful_shots']=old['successful_shots']
   new['compiled_T']=new['T_count']
   new['complete_task_seconds']=new['seconds_per_attempt']*new['successful_shots']
   new['complete_task_hours']=new['complete_task_seconds']/3600
   assert new['E_total']<=new['implementation_allowance']+1e-14
   sensitivity.append(new)
  pd.DataFrame(sensitivity).to_csv(output/'implementation_budget_sensitivity.csv',index=False)
 (output/'audit.json').write_text(json.dumps({'data_block_rounding':'ceil applied before physical area and error volumes','prep_area_policy':'max(Q_surface,Q_Gidney)','capacity_shots':815,'capacity_time_budget_s':86400,'capacity_N_T_range':[45,125],'sensitivity_N_T_range':[45,125],'error_budget_N_T_range':[45,200],'sampling_budgets_unchanged':True},indent=2)+'\n')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rates", nargs="+", type=int, choices=range(2, 10), default=[5])
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--publish-figure-data", action="store_true", help="Update S8–S10 Source_Data plot CSVs after the complete eight-rate sweep")
    args = parser.parse_args()
    rates = sorted(set(args.rates))
    with staged_sweep(args.output, [k / 10000 for k in rates]) as output:
        generate(rates, output)
    if args.publish_figure_data:
        if rates!=list(range(2,10)):
            raise ValueError('Publishing Fig. S8–S10 requires all physical-error rates 2–9')
        import shutil
        destinations={
            'S08':('implementation_budget_sensitivity.csv',),
            'S09':('native_T_sensitivity.csv','supply_distances.csv','supply_optima.csv','control_distance_curves.csv','control_distance_optima.csv'),
            'S10':('capacity_selected_designs.csv',),
        }
        for figure,names in destinations.items():
            target=figure_data(figure)/'plot_values'
            for name in names:
                if not (target/name).is_file():raise FileNotFoundError(target/name)
                shutil.copyfile(args.output/name,target/name)
        print('Published S8–S10 plot inputs directly into Source_Data figure folders',flush=True)
    print('supplement complete', flush=True)
