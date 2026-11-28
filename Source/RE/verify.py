"""Check reproducibility of stored resource-design evaluations.

Reevaluate stored architectures and synthesis selections with their recorded
calibration inputs, then compare costs with the reference tables. The current
second-order growth designs use the updated calibration; the other publication
records retain the previous calibration where it differs.

Run from the Codes directory:
    python -m Source.RE.verify

This suite does not repeat an exhaustive architecture search.
"""
from __future__ import annotations
from pathlib import Path
from dataclasses import replace
import csv,json,math
from . import model as resource_model
from .preparation import make_config, load_workloads
from .geometry import raw_geometry
from .optimization import select_work
from .pareto import pareto_indices_4d
from Source.data_locations import shared_data

ROOT=Path(__file__).resolve().parents[2]
DATA=shared_data('RE_common','RE')
PREVIOUS_CALIBRATION=shared_data('Gidney_cultivation','update_20261127/previous_calibration')

# Reuse the stored code distances and factory unit counts to recompute area,
# errors, synthesis cost, and runtime.
# Additional checks cover patch rounding, level-local storage, mean routing
# lengths, and a basic Pareto case.
#
# Passing these checks establishes consistency with the stored evaluations.
# It does not independently establish global optimality or validate the
# physical error model.
def check():
    workloads=load_workloads(DATA/'workloads.json')
    totals={'selected':0,'capped_selected':0,'l2':0}
    for tag,filename in [('selected','reference_selected.csv'),
                         ('capped_selected','reference_capped_selected.csv')]:
        with (DATA/filename).open(newline='') as fh:
            for r in csv.DictReader(fh):
                model=r['model'];p_phys=float(r['p_phys']);block=r['block'];
                d_core=int(r['d_core']);d_levels=tuple(map(int,r['distances'].split(':')));
                u_levels=tuple(map(int,r['u_levels'].split(':')));rate_index=round(p_phys*1e4)
                prep_table=DATA/f'prep_p{rate_index}.csv'
                previous=PREVIOUS_CALIBRATION/f'prep_p{rate_index}.csv'
                # Match the input provenance of the record under test, rather
                # than comparing an old design with a different calibration.
                if model!='long298' and previous.exists():
                    prep_table=previous
                cfg=make_config(p_phys,prep_table,block,workloads[model])
                geom=raw_geometry(p_phys,block,cfg,d_core,d_levels,u_levels)
                assert geom is not None,(model,p_phys,block,'missing geometry')
                rec=select_work(geom,model,workloads[model])
                assert rec is not None,(model,p_phys,block,'infeasible precision')
                for key in ('physical_qubits','Q_core','Q_MSF','N_T','T_count',
                       'complete_task_seconds','seconds_per_attempt',
                       'E_core','E_MSF','E_synth','E_total',
                       'core_period_s','service_period_s'):
                    if key in r and r[key] not in (None,''):
                        assert math.isclose(float(r[key]),float(rec[key]),
                            rel_tol=5e-11,abs_tol=3e-10),(tag,model,p_phys,block,key,r[key],rec[key])
                assert math.isclose(float(r['correction_storage_core_tiles']),
                    geom['correction_storage_core_tiles'],rel_tol=1e-13)
                levels=[float(x) for x in str(r['correction_storage_level_tiles']).split(':')]
                got=[float(x) for x in str(geom['correction_storage_level_tiles']).split(':')]
                assert levels==got,(levels,got)
                totals[tag]+=1;totals['l2']+=int(len(d_levels)==2)
    cfg=make_config(.0005,DATA/'prep_p5.csv','fast',workloads['direct'])
    assert cfg.t_round==1e-6 and cfg.t_react==10e-6
    for name, work in workloads.items():
        config=make_config(.0005,DATA/'prep_p5.csv','fast',work)
        e_in=3.2e-6
        output,accepted=resource_model.distillation_level_metrics(e_in,.0005,13,config)
        e_cliff=resource_model.memory_error(13,.0005,config)
        assert output == 35*e_in**3 + 7.1*e_cliff
        expected=1.0-15*e_in-356*e_cliff
        assert abs(accepted-expected) < 2e-16
    for level,spacing in [(1,11),(2,10)]:
        for units in (1,2,3,4,17):
            offsets=[abs(j-(units-1)/2) for j in range(units)]
            mean=1+spacing*sum(offsets)/units
            assert math.isclose(resource_model.mean_silva_path_patches(units,level),mean)
    assert resource_model.data_block_tiles(9,'compact')==17
    assert resource_model.data_block_tiles(9,'fast')==28
    assert resource_model.physical_patch_qubits(17)==2*18*18
    assert resource_model.level1_c_prep(17,cfg)==math.ceil(23/17)
    for t_round in (350e-9,1e-6):
        test=replace(cfg,t_round=t_round)
        for d_levels,u_levels in (((13,),(2,)),((13,27),(20,3)),((17,25),(40,8))):
            factory=[]
            for d_core in (25,35,49):
                q=resource_model.compute_total_qubits(len(d_levels),u_levels,d_levels,d_core=d_core,cfg=test)
                for storage_tiles,distance in zip(q['N_corr_bus_levels'],d_levels):
                    assert storage_tiles==1.5*math.ceil(10e-6/(distance*t_round)-1e-12)
                factory.append(q['Q_MSF'])
            assert math.isclose(min(factory),max(factory),rel_tol=1e-12)
    assert resource_model.reaction_storage_tiles(13,replace(cfg,t_round=350e-9))==4.5
    assert resource_model.reaction_storage_tiles(35,replace(cfg,t_round=350e-9))==1.5
    assert pareto_indices_4d([1,2,3],[1,2,1],[1,1,2],[1,2,1]).tolist()==[0]
    print(json.dumps({'status':'PASS','reference_rows':totals,
        'factory_distance_invariance':'PASS', 'tile_rounding':'PASS',
        'QEC_round':'1us','error_model':'full Silva 15-to-1 acceptance (S48)'},indent=2))

if __name__=='__main__':check()
