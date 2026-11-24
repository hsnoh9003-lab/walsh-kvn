"""Reoptimize the saved resource designs after a factory-calibration update.

The raw supplied cultivation records remain inputs, not newly simulated data.
All architecture searches use the same canonical model and workload inventory.
Run from the package root with --apply to update the reference CSVs.
"""
from pathlib import Path
import argparse, json, time
from .preparation import load_workloads, make_prep_table, read_csv, write_csv, design_key
from .optimization import select_designs
from Source.data_locations import shared_data

ROOT=Path(__file__).resolve().parents[2]
RAW_DIRECTORY = shared_data('Gidney_cultivation','raw')
RE_DIRECTORY = shared_data('RE_common','RE')


def refresh_preparation_inputs(rates):
    """Regenerate the per-rate source views and preparation tables from the supplied shots."""
    for rate in sorted(set(rates)):
        p_phys = rate / 10000
        raw = RAW_DIRECTORY / f'custom_p{rate}e-4_d1_3_d2.csv'
        make_prep_table(raw, p_phys, RE_DIRECTORY / f'prep_p{rate}.csv')


def run(rates,models,output,apply=False):
    workloads=load_workloads(RE_DIRECTORY/'workloads_v38.json')
    selected=[];capped=[];diagnostics=[];certificates={};start=time.time()
    for rate_index in rates:
        for name in models:
            prep_table = RE_DIRECTORY/f'prep_p{rate_index}.csv'
            minimum, fastest, capped_rows, statistics, certificate = select_designs(
                name, rate_index/10000, prep_table, workloads[name],
            )
            capped.extend(capped_rows)
            diagnostics.extend(statistics)
            for row in statistics:
                print(name,rate_index,row["block"],"seconds",round(row["seconds"],2),flush=True)
            certificates[f'{name}_p{rate_index}']=certificate
            selected.extend((minimum,fastest))
            write_csv(output/'updated_selected.csv',selected)
            write_csv(output/'updated_capped_selected.csv',capped,columns=list(minimum) if not capped else None)
            (output/'search_audit.json').write_text(json.dumps(dict(
                elapsed_seconds=time.time()-start,completed_model_rates=len(selected)//2,
                diagnostics=diagnostics,uncapped_certificates=certificates),indent=2)+'\n')
            print('SELECTED',name,rate_index,minimum['physical_qubits'],fastest['physical_qubits'],
                  fastest['complete_task_seconds']/3600,flush=True)
    if apply:
        for filename,updates,is_capped in [('v38_reference_selected.csv',selected,False),
                ('v38_reference_capped_selected.csv',capped,True)]:
            old=read_csv(RE_DIRECTORY/filename);by_key={design_key(row,is_capped):row for row in updates}
            count=0;merged=[]
            for row in old:
                replacement=by_key.get(design_key(row,is_capped))
                if replacement:
                    row={**row,**replacement};count+=1
                    for k,scale in [('complete_task_hours',3600),('complete_task_days',86400),
                                    ('complete_task_years',365.25*86400)]:
                        if k in row:row[k]=float(row['complete_task_seconds'])/scale
                    if 'seconds_per_accepted_shot' in row:
                        row['seconds_per_accepted_shot']=float(row['seconds_per_attempt'])/float(row['success_probability'])
                elif is_capped and row['model'] in models and round(float(row['p_phys'])*10000) in rates:
                    # This block has no feasible design at the requested total-area cap.
                    continue
                merged.append(row)
            assert count==len(updates),(filename,count,len(updates))
            write_csv(RE_DIRECTORY/filename,merged,columns=list(old[0]) if old and not merged else None)
    return selected,capped

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rates',type=int,nargs='+',default=[2,3,4,6,7,8,9])
    p.add_argument('--models',nargs='+',default=list(load_workloads(RE_DIRECTORY/'workloads_v38.json')))
    p.add_argument('--output',type=Path,default=ROOT/'Build/Gidney_Input_Update')
    p.add_argument('--apply',action='store_true')
    a=p.parse_args();a.output.mkdir(exist_ok=True,parents=True)
    refresh_preparation_inputs(a.rates)
    run(a.rates,a.models,a.output,a.apply)
