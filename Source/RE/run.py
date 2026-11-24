"""Standalone v38 resource estimation entrypoint.

Usage with existing preparation CSVs (source data remains external):
  python -m Source.RE.run --output Build/Full_RE

Expensive complete enumeration is explicitly invoked, never during plotting.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
from Source.data_locations import shared_data
from .preparation import load_workloads, write_csv
from .optimization import select_designs

def sweep(workloads,prep_by_rate,out,*,rates=range(2,10),blocks=('compact','intermediate','fast'),
          q_max=1_000_000,nt_min=45,nt_max=200,d_min=13,d_max=49,
          uncapped=True):
    """Recompute architectures for plain-dictionary workloads and preparation paths.

    Write both the global optima and capped per-block resource records.
    """
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    selected=[];capped=[];audit={}
    for rate_index in rates:
        p_phys=rate_index/10000;prep_table=Path(prep_by_rate[rate_index])
        for name,workload in workloads.items():
            minimum, fastest, capped_rows, _, certificate = select_designs(
                name, p_phys, prep_table, workload, blocks=blocks,
                n_T_min=nt_min, n_T_max=nt_max, d_min=d_min, d_max=d_max, Q_cap=q_max,
                uncapped=uncapped, output_dir=out/f"{name}_p{rate_index}",
            )
            capped.extend(capped_rows)
            if certificate is not None:
                audit[f"{name}_p{rate_index}"] = certificate
            selected.extend((minimum,fastest))
    write_csv(out/'selected.csv',selected)
    write_csv(out/'capped_selected.csv',capped,columns=list(selected[0]) if selected and not capped else None)
    (out/'audit.json').write_text(json.dumps(dict(uncapped_fastest_audits=audit,
        data_block_rounding='ceil',distance_search=[d_min,d_max,2],
        synthesis_precision=[nt_min,nt_max],distillation_levels_max=2,
        capped_design_qubit_limit=q_max,
        final_fastest_qubit_cap=None if uncapped else q_max),indent=2)+'\n')
    return selected

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workloads',type=Path,default=shared_data('RE_common','RE','workloads_v38.json'))
    ap.add_argument('--prep-dir',type=Path,default=shared_data('RE_common','RE'))
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--p',nargs='+',type=int,default=list(range(2,10)))
    ap.add_argument('--models',nargs='+')
    ap.add_argument('--blocks',nargs='+',choices=['compact','intermediate','fast'],default=['compact','intermediate','fast'])
    ap.add_argument('--capped-only',action='store_true')
    args=ap.parse_args()
    workloads=load_workloads(args.workloads)
    if args.models: workloads={n:workloads[n] for n in args.models}
    prep={p:args.prep_dir/f'prep_p{p}.csv' for p in args.p}
    missing=[str(v) for v in prep.values() if not v.is_file()]
    if missing:raise FileNotFoundError('Preparation input CSVs not provided: '+', '.join(missing))
    sweep(workloads,prep,args.output,rates=args.p,blocks=args.blocks,uncapped=not args.capped_only)

if __name__=='__main__':main()
