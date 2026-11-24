"""Input preparation and explicit per-workload v38 RE configuration.

The compiled logical width comes directly from each workload inventory.
"""
from __future__ import annotations
from pathlib import Path
import csv
import numpy as np
from . import model

def read_csv(path):
    with Path(path).open(newline='',encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

def write_csv(path,rows,columns=None):
    path.parent.mkdir(parents=True,exist_ok=True)
    if columns is None:
        columns=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)

def make_prep_table(path,p_phys,out):
    rows=read_csv(path)
    if {'degree','e_prep','P_prep'}.issubset(rows[0]):
        return path
    required={'p_phys','d1','d2','r1','r2','gap_db','shots','discards','kept',
              'logical_errors','P_prep','e_out_S','e_out_S_95_lo','e_out_S_95_hi',
              'e_out_T_est','e_out_T_est_95_lo','e_out_T_est_95_hi'}
    if not required.issubset(rows[0]): raise ValueError('Missing measured prep columns in '+str(path))
    # A supplied sweep can contain several physical error probabilities.
    rows = [row for row in rows if float(row['p_phys']) == p_phys]
    if not rows:
        raise ValueError(f'No raw preparation records for p_phys={p_phys} in {path}')
    best={}
    for r in rows:
        d=int(float(r['d2'])); P_prep=float(r['P_prep']); e_prep=float(r['e_out_T_est'])
        if P_prep<=0 or not np.isfinite(e_prep) or e_prep<0: continue
        rank=(e_prep,-P_prep,float(r['gap_db']))
        if d not in best or rank<best[d][0]: best[d]=(rank,r)
    chosen=[]
    for d,(_,r) in sorted(best.items()):
        chosen.append(dict(degree=d,d2=d,p_phys=p_phys,cultivation_d1=int(float(r['d1'])),
          r1=r['r1'],r2=r['r2'],gap_db=r['gap_db'],shots=r['shots'],discards=r['discards'],
          kept=r['kept'],errors=r['logical_errors'],P_prep=r['P_prep'], expected_attempts=1/float(r['P_prep']),
          e_S_sim=r['e_out_S'], e_S_95_lo=r['e_out_S_95_lo'], e_S_95_hi=r['e_out_S_95_hi'],
          e_prep=r['e_out_T_est'],e_prep_95_lo=r['e_out_T_est_95_lo'], e_prep_95_hi=r['e_out_T_est_95_hi'],
          T_over_S_factor=2.0, selection_rule='min_estimated_T_error_then_max_P_prep'))
    write_csv(out,chosen)
    return out

def make_config(p_phys, table, block, workload, *, d_min=13, d_max=49):
    cfg = model.ModelConfig(
        n_L=workload["logical_qubits"], data_block=block, p_phys=p_phys,
        d_min=d_min, d_max=d_max, prep_degree_table_path=str(Path(table).resolve()),
    )
    cfg.validate()
    return cfg

def load_workloads(path):
    """Load current_models.json without hard-coding a Figure or workload name."""
    import json
    raw=json.loads(Path(path).read_text())
    clean={}
    for name,r in raw.items():
        is_wake = r.get('is_wake', name in {'direct', 'generator', 'long298', 'restart40'})
        if not isinstance(is_wake, bool):
            raise ValueError(f'is_wake must be a Boolean for {name}')
        clean[name]=dict(logical_qubits=int(r['logical_qubits']), is_wake=is_wake,
            RZ=int(float(r['RZ'])),
            direct_T=int(r['native_T']),Clifford=int(r.get('Clifford',0)),
            shots=int(r['successful_shots']),success=float(r['success_probability']),
            canonicalization_error_bound=float(r.get('source_angle_error_bound') or 0.0),
            precision_min=45,precision_max=200,
            shot_basis='accepted samples',
            workload_scope='published nonlinear example' if name=='nj_lchs'
                else 'same ensemble field-error task')
    return clean


def design_key(row, capped=False):
    selection = row["selection"]
    if not capped and row["model"] == "2d_interior" and selection == "fastest_under_cap":
        selection = "fastest_uncapped"
    return (row["model"], round(float(row["p_phys"]) * 10000), selection,
            row["block"] if capped else "")
