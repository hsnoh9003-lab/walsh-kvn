"""Direct v38 optimization API; calculations import one canonical model only.

No Figure or Supplementary modules are imported. CSV output is optional.
"""
from __future__ import annotations
import json,time
from pathlib import Path
import numpy as np
import pandas as pd
from .preparation import make_config
from .geometry import raw_geometry, factory_distance_choices, unit_options
from .pareto import architecture_frontier
from .model import ETA
from .uncapped import find_fastest, select_fastest_with_tolerance

def select_work(geometry, name, workload, n_T=None, eta=ETA):
    """Choose the first feasible n_T and emit the archived design-row schema (S36–S42)."""
    n_T = np.arange(45, 201, dtype=np.int64) if n_T is None else np.asarray(n_T, dtype=np.int64)
    N_T_native, N_Rz = workload["direct_T"], workload["RZ"]
    N_T = N_T_native + N_Rz*n_T
    E_core = -np.expm1(geometry["log_core_survival_per_T"]*N_T)
    E_MSF = -np.expm1(geometry["log_magic_survival_per_T"]*N_T)
    E_synth = N_Rz*np.exp2(-n_T/3.)
    E_model = E_core + E_MSF - E_core*E_MSF + E_synth + workload["canonicalization_error_bound"]
    feasible = np.flatnonzero(E_model <= eta + 1e-15)
    if not len(feasible):
        return None
    j = int(feasible[0])
    tau_T = geometry["service_period_s"]
    t_attempt = float(N_T[j]*tau_T)
    S, P_s = workload["shots"], workload["success"]
    t_task = t_attempt*S/P_s
    # In the archived CSV, N_T denotes n_T and T_count denotes total N_T.
    return dict(geometry, model=name, selection="", N_T=int(n_T[j]), T_count=int(N_T[j]),
                RZ=N_Rz, native_T=N_T_native, E_core=float(E_core[j]), E_MSF=float(E_MSF[j]),
                E_synth=float(E_synth[j]), E_total=float(E_model[j]),
                source_angle_error_bound=workload["canonicalization_error_bound"],
                seconds_per_attempt=t_attempt, successful_shots=S, success_probability=P_s,
                complete_task_seconds=t_task,
                V_act=geometry["core_T_cycle_factor"]*geometry["active_tiles"]*int(N_T[j]),
                V_idle=geometry["core_T_cycle_factor"]*geometry["Q_core_patches"]*int(N_T[j]))

def run_group(name, p_phys, block, prep_file, workload, *, output_dir=None,
              nt_min=45, nt_max=200, d_min=13, d_max=49, q_max=1_000_000):
    """Enumerate a v38 architecture family and jointly select precision.

    Returns (selected, geometries, candidates, diagnostics). Workload is a
    plain dictionary; never modifies global model or module import state.
    The four-dimensional architecture Pareto filter is checked against the
    exhaustive candidate optimum. Returned geometry and candidate tables use the
    requested total-area cap. A minimum-qubit seed is also returned if it lies above it.
    """
    if not q_max > 0:
        raise ValueError("q_max must be positive")
    if nt_min < 1 or nt_max < nt_min:
        raise ValueError("Invalid synthesis precision range")
    cfg=make_config(p_phys,Path(prep_file),block,workload,d_min=d_min,d_max=d_max)
    n_T_grid=np.arange(nt_min,nt_max+1,dtype=np.int64)
    geometries=[];rows=[];tested=0;t0=time.time()
    # The cap applies only to total area for the explicitly capped outputs.
    # Keep a feasible minimum even when every design exceeds the reporting cap.
    outside_minimum = None
    geometry_columns = None
    def footprint(row):
        return row['physical_qubits'], row['seconds_per_attempt']

    for d_core in range(d_min,d_max+1,2):
        for d_levels in factory_distance_choices(d_core,cfg):
            for u_levels in unit_options(p_phys,d_core,d_levels,cfg):
                tested+=1
                try:geometry=raw_geometry(p_phys,block,cfg,d_core,d_levels,u_levels)
                except (ValueError,OverflowError,KeyError,ZeroDivisionError):continue
                if geometry is None:continue
                if output_dir is not None and geometry_columns is None:
                    geometry_columns = list(geometry)
                if geometry['physical_qubits'] > q_max:
                    if rows or (outside_minimum is not None and
                                geometry['physical_qubits'] > outside_minimum['physical_qubits']):
                        continue
                    record = select_work(geometry, name, workload, n_T_grid)
                    if record and (outside_minimum is None or footprint(record) < footprint(outside_minimum)):
                        outside_minimum = record
                    continue
                geometries.append(geometry)
                record=select_work(geometry,name,workload,n_T_grid)
                if record: rows.append(record)
    frontier_indices=architecture_frontier(pd.DataFrame(geometries)) if geometries else []
    allowed={(int(geometries[i]['d_core']),geometries[i]['distances'],geometries[i]['u_levels']) for i in frontier_indices}
    def geometry_key(row):return (int(row['d_core']),str(row['distances']),str(row['u_levels']))
    selected=[]
    if rows:
        from_frontier=[row for row in rows if geometry_key(row) in allowed]
        for label in ('minimum_qubits', 'fastest_under_cap'):
            if label == 'minimum_qubits':
                winner=min(from_frontier,key=footprint);exhaustive=min(rows,key=footprint)
            else:
                winner=select_fastest_with_tolerance(from_frontier)
                exhaustive=select_fastest_with_tolerance(rows)
            if footprint(winner) != footprint(exhaustive):
                raise AssertionError(f'4D Pareto changed {label} optimum for {name},{p_phys},{block}')
            selected.append(dict(winner,selection=label))
    elif outside_minimum is not None:
        selected.append(dict(outside_minimum, selection='minimum_qubits'))
    statistics=dict(model=name,p_phys=p_phys,block=block,tested=tested,valid_geometries=len(geometries),
        feasible_candidates=len(rows),frontier_rows=len(frontier_indices),
        frontier_objectives=['Q_total','e_core_per_T','e_MSF','tau_T'],
        seconds=time.time()-t0)
    if output_dir is not None:
        out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
        prefix=f'{name}_p{round(p_phys*1e4)}_{block}'
        geometry_table = pd.DataFrame(geometries, columns=geometry_columns)
        candidate_columns = list(selected[0]) if selected else None
        geometry_table.to_csv(out/(prefix+'_geometries.csv'),index=False)
        pd.DataFrame(rows, columns=candidate_columns).to_csv(out/(prefix+'_candidates.csv'),index=False)
        geometry_table.iloc[frontier_indices].to_csv(out/(prefix+'_architecture_frontier_4d.csv'),index=False)
        pd.DataFrame(selected, columns=candidate_columns).to_csv(out/(prefix+'_selected.csv'),index=False)
        (out/(prefix+'_statistics.json')).write_text(json.dumps(statistics,indent=2)+'\n')
    return selected,geometries,rows,statistics


def select_designs(name, p_phys, prep_table, workload, *,
                   blocks=("compact", "intermediate", "fast"), Q_cap=1_000_000,
                   n_T_min=45, n_T_max=200, d_min=13, d_max=49,
                   uncapped=True, output_dir=None):
    """Select both task objectives across the requested data-block families."""
    families = []
    diagnostics = []
    for block in blocks:
        selected, _, _, statistics = run_group(
            name, p_phys, block, prep_table, workload, output_dir=output_dir,
            nt_min=n_T_min, nt_max=n_T_max, d_min=d_min, d_max=d_max, q_max=Q_cap,
        )
        families.extend(selected)
        diagnostics.append(statistics)
    capped = [row for row in families if row["physical_qubits"] <= Q_cap]
    eligible = families if uncapped else capped
    if not eligible:
        raise RuntimeError(f"No feasible design in the requested search for {name}, p={p_phys}")
    minimum = min(eligible, key=lambda row: (row["physical_qubits"], row["seconds_per_attempt"]))
    minimum = dict(minimum, selection="minimum_qubits")
    incumbent = select_fastest_with_tolerance(eligible)
    if uncapped:
        fastest, certificate = find_fastest(name, workload, incumbent, prep_table)
    else:
        fastest = dict(incumbent, selection="fastest_under_cap")
        certificate = None
    return minimum, fastest, capped, diagnostics, certificate
