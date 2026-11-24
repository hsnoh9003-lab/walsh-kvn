"""Uncapped fastest-runtime RE search with v38 pruning logic.

Directly imports the single resource model and geometry modules.
No Figure paths, mutable global workload registry, or legacy RE backend.
"""
from __future__ import annotations
from itertools import count
import math
from .model import ModelConfig, block_cycles_per_T, evaluate_factory_distances
from .preparation import make_config
from .geometry import factory_distance_choices, raw_geometry
from .model import ETA
from .factory_sizing import interlevel_units
from .core_timing import core_interval_seconds
DECAY=math.log(2)/3
BLOCKS=('fast','intermediate','compact')
FASTEST_RELATIVE_TIME_TOLERANCE = 1e-6


def synthesis_floor(circuit):
    eta_available = ETA - circuit["canonicalization_error_bound"]
    return math.ceil(math.log(circuit["RZ"] / eta_available) / DECAY)

def errors(geometry, circuit, n_T):
    N_T = circuit["direct_T"] + circuit["RZ"] * n_T
    E_core = -math.expm1(N_T * geometry["log_core_survival_per_T"])
    E_MSF = -math.expm1(N_T * geometry["log_magic_survival_per_T"])
    E_synth = circuit["RZ"] * 2 ** (-n_T / 3)
    E_model = E_core + E_MSF - E_core * E_MSF + E_synth + circuit["canonicalization_error_bound"]
    return E_core, E_MSF, E_synth, E_model

def first_feasible_precision(geometry, circuit, n_min):
    """Return the smallest feasible synthesis cost n_T."""
    hazard = -geometry["log_core_survival_per_T"] - geometry["log_magic_survival_per_T"]
    if hazard == 0:
        return n_min
    if hazard * circuit["RZ"] >= DECAY:
        return None
    n_opt = (math.log(DECAY / hazard) + hazard * circuit["direct_T"]) / (
        DECAY - hazard * circuit["RZ"]
    )
    n_at_min_error = min(
        (max(n_min, math.floor(n_opt)), max(n_min, math.ceil(n_opt))),
        key=lambda n_T: errors(geometry, circuit, n_T)[-1],
    )
    if errors(geometry, circuit, n_at_min_error)[-1] > ETA:
        return None
    low, high = n_min, n_at_min_error
    while low < high:
        mid = (low + high) // 2
        if errors(geometry, circuit, mid)[-1] <= ETA:
            high = mid
        else:
            low = mid + 1
    return low

def fastest_key(row):
    return (row["seconds_per_attempt"], row["physical_qubits"], row["block"],
            row["d_core"], row["distances"], row["u_levels"])

def fastest_time_limit(minimum_time, relative_tolerance=FASTEST_RELATIVE_TIME_TOLERANCE):
    """Allow a relative runtime difference from the true minimum."""
    if not 0 <= relative_tolerance < math.inf:
        raise ValueError("Relative time tolerance must be finite and nonnegative")
    return minimum_time * (1 + relative_tolerance)


def select_fastest_with_tolerance(rows, relative_tolerance=FASTEST_RELATIVE_TIME_TOLERANCE):
    """Minimize qubits among designs within relative tolerance of fastest time.

    Compare against the global minimum runtime, not against the previous
    candidate: pairwise approximate ties are not transitive and would make the
    result depend on enumeration order.
    """
    rows = list(rows)
    if not rows:
        raise ValueError("At least one feasible design is required")
    minimum_time = min(row["seconds_per_attempt"] for row in rows)
    limit = fastest_time_limit(minimum_time, relative_tolerance)
    eligible = (row for row in rows
                if row["seconds_per_attempt"] <= limit)
    return min(eligible, key=lambda row: (row["physical_qubits"],) + fastest_key(row))

def make_record(model, geometry, circuit, n_T):
    N_T = circuit['direct_T'] + circuit['RZ'] * n_T
    E_core, E_MSF, E_synth, E_model = errors(geometry, circuit, n_T)
    tau_T = geometry['service_period_s']
    S, P_s = circuit['shots'], circuit['success']
    t_attempt = N_T * tau_T
    t_accepted_shot = t_attempt / P_s
    t_task = t_accepted_shot * S
    return dict(
        geometry, model=model, selection='fastest_uncapped',
        RZ=circuit['RZ'], native_T=circuit['direct_T'], N_T=n_T, T_count=N_T,
        E_core=E_core, E_MSF=E_MSF, E_synth=E_synth, E_total=E_model,
        source_angle_error_bound=circuit['canonicalization_error_bound'],
        seconds_per_attempt=t_attempt, seconds_per_accepted_shot=t_accepted_shot,
        successful_shots=S, success_probability=P_s,
        complete_task_seconds=t_task,
        shot_basis=circuit['shot_basis'], workload_scope=circuit['workload_scope'],
        V_act=geometry['core_T_cycle_factor'] * geometry['active_tiles'] * N_T,
        V_idle=geometry['core_T_cycle_factor'] * geometry['Q_core_patches'] * N_T,
    )


def find_fastest(model, circuit, incumbent, prep):
    p_phys = incumbent['p_phys']
    n_min = max(circuit['precision_min'], synthesis_floor(circuit))
    N_T_min = circuit['direct_T'] + circuit['RZ'] * n_min
    
    # Each consumption interval is at least b * d_core * t_round. The initial design supplies a
    # feasible time bound even with unlimited area.
    t_round = ModelConfig.t_round
    d_max = math.ceil(fastest_time_limit(incumbent['seconds_per_attempt'])
                      / (N_T_min * t_round))
    if d_max % 2 == 0:
        d_max += 1
    incumbent_block = incumbent['block']
    incumbent_cfg = make_config(p_phys, prep, incumbent_block, circuit, d_max=d_max)
    incumbent_d_levels = tuple(map(int, incumbent['distances'].split(':')))
    incumbent_u_levels = tuple(map(int, incumbent['u_levels'].split(':')))
    incumbent_geometry = raw_geometry(p_phys, incumbent_block, incumbent_cfg, int(incumbent['d_core']),
                                         incumbent_d_levels, incumbent_u_levels)
    best = make_record(model, incumbent_geometry, circuit, int(incumbent['N_T']))
    assert best['physical_qubits'] == incumbent['physical_qubits']
    assert math.isclose(best['seconds_per_attempt'], incumbent['seconds_per_attempt'], rel_tol=1e-13)
    minimum_time = best['seconds_per_attempt']
    near_fastest = [best]
    visited = {}

    for block in BLOCKS:
        config = make_config(p_phys, prep, block, circuit, d_max=d_max)
        assert config.t_round == t_round
        b = block_cycles_per_T(block)
        checked = 0
        for d_core in range(config.d_min, d_max + 1, 2):
            tau_core = core_interval_seconds(b, d_core, config.t_round, config.t_react)
            if N_T_min * tau_core > fastest_time_limit(minimum_time):
                break
            for d_levels in factory_distance_choices(d_core, config):
                factory = evaluate_factory_distances(p_phys, d_core, d_levels, config)
                if factory is None:
                    continue
                for u_top in count(1):
                    u_levels = interlevel_units(p_phys, d_core, factory, u_top, config)
                    if u_levels is None:
                        break
                    geometry = raw_geometry(p_phys, block, config, d_core, d_levels, u_levels)
                    if geometry is None:
                        break
                    checked += 1
                    # With fixed distances, more units increase the connection
                    # error. A factory that already exceeds the entire budget
                    # cannot become feasible by adding units.
                    if -math.expm1(N_T_min * geometry['log_magic_survival_per_T']) > ETA:
                        break
                    if N_T_min * geometry['service_period_s'] <= fastest_time_limit(minimum_time):
                        n_T = first_feasible_precision(geometry, circuit, n_min)
                        if n_T is not None and n_T <= circuit['precision_max']:
                            candidate = make_record(model, geometry, circuit, n_T)
                            runtime = candidate['seconds_per_attempt']
                            if runtime < minimum_time:
                                minimum_time = runtime
                                near_fastest = [row for row in near_fastest if
                                    row['seconds_per_attempt']
                                    <= fastest_time_limit(minimum_time)]
                            if runtime <= fastest_time_limit(minimum_time):
                                near_fastest.append(candidate)
                    # Extra factories after saturation lengthen the bus and
                    # increase area while the core service period stays fixed.
                    if geometry['service_period_s'] == tau_core:
                        break
        visited[block] = checked

    best = select_fastest_with_tolerance(near_fastest)
    assert best['E_total'] <= ETA
    assert best['seconds_per_attempt'] <= fastest_time_limit(minimum_time)
    return best, dict(model=model, p_phys=p_phys, synthesis_floor=n_min,
                      d_core_upper_bound=d_max, checked_geometries=visited,
                      incumbent_seconds_per_attempt=incumbent['seconds_per_attempt'],
                      absolute_minimum_seconds_per_attempt=minimum_time,
                      fastest_relative_time_tolerance=FASTEST_RELATIVE_TIME_TOLERANCE,
                      near_fastest_candidate_count=len(near_fastest))
