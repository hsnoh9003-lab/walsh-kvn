"""Evaluate supply policies and step capacity using the common RE model.

These functions perform numerical calculations without plotting or
automatically creating data files.
"""

import numpy as np
from .model import ETA

# The baseline sensitivity workload contains 4803 arbitrary Rz rotations
# and requires 815 accepted samples.
# Its default synthesis range is 45 through 125, whereas the general
# resource search uses 45 through 200.
# Applying these routines to another workload requires revisiting the
# rotation count, sampling requirement, and any additional angle-error bound.
N_Rz_baseline = 4803
n_T_grid_baseline = np.arange(45, 126)
BLOCKS = ("compact", "intermediate", "fast")
S_BASELINE = 815


def first_feasible_cost_indices(log_s_core, log_s_MSF, N_T_native, N_Rz, n_T_grid, eta):
    """Exact finite synthesis search, vectorized over unresolved architectures.

    Costs whose synthesis error alone exceeds the budget cannot be feasible.
    Every remaining declared cost is visited in ascending order, so this is
    the same first-feasible selection as the original scalar/Numba loop.
    """
    out = np.full(len(log_s_core), -1, dtype=np.int32)
    for j, n_T in enumerate(n_T_grid):
        E_synth = N_Rz * 2 ** (-float(n_T) / 3.0)
        if E_synth > eta + 1e-15:
            continue
        unresolved = np.flatnonzero(out < 0)
        if not len(unresolved):
            break
        N_T = N_T_native + N_Rz * n_T
        E_core = -np.expm1(log_s_core[unresolved] * N_T)
        E_MSF = -np.expm1(log_s_MSF[unresolved] * N_T)
        feasible = E_core + E_MSF - E_core * E_MSF + E_synth <= eta + 1e-15
        out[unresolved[feasible]] = j
    return out


def pick(designs, policy):
    # supply_enforced selects among designs that meet the core's mean demand.
    # delay_allowed selects among designs with exactly one top-level unit.
    # The latter is a restricted one-unit comparison, not an unrestricted
    # optimization over all architectures that permit waiting.
    # Both policies minimize qubit count before considering runtime.
    eligible = (
        designs[designs.n_units.eq(1)].copy()
        if policy == "delay_allowed"
        else designs[designs.supply_mode == "enforced"]
    )
    if eligible.empty:
        return None
    selected = (
        eligible.sort_values(
            ["physical_qubits", "seconds_per_attempt", "d_core", "block"]
        )
        .iloc[0]
        .to_dict()
    )
    return dict(policy=policy, **selected)


def scan_pair_fast(geometries, N_T_native):
    """Choose minimum-qubit policies without materializing full feasible DataFrames.

    The lexicographic selection uses (physical_qubits,
    seconds_per_attempt, d_core, block). All error and timing expressions match.
    """
    log_s_core = geometries.log_core_survival_per_T.to_numpy(dtype=np.float64)
    log_s_MSF = geometries.log_magic_survival_per_T.to_numpy(dtype=np.float64)
    Q = geometries.physical_qubits.to_numpy(dtype=np.int64)
    tau_T = geometries.service_period_s.to_numpy(dtype=np.float64)
    d_core = geometries.d_core.to_numpy(dtype=np.int32)
    u_top = geometries.n_units.to_numpy(dtype=np.int32)
    supply_met = geometries.supply_mode.eq("enforced").to_numpy()
    j = first_feasible_cost_indices(
        log_s_core, log_s_MSF, int(N_T_native), N_Rz_baseline, n_T_grid_baseline, ETA
    )
    valid = j >= 0
    if not valid.any():
        return None, None
    t_attempt = np.where(
        valid,
        (N_T_native + N_Rz_baseline * n_T_grid_baseline[np.maximum(0, j)]) * tau_T,
        np.inf,
    )

    def one(mask, policy):
        ids = np.flatnonzero(valid & mask)
        if len(ids) == 0:
            return None
        # pandas sorts tie-breakers in the same order; no substantive ties expected.
        ranks = np.lexsort(
            (geometries.block.to_numpy()[ids], d_core[ids], t_attempt[ids], Q[ids])
        )
        ii = int(ids[int(ranks[0])])
        idx = int(j[ii])
        N_T = int(N_T_native + N_Rz_baseline * n_T_grid_baseline[idx])
        E_core = float(-np.expm1(log_s_core[ii] * N_T))
        E_MSF = float(-np.expm1(log_s_MSF[ii] * N_T))
        E_synth = float(N_Rz_baseline * np.exp2(-n_T_grid_baseline[idx] / 3.0))
        E_model = E_core + E_MSF - E_core * E_MSF + E_synth
        r = geometries.iloc[ii].to_dict()
        r.update(
            policy=policy,
            native_T=int(N_T_native),
            RZ=N_Rz_baseline,
            synthesis_T_per_RZ=int(n_T_grid_baseline[idx]),
            T_core=N_T,
            E_core=E_core,
            E_MSF=E_MSF,
            E_synth=E_synth,
            E_total=float(E_model),
            seconds_per_attempt=float(t_attempt[ii]),
            complete_task_seconds=S_BASELINE * float(t_attempt[ii]),
            mean_supply_wait_per_T_s=float(
                tau_T[ii] - geometries.iloc[ii].core_period_s
            ),
        )
        return r

    enforced = one(supply_met, "supply_enforced")
    waiting = one(u_top == 1, "delay_allowed")
    return enforced, waiting


def compare(enforced, waiting):
    return dict(
        p_phys=enforced["p_phys"],
        native_T=int(enforced["native_T"]),
        RZ=N_Rz_baseline,
        strict_Q=int(enforced["physical_qubits"]),
        relaxed_Q=int(waiting["physical_qubits"]),
        strict_time_s=enforced["seconds_per_attempt"],
        relaxed_time_s=waiting["seconds_per_attempt"],
        qubit_change_pct=100
        * (waiting["physical_qubits"] / enforced["physical_qubits"] - 1),
        runtime_change_pct=100
        * (waiting["seconds_per_attempt"] / enforced["seconds_per_attempt"] - 1),
        strict_d_core=int(enforced["d_core"]),
        relaxed_d_core=int(waiting["d_core"]),
        strict_factory_units=int(enforced["n_units"]),
        relaxed_factory_units=int(waiting["n_units"]),
        strict_factory_distances=enforced["distances"],
        relaxed_factory_distances=waiting["distances"],
        strict_T_per_RZ=int(enforced["synthesis_T_per_RZ"]),
        relaxed_T_per_RZ=int(waiting["synthesis_T_per_RZ"]),
        strict_T_core=int(enforced["T_core"]),
        relaxed_T_core=int(waiting["T_core"]),
        strict_core_period_s=enforced["core_period_s"],
        relaxed_core_period_s=waiting["core_period_s"],
        strict_service_period_s=enforced["service_period_s"],
        relaxed_service_period_s=waiting["service_period_s"],
        strict_wait_s=enforced["mean_supply_wait_per_T_s"],
        relaxed_wait_s=waiting["mean_supply_wait_per_T_s"],
        strict_block=enforced["block"],
        relaxed_block=waiting["block"],
    )


def joint_capacity(
    frame,
    scheme,
    n_T_min,
    n_T_max,
    t_budget_per_accepted,
    N_Rz_33,
    N_Rz_step,
    N_T_native_33,
    P_s,
):
    """Maximize the feasible step count over architectures and synthesis costs.

    Split the 33-step inventory into initialization and repeated-step costs.
    The native T inventory must divide into 33 identical steps, giving zero
    native-T initialization cost in the current decomposition.

    Keep the supplied circuit acceptance probability fixed as the step count
    changes. Enforce both implementation-error and expected-runtime budgets.

    The resulting capacity does not independently certify physical-field
    accuracy at the extended evolution time.
    """
    if not (0 < P_s <= 1):
        raise ValueError("P_good must be in (0, 1]")
    N_T_native_step = N_T_native_33 // 33
    if N_T_native_step * 33 != N_T_native_33:
        raise ValueError("Native T inventory must divide into 33 identical steps")
    N_Rz_init = N_Rz_33 - 33 * N_Rz_step
    N_T_native_init = N_T_native_33 - 33 * N_T_native_step
    assert N_Rz_init >= 0 and N_T_native_init >= 0
    tau_T = frame.service_period_s.to_numpy(dtype=float)[None, :]
    log_s_per_T = (
        frame.log_core_survival_per_T + frame.log_magic_survival_per_T
    ).to_numpy(dtype=float)[None, :]
    n_T_grid = np.arange(n_T_min, n_T_max + 1, dtype=np.int64)[:, None]
    epsilon_R = np.exp2(-n_T_grid.astype(float) / 3)
    N_T_step = N_T_native_step + N_Rz_step * n_T_grid
    N_T_init = N_T_native_init + N_Rz_init * n_T_grid
    # The caller supplies the total task-time budget divided by the required
    # number of accepted samples.
    # Since one accepted sample requires attempt_runtime/P_s time in expectation,
    # the corresponding attempt-time allowance is t_budget_per_accepted*P_s.
    time_limited_steps = np.floor(
        (t_budget_per_accepted * P_s / tau_T - N_T_init) / N_T_step
    ).astype(np.int64)

    def error(k):
        N_T = N_T_init + N_T_step * k
        N_Rz = N_Rz_init + N_Rz_step * k
        return -np.expm1(log_s_per_T * N_T) + N_Rz * epsilon_R

    init_valid = error(0) <= ETA
    valid = (time_limited_steps >= 0) & init_valid
    # Find the largest error-feasible integer K in [0,k_time], separately for
    # every architecture and n_T.  Since error(K) is monotonic, bisection is exact.
    hi = np.maximum(time_limited_steps + 1, 0)
    lo = np.zeros_like(hi)
    while np.any(hi - lo > 1):
        mid = (hi + lo) // 2
        yes = error(mid) <= ETA
        lo = np.where(yes, mid, lo)
        hi = np.where(yes, hi, mid)
    best_k = np.where(valid, lo, -1)
    N_t_max = int(best_k.max())
    if N_t_max < 0:
        return None
    # Tie-break 1: shorter runtime; 2: fewer qubits; 3: smaller n_T;
    # 4: deterministic source row order.
    ni, gi = np.nonzero(best_k == N_t_max)
    runtimes = (N_T_init[ni, 0] + N_T_step[ni, 0] * N_t_max) * tau_T[0, gi] / P_s
    qubits = frame.physical_qubits.to_numpy(dtype=np.int64)[gi]
    order = np.lexsort((gi, n_T_grid[ni, 0], qubits, runtimes))
    idx = int(order[0])
    nidx, gidx = int(ni[idx]), int(gi[idx])
    n_T = int(n_T_grid[nidx, 0])
    row = frame.iloc[gidx]
    N_T = int(N_T_init[nidx, 0] + N_T_step[nidx, 0] * N_t_max)
    t_attempt = float(N_T * tau_T[0, gidx])
    t_accepted_shot = t_attempt / P_s
    E_model = float(error(N_t_max)[nidx, gidx])
    assert t_accepted_shot <= t_budget_per_accepted + 1e-7 and E_model <= ETA + 1e-13
    t_attempt_next = float(
        (N_T_init[nidx, 0] + N_T_step[nidx, 0] * (N_t_max + 1)) * tau_T[0, gidx]
    )
    t_accepted_next = t_attempt_next / P_s
    E_model_next = float(error(N_t_max + 1)[nidx, gidx])
    assert t_accepted_next > t_budget_per_accepted - 1e-7 or E_model_next > ETA - 1e-13
    return {
        **row.to_dict(),
        "scheme": scheme,
        "N_T": n_T,
        "max_steps": N_t_max,
        "max_runtime_seconds": t_accepted_shot,
        "attempt_runtime_seconds": t_attempt,
        "accepted_shot_runtime_seconds": t_accepted_shot,
        "good_branch_probability": P_s,
        "t_step_seconds": float(N_T_step[nidx, 0] * tau_T[0, gidx] / P_s),
        "t_init_seconds": float(N_T_init[nidx, 0] * tau_T[0, gidx] / P_s),
        "t_step_attempt_seconds": float(N_T_step[nidx, 0] * tau_T[0, gidx]),
        "t_init_attempt_seconds": float(N_T_init[nidx, 0] * tau_T[0, gidx]),
        "E_total_at_max": E_model,
        "E_total_next_step": E_model_next,
        "next_step_runtime_seconds": float(t_accepted_next),
        "next_step_attempt_seconds": t_attempt_next,
        "RZ_total33": N_Rz_33,
        "native_T_total33": N_T_native_33,
        "RZ_step": N_Rz_step,
        "native_T_step": N_T_native_step,
        "RZ_init": N_Rz_init,
        "native_T_init": N_T_native_init,
        "total_T_count_at_max": N_T,
        "E_synth_at_max": float((N_Rz_init + N_Rz_step * N_t_max) * epsilon_R[nidx, 0]),
        "E_stochastic_at_max": float(-np.expm1(log_s_per_T[0, gidx] * N_T)),
    }
