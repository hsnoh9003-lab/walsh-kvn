"""Core timing in seconds and surface-code exposure (S56a–S57b and S67a–S69)."""

import math


# Minimum interval between T-state consumptions when factory supply is adequate.
# The interval is limited by either the data-block operation or classical
# reaction, whichever takes longer.
# It is distinct from a factory batch duration and the total circuit runtime.
def core_interval_seconds(b, d_core, t_round, t_react):
    """Return the slower gate/reaction interval in seconds per T-state consumption."""
    assert b > 0 and d_core > 0 and t_round > 0 and t_react >= 0
    return max(float(b) * d_core * t_round, float(t_react))


"""Return additional core exposure per consumed T state in patch-rounds.

tau_T-tau_gate includes both classical-reaction delay and factory-supply
waiting. Supply waiting alone is measured by tau_T-tau_core.
"""


def extra_idle_rounds_per_T(N_core, tau_gate, tau_T, t_round):
    """Patch-rounds beyond b*d_core*t_round per consumed T state (S68–S69)."""
    return N_core * max(0.0, tau_T - tau_gate) / t_round


def conservative_core_error(
    N_T, N_core, V_act, tau_core, tau_T, e_mem, p_L, t_round, b, *, d_core
):
    """Combine patch-cycle and patch-round exposures into per-attempt core error."""
    tau_gate = b * d_core * t_round
    if t_round <= 0 or tau_gate <= 0 or not tau_gate <= tau_core <= tau_T:
        raise ValueError("Invalid core or consumption interval")
    if not (0 <= e_mem < 1 and 0 <= p_L < 1):
        raise ValueError("Invalid memory error probabilities")
    N_T = float(N_T)
    # Baseline active and memory exposures are measured in patch-cycles and
    # therefore use the one-cycle memory error e_mem.
    # Additional reaction and supply exposure is measured in patch-rounds and
    # uses the per-round logical error p_L.
    # Combine their log-survival contributions without mixing exposure units.
    V_idle_base = N_T * float(N_core) * float(b)
    extra_patch_rounds = N_T * extra_idle_rounds_per_T(N_core, tau_gate, tau_T, t_round)
    log_s_attempt = (V_act + V_idle_base) * math.log1p(
        -e_mem
    ) + extra_patch_rounds * math.log1p(-p_L)
    return {
        "T_count_actual": N_T,
        "V_act": V_act,
        "V_idle_base": V_idle_base,
        "V_idle_extra_QEC_rounds": extra_patch_rounds,
        "V_idle": V_idle_base,
        "V_total": V_act + V_idle_base,
        "E_core": -math.expm1(log_s_attempt),
        "delay_factor": tau_T / tau_core,
        "additional_wait_seconds_per_T": max(0.0, tau_T - tau_core),
        "additional_reaction_seconds_per_T": tau_core - tau_gate,
        "error_model": "base active and idle (e_mem) + reaction and supply idle (p_round)",
    }
