"""Evaluate core and distillation architectures using the adopted Silva model."""

import math
from pathlib import Path
from . import model
from .factory_sizing import interlevel_units, steady_state_units
from .core_timing import core_interval_seconds, extra_idle_rounds_per_T


# Select the first-level distance from measured preparation data.
# For two-level factories, enumerate odd distances satisfying
# d_1 <= d_2 <= d_core.
#
# This enumerator currently generates one- and two-level factories.
# Increasing ModelConfig.L_max alone does not implement a three-level search.
def factory_distance_choices(d_core, cfg):
    """Enumerate measured first-level and ordered second-level code distances."""
    measured = model.gidney_degree_table(cfg)
    d_first = sorted(d for d in measured if cfg.d_min <= d <= d_core and d % 2 == 1)
    for d_1 in d_first:
        yield (d_1,)
    if cfg.L_max >= 2:
        for d_1 in d_first:
            for d_2 in range(d_1, d_core + 1, 2):
                yield (d_1, d_2)


def raw_geometry(p_phys, block, cfg, d_core, d_levels, u_levels):
    """Evaluate architecture area, supply rate and per-T log survival; reject starvation."""
    factory = model.evaluate_factory_distances(p_phys, d_core, d_levels, cfg, u_levels)
    if factory is None:
        return None
    tau_levels = [
        model.distillation_duration(i + 1, d_l, cfg) for i, d_l in enumerate(d_levels)
    ]
    P_levels = factory["P_levels"]
    if len(d_levels) == 2:
        D_1 = u_levels[0] * P_levels[0] / tau_levels[0]
        C_2 = cfg.distill_input_count * u_levels[1] / tau_levels[1]
        if D_1 + 1e-18 < C_2:
            return None
    area = model.compute_total_qubits(
        len(d_levels), u_levels, d_levels, d_core=d_core, cfg=cfg
    )
    b = model.block_cycles_per_T(block)
    # tau_gate: nominal data-block operation interval.
    # tau_core: minimum core interval including classical reaction.
    # tau_MSF: mean interval between usable final-factory outputs.
    # tau_T: actual consumption interval, max(tau_core, tau_MSF).
    #
    # D_L already includes the final-level acceptance probability.
    # Do not apply the same acceptance penalty again to the circuit runtime.
    tau_gate = b * d_core * cfg.t_round
    tau_core = core_interval_seconds(b, d_core, cfg.t_round, cfg.t_react)
    D_L = u_levels[-1] * P_levels[-1] / tau_levels[-1]
    tau_MSF = 1 / D_L
    tau_T = max(tau_core, tau_MSF)
    N_active = model.data_block_tiles(cfg.n_L, block) + model.CORE_FIXED_TILES
    N_core = area["Q_core"] / model.physical_patch_qubits(d_core)
    e_mem = model.memory_error(d_core, p_phys, cfg)
    p_L = model.logical_error_per_round(p_phys, d_core, cfg)
    e_MSF = factory["e_MSF"]
    # Store core reliability as a log-survival contribution per consumed T state.
    # The circuit's total T-count is applied later during workload evaluation.
    # This permits the same architecture to be evaluated at different synthesis
    # costs and circuit lengths.
    # The per-attempt core error is -expm1(N_T*log_s_core).
    extra_rounds = extra_idle_rounds_per_T(N_core, tau_gate, tau_T, cfg.t_round)
    log_s_core = b * (N_active + N_core) * math.log1p(
        -e_mem
    ) + extra_rounds * math.log1p(-p_L)
    return dict(
        p_phys=p_phys,
        block=block,
        logical_qubits=cfg.n_L,
        physical_qubits=math.ceil(area["Q"]),
        Q_core=area["Q_core"],
        Q_MSF=area["Q_MSF"],
        QEC_round_seconds=cfg.t_round,
        correction_storage_core_tiles=area["N_corr_bus_core"],
        correction_storage_level_tiles=":".join(map(str, area["N_corr_bus_levels"])),
        d_core=d_core,
        L=len(d_levels),
        distances=":".join(map(str, d_levels)),
        u_levels=":".join(map(str, u_levels)),
        n_units=u_levels[-1],
        core_period_s=tau_core,
        service_period_s=tau_T,
        delay_factor=tau_T / tau_core,
        factory_supply_rate=D_L,
        core_consumption_rate=1 / tau_core,
        p_T_final=e_MSF,
        log_core_survival_per_T=log_s_core,
        log_magic_survival_per_T=math.log1p(-e_MSF),
        active_tiles=N_active,
        Q_core_patches=N_core,
        core_T_cycle_factor=b,
        p_round=p_L,
        e_mem=e_mem,
        supply_mode="enforced" if tau_T <= tau_core * (1 + 1e-9) else "delay_aware",
        V_wait_per_T_QEC_rounds=N_core * (tau_T - tau_core) / cfg.t_round,
        V_reaction_per_T_QEC_rounds=N_core * (tau_core - tau_gate) / cfg.t_round,
        V_idle_extra_per_T_QEC_rounds=extra_rounds,
        prep_file=Path(cfg.prep_degree_table_path).name,
        c_prep_effective=model.level1_c_prep(d_levels[0], cfg),
        c_prep_timing_model="gidney_13_rounds",
        physical_patch_model=model.PHYSICAL_PATCH_MODEL,
        preparation_area_policy="max_padded_patch_recorded_complete_preparation",
        preparation_time_seconds=13 * cfg.t_round + cfg.t_react,
    )


def unit_options(p_phys, d_core, d_levels, cfg):
    """Enumerate interlevel-compatible factory counts through the steady-supply limit."""
    factory = model.evaluate_factory_distances(p_phys, d_core, d_levels, cfg)
    if factory is None:
        return []
    u_max = steady_state_units(d_core, factory, cfg)
    if u_max is None:
        try:
            u_max, _ = model.size_factory_units(factory, d_core, cfg)
        except (ValueError, OverflowError):
            return []
    choices = []
    for u_top in range(1, u_max[-1] + 1):
        units = interlevel_units(
            p_phys, d_core, factory, u_top, cfg, iteration_limit=40
        )
        if units is not None:
            choices.append(units)
    return choices
