"""Parallel factory counts for the interlevel throughput constraints (S56–S58)."""

import math
from .model import distillation_duration, evaluate_factory_distances, size_factory_units


def steady_state_units(d_core, factory, cfg):
    """Find a stable supply-enforced configuration using its own routing errors.

    Unit counts affect both throughput and routing length. Routing errors can
    alter the input error and acceptance probability of subsequent levels.
    Alternate factory evaluation and unit sizing until the counts stabilize.

    Returning None indicates that this procedure did not obtain a valid stable
    configuration; it is not a proof that every possible architecture fails.
    """
    if cfg.p_phys >= cfg.p_th:
        return None
    try:
        u_levels, throughput_ok = size_factory_units(factory, d_core, cfg)
        seen = set()
        # The iteration limit and repeated-state check guard against nonconvergence.
        # They are numerical search controls rather than physical resource limits.
        for _ in range(24):
            if not throughput_ok or u_levels in seen:
                return None
            seen.add(u_levels)
            corrected = evaluate_factory_distances(
                cfg.p_phys, d_core, factory["d_levels"], cfg, u_levels
            )
            if corrected is None:
                return None
            updated, throughput_ok = size_factory_units(corrected, d_core, cfg)
            if updated == u_levels:
                return updated if throughput_ok else None
            u_levels = updated
    except (ValueError, OverflowError):
        return None
    return None


def interlevel_units(p_phys, d_core, factory, u_top, cfg, iteration_limit=None):
    """Determine the lower-level units required by a fixed top-level unit count.

    A single-level factory requires no interlevel sizing.
    For a two-level factory, enforce D_1 >= C_2.

    This condition concerns supply between distillation levels. Whether the
    final output rate meets the core's maximum consumption rate is evaluated
    separately.
    """
    d_levels = factory["d_levels"]
    if len(d_levels) == 1:
        return (u_top,)
    tau_1 = distillation_duration(1, d_levels[0], cfg)
    tau_2 = distillation_duration(2, d_levels[1], cfg)
    inputs_per_batch = cfg.distill_input_count * u_top * tau_1 / tau_2
    u_1 = max(1, math.ceil(inputs_per_batch / factory["P_levels"][0] - 1e-12))
    iterations = 0
    while iteration_limit is None or iterations < iteration_limit:
        corrected = evaluate_factory_distances(
            p_phys, d_core, d_levels, cfg, (u_1, u_top)
        )
        if corrected is None:
            return None
        required = math.ceil(inputs_per_batch / corrected["P_levels"][0] - 1e-12)
        if required <= u_1:
            return (u_1, u_top)
        u_1 = required
        iterations += 1
    return None
