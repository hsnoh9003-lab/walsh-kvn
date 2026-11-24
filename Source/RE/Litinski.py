"""Litinski factory protocols evaluated with the common surface-code core.

The two factory models supply their own areas, output probabilities and batch
times. The comparison uses actual code distances for routing error and padded
component dimensions for physical area (Supplementary Note 11, equations (S70)–(S74)).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

from . import model
from .core_timing import conservative_core_error, core_interval_seconds
from .preparation import read_csv


# The factory path and A Game exposure use the published per-round calibration.
_FACTORY_CALIBRATION = model.ModelConfig(logical_prefactor=0.1, p_th=0.01)


@dataclass(frozen=True)
class Protocol:
    paper: str
    name: str
    level: int
    tiles: int | None
    storage_tiles: int
    cycles: float  # Physical surface-code rounds per batch.
    output_states: int
    success: float
    p_out: float
    sourced: str
    output_face_tiles: int | None = None
    output_code_distance: int | None = None

    @property
    def rounds_per_output(self):
        """Mean QEC rounds per delivered state. Catalogue times already include retries."""
        return self.cycles / (self.output_states * self.success)


def factory_candidates_game(p_phys, d_out, data_block, modified=False):
    """Return the three published A Game protocols at the chosen output distance."""
    if modified:
        raise ValueError("The comparison uses the published factory layouts")
    P_15 = (1 - p_phys)**15
    P_116 = (1 - p_phys)**116
    return [
        Protocol(
            "game", "15-to-1_11tile", 1, 11, int(data_block == "fast"),
            11*d_out, 1, P_15, 35*p_phys**3, "Game Sec 3.3; Fig 17(c), Fig 23", 3, d_out,
        ),
        Protocol(
            "game", "116-to-12_81tile", 1, 81, 12,
            50*d_out, 12, P_116, 41.25*p_phys**4, "Game Sec 3.4; Fig 20/22(b)", 17, d_out,
        ),
        Protocol(
            "game", "225-to-1_176tile", 2, 176, 0, 15*d_out, 1, P_15, 1500625*p_phys**9,
            "Game Fig. 19/Sec. 3.5; throughput success approximates first-level supply only",
            21, d_out,
        ),
    ]


def read_numerical(path):
    """Return the usable Not as Costly catalogue rows in source order."""
    rows = read_csv(path)
    if not rows:
        raise RuntimeError("Empty numerical source")
    return [row for row in rows if row["status"] == "ok"]


def candidate_nac(row):
    """Construct a factory from its tabulated mean batch duration and output error."""
    return Protocol(
        "not_as_costly", row["name"], int(row["level"]), None, 0,
        float(row["cycles_per_batch"]), int(row["output_states"]), 1.0, float(row["p_out"]),
        "OFFICIAL saved numerical density-matrix output", None, int(row["d_out"]),
    )


def _distances(row, keys, increment):
    return tuple(int(float(row[key])) + increment for key in keys)


def official_factory_qubits(row, pitch_increment=1):
    """Return the NAC factory area. Increment zero gives the original catalogue area."""
    pitch_increment = int(pitch_increment)
    d_X, d_Z, d_m = _distances(row, ("dx", "dz", "dm"), pitch_increment)
    width_1 = d_X + 4*d_Z
    kind = row["kind"]
    if kind == "one":
        area = width_1*3*d_X + 2*d_m
    elif kind == "small_one":
        area = 2*d_X*width_1 + d_m
    else:
        d_X2, d_Z2, d_m2 = _distances(row, ("dx2", "dz2", "dm2"), pitch_increment)
        if kind == "small_two":
            area = ((d_X2 + 4*d_Z2 + d_m2)*2*d_X2 + width_1*3*d_X
                    + 2*d_m + 2*d_m2*d_m2)
        elif kind in ("two15", "two20"):
            n_L1 = int(float(row["nl1"]))
            width_2 = d_X2 + 4*d_Z2 if kind == "two15" else 4*d_X2 + 3*d_Z2
            area = (width_2*3*d_X2 + n_L1*(width_1*(3*d_X + Fraction(d_m2, 2)) + 2*d_m)
                    + 20*d_m2*d_m2 + 2*d_X2*d_m2)
        else:
            raise ValueError(f"Unsupported factory kind {kind}")
    return 2*math.ceil(area)


def nac_factory_length(row, pitch_increment=0):
    """External factory extent in lattice spacings for the catalogue layout."""
    d_X, d_Z = _distances(row, ("dx", "dz"), pitch_increment)
    width_1 = d_X + 4*d_Z
    kind = row["kind"]
    if kind in ("one", "small_one"):
        height_1 = (2 if kind == "small_one" else 3)*d_X
        side = float(max(width_1, height_1))
        return dict(length=side, L_L1=side, L_L2=0., L_inter=0., L_bridge=0.,
                    provenance="Fig11/Fig18: max(actual main block width, height)")
    d_X2, d_Z2, d_m2 = _distances(row, ("dx2", "dz2", "dm2"), pitch_increment)
    if kind == "small_two":
        width_2 = d_X2 + 4*d_Z2
        L_bridge = max(0, 3*d_X - 2*d_X2)
        return dict(length=float(width_1 + width_2 + 2*d_m2 + L_bridge),
                    L_L1=float(width_1), L_L2=float(width_2), L_inter=float(2*d_m2),
                    L_bridge=float(L_bridge),
                    provenance="Fig19 independent small-footprint heuristic, NOT standard Fig13 expression")
    if kind not in ("two15", "two20"):
        raise ValueError(kind)
    n_L1 = int(float(row["nl1"]))
    width_2 = d_X2 + 4*d_Z2 if kind == "two15" else 4*d_X2 + 3*d_Z2
    L_L1 = 0.5*n_L1*width_1
    L_inter = 6*d_m2
    L_bridge = max(0, 3*d_X - d_X2, 2*d_m2 - d_X2)
    return dict(length=float(L_L1 + width_2 + L_inter + L_bridge),
                L_L1=L_L1, L_L2=float(width_2), L_inter=float(L_inter), L_bridge=float(L_bridge),
                provenance="Fig13 symmetric external long-side + both intermediate 3dm2 + L_bridge heuristic")


def _factory_extents(protocol, row, pitch_increment):
    if protocol.paper != "game":
        return nac_factory_length(row, pitch_increment)
    if not protocol.output_face_tiles:
        raise ValueError(f"No output-face geometry for {protocol.name}")
    pitch = protocol.output_code_distance + pitch_increment
    L_out = float(protocol.output_face_tiles)*pitch
    L_bridge = 3.0*pitch if "225-to-1" in protocol.name else 0.0
    return dict(length=L_out + L_bridge, L_L1=L_out, L_L2=0., L_inter=0.,
                L_bridge=L_bridge, provenance=protocol.sourced)


def _connection_geometry(protocol, row, u, block):
    """Evaluate mean routing exposure and the physical area of the full connection."""
    n_out = protocol.output_states
    d_out = protocol.output_code_distance
    for number in (u, n_out, d_out):
        if int(number) != number or number < 1:
            raise ValueError("Unit count, output count and output distance must be positive integers")
    u, n_out, d_out = int(u), int(n_out), int(d_out)

    raw = _factory_extents(protocol, row, 0)
    padded = _factory_extents(protocol, row, 1)
    L_bridge, L_bridge_area = raw["L_bridge"], padded["L_bridge"]
    L_out = raw["length"] - L_bridge
    L_out_area = padded["length"] - L_bridge_area
    if not all(math.isfinite(x) and x >= 0 for x in (L_out, L_bridge, L_out_area, L_bridge_area)):
        raise ValueError("Connection lengths must be finite and nonnegative")

    L_path = L_out*(u - 1) + n_out*d_out
    if not math.isfinite(L_path):
        raise ValueError("Connection path length must be finite")
    if u == 1:
        mean_length = 0.0
    elif u % 2 == 0:
        mean_length = u*L_path/(4*(u - 1))
    else:
        mean_length = (u + 1)*L_path/(4*u)
    N_path = (mean_length + L_bridge)/d_out

    pitch = d_out + 1
    L_path_area = L_out_area*(u - 1) + n_out*pitch
    Q_surface = model.physical_patch_qubits(d_out)
    qubits_per_length = Q_surface/pitch
    Q_path_horizontal = qubits_per_length*L_path_area
    Q_path_bridges = qubits_per_length*u*n_out*L_bridge_area
    Q_path = Q_path_horizontal + Q_path_bridges
    if protocol.paper == "game" and block == "fast" and protocol.name == "116-to-12_81tile" and u == 5:
        # The five-factory layout in A Game, Fig. 23(b), reserves four connector tiles.
        Q_path = max(Q_path, 4*Q_surface)

    return dict(
        L_unit=raw["length"], L_L1=raw["L_L1"], L_L2=raw["L_L2"], L_inter=raw["L_inter"],
        L_bridge=L_bridge, L_out=L_out, L_path_total=L_path, geometry_source=raw["provenance"],
        length_units="physical lattice-site spacing", n_conn=Q_path/Q_surface,
        n_conn_active=N_path, n_conn_geometry=L_path, connection_policy="attachment_rectangles",
        connection_assumption="full-span actual-distance error path; componentwise padded area rectangles",
        path_mean_length=mean_length, N_path=N_path,
    ), dict(
        Q_path_horizontal=Q_path_horizontal, Q_path_bridges=Q_path_bridges,
        L_out_area=L_out_area, L_bridge_area=L_bridge_area,
        L_path_area=L_path_area, output_pitch_area=pitch,
    ), Q_path


def evaluate_units(case, p_phys, n_T, d_core, block, protocol, metadata, counts, u):
    """Evaluate one factory layout and return the comparison CSV fields for one attempt."""
    if case not in {"game_silva_v2", "nac_silva_v2"}:
        raise ValueError(f"Unknown factory comparison: {case}")
    d_out = protocol.output_code_distance
    if d_out > d_core or u < 1:
        return None

    cfg = model.ModelConfig(n_L=counts["Q_logic"], data_block=block, p_phys=p_phys)
    t_round = cfg.t_round
    N_T = counts["nT"]
    is_game = protocol.paper == "game"
    if is_game:
        unit_tiles = protocol.tiles + protocol.storage_tiles
        Q_factory = u*unit_tiles*model.physical_patch_qubits(d_out)
    else:
        Q_factory = u*float(official_factory_qubits(metadata))
    rounds_per_output = protocol.rounds_per_output

    routing, allocation, Q_path = _connection_geometry(protocol, metadata, u, block)
    N_core = model.core_tile_count(d_core, cfg)["N_core_tiles"]
    Q_core = N_core*model.physical_patch_qubits(d_core)
    Q_total = Q_core + Q_factory + Q_path
    if not math.isfinite(Q_total):
        return None

    b = model.block_cycles_per_T(block)
    tau_core = core_interval_seconds(b, d_core, t_round, cfg.t_react)
    tau_MSF = (rounds_per_output/u)*t_round
    tau_T = max(tau_core, tau_MSF)
    core_rounds = tau_core/t_round
    service_rounds = tau_T/t_round
    wait_rounds = max(0., service_rounds - core_rounds)

    V_act, V_idle = model.core_volumes(N_T, cfg, d_core)
    V_core = V_act + V_idle
    e_mem = model.memory_error(d_core, p_phys, cfg)
    p_L = model.logical_error_per_round(p_phys, d_core, cfg)
    volumes = conservative_core_error(N_T, N_core, V_act, tau_core, tau_T, e_mem, p_L,
                                      t_round, b, d_core=d_core)
    E_core = volumes["E_core"]

    p_L_out = model.logical_error_per_round(p_phys, d_out, _FACTORY_CALIBRATION)
    e_path = model.repeated_failure_probability(p_L_out, routing["N_path"]*d_out)
    e_grow = model.growth_error(d_out, d_core, p_phys, cfg)
    e_MSF = model.combine_independent(protocol.p_out, e_path, e_grow)
    E_factory_base = E_factory = 0.0
    if is_game:
        tile_count_times_T = u*unit_tiles*N_T
        E_factory_base = model.repeated_failure_probability(p_L_out, tile_count_times_T*core_rounds)
        E_factory = model.repeated_failure_probability(p_L_out, tile_count_times_T*service_rounds)
    E_magic = model.repeated_failure_probability(e_MSF, N_T)
    E_MSF = model.combine_independent(E_magic, E_factory)
    E_stoch = model.combine_independent(E_core, E_MSF)
    E_synth = counts["E_synth"]

    # The table includes both the baseline exposure and the actual supply-limited exposure.
    return dict(
        case=case, paper=protocol.paper, protocol=protocol.name, level=protocol.level,
        p_phys=p_phys, nt=n_T, d_core=d_core, d_out=d_out, core_style=block, factory_units=u,
        physical_variant=model.PHYSICAL_PATCH_MODEL, output_states=protocol.output_states,
        **routing, E_path=e_path, E_growth=e_grow, **allocation,
        E_factory_logical=E_factory, p_out_official=protocol.p_out, p_out_effective=e_MSF,
        p_success=protocol.success, factory_cycles_per_batch=protocol.cycles,
        factory_cycles_per_output=rounds_per_output, q_core=Q_core, q_factory=Q_factory,
        q_path=Q_path, q_total=Q_total, E_core=E_core, E_MSF=E_MSF,
        E_stoch=E_stoch, E_synth=E_synth, E_total=E_stoch + E_synth, core_volume=V_core,
        fault_scope="common_alpha1_silva_core_plus_actual_distance_path_and_growth",
        source=protocol.sourced, alpha_core=1.0, E_factory_logical_base=E_factory_base,
        E_core_original=model.repeated_failure_probability(e_mem, V_core), E_wait=None,
    ) | volumes | dict(
        Q_core_patches=N_core, core_cycle_seconds=tau_core, service_cycle_seconds=tau_T,
        units_enforced=max(1, math.ceil(rounds_per_output/core_rounds - 1e-12)),
        supply_mode="enforced" if wait_rounds < 1e-10 else "delay_aware",
        core_cycles_per_T=core_rounds, effective_cycles_per_T=service_rounds,
        wait_cycles_per_T=wait_rounds, runtime_seconds=N_T*tau_T, runtime_core_seconds=N_T*tau_core,
    )
