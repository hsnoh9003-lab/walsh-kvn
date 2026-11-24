"""Surface-code core and distillation costs for compiled circuit inventories."""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict
from Source.data_locations import source_file

from .core_timing import core_interval_seconds

ETA = 1e-4
CORE_FIXED_TILES = 18.0
PHYSICAL_PATCH_MODEL = "2dplus1_sq"


@dataclass(frozen=True)
class ModelConfig:
    n_L: int = 11
    data_block: str = "compact"

    # Hardware times are in seconds.
    t_round: float = 1e-6
    t_react: float = 10e-6
    p_phys: float = 1e-4
    p_th: float = 1e-2
    logical_prefactor: float = 0.1

    # First-stage distances require measured Gidney preparation data.
    d_min: int = 3
    d_max: int = 223
    L_max: int = 2
    d_cult: int = 3
    prep_degree_table_path: str = ""
    gidney_footprint_table_path: str = str(source_file("Data/RE/gidney_circuit_footprints.csv"))

    O_1: int = 13
    O_higher: int = 15
    distill_input_count: int = 15
    distill_cubic_coeff: float = 35.0
    distill_logical_coeff: float = 7.1
    accept_input_coeff: float = 15.0

    @property
    def W_seconds(self) -> float:
        """Read-only spelling used by Data/Radial128/reproduce_resources.py."""
        return self.t_round

    def validate(self) -> None:
        if self.data_block not in {"compact", "intermediate", "fast"}:
            raise ValueError("Unknown data_block")
        if self.n_L < 1 or int(self.n_L) != self.n_L:
            raise ValueError("n_L must be a positive integer")
        if self.d_min < 3 or self.d_min % 2 == 0:
            raise ValueError("d_min must be an odd integer >= 3")
        if self.d_max < self.d_min or self.d_max % 2 == 0:
            raise ValueError("d_max must be an odd integer >= d_min")
        if not 1 <= self.L_max <= 3:
            raise ValueError("L_max must be between one and three")
        if not math.isfinite(self.p_phys) or self.p_phys <= 0:
            raise ValueError("p_phys must be finite and positive")
        if not math.isfinite(self.t_round) or self.t_round <= 0:
            raise ValueError("t_round must be finite and positive")
        if not math.isfinite(self.t_react) or self.t_react < 0:
            raise ValueError("t_react must be finite and nonnegative")


def clamp_probability(x: float) -> float:
    return min(max(float(x), 0.0), 1.0)


def combine_independent(*errors: float) -> float:
    """Independent failure composition: 1 - product_i(1-e_i)."""
    log_survival = 0.0
    for e in errors:
        e = clamp_probability(e)
        if e >= 1.0:
            return 1.0
        log_survival += math.log1p(-e)
    # -expm1(x) evaluates 1-exp(x) without cancelling tiny probabilities.
    return clamp_probability(-math.expm1(log_survival))


def repeated_failure_probability(p: float, locations: float) -> float:
    """1-(1-p)^locations, allowing a noninteger expected location volume."""
    p = clamp_probability(p)
    locations = max(float(locations), 0.0)
    if locations == 0.0 or p == 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    # Stable for tiny p and large location counts.
    return clamp_probability(-math.expm1(locations * math.log1p(-p)))


def ceil_int(x: float) -> int:
    return int(math.ceil(float(x) - 1e-15))


@lru_cache(maxsize=16)
def _load_gidney_degree_table_cached(path_str: str) -> Dict[int, Dict[str, float]]:
    """Load a degree-indexed Gidney preparation table from CSV.

    Required columns: degree (or d2), e_prep, P_prep.

    For the current Gidney table, ``e_prep`` is already the T-state estimate
    e_T ~= 2*e_S, where e_S=errors/kept comes from the stabilizer end-to-end
    simulation. The table already includes this factor.  The table is deliberately exact: no interpolation/extrapolation
    is performed.
    """
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Gidney degree table not found: {path}")

    rows: Dict[int, Dict[str, float]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        degree_col = "degree" if "degree" in fields else ("d2" if "d2" in fields else None)
        required = {"e_prep", "P_prep"}
        if degree_col is None or not required.issubset(fields):
            raise ValueError(
                f"{path} must contain degree (or d2), e_prep, and P_prep columns; "
                f"found {sorted(fields)}"
            )
        for raw in reader:
            d = int(float(raw[degree_col]))
            e = float(raw["e_prep"])
            P = float(raw["P_prep"])
            if d in rows:
                raise ValueError(f"Duplicate degree={d} in {path}")
            if d <= 0 or not (0.0 <= e <= 1.0) or not (0.0 < P <= 1.0):
                raise ValueError(f"Invalid Gidney table row for degree={d}: e_prep={e}, P_prep={P}")
            rows[d] = {"e_prep": e, "P_prep": P}
    if not rows:
        raise ValueError(f"No rows found in Gidney degree table: {path}")
    return rows


def gidney_degree_table(cfg: ModelConfig) -> Dict[int, Dict[str, float]]:
    if not cfg.prep_degree_table_path:
        raise ValueError("A measured Gidney preparation table is required")
    return _load_gidney_degree_table_cached(cfg.prep_degree_table_path)


def gidney_degree_table_row(d_1: int, cfg: ModelConfig) -> Dict[str, float]:
    d = int(d_1)
    table = gidney_degree_table(cfg)
    if d not in table:
        available = tuple(sorted(table))
        raise ValueError(
            f"No Gidney preparation data for final degree/d2={d}. "
            f"Available exact degrees are {available}. No interpolation/extrapolation is used."
        )
    return table[d]


def logical_error_per_round(p_phys: float, d: int, cfg: ModelConfig) -> float:
    p_phys = float(p_phys)
    d = int(d)
    if p_phys <= 0:
        return 0.0
    return clamp_probability(
        cfg.logical_prefactor * (p_phys / cfg.p_th) ** ((d + 1) / 2.0)
    )


def memory_error(d: int, p_phys: float, cfg: ModelConfig) -> float:
    """Memory failure over d rounds of the surface code."""
    return repeated_failure_probability(logical_error_per_round(p_phys, d, cfg), d)


def growth_error(d_from: int, d_to: int, p_phys: float, cfg: ModelConfig) -> float:
    """A growing patch spends one memory cycle at the source distance (S49)."""
    return memory_error(d_from, p_phys, cfg) if int(d_to) > int(d_from) else 0.0


def distillation_level_metrics(e_in: float, p_phys: float, d: int,
                               cfg: ModelConfig) -> tuple[float, float]:
    """Return the output error and acceptance probability of one distillation batch."""
    e_cliff = memory_error(d, p_phys, cfg)
    e_out = clamp_probability(cfg.distill_cubic_coeff * float(e_in)**3
                              + cfg.distill_logical_coeff * e_cliff)
    P_l = 1.0 - cfg.accept_input_coeff * float(e_in) - 356.0 * e_cliff
    return e_out, P_l


def evaluate_factory_distances(p_phys, d_core, d_levels, cfg, u_levels=None):
    """Evaluate a factory chain, with output-routing exposure once unit counts are known."""
    d_levels = tuple(map(int, d_levels))
    L = len(d_levels)
    if not 1 <= L <= cfg.L_max:
        raise ValueError("Invalid distillation level count")
    if u_levels is not None:
        u_levels = tuple(map(int, u_levels))
        if len(u_levels) != L or min(u_levels) < 1:
            raise ValueError("Each distillation level needs a positive unit count")
    if any(d % 2 == 0 or not cfg.d_min <= d <= min(cfg.d_max, d_core) for d in d_levels):
        return None
    if any(left > right for left, right in zip(d_levels, d_levels[1:])):
        return None

    d_prev = d_levels[0]
    e_out = gidney_degree_table_row(d_prev, cfg)["e_prep"]
    P_levels = []
    for i, d_l in enumerate(d_levels):
        e_in = combine_independent(e_out, growth_error(d_prev, d_l, p_phys, cfg))
        e_out, P_l = distillation_level_metrics(e_in, p_phys, d_l, cfg)
        if P_l <= 0:
            return None
        P_levels.append(P_l)
        if u_levels is not None:
            N_path = mean_silva_path_patches(u_levels[i], level=i + 1)
            p_L = logical_error_per_round(p_phys, d_l, cfg)
            e_path = clamp_probability(-math.expm1((N_path*d_l)*math.log1p(-p_L)))
            e_out = combine_independent(e_out, e_path)
        d_prev = d_l
    e_MSF = combine_independent(e_out, growth_error(d_prev, d_core, p_phys, cfg))
    return {"d_levels": d_levels, "P_levels": tuple(P_levels), "e_MSF": e_MSF}


def silva_output_spacing(level: int) -> int:
    """Return the spacing between adjacent outputs in source-distance patches."""
    if int(level) != level or level < 1:
        raise ValueError("distillation level must be a positive integer")
    return 11 if level == 1 else 10


def mean_silva_path_patches(units: int, level: int = 1) -> float:
    """Mean output-to-centre distance, including the one-patch connection (S50)."""
    u = int(units)
    if u != units or u < 1:
        raise ValueError("factory units must be a positive integer")
    spacing = silva_output_spacing(level)
    # The sum of the absolute offsets from the midpoint is floor(u squared / 4).
    return 1.0 + spacing * ((u * u) // 4) / u


def data_block_tiles(n_L: int, data_block: str) -> int:
    """Return the data-block patch count N_blk from Table S6."""
    if int(n_L) != n_L or n_L < 1:
        raise ValueError("n_L must be a positive integer")
    n_L = int(n_L)
    if data_block == "compact":
        return math.ceil(1.5*n_L + 3.0)
    if data_block == "intermediate":
        return 2*n_L + 4
    if data_block == "fast":
        return math.ceil(2.0*n_L + math.sqrt(8.0*n_L) + 1.0)
    raise ValueError(f"Unknown data_block={data_block!r}")


def block_cycles_per_T(data_block: str) -> float:
    if data_block == "compact":
        return 9.0
    if data_block == "intermediate":
        return 5.0
    if data_block == "fast":
        return 1.0
    raise ValueError(f"Unknown data_block={data_block!r}")


def reaction_storage_tiles(d_region: int, cfg: ModelConfig) -> float:
    """Correction storage at the core or factory level's own distance."""
    return 1.5 * ceil_int(cfg.t_react / (cfg.t_round * int(d_region)))


def core_tile_count(d_core: int, cfg: ModelConfig) -> dict:
    N_blk = data_block_tiles(cfg.n_L, cfg.data_block)
    N_corr_bus = reaction_storage_tiles(d_core, cfg)
    N_core = N_blk + CORE_FIXED_TILES + N_corr_bus
    # These keys are also consumed by the standalone Litinski table generator.
    return {"data_block_tiles": N_blk, "reaction_storage_tiles": N_corr_bus,
            "N_core_tiles": N_core}


def level1_c_prep(d_1: int, cfg: ModelConfig) -> float:
    """Preparation and decoding in units of d_first rounds."""
    return float(ceil_int((13.0 + cfg.t_react / cfg.t_round) / int(d_1)))


def physical_patch_qubits(d: int) -> int:
    """Physical qubits per padded surface-code patch."""
    return 2 * (int(d) + 1) ** 2


@lru_cache(maxsize=24)
def _read_gidney_footprints(path: str):
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return {(int(row["d_cult"]), int(row["d_esc"])): int(row["physical_qubits_q"])
            for row in rows}


def gidney_physical_qubits(d_esc: int, cfg: ModelConfig) -> int:
    """Recorded physical qubits from Gidney Stim circuit q metadata.

    The observed 2*d_esc**2+4 law is deliberately NOT extrapolated.
    """
    data = _read_gidney_footprints(cfg.gidney_footprint_table_path)
    key = (int(cfg.d_cult), int(d_esc))
    if key not in data:
        raise ValueError(f"No recorded Gidney circuit footprint for {key}; do not extrapolate.")
    return data[key]


def distillation_duration(level: int, d_l: int, cfg: ModelConfig) -> float:
    O_l = cfg.O_1 if int(level) == 1 else cfg.O_higher
    return float(O_l) * cfg.t_round * int(d_l)


def size_factory_units(factory, d_core, cfg):
    """Find the minimum units meeting D_L >= C_core and D_(l-1) >= C_l."""
    d_levels = factory["d_levels"]
    P_levels = factory["P_levels"]
    if any(P_l <= 0 for P_l in P_levels):
        raise ValueError("Factory acceptance probability must be positive")
    b = block_cycles_per_T(cfg.data_block)
    C_required = 1.0 / core_interval_seconds(b, d_core, cfg.t_round, cfg.t_react)
    u_levels = [0] * len(d_levels)
    throughput_ok = True
    for i in reversed(range(len(d_levels))):
        tau_l = distillation_duration(i+1, d_levels[i], cfg)
        u_l = max(1, ceil_int(C_required * tau_l / P_levels[i]))
        D_l = u_l * P_levels[i] / tau_l
        throughput_ok = throughput_ok and D_l + 1e-18 >= C_required
        u_levels[i] = u_l
        C_required = cfg.distill_input_count * u_l / tau_l
    return tuple(u_levels), throughput_ok


def compute_total_qubits(L, u_levels, code_distances, *, d_core, cfg):
    """Evaluate core and factory areas at the level-local pitches (S59–S66)."""
    d_levels = tuple(map(int, code_distances))
    u_levels = tuple(map(int, u_levels))
    if L < 1 or len(d_levels) != L or len(u_levels) != L:
        raise ValueError("One code distance and unit count are required per level")

    core = core_tile_count(d_core, cfg)
    Q_core = core["N_core_tiles"] * physical_patch_qubits(d_core)
    P_prep = gidney_degree_table_row(d_levels[0], cfg)["P_prep"]
    c_prep = level1_c_prep(d_levels[0], cfg)
    prep_tiles_per_unit = (15.0/13.0) * (c_prep + P_prep) / P_prep
    Q_MSF = 0.0
    storage_by_level = []
    for i, (d_l, u_l) in enumerate(zip(d_levels, u_levels)):
        N_corr_bus = reaction_storage_tiles(d_l, cfg)
        Q_surface = physical_patch_qubits(d_l)
        spacing = silva_output_spacing(i + 1)
        fixed = 29 if i == 0 else 28
        # One shared bus has 1 + spacing*(u_l - 1) patches (S59).
        if i == 0:
            n_tiles = u_l * (fixed + spacing + prep_tiles_per_unit + N_corr_bus) - (spacing - 1)
            n_0 = prep_tiles_per_unit * u_l
            Q_prep_unit = max(Q_surface, gidney_physical_qubits(d_l, cfg))
            Q_MSF += (n_tiles - n_0) * Q_surface + n_0 * Q_prep_unit
        else:
            n_tiles = u_l * (fixed + spacing + N_corr_bus) - (spacing - 1)
            Q_MSF += n_tiles * Q_surface
        storage_by_level.append(N_corr_bus)
    return {"Q": Q_core + Q_MSF, "Q_core": Q_core, "Q_MSF": Q_MSF,
            "N_corr_bus_core": core["reaction_storage_tiles"],
            "N_corr_bus_levels": tuple(storage_by_level)}


def core_volumes(N_T: int, cfg: ModelConfig, d_core: int) -> tuple[float, float]:
    """Return V_act and V_idle,base in patch-cycles (S67a)–(S67b)."""
    b = block_cycles_per_T(cfg.data_block)
    N_core = core_tile_count(d_core, cfg)["N_core_tiles"]
    N_active = data_block_tiles(cfg.n_L, cfg.data_block) + CORE_FIXED_TILES
    return b * N_active * float(N_T), b * float(N_T) * N_core
