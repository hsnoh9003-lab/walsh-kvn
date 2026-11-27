#!/usr/bin/env python3
"""Build the published Gidney P_prep--e_prep trade-off from raw Sinter stats.

Raw input: the Sinter stats.csv files produced by generate_raw_stats.py, or
previously aggregated per-p Sinter stats.csv files. Gap thresholds are
reconstructed with the original Cultiv split_by_gap_threshold algorithm.

The publication's CSV column e_out_T_est is the conservative preparation
error estimate e_prep = min(1, 2*e_out_S); e_out_S=errors/kept and
P_prep=kept/shots. Both preparation error and exact 95% binomial intervals
are computed from counts. There is no fit, extrapolation, or smoothing.

The separately acquired p=5e-4 dataset uses 1-dB gap thresholds (0..78);
the other seven rates use 5-dB thresholds, including the final unrounded
boundary. All eight output files use the same published 17-column schema.

Examples, from the GitHub repository root:
  python -m Source.Gidney.build_tradeoff --verify-published
  python -m Source.Gidney.build_tradeoff --raw-root ../Source_Data/Generated_Outputs/Gidney/raw_stats \
      --output-dir ../Source_Data/Generated_Outputs/Gidney/tradeoff --p 5

This tool never overwrites the published Source_Data inputs unless an
operator explicitly names their directory as --output-dir.
"""

from __future__ import annotations

# Direct-file execution must locate Codes/ independently of the working directory.
if __package__ in (None, ""):
    import sys as _path_sys
    from pathlib import Path as _Path
    _codes = next(p for p in _Path(__file__).resolve().parents
                  if (p / "Source/data_locations.py").is_file())
    _path_sys.path.insert(0, str(_codes))

from Source.data_locations import generated_outputs_root

import argparse
import csv
import math
from pathlib import Path
from typing import Any

from Source.data_locations import source_data_root
from Source.Gidney.generate_raw_stats import (
    D1,
    D2_VALUES,
    DECODER,
    GATESET,
    PHYSICAL_ERROR_INDICES,
    R2,
    shots_for,
)

T_OVER_S = 2.0
GAP_ROUNDING_DB = 5
LAST_GAP_DB = {2: 90, 3: 84, 4: 81, 5: 78, 6: 75, 7: 73, 8: 72, 9: 70}
COLUMNS = (
    "p_phys",
    "d1",
    "d2",
    "r1",
    "r2",
    "gap_db",
    "shots",
    "discards",
    "kept",
    "logical_errors",
    "P_prep",
    "e_out_S",
    "e_out_S_95_lo",
    "e_out_S_95_hi",
    "e_out_T_est",
    "e_out_T_est_95_lo",
    "e_out_T_est_95_hi",
)


def filename(p_index: int) -> str:
    """Name the published trade-off table for p_phys = p_index * 1e-4."""
    return f"custom_p{p_index}e-4_d1_3_d2.csv"


def gap_grid(p_index: int) -> list[int]:
    """Original 1-dB/5-dB thresholds, retaining the final unrounded gap."""
    maximum = LAST_GAP_DB[p_index]
    stride = 1 if p_index == 5 else GAP_ROUNDING_DB
    result = list(range(0, maximum + 1, stride))
    if result[-1] != maximum:
        result.append(maximum)
    return result


def clopper_pearson(errors: int, kept: int) -> tuple[float, float]:
    """Two-sided Clopper--Pearson 95% confidence interval."""
    from scipy.stats import beta

    if kept <= 0:
        raise ValueError("Clopper-Pearson interval requires kept > 0")
    lower = 0.0 if errors == 0 else float(beta.ppf(0.025, errors, kept - errors + 1))
    upper = 1.0 if errors == kept else float(beta.ppf(0.975, errors + 1, kept - errors))
    return lower, upper


def row_from_counts(
    p_index: int,
    d2: int,
    gap_db: int,
    shots: int,
    discards: int,
    errors: int,
) -> dict[str, Any]:
    """Compute the published columns, including the conservative T-state proxy."""
    kept = shots - discards
    if shots <= 0 or kept <= 0 or not (0 <= errors <= kept):
        raise ValueError(f"Invalid counts p={p_index}e-4, d2={d2}, gap={gap_db}")
    error_s = errors / kept
    lower, upper = clopper_pearson(errors, kept)
    return dict(
        zip(
            COLUMNS,
            (
                p_index * 1e-4,
                D1,
                d2,
                D1,
                R2,
                gap_db,
                shots,
                discards,
                kept,
                errors,
                kept / shots,
                error_s,
                lower,
                upper,
                min(1.0, T_OVER_S * error_s),
                min(1.0, T_OVER_S * lower),
                min(1.0, T_OVER_S * upper),
            ),
        )
    )


def source_stats_files(raw_root: Path, p_index: int) -> list[Path]:
    """Support both reproducible per-distance runs and original combined files."""
    parent = raw_root / f"p{p_index}e-4"
    individual = [parent / f"d2_{d2}" / "stats.csv" for d2 in D2_VALUES]
    if all(path.is_file() for path in individual):
        return individual
    combined = parent / "stats.csv"
    if combined.is_file():
        return [combined]
    missing = [path for path in individual if not path.is_file()]
    raise FileNotFoundError(
        f"Missing raw Sinter stats for p={p_index}e-4: {missing[:3]}; "
        f"alternatively place one aggregate stats.csv at {combined}"
    )


def reconstruct_rate(raw_root: Path, p_index: int) -> list[dict[str, Any]]:
    """Apply Cultiv's gap split without altering Sinter errors or discards."""
    try:
        import sinter
        import cultiv
    except ImportError as exc:
        raise RuntimeError(
            "The original Cultiv and Sinter packages are required"
        ) from exc

    raw = []
    for stats_file in source_stats_files(raw_root, p_index):
        raw.extend(sinter.read_stats_from_csv_files(stats_file))
    if not raw:
        raise ValueError(f"No Sinter stats for p={p_index}e-4")
    gap_stats = cultiv.split_by_gap_threshold(
        raw,
        gap_rounding=1 if p_index == 5 else GAP_ROUNDING_DB,
        keep_zero=True,
    )

    rows = []
    for stat in gap_stats:
        meta = dict(stat.json_metadata)
        if (
            meta.get("c") != "end2end-inplace-distillation"
            or meta.get("noise") != "uniform"
            or meta.get("g") != GATESET
            or (stat.decoder or DECODER) != DECODER
        ):
            continue
        if not math.isclose(
            float(meta.get("p", math.nan)), p_index * 1e-4, abs_tol=1e-12
        ):
            continue
        if int(meta.get("d1", -1)) != D1:
            continue
        d2 = int(meta["d2"])
        if d2 not in D2_VALUES or (
            int(meta.get("r1", -1)),
            int(meta.get("r2", -1)),
        ) != (D1, R2):
            continue
        shots, discards, errors = int(stat.shots), int(stat.discards), int(stat.errors)
        if shots <= discards:
            continue
        rows.append(
            row_from_counts(
                p_index,
                d2,
                int(meta.get("gap", 0)),
                shots,
                discards,
                errors,
            )
        )
    rows.sort(key=lambda row: (row["p_phys"], row["d1"], row["d2"], row["gap_db"]))
    validate_grid(rows, p_index, require_published_shots=False)
    return rows


def validate_grid(
    rows: list[dict[str, Any]],
    p_index: int,
    *,
    require_published_shots: bool,
) -> None:
    """Check all expected distance/gap pairs and physical input settings."""
    expected = {(d2, gap) for d2 in D2_VALUES for gap in gap_grid(p_index)}
    observed = set()
    for row in rows:
        d2 = int(row["d2"])
        gap = int(row["gap_db"])
        pair = d2, gap
        if pair in observed or pair not in expected:
            raise ValueError(
                f"Unexpected or duplicate (d2,gap) for p={p_index}: {pair}"
            )
        observed.add(pair)
        if not math.isclose(float(row["p_phys"]), p_index * 1e-4, abs_tol=1e-12):
            raise ValueError("Incorrect physical error probability")
        if (int(row["d1"]), int(row["r1"]), int(row["r2"])) != (D1, D1, R2):
            raise ValueError(f"Wrong cultivation configuration: {pair}")
        if require_published_shots and int(row["shots"]) != shots_for(p_index, d2):
            raise ValueError(f"Published shot count differs: p={p_index}, d2={d2}")
    if observed != expected:
        raise ValueError(
            f"p={p_index}: missing {len(expected-observed)} of {len(expected)} distance/gap pairs"
        )


def check_published(path: Path, p_index: int) -> int:
    """Recalculate published CSV quantities independently from stored counts."""
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != COLUMNS:
            raise ValueError(f"Unexpected columns: {path}")
        rows = list(reader)
    validate_grid(rows, p_index, require_published_shots=True)
    count_fields = ("shots", "discards", "kept", "logical_errors")
    estimate_fields = COLUMNS[10:]
    for row in rows:
        d2, gap = int(row["d2"]), int(row["gap_db"])
        shots, discards, kept, errors = (int(row[name]) for name in count_fields)
        if kept != shots - discards:
            raise ValueError(f"Wrong kept count: p={p_index}, d2={d2}, gap={gap}")
        reconstructed = row_from_counts(p_index, d2, gap, shots, discards, errors)
        for name in estimate_fields:
            if not math.isclose(
                float(row[name]), reconstructed[name], rel_tol=1e-11, abs_tol=1e-15
            ):
                raise ValueError(f"Wrong {name}: p={p_index}, d2={d2}, gap={gap}")
    return len(rows)


def write_csv(rows: list[dict[str, Any]], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--raw-root", type=Path, default=generated_outputs_root() / "Gidney/raw_stats"
    )
    ap.add_argument(
        "--output-dir", type=Path, default=generated_outputs_root() / "Gidney/tradeoff"
    )
    ap.add_argument(
        "--p",
        type=int,
        nargs="+",
        choices=PHYSICAL_ERROR_INDICES,
        help="Physical error probabilities in units of 1e-4 (default: all)",
    )
    ap.add_argument(
        "--verify-published",
        action="store_true",
        help="Validate all published CSVs using only scipy; do not write anything",
    )
    args = ap.parse_args()
    rates = tuple(dict.fromkeys(args.p or PHYSICAL_ERROR_INDICES))

    if args.verify_published:
        published = source_data_root() / "Shared_source_data/Gidney_cultivation/raw"
        for rate in rates:
            path = published / filename(rate)
            n = check_published(path, rate)
            print(
                f"PASS p={rate}e-4: {n} published rows, all counts and intervals verified"
            )
        return

    if (
        args.output_dir.resolve()
        == (source_data_root() / "Shared_source_data/Gidney_cultivation/raw").resolve()
    ):
        ap.error(
            "Refusing to overwrite the canonical Source Data path; choose an output under Generated_Outputs/"
        )
    for rate in rates:
        rows = reconstruct_rate(args.raw_root, rate)
        destination = args.output_dir / filename(rate)
        write_csv(rows, destination)
        print(f"Generated {destination} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
