"""Generate the signed-MPS versus positive-root Q5 input-phase comparison.

Inputs are the two published Q5 upwind tables, produced with identical cell
probabilities but different initial amplitude phases. The resulting CSV is a
publication source table, not an independent numerical-stencil validation.

Run: python -m Source.KvN.q5_phase_comparison
Output: Source_Data/Generated_Outputs/Q5_Upwind_Comparison/input_phase_comparison.csv
"""

from __future__ import annotations

import csv
from pathlib import Path

from Source.data_locations import generated_outputs_root, shared_data

METRICS = (
    "native_cell_TV",
    "aggregated_64_bin_TV",
    "relative_continuum_moment_error",
    "squared_output_norm",
)


def read_records(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def phase_rows(primary: list[dict], control: list[dict]) -> list[dict]:
    """Compute percentage-point changes from matched signed/positive input runs."""
    if len(primary) != len(control) or not primary:
        raise ValueError("Both Q5 sweeps must have the same nonzero number of rows")
    result = []
    for signed, positive in zip(primary, control):
        if (signed["cells"], signed["method"]) != (
            positive["cells"], positive["method"]
        ):
            raise ValueError("Q5 sweeps have different grid/method ordering")
        row = {"cells": int(signed["cells"]), "method": signed["method"]}
        for key in METRICS:
            row[key + "_change_percentage_points"] = 100 * (
                float(signed[key]) - float(positive[key])
            )
        row["mps_negative_amplitude_probability"] = float(
            signed["negative_amplitude_probability"]
        )
        row["positive_root_negative_amplitude_probability"] = float(
            positive["negative_amplitude_probability"]
        )
        result.append(row)
    return result


def main() -> Path:
    directory = shared_data("Radial_refinement", "Q5_Upwind_Comparison")
    rows = phase_rows(
        read_records(directory / "q5_upwind_comparison.csv"),
        read_records(directory / "positive_root_control/q5_upwind_comparison.csv"),
    )
    output = generated_outputs_root() / "Q5_Upwind_Comparison/input_phase_comparison.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Generated {len(rows)} Q5 input-phase comparison rows: {output}")
    return output


if __name__ == "__main__":
    main()
