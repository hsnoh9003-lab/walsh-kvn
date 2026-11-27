"""Export selected views of retained sweeps, coefficients and coordinates.

These are projections of existing scientific inputs, not fresh CFD or hardware
calculations. Sweeps are filtered to the displayed model families; model
coefficients are paired by monomial powers; modal coordinates are normalized
with the same R, c0 and c2 as the reduced wake model. The output schema is kept
compatible with the deposited figure tables.
"""

# Direct-file execution must locate Codes/ independently of the working directory.
if __package__ in (None, ""):
    import sys as _path_sys
    from pathlib import Path as _Path
    _codes = next(p for p in _Path(__file__).resolve().parents
                  if (p / "Source/data_locations.py").is_file())
    _path_sys.path.insert(0, str(_codes))


import argparse
import csv
import json
from pathlib import Path

import numpy as np
try:
    from Source.Data_Production.data_assets import fixed_asset
except ModuleNotFoundError:
    from data_assets import fixed_asset
from Source.data_locations import source_data_root

HERE = Path(__file__).resolve().parent


def save(path, header, values):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(values)


def sweeps(out):
    """Filter complete sweeps to plotted models and translate only their display labels."""
    shown = {"quadratic", "cubic_interior", "quintic_multistable", "coupled_cubic_2d"}
    for name in ("spatial_refinement.csv", "time_refinement.csv"):
        from Source.Data_Production.data_assets import fixed_asset

        with fixed_asset("inputs", "full_sweep", name).open(
            encoding="utf-8-sig", newline=""
        ) as file:
            reader = csv.reader(file)
            header = next(reader)
            case = header.index("case")
            label = header.index("label")
            selected = []
            for row in reader:
                if row[case] in shown:
                    row[label] = {"C3i": "C3", "Q5m": "Q5"}.get(row[label], row[label])
                    selected.append(row)
            save(out / name, header, selected)


def coefficients(out):
    """Pair monomials across model fits and export nonzero coefficients and differences."""
    from Source.data_locations import source_data_root

    inputs = source_data_root() / "Shared_source_data/CFD_ROM/Model"
    with np.load(inputs / "reported_model_scaling.npz") as deployed, np.load(
        inputs / "candidate_K2.npz"
    ) as candidate:
        np.testing.assert_array_equal(deployed["powers"], candidate["powers"])
        rows = []
        for power, aa, bb in zip(
            deployed["powers"],
            deployed["coefficients"],
            candidate["physical_coefficients"],
            strict=True,
        ):
            for equation, (a, b) in enumerate(zip(aa, bb, strict=True), 1):
                if a or b:
                    rows.append([equation, *power, a, b, b - a])
        save(
            out / "coefficient_comparison.csv",
            [
                "equation",
                "a1_power",
                "a2_power",
                "a3_power",
                "deployed",
                "candidate",
                "change",
            ],
            rows,
        )


def coordinates(data, out):
    """Convert CFD modal coordinates to normalized wake coordinates and export the prepared radial input."""
    inputs = data / "Shared_source_data/CFD_ROM/CFD_ROM/Re100/data"
    config = json.loads((inputs / "configuration.json").read_text())["parameters"]
    with (
        data / "Figures/Fig_1/plot_inputs/CFD_A_POD_coefficients_t29p04_to49p68.csv"
    ).open(newline="") as file:
        raw = list(csv.DictReader(file))
    # Dimensionless coordinates: x=a1/R, y=a2/R,
    # z=(a3-c0)/(c2*R**2); z is a shifted modal coordinate.
    radius = config["R"]
    save(
        out / "CFD_A_POD_normalized_xyz.csv",
        ["tstar", "x", "y", "z"],
        (
            (
                r["tstar"],
                float(r["a1"]) / radius,
                float(r["a2"]) / radius,
                (float(r["a3"]) - config["c0"]) / (config["c2"] * radius**2),
            )
            for r in raw
        ),
    )
    with np.load(inputs / "direct_quantum_history.npz") as history:
        r = history["r"]
        phase = config["initial_phase"]
        save(
            out / "Prepared_Initial_Radial_Probability_64cells.csv",
            [
                "radial_cell_midpoint",
                "prepared_probability",
                "a1_normalized",
                "a2_normalized",
            ],
            zip(
                r,
                history["initial_probability"],
                r * np.cos(phase),
                r * np.sin(phase),
                strict=True,
            ),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sweeps(args.output)
    coefficients(args.output)
    coordinates(args.source_data, args.output)
