"""Plot diagnostics from a completed grid/time-step refinement.

Inputs
------
A directory produced by Source.KvN.grid_step_scaling.

Outputs
-------
Diagnostic plots in the supplied refinement-results directory.

Run from the repository root
----------------------------
python -m Figure_Plots.plot_grid_step_scaling ../Source_Data/Generated_Outputs/Grid_Step_Refinement

Scope
-----
This is an optional diagnostic view of a calculation, not a second publication-figure implementation.
"""

# Direct-file execution must locate Codes/ independently of the working directory.
if __package__ in (None, ""):
    import sys as _path_sys
    from pathlib import Path as _Path
    _codes = next(p for p in _Path(__file__).resolve().parents
                  if (p / "Source/data_locations.py").is_file())
    _path_sys.path.insert(0, str(_codes))


from pathlib import Path
import argparse
import csv
import json

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from Source.data_locations import source_data_root


def read(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def write(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(directory):
    """Plot completed grid/time sweeps; no transport or hardware search is run here."""
    polynomial = read(directory / "polynomial_grid_step_sweep.csv")
    rows = []
    for model in [
        "quadratic",
        "cubic_interior",
        "quintic_multistable",
        "coupled_cubic_2d",
    ]:
        subset = [
            x
            for x in polynomial
            if x["model"] == model
            and not (
                model == "quintic_multistable" and x["input"] == "boundary_clipped"
            )
        ]
        inputs = sorted(set(x["input"] for x in subset))
        for factor in [1, 2, 4]:
            cells = int(
                next(
                    x["cells_per_axis"]
                    for x in subset
                    if int(x["refinement_factor"]) == factor
                )
            )
            for policy, steps in [
                ("fixed_steps", 64),
                ("proportional_steps", 64 * factor),
            ]:
                selected = [
                    x
                    for x in subset
                    if int(x["refinement_factor"]) == factor
                    and int(x["steps"]) == steps
                ]
                largest = max(
                    selected, key=lambda x: float(x["temporal_relative_moment_error"])
                )
                rows.append(
                    {
                        "model": model,
                        "refinement_factor": factor,
                        "cells_per_refined_axis": cells,
                        "steps": steps,
                        "policy": policy,
                        "time_diagnostic_percent": 100
                        * float(largest["temporal_relative_moment_error"]),
                        "input_at_maximum": largest["input"],
                        "number_of_inputs": len(inputs),
                        "reference": "exact finite-grid exponential",
                        "reference_check_percent": 0.0,
                    }
                )
    for run in ["late40", "long298"]:
        base_steps = {"late40": 33, "long298": 264}[run]
        data = source_data_root() / "Shared_source_data/MPS_KvN_inputs/KvN/inputs" / run
        config = json.loads((data / "protocol.json").read_text())
        metric = np.array(
            json.loads((data / "terminal_readout.json").read_text())["field_metric"]
        )
        for factor in [1, 2, 4] if run == "late40" else [1, 2]:
            nr, nz = (
                config["grid_r_theta_z"][0] * factor,
                config["grid_r_theta_z"][2] * factor,
            )
            for policy, steps in [
                ("fixed_steps", base_steps),
                ("proportional_steps", base_steps * factor),
            ]:
                first = directory / f"{run}_r{nr}_z{nz}_steps{steps}.json"
                second = directory / f"{run}_r{nr}_z{nz}_steps{2*steps}.json"
                if not first.exists() or not second.exists():
                    assert run == "late40" and factor == 4 and policy == "fixed_steps"
                    continue
                delta = np.array(json.loads(first.read_text())["mean"]) - np.array(
                    json.loads(second.read_text())["mean"]
                )
                rows.append(
                    {
                        "model": run,
                        "refinement_factor": factor,
                        "cells_per_refined_axis": nr,
                        "steps": steps,
                        "policy": policy,
                        "time_diagnostic_percent": 100
                        * float(np.sqrt(max(0.0, delta @ metric @ delta))),
                        "input_at_maximum": run,
                        "number_of_inputs": 1,
                        "reference": "direct Nt versus 2Nt terminal field difference",
                        "reference_check_percent": 0.0,
                    }
                )
    write(directory / "grid_step_summary.csv", rows)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "stix",
            "font.size": 9,
            "axes.linewidth": 0.6,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(8, 4.6), layout="constrained")
    labels = [
        "Quadratic",
        "Cubic",
        "Quintic",
        "Coupled cubic",
        "Three-coordinate, late",
        "Three-coordinate, growth",
    ]
    models = [
        "quadratic",
        "cubic_interior",
        "quintic_multistable",
        "coupled_cubic_2d",
        "late40",
        "long298",
    ]
    for ax, model, title in zip(axes.ravel(), models, labels):
        for policy, color, marker, label in [
            ("fixed_steps", "#c47736", "o", r"Fixed $N_t$"),
            ("proportional_steps", "#286590", "s", r"$N_t\propto N$"),
        ]:
            selected = [
                x for x in rows if x["model"] == model and x["policy"] == policy
            ]
            ax.plot(
                [x["cells_per_refined_axis"] for x in selected],
                [x["time_diagnostic_percent"] for x in selected],
                color=color,
                marker=marker,
                ms=3,
                lw=0.8,
                label=label,
            )
        ticks = sorted(
            set(x["cells_per_refined_axis"] for x in rows if x["model"] == model)
        )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(ticks, [str(x) for x in ticks])
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(r"Cells per refined axis, $N$")
        ax.set_ylabel(
            "Moment error (%)"
            if model not in ["late40", "long298"]
            else "Step-doubling change (%)"
        )
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.savefig(directory / "grid_step_scaling.pdf")
    fig.savefig(directory / "grid_step_scaling.png")
    plt.close(fig)
    summary = {
        "models": 6,
        "polynomial_sweep_rows": len(polynomial),
        "summary_rows": len(rows),
        "original_inputs_kept": True,
        "additional_quintic_boundary_diagnostic": True,
        "three_coordinate_refined_axes": ["r", "z"],
        "angular_cells_fixed": 32,
    }
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for row in rows:
        print(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    main(parser.parse_args().directory)
