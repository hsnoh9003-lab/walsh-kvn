"""Replay retained budget and preparation programs and export trajectory data.

The preserved programs run in temporary directories because some write beside
their inputs. Budget rows use the figure protocol's common shot count;
trajectory CSVs read the saved coupled histories; preparation records rerun the
retained fitting routine. These are different reproduction levels and should
not all be interpreted as a new transport simulation.
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
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from trajectory_check import check_all
try:
    from Source.Data_Production.data_assets import fixed_asset
except ModuleNotFoundError:
    from data_assets import fixed_asset

HERE = Path(__file__).resolve().parent
EXPERIMENTS = fixed_asset("inputs", "retained_experiments")
PROGRAMS = HERE / "producers" / "experiments"
CASES = ("interior", "boundary_clipped", "near_stable", "near_unstable")


def write_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export_budget(output, source_data):
    # The archived program resolves all inputs beside itself. Keep its writes isolated.
    """Run the original budget builder in isolation and apply the common plotting shot count."""
    with TemporaryDirectory(prefix="kvn_budget_") as name:
        run = Path(name)
        shutil.copytree(EXPERIMENTS / "budget", run, dirs_exist_ok=True)
        script = run / "build_relative_data.py"
        shutil.copyfile(PROGRAMS / script.name, script)
        subprocess.run([sys.executable, str(script)], cwd=run, check=True)
        result = "relative_accuracy_budget.csv"
        with (run / "updated_data" / result).open(newline="") as file:
            rows = list(csv.DictReader(file))
        protocol = (
            source_data
            / "Figures/Fig_5/plot_inputs/panels/a_moments/accuracy_protocol.json"
        )
        common_shots = json.loads(protocol.read_text())["panel_a_shots"]
        for row in rows:
            row["panel_a_comparison_shots"] = (
                common_shots if row["displayed_in_panel_a"] == "True" else ""
            )
            row["resource_accepted_shots"] = row["accepted_shots"]
        write_rows(output / result, rows)


def export_trajectories(output):
    """Export saved circuit and matched-characteristic means for four coupled inputs."""
    rows = []
    for case in CASES:
        key = "coupled_cubic_2d__" + case
        with np.load(EXPERIMENTS / "trajectories" / (key + ".npz")) as run:
            for time, quantum, reference in zip(
                run["times"],
                run["moment_history"][:, :2],
                run["prepared_cell_reference_moments"][:, :2],
                strict=True,
            ):
                rows.append(
                    dict(
                        id=key,
                        t=float(time),
                        circuit_mean_x=float(quantum[0]),
                        circuit_mean_y=float(quantum[1]),
                        characteristic_mean_x=float(reference[0]),
                        characteristic_mean_y=float(reference[1]),
                        mean_error_norm=float(np.linalg.norm(quantum - reference)),
                    )
                )
    write_rows(output / "coupled_mean_trajectories.csv", rows)


def export_preparation(output):
    """Recompute the deterministic MPS fits from the canonical KvN implementation."""
    path = PROGRAMS / "mps" / "train_gaussian_pqc.py"
    spec = importlib.util.spec_from_file_location("archived_gaussian_preparation", path)
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    rows = []
    with TemporaryDirectory(prefix="kvn_mps_refit_") as name:
        engine.OUT = Path(name)
        for path in sorted((EXPERIMENTS / "mps").glob("*.json")):
            archived = json.loads(path.read_text())
            family, n = archived["family"], archived["n_qubits"]
            engine.fit_one(family, n)
            fit = json.loads((engine.OUT / "fits" / path.name).read_text())
            layers = fit["training"]["MPS"]["layers"]
            if layers != archived["training"]["MPS"]["layers"]:
                raise ValueError(f"Layer count changed for {path.stem}")
            quality = fit["quality"]
            metrics = (
                "amplitude_l2_error",
                "fidelity",
                "probability_l1_error",
                "negative_amplitude_probability_mass",
            )
            np.testing.assert_allclose(
                [quality[k] for k in metrics],
                [archived["quality"][k] for k in metrics],
                atol=5e-14,
                rtol=1e-12,
            )
            rows.append(
                {
                    "Input": family,
                    "n": n,
                    "Cells / layers": f"{2**n} / {layers}",
                    "Amplitude L2 error": f"{quality['amplitude_l2_error']:.5g}",
                    "Fidelity": f"{quality['fidelity']:.8f}",
                    "Total variation": f"{quality['probability_l1_error'] / 2:.5g}",
                    "Negative weight": f"{quality['negative_amplitude_probability_mass']:.4g}",
                }
            )
    write_rows(output / "Supplementary_Data_3.csv", rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data", type=Path, required=True)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    export_budget(args.output, args.source_data)
    with TemporaryDirectory(prefix="kvn_trajectory_") as scratch:
        check_all(
            args.release_root.resolve(), EXPERIMENTS / "trajectories", Path(scratch)
        )
    export_trajectories(args.output)
    export_preparation(args.output)
