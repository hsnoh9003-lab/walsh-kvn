"""Replay preserved scientific programs with checked inputs in an isolated directory.

jobs.json specifies each program's inputs, SHA-256 digests, commands and outputs.
The retained scientific producer is copied to a temporary working folder before its
relative paths are populated. Label renumbering affects exported text only.
This wrapper preserves the original program files and avoids writing into the
deposited inputs. Select a job with --job and provide --source-data and --output.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import numpy as np
try:
    from Source.Data_Production.data_assets import fixed_asset, fixed_root
except ModuleNotFoundError:
    from data_assets import fixed_asset, fixed_root

ROOT = Path(__file__).resolve().parent
JOBS = json.loads(fixed_asset("jobs.json").read_text())


def write_csv(path, header, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


def renumber(path, target, column, replacements):
    """Translate labels in one named column without parsing numeric fields.

    Replacements apply sequentially to cell text; CSV quoting may change, but
    all other cell strings retain their stored precision.
    """
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)
        header = next(reader)
        index = header.index(column)
        rows = list(reader)
    for row in rows:
        for old, new in replacements.items():
            row[index] = row[index].replace(old, new)
    write_csv(target, header, rows)


def run(job, source_data, out):
    """Check input hashes, run one historical job in isolation and collect its declared outputs."""
    spec = JOBS[job]
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="kvn_replay_") as temp:
        work = Path(temp)
        shutil.copytree(ROOT / "producers" / job, work, dirs_exist_ok=True)
        missing_pod_coefficients = None
        for relative, entry in spec["inputs"].items():
            parent = source_data if entry["location"] == "source_data" else fixed_root(source_data)
            src = parent / entry["path"]
            if (
                job == "pod"
                and not src.is_file()
                and (
                    entry["path"]
                    == "Figures/Fig_1/plot_inputs/CFD_A_POD_coefficients_t29p04_to49p68.csv"
                )
            ):
                missing_pod_coefficients = relative
                continue
            if hashlib.sha256(src.read_bytes()).hexdigest() != entry["sha256"]:
                raise ValueError(f"Unexpected input digest for {src}")
            path = work / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, path)
        if missing_pod_coefficients:
            # The archived plotting CSV contains only a sampled projection of
            # the original flow snapshots. Reconstruct it from the snapshots
            # instead of requiring the output itself as a purported input.
            vendor = work / "data_pipeline/analysis_vendor"
            sys.path.insert(0, str(vendor))
            try:
                from original_rom_functions import make_basis

                source = (
                    work
                    / "source/Figure1/source/cfd_original/data/snapshots_Re100_original.npz"
                )
                with np.load(source) as container:
                    snapshots = {name: container[name] for name in container.files}
                _, coordinates, _ = make_basis(snapshots)
                target_times = 29.04 + 0.24 * np.arange(87)
                indices = [
                    int(np.argmin(abs(snapshots["times"] - t))) for t in target_times
                ]
                np.testing.assert_allclose(
                    snapshots["times"][indices], target_times, atol=1e-12, rtol=0
                )
                target = work / missing_pod_coefficients
                target.parent.mkdir(parents=True, exist_ok=True)
                np.savetxt(
                    target,
                    np.c_[snapshots["times"][indices], coordinates[indices]],
                    delimiter=",",
                    header="tstar,a1,a2,a3",
                    comments="",
                )
            finally:
                sys.path.remove(str(vendor))
        env = dict(
            os.environ,
            OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            MPLBACKEND="Agg",
            PYTHONDONTWRITEBYTECODE="1",
        )
        for args in spec["commands"]:
            subprocess.run([sys.executable, *args], cwd=work, env=env, check=True)
        for entry in spec["outputs"]:
            src = work / entry
            if job == "q5":
                with src.open(newline="") as file:
                    reader = csv.reader(file)
                    header = next(reader)
                    method = header.index("method")
                    rows = [row for row in reader if row[method] == "exact_generator"]
                if len(rows) != 8:
                    raise ValueError("Expected eight independently propagated grids")
                write_csv(out / "previous_exact_generator_rows.csv", header, rows)
            elif src.name == "figure2_metrics.csv":
                renumber(src, out / "figure4_metrics.csv", "window", {"Fig2_": "Fig4_"})
            elif src.name == "numerical_checks.csv":
                renumber(
                    src,
                    out / src.name,
                    "definition",
                    {
                        "Fig. 2e": "Fig. 4f",
                        "Supplementary Data 8": "Supplementary Data 7",
                        "reported in SI v30": "reported in supplementary source data",
                    },
                )
            else:
                name = (
                    "figure4_comparison.csv"
                    if src.name == "figure2_comparison.csv"
                    else src.name
                )
                shutil.copyfile(src, out / name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", choices=sorted(JOBS), required=True)
    parser.add_argument("--source-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.job, args.source_data, args.output)
