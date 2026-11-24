"""Replay the recovered original Re100 CFD drag/lift generator.

The working solver has readability edits only; its original is preserved in
Source/CFD/force_replay/audit. This entry point supplies the Source_Data input,
exports valid force rows, and optionally checks source identity and outputs.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from Source.data_locations import REPO_ROOT, source_file, set_source_data_root

# RECOVERED: solver directory in the code package.
RECOVERED = REPO_ROOT / "Source/CFD/force_replay"
# Historical input paths used as keys in the Source_Data manifest.
STEADY_INPUT = "Source/CFD/replay_fem/inputs/steady_coarse.npz"
ARCHIVE = "Cylinder_Forces_Re100_20261110/"


def export_forces(output: Path) -> dict:
    """Use the original valid-pressure filter and the Figure 4 time selection."""
    # raw: every saved step, including t=0 without a computed pressure.
    raw = pd.read_csv(output / "forces.csv")
    # valid: steps with pressure available; this flag is not an accuracy test.
    valid = raw.loc[
        raw["pressure_valid"].astype(str).str.lower().isin(["true", "1"])
    ].copy()
    if valid.empty:
        raise ValueError("No time steps with computed pressure were found.")
    valid.to_csv(output / "forces_cfd.csv", index=False)

    # result: row counts returned to the run's audit record.
    result = {"raw_rows": len(raw), "valid_rows": len(valid)}
    if float(valid["time"].iloc[-1]) >= 60:
        # The published figure was exported after reading forces_cfd.csv.
        # Preserve that read/write sequence, including pandas float formatting.
        exported = pd.read_csv(output / "forces_cfd.csv")
        figure = exported.loc[exported["time"] <= 60].copy()
        figure.to_csv(output / "fig4f_fem_forces.csv", index=False)
        result["figure_rows"] = len(figure)
    return result


def verify_solver_sources() -> dict:
    """Verify archived bytes and the working solver's unchanged executable AST."""
    # provenance: archived file hashes and the reviewed working-copy record.
    provenance = json.loads((RECOVERED / "provenance.json").read_text())
    for relative, digest in provenance["files"].items():
        assert (
            hashlib.sha256((RECOVERED / relative).read_bytes()).hexdigest() == digest
        ), relative

    # working: reviewed solver metadata; source: its current file contents.
    working = provenance["working_copy"]
    source = (RECOVERED / working["path"]).read_bytes()
    assert hashlib.sha256(source).hexdigest() == working["sha256"], working["path"]

    # trees: executable syntax of the working and original solvers.
    # Parsing ignores comments/formatting; remove only the module docstring.
    trees = []
    for content in (source, (RECOVERED / working["original"]).read_bytes()):
        tree = ast.parse(content)
        if ast.get_docstring(tree) is not None:
            tree.body.pop(0)
        trees.append(ast.dump(tree))
    assert trees[0] == trees[1], "Working solver executable AST differs from the original."
    return {
        "original_solver_files_unchanged": True,
        "working_solver_ast_matches_original": True,
        "working_solver_sha256": working["sha256"],
    }


def verify_outputs(output: Path) -> dict:
    """Check the complete recorded 4500-step case against independent inputs."""
    # source_checks: original-file integrity and unchanged solver computation.
    source_checks = verify_solver_sources()
    # raw_reference: archived CSV path; reference/actual: archived/replayed tables.
    raw_reference = source_file(ARCHIVE + "archived_replay/forces.csv")
    reference = pd.read_csv(raw_reference)
    actual = pd.read_csv(output / "forces.csv")
    assert len(actual) == len(reference) == 4501, "Expected t=0 plus 4500 time steps."
    assert actual.columns.tolist() == reference.columns.tolist()
    assert actual["pressure_valid"].equals(reference["pressure_valid"])

    # differences: maximum absolute difference for each numeric CSV column.
    differences = {}
    for column in actual.columns.drop("pressure_valid"):
        left, right = actual[column].to_numpy(), reference[column].to_numpy()
        np.testing.assert_array_equal(left, right, err_msg=column)
        differences[column] = float(np.nanmax(np.abs(left - right)))

    # checks/references: byte-equality results and the corresponding reference files.
    checks = {}
    references = {
        "forces.csv": raw_reference,
        "forces_cfd.csv": source_file(ARCHIVE + "results/forces_cfd.csv"),
        "fig4f_fem_forces.csv": source_file("Figure/F4/data/force_output/fig4f_fem_forces.csv"),
    }
    for name, reference_path in references.items():
        checks[name] = (output / name).read_bytes() == reference_path.read_bytes()
        assert checks[name], name

    # arrays: exact equality of each saved field/state array, grouped by NPZ file.
    arrays = {}
    for name, original in [
        ("snapshots.npz", "Data/CFD_ROM/Re100/data/source_snapshots.npz"),
        ("final_state.npz", ARCHIVE + "archived_replay/final_state.npz"),
    ]:
        with (
            np.load(output / name) as actual_npz,
            np.load(source_file(original)) as reference_npz,
        ):
            arrays[name] = {
                key: bool(np.array_equal(actual_npz[key], reference_npz[key]))
                for key in reference_npz.files
            }
        assert all(arrays[name].values()), name
    return {
        "force_csv_bytes_equal": checks,
        "maximum_absolute_differences": differences,
        "arrays_equal": arrays,
        **source_checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path,
        help="Override DOI cache / KVN_SOURCE_DATA / sibling Source_Data.",
    )
    parser.add_argument(
        "--output", type=Path, default=REPO_ROOT / "Build/CFD/force_replay",
    )
    parser.add_argument(
        "--end", type=float, default=90.0,
        help="Final nondimensional time; dt is fixed at 0.02.",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Require the recorded full 90-time-unit case to match.",
    )
    parser.add_argument(
        "--export-only", action="store_true",
        help="Export/check an existing output directory without solving CFD.",
    )
    args = parser.parse_args()

    # The archived solver uses dt=0.02; end must fall on a full time step.
    if (
        not math.isfinite(args.end)
        or args.end < .02
        or abs(args.end / .02 - round(args.end / .02)) > 1e-9
    ):
        parser.error("--end must be at least 0.02 and an integer multiple of 0.02.")
    if args.verify and args.end != 90.0:
        parser.error("--verify requires --end 90 for the complete archived case.")

    # Apply the explicit data root before resolving the manifest's input path.
    try:
        if args.data_root is not None:
            set_source_data_root(args.data_root)
        # initial: resolved external FEM mesh and steady velocity/pressure archive.
        initial = source_file(STEADY_INPUT)
    except (FileNotFoundError, KeyError) as error:
        parser.error(
            f"{error}\nSupply a local Source_Data directory with --data-root or KVN_SOURCE_DATA, or publish DOI 10.5281/zenodo.24505927."
        )

    # output: destination for replayed results; export-only reuses existing results.
    output = args.output.expanduser().resolve()
    if not args.export_only:
        subprocess.run([
            sys.executable, "-B", str(RECOVERED / "cfd_force_replay.py"),
            "--input", str(initial),
            "--mesh", "coarse",
            "--case", "train",
            "--dt", "0.02",
            "--end", str(args.end),
            "--out", str(output),
        ], check=True)

    # result: export counts plus optional source/output verification results.
    result = export_forces(output)
    if args.verify:
        result["verification"] = verify_outputs(output)
    (output / "force_reproduction_audit.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
