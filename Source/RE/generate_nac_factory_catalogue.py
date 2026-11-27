#!/usr/bin/env python3
"""Regenerate the numerical factory catalogue using the upstream Litinski implementation.

Inputs
------
An explicit path to litinski/magicstates/Python, candidate configurations and archived reference rows.

Outputs
-------
The --output CSV and checkpoint/run metadata; default publication mode targets the selected dataset.

Run from the repository root
----------------------------
python -m Source.RE.generate_nac_factory_catalogue --litinski-python /path/to/magicstates/Python --output ../Source_Data/Generated_Outputs/Litinski_catalogue.csv

Scope
-----
The upstream distillation model is kept separate from hardware-layout corrections in Source.RE.Litinski.

Implementation details
----------------------
Regenerate the NAC numerical factory catalogue from Litinski's public code.

Source: D. Litinski, "Magic State Distillation: Not as Costly as You Think,"
Quantum 3, 205 (2019); https://github.com/litinski/magicstates.

The public Python functions return the postselected output error, physical
qubit count, average batch duration (including distillation retries), and
number of output T states. This script records those quantities without
changing the original noise model or introducing resource-estimation costs.

In particular, NO d+1 area padding, connection geometry, output-state growth
error, or common-core correction is applied here. Those modifications belong
to Source/RE/Litinski.py, not to the published NAC source catalogue.

Usage from the GitHub repository root (containing Source/ and Figure_Plots/)::

    git clone https://github.com/litinski/magicstates.git ../magicstates
    python -m Source.RE.generate_nac_factory_catalogue         --litinski-python ../magicstates/Python

By default, the script reproduces the 126 (factory, p_phys) combinations in
the archived source CSV and, after all comparisons pass, writes the final
result to its original Source_Data path. Until completion, the published CSV
is preserved and checkpoints are kept in Generated_Outputs/Regenerated/Litinski/. Use
--full-grid with a distinct --output to evaluate all 13 x 10 = 130 pairs,
including four unpublished configurations at 9e-4.

The source code's twolevel20to4.py contains a debugging breakpoint(); this
script temporarily disables that hook during the original numerical calls.
No numerical statement or algorithm in the original functions is modified.

Numerical reproduction uses 80 decimal digits, matching the earlier
official_data.py adapter. The compatibility wrapper around scipy.optimize.root
converts its one-element vector argument to a scalar for the upstream scalar
callbacks without replacing SciPy's solver or changing the factory equations.
The run metadata records the exact input/source checksums, software versions,
and job selection, and is checked before resuming an interrupted calculation.
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
import builtins
import csv
import hashlib
import importlib
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch


# Match the existing Source_Data schema and column order. "qubits" and
# "qubits_unpadded" both contain the author's original (unpadded) result.
FIELDS = (
    "name",
    "kind",
    "level",
    "dx",
    "dz",
    "dm",
    "dx2",
    "dz2",
    "dm2",
    "nl1",
    "d_out",
    "p_phys",
    "p_out",
    "qubits",
    "cycles_per_batch",
    "output_states",
    "cycles_per_output",
    "factory_source",
    "official_source_sha256",
    "status",
    "error",
    "qubits_unpadded",
    "area_policy",
)

# Each catalogue kind maps to the corresponding function in the author's
# repository. All function arguments are original code distances, with no
# geometric padding. The 20-to-4 routine produces four states per batch.
FUNCTIONS = {
    "one": ("onelevel15to1", "cost_of_one_level_15to1", ("dx", "dz", "dm")),
    "small_one": (
        "smallfootprint",
        "cost_of_one_level_15to1_small_footprint",
        ("dx", "dz", "dm"),
    ),
    "two15": (
        "twolevel15to1",
        "cost_of_two_level_15to1",
        ("dx", "dz", "dm", "dx2", "dz2", "dm2", "nl1"),
    ),
    "two20": (
        "twolevel20to4",
        "cost_of_two_level_20to4",
        ("dx", "dz", "dm", "dx2", "dz2", "dm2", "nl1"),
    ),
    "small_two": (
        "smallfootprint",
        "cost_of_two_level_15to1_small_footprint",
        ("dx", "dz", "dm", "dx2", "dz2", "dm2"),
    ),
}

DEFAULT_RATES = tuple(Decimal(i) / Decimal(10000) for i in range(1, 11))
SOURCE_FILES = (
    "definitions.py",
    "magic_state_factory.py",
    "onelevel15to1.py",
    "twolevel15to1.py",
    "twolevel20to4.py",
    "smallfootprint.py",
)


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read an input catalogue without changing its original string fields."""
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def inputs_from_project() -> Path:
    """Resolve the project's published Source_Data directory when installed."""
    codes_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(codes_root))
    from Source.data_locations import shared_data

    return shared_data("RE_common", "RE", "Litinski")


def source_digest(source_dir: Path) -> str:
    """Use the original official_data.py source-fingerprint convention.

    The digest includes the sorted file names and their exact bytes. Matching
    this convention does not imply that the current upstream source bytes
    match those used for the archived CSV; never copy its historical digest.
    """
    digest = hashlib.sha256()
    for name in sorted(SOURCE_FILES):
        path = source_dir / name
        digest.update(name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def load_original_functions(source_dir: Path, decimal_precision: int):
    """Set precision *before* importing the original numerical functions.

    The public definitions.py constructs mpmath matrices at import time.
    Raising mp.dps only after that import leaves those matrices at the
    default precision and changes the resulting very small output errors.
    """
    missing = [name for name in SOURCE_FILES if not (source_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing public-source files in {source_dir}: {missing}"
        )
    from mpmath import mp

    mp.dps = decimal_precision
    sys.path.insert(
        0, str(source_dir)
    )  # The public modules use local absolute imports.

    functions = {
        kind: getattr(importlib.import_module(module), function)
        for kind, (module, function, _) in FUNCTIONS.items()
    }
    return mp, functions


@contextmanager
def disable_upstream_breakpoint():
    """Suppress the unconditional breakpoint in the original 20-to-4 module.

    This replaces only the debugger hook while a factory function runs; it
    does not alter its density-matrix, acceptance, or timing calculations.
    """
    with patch.object(builtins, "breakpoint", lambda *args, **kwargs: None):
        yield


@contextmanager
def scipy_scalar_root_compatibility():
    """Preserve the original adapter's one-dimensional SciPy compatibility.

    The published auxiliary full-code-distance callbacks expect scalar d,
    while scipy.optimize.root supplies a length-one NumPy array. The original
    solver and the published formula are unchanged, and the patch is restored
    after each factory evaluation (including failures).
    """
    from scipy import optimize
    import numpy as np

    original_root = optimize.root

    def compatible_root(fun, x0, *args, **kwargs):
        def scalar_callback(x, *callback_args):
            arr = np.asarray(x)
            if arr.size == 1:
                return fun(float(arr.reshape(-1)[0]), *callback_args)
            return fun(x, *callback_args)

        return original_root(scalar_callback, x0, *args, **kwargs)

    try:
        optimize.root = compatible_root
        yield
    finally:
        optimize.root = original_root


def evaluate(candidate: dict[str, str], p_text: str, functions, mp, digest: str):
    """Evaluate one published factory configuration at one physical error.

    The author's MagicStateFactory provides p_out, qubits, batch cycles, and
    output count directly. Its batch duration already contains the expected
    retry penalty; do not divide by the acceptance probability again.
    For 20-to-4, the author's error field is already per output T state;
    only its batch duration is divided by the number of outputs below.
    """
    kind = candidate["kind"]
    if kind not in FUNCTIONS:
        raise ValueError(f"Unknown NAC factory kind {kind!r}")
    argument_names = FUNCTIONS[kind][2]
    values = [int(candidate[key]) for key in argument_names]

    with disable_upstream_breakpoint(), scipy_scalar_root_compatibility():
        factory = functions[kind](mp.mpf(p_text), *values)

    error = float(factory.distilled_magic_state_error_rate)
    batch = float(factory.distillation_time_in_cycles)
    qubits = int(factory.qubits)
    n_out = int(factory.n_t_gates_produced_per_distillation)
    if not (0 <= error <= 1 and math.isfinite(error)):
        raise ValueError(f"Invalid output error for {candidate['name']}: {error}")
    if not (batch > 0 and math.isfinite(batch) and qubits > 0 and n_out > 0):
        raise ValueError(f"Invalid factory costs for {candidate['name']}")

    return {
        **{key: candidate.get(key, "") for key in FIELDS[:11]},
        "p_phys": p_text,
        "p_out": repr(error),
        "qubits": str(qubits),
        "cycles_per_batch": repr(batch),
        "output_states": str(n_out),
        "cycles_per_output": repr(batch / n_out),
        "factory_source": "official_litinski_magicstates_function",
        "official_source_sha256": digest,
        "status": "ok",
        "error": "",
        "qubits_unpadded": str(qubits),
        "area_policy": "original_official_geometry_attachment_baseline",
    }


def jobs_to_run(candidates, reference, full_grid: bool, rates):
    """Select original candidate/rate pairs without fabricating missing data.

    Default: match the 126 archived source rows. Full-grid mode explicitly
    includes the four configurations omitted at p_phys=9e-4 in the archive.
    """
    by_name = {row["name"]: row for row in candidates}
    if len(by_name) != len(candidates):
        raise ValueError("Duplicate factory names in candidate catalogue")
    if full_grid:
        distinct_rates = (
            sorted({Decimal(x) for x in rates}) if rates else list(DEFAULT_RATES)
        )
        return [
            (by_name[name], str(rate))
            for rate in distinct_rates
            for name in sorted(by_name)
        ]
    if not reference:
        raise ValueError(
            "The archived reference CSV is required unless --full-grid is given"
        )
    selected_rates = {Decimal(x) for x in rates} if rates else None
    jobs = []
    for row in reference:
        if row["status"] != "ok":
            continue
        name, p_text = row["name"], row["p_phys"]
        if selected_rates is not None and Decimal(p_text) not in selected_rates:
            continue
        candidate = by_name.get(name)
        if candidate is None:
            raise ValueError(f"Published factory {name!r} is absent from candidates")
        # The archived table serializes some integer distances as "25.0"
        # whereas the input catalogue uses "25". Compare their values, not
        # the incidental formatting of the CSV strings.
        same_geometry = candidate["kind"] == row["kind"] and all(
            (
                candidate[key] == row[key]
                if key in ("name", "kind")
                else (
                    Decimal(candidate[key]) == Decimal(row[key])
                    if candidate[key] and row[key]
                    else candidate[key] == row[key]
                )
            )
            for key in FIELDS[:11]
        )
        if not same_geometry:
            raise ValueError(
                f"Geometry differs between candidate and source CSV: {name}"
            )
        jobs.append((candidate, p_text))
    if len({(row["name"], p) for row, p in jobs}) != len(jobs):
        raise ValueError("Duplicate factory/rate pairs in archived source")
    return jobs


def verify_against_reference(result, reference_row, rtol: float):
    """Check original numerical outputs, excluding provenance metadata.

    This does not assert equality of the source digest: a newly fetched
    upstream revision may produce the same numbers with different source bytes.
    """
    for key in ("p_out", "cycles_per_batch", "cycles_per_output"):
        current, archived = float(result[key]), float(reference_row[key])
        if not math.isclose(current, archived, rel_tol=rtol, abs_tol=0.0):
            raise AssertionError(
                f"{result['name']} at p={result['p_phys']}: "
                f"{key}={current} != archived {archived} (rtol={rtol})"
            )
    for key in ("qubits", "output_states", "qubits_unpadded"):
        if int(result[key]) != int(reference_row[key]):
            raise AssertionError(
                f"{result['name']}: {key} differs from archived source"
            )


def write_csv_atomic(path: Path, rows: list[dict[str, str]]):
    """Checkpoint results atomically so long numerical runs can be resumed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_name = handle.name
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp_name, path)
    finally:
        if temp_name is not None and os.path.exists(temp_name):
            os.unlink(temp_name)


def run_metadata(
    source_dir: Path,
    digest: str,
    candidate_csv: Path,
    reference_csv: Path | None,
    jobs,
    args,
) -> dict:
    """Record the source, numerical environment, and exact selected jobs.

    A matching source SHA alone is insufficient for safe checkpoint reuse:
    mpmath precision, installed numerical libraries, and requested jobs must
    also agree. Individual upstream-file hashes help trace source revisions.
    """
    import mpmath
    import numpy
    import scipy

    checkout = source_dir.parent if source_dir.name == "Python" else source_dir
    official_commit = None
    if (checkout / ".git").exists():
        try:
            official_commit = subprocess.run(
                ["git", "-C", str(checkout), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass

    return {
        "format_version": 1,
        "official_source_sha256": digest,
        "official_git_commit_if_available": official_commit,
        "official_python_file_sha256": {
            name: hashlib.sha256((source_dir / name).read_bytes()).hexdigest()
            for name in sorted(SOURCE_FILES)
        },
        "candidate_csv_sha256": hashlib.sha256(candidate_csv.read_bytes()).hexdigest(),
        "reference_csv_sha256": (
            hashlib.sha256(reference_csv.read_bytes()).hexdigest()
            if reference_csv is not None and reference_csv.is_file()
            else None
        ),
        "python_version": platform.python_version(),
        "numpy_version": numpy.__version__,
        "scipy_version": scipy.__version__,
        "mpmath_version": mpmath.__version__,
        "mpmath_decimal_precision": args.mp_dps,
        "reference_relative_tolerance": args.rtol,
        "selected_jobs": [[candidate["name"], p] for candidate, p in jobs],
    }


def write_json_atomic(path: Path, record: dict):
    """Write provenance metadata without leaving a partial JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_name = handle.name
            json.dump(record, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(temp_name, path)
    finally:
        if temp_name is not None and os.path.exists(temp_name):
            os.unlink(temp_name)


def main(argv=None):
    """Evaluate requested official factory candidates, with provenance-checked resume and reference comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--litinski-python",
        type=Path,
        required=True,
        help="Path to the original litinski/magicstates/Python directory",
    )
    parser.add_argument(
        "--candidate-csv",
        type=Path,
        help="Candidate catalogue; defaults to the published Source_Data",
    )
    parser.add_argument(
        "--reference-csv",
        type=Path,
        help="Archived numerical catalogue used to match/verify published rows",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Final output CSV (default: original Source_Data/Shared_source_data/RE_common/RE/Litinski/... path)",
    )
    parser.add_argument(
        "--full-grid",
        action="store_true",
        help="Evaluate all 13 candidates at each requested p_phys",
    )
    parser.add_argument(
        "--rates",
        nargs="+",
        metavar="P_PHYS",
        help="Subset or override of physical error probabilities",
    )
    parser.add_argument(
        "--kind",
        choices=sorted(FUNCTIONS),
        help="Evaluate one factory family (useful for validation)",
    )
    parser.add_argument(
        "--limit", type=int, help="Evaluate only the first N jobs (smoke tests)"
    )
    parser.add_argument(
        "--mp-dps",
        type=int,
        default=80,
        help="mpmath decimal precision for original simulations (default: 80)",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-6,
        help="Relative tolerance when checking archived numerical results",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume matching checkpoint and run settings",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Start a fresh run, replacing an earlier generated output/checkpoint",
    )
    args = parser.parse_args(argv)

    if args.mp_dps < 40 or args.rtol <= 0:
        parser.error("Require --mp-dps >= 40 and --rtol > 0")
    if args.resume and args.overwrite:
        parser.error("Choose only one of --resume and --overwrite")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    codes_root = Path(__file__).resolve().parents[2]
    data_dir = None
    if (
        args.candidate_csv is None
        or args.output is None
        or (args.reference_csv is None and not args.full_grid)
    ):
        data_dir = inputs_from_project()
    candidate_csv = (
        args.candidate_csv or data_dir / "not_as_costly_factory_candidate_catalog.csv"
    )
    reference_csv = args.reference_csv or (
        data_dir / "not_as_costly_official_factory_numerical.csv"
        if data_dir is not None
        else None
    )
    output_csv = args.output or (
        data_dir / "not_as_costly_official_factory_numerical.csv"
    )
    output_csv = output_csv.expanduser().resolve()
    reference_csv = (
        reference_csv.expanduser().resolve() if reference_csv is not None else None
    )
    candidate_csv = candidate_csv.expanduser().resolve()
    in_place = reference_csv is not None and output_csv == reference_csv
    if output_csv == candidate_csv.expanduser().resolve():
        parser.error("Refusing to overwrite the factory candidate catalogue")
    if in_place:
        if not output_csv.is_file():
            parser.error(
                "Original Source_Data numerical CSV is required for in-place verification"
            )
        if (
            args.kind is not None
            or args.rates is not None
            or args.limit is not None
            or args.full_grid
        ):
            parser.error(
                "In-place publishing requires all 126 archived jobs; use --output for a subset/full-grid test"
            )
        # Preserve the archived CSV throughout the expensive calculation. Only
        # the fully verified result replaces it, atomically, at the end.
        path_tag = hashlib.sha256(str(output_csv).encode("utf-8")).hexdigest()[:10]
        checkpoint_csv = (
            generated_outputs_root()
            / "Regenerated"
            / "Litinski"
            / f"{output_csv.stem}.{path_tag}.inprogress.csv"
        )
    else:
        checkpoint_csv = output_csv
    if checkpoint_csv.exists() and not (args.resume or args.overwrite):
        parser.error(
            f"Checkpoint/output exists: {checkpoint_csv}. Use --resume or --overwrite"
        )

    candidates = read_rows(candidate_csv)
    reference = (
        read_rows(reference_csv)
        if reference_csv is not None and reference_csv.is_file()
        else []
    )
    jobs = jobs_to_run(candidates, reference, args.full_grid, args.rates)
    if args.kind is not None:
        jobs = [
            (candidate, p) for candidate, p in jobs if candidate["kind"] == args.kind
        ]
    if args.limit is not None:
        jobs = jobs[: args.limit]
    if not jobs:
        parser.error("No factory/rate combinations were selected")
    archived = {(r["name"], Decimal(r["p_phys"])): r for r in reference}

    source_dir = args.litinski_python.expanduser().resolve()
    digest = source_digest(source_dir)
    metadata_path = checkpoint_csv.with_suffix(".run_metadata.json")
    metadata = run_metadata(
        source_dir, digest, candidate_csv, reference_csv, jobs, args
    )
    if args.resume and metadata_path.is_file():
        with metadata_path.open(encoding="utf-8") as handle:
            previous = json.load(handle)
        if previous != metadata:
            changed = sorted(
                k
                for k in set(previous) | set(metadata)
                if previous.get(k) != metadata.get(k)
            )
            raise ValueError(
                "Cannot resume with changed numerical inputs or environment: "
                + ", ".join(changed)
                + ". Use the original settings or choose a new output CSV."
            )
    elif args.resume and checkpoint_csv.is_file():
        raise ValueError(
            f"Cannot verify resume settings: missing {metadata_path}. "
            "Choose a new output or restart with --overwrite."
        )
    elif metadata_path.exists() and not args.overwrite and not args.resume:
        parser.error(
            f"Existing run metadata: {metadata_path}. Use --resume or --overwrite"
        )

    if not metadata_path.exists() or args.overwrite:
        write_json_atomic(metadata_path, metadata)
    if args.overwrite and checkpoint_csv.is_file():
        checkpoint_csv.unlink()  # Never removes the archived reference in in-place mode.

    mp, original_functions = load_original_functions(source_dir, args.mp_dps)

    completed = (
        read_rows(checkpoint_csv) if args.resume and checkpoint_csv.is_file() else []
    )
    saved = {(r["name"], Decimal(r["p_phys"])): r for r in completed}
    if len(saved) != len(completed):
        raise ValueError("Duplicate factory/rate pairs in resume file")
    requested = {(r["name"], Decimal(p)) for r, p in jobs}
    if any(key not in requested for key in saved):
        raise ValueError("Resume CSV contains jobs outside the recorded selection")
    for row in completed:
        if row["official_source_sha256"] != digest:
            raise ValueError("Cannot resume: original source files changed")
    print(f"Official source SHA256: {digest}")
    print(f"Jobs: {len(jobs)}; already completed: {len(completed)}")
    for index, (candidate, p_text) in enumerate(jobs, 1):
        key = (candidate["name"], Decimal(p_text))
        if key in saved:
            result = saved[key]
            if key in archived:
                verify_against_reference(result, archived[key], args.rtol)
        else:
            result = evaluate(candidate, p_text, original_functions, mp, digest)
            saved[key] = result
            write_csv_atomic(
                checkpoint_csv,
                [
                    saved[(r["name"], Decimal(p))]
                    for r, p in jobs
                    if (r["name"], Decimal(p)) in saved
                ],
            )
            # Checkpoint expensive outputs even if reference verification fails.
            # An unsuccessful comparison will stop the run, but its numerical
            # result remains available for inspection without recomputation.
            if key in archived:
                verify_against_reference(result, archived[key], args.rtol)
        print(
            f"[{index}/{len(jobs)}] {candidate['name']} p={p_text}: "
            f"p_out={result['p_out']}, cycles={result['cycles_per_batch']}",
            flush=True,
        )
    if in_place:
        # Verify that nobody has changed the source reference during the run.
        current_digest = hashlib.sha256(reference_csv.read_bytes()).hexdigest()
        if current_digest != metadata["reference_csv_sha256"]:
            raise RuntimeError(
                "Archived Source_Data CSV changed during this run; "
                "verified checkpoints are preserved, but publishing stopped"
            )
        final_rows = [saved[(r["name"], Decimal(p))] for r, p in jobs]
        write_csv_atomic(output_csv, final_rows)
    print(f"Saved {len(jobs)} factory evaluations to {output_csv}")
    print(f"Run metadata: {metadata_path}")


if __name__ == "__main__":
    main()
