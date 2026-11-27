"""Regenerate CSV groups independently and compare them with the dataset.

Comparison distinguishes byte identity, parsed-cell identity and numerical
equivalence. Differing files must have matching headers and row counts;
differing numeric fields may agree within ABS_TOL + REL_TOL*abs(reference).
Identity shortcuts and equal nonfinite values do not establish scientific
validity: allowed missing values and physical ranges belong to each producer's
schema checks.

Each producer writes to a fresh temporary folder so old outputs cannot satisfy
the check. This public workflow retains the deposited as-printed tables as
fixed inputs and does not require a manuscript.
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
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
try:
    from Source.Data_Production.data_assets import fixed_asset
except ModuleNotFoundError:
    from data_assets import fixed_asset
# Resolve contracts after parsing the explicit dataset, never at import time.
CONTRACTS = []


def load_contracts(data: Path) -> list[dict]:
    contracts = json.loads(fixed_asset("verified_outputs.json", data=data).read_text())
    if not contracts:
        raise ValueError("The selected dataset has no reproduction contracts")
    return contracts


ABS_TOL = 1e-12
REL_TOL = 1e-10
TEXT_COLUMNS = {"pair_bits", "source", "id", "name", "distances", "u_levels"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.reader(file))


def compare(reference, actual):
    """Return the equality mode and largest compared absolute numeric difference.

    SHA_IDENTICAL compares bytes; CSV_IDENTICAL compares parsed strings;
    NUMERIC_EQUIVALENT also permits numeric formatting and the stated tolerance.
    Named text columns never use numeric tolerance. Equal NaNs and equal signed
    infinities are accepted here; this comparator does not validate whether a
    producer's schema permits them. Identity shortcuts do not validate schema.
    """
    if digest(reference) == digest(actual):
        return "SHA_IDENTICAL", 0.0
    expected, found = read(reference), read(actual)
    if (
        not expected
        or not found
        or expected[0] != found[0]
        or len(expected) != len(found)
    ):
        raise ValueError("Schema or row count differs")
    # Exact string equality is recorded separately from numerical equality.
    # Both checks require the same column order and number of rows.
    if expected == found:
        return "CSV_IDENTICAL", 0.0
    # Track the largest absolute discrepancy; relative tolerance is evaluated
    # against each reference field rather than this aggregate value.
    maximum = 0.0
    for index, (left, right) in enumerate(zip(expected[1:], found[1:], strict=True), 1):
        if len(left) != len(right):
            raise ValueError(f"Row {index} has a different number of cells")
        for label, a, b in zip(expected[0], left, right, strict=True):
            if a == b:
                continue
            if label in TEXT_COLUMNS:
                raise ValueError(f"Row {index}, {label}: {a!r} versus {b!r}")
            try:
                x, y = float(a), float(b)
            except ValueError as err:
                raise ValueError(f"Row {index}, {label}: {a!r} versus {b!r}") from err
            if math.isnan(x) and math.isnan(y) or x == y:
                continue
            delta = abs(x - y)
            if not math.isfinite(delta) or delta > ABS_TOL + REL_TOL * abs(x):
                raise ValueError(f"Row {index}, {label}: {a!r} versus {b!r}")
            maximum = max(maximum, delta)
    return "NUMERIC_EQUIVALENT", maximum


def produce(group, release, data, si, out):
    """Dispatch one producer group into a fresh output folder with bounded thread counts."""
    env = dict(
        os.environ,
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        PYTHONDONTWRITEBYTECODE="1",
        MPLBACKEND="Agg",
    )
    code_root = HERE.parents[1]
    env["KVN_SOURCE_DATA"] = str(data)
    env["PYTHONPATH"] = str(code_root) + os.pathsep + env.get("PYTHONPATH", "")
    if group == "s8":
        env["KVN_SOURCE_DATA"] = str(data)
        program = HERE.parent / "RE/generate_supplement_data.py"
        subprocess.run(
            [sys.executable, str(program), "--s8-only", "--output", str(out)],
            env=env,
            check=True,
        )
        return
    if group == "ibm":
        code_root = HERE.parents[1]
        env.update(
            PYTHONPATH=str(code_root),
            KVN_SOURCE_DATA=str(data),
            KVN_IBM_OUTPUT=str(out),
        )
        subprocess.run(
            [sys.executable, str(code_root / "Hardware/IBM/code/analyze_mps_ibm.py")],
            cwd=code_root,
            env=env,
            check=True,
        )
        return
    if group.startswith("history-"):
        args = [
            "replay_history.py",
            "--job",
            group.removeprefix("history-"),
            "--source-data",
            str(data),
        ]
    elif group == "retained_experiments":
        args = [
            "retained_experiments.py",
            "--source-data",
            str(data),
            "--release-root",
            str(release),
        ]
    elif group == "printed_tables":
        raise ValueError("Printed-table extraction is available only in the internal validation workspace")
    else:
        args = [group + ".py", "--source-data", str(data)]
    subprocess.run(
        [sys.executable, str(HERE / args[0]), *args[1:], "--output", str(out)],
        env=env,
        check=True,
    )


def main():
    """Regenerate requested groups and write per-file equality and digest results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release-root",
        type=Path,
        default=HERE.parents[2],
        help="Parent of the code repository; Source_Data is expected here unless --source-data is supplied",
    )
    parser.add_argument(
        "--source-data", type=Path, help="Override the release Source_Data directory"
    )
    parser.add_argument("--group", default="all", help="Named numerical producer group, or all")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory for the verification report and copies of fresh outputs",
    )
    args = parser.parse_args()
    release = args.release_root.resolve()
    data = (args.source_data or release / "Source_Data").resolve()
    contracts = load_contracts(data)
    available = sorted({c["group"] for c in contracts if c["group"] != "printed_tables"})
    if args.group not in ("all", *available):
        parser.error(f"Choose a public numerical group: {', '.join(available)}")
    destination = args.output.resolve()
    if destination.is_relative_to(data):
        raise ValueError("Choose an output directory outside the published dataset")
    destination.mkdir(parents=True, exist_ok=True)
    groups = available if args.group == "all" else [args.group]
    records = []
    for group in groups:
        # Stale files from earlier runs cannot satisfy a comparison.
        with tempfile.TemporaryDirectory(prefix="kvn_csv_check_") as temp:
            out = Path(temp)
            produce(group, release, data, None, out)
            for contract in (c for c in contracts if c["group"] == group):
                source = data / contract["source"]
                result = out / contract["output"]
                mode, maximum = compare(source, result)
                records.append(
                    dict(
                        **contract,
                        comparison=mode,
                        max_abs_difference=maximum,
                        sha256_source=digest(source),
                        sha256_output=digest(result),
                    )
                )
            shutil.copytree(out, destination / group, dirs_exist_ok=True)
    records.sort(key=lambda x: x["id"])
    report = destination / ("verification_" + args.group + ".csv")
    with report.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"Verified {len(records)} paths. Report: {report}")


if __name__ == "__main__":
    main()
