"""Publish rate sweeps while preserving results for conditions outside a rerun."""
import csv
from contextlib import closing, contextmanager
from hashlib import sha256
import os
from pathlib import Path
import shutil
import sqlite3
from tempfile import TemporaryDirectory, gettempdir


def read_table(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or ()), list(reader)


def merge_table(staged, existing, rates, schemes):
    fields, fresh = read_table(staged)
    previous_fields, previous = read_table(existing) if existing.exists() else ([], [])
    if not fields:
        raise ValueError(f"Missing column names in {staged.name}")

    def group(row):
        scheme = row["scheme"] if schemes is not None else ""
        return scheme, round(float(row["p_phys"]), 12)

    def requested(row):
        scheme, rate = group(row)
        return rate in rates and (schemes is None or scheme in schemes)

    if any(not requested(row) for row in fresh):
        raise ValueError(f"{staged.name} contains unrequested sweep conditions")
    retained = [row for row in previous if not requested(row)]
    if existing.is_file() and not fresh and len(retained) == len(previous):
        shutil.copyfile(existing, staged)
        return
    if not retained:
        return
    # Pandas' dtype inference can round previously saved numbers. Copy retained
    # fields as CSV strings, including empty fields and multi-level distances.
    columns = list(dict.fromkeys(previous_fields + fields))
    combined = sorted(retained + fresh, key=group)
    with staged.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(combined)


@contextmanager
def staged_sweep(destination, rates, schemes=None, synchronize_metadata=None):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    rates = {round(float(p), 12) for p in rates}
    schemes = None if schemes is None else set(schemes)
    with TemporaryDirectory(dir=destination, prefix=".sweep-") as folder:
        stage = Path(folder)
        yield stage
        outputs = sorted(p for p in stage.iterdir() if p.is_file())
        # The OS releases this cross-platform writer lock if a process exits.
        token = sha256(os.fsencode(destination)).hexdigest()
        lock = Path(gettempdir()) / f"kvn-resource-sweep-{token}.sqlite3"
        with closing(sqlite3.connect(lock, timeout=60)) as connection:
            connection.execute("BEGIN IMMEDIATE")
            if synchronize_metadata is not None:
                synchronize_metadata(stage, destination)
            backup = stage / "previous"
            backup.mkdir()
            for path in outputs:
                target = destination / path.name
                if path.suffix == ".csv":
                    merge_table(path, target, rates, schemes)
                if target.exists():
                    shutil.copy2(target, backup / path.name)
            committed = []
            try:
                for path in outputs:
                    target = destination / path.name
                    os.replace(path, target)
                    committed.append(target)
            except BaseException:
                for target in reversed(committed):
                    saved = backup / target.name
                    if saved.exists():
                        os.replace(saved, target)
                    else:
                        target.unlink()
                raise
