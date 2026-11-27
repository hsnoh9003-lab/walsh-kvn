"""Merge partial parameter sweeps without discarding other stored conditions.

Inputs
------
Existing tables, newly calculated rows and the keys identifying a sweep condition.

Outputs
-------
Validated merged CSVs and, when publishing, refreshed size/checksum metadata.

Run from the repository root
----------------------------
Imported by generate_supplement_data and generate_litinski_data; not a standalone command.

Scope
-----
Out-of-scope rows are rejected. Requested conditions replace their old rows;
duplicate-key validation is the producer's responsibility. Writers serialize
updates and restore committed files on a caught replacement failure. This is
not a crash-recovery journal or a transaction spanning a complete workflow.
"""

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
    """Merge the requested physical-error fractions/schemes into a staged CSV.

    Unrequested rows retain their exact stored strings and precision. Rates
    are matched after rounding to 12 decimal places; ``schemes=None`` means
    that scheme is not part of the selection key.
    """
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
def writer_lock(destination):
    # The operating system releases this cross-platform lock if a process exits.
    """Serialize writers to one output destination using an operating-system-released lock."""
    token = sha256(os.fsencode(Path(destination).resolve())).hexdigest()
    lock = Path(gettempdir()) / f"kvn-resource-sweep-{token}.sqlite3"
    with closing(sqlite3.connect(lock, timeout=60)) as connection:
        connection.execute("BEGIN IMMEDIATE")
        yield


def publish_dataset_csvs(root, sources):
    """Replace selected dataset CSVs and their manifest digests under one writer lock.

    ``sources`` maps destination Paths to staged source Paths. All targets must
    have manifest records. Backups restore committed files if replacement
    raises; unchanged payloads and unrelated manifest rows are preserved.
    """
    import io

    manifest = root / "manifest.csv"
    with writer_lock(root), TemporaryDirectory(dir=root, prefix=".publish-") as folder:
        stage = Path(folder)
        files = {target: source.read_bytes() for target, source in sources.items()}
        fields, rows = read_table(manifest)
        records = {row["published_path"]: row for row in rows}
        for target, data in files.items():
            row = records[target.relative_to(root).as_posix()]
            row.update(sha256=sha256(data).hexdigest(), size_bytes=str(len(data)))
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        files[manifest] = stream.getvalue().encode("utf-8")
        changes = []
        for index, (target, data) in enumerate(files.items()):
            previous = stage / f"previous-{index}" if target.exists() else None
            if previous is not None:
                if target.read_bytes() == data:
                    continue
                shutil.copy2(target, previous)
            fresh = stage / str(index)
            fresh.write_bytes(data)
            changes.append((fresh, target, previous))
        committed = []
        try:
            for fresh, target, previous in changes:
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(fresh, target)
                committed.append((target, previous))
        except BaseException:
            for target, previous in reversed(committed):
                if previous is None:
                    target.unlink(missing_ok=True)
                else:
                    os.replace(previous, target)
            raise


@contextmanager
def staged_sweep(destination, rates, schemes=None, synchronize_metadata=None):
    """Yield a staging directory, then merge and commit its files under a lock.

    No staged files are published if the calculation inside the context raises.
    On exit, requested rate/scheme rows replace existing rows; other rows stay.
    A caught commit failure restores earlier replacements. The optional
    metadata callback runs before merging and is not itself rolled back here.
    """
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    rates = {round(float(p), 12) for p in rates}
    schemes = None if schemes is None else set(schemes)
    with TemporaryDirectory(dir=destination, prefix=".sweep-") as folder:
        stage = Path(folder)
        yield stage
        outputs = sorted(p for p in stage.iterdir() if p.is_file())
        with writer_lock(destination):
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
