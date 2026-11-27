"""Copy only the archived inputs needed by demo.ipynb; never modify the deposit."""

from __future__ import annotations

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
from pathlib import Path
import shutil

from Source.data_locations import set_source_data_root, source_data_root, source_file


def prepare(source: Path, output: Path) -> tuple[int, int]:
    """Copy hash-checked demo inputs/previews and return their count and byte size.

    ``output`` must not already exist. Selecting ``source`` also changes the
    process-wide data resolver; callers that need the old root must restore it.
    The subset keeps input bytes and manifest rows, not full-reproduction data.
    """
    root = set_source_data_root(source)
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError("Choose a new output directory for the demo subset")
    originals = [
        "Data/Model/baseline_inputs.npz",
        "Data/Model/candidate_K2.npz",
        "Data/RE/gidney_circuit_footprints.csv",
    ]
    paths = {source_file(name).relative_to(root).as_posix() for name in originals}
    paths.update(
        [
            "Figures/Fig_4/plot_inputs/model_degree/selected_by_library.csv",
            "Figures/Fig_4/plot_inputs/plot_values/Re100/fig4f_fem_forces.csv",
            "Figures/Fig_4/plot_inputs/plot_values/Re100/fig4f_walsh_forces.csv",
            "Figures/Fig_4/force_sampling/force_error_scales.csv",
            "Figures/Fig_5/plot_inputs/wake/data/restart40/terminal_readout.json",
            "Shared_source_data/RE_common/RE/prep_p5.csv",
            "Shared_source_data/RE_common/RE/reference_selected.csv",
            "Additional_Source_Data/Example_Inputs/resource_workload.json",
        ]
    )
    paths.update(
        f"Additional_Source_Data/Demo_Previews/F{i}/{name}"
        for i, name in enumerate(
            ("Figure_1.png", "Figure_2.png", "Figure3.png", "Figure_4.png", "Figure5_updated.png"),
            start=1,
        )
    )
    with (root / "manifest.csv").open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames
        rows = [row for row in reader if row["published_path"] in paths]
    if {row["published_path"] for row in rows} != paths:
        raise ValueError("Demo inputs are missing from the manifest")
    # Verify everything before creating the subset.
    for row in rows:
        actual = hashlib.sha256((root / row["published_path"]).read_bytes()).hexdigest()
        if actual != row["sha256"]:
            raise ValueError("Archived checksum mismatch: " + row["published_path"])
    output.mkdir(parents=True)
    for relative in sorted(paths):
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / relative, target)
    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    with (root / "Source_Data_Index.csv").open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        index_columns = reader.fieldnames
        index = [row for row in reader if row["path_from_source_data_root"] in paths]
    with (output / "Source_Data_Index.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=index_columns)
        writer.writeheader()
        writer.writerows(index)
    for name in ("LICENSE", "CITATION.cff"):
        if (root / name).is_file():
            shutil.copy2(root / name, output / name)
    (output / "README.md").write_text(
        "# Demo input subset\n\n"
        "These unchanged payloads are a subset of the Zenodo Source_Data deposit. "
        "Use this directory with demo.ipynb via KVN_SOURCE_DATA. It is not the "
        "complete dataset needed for quick_check, figure plotters or full reproduction.\n",
        encoding="utf-8",
    )
    return len(rows), sum(int(row["size_bytes"]) for row in rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    count, size = prepare(args.source_data or source_data_root(), args.output)
    print(f"Demo subset: {count} unchanged payloads, {size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
