"""Export a figure's exact DOI-hosted inputs without altering Source Data."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def export(figure_root: Path, destination: Path | None = None) -> Path:
    """Copy archived figure inputs to an export directory without recomputation."""
    from Figure_Plots.common.paths import figure_root as resolve_figure_root

    candidate = Path(figure_root)
    figure_root = (
        candidate.resolve()
        if candidate.is_dir()
        else resolve_figure_root(str(figure_root))
    )
    repository = figure_root.parents[1]
    sys.path.insert(0, str(repository))
    from Source.data_locations import source_data_root, generated_outputs_root

    if figure_root.name.startswith("F"):
        key = "Fig_" + figure_root.name[1:]
    elif figure_root.name.startswith("S"):
        key = "Fig_S" + str(int(figure_root.name[1:]))
    elif figure_root.name == "ED1":
        key = "Extended_Data_Fig_1"
    else:
        raise ValueError(f"Not a recognized figure root: {figure_root}")

    source = source_data_root() / "Figures" / key
    if not source.is_dir():
        raise FileNotFoundError(f"This figure has no published Source Data: {source}")
    output = (
        Path(destination)
        if destination
        else generated_outputs_root()
        / "Figure_Source_Exports"
        / figure_root.name
    ).resolve()
    if output == source or source in output.parents:
        raise ValueError("Source Data is read-only; choose a separate output directory")
    output.mkdir(parents=True, exist_ok=True)
    count = 0
    for item in source.rglob("*"):
        if item.is_file():
            target = output / item.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)
            count += 1
    print(f"Exported {count} original source files from {source} to {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("figure_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    export(args.figure_root, args.output)


if __name__ == "__main__":
    main()
