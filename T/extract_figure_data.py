"""Compatibility wrapper for legacy per-figure extract_data.py entry points.

The public dataset already contains the source tables and editable masters.
Export their exact bytes from the DOI dataset into a local Build/ directory.
"""

from pathlib import Path

from export_published_figure_inputs import export


def run(root: Path) -> Path:
    return export(root)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("figure_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    export(args.figure_root, args.output)
