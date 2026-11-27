"""Read-only Source Data resolver, with a Zenodo DOI download fallback.

Use ``Source_Data/`` next to the cloned GitHub repository or set ``KVN_SOURCE_DATA`` for
offline use. Otherwise, the first data access downloads the public Zenodo
``Source_Data.zip`` into a persistent cache (DOI: 10.5281/zenodo.24505927).
"""

from __future__ import annotations

import csv
import argparse
import os
from functools import lru_cache
from pathlib import Path

from .zenodo_download import cached_path, download_source_data

REPO_ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def source_data_root() -> Path:
    """Find a manifest-bearing data directory, falling back to the configured downloader."""
    explicit = os.environ.get("KVN_SOURCE_DATA")
    candidates = (
        [Path(explicit)]
        if explicit
        else [
            REPO_ROOT.parent / "Source_Data",
            REPO_ROOT / "Source_Data",
            cached_path(),
        ]
    )
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if (resolved / "manifest.csv").is_file():
            return resolved
    if explicit:
        raise FileNotFoundError(f"KVN_SOURCE_DATA={explicit} has no manifest.csv")
    return download_source_data()


def set_source_data_root(directory: str | Path) -> Path:
    """Apply an explicit dataset override and invalidate both resolver caches."""
    root = Path(directory).expanduser().resolve()
    if not (root / "manifest.csv").is_file():
        raise FileNotFoundError(f"Source_Data directory {root} has no manifest.csv")
    os.environ["KVN_SOURCE_DATA"] = str(root)
    source_data_root.cache_clear()
    _original_to_published.cache_clear()
    return root


@lru_cache(maxsize=1)
def _original_to_published() -> dict[str, str]:
    """Map historical input paths to published paths using the dataset manifest."""
    with (source_data_root() / "manifest.csv").open(
        newline="", encoding="utf-8-sig"
    ) as handle:
        records = csv.DictReader(handle)
        return {
            row["original_path"].replace("\\", "/"): row["published_path"]
            for row in records
            if row["original_path"] and row["published_path"]
        }


def published_path(original_relative_path: str) -> Path:
    """Resolve a documented input/output path, including before it is written."""
    key = original_relative_path.replace("\\", "/").lstrip("./")
    external_prefix = "Comparisons/KvN_LCHS/source_replay/QuCF/"
    if key.startswith(external_prefix):
        from .qucf_inputs import input_path

        return input_path(key[len(external_prefix) :])
    published = _original_to_published().get(key)
    if published is None:
        raise KeyError(f"No Source_Data manifest entry for {key}")
    return source_data_root() / published


def source_file(original_relative_path: str) -> Path:
    """Return an existing file in Source_Data by its old repository path."""
    target = published_path(original_relative_path)
    if not target.is_file():
        raise FileNotFoundError(target)
    return target


def figure_data(figure: str) -> Path:
    """Current Figure-local plot inputs (F1, ..., S10, ED1)."""
    if figure.startswith("F"):
        directory = "Figures/Fig_" + figure[1:]
    elif figure.startswith("S"):
        directory = "Figures/Fig_S" + str(int(figure[1:]))
    elif figure == "ED1":
        directory = "Figures/Extended_Data_Fig_1"
    else:
        raise ValueError(figure)
    return source_data_root() / directory / "plot_inputs"


def shared_data(*parts: str) -> Path:
    """Resolve a shared dataset path, including the separately cached QuCF inputs."""
    if parts[:2] == ("RE_common", "LCHS_source_inputs"):
        from .qucf_inputs import input_root

        return input_root() / Path(*parts[2:])
    return source_data_root() / "Shared_source_data" / Path(*parts)


def generated_outputs_root() -> Path:
    """Stage non-publication results in Source_Data, never alongside code."""
    return source_data_root() / "Generated_Outputs"


def source_label(path: str | Path) -> str:
    """Stable provenance label for project files and external Source_Data files."""
    resolved = Path(path).resolve()
    for root, prefix in ((source_data_root(), "Source_Data/"), (REPO_ROOT, "")):
        try:
            return prefix + resolved.relative_to(root).as_posix()
        except ValueError:
            continue
    return resolved.as_posix()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Locate/download Zenodo Source Data")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Fetch public DOI data if it is not already local",
    )
    parser.parse_args()
    print(source_data_root())
