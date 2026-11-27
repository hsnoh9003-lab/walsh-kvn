"""Resolve archived reproduction inputs from the Source_Data dataset.

All fixed arrays, acquired records, schemas and saved experiment checkpoints
belong to Source_Data.  This module only locates them; it never copies them
into the executable code tree.
"""

from pathlib import Path
import sys

try:
    from Source.data_locations import source_data_root
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from Source.data_locations import source_data_root


ASSETS = Path("Additional_Source_Data/Reproduction_Inputs")


def fixed_root(data: Path | None = None) -> Path:
    """Return the root containing the retained non-publication inputs."""
    return (Path(data) if data is not None else source_data_root()) / ASSETS


def fixed_asset(*parts: str, data: Path | None = None) -> Path:
    """Locate a fixed input without depending on a Codes-relative data file."""
    return fixed_root(data).joinpath(*parts)
