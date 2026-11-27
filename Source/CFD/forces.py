"""Read independently generated FEM forces and fitted force-observable coefficients.

Input CSVs come from the CFD pipeline; these helpers do not run a solver."""

from pathlib import Path
import pandas as pd
from Source.data_locations import source_file


def _force_file(force_data, names, original):
    """Resolve a supplied CSV, a directory containing it, or the published input."""
    if force_data is None:
        return source_file(original)
    path = Path(force_data)
    if path.is_file():
        return path
    for name in names:
        candidate = path / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Expected {', '.join(names)} in {path}, or pass a CSV file."
    )


def read_fem_forces(force_data: Path | None = None, end_time: float = 60):
    """Read the FEM force trajectory through the exact stored end_time; require valid pressure."""
    path = _force_file(
        force_data,
        ("forces_cfd.csv", "fig4f_fem_forces.csv"),
        "Figure/F4/data/force_output/fig4f_fem_forces.csv",
    )
    fem = pd.read_csv(path)
    fem = fem.loc[fem["time"] <= end_time].copy()
    assert fem["pressure_valid"].all() and fem["time"].iloc[-1] == end_time
    assert fem["time"].is_monotonic_increasing
    return fem


def read_force_coefficients(force_data: Path | None = None):
    """Read the fitted force-observable coefficient table without recomputing CFD."""
    path = _force_file(
        force_data,
        ("force_coefficients.csv",),
        "Figure/F4/data/plot_values/Re100/force_coefficients.csv",
    )
    return pd.read_csv(path)
