"""Read independently generated FEM force trajectories and fitted coefficient tables.

These are the exact validation/readout assertions used in F/F2's force_model/observables.py.
No CFD solver is implied; input CSVs are supplied by the original CFD pipeline.
"""
from pathlib import Path
import pandas as pd
from Source.data_locations import source_file


def _force_file(force_data, names, original):
    if force_data is None:
        return source_file(original)
    path = Path(force_data)
    if path.is_file():
        return path
    for name in names:
        candidate = path / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Expected {', '.join(names)} in {path}, or pass a CSV file.")


def read_fem_forces(force_data: Path | None = None, end_time: float = 60):
    path = _force_file(force_data, ("forces_cfd.csv", "fig4f_fem_forces.csv"),
                       "Figure/F4/data/force_output/fig4f_fem_forces.csv")
    fem = pd.read_csv(path)
    fem = fem.loc[fem["time"] <= end_time].copy()
    assert fem["pressure_valid"].all() and fem["time"].iloc[-1] == end_time
    assert fem["time"].is_monotonic_increasing
    return fem


def read_force_coefficients(force_data: Path | None = None):
    path = _force_file(force_data, ("force_coefficients.csv",),
                       "Figure/F4/data/plot_values/Re100/force_coefficients.csv")
    return pd.read_csv(path)
