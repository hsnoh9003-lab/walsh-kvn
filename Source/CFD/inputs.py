"""Resolve published FEM inputs through the common local/Zenodo data resolver."""

from pathlib import Path

from Source.data_locations import source_file

STEADY_INPUT = "Source/CFD/replay_fem/inputs/steady_coarse.npz"


def steady_input(mesh="coarse", reynolds=100.0, input_path=None):
    """Use an explicit newly generated field, or the published Re100 coarse field."""
    if input_path is not None:
        path = Path(input_path).expanduser().resolve()
    else:
        if mesh != "coarse" or reynolds != 100.0:
            raise FileNotFoundError(
                "The published steady input supports only Re=100 and mesh=coarse. "
                "Generate a matching steady field and pass --input /path/to/field.npz."
            )
        path = source_file(STEADY_INPUT)
    if not path.is_file():
        raise FileNotFoundError(f"Missing steady FEM input: {path}")
    return path
