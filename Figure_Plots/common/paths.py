"""Shared locations for figure code and generated artwork."""

from pathlib import Path
from Source.data_locations import generated_outputs_root

REPOSITORY = Path(__file__).resolve().parents[2]
GROUPS = {
    "main": ("F1", "F3", "F4", "F5"),
    "supplementary": ("S01", "S03", "S04", "S05", "S08", "S09", "S10"),
    "extended": ("ED1",),
}
AVAILABLE = tuple(name for names in GROUPS.values() for name in names)


def canonical_name(name: str) -> str:
    name = name.upper()
    if name.startswith("S") and name[1:].isdigit():
        name = f"S{int(name[1:]):02d}"
    if name not in AVAILABLE:
        raise ValueError(
            f'No public plotter for {name}; available: {", ".join(AVAILABLE)}'
        )
    return name


def figure_root(name: str) -> Path:
    return REPOSITORY / "Figure_Plots" / canonical_name(name)


def output_dir(name: str) -> Path:
    """Locate generated artwork separately from the archived figure inputs."""
    return generated_outputs_root() / "Figures" / canonical_name(name)
