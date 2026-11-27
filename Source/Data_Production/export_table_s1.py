"""Export full-precision model parameters underlying Table S1.

Inputs
------
Fitted coefficients in model_configuration.json and the phase/input record in configuration.json.

Outputs
-------
The Table S1 numerical-source CSV in the selected writable Source_Data tree.

Run from the repository root
----------------------------
python -m Source.Data_Production.export_table_s1 --source-data ../Source_Data_Reproduced

Scope
-----
Rounded printed table entries are not used to reconstruct full-precision parameters.
"""

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
import json
from pathlib import Path

from Source.data_locations import source_data_root


def produce(root: Path) -> Path:
    """Check the two parameter records agree, then write the six full-precision values."""
    target = root / "Tables" / "Table_S1"
    model = json.loads(
        (target / "model_configuration.json").read_text(encoding="utf-8")
    )
    config = json.loads((target / "configuration.json").read_text(encoding="utf-8"))
    p_model, p_config = model["parameters"], config["parameters"]
    for key in ("mu", "R", "omega", "gamma_normalized", "c2"):
        if p_model[key] != p_config[key]:
            raise ValueError(f"Table S1 configuration disagreement: {key}")
    rows = [
        ("mu", p_model["mu"], "R", p_model["R"]),
        ("omega", p_model["omega"], "gamma", p_model["gamma_normalized"]),
        ("theta0", p_config["initial_phase"], "c2", p_model["c2"]),
    ]
    output = target / "Table_S1_full_precision.csv"
    target.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ("parameter_left", "value_left", "parameter_right", "value_right")
        )
        writer.writerows(rows)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-data",
        type=Path,
        help="Source_Data root (defaults to the configured public input root)",
    )
    arguments = parser.parse_args()
    print(produce(arguments.source_data or source_data_root()))
