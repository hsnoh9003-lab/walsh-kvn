"""Export selected archived refinement rows, keeping their numeric values intact."""

# Direct-file execution must locate Codes/ independently of the working directory.
if __package__ in (None, ""):
    import sys as _path_sys
    from pathlib import Path as _Path
    _codes = next(p for p in _Path(__file__).resolve().parents
                  if (p / "Source/data_locations.py").is_file())
    _path_sys.path.insert(0, str(_codes))



from pathlib import Path
import csv
import sys
from plot import CASES, FIGURE as ROOT

sys.path.insert(0, str(ROOT.parents[1]))
from Source.data_locations import generated_outputs_root
from Source.data_locations import figure_data


def extract(output):
    output.mkdir(exist_ok=True, parents=True)
    for source in sorted(figure_data("S03").glob("*_refinement.csv")):
        with source.open(newline="") as original, (output / source.name).open(
            "w", newline=""
        ) as target:
            rows = csv.DictReader(original)
            writer = csv.DictWriter(target, rows.fieldnames)
            writer.writeheader()
            for row in rows:
                if row["case"] in CASES:
                    row["label"] = CASES[row["case"]]
                    writer.writerow(row)


if __name__ == "__main__":
    extract(
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else generated_outputs_root() / "Figure_Source_Exports/S03/sweeps"
    )
