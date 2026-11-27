"""Calculate source diagnostics for Extended Data Figure 1.

Inputs
------
Archived IBM measurement summaries, preparation records and reference diagnostics.

Outputs
-------
Generated_Outputs/Regenerated/ED1/Source_Data_Extended_Data_Figure_1.csv and IBM_Hardware_Reference_Diagnostics.csv.

Run from the repository root
----------------------------
python -m Figure_Plots.ED1.generate_data

Scope
-----
Runs offline, using acquired measurements. It does not submit jobs to a quantum provider.
"""

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
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from Source.data_locations import generated_outputs_root
from Source.data_locations import published_path

DATA = published_path("Hardware/IBM/data/ibm_mps_calibration_full_precision.csv").parent


def export_source_data(rows):
    # The continuum reference uses the prepared distribution and is independent of shots.
    """Combine acquired shot counts with the matched continuum probability.

    The continuum calculation starts from the same prepared distribution as
    the quantum circuit; measured counts and mitigation results are not refitted."""
    sys.path.insert(0, str(ROOT / "Hardware/IBM/code"))
    from analyze_mps_ibm import mps_state, target_masses
    from reduce_shots import continuum

    initial = {}
    publication = []
    shots = {}
    for row in rows:
        n, nt, k = (int(row[x]) for x in ("qubits", "nt", "k"))
        if n not in initial:
            initial[n] = abs(mps_state(np.sqrt(target_masses(n)), n)) ** 2
        published = dict(row)
        published["acquired_shots"] = row["shots"]
        published["P_continuum"] = continuum(initial[n], k * (49.8291 - 40) / nt, "P")
        publication.append(published)
        shots[(row["cells"], row["nt"], row["k"])] = row["shots"]
    columns = list(rows[0])
    columns.insert(columns.index("shots") + 1, "acquired_shots")
    columns.insert(columns.index("r_continuum"), "P_continuum")
    destination = generated_outputs_root() / "Regenerated/ED1"
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "Source_Data_Extended_Data_Figure_1.csv").open(
        "w", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(publication)
    with (DATA / "ibm_mps_reference_diagnostics.csv").open() as file:
        diagnostics = list(csv.DictReader(file))
    columns = list(diagnostics[0])
    columns.insert(columns.index("k") + 1, "shots")
    for row in diagnostics:
        row["shots"] = shots[(row["cells"], row["nt"], row["k"])]
    with (destination / "IBM_Hardware_Reference_Diagnostics.csv").open(
        "w", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(diagnostics)


def main():
    """Export matched-continuum diagnostics from the archived IBM analysis rows."""
    with (DATA / "ibm_mps_calibration_full_precision.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    export_source_data(rows)
    print(
        "Wrote Extended Data Figure 1 diagnostics to Generated_Outputs/Regenerated/ED1"
    )


if __name__ == "__main__":
    main()
