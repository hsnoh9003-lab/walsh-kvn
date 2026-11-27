"""Refresh the Litinski design curves used by Supplementary Figure S8b.

Inputs
------
Completed Generated_Outputs/Regenerated/Litinski results and the existing three-model plot table.

Outputs
-------
Updated Fig. S8b CSVs and matching checksum metadata in the writable dataset.

Run from the repository root
----------------------------
python -m Source.RE.update_litinski_s8b

Scope
-----
The Silva model rows and plot labels are retained; only the two Litinski model rows are updated.
"""

from __future__ import annotations

# Direct-file execution must locate Codes/ independently of the working directory.
if __package__ in (None, ""):
    import sys as _path_sys
    from pathlib import Path as _Path
    _codes = next(p for p in _Path(__file__).resolve().parents
                  if (p / "Source/data_locations.py").is_file())
    _path_sys.path.insert(0, str(_codes))

from Source.Data_Production.data_assets import fixed_asset

# Public Source Data mapping (single canonical file per published input).
import sys as _kvn_path_sys
from pathlib import Path as _kvn_path_Path

_kvn_path_sys.path.insert(0, str(_kvn_path_Path(__file__).resolve().parents[1]))
from Source.data_locations import generated_outputs_root
from Source.data_locations import published_path
from Source.data_locations import source_data_root

import json
from pathlib import Path

from Source.RE.preparation import read_csv, write_csv
from Source.RE.sweep_output import publish_dataset_csvs


ROOT = Path(__file__).resolve().parents[1]
DATA = published_path("Supply/S08/data/plot_values/new_backend.csv")
LITINSKI = (
    generated_outputs_root() / "Regenerated/Litinski/litinski_nine_qubit_comparison.csv"
)
LABELS = {"game": "A Game of Surface Codes", "not_as_costly": "Not as Costly"}
PATH_COLUMNS = {
    "q_connection": "q_path",
    "Q_connection_horizontal": "Q_path_horizontal",
    "Q_connection_bridges": "Q_path_bridges",
}


def path_column_names(row):
    """Use the same Q_path field names for retained Silva and updated Litinski rows."""
    return {PATH_COLUMNS.get(name, name): value for name, value in row.items()}


def key(row):
    """Index the selected factory family and physical error rate."""
    return row["paper"], round(float(row["p_phys"]) * 10000)


def published_wake_designs(rows):
    """Select one scheme A design for each factory and physical error in Fig. S8b."""
    expected = {(paper, p / 10000) for paper in LABELS for p in range(2, 10)}
    matched = {}
    for row in rows:
        if (row.get("scheme") or "A") != "A":
            continue
        condition = row["paper"], float(row["p_phys"])
        if condition not in expected:
            continue
        lookup = key(row)
        if lookup in matched:
            raise ValueError(f"Multiple scheme A designs for {condition}")
        matched[lookup] = row
    if len(matched) != len(expected):
        raise ValueError(
            "Fig. S8b requires scheme A results for both factories at all eight physical errors"
        )
    return matched


# Update the 16 Litinski rows using scheme A results for both factory models
# at all eight physical-error values.
# Retain the existing Silva rows and write the updated Figure S8b input table.
def main():
    """Merge a complete Litinski sweep with the retained baseline for Fig. S8b."""
    if not LITINSKI.is_file():
        raise FileNotFoundError(f"Regenerate Litinski data first: {LITINSKI}")
    matched = published_wake_designs(read_csv(LITINSKI))
    if DATA.is_file():
        original = [path_column_names(row) for row in read_csv(DATA)]
    else:
        # Cold start: build the Fig. S8b comparison directly from resource
        # searches; no previously published Fig. S8b CSV is needed.
        direct = [
            r
            for r in read_csv(published_path("Data/RE/reference_capped_selected.csv"))
            if r["model"] == "direct"
            and r["block"] == "fast"
            and r["selection"] == "minimum_qubits"
        ]
        if len(direct) != 8:
            raise ValueError(
                "The reference resource sweep must supply eight direct designs"
            )
        original = []
        for rate in range(2, 10):
            for paper in ("game", "not_as_costly"):
                original.append(dict(paper=paper, p_phys=str(rate / 10000)))
        original.extend(
            dict(row, paper="", scheme="A", comparison_label="Original 33-step data")
            for row in sorted(direct, key=lambda r: float(r["p_phys"]))
        )
    current = []
    updated = []
    preserved_silva = 0
    for row in original:
        if row["paper"] not in LABELS:
            current.append(row)
            preserved_silva += 1
            continue
        new = matched[key(row)]
        assert float(new["E_total"]) <= 1e-4 + 1e-12
        current.append(
            {**new, "comparison_label": LABELS[new["paper"]], "block": "fast"}
        )
        prior_qubits = row.get("physical_qubits")
        updated.append(
            {
                "paper": row["paper"],
                "p_phys": row["p_phys"],
                "old_physical_qubits": (
                    int(float(prior_qubits)) if prior_qubits else None
                ),
                "new_physical_qubits": int(float(new["physical_qubits"])),
            }
        )
    assert len(updated) == 16 and preserved_silva == 8
    # The source families have complementary fields; preserve the canonical
    # union schema from the export protocol without requiring a result CSV.
    columns = json.loads(
        fixed_asset("catalogue", "fig_s8b_schema.json").read_text()
    )
    import csv
    from tempfile import TemporaryDirectory

    # Publish the data and its manifest record together. The existing helper
    # provides a writer lock and rollback if the replacement fails.
    with TemporaryDirectory(prefix="s8b-stage-") as folder:
        staged = Path(folder) / DATA.name
        with staged.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(current)
        publish_dataset_csvs(source_data_root(), {DATA: staged})
    report = {
        "status": "PASS",
        "figure": "Supplementary Figure S8b",
        "updated_Litinski_rows": len(updated),
        "preserved_Silva_rows": preserved_silva,
        "changed_qubit_points": [
            row
            for row in updated
            if row["old_physical_qubits"] is not None
            and row["old_physical_qubits"] != row["new_physical_qubits"]
        ],
    }
    (LITINSKI.parent / "s8b_application_audit.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
