"""Refresh the Litinski factory curves in v100 Supplementary Figure S8b.

Preserve the three v100 plot labels, the Silva rows, and the figure numbering.
Only the two Litinski model rows at each physical error probability are updated.
"""
from __future__ import annotations

# Public Source Data mapping (single canonical file per published input).
import sys as _kvn_path_sys
from pathlib import Path as _kvn_path_Path
_kvn_path_sys.path.insert(0, str(_kvn_path_Path(__file__).resolve().parents[1]))
from Source.data_locations import published_path

import json
from pathlib import Path

from Source.RE.preparation import read_csv, write_csv


ROOT = Path(__file__).resolve().parents[1]
DATA = published_path('Supply/S08/data/plot_values/new_backend.csv')
LITINSKI = ROOT/'Build/Regenerated/Litinski/litinski_nine_qubit_comparison.csv'
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
        raise ValueError("Fig. S8b requires scheme A results for both factories at all eight physical errors")
    return matched


def main():
    original = [path_column_names(row) for row in read_csv(DATA)]
    if not LITINSKI.is_file():
        raise FileNotFoundError(f'Regenerate Litinski data first: {LITINSKI}')
    matched = published_wake_designs(read_csv(LITINSKI))
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
        current.append({**new, "comparison_label": LABELS[new["paper"]],
                        "block": "fast"})
        updated.append({"paper": row["paper"], "p_phys": row["p_phys"],
                        "old_physical_qubits": int(float(row["physical_qubits"])),
                        "new_physical_qubits": int(float(new["physical_qubits"]))})
    assert len(updated) == 16 and preserved_silva == 8
    write_csv(DATA, current)
    report = {"status": "PASS", "figure": "Supplementary Figure S8b",
              "updated_Litinski_rows": len(updated), "preserved_Silva_rows": preserved_silva,
              "changed_qubit_points": [row for row in updated if
                                      row["old_physical_qubits"] != row["new_physical_qubits"]]}
    (LITINSKI.parent/'v100_s8b_application_audit.json').write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
