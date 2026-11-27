"""Assemble Table 1 from full-precision selected-resource records.

Inputs
------
The Table 1 specification and selected design/field records in Source_Data.

Outputs
-------
Generated_Outputs/Regenerated/Table_1.csv, or the path supplied with --output.

Run from the repository root
----------------------------
python -m Source.RE.export_table1 --check

Scope
-----
--check compares the exported table with the archived table; it does not optimize a new design.
"""

# Direct-file execution must locate Codes/ independently of the working directory.
if __package__ in (None, ""):
    import sys as _path_sys
    from pathlib import Path as _Path
    _codes = next(p for p in _Path(__file__).resolve().parents
                  if (p / "Source/data_locations.py").is_file())
    _path_sys.path.insert(0, str(_codes))


# Public Source Data mapping (single canonical file per published input).
import sys as _kvn_path_sys
from pathlib import Path as _kvn_path_Path

_kvn_path_sys.path.insert(0, str(_kvn_path_Path(__file__).resolve().parents[1]))
from Source.data_locations import generated_outputs_root
from Source.data_locations import published_path
from pathlib import Path
import csv, json, math, argparse

ROOT = Path(__file__).resolve().parents[1]


def rows(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build():
    """Join the table specification to one full-precision design per row."""
    spec = json.loads(
        (published_path("Data/Table_1_specification.json")).read_text(encoding="utf-8")
    )
    output = []
    for entry in spec:
        entry = dict(entry)
        if entry["source_model"] == "classical_eulerian":
            # This row is an asymptotic operation count, not a hardware design.
            assert entry["selection"] == "asymptotic_bound"
            assert not any(entry[key] for key in (
                "logical_qubits", "accepted_shots", "physical_qubits",
                "t_gates_per_attempt", "total_task_seconds",
            ))
            output.append(entry)
            continue
        record = entry["source_record"]
        selected = [
            r
            for r in rows(published_path(record))
            if r["model"] == entry["source_model"]
            and r["selection"] == entry["selection"]
            and math.isclose(
                float(r["p_phys"]), float(entry["physical_error_probability"])
            )
        ]
        assert len(selected) == 1, (record, entry["source_model"], len(selected))
        r = selected[0]
        ints = {
            "logical_qubits": "logical_qubits",
            "accepted_shots": "successful_shots",
            "physical_qubits": "physical_qubits",
            "t_gates_per_attempt": "T_count",
            "native_T": "native_T",
            "arbitrary_Rz": "RZ",
            "synthesis_T_per_Rz": "N_T",
        }
        for out, key in ints.items():
            entry[out] = str(int(float(r[key])))
        sec = float(r["complete_task_seconds"])
        for out, den in [
            ("total_task_seconds", 1),
            ("total_task_hours", 3600),
            ("total_task_days", 86400),
            ("total_task_years", 86400 * 365.25),
        ]:
            entry[out] = str(sec / den)
        output.append(entry)
    return output


def main():
    """Write Table 1 and optionally compare every field with its archived value."""
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output",
        type=Path,
        default=generated_outputs_root() / "Regenerated/Table_1.csv",
    )
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    out = build()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    if a.check:
        prior = rows(published_path("Data/Table_1.csv"))
        assert len(prior) == len(out)
        for x, y in zip(prior, out):
            for k in x:
                if x[k] == y[k]:
                    continue
                try:
                    assert math.isclose(
                        float(x[k]), float(y[k]), rel_tol=1e-12, abs_tol=1e-14
                    ), (x["calculation"], k, x[k], y[k])
                except ValueError:
                    raise AssertionError((x["calculation"], k, x[k], y[k]))
    print("Table 1 export: PASS", len(out), "rows")


if __name__ == "__main__":
    main()
