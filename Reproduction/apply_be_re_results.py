"""Apply the verified one-clean BE–KvN RE sweep to v100's reference mirrors.

Only generator (BE–KvN) rows are replaced. Other workloads, manuscript figure
numbers, and Supplementary table identities remain untouched. Run after the
BE one-clean workload has been compiled and its RE sweep has finished.
"""
from __future__ import annotations

import json
from pathlib import Path

from Source.RE.preparation import read_csv, write_csv
from Source.data_locations import shared_data


ROOT = Path(__file__).resolve().parents[1]
BE_RESULTS = ROOT / "Build/RE_BE_one_clean"
REFERENCE = shared_data('RE_common','RE')


def design_key(row, capped=False):
    return (round(float(row["p_phys"]) * 1e4), row["selection"],
            row["block"] if capped else None)


def apply_one(reference_filename, new_filename, *, capped=False):
    old = read_csv(REFERENCE / reference_filename)
    fresh = read_csv(BE_RESULTS / new_filename)
    assert fresh and all(row["model"] == "generator" for row in fresh)
    by_key = {design_key(row, capped): row for row in fresh}
    assert len(by_key) == len(fresh), "Duplicate entries in new BE results"
    assert {round(float(row["p_phys"]) * 1e4) for row in fresh} == set(range(2, 10))
    used = set()
    merged = []
    for row in old:
        if row["model"] != "generator":
            merged.append(row)
            continue
        key = design_key(row, capped)
        if key not in by_key:
            raise AssertionError(f"Existing BE design has no updated counterpart: {key}")
        new = by_key[key]
        assert int(new["logical_qubits"]) == 25
        assert int(float(new["RZ"])) == 7677481
        assert int(new["native_T"]) == 511970388
        updated = dict(row, **new)
        if "seconds_per_accepted_shot" in updated:
            updated["seconds_per_accepted_shot"] = (
                float(updated["seconds_per_attempt"]) / float(updated["success_probability"]))
        seconds = float(updated["complete_task_seconds"])
        for field, scale in (("complete_task_hours", 3600),
                             ("complete_task_days", 86400),
                             ("complete_task_years", 365.25 * 86400)):
            if field in updated:
                updated[field] = seconds / scale
        used.add(key)
        merged.append(updated)
    assert used == set(by_key), ("BE design coverage mismatch", len(used), len(by_key))
    write_csv(REFERENCE / reference_filename, merged)
    return {"replaced_BE_rows": len(used),
            "unchanged_non_BE_rows": sum(row["model"] != "generator" for row in old)}


def main():
    workload = json.loads((REFERENCE / "workloads_v38.json").read_text(encoding="utf-8"))
    assert int(workload["generator"]["logical_qubits"]) == 25
    selected = apply_one("v38_reference_selected.csv", "selected.csv")
    capped = apply_one("v38_reference_capped_selected.csv", "capped_selected.csv", capped=True)
    record = {"status": "PASS", "source": str(BE_RESULTS.relative_to(ROOT)),
              "selected": selected, "capped": capped,
              "preserved_models": "Every reference row except model=generator"}
    (BE_RESULTS / "v100_application_audit.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
