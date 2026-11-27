"""Reoptimize resource designs using updated preparation measurements.

The supplied cultivation records are inputs; no new Monte Carlo simulation
is performed.

Command-line execution refreshes prep_p*.csv before running the search.
The --apply option controls only whether the recomputed design selections
are merged into the reference design CSVs.
"""

# Direct-file execution must locate Codes/ independently of the working directory.
if __package__ in (None, ""):
    import sys as _path_sys
    from pathlib import Path as _Path
    _codes = next(p for p in _Path(__file__).resolve().parents
                  if (p / "Source/data_locations.py").is_file())
    _path_sys.path.insert(0, str(_codes))


from Source.data_locations import generated_outputs_root

from pathlib import Path
import argparse, json, time
from .preparation import (
    load_workloads,
    make_prep_table,
    read_csv,
    write_csv,
    design_key,
)
from .optimization import select_designs
from Source.data_locations import shared_data
from Source.Data_Production.data_assets import fixed_asset

ROOT = Path(__file__).resolve().parents[2]
RAW_DIRECTORY = shared_data("Gidney_cultivation", "raw")
RE_DIRECTORY = shared_data("RE_common", "RE")


def refresh_preparation_inputs(rates):
    """Regenerate per-rate RE preparation tables from supplied cultivation records.

    The destination files are the preparation inputs used by subsequent
    resource-estimation searches.
    """
    for rate in sorted(set(rates)):
        p_phys = rate / 10000
        raw = RAW_DIRECTORY / f"custom_p{rate}e-4_d1_3_d2.csv"
        make_prep_table(raw, p_phys, RE_DIRECTORY / f"prep_p{rate}.csv")


def run(rates, models, output, apply=False, workload_overrides=None):
    workloads = load_workloads(RE_DIRECTORY / "workloads.json")
    if workload_overrides is not None:
        if apply:
            raise ValueError(
                "Historical workload replay cannot replace current selections"
            )
        overrides = load_workloads(workload_overrides)
        if not overrides.keys() <= workloads.keys():
            raise ValueError("Historical workload overrides contain unknown models")
        workloads.update(overrides)
    selected = []
    capped = []
    diagnostics = []
    certificates = {}
    start = time.time()
    for rate_index in rates:
        for name in models:
            prep_table = RE_DIRECTORY / f"prep_p{rate_index}.csv"
            minimum, fastest, capped_rows, statistics, certificate = select_designs(
                name,
                rate_index / 10000,
                prep_table,
                workloads[name],
            )
            capped.extend(capped_rows)
            diagnostics.extend(statistics)
            # Match the public full-precision schema, including derived
            # workload metadata and time units for every selected design.
            for row in [minimum, fastest, *capped_rows]:
                workload = workloads[name]
                row["shot_basis"] = workload["shot_basis"]
                row["workload_scope"] = workload["workload_scope"]
                row["Clifford"] = workload["Clifford"]
                if workload_overrides is None:
                    row["run"] = workload.get("run", "")
                    row["steps"] = workload.get("steps", "")
                row["seconds_per_accepted_shot"] = float(
                    row["seconds_per_attempt"]
                ) / float(row["success_probability"])
                for unit, scale in [
                    ("hours", 3600),
                    ("days", 86400),
                    ("years", 86400 * 365.25),
                ]:
                    row["complete_task_" + unit] = (
                        float(row["complete_task_seconds"]) / scale
                    )
            for row in statistics:
                print(
                    name,
                    rate_index,
                    row["block"],
                    "seconds",
                    round(row["seconds"], 2),
                    flush=True,
                )
            certificates[f"{name}_p{rate_index}"] = certificate
            selected.extend((minimum, fastest))
            write_csv(output / "updated_selected.csv", selected)
            write_csv(
                output / "updated_capped_selected.csv",
                capped,
                columns=list(minimum) if not capped else None,
            )
            (output / "search_audit.json").write_text(
                json.dumps(
                    dict(
                        elapsed_seconds=time.time() - start,
                        completed_model_rates=len(selected) // 2,
                        diagnostics=diagnostics,
                        uncapped_certificates=certificates,
                    ),
                    indent=2,
                )
                + "\n"
            )
            print(
                "SELECTED",
                name,
                rate_index,
                minimum["physical_qubits"],
                fastest["physical_qubits"],
                fastest["complete_task_seconds"] / 3600,
                flush=True,
            )
    # Match updates to reference rows using design_key().
    # Canonical publication order is model, physical error, block/selection.
    order = {name: index for index, name in enumerate(models)}
    sort_key = lambda row: (
        float(row["p_phys"]),
        order[row["model"]],
        ("compact", "intermediate", "fast").index(row["block"]),
        row["selection"] != "minimum_qubits",
    )
    selected.sort(key=sort_key)
    capped.sort(key=sort_key)
    schema = json.loads(
        fixed_asset("catalogue", "re_selected_schema.json").read_text()
    )
    capped_schema = json.loads(
        fixed_asset("catalogue", "re_capped_schema.json").read_text()
    )
    if workload_overrides is not None:
        schema = [name for name in schema if name not in ("run", "steps")]
        capped_schema = [name for name in capped_schema if name not in ("run", "steps")]
    write_csv(output / "updated_selected.csv", selected, columns=schema)
    write_csv(output / "updated_capped_selected.csv", capped, columns=capped_schema)
    # Preserve workloads and physical-error values outside the requested update.
    # Remove previously stored capped rows that are infeasible in the new search.
    # Refresh derived time fields when replacing a design.
    if apply:
        for filename, updates, is_capped in [
            ("reference_selected.csv", selected, False),
            ("reference_capped_selected.csv", capped, True),
        ]:
            old = read_csv(RE_DIRECTORY / filename)
            by_key = {design_key(row, is_capped): row for row in updates}
            count = 0
            merged = []
            for row in old:
                replacement = by_key.get(design_key(row, is_capped))
                if replacement:
                    row = {**row, **replacement}
                    count += 1
                    for k, scale in [
                        ("complete_task_hours", 3600),
                        ("complete_task_days", 86400),
                        ("complete_task_years", 365.25 * 86400),
                    ]:
                        if k in row:
                            row[k] = float(row["complete_task_seconds"]) / scale
                    if "seconds_per_accepted_shot" in row:
                        row["seconds_per_accepted_shot"] = float(
                            row["seconds_per_attempt"]
                        ) / float(row["success_probability"])
                elif (
                    is_capped
                    and row["model"] in models
                    and round(float(row["p_phys"]) * 10000) in rates
                ):
                    # This block has no feasible design at the requested total-area cap.
                    continue
                merged.append(row)
            assert count == len(updates), (filename, count, len(updates))
            write_csv(
                RE_DIRECTORY / filename,
                merged,
                columns=list(old[0]) if old and not merged else None,
            )
    return selected, capped


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rates", type=int, nargs="+", default=[2, 3, 4, 5, 6, 7, 8, 9])
    p.add_argument(
        "--models",
        nargs="+",
        default=list(load_workloads(RE_DIRECTORY / "workloads.json")),
    )
    p.add_argument(
        "--output", type=Path, default=generated_outputs_root() / "Gidney_Input_Update"
    )
    p.add_argument("--apply", action="store_true")
    p.add_argument(
        "--preserve-prep",
        action="store_true",
        help="Use the retained measured preparation tables without rewriting them",
    )
    p.add_argument(
        "--workload-overrides",
        type=Path,
        help="Replay historical workload records without changing current selections",
    )
    a = p.parse_args()
    a.output.mkdir(exist_ok=True, parents=True)
    if not a.preserve_prep:
        refresh_preparation_inputs(a.rates)
    run(a.rates, a.models, a.output, a.apply, a.workload_overrides)
