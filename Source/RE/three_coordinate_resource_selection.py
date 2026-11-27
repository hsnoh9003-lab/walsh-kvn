"""Compile and reoptimize the tested three-coordinate time-step candidates.

Inputs
------
Time-convergence candidates, fixed baseline workloads, signed prepared states and hardware calibration.

Outputs
-------
Generated_Outputs/Three_Coordinate_Resource_Selection/, or --output.

Run from the repository root
----------------------------
python -m Source.RE.three_coordinate_resource_selection --workers 2

Scope
-----
The analysis compares feasible tested candidates under the production resource model; it is not a continuous optimization over time step.
"""

# Direct-file execution must locate Codes/ independently of the working directory.
if __package__ in (None, ""):
    import sys as _path_sys
    from pathlib import Path as _Path
    _codes = next(p for p in _Path(__file__).resolve().parents
                  if (p / "Source/data_locations.py").is_file())
    _path_sys.path.insert(0, str(_codes))


import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import os
from pathlib import Path
import sys
import time

from Source.data_locations import generated_outputs_root, source_data_root, set_source_data_root

ROOT = Path(__file__).resolve().parents[2]
DATA = None
OUTPUT = None
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
sys.path.insert(0, str(ROOT))

from Source.RE import preparation as S, optimization as O
from Source.RE.uncapped import select_fastest_with_tolerance


def load_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def prepare():
    """Build native gate inventories only for tested counts passing the stored field and shot criteria."""
    from Source.KvN.ensemble import count_native
    from Source.ROM.model import load

    model = load("candidate")
    base = DATA / "Additional_Source_Data/Three_Coordinate_Time_Convergence/baseline"
    current = DATA / "Shared_source_data/MPS_KvN_inputs/KvN/inputs"
    candidates = S.read_csv(
        DATA
        / "Additional_Source_Data/Three_Coordinate_Time_Convergence/time_convergence.csv"
    )
    workloads = {}
    for row in candidates:
        # Retain only counts below the sampled-time bias threshold and with a
        # positive protected final-time sampling budget. Recompute counts for each;
        # do not rescale the old runtime by the ratio of step counts.
        if row["below_target_at_all_67_times"] != "True" or not row["accepted_shots"]:
            continue
        run, steps = row["run"], int(row["steps"])
        config = load_json(base / run / "protocol.json")
        prep = load_json(current / run / "compiled_inventory.json")
        published = config["steps"]
        config["steps"] = steps
        inv = count_native(model, config, prep)
        if steps == published:
            reference = load_json(base / run / "expected_gate_inventory.json")
            for key in (
                "logical_qubits",
                "position_qubits",
                "arbitrary_Rz",
                "native_T",
                "Clifford",
                "by_gate",
            ):
                assert inv[key] == reference[key], (run, key, inv[key], reference[key])
        name = f"{run}_Nt{steps}"
        write_json(OUTPUT / "inventories" / (name + ".json"), inv)
        workloads[name] = dict(
            logical_qubits=inv["logical_qubits"],
            is_wake=True,
            RZ=inv["arbitrary_Rz"],
            direct_T=inv["native_T"],
            Clifford=inv["Clifford"],
            shots=int(row["accepted_shots"]),
            success=1.0,
            canonicalization_error_bound=0.0,
            precision_min=45,
            precision_max=200,
            shot_basis="accepted samples",
            workload_scope="same ensemble field-error task",
            run=run,
            steps=steps,
        )
    write_json(OUTPUT / "workloads.json", workloads)
    print(
        "Compiled",
        len(workloads),
        "feasible workloads; original inventories reproduced.",
        flush=True,
    )
    return workloads


def evaluate(name, w, rate, data_root, output_root):
    """Optimize one workload and physical error rate, retaining capped and uncapped designs."""
    DATA = Path(data_root)
    OUTPUT = Path(output_root)
    tick = time.monotonic()
    # Calibration-table indices encode p_phys = rate * 10**-4.
    table = DATA / f"Shared_source_data/RE_common/RE/prep_p{rate}.csv"
    minimum, fastest, capped, diagnostics, certificate = O.select_designs(
        name, rate / 10000.0, table, w, uncapped=True
    )
    capfast = select_fastest_with_tolerance(capped)
    selected = [minimum, fastest, dict(capfast, selection="fastest_under_cap")]
    for row in selected + capped:
        row.update(
            run=w["run"],
            steps=w["steps"],
            Clifford=w["Clifford"],
            seconds_per_accepted_shot=row["seconds_per_attempt"] / w["success"],
            complete_task_hours=row["complete_task_seconds"] / 3600.0,
            complete_task_days=row["complete_task_seconds"] / 86400.0,
            complete_task_years=row["complete_task_seconds"] / (86400.0 * 365.25),
            shot_basis=w["shot_basis"],
            workload_scope=w["workload_scope"],
        )
    out = OUTPUT / f"{name}_p{rate}"
    out.mkdir(exist_ok=True)
    S.write_csv(out / "selected.csv", selected)
    S.write_csv(out / "capped_selected.csv", capped)
    write_json(
        out / "audit.json",
        dict(
            statistics=diagnostics,
            uncapped_certificate=certificate,
            elapsed_seconds=time.monotonic() - tick,
        ),
    )
    print(
        json.dumps(
            dict(
                name=name,
                rate=rate,
                hours=fastest["complete_task_hours"],
                qubits=fastest["physical_qubits"],
                capped_hours=selected[-1]["complete_task_hours"],
            )
        ),
        flush=True,
    )
    return selected


def collect_results(output):
    """Export the dataset tables from completed per-candidate searches."""
    import shutil

    selected, capped = [], []
    audit_dir = output / "audits"
    audit_dir.mkdir(exist_ok=True)
    for path in sorted(output.glob("*_p*/selected.csv")):
        selected.extend(S.read_csv(path))
        capped.extend(S.read_csv(path.parent / "capped_selected.csv"))
        shutil.copyfile(
            path.parent / "audit.json", audit_dir / (path.parent.name + ".json")
        )
    ordering = lambda row: (
        float(row["p_phys"]),
        row["run"],
        int(row["steps"]),
        row["selection"],
        row["block"],
    )
    selected.sort(key=ordering)
    capped.sort(key=ordering)
    S.write_csv(output / "candidate_selected.csv", selected)
    S.write_csv(output / "candidate_capped_selected.csv", capped)
    winners = []
    groups = sorted({(float(row["p_phys"]), row["run"]) for row in selected})
    for rate, run in groups:
        eligible = [
            row
            for row in selected
            if float(row["p_phys"]) == rate
            and row["run"] == run
            and row["selection"] == "fastest_uncapped"
        ]
        winners.append(
            min(eligible, key=lambda row: float(row["complete_task_seconds"]))
        )
    S.write_csv(output / "fastest_tested_by_rate.csv", winners)


def main(argv=None):
    """Prepare candidate workloads, run independent hardware searches and collect the winners."""
    global DATA, OUTPUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--rates", nargs="+", type=int, default=list(range(2, 10)))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--reuse-inventories", action="store_true")
    parser.add_argument("--source-data", type=Path)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    DATA = (
        set_source_data_root(args.source_data)
        if args.source_data
        else source_data_root()
    )
    os.environ["KVN_SOURCE_DATA"] = str(DATA)
    OUTPUT = (args.output or generated_outputs_root() / 'Three_Coordinate_Resource_Selection').expanduser().resolve()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    workloads = (
        load_json(OUTPUT / "workloads.json") if args.reuse_inventories else prepare()
    )
    if args.models:
        workloads = {name: workloads[name] for name in args.models}
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(evaluate, name, w, rate, str(DATA), str(OUTPUT))
            for rate in args.rates
            for name, w in workloads.items()
        ]
        for future in as_completed(futures):
            rows.extend(future.result())
    rows.sort(key=lambda r: (r["p_phys"], r["run"], r["steps"], r["selection"]))
    collect_results(OUTPUT)


if __name__ == "__main__":
    main()
