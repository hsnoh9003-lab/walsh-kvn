"""Run resource-estimation searches for compiled circuit workloads.

The inputs are circuit inventories and measured magic-state preparation
tables. The search produces minimum-qubit designs, capped per-block designs,
and, by default, fastest designs without a physical-qubit cap.

Run from the Codes directory:
    python -m Source.RE.run --output ../Source_Data/Generated_Outputs/Full_RE

Complete architecture enumeration is performed only when explicitly invoked.
"""

from __future__ import annotations

# Direct-file execution must locate Codes/ independently of the working directory.
if __package__ in (None, ""):
    import sys as _path_sys
    from pathlib import Path as _Path
    _codes = next(p for p in _Path(__file__).resolve().parents
                  if (p / "Source/data_locations.py").is_file())
    _path_sys.path.insert(0, str(_codes))

import argparse, json
from pathlib import Path
from Source.data_locations import shared_data
from .preparation import load_workloads, write_csv
from .optimization import select_designs


# Each workload specifies the logical width, gate inventory, required number
# of accepted samples, and circuit acceptance probability.
# prep_by_rate maps each physical-error index to its preparation input table.
#
# Architecture and synthesis-cost selection are delegated to select_designs().
# q_max limits the explicitly capped results. When uncapped=True, the fastest
# design is subsequently searched for without a physical-qubit cap.
def sweep(
    workloads,
    prep_by_rate,
    out,
    *,
    rates=range(2, 10),
    blocks=("compact", "intermediate", "fast"),
    q_max=1_000_000,
    nt_min=45,
    nt_max=200,
    d_min=13,
    d_max=49,
    uncapped=True,
    canonical_export=False,
):
    """Recompute architectures for plain-dictionary workloads and preparation paths.

    Write both the global optima and capped per-block resource records.
    """
    rates = tuple(rates)
    if not workloads or not rates or not blocks:
        raise ValueError(
            "Provide at least one workload, physical-error rate and data block"
        )
    if q_max <= 0 or not 1 <= nt_min <= nt_max:
        raise ValueError("q_max must be positive and 1 <= nt_min <= nt_max")
    if d_min < 3 or d_max < d_min or d_min % 2 != 1 or d_max % 2 != 1:
        raise ValueError("Distances must be odd with 3 <= d_min <= d_max")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    selected = []
    capped = []
    audit = {}
    # The rate index is expressed in units of 1e-4.
    # For example, index 5 selects p_phys=5e-4 and the corresponding prep_p5.csv.
    for rate_index in rates:
        p_phys = rate_index / 10000
        prep_table = Path(prep_by_rate[rate_index])
        for name, workload in workloads.items():
            minimum, fastest, capped_rows, _, certificate = select_designs(
                name,
                p_phys,
                prep_table,
                workload,
                blocks=blocks,
                n_T_min=nt_min,
                n_T_max=nt_max,
                d_min=d_min,
                d_max=d_max,
                Q_cap=q_max,
                uncapped=uncapped,
                output_dir=out / f"{name}_p{rate_index}",
            )
            capped.extend(capped_rows)
            if certificate is not None:
                audit[f"{name}_p{rate_index}"] = certificate
            selected.extend((minimum, fastest))
    # The published schema reports attempt time as well as accepted-shot time.
    # Fill the latter from the computed heralding/acceptance probability.
    for row in [*selected, *capped]:
        if row.get("seconds_per_accepted_shot") in (None, ""):
            row["seconds_per_accepted_shot"] = float(
                row["seconds_per_attempt"]
            ) / float(row["success_probability"])
        if canonical_export:
            workload = workloads[row["model"]]
            row["shot_basis"] = workload["shot_basis"]
            row["workload_scope"] = workload["workload_scope"]
            row["Clifford"] = workload["Clifford"]
            row["run"] = workload.get("run", "")
            row["steps"] = workload.get("steps", "")
            row["seconds_per_accepted_shot"] = float(
                row["seconds_per_attempt"]
            ) / float(row["success_probability"])
            for name, seconds in [
                ("hours", 3600),
                ("days", 86400),
                ("years", 86400 * 365.25),
            ]:
                row["complete_task_" + name] = (
                    float(row["complete_task_seconds"]) / seconds
                )
    if canonical_export:
        model_order = {name: i for i, name in enumerate(workloads)}
        sort_key = lambda row: (
            model_order[row["model"]],
            float(row["p_phys"]),
            ("compact", "intermediate", "fast").index(row["block"]),
            row["selection"] != "minimum_qubits",
        )
        selected.sort(key=sort_key)
        capped.sort(key=sort_key)
        from Source.Data_Production.data_assets import fixed_asset

        schema = json.loads(fixed_asset("catalogue", "re_selected_schema.json").read_text())
    # selected.csv contains the designs selected across data-block families.
    # capped_selected.csv contains the per-block selections that satisfy q_max.
    # audit.json records the declared search ranges and uncapped-search bounds.
    # The minimum-qubit search and uncapped fastest search have distinct scopes.
    write_csv(
        out / "selected.csv", selected, columns=schema if canonical_export else None
    )
    write_csv(
        out / "capped_selected.csv",
        capped,
        columns=(
            json.loads(fixed_asset("catalogue", "re_capped_schema.json").read_text())
            if canonical_export
            else (list(selected[0]) if selected and not capped else None)
        ),
    )
    (out / "audit.json").write_text(
        json.dumps(
            dict(
                uncapped_fastest_audits=audit,
                data_block_rounding="ceil",
                data_blocks=list(blocks),
                distance_search=[d_min, d_max, 2],
                synthesis_precision=[nt_min, nt_max],
                distillation_levels_max=2,
                capped_design_qubit_limit=q_max,
                final_fastest_qubit_cap=None if uncapped else q_max,
            ),
            indent=2,
        )
        + "\n"
    )
    return selected


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--workloads",
        type=Path,
        help="Workload JSON; defaults to the publication workload when omitted",
    )
    ap.add_argument(
        "--prep-dir",
        type=Path,
        help="Directory of prep_pN.csv inputs; defaults to publication inputs",
    )
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--p", nargs="+", type=int, default=list(range(2, 10)))
    ap.add_argument("--models", nargs="+")
    ap.add_argument(
        "--blocks",
        nargs="+",
        choices=["compact", "intermediate", "fast"],
        default=["compact", "intermediate", "fast"],
    )
    ap.add_argument("--capped-only", action="store_true")
    ap.add_argument(
        "--q-max",
        type=int,
        default=1_000_000,
        help="Physical-qubit cap for capped selections",
    )
    ap.add_argument(
        "--d-min", type=int, default=13, help="Minimum enumerated odd code distance"
    )
    ap.add_argument(
        "--d-max",
        type=int,
        default=49,
        help="Maximum enumerated odd core distance; uncapped search may extend it",
    )
    ap.add_argument(
        "--nt-min", type=int, default=45, help="Minimum T gates per synthesized RZ"
    )
    ap.add_argument(
        "--nt-max",
        type=int,
        default=200,
        help="Maximum T gates per RZ in the initial enumeration",
    )
    ap.add_argument(
        "--canonical-export",
        action="store_true",
        help="Use the complete canonical full-sweep Source Data schema",
    )
    args = ap.parse_args()
    if args.workloads is None:
        args.workloads = shared_data("RE_common", "RE", "workloads.json")
    if args.prep_dir is None:
        args.prep_dir = shared_data("RE_common", "RE")
    workloads = load_workloads(args.workloads)
    if args.models:
        unknown = set(args.models) - workloads.keys()
        if unknown:
            ap.error("Unknown workload names: " + ", ".join(sorted(unknown)))
        workloads = {n: workloads[n] for n in args.models}
    if any(p not in range(2, 10) for p in args.p):
        ap.error("--p uses preparation-table indices 2 through 9 (index times 1e-4)")
    prep = {p: args.prep_dir / f"prep_p{p}.csv" for p in args.p}
    missing = [str(v) for v in prep.values() if not v.is_file()]
    if missing:
        raise FileNotFoundError(
            "Preparation input CSVs not provided: " + ", ".join(missing)
        )
    sweep(
        workloads,
        prep,
        args.output,
        rates=args.p,
        blocks=args.blocks,
        q_max=args.q_max,
        d_min=args.d_min,
        d_max=args.d_max,
        nt_min=args.nt_min,
        nt_max=args.nt_max,
        uncapped=not args.capped_only,
        canonical_export=args.canonical_export,
    )


if __name__ == "__main__":
    main()
