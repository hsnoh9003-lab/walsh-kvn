"""Search Litinski factory designs for the fixed nine-qubit workload.

Inputs
------
The factory catalogue, calibration data, circuit counts and requested physical-error probabilities.

Outputs
-------
Generated_Outputs/Regenerated/Litinski/ design tables and run metadata.

Run from the repository root
----------------------------
python -m Source.RE.generate_litinski_data --p 0.0005 --schemes A

Scope
-----
The objective is minimum physical-qubit count with runtime/error tie-breakers. Partial runs preserve other stored sweep conditions.
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
from Source.data_locations import published_path, source_label
import os

for k in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[k] = "1"
from pathlib import Path
import json, time, math, argparse, hashlib, tempfile
import numpy as np
import pandas as pd
from Source.RE.sweep_output import staged_sweep

ROOT = Path(__file__).resolve().parents[1]
from Source.RE import Litinski, model
from Source.RE.core_timing import core_interval_seconds


def calculate(args, out):
    """Search core distance and synthesis precision with the specified factory catalogue and workload."""
    start = time.time()
    allrows = []
    logs = []
    cat = published_path(
        "Data/RE/Litinski/not_as_costly_official_factory_numerical.csv"
    )
    raw = Litinski.read_numerical(cat)
    by_p = {p: [r for r in raw if abs(float(r["p_phys"]) - p) < 1e-12] for p in args.p}
    # nts is the tested T-gate cost PER arbitrary rotation, not a time-step count.
    # More synthesis T gates reduce angle error but increase circuit fault exposure.
    nts = np.arange(45, 126, dtype=int)
    inventories = {
        "A": dict(
            native=args.native_a,
            rz=args.rz,
            step_native=args.native_a // 33,
            step_rz=args.step_rz,
        ),
        "B": dict(
            native=args.native_b,
            rz=args.rz,
            step_native=args.native_b // 33,
            step_rz=args.step_rz,
        ),
    }
    for scheme in args.schemes:
        inv = inventories[scheme]
        assert inv["native"] == 33 * inv["step_native"]

        def counts(nt):
            return dict(
                nT=inv["native"] + inv["rz"] * nt,
                nRZ=inv["rz"],
                Q_logic=9,
                E_synth=inv["rz"] * 2.0 ** (-nt / 3),
                N_T=nt,
                step_nT=inv["step_native"] + inv["step_rz"] * nt,
                init_nT=(inv["rz"] - 33 * inv["step_rz"]) * nt,
            )

        # Add native T gates to synthesized rotations. syn is the summed angle-error
        # allowance from the declared synthesis model, at each tested precision.
        T = inv["native"] + inv["rz"] * nts
        syn = inv["rz"] * np.exp2(-nts / 3)
        for p in args.p:
            config = model.ModelConfig(n_L=9, data_block="fast", p_phys=p)
            block_period = model.block_cycles_per_T(config.data_block)
            for case in ["game_silva_v2", "nac_silva_v2"]:
                best = None
                num = 0
                feas = 0
                for dc in range(13, 50, 2):
                    core_rounds = (
                        core_interval_seconds(
                            block_period, dc, config.t_round, config.t_react
                        )
                        / config.t_round
                    )
                    if case.startswith("game"):
                        factories = [
                            (f, None)
                            for d in range(11, dc + 1, 2)
                            for f in Litinski.factory_candidates_game(p, d, "fast")
                        ]
                    else:
                        factories = [
                            (Litinski.candidate_nac(row), row) for row in by_p[p]
                        ]
                    for fac, meta in factories:
                        if fac.output_code_distance > dc:
                            continue
                        # u counts parallel factory units. Test from one unit through enough
                        # average supply to match the core; fewer units allow the core to wait.
                        upper = max(
                            1, math.ceil(fac.rounds_per_output / core_rounds - 1e-12)
                        )
                        for u in range(1, upper + 1):
                            r = Litinski.evaluate_units(
                                case, p, 45, dc, "fast", fac, meta, counts(45), u
                            )
                            if r is None:
                                continue
                            num += 1
                            if r["q_total"] > 1e6:
                                continue
                            if (
                                best is not None
                                and r["q_total"] > best["q_total"] + 1e-8
                            ):
                                continue
                            # Fixed architecture faults are independent per-T hazards in these
                            # declared backends. Vectorization is numerically checked after selection.
                            c0 = counts(45)["nT"]
                            hc = (
                                -np.log1p(-r["E_core"]) / c0
                                if r["E_core"] < 1
                                else np.inf
                            )
                            hm = (
                                -np.log1p(-r["E_MSF"]) / c0
                                if r["E_MSF"] < 1
                                else np.inf
                            )
                            err = -np.expm1(-(hc + hm) * T) + syn
                            ix = np.flatnonzero(err <= 1e-4)
                            if not len(ix):
                                continue
                            j = int(ix[0])
                            nt = int(nts[j])
                            z = Litinski.evaluate_units(
                                case, p, nt, dc, "fast", fac, meta, counts(nt), u
                            )
                            if z is None or z["E_total"] > 1e-4 + 1e-14:
                                raise AssertionError(
                                    "Vectorized candidate differs from full evaluator"
                                )
                            assert abs(z["E_total"] - err[j]) < 1e-10
                            feas += 1
                            # This comparison minimizes physical qubits first; runtime and error
                            # break ties. It is not the unconstrained fastest-design objective.
                            if best is None or (
                                z["q_total"],
                                z["runtime_seconds"],
                                z["E_total"],
                            ) < (
                                best["q_total"],
                                best["runtime_seconds"],
                                best["E_total"],
                            ):
                                best = z
                logs.append(
                    dict(
                        scheme=scheme,
                        p_phys=p,
                        case=case,
                        architectures=num,
                        feasible=feas,
                    )
                )
                if best:
                    allrows.append(
                        {
                            **best,
                            "scheme": scheme,
                            "logical_qubits": 9,
                            "RZ": inv["rz"],
                            "native_T": inv["native"],
                            "Q_max": 1000000,
                            "physical_qubits": int(math.ceil(best["q_total"])),
                        }
                    )
                print(
                    scheme,
                    p,
                    case,
                    "Q",
                    round(best["q_total"]) if best else None,
                    "elapsed",
                    round(time.time() - start, 1),
                    flush=True,
                )
    results = pd.DataFrame(allrows)
    if results.empty:
        schema = published_path(
            "Data/Regenerated/Litinski/litinski_nine_qubit_comparison.csv"
        )
        results = (
            pd.read_csv(schema, nrows=0)
            if schema.is_file()
            else pd.DataFrame(
                columns=[
                    "p_phys",
                    "scheme",
                    "case",
                    "paper",
                    "protocol",
                    "physical_qubits",
                ]
            )
        )
    results.to_csv(out / "litinski_nine_qubit_comparison.csv", index=False)
    pd.DataFrame(logs).to_csv(out / "litinski_search_counts.csv", index=False)
    certpath = published_path("Data/Radial128/grid64/flow_report.json")
    certification = {
        "status": "circuit verification file not present; certify declared inventory separately"
    }
    if certpath.exists():
        verified = json.loads(certpath.read_text())
        assert verified["logical_qubits"] == 9
        assert verified["counts"]["RZ"] == args.rz
        assert verified["counts"]["native_T"] == args.native_a
        certification = {
            "source": source_label(certpath),
            "sha256": hashlib.sha256(certpath.read_bytes()).hexdigest(),
            "status": "Counts match separately verified9q33-step arbitrary-rotation circuit; synthesis remains modelconditional",
        }
    (out / "litinski_metadata.json").write_text(
        json.dumps(
            dict(
                workload=inventories,
                elapsed_seconds=time.time() - start,
                scope="same nine-qubit 33-step workload; common alpha1 Silva core; unchanged factory timing/error calibration; surface-patch-density padded componentwise path allocation; actual-distance full-span errors; catalogue p=9e-4 has 9 of13 rows",
                W_seconds=1e-6,
                reaction_seconds=10e-6,
                d_range=[13, 49],
                N_T_range=[45, 125],
                Q_max=1e6,
                physical_patch_model="2dplus1_sq",
                area_policy="Q_path=Q_surface(d_out)*(L_path_area+u*n_out*L_bridge_area)/(d_out+1); error_lengths_unchanged",
                active_volume_alpha=1.0,
                error_target=1e-4,
                counts_certification=certification,
            ),
            indent=2,
        )
    )


def execute(args):
    """Stage a partial rate/scheme sweep and merge it without deleting unrequested results."""
    destination = Path(
        getattr(args, "output", generated_outputs_root() / "Regenerated/Litinski")
    )
    args.p = sorted(set(args.p))
    args.schemes = list(dict.fromkeys(args.schemes))

    def synchronize_metadata(stage, final):
        metadata_name = "litinski_metadata.json"
        table_name = "litinski_nine_qubit_comparison.csv"
        if not (final / metadata_name).is_file() or not (final / table_name).is_file():
            return
        previous = json.loads((final / metadata_name).read_text(encoding="utf-8"))
        current = json.loads((stage / metadata_name).read_text(encoding="utf-8"))
        rows = pd.read_csv(final / table_name)
        retained = set(rows.loc[~rows.p_phys.round(12).isin(args.p), "scheme"])
        for scheme, inventory in previous["workload"].items():
            if scheme not in args.schemes:
                current["workload"][scheme] = inventory
            elif scheme in retained and current["workload"][scheme] != inventory:
                raise ValueError(
                    "Use --output for a different gate inventory or rerun all its rates."
                )
        (stage / metadata_name).write_text(
            json.dumps(current, indent=2), encoding="utf-8"
        )

    if getattr(args, "replace_all", False):
        # A publication-wide rebuild must not inherit unrelated cached scheme
        # rows from a previous partial research sweep in Generated_Outputs/.
        destination.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="litinski_public_", dir=destination
        ) as folder:
            stage = Path(folder)
            calculate(args, stage)
            for path in stage.iterdir():
                path.replace(destination / path.name)
    else:
        with staged_sweep(
            destination, args.p, args.schemes, synchronize_metadata
        ) as output:
            calculate(args, output)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", nargs="+", type=float, default=[5e-4])
    ap.add_argument("--schemes", nargs="+", choices=["A", "B"], default=["A"])
    ap.add_argument(
        "--output", type=Path, default=generated_outputs_root() / "Regenerated/Litinski"
    )
    ap.add_argument("--rz", type=int, default=4803)
    ap.add_argument("--step-rz", type=int, default=143)
    ap.add_argument("--native-a", type=int, default=7392)
    ap.add_argument("--native-b", type=int, default=12936)
    ap.add_argument(
        "--replace-all",
        action="store_true",
        help="Replace a complete publication sweep rather than merging cached schemes",
    )
    execute(ap.parse_args())
