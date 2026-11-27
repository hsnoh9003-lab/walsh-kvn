"""Assemble publication tables from full-precision numerical records.

The functions join model settings, prepared inputs, circuit inventories and
selected resource designs using the archived labels. They also recompute joint
covariances and generator spectral norms from arrays. COLUMNS fixes the output
schema; readout shots are not inferred by rounding a printed table. Generated
CSV files go to the supplied output directory.
"""

# Direct-file execution must locate Codes/ independently of the working directory.
if __package__ in (None, ""):
    import sys as _path_sys
    from pathlib import Path as _Path
    _codes = next(p for p in _Path(__file__).resolve().parents
                  if (p / "Source/data_locations.py").is_file())
    _path_sys.path.insert(0, str(_codes))


import argparse
import csv
import json
import math
from pathlib import Path
from collections import defaultdict

import numpy as np
try:
    from Source.Data_Production.data_assets import fixed_asset
except ModuleNotFoundError:
    from data_assets import fixed_asset

HERE = Path(__file__).resolve().parent
COLUMNS = json.loads(fixed_asset("columns.json").read_text())


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write(path, records, keys=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = keys or COLUMNS[path.name]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in records:
            writer.writerow({k: row[k] for k in keys})


def case_settings(root, out):
    """Export solver settings and measured snapshot intervals from provenance records."""
    forecast = root / "Shared_source_data/CFD_ROM/Physical_Forecast_Audit"
    saved = load(forecast / "input_provenance.json")
    result = []
    with np.load(forecast / "audit_inputs.npz") as inputs:
        for row in saved["cases"]:
            sim = row["simulation"]
            amp, x, y, width = sim["perturbation"]
            dt = np.median(np.diff(inputs[row["case"] + "_times"]))
            result.append(
                dict(
                    case=row["case"],
                    solver_case=row["solver_case"],
                    streamfunction_amplitude=amp,
                    centre_x=x,
                    centre_y=y,
                    width=width,
                    dt=sim["dt"],
                    end=sim["end"],
                    snapshot_interval=round(float(dt), 10),
                    **{
                        k: sim[k]
                        for k in ("velocity_dofs", "pressure_dofs", "triangles")
                    },
                )
            )
    write(out / "case_settings.csv", result)


def resource_selections(root, out):
    """Join per-input gate inventories, shot budgets and selected hardware records."""
    cm = root / "Shared_source_data"
    panels = root / "Figures/Fig_5/plot_inputs/panels"
    mps = cm / "MPS_KvN_inputs"
    selected = rows(mps / "One_Clean_Three_Coordinate/resources/selected.csv")
    budget = rows(panels / "a_moments/relative_accuracy_budget.csv")
    all_moments = [
        dict(row, moment_relative_RMS_error=row["deterministic_relative_RMS"])
        for row in budget
    ]
    write(out / "selected_moment_errors.csv", all_moments)
    interior = next(
        r for r in budget if r["model"] == "2D C3" and r["input"] == "interior"
    )
    # Per-input state preparation changes RZ and CX. Read each retained gate inventory.
    counts = load(fixed_asset("inputs", "publication", "coupled_compiled_counts.json"))
    benchmark = load(
        mps
        / "One_Clean_Three_Coordinate/polynomial_benchmarks/coupled32_inventory_and_replay.json"
    )
    matched_workloads = []
    for r in budget:
        if r["model"] != "2D C3":
            continue
        matched_workloads.append(
            dict(
                r,
                ic=r["input"],
                dimension=2,
                steps=benchmark["steps"],
                logical_qubits=benchmark["logical_qubits"],
                implementation_allowance=r["implementation_TV_allowance"],
                reference="characteristics of the actual MPS-prepared piecewise-uniform input",
                RMS_scope="Euclidean moment error divided by Euclidean reference moment norm",
                **counts[r["id"]],
            )
        )
    write(out / "workloads.csv", matched_workloads)
    for run, model in (
        ("long298", "long298"),
        ("late40", "restart40"),
        (None, "2d_interior"),
    ):
        result = []
        designs = [v for v in selected if v["model"] == model]
        for row in designs:
            row = row.copy()
            row["expected_attempts"] = float(row["successful_shots"]) / float(
                row["success_probability"]
            )
            if run:
                inputs = mps / "KvN/inputs" / run
                b = load(inputs / "expected_field_budget.json")
                prep = load(inputs / "compiled_inventory.json")
                row.update(
                    run=model,
                    E_total_including_angle_omission=float(row["E_total"])
                    + float(row["source_angle_error_bound"]),
                    input_preparation="MPS",
                    MPS_bond_dimension=prep["bond_dimension"],
                    input_protocol="Independent Cartesian Gaussian centred on the ODE characteristic",
                    **{
                        k: b[k]
                        for k in (
                            "target_relative_field_rms",
                            "noiseless_deterministic_relative_field_bias",
                            "expected_relative_field_rms_bound",
                        )
                    },
                )
            else:
                row.update(
                    {
                        k: interior[k]
                        for k in (
                            "target_expected_relative_RMS",
                            "model_based_expected_relative_RMS_bound",
                            "cells_per_coordinate",
                        )
                    }
                )
            result.append(row)
        destination = (
            out / model / "selected_wake_resources.csv"
            if run
            else out / "selected_resources.csv"
        )
        write(destination, result)

    ref = rows(cm / "RE_common/RE/reference_selected.csv")
    archive = [
        r
        for name in ("direct", "nj_lchs", "long298")
        for r in ref
        if r["model"] == name and float(r["p_phys"]) == 5e-4
    ]
    write(out / "baseline_resources.csv", archive)
    cap = rows(cm / "Radial_refinement/Radial128/resources/capped_selected.csv")
    for block in ("compact", "intermediate", "fast"):
        write(
            out / f"radial128_p5_{block}_selected.csv",
            [r for r in cap if r["block"] == block and float(r["p_phys"]) == 5e-4],
        )
    sensitivity = rows(
        root / "Figures/Fig_S9/plot_inputs/plot_values/native_T_sensitivity.csv"
    )
    design_points = [
        (0.0003, 7392),
        (0.0003, 3210771),
        (0.0004, 510000),
        (0.0005, 200000),
        (0.0006, 1140000),
    ]
    by_case = {(float(r["p_phys"]), int(r["native_T"])): r for r in sensitivity}
    write(out / "Table_S8_current.csv", [by_case[k] for k in design_points])

    # Source table for SI S9 uses different time units for the block-encoded task.
    be = rows(cm / "RE_common/Regenerated/RE_BE_one_clean/selected.csv")
    result = []
    for label, model, selection in (
        ("Walsh–KvN A field", "direct", ref),
        ("BE–KvN field", "generator", be),
    ):
        for r in selection:
            if float(r["p_phys"]) != 5e-4 or (model and r["model"] != model):
                continue
            smallest = r["selection"] == "minimum_qubits"
            short, unit = (
                (("d", "yr") if smallest else ("h", "d"))
                if model == "generator"
                else ("s", "h")
            )
            sec_per_unit = {"s": 1, "h": 3600, "d": 86400, "yr": 365.25 * 86400}
            result.append(
                dict(
                    workload=label,
                    selection=("Minimum-qubit, " if smallest else "Fastest, ")
                    + r["block"],
                    physical_qubits=r["physical_qubits"],
                    accepted_shots=r["successful_shots"],
                    complete_task_time=f'{float(r["complete_task_seconds"])/sec_per_unit[unit]:.9f}',
                    time_unit=unit,
                    physical_error_probability=r["p_phys"],
                    mean_time_per_accepted_shot=f'{(float(r["seconds_per_attempt"])/float(r["success_probability"]))/sec_per_unit[short]:.9f}',
                    per_shot_time_unit=short,
                )
            )
    # Scheme B's fastest design is recorded in the retained numerical input.
    # For its minimum-footprint design the source uses the same geometry and N_T
    # as A, with the B native inventory. Recover that time from the current A service period.
    cache = load(cm / "RE_common/Numerical_Details/input/matched_field_designs.json")
    stored = [dict(zip(cache[0], v)) for v in cache[1:] if v[0] == "Walsh–KvN B field"]
    geom = next(
        r
        for r in ref
        if r["model"] == "direct"
        and r["selection"] == "minimum_qubits"
        and float(r["p_phys"]) == 0.0005
    )
    counts = next(
        r
        for r in rows(root / "Tables/Table_S2/numerical_source/wake_gate_inventory.csv")
        if r["address_shift"] == "B"
    )
    entry = next(r for r in stored if r["selection"].startswith("Minimum"))
    if int(entry["physical_qubits"]) != int(geom["physical_qubits"]):
        raise ValueError("Scheme B and A minimum-footprint geometries disagree")
    per = (
        int(counts["native_T_including_T_dagger"])
        + int(geom["N_T"]) * int(counts["arbitrary_RZ"])
    ) * float(geom["service_period_s"])
    entry.update(
        mean_time_per_accepted_shot=f"{per:.9f}",
        complete_task_time=f'{per*int(entry["accepted_shots"])/3600:.9f}',
    )
    result[2:2] = stored
    write(out / "matched_field_designs_current.csv", result)


def registers(root, out):
    """Export spatial registers, preparation costs and protected shot budgets for both wake intervals."""
    mps = root / "Shared_source_data/MPS_KvN_inputs"
    candidates = load(mps / "MPS/grid_selection.json")["candidates"]
    write(
        out / "grid_selection.csv",
        [{k: r.get(k, "") for k in COLUMNS["grid_selection.csv"]} for r in candidates],
    )
    summaries, shots = [], []
    for run in ("long298", "late40"):
        src = mps / "KvN/inputs" / run
        b = load(src / "expected_field_budget.json")
        inv = load(src / "expected_gate_inventory.json")
        prep = load(src / "compiled_inventory.json")
        protocol = load(src / "protocol.json")
        logical, position = inv["logical_qubits"], inv["position_qubits"]
        summaries.append(
            dict(
                run=("restart40" if run == "late40" else run),
                selected_grid=protocol["grid_r_theta_z"],
                position_qubits=position,
                logical_qubits=logical,
                transport_work_qubits=logical - position,
                preparation_work_qubits=prep["work_qubits"],
                bond_dimension=prep["bond_dimension"],
                preparation_fidelity=prep["fidelity"],
                preparation_RZ=prep["RZ"],
                preparation_native_T=prep["native_T"],
                whole_RZ=inv["arbitrary_Rz"],
                whole_native_T=inv["native_T"],
                shots=b["minimum_accepted_shots_under_rms_bound"],
                deterministic_relative_field_bias=b[
                    "noiseless_deterministic_relative_field_bias"
                ],
            )
        )
        shots.append(
            dict(
                run=run,
                eta=b["implementation_TV_allowance"],
                target_relative_RMS=b["target_relative_field_rms"],
                shots=b["minimum_accepted_shots_under_rms_bound"],
                bias=b["noiseless_deterministic_relative_field_bias"],
                bias_bound=b["worst_case_relative_field_bias_bound"],
                variance=b["noiseless_relative_field_variance_per_shot"],
                variance_bound=b["relative_field_variance_per_shot_bound"],
            )
        )
    write(out / "MPS_preparation_summary.csv", summaries)
    write(out / "three_coordinate_shot_replay.csv", shots)


def s7(root, out):
    """Collect the factory code distances and cultivation degrees actually selected by the searches."""
    mapping = {
        r["original_path"]: r["published_path"] for r in rows(root / "manifest.csv")
    }
    sources = [
        location + "/" + name + ".csv"
        for location, stem in (
            ("Data/RE", "reference_"),
            ("Data/Regenerated/RE_BE_one_clean", ""),
            ("Data/One_Clean_Three_Coordinate/resources", ""),
            ("Data/Radial128/resources", ""),
        )
        for name in (stem + "selected", stem + "capped_selected")
    ]
    sources.extend(
        [
            "Figures/Fig_S7/plot_inputs/baseline_policy_optima.csv",
            "Supply/S10/data/plot_values/capacity_selected_designs.csv",
            "Supply/S08/data/plot_values/implementation_budget_sensitivity.csv",
        ]
    )
    result = []
    for source in sources:
        groups = defaultdict(list)
        for r in rows(root / mapping.get(source, source)):
            dc = int(float(r["d_core"]))
            ds = [int(float(x)) for x in r["distances"].split(":")]
            groups["b"].append(dc)
            groups["c"].append(ds[0])
            groups["a"].extend(ds[1:])
        for panel, name, positions in (
            ("a", "higher-level distillation unit", 10),
            ("b", "core buffer", 6),
            ("c", "first-level distillation unit", 8),
        ):
            ds = groups[panel]
            if ds:
                patches = [math.ceil(10 / d - 1e-12) for d in ds]
                result.append(
                    dict(
                        panel=panel,
                        region=name,
                        source_table=source,
                        region_records=len(ds),
                        min_code_distance=min(ds),
                        max_code_distance=max(ds),
                        reaction_time_us=10.0,
                        round_time_us=1.0,
                        min_storage_patches_required=min(patches),
                        max_storage_patches_required=max(patches),
                        storage_positions_shown=positions,
                    )
                )
    write(out / "Source_Data_Figure_S7.csv", result)


def generator_norm_rows(root):
    """Diagonalize the normalized radial/phase generator blocks and compare their norm with block-encoding alpha."""
    case = root / "Shared_source_data/CFD_ROM/CFD_ROM/Re100/data"
    config = load(case / "configuration.json")["parameters"]
    with np.load(case / "direct_quantum_history.npz") as h:
        r = h["r"]
        faces = np.arange(1, len(r)) / len(r)
        rates = (faces - faces**3) * len(r) / 2
        skew = np.diag(rates, -1) - np.diag(rates, 1)
        eig = np.column_stack(
            [
                np.linalg.eigvalsh(
                    1j * skew
                    - np.diag(q * config["gamma_normalized"] / config["mu"] * r**2)
                )
                for q in (0, 1, 1, 2)
            ]
        )
        alpha = load(
            root / "Tables/Table_S11/BE_fullwindow_inventory/literal_joint_counts.json"
        )["normalization_alpha"]
        radial, joint_norm = abs(eig[:, 0]).max(), abs(eig).max()
        labels = (
            "radial_generator_spectral_norm",
            "joint_generator_spectral_norm",
            "block_encoding_alpha",
            "alpha_over_joint_norm",
        )
        records = [
            dict(quantity=k, value=v, definition="normalized time tau = mu(t* - 40)")
            for k, v in zip(labels, (radial, joint_norm, alpha, alpha / joint_norm))
        ]
    return eig, records


def ensure_generator_norm(root):
    """Recover the generator spectral norm from upstream numerical arrays."""
    target = root / "Shared_source_data/RE_common/Numerical_Details/generator_norm.csv"
    if not target.is_file():
        _, records = generator_norm_rows(root)
        write(target, records)
    return target


def diagnostics(root, out):
    """Recompute joint covariance and generator norms, then export the declared hardware settings."""
    case = root / "Shared_source_data/CFD_ROM/CFD_ROM/Re100/data"
    config = load(case / "configuration.json")["parameters"]
    with np.load(case / "joint_readout_inputs.npz") as joint:
        p, y = joint["direct_p"], joint["direct_values"]
        mean = p @ y
        # E[v_i*v_j]-E[v_i]E[v_j] retains correlations between observables
        # read from the same shot; this is the covariance exported in Data 6.
        cov = (y.T * p) @ y - np.outer(mean, mean)
        labels = ["ξ1", "ξ2", "ξ3"]
        write(
            out / "Supplementary_Data_6.csv",
            [
                {"Σ": label, **dict(zip(labels, (f"{a:.9f}" for a in row)))}
                for label, row in zip(labels, cov)
            ],
        )
    eig, norm_rows = generator_norm_rows(root)
    np.savetxt(
        out / "generator_eigenvalues.csv",
        eig,
        delimiter=",",
        header=",".join(COLUMNS["generator_eigenvalues.csv"]),
        comments="",
    )
    write(out / "generator_norm.csv", norm_rows)
    inp = load(
        root
        / "Shared_source_data/RE_common/Numerical_Details/input/hardware_and_search.json"
    )
    write(
        out / "hardware_and_search.csv",
        [dict(zip(inp[0], r, strict=True)) for r in inp[1:]],
    )
    subset = rows(root / "Figures/Fig_5/plot_inputs/wake/data/long298/trajectory.csv")
    keys = (
        "time",
        "reference_mean_x",
        "reference_mean_y",
        "reference_mean_z",
        "reference_mean_z_minus_r2",
    )
    write(out / "common_reference.csv", subset, keys)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for produce in (case_settings, registers, s7, diagnostics, resource_selections):
        produce(args.source_data, args.output)
