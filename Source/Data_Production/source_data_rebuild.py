"""Manage the mapping from numerical producers to publication data files.

Inputs
------
Producer definitions and inputs in Source_Data; transient output under the working Source_Data/Generated_Outputs/.

Outputs
-------
CSV files staged or copied into the explicitly selected writable Source_Data tree.

Run from the repository root
----------------------------
python -m Source.Data_Production.source_data_rebuild list

Scope
-----
bootstrap creates the catalogue; reproduce runs a producer; publish copies generated outputs; verify checks file sizes and SHA-256 digests against the manifest.
"""

from __future__ import annotations

# Direct-file execution must locate Codes/ independently of the working directory.
if __package__ in (None, ""):
    import sys as _path_sys
    from pathlib import Path as _Path
    _codes = next(p for p in _Path(__file__).resolve().parents
                  if (p / "Source/data_locations.py").is_file())
    _path_sys.path.insert(0, str(_codes))


import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


CODES = Path(__file__).resolve().parents[2]
from Source.Data_Production.data_assets import fixed_asset
from Source.data_locations import source_data_root


@dataclass(frozen=True)
class Route:
    """One producer output path, relative to Codes, and its dataset destination."""

    group: str
    produced_in_codes: str
    published_in_source_data: str


def all_routes() -> tuple[Route, ...]:
    """Published CSV destinations for every noncanonical writer identified.

    Source paths refer to the *actual* generated filename (which occasionally
    differs from the figure/table filename). Duplicate copies of an identical
    scientific result are represented by separate destination routes.
    """
    output = []

    def one(group, src, dst):
        output.append(Route(group, src, dst))

    def many(group, src_dir, dst_dir, filenames):
        for name in filenames:
            one(group, f"{src_dir}/{name}", f"{dst_dir}/{name}")

    ed = "Figures/Extended_Data_Fig_1"
    one(
        "ibm_figure",
        "Generated_Outputs/Regenerated/ED1/Source_Data_Extended_Data_Figure_1.csv",
        f"{ed}/Source_Data_Extended_Data_Figure_1.csv",
    )
    many(
        "ibm_measurements",
        "Generated_Outputs/IBM",
        f"{ed}/hardware_inputs",
        (
            "ibm_mps_calibration_full_precision.csv",
            "ibm_mps_gate_counts.csv",
            "ibm_mps_preparation_validation.csv",
            "ibm_mps_reference_diagnostics.csv",
        ),
    )

    gidney = "Shared_source_data/Gidney_cultivation"
    many(
        "gidney_update",
        "Generated_Outputs/Gidney_Input_Update",
        f"{gidney}/audit",
        ("updated_selected.csv", "updated_capped_selected.csv"),
    )
    for number in range(2, 10):
        name = f"custom_p{number}e-4_d1_3_d2.csv"
        one(
            "gidney_trials",
            f"Generated_Outputs/Gidney/tradeoff/{name}",
            f"{gidney}/raw/{name}",
        )

    re = "Shared_source_data/RE_common/Regenerated"
    many(
        "litinski",
        "Generated_Outputs/Regenerated/Litinski",
        f"{re}/Litinski",
        ("litinski_nine_qubit_comparison.csv", "litinski_search_counts.csv"),
    )
    many(
        "be_resources",
        "Generated_Outputs/RE_BE_one_clean",
        f"{re}/RE_BE_one_clean",
        ("selected.csv", "capped_selected.csv"),
    )

    f4 = "Figures/Fig_4"
    sample = "Generated_Outputs/Force_Sampling"
    one(
        "force_sampling",
        f"{sample}/force_815_shots.csv",
        f"{f4}/explicit_source_data/Source_Data_Figure_4f.csv",
    )
    one(
        "force_sampling",
        f"{sample}/force_815_shots.csv",
        f"{f4}/plot_inputs/force_output/force_815_shots.csv",
    )
    many(
        "force_sampling",
        sample,
        f"{f4}/force_sampling",
        (
            "force_815_shots.csv",
            "force_error_scales.csv",
            "sampled_outcome_counts.csv",
            "summary.csv",
        ),
    )

    fem = "Generated_Outputs/CFD/force_replay"
    one(
        "cfd_forces",
        f"{fem}/fig4f_fem_forces.csv",
        f"{f4}/plot_inputs/force_output/fig4f_fem_forces.csv",
    )
    one(
        "cfd_forces",
        f"{fem}/fig4f_fem_forces.csv",
        f"{f4}/plot_inputs/plot_values/Re100/fig4f_fem_forces.csv",
    )
    one(
        "cfd_forces",
        f"{fem}/forces.csv",
        "Shared_source_data/CFD_ROM/FEM/force_reference/forces.csv",
    )
    one(
        "cfd_forces",
        f"{fem}/forces_cfd.csv",
        "Shared_source_data/CFD_ROM/FEM/force_reference/forces_cfd.csv",
    )

    sindy = "Generated_Outputs/SINDy_Degree/additional_cases"
    sindy_shared = "Shared_source_data/CFD_ROM/SINDy_Degree/additional_cases"
    many(
        "sindy_cases",
        sindy,
        sindy_shared,
        (
            "case_errors.csv",
            "replayed_validation_folds.csv",
            "three_case_summary.csv",
            "all_case_summary.csv",
        ),
    )
    one(
        "sindy_cases",
        f"{sindy}/case_errors.csv",
        f"{f4}/plot_inputs/model_degree/additional_case_errors.csv",
    )
    one(
        "sindy_cases",
        f"{sindy}/all_case_summary.csv",
        f"{f4}/plot_inputs/model_degree/additional_case_summary.csv",
    )

    f5_wake = "Figures/Fig_5/plot_inputs/wake/data"
    mps_shared = "Shared_source_data/MPS_KvN_inputs"
    for run, figure_run in (("late40", "restart40"), ("long298", "long298")):
        ref = f"Generated_Outputs/KvN/ensemble/candidate_{run}_common/reference.csv"
        one(
            "mps_reference",
            ref,
            f"{mps_shared}/MPS/domain_selection/{run}/reference.csv",
        )
        one("mps_reference", ref, f"{f5_wake}/{figure_run}/reference.csv")
        propagated = f"Generated_Outputs/Three_Coordinate_MPS_Replay/{run}/propagation/trajectory.csv"
        one(
            "mps_trajectory",
            propagated,
            f"{mps_shared}/Three_Coordinate_MPS_Replay/{run}/propagation/trajectory.csv",
        )
        one("mps_trajectory", propagated, f"{f5_wake}/{figure_run}/trajectory.csv")

    radial = "Shared_source_data/Radial_refinement"
    one(
        "q5_validation",
        "Generated_Outputs/Q5_Upwind_Comparison/input_phase_comparison.csv",
        f"{radial}/Q5_Upwind_Comparison/input_phase_comparison.csv",
    )
    one(
        "q5_upwind",
        "Generated_Outputs/Q5_Upwind_Comparison/mps/q5_upwind_comparison.csv",
        f"{radial}/Q5_Upwind_Comparison/q5_upwind_comparison.csv",
    )
    one(
        "q5_upwind",
        "Generated_Outputs/Q5_Upwind_Comparison/positive_root/q5_upwind_comparison.csv",
        f"{radial}/Q5_Upwind_Comparison/positive_root_control/q5_upwind_comparison.csv",
    )

    coord = "One_Clean_Three_Coordinate/resources"
    names = ["selected.csv", "capped_selected.csv"] + [
        f"{prefix}_p{rate}.csv"
        for prefix in ("selected", "capped_selected")
        for rate in range(2, 10)
    ]
    many(
        "three_coordinate_resources",
        f"Generated_Outputs/{coord}",
        f"{mps_shared}/{coord}",
        names,
    )

    audit = "Generated_Outputs/Replayed_Physical_Audit"
    cfd_audit = "Shared_source_data/CFD_ROM/Physical_Forecast_Audit"
    many(
        "physical_audit",
        audit,
        cfd_audit,
        ("extended_window.csv", "physical_forecast_metrics.csv", "short_window.csv"),
    )
    one(
        "physical_audit",
        f"{audit}/short_window.csv",
        "Figures/Fig_S1/plot_inputs/plot_values/S1a_short_window_CFD_audit.csv",
    )
    one(
        "physical_audit",
        f"{audit}/extended_window.csv",
        "Figures/Fig_S1/plot_inputs/plot_values/S1b_extended_window_CFD_audit.csv",
    )

    many(
        "grid_oscillation",
        "Generated_Outputs/Grid_Oscillation_Diagnostics",
        f"{radial}/Grid_Oscillation_Diagnostics",
        (
            "q5_boundary_cell_masses.csv",
            "radial_cell_masses.csv",
            "radial_history_diagnostics.csv",
        ),
    )
    one(
        "radial128_flow",
        "Generated_Outputs/Radial_Refinement/grid_comparison.csv",
        f"{radial}/Radial128/grid_comparison.csv",
    )
    many(
        "radial128_resources",
        "Generated_Outputs/Radial_Refinement/resources",
        f"{radial}/Radial128/resources",
        ("capped_selected.csv", "selected.csv"),
    )

    one(
        "supplement_policy",
        "Generated_Outputs/Regenerated/Supplementary/baseline_policy_optima.csv",
        "Figures/Fig_S7/plot_inputs/baseline_policy_optima.csv",
    )
    one(
        "radial_step_compare",
        "Generated_Outputs/Step_Shot_Comparison/hardware_16_33.csv",
        "Additional_Source_Data/Source_data_S1_64cell_wake_step_count/hardware_16_33.csv",
    )
    one(
        "radial_step_diagnostic",
        "Generated_Outputs/KvN/radial_step_check/step_shot_tradeoff.csv",
        "Additional_Source_Data/Source_data_S1_64cell_wake_step_count/step_shot_tradeoff.csv",
    )
    one(
        "accuracy_target",
        "Generated_Outputs/Accuracy_Target_Sensitivity/fixed_design_targets.csv",
        "Shared_source_data/RE_common/Accuracy_Target_Sensitivity/fixed_design_targets.csv",
    )
    one(
        "field_sampling",
        "Generated_Outputs/Field_Sampling_Replay/Table_S13_sampling_verification.csv",
        "Supplementary_Data/Supplementary_Data_7.csv",
    )
    one(
        "normalization",
        "Generated_Outputs/Normalization_Sensitivity/result.csv",
        "Shared_source_data/RE_common/Normalization_Sensitivity/result.csv",
    )
    many(
        "wake_upwind",
        "Generated_Outputs/Wake_Upwind_Refinement",
        f"{radial}/Wake_Upwind_Refinement",
        ("discretization_comparison.csv", "existing_circuit_reference.csv"),
    )
    one(
        "table1",
        "Generated_Outputs/Regenerated/Table_1.csv",
        "Tables/Table_1/Table_1.csv",
    )
    many(
        "lchs_counts",
        "Generated_Outputs/Comparisons/KvN_LCHS/source_replay/audit/results",
        "Tables/Table_S11/LCHS_source_counts",
        ("dt1_stmc_histogram.csv", "t2_stmc_histogram.csv"),
    )

    one(
        "coupled_time_checks",
        "Generated_Outputs/Coupled_Time_Check/coupled_time_convergence.csv",
        "Figures/Fig_S5/plot_inputs/plot_values/coupled_time_convergence.csv",
    )
    time_base = "Additional_Source_Data/Three_Coordinate_Time_Convergence"
    for run, counts in (
        ("late40", (8, 16, 33, 66, 132, 264, 528)),
        ("long298", (66, 132, 264, 528, 1056, 2112)),
    ):
        for steps in counts:
            name = f"{run}/steps_{steps}_trajectory.csv"
            one(
                "three_coordinate_time_checks",
                f"Generated_Outputs/Three_Coordinate_Time_Convergence/{name}",
                f"{time_base}/{name}",
            )
        one(
            "three_coordinate_time_checks",
            f"Generated_Outputs/Three_Coordinate_Time_Convergence/{run}_time_convergence.csv",
            f"{time_base}/{run}_time_convergence.csv",
        )
    one(
        "three_coordinate_time_checks",
        "Generated_Outputs/Three_Coordinate_Time_Convergence/time_convergence.csv",
        f"{time_base}/time_convergence.csv",
    )
    many(
        "three_coordinate_time_resources",
        "Generated_Outputs/Three_Coordinate_Resource_Selection",
        "Additional_Source_Data/Three_Coordinate_Resource_Selection",
        (
            "candidate_selected.csv",
            "candidate_capped_selected.csv",
            "fastest_tested_by_rate.csv",
        ),
    )
    many(
        "radial_time_selection",
        "Generated_Outputs/Radial_Time_Selection",
        "Additional_Source_Data/Radial_Time_Convergence",
        (
            "time_convergence.csv",
            "candidate_fastest.csv",
            "fastest_tested_by_rate.csv",
            "selected_designs_p5.csv",
        ),
    )
    many(
        "grid_step_scaling",
        "Generated_Outputs/Grid_Step_Refinement",
        "Additional_Source_Data/Grid_Step_Refinement",
        (
            "polynomial_grid_step_sweep.csv",
            "late40_grid_step_sweep.csv",
            "long298_grid_step_sweep.csv",
            "grid_step_summary.csv",
        ),
    )

    if len(output) != 114 or len({r.published_in_source_data for r in output}) != 114:
        raise AssertionError(
            "The noncanonical CSV registry must cover exactly 114 unique outputs"
        )
    return tuple(output)


# Exact one-command producer entry points wherever a self-contained invocation
# is possible. Other groups require a declared input choice or external solver;
# their commands are documented separately in the reproduction guide.
RUN_COMMANDS: dict[str, tuple[tuple[str, ...], ...]] = {
    "table1": (("-m", "Source.RE.export_table1"),),
    "force_sampling": (("-m", "Source.ROM.force_sampling"),),
    "field_sampling": (
        ("-m", "Source.ROM.field_sampling_verification"),
    ),
    "sindy_cases": (("-m", "Source.POD_SINDy.evaluate_degree_cases"),),
    "physical_audit": (("-m", "Source.CFD.physical_audit"),),
    "normalization": (("-m", "Source.RE.normalization_sensitivity"),),
    "accuracy_target": (("-m", "Source.ROM.accuracy_target_sensitivity"),),
    "grid_oscillation": (
        ("-m", "Source.KvN.grid_oscillation_diagnostics"),
    ),
    "wake_upwind": (("-m", "Source.KvN.wake_upwind_refinement"),),
    "radial128_flow": (
        ("-m", "Source.KvN.radial_refinement_flow"),
    ),
    "radial128_resources": (
        ("-m", "Source.RE.radial_refinement_resources"),
    ),
    "three_coordinate_resources": (
        ("-m", "Source.RE.reestimate_one_clean_three_coordinate"),
    ),
    "supplement_policy": (
        (
            "-m",
            "Source.RE.generate_supplement_data",
            "--rates",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
        ),
    ),
    "ibm_measurements": (("Hardware/IBM/code/analyze_mps_ibm.py",),),
    "ibm_figure": (("Figure_Plots/ED1/generate_data.py",),),
    "lchs_counts": (("Comparisons/KvN_LCHS/source_replay/audit/count_circuit.py",),),
    "litinski": (
        (
            "-m",
            "Source.RE.generate_litinski_data",
            "--p",
            "0.0002",
            "0.0003",
            "0.0004",
            "0.0005",
            "0.0006",
            "0.0007",
            "0.0008",
            "0.0009",
            "--schemes",
            "A",
            "--replace-all",
        ),
    ),
    "cfd_forces": (("-m", "Source.CFD.reproduce_cfd_forces"),),
    "be_resources": (
        (
            "-m",
            "Source.RE.run",
            "--models",
            "generator",
            "--output",
            "Generated_Outputs/RE_BE_one_clean",
        ),
    ),
    "gidney_update": (
        (
            "-m",
            "Source.RE.update_gidney_inputs",
            "--rates",
            "2",
            "3",
            "4",
            "6",
            "7",
            "8",
            "9",
            "--preserve-prep",
            "--workload-overrides",
            "{source_data}/Additional_Source_Data/Three_Coordinate_Time_Convergence/baseline/workloads.json",
        ),
    ),
    "coupled_time_checks": (
        (
            "Source/KvN/generate_fig_s5_trajectories.py",
            "--output",
            "Generated_Outputs/Coupled_Time_Check",
        ),
    ),
    "three_coordinate_time_checks": (
        (
            "Source/KvN/three_coordinate_time_convergence.py",
            "late40",
            "--threads",
            "2",
        ),
        (
            "Source/KvN/three_coordinate_time_convergence.py",
            "long298",
            "--threads",
            "6",
            "--plot",
        ),
    ),
    "three_coordinate_time_resources": (
        ("Source/RE/three_coordinate_resource_selection.py", "--workers", "3"),
    ),
    "mps_reference": (
        (
            "-m",
            "Source.KvN.ensemble",
            "late40",
            "--policy",
            "common",
            "--model",
            "candidate",
        ),
        (
            "-m",
            "Source.KvN.ensemble",
            "long298",
            "--policy",
            "common",
            "--model",
            "candidate",
        ),
    ),
    "mps_trajectory": (("-m", "Source.KvN.verify_three_coordinate_mps", "both"),),
    "q5_upwind": (
        (
            "Source/KvN/q5_upwind_comparison.py",
            "{source_data}/Shared_source_data/Radial_refinement/Grid_Oscillation_Diagnostics/inputs/scalar_conservative_cell_masses.csv",
            "Generated_Outputs/Q5_Upwind_Comparison/mps",
        ),
        (
            "Source/KvN/q5_upwind_comparison.py",
            "{source_data}/Shared_source_data/Radial_refinement/Grid_Oscillation_Diagnostics/inputs/scalar_conservative_cell_masses.csv",
            "Generated_Outputs/Q5_Upwind_Comparison/positive_root",
            "--input-convention",
            "positive_root",
        ),
    ),
    # The phase source table consumes both published Q5 sweeps. Independent
    # stencil and full rerun checks live only in Internal_Workspace/Validation.
    "q5_validation": (("-m", "Source.KvN.q5_phase_comparison"),),
    "radial_step_diagnostic": (
        (
            "-m",
            "Source.KvN.radial_step_diagnostic",
            "--case",
            "{source_data}/Shared_source_data/Radial_refinement/Radial128/inputs",
        ),
    ),
    "radial_step_compare": (
        (
            "-m",
            "Source.KvN.radial_step_comparison",
            "--output",
            "Generated_Outputs/Step_Shot_Comparison/hardware_16_33.csv",
        ),
    ),
    "radial_time_selection": (("-m", "Source.KvN.radial_time_selection"),),
    "grid_step_scaling": (
        ("-m", "Source.KvN.grid_step_scaling", "polynomial"),
        ("-m", "Source.KvN.grid_step_scaling", "late40", "--threads", "2"),
        ("-m", "Source.KvN.grid_step_scaling", "long298", "--threads", "6"),
        (
            "-m",
            "Figure_Plots.plot_grid_step_scaling",
            "Generated_Outputs/Grid_Step_Refinement",
        ),
    ),
}


def default_source_data() -> Path:
    return Path(
        os.environ.get("KVN_SOURCE_DATA", str(CODES.parent / "Source_Data"))
    ).resolve()


def bootstrap(root: Path) -> None:
    """Install static indices and full-precision wake parameters in an empty root."""
    root.mkdir(parents=True, exist_ok=True)
    for name in ("manifest.csv", "Source_Data_Index.csv"):
        target = root / name
        if not target.exists():
            shutil.copyfile(source_data_root() / name, target)
    t1 = root / "Tables/Table_S1"
    t1.mkdir(parents=True, exist_ok=True)
    for name in ("model_configuration.json", "configuration.json"):
        target = t1 / name
        if not target.exists():
            shutil.copyfile(fixed_asset("catalogue", name), target)
    from Source.Data_Production.export_table_s1 import produce

    if not (t1 / "Table_S1_full_precision.csv").is_file():
        produce(root)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_header(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return next(csv.reader(stream))


def verify_manifest(root: Path) -> tuple[int, int, int]:
    """Compare available files with published SHA-256 and size metadata.

    This verifies byte-for-byte fidelity, not scientific equivalence. An
    incomplete Source_Data is normal during reproduction: missing entries are
    counted and reported separately from hash mismatches.
    """
    matched = missing = mismatched = 0
    with (root / "manifest.csv").open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        target = root / row["published_path"]
        if not target.is_file():
            missing += 1
            continue
        expected_sha = row.get("sha256", "")
        expected_size = row.get("size_bytes", "")
        if (
            expected_sha
            and digest(target) != expected_sha
            or expected_size
            and target.stat().st_size != int(expected_size)
        ):
            mismatched += 1
            print(f"HASH_MISMATCH {row['published_path']}")
        else:
            matched += 1
    return matched, missing, mismatched


def publish(group: str, data_root: Path, overwrite: bool = False) -> tuple[int, int]:
    """Preflight a group, then replace each file atomically; return written/unchanged counts.

    Existing different content requires ``overwrite`` and matching headers.
    A failure during preflight writes nothing. After commits begin, a later
    failure can leave earlier files updated: this is not a group transaction.
    The archive manifest is not rewritten by this publisher; use numerical
    comparison for regenerated results and digest checks for archived bytes.
    """
    selected = [r for r in all_routes() if group in ("all", r.group)]
    if not selected:
        raise ValueError(f"Unknown group: {group}")
    pending = []
    identical = 0
    for route in selected:
        if route.produced_in_codes.startswith('Generated_Outputs/'):
            src = data_root / route.produced_in_codes
        elif route.produced_in_codes.startswith('Hardware/IBM/data/'):
            src = data_root / 'Figures/Extended_Data_Fig_1/hardware_inputs' / Path(route.produced_in_codes).name
        else:
            src = CODES / route.produced_in_codes
        dst = data_root / route.published_in_source_data
        if not src.is_file():
            raise FileNotFoundError(
                f"Run producer {route.group!r} first; missing {src}"
            )
        if dst.exists() and digest(src) == digest(dst):
            identical += 1
            continue
        if dst.exists():
            if file_header(src) != file_header(dst):
                raise ValueError(
                    f"Column mismatch for {route.published_in_source_data}"
                )
            if not overwrite:
                raise FileExistsError(
                    f"Different published CSV exists: {dst}; inspect and rerun with --overwrite"
                )
        pending.append((src, dst))
    for src, dst in pending:
        dst.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=dst.parent, prefix=".source-data-", suffix=".tmp", delete=False
        ) as temp:
            temp_path = Path(temp.name)
        try:
            shutil.copyfile(src, temp_path)
            os.replace(temp_path, dst)
        finally:
            temp_path.unlink(missing_ok=True)
    return len(pending), identical


def run_producer(group: str, root: Path) -> None:
    """Run a registered command with the selected dataset and stop if the producer fails."""
    commands = RUN_COMMANDS.get(group)
    if commands is None:
        raise ValueError(
            f"Producer group {group!r} needs an explicit external/input-specific command. See reproduction guide"
        )
    env = os.environ.copy()
    env["KVN_SOURCE_DATA"] = str(root)
    # Direct script entry points also import the top-level Source package.
    env["PYTHONPATH"] = str(CODES) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    env.setdefault("MPLBACKEND", "Agg")
    for args in commands:
        resolved = [arg.replace("{source_data}", str(root.resolve())) for arg in args]
        resolved = [str(root / arg) if arg.startswith('Generated_Outputs/') else arg for arg in resolved]
        print(f"Running {group}: {sys.executable} {' '.join(resolved)}", flush=True)
        subprocess.run([sys.executable, *resolved], cwd=CODES, env=env, check=True)


def main() -> None:
    """Dispatch catalogue, regeneration, publication and verification actions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("bootstrap", "list", "publish", "reproduce", "verify")
    )
    parser.add_argument("group", nargs="?", default="all")
    parser.add_argument("--source-data", type=Path, default=default_source_data())
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing different published CSV values after header checks",
    )
    args = parser.parse_args()
    if args.action == "bootstrap":
        bootstrap(args.source_data)
        print(
            f"Initialized {args.source_data}; external scientific inputs must still be supplied"
        )
        return
    if args.action == "verify":
        matched, missing, mismatched = verify_manifest(args.source_data)
        print(
            json.dumps(
                dict(matched=matched, missing=missing, hash_mismatches=mismatched),
                ensure_ascii=False,
            )
        )
        if mismatched:
            raise SystemExit(1)
        return
    groups = Counter(r.group for r in all_routes())
    if args.action == "list":
        for group, count in sorted(groups.items()):
            print(
                f"{group:29s} {count:2d} outputs; automatic_run={'yes' if group in RUN_COMMANDS else 'manual'}"
            )
        print(f"Total mapped noncanonical CSV outputs: {sum(groups.values())}")
        return
    if args.group not in groups:
        parser.error(f"Choose a named group from 'list' (not {args.group!r})")
    if args.action == "reproduce":
        run_producer(args.group, args.source_data)
    written, identical = publish(args.group, args.source_data, args.overwrite)
    print(
        json.dumps(
            dict(
                group=args.group,
                written=written,
                unchanged=identical,
                source_data=str(args.source_data),
            ),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
