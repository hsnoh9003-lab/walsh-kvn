"""Test time-step convergence for the two three-coordinate wake intervals.

Inputs
------
Fixed polar grid, signed prepared MPS input, drift coefficients and continuum reference.

Outputs
-------
Generated_Outputs/Three_Coordinate_Time_Convergence/, or --output, with per-run diagnostics.

Run from the repository root
----------------------------
python -m Source.KvN.three_coordinate_time_convergence late40 --threads 2

Scope
-----
Only the number of symmetric transport steps varies. Differences to a finer run are refinement diagnostics, not exact-grid error bounds.
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
import ctypes
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import time

for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))
from Source.data_locations import generated_outputs_root
from Source.data_locations import set_source_data_root, shared_data
from Source.KvN import transport
from Source.ROM.field_budget import field_budget
from Source.ROM.model import load

DEFAULT_STEPS = {
    "late40": (8, 16, 33, 66, 132, 264, 528),
    "long298": (66, 132, 264, 528, 1056, 2112),
}
TARGET_RMS = 0.01
IMPLEMENTATION_ALLOWANCE = 1e-4


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def prepared_input(path):
    """Contract the archived signed MPS factors in their original bit order."""
    with np.load(path) as blocks:
        state = blocks["base"]
        indices = sorted(int(key[2:]) for key in blocks.files if key.startswith("U_"))
        for index in reversed(indices):
            basis = blocks[f"U_{index}"]
            state = (basis @ state.reshape(basis.shape[1], -1)).ravel()
    np.testing.assert_allclose(np.vdot(state, state), 1.0, rtol=0, atol=2e-13)
    return state


def matching_kernel(output, threads):
    """Compile the unchanged production matching source with optional OpenMP."""
    source = CODE_ROOT / "Source/KvN/matching.cpp"
    extension = ".dll" if os.name == "nt" else ".so"
    library = output / ("matching_" + digest(source)[:12] + f"_{threads}" + extension)
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["OMP_DYNAMIC"] = "FALSE"
    lib, command = transport.build_cpp_library(source, library, openmp=True)
    # Resolve a function exported by the kernel itself, not a transitive
    # dependency's omp_set_num_threads symbol (unreliable on Windows).
    set_threads = lib.matching_set_threads
    set_threads.argtypes = [ctypes.c_int]
    set_threads.restype = None
    set_threads(ctypes.c_int(threads))
    step = lib.matching_step
    step.argtypes = [
        np.ctypeslib.ndpointer(dtype=dtype, flags="C_CONTIGUOUS")
        for dtype in (np.complex128, np.int32, np.float64)
    ] + [ctypes.c_int] * 5
    step.restype = None
    # Keep execution paths local, but make the exported provenance portable.
    recorded_command = ["Source/KvN/matching.cpp" if arg == str(source.resolve())
                        else "<output>/" + library.name if arg == str(library.resolve())
                        else arg for arg in command]
    return step, {"source_sha256": digest(source), "compiler_command": recorded_command,
                  "compiler_working_directory": "Codes",
                  "output_placeholder": "The output directory selected by the caller.",
                  "threads": threads}


def terminal_readout(state, centres, archived):
    """Transform the angular Fourier register back to position and compute joint x,y,z moments."""
    radial, angular, vertical = centres
    position_state = np.fft.ifft(np.fft.ifftshift(state, axes=2), axis=2, norm="ortho")
    probability = abs(position_state) ** 2
    planar = probability.sum(axis=0)
    vertical_mass = probability.sum(axis=(1, 2))
    xy = (radial[:, None] * np.cos(angular), radial[:, None] * np.sin(angular))
    mean = np.array(
        [(planar * xy[0]).sum(), (planar * xy[1]).sum(), vertical @ vertical_mass]
    )
    # Accumulate E[x_i*x_j] from the joint position probability. In particular,
    # x-z and y-z terms must retain correlations between registers.
    second = np.zeros((3, 3))
    for i in (0, 1):
        for j in range(i, 2):
            second[i, j] = second[j, i] = (planar * xy[i] * xy[j]).sum()
        second[i, 2] = second[2, i] = np.einsum(
            "zrt,rt,z->", probability, xy[i], vertical
        )
    second[2, 2] = vertical_mass @ (vertical * vertical)
    return {
        "outcome_mean": mean.tolist(),
        "reference_outcome_mean": archived["reference_outcome_mean"],
        "covariance": (second - np.outer(mean, mean)).tolist(),
        "field_metric": archived["field_metric"],
        "probability_sum": float(probability.sum()),
        "grid_centres": [axis.tolist() for axis in centres],
    }


def coordinate_mean(state, centres):
    """Return Cartesian modal means and the probability norm at one saved time."""
    radial, angular, vertical = centres
    position_state = np.fft.ifft(np.fft.ifftshift(state, axes=2), axis=2, norm="ortho")
    probability = abs(position_state) ** 2
    planar = probability.sum(axis=0)
    mean = np.array(
        [
            (planar * radial[:, None] * np.cos(angular)).sum(),
            (planar * radial[:, None] * np.sin(angular)).sum(),
            vertical @ probability.sum(axis=(1, 2)),
        ]
    )
    return mean, float(probability.sum())


def summarize(output, run, checkpoints):
    """Compare each tested count with the next and finest tested runs, not an exact continuum state."""
    records = [
        json.loads(path.read_text())
        for path in sorted((output / run).glob("steps_*.json"))
    ]
    records.sort(key=lambda record: record["steps"])
    metric = np.array(records[-1]["readout"]["field_metric"])
    provenance = json.loads((output / run / "provenance.json").read_text())
    reference_norms = np.array(provenance["reference_field_norm_squared"])
    histories = []
    for record in records:
        table = np.genfromtxt(
            output / run / f'steps_{record["steps"]}_trajectory.csv',
            names=True,
            delimiter=",",
        )
        histories.append(np.column_stack([table[f"mean_{axis}"] for axis in "xyz"]))

    def peak_change(first, second):
        difference = first - second
        squared = np.einsum("ij,jk,ik->i", difference, metric, difference)
        return float(
            np.sqrt(
                np.maximum(0, squared) * reference_norms[-1] / reference_norms
            ).max()
        )

    # The finest tested run is a temporal-convergence comparator. It still
    # uses this spatial grid and is not the continuum reference used for bias.
    finest = np.array(records[-1]["readout"]["outcome_mean"])
    finest_state = np.load(checkpoints / f'{records[-1]["steps"]}.npy')
    rows = []
    for index, record in enumerate(records):
        mean = np.array(record["readout"]["outcome_mean"])
        difference = mean - finest
        next_change = None
        next_steps = None
        if index + 1 < len(records):
            next_steps = records[index + 1]["steps"]
            change = mean - np.array(records[index + 1]["readout"]["outcome_mean"])
            next_change = float(np.sqrt(max(0.0, change @ metric @ change)))
        current_state = np.load(checkpoints / f'{record["steps"]}.npy')
        overlap = np.vdot(finest_state, current_state)
        phase = overlap / abs(overlap) if abs(overlap) else 1.0
        state_distance = float(np.linalg.norm(current_state - phase * finest_state))
        budget = record["budget"]
        shots = budget["minimum_accepted_shots_under_rms_bound"]
        rows.append(
            {
                "run": run,
                "steps": record["steps"],
                "dt": record["dt"],
                "published_steps": record["published_steps"],
                "radial_cells": record["grid_r_theta_z"][0],
                "angular_cells": record["grid_r_theta_z"][1],
                "vertical_cells": record["grid_r_theta_z"][2],
                "noiseless_relative_field_bias": budget[
                    "noiseless_deterministic_relative_field_bias"
                ],
                "relative_field_bias_bound": budget[
                    "worst_case_relative_field_bias_bound"
                ],
                "relative_field_variance_bound": budget[
                    "relative_field_variance_per_shot_bound"
                ],
                "peak_sampled_relative_field_bias": record[
                    "peak_sampled_relative_field_bias"
                ],
                "below_target_at_all_67_times": record[
                    "peak_sampled_relative_field_bias"
                ]
                < TARGET_RMS,
                "accepted_shots": shots,
                "step_shot_product": None if shots is None else shots * record["steps"],
                "next_finer_steps": next_steps,
                "relative_field_change_to_next_finer": next_change,
                "peak_relative_field_change_to_next_finer": (
                    peak_change(histories[index], histories[index + 1])
                    if index + 1 < len(records)
                    else None
                ),
                "finest_tested_steps": records[-1]["steps"],
                "relative_field_change_to_finest": float(
                    np.sqrt(max(0.0, difference @ metric @ difference))
                ),
                "peak_relative_field_change_to_finest": peak_change(
                    histories[index], histories[-1]
                ),
                "phase_aligned_state_distance_to_finest": state_distance,
                "probability_sum": record["readout"]["probability_sum"],
                "wall_seconds": record["wall_seconds"],
            }
        )
    with (output / f"{run}_time_convergence.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def run_case(args):
    """Replay one fixed three-coordinate input/grid for a range of step counts."""
    if args.source_data:
        set_source_data_root(args.source_data)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    destination = output / args.run
    destination.mkdir(exist_ok=True)
    checkpoints = output / "_checkpoints" / args.run
    checkpoints.mkdir(parents=True, exist_ok=True)
    inputs = shared_data("MPS_KvN_inputs", "KvN", "inputs", args.run)
    replay = shared_data("MPS_KvN_inputs", "Three_Coordinate_MPS_Replay", args.run)
    frozen = (
        shared_data("MPS_KvN_inputs").parents[1]
        / "Additional_Source_Data/Three_Coordinate_Time_Convergence/baseline"
        / args.run
    )
    baseline = frozen if frozen.is_dir() else inputs
    protocol_path = baseline / "protocol.json"
    mps_path = replay / "mps_blocks.npz"
    raw_path = baseline / "terminal_readout.json"
    expected_path = baseline / "expected_field_budget.json"
    reference_path = shared_data(
        "MPS_KvN_inputs", "MPS", "domain_selection", args.run, "reference.csv"
    )
    config = json.loads(protocol_path.read_text())
    archived = json.loads(raw_path.read_text())
    expected = json.loads(expected_path.read_text())
    amplitude = prepared_input(mps_path)
    # nr, nt and nz are spatial cell counts in r, theta and z. Here nt is
    # the ANGULAR grid size; the separate variable count/steps controls time.
    nr, nt, nz = config["grid_r_theta_z"]
    initial = np.ascontiguousarray(
        np.fft.fftshift(
            np.fft.fft(amplitude.reshape(nz, nr, nt), axis=2, norm="ortho"), axes=2
        )
    )
    radial = (
        config["rrange"][0] + np.diff(config["rrange"])[0] * (np.arange(nr) + 0.5) / nr
    )
    angular = 2 * np.pi * np.arange(nt) / nt
    vertical = (
        config["zrange"][0] + np.diff(config["zrange"])[0] * (np.arange(nz) + 0.5) / nz
    )
    centres = (radial, angular, vertical)
    model = load("candidate")
    reference_csv = np.genfromtxt(reference_path, delimiter=",", names=True)
    reference_times = reference_csv["time"]
    reference_means = np.column_stack(
        [reference_csv[f"reference_mean_{axis}"] for axis in "xyz"]
    )
    np.testing.assert_allclose(
        reference_means[-1], archived["reference_outcome_mean"], atol=2e-12, rtol=0
    )
    field_modes = model.modes[:, model.mask]
    base_field = model.base[model.mask]
    field_gram = field_modes @ field_modes.T
    reference_field_norm_squared = (
        base_field @ base_field
        + 2 * reference_means @ (field_modes @ base_field)
        + np.einsum("ij,jk,ik->i", reference_means, field_gram, reference_means)
    )
    np.testing.assert_allclose(
        field_gram / reference_field_norm_squared[-1],
        archived["field_metric"],
        atol=2e-13,
        rtol=2e-12,
    )
    duration = np.diff(config["time_window"])[0]
    step, compilation = matching_kernel(output, args.threads)
    steps_to_test = sorted(set(args.steps or DEFAULT_STEPS[args.run]))
    assert steps_to_test and min(steps_to_test) > 0
    provenance = {
        "run": args.run,
        "model": "candidate quadratic three-coordinate wake",
        "protocol": config,
        "tested_steps": steps_to_test,
        "input": "Signed amplitudes contracted from the archived, gate-verified MPS factors",
        "input_file_sha256": digest(mps_path),
        "protocol_sha256": digest(protocol_path),
        "archived_readout_sha256": digest(raw_path),
        "archived_budget_sha256": digest(expected_path),
        "reference_trajectory_sha256": digest(reference_path),
        "coefficient_sha256": sha256(model.coefficients.tobytes()).hexdigest(),
        "prepared_amplitude_sha256": sha256(amplitude.tobytes()).hexdigest(),
        "negative_amplitude_probability": float(np.sum(amplitude[amplitude < 0] ** 2)),
        "target_relative_RMS": TARGET_RMS,
        "implementation_TV_allowance": IMPLEMENTATION_ALLOWANCE,
        "compiler": compilation,
        "reference": "Same terminal continuum Gaussian-ensemble means and field metric as the published run",
        "interpretation": "Differences to finer runs diagnose time discretization; they are not exact-grid error bounds. Step times shots is a propagation-work proxy, not physical runtime.",
        "trajectory_diagnostics": "67 archived reference times. At a noninteger step position, a copy receives one fractional matching step for observation; the main propagation is unchanged.",
        "reference_field_norm_squared": reference_field_norm_squared.tolist(),
    }
    # Check that the optional parallel kernel implements the production step.
    baseline_dt = duration / config["steps"]
    layers, tables = transport.rotation_tables(
        radial,
        vertical,
        np.arange(-nt // 2, nt // 2),
        model.coefficients,
        baseline_dt,
        config["stencil"],
    )
    serial = initial.copy()
    parallel = initial.copy()
    transport.kernel()(serial, layers, tables, nr, nt, nz, len(layers), tables.shape[1])
    step(parallel, layers, tables, nr, nt, nz, len(layers), tables.shape[1])
    kernel_error = float(np.max(abs(serial - parallel)))
    assert kernel_error < 1e-14, kernel_error
    provenance["parallel_vs_production_one_step_max_abs_difference"] = kernel_error
    del serial, parallel
    save_json(destination / "provenance.json", provenance)
    fingerprint = sha256(
        json.dumps(
            {
                key: provenance[key]
                for key in (
                    "input_file_sha256",
                    "protocol_sha256",
                    "archived_readout_sha256",
                    "coefficient_sha256",
                    "reference_trajectory_sha256",
                )
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()

    for count in steps_to_test:
        result_path = destination / f"steps_{count}.json"
        checkpoint = checkpoints / f"{count}.npy"
        if args.resume and result_path.exists() and checkpoint.exists():
            old = json.loads(result_path.read_text())
            assert old["input_fingerprint"] == fingerprint
            assert old["checkpoint_sha256"] == digest(checkpoint)
            print(f"Reusing {args.run}, {count} steps", flush=True)
            continue
        dt = duration / count
        layers, tables = transport.rotation_tables(
            radial,
            vertical,
            np.arange(-nt // 2, nt // 2),
            model.coefficients,
            dt,
            config["stencil"],
        )
        state = initial.copy()
        # Compare all 67 reference times even when they fall between full steps.
        # A temporary copy takes the fractional diagnostic step; the propagated
        # state and the circuit's counted full steps are unchanged by this readout.
        sample_positions = (
            (reference_times - config["time_window"][0]) * count / duration
        )
        sample_steps = np.floor(sample_positions + 1e-10).astype(int)
        fractional_steps = np.clip(sample_positions - sample_steps, 0, 1)
        means = []
        norms = []
        tick = time.monotonic()
        for index in range(count + 1):
            for sample_index in np.flatnonzero(sample_steps == index):
                fraction = fractional_steps[sample_index]
                sampled = state
                if fraction > 1e-10:
                    sampled = state.copy()
                    partial_layers, partial_tables = transport.rotation_tables(
                        radial,
                        vertical,
                        np.arange(-nt // 2, nt // 2),
                        model.coefficients,
                        fraction * dt,
                        config["stencil"],
                    )
                    step(
                        sampled,
                        partial_layers,
                        partial_tables,
                        nr,
                        nt,
                        nz,
                        len(partial_layers),
                        partial_tables.shape[1],
                    )
                sample_mean, sample_norm = coordinate_mean(sampled, centres)
                means.append(sample_mean)
                norms.append(sample_norm)
            if index == count:
                break
            step(state, layers, tables, nr, nt, nz, len(layers), tables.shape[1])
            if (index + 1) % max(1, count // 4) == 0:
                print(
                    f"{args.run}: {count} steps, {index + 1}/{count}, {time.monotonic() - tick:.1f}s",
                    flush=True,
                )
        raw = terminal_readout(state, centres, archived)
        means = np.asarray(means)
        assert means.shape == reference_means.shape == (67, 3)
        delta = means - reference_means
        trajectory_error = np.sqrt(
            np.einsum("ij,jk,ik->i", delta, field_gram, delta)
            / reference_field_norm_squared
        )
        trajectory_path = destination / f"steps_{count}_trajectory.csv"
        with trajectory_path.open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "time",
                    "mean_x",
                    "mean_y",
                    "mean_z",
                    "reference_x",
                    "reference_y",
                    "reference_z",
                    "relative_field_bias",
                    "probability_sum",
                    "diagnostic_fractional_step",
                ]
            )
            writer.writerows(
                [t, *mean, *reference, error, norm, fraction]
                for t, mean, reference, error, norm, fraction in zip(
                    reference_times,
                    means,
                    reference_means,
                    trajectory_error,
                    norms,
                    fractional_steps,
                )
            )
        budget = field_budget(raw, IMPLEMENTATION_ALLOWANCE, TARGET_RMS)
        baseline_checked = count == config["steps"]
        if baseline_checked:
            np.testing.assert_allclose(
                raw["outcome_mean"], archived["outcome_mean"], atol=2e-10, rtol=0
            )
            np.testing.assert_allclose(
                raw["covariance"], archived["covariance"], atol=2e-10, rtol=0
            )
            for key in (
                "noiseless_deterministic_relative_field_bias",
                "noiseless_relative_field_variance_per_shot",
            ):
                np.testing.assert_allclose(
                    budget[key], expected[key], atol=2e-10, rtol=0
                )
            assert (
                budget["minimum_accepted_shots_under_rms_bound"]
                == expected["minimum_accepted_shots_under_rms_bound"]
            )
            published_summary = json.loads((baseline / "summary.json").read_text())
            np.testing.assert_allclose(
                trajectory_error.max() * 100,
                published_summary["peak_sampled_field_error_percent"],
                atol=2e-9,
                rtol=0,
            )
        np.save(checkpoint, state)
        record = {
            "run": args.run,
            "steps": count,
            "dt": dt,
            "published_steps": config["steps"],
            "grid_r_theta_z": config["grid_r_theta_z"],
            "time_window": config["time_window"],
            "readout": raw,
            "budget": budget,
            "wall_seconds": time.monotonic() - tick,
            "peak_sampled_relative_field_bias": float(trajectory_error.max()),
            "max_sampled_norm_drift": float(np.max(abs(np.array(norms) - 1))),
            "trajectory_sha256": digest(trajectory_path),
            "published_readout_and_budget_verified": baseline_checked,
            "input_fingerprint": fingerprint,
            "checkpoint_sha256": digest(checkpoint),
        }
        save_json(result_path, record)
        print(
            json.dumps(
                {
                    "run": args.run,
                    "steps": count,
                    "bias_bound_percent": 100
                    * budget["worst_case_relative_field_bias_bound"],
                    "accepted_shots": budget["minimum_accepted_shots_under_rms_bound"],
                }
            ),
            flush=True,
        )
    summarize(output, args.run, checkpoints)
    combine_runs(output, args.plot)


def combine_runs(output, plot=False):
    """Combine completed intervals and optionally plot error versus tested step count."""
    rows = []
    for run in DEFAULT_STEPS:
        path = output / f"{run}_time_convergence.csv"
        if path.is_file():
            with path.open(newline="") as file:
                rows.extend(csv.DictReader(file))
    with (output / "time_convergence.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if not plot:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from Figure_Plots.diagnostic_style import diagnostic_style, COLORS

    with diagnostic_style():
        fig, axes = plt.subplots(1, 2, figsize=(10, 5.8))
        for ax, run, title in zip(
            axes, DEFAULT_STEPS, ("Late interval", "Growth interval")
        ):
            selected = [row for row in rows if row["run"] == run]
            if not selected:
                ax.set_visible(False)
                continue
            count = np.array([int(row["steps"]) for row in selected])
            bound = 100 * np.array(
                [float(row["relative_field_bias_bound"]) for row in selected]
            )
            peak = 100 * np.array(
                [float(row["peak_sampled_relative_field_bias"]) for row in selected]
            )
            change = 100 * np.array(
                [float(row["relative_field_change_to_finest"]) for row in selected]
            )
            ax.plot(
                count,
                bound,
                "o-",
                ms=3,
                lw=0.8,
                color=COLORS[0],
                label="Final bias bound (including implementation)",
            )
            ax.plot(
                count,
                peak,
                "s-",
                ms=3,
                lw=0.8,
                color=COLORS[1],
                label="Maximum bias over 67 times",
            )
            positive = change > 0
            ax.plot(
                count[positive],
                change[positive],
                "^--",
                ms=3,
                lw=0.8,
                color=COLORS[2],
                label="Change to finest run",
            )
            ax.axhline(
                100 * TARGET_RMS,
                color="0.45",
                linestyle=":",
                lw=0.8,
                label="1% field RMS target",
            )
            archived_steps = int(selected[0]["published_steps"])
            ax.axvline(archived_steps, color="0.55", linestyle="--", lw=0.8)
            ax.text(
                0.98,
                0.98,
                f"Archived baseline: {archived_steps} steps",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 1.0,
                    "pad": 2,
                },
            )
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_xticks(count, [str(n) for n in count], rotation=40)
            ax.set_xlabel(r"Time steps, $N_t$")
            ax.set_ylabel("Relative field error (%)")
            ax.set_title(title)
            ax.spines[["top", "right"]].set_visible(False)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0)
        )
        fig.text(
            0.06,
            0.01,
            "Bias is measured against the corresponding continuum ensemble. The final bias bound includes the 10⁻⁴ implementation allowance.\n"
            "The peak is the noiseless maximum over 67 saved times. The 1% field RMS target also includes sampling error.\n"
            "Vertical lines mark archived 66/528-step baselines, not the current 33/264-step selections.",
            fontsize=9,
            va="bottom",
        )
        fig.tight_layout(rect=(0, 0.16, 1, 0.88))
        for extension in ("png", "pdf"):
            metadata = (
                {"CreationDate": None, "ModDate": None} if extension == "pdf" else None
            )
            fig.savefig(
                output / f"three_coordinate_time_convergence.{extension}",
                metadata=metadata,
            )
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", choices=tuple(DEFAULT_STEPS))
    parser.add_argument("--steps", nargs="+", type=int)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--source-data", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=generated_outputs_root() / "Three_Coordinate_Time_Convergence",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--plot", action="store_true", help="Also plot the completed run summaries"
    )
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    run_case(args)
