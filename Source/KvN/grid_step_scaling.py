"""Compare spatial and temporal refinement at fixed prepared input and duration.

Inputs
------
Polynomial benchmark inputs or one of the late40/long298 three-coordinate runs.

Outputs
-------
Generated_Outputs/Grid_Step_Refinement/, or --output, with numerical diagnostics and run metadata.

Run from the repository root
----------------------------
python -m Source.KvN.grid_step_scaling polynomial

Scope
-----
Polynomial tests use a finite-grid matrix exponential. Wake Richardson diagnostics are convergence estimates, not rigorous exact-grid error bounds.
"""

from __future__ import annotations

# Direct-file execution must locate Codes/ independently of the working directory.
if __package__ in (None, ""):
    import sys as _path_sys
    from pathlib import Path as _Path
    _codes = next(p for p in _Path(__file__).resolve().parents
                  if (p / "Source/data_locations.py").is_file())
    _path_sys.path.insert(0, str(_codes))

from Source.data_locations import generated_outputs_root

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
from scipy.sparse.linalg import expm_multiply


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_label(path, args):
    resolved = Path(path).resolve()
    for root, prefix in [
        (args.code_root, "Codes/"),
        (args.source_data, "Source_Data/"),
        (args.output, "output/"),
    ]:
        try:
            return prefix + resolved.relative_to(root).as_posix()
        except ValueError:
            continue
    return str(path)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def phase_distance(first, second):
    """Remove a physically irrelevant global phase before comparing state vectors."""
    overlap = np.vdot(second, first)
    phase = overlap / abs(overlap) if abs(overlap) else 1.0
    return float(np.linalg.norm(first - phase * second))


def prolong(state, factor, axes):
    """Split each old cell equally; preserve its signed/complex amplitude."""
    result = state.copy()
    for axis in axes:
        result = np.repeat(result, factor, axis=axis) / np.sqrt(factor)
    np.testing.assert_allclose(
        np.vdot(result, result), np.vdot(state, state), atol=1e-12, rtol=0
    )
    return np.ascontiguousarray(result)


def polynomial(args):
    """Vary cells and time steps for fixed-input one- and two-dimensional polynomial cases."""
    os.environ["POLYNOMIAL_VALIDATION_OUTPUT"] = str(args.output / "_model_work")
    from Source.KvN import polynomial_validation as model
    from Source.Data_Production.data_assets import fixed_asset

    inputs = fixed_asset("inputs", data=args.source_data)
    cases = {c["name"]: c for c in model.declared_cases()}
    specifications = []
    for name in ("quadratic", "cubic_interior", "quintic_multistable"):
        for location in ("interior", "near_stable", "near_unstable"):
            specifications.append(
                (name, location, inputs / "polynomial" / f"{name}__{location}.npz")
            )
    specifications.append(
        (
            "quintic_multistable",
            "boundary_clipped",
            args.source_data
            / "Shared_source_data/Radial_refinement/Grid_Oscillation_Diagnostics/inputs/quintic_multistable__boundary_clipped.npz",
        )
    )
    specifications += [
        (
            "coupled_cubic_2d",
            "interior",
            args.source_data
            / "Shared_source_data/MPS_KvN_inputs/Figure5_Bias_Analysis/input/grid32/coupled_cubic_2d__interior.npz",
        ),
        (
            "coupled_cubic_2d",
            "boundary_clipped",
            inputs
            / "retained_experiments/trajectories/coupled_cubic_2d__boundary_clipped.npz",
        ),
    ]
    for location in ("near_stable", "near_unstable"):
        specifications.append(
            (
                "coupled_cubic_2d",
                location,
                inputs / "polynomial" / f"coupled_cubic_2d__{location}.npz",
            )
        )
    rows, provenance = [], []
    for name, location, source in specifications:
        case = cases[name]
        dimension = case["D"]
        base_cells = 64 if dimension == 1 else 32
        with np.load(source) as data:
            initial = (
                data["initial_state"]
                if "initial_state" in data
                else data["state_history"][0]
            ).copy()
        budget_path = inputs / "retained_experiments/budget"
        if dimension == 2:
            budget_path /= "grid32"
        budget_path /= f"{name}__{location}.npz"
        with np.load(budget_path) as data:
            reference = data["terminal_reference_moments"].copy()
            archived_probability = data["probability_history"][-1].copy()
        norm = np.linalg.norm(reference)
        assert abs(np.vdot(initial, initial).real - 1) < 1e-10
        provenance.append(
            {
                "case": name,
                "input": location,
                "source": source_label(source, args),
                "sha256": digest(source),
                "base_cells_per_axis": base_cells,
                "reference_moments": reference.tolist(),
            }
        )
        # Refine the prepared cell density by equal subdivision. The step sweep
        # then measures temporal error against exact evolution on each new grid;
        # it does not independently refit a Gaussian at every resolution.
        for factor in (1, 2, 4):
            cells = base_cells * factor
            state0 = prolong(
                initial.reshape((base_cells,) * dimension), factor, range(dimension)
            ).ravel()
            generator, pairs = model.generator_and_pairs(case, cells)
            exact = expm_multiply(generator, state0, traceA=0.0)
            observables = model.observable_values(model.coordinates(cells, dimension))
            exact_mean = observables @ abs(exact) ** 2
            assert abs(np.vdot(exact, exact).real - 1) < 2e-11
            for count in (32, 64, 128, 256, 512):
                tick = time.monotonic()
                final, _ = model.pair_evolution(
                    state0, pairs, dimension, 1 / count, count, save_every=count
                )
                mean = observables @ abs(final) ** 2
                if factor == 1 and count == 64:
                    np.testing.assert_allclose(
                        abs(final) ** 2, archived_probability, atol=3e-11, rtol=2e-9
                    )
                rows.append(
                    {
                        "model": name,
                        "input": location,
                        "dimensions": dimension,
                        "base_cells_per_axis": base_cells,
                        "refinement_factor": factor,
                        "cells_per_axis": cells,
                        "steps": count,
                        "duration": 1.0,
                        "dt": 1 / count,
                        "temporal_relative_moment_error": float(
                            np.linalg.norm(mean - exact_mean) / norm
                        ),
                        "phase_aligned_state_error": phase_distance(final, exact),
                        "total_relative_moment_bias": float(
                            np.linalg.norm(mean - reference) / norm
                        ),
                        "exact_grid_relative_moment_bias": float(
                            np.linalg.norm(exact_mean - reference) / norm
                        ),
                        "norm_defect": float(abs(np.vdot(final, final).real - 1)),
                    }
                )
            print(f"{name}/{location}: {cells} cells per axis complete", flush=True)
            write_csv(args.output / "polynomial_grid_step_sweep.csv", rows)
    write_json(
        args.output / "polynomial_provenance.json",
        {
            "inputs": provenance,
            "duration": 1.0,
            "cell_refinement_factors": [1, 2, 4],
            "tested_steps": [32, 64, 128, 256, 512],
            "initial_state_refinement": "Uniform subdivision of the original signed cell amplitudes; no refit or replacement by a positive square root.",
            "temporal_reference": "scipy.sparse.linalg.expm_multiply of the unchanged finite-grid generator on each refined grid",
            "observable_error_normalization": "Euclidean norm of the fixed matched continuum reference moments",
            "production_replay": "Every available original 64-step terminal probability reproduced before using its refinements.",
            "note": "The boundary-clipped quintic is a diagnostic case; it is not a feasible 1% main result.",
        },
    )


def wake(args):
    """Refine the radial/vertical wake grids with fixed angular resolution and compare time-step policies."""
    sys.path.insert(0, str(args.code_root))
    from Source.data_locations import set_source_data_root

    set_source_data_root(args.source_data)
    from Source.KvN import transport
    from Source.ROM.model import load
    from Source.KvN.three_coordinate_time_convergence import (
        prepared_input,
        matching_kernel,
        coordinate_mean,
    )

    run = args.task
    input_root = args.source_data / "Shared_source_data/MPS_KvN_inputs"
    config_path = input_root / "KvN/inputs" / run / "protocol.json"
    config = json.loads(config_path.read_text())
    mps_path = input_root / "Three_Coordinate_MPS_Replay" / run / "mps_blocks.npz"
    nr0, nt, nz0 = config["grid_r_theta_z"]
    prepared = prepared_input(mps_path).reshape(nz0, nr0, nt)
    # Refinement acts on physical r,z cells; the same Fourier basis is retained.
    initial_fourier = np.ascontiguousarray(
        np.fft.fftshift(np.fft.fft(prepared, axis=2, norm="ortho"), axes=2)
    )
    duration = float(np.ptp(config["time_window"]))
    base_steps = config["steps"]
    assert base_steps == {"late40": 33, "long298": 264}[run]
    reference_path = input_root / "KvN/inputs" / run / "terminal_readout.json"
    archived = json.loads(reference_path.read_text())
    metric = np.asarray(archived["field_metric"])
    reference_mean = np.asarray(archived["reference_outcome_mean"])
    model = load("candidate")
    kernel, compiler = matching_kernel(args.output, args.threads)
    field_error = lambda v: float(np.sqrt(max(0.0, v @ metric @ v)))
    factors = [1, 2, 4] if run == "late40" else [1, 2]
    rows = []
    for factor in factors:
        nr, nz = nr0 * factor, nz0 * factor
        initial = prolong(initial_fourier, factor, (0, 1))
        r = config["rrange"][0] + np.ptp(config["rrange"]) * (np.arange(nr) + 0.5) / nr
        z = config["zrange"][0] + np.ptp(config["zrange"]) * (np.arange(nz) + 0.5) / nz
        angular = 2 * np.pi * np.arange(nt) / nt
        centres = (r, angular, z)
        modes = np.arange(-nt // 2, nt // 2)
        results = []
        # The primary grid-scaling diagnostic is the directly measured change
        # between Nt and 2*Nt, which does not require a zero-step reference.
        # Additional refinements also report extrapolation diagnostics.
        multipliers = [1, 2, 4, 8, 16] if run == "late40" else [1, 2, 4, 8]
        if run == "late40" and factor == 4:
            multipliers.insert(0, 0.5)
        for multiplier in multipliers:
            count = int(base_steps * multiplier * max(1, factor // 2))
            path = args.output / f"{run}_r{nr}_z{nz}_steps{count}.json"
            fingerprint = hashlib.sha256(
                (
                    digest(config_path)
                    + digest(mps_path)
                    + str((nr, nt, nz, count))
                    + digest(args.code_root / "Source/KvN/matching.cpp")
                ).encode()
            ).hexdigest()
            if path.exists():
                result = json.loads(path.read_text())
                assert result["fingerprint"] == fingerprint
                print(f"Reusing {run} {nr}x{nt}x{nz}, {count} steps", flush=True)
            else:
                layers, tables = transport.rotation_tables(
                    r, z, modes, model.coefficients, duration / count, config["stencil"]
                )
                state = initial.copy()
                tick = time.monotonic()
                for i in range(count):
                    kernel(
                        state, layers, tables, nr, nt, nz, len(layers), tables.shape[1]
                    )
                    if (i + 1) % max(1, count // 4) == 0:
                        print(
                            f"{run} {nr}x{nt}x{nz}, {count} steps: {i+1}/{count}, {time.monotonic()-tick:.1f}s",
                            flush=True,
                        )
                mean, norm = coordinate_mean(state, centres)
                result = {
                    "model": run,
                    "refinement_factor": factor,
                    "grid_r_theta_z": [nr, nt, nz],
                    "steps": count,
                    "duration": duration,
                    "dt": duration / count,
                    "mean": mean.tolist(),
                    "norm_defect": abs(norm - 1),
                    "wall_seconds": time.monotonic() - tick,
                    "total_relative_field_bias": field_error(mean - reference_mean),
                    "fingerprint": fingerprint,
                }
                assert result["norm_defect"] < 1e-9
                if factor == 1 and count == base_steps:
                    np.testing.assert_allclose(
                        mean, archived["outcome_mean"], atol=5e-10, rtol=0
                    )
                    result["production_mean_replay_max_abs"] = float(
                        np.max(abs(mean - np.array(archived["outcome_mean"])))
                    )
                write_json(path, result)
                del state
            results.append(result)
        means = [np.array(row["mean"]) for row in results]
        richardson = [(4 * b - a) / 3 for a, b in zip(means, means[1:])]
        extrapolation_change = field_error(richardson[-1] - richardson[-2])
        next_difference = [field_error(b - a) for a, b in zip(means, means[1:])]
        orders = [
            float(np.log2(a / b)) if a > 0 and b > 0 else None
            for a, b in zip(next_difference, next_difference[1:])
        ]
        for i, result in enumerate(results):
            rows.append(
                {
                    "run": run,
                    "refinement_factor": factor,
                    "radial_cells": nr,
                    "angular_cells": nt,
                    "shift_cells": nz,
                    "steps": result["steps"],
                    "duration": duration,
                    "dt": result["dt"],
                    "estimated_temporal_relative_field_error": field_error(
                        means[i] - richardson[-1]
                    ),
                    "total_relative_field_bias": result["total_relative_field_bias"],
                    "next_refinement_field_difference": (
                        next_difference[i] if i < len(next_difference) else None
                    ),
                    "observed_time_order": orders[i] if i < len(orders) else None,
                    "reference_extrapolation_change": extrapolation_change,
                    "finest_steps": results[-1]["steps"],
                    "norm_defect": result["norm_defect"],
                }
            )
        write_csv(args.output / f"{run}_grid_step_sweep.csv", rows)
        write_json(
            args.output / f"{run}_r{nr}_z{nz}_reference.json",
            {
                "reference_mean": richardson[-1].tolist(),
                "preceding_extrapolation_mean": richardson[-2].tolist(),
                "reference_extrapolation_change": extrapolation_change,
                "observed_orders": orders,
                "interpretation": "Richardson diagnostic, not a rigorous exact-grid error bound.",
            },
        )
        del initial
    compiler["compiler_command"] = [
        source_label(x, args) if Path(x).is_absolute() else x
        for x in compiler["compiler_command"]
    ]
    write_json(
        args.output / f"{run}_provenance.json",
        {
            "run": run,
            "protocol": config,
            "config_sha256": digest(config_path),
            "mps_sha256": digest(mps_path),
            "coefficient_sha256": hashlib.sha256(
                model.coefficients.tobytes()
            ).hexdigest(),
            "compiler": compiler,
            "refinement_factors": factors,
            "input_refinement": "Uniform subdivision of original signed/complex cell amplitudes on r and z, retaining the 32-mode angular basis.",
            "fixed_within_run": [
                "physical domain",
                "initial cell density and phases",
                "total duration",
                "polynomial drift",
                "difference stencil",
                "angular basis",
                "field metric",
            ],
            "reference": "Fourth-order Richardson extrapolation of the final two time-refinement results, checked against the preceding extrapolation.",
            "primary_comparison": "Direct terminal field difference between Nt and 2*Nt on the same grid, compared between grids at fixed Nt or fixed Nt/N. This requires no extrapolated reference.",
            "scope": "Time-splitting error at fixed grid, with a controlled refinement between grids; no new state-preparation circuit or physical resource optimization.",
        },
    )


def main(argv=None):
    """Resolve the selected code/data roots before starting a refinement run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=["polynomial", "late40", "long298"])
    parser.add_argument(
        "--code-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--source-data", type=Path)
    parser.add_argument(
        "--output", type=Path,
        help="Output directory (default: selected dataset's Generated_Outputs/Grid_Step_Refinement)",
    )
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    args.code_root = args.code_root.expanduser().resolve()
    sys.path.insert(0, str(args.code_root))
    from Source.data_locations import set_source_data_root, source_data_root

    args.source_data = (
        set_source_data_root(args.source_data)
        if args.source_data
        else source_data_root()
    )
    args.output = (args.output or generated_outputs_root() / "Grid_Step_Refinement").expanduser().resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    (polynomial if args.task == "polynomial" else wake)(args)


if __name__ == "__main__":
    main()
