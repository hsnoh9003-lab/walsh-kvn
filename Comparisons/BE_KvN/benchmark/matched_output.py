#!/usr/bin/env python3
"""Replay the Re=100 64-cell output task while replacing only the radial map."""

from __future__ import annotations

import argparse
from io import BytesIO
from itertools import product
import json
from math import ceil, sqrt
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from scipy.linalg import expm, polar


DATA = "Figure2/unified_run/cases/Re100/data/"


def read_npz(archive, filename):
    with np.load(BytesIO(archive.read(DATA + filename)), allow_pickle=False) as data:
        return dict(data)


def skew_from_faces(values, spacing):
    return np.diag(values, -1) / (2 * spacing) - np.diag(values, 1) / (2 * spacing)


def radial_generators(radius):
    """Build the SI face stencil and zero-wrap symmetric cell stencil."""
    spacing = 1 / len(radius)
    np.testing.assert_array_equal(radius, (np.arange(len(radius)) + 0.5) * spacing)
    faces = np.arange(1, len(radius)) * spacing
    cell_drift = radius - radius**3
    face_drift = faces - faces**3
    return {
        "exact_face": skew_from_faces(face_drift, spacing),
        "exact_cell_centered": skew_from_faces((cell_drift[:-1] + cell_drift[1:]) / 2, spacing),
    }


def replay(radial_map, initial, radius, substeps, probe_phase):
    occupations = np.array([0, 1, 1, 2])
    half_kick = np.exp(0.5j * probe_phase * occupations[:, None] * radius**2)
    history = np.empty((substeps + 1, 4, len(radius)), dtype=complex)
    history[0] = initial / 2
    for index in range(substeps):
        history[index + 1] = ((history[index] * half_kick) @ radial_map.T) * half_kick
    return read_probes(history)


def replay_joint_exact(generator, initial, radius, elapsed, probe_rate):
    """Exact finite-grid evolution under simultaneous transport and phase."""
    history = np.empty((len(elapsed), 4, len(radius)), dtype=complex)
    for branch, occupation in enumerate((0, 1, 1, 2)):
        joint_hamiltonian = 1j * generator - np.diag(probe_rate * occupation * radius**2)
        eigenvalues, eigenvectors = np.linalg.eigh(joint_hamiltonian)
        weights = eigenvectors.conj().T @ initial / 2
        history[:, branch] = (np.exp(-1j * elapsed[:, None] * eigenvalues) * weights) @ eigenvectors.T
    return read_probes(history)


def read_probes(history):
    hadamard = np.array([[1, 1], [1, -1]]) / sqrt(2)
    readout = np.kron(hadamard @ np.diag([1, -1j]), hadamard)
    survival = (abs(history) ** 2).sum(axis=(1, 2))
    outcome_weights = (abs(readout @ history) ** 2).reshape(len(history), -1)
    probabilities = outcome_weights / survival[:, None]
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-13, rtol=0)
    return history / np.sqrt(survival[:, None, None]), probabilities, survival


def field_coefficients(moments, reference):
    rotate_xy = (moments[:, 0] + 1j * moments[:, 1]) * np.exp(1j * reference["known_phase"])
    radius_scale = float(reference["R"])
    shift = float(reference["c0"]) + float(reference["c2"]) * radius_scale**2 * moments[:, 2]
    return np.c_[radius_scale * rotate_xy.real, radius_scale * rotate_xy.imag, shift]


def splitting_errors(generator, radial_map, radius, substeps, radial_step, probe_phase):
    """Compare the archive's split phase-kick schedule to joint exact grid evolution."""
    step_error = 0.
    final_error = 0.
    for occupation in (0, 1, 2):
        kick_generator = 1j * probe_phase * occupation * radius**2
        half_kick = np.exp(0.5 * kick_generator)
        split_step = half_kick[:, None] * radial_map * half_kick[None, :]
        joint_generator = radial_step * generator + np.diag(kick_generator)
        exact_step = expm(joint_generator)
        exact_final = expm(substeps * joint_generator)
        step_error = max(step_error, float(np.linalg.norm(split_step - exact_step, 2)))
        final_error = max(
            final_error, float(np.linalg.norm(np.linalg.matrix_power(split_step, substeps) - exact_final, 2))
        )
    return {"step_operator_2_norm": step_error, "full_window_operator_2_norm": final_error}


def output_metrics(probabilities, values, target, field_metric, field_target, tv):
    """Apply the source shot model with a supplied conditioned TV bound."""
    mean = probabilities @ values
    covariance = (values.T * probabilities) @ values - np.outer(mean, mean)
    error = mean - target
    value_range = np.ptp(values, axis=0)
    product_range = np.ptp(values[:, :, None] * values[:, None, :], axis=0)
    change = (
        product_range + np.outer(abs(mean), value_range) + np.outer(value_range, abs(mean))
    ) * tv + np.outer(value_range, value_range) * tv**2
    bias2 = float(error @ field_metric @ error)
    variance = float(np.sum(field_metric * covariance))
    variance_bound = variance + float(np.sum(abs(field_metric) * change))
    corner_errors = error + np.array(list(product((-1, 1), repeat=3))) * value_range * tv
    bias2_bound = float(np.max(np.einsum("ni,ij,nj->n", corner_errors, field_metric, corner_errors)))
    denominator = field_target**2 - bias2_bound
    shots = max(1, ceil(variance_bound / denominator)) if denominator > 0 else None
    return {
        "mean": mean.tolist(),
        "covariance": covariance.tolist(),
        "relative_field_bias": sqrt(bias2),
        "relative_field_variance": variance,
        "conditional_measurement_tv_bound": tv,
        "relative_field_bias_upper_bound": sqrt(bias2_bound),
        "relative_field_variance_upper_bound": variance_bound,
        "accepted_shots_for_field_target": shots,
        "rms_bound_at_814_accepted_shots": sqrt(bias2_bound + variance_bound / 814),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("release_out/Figure2_Source.zip"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--operator-tolerance", type=float, default=1e-4)
    parser.add_argument("--field-target", type=float, default=0.01)
    parser.add_argument("--implementation-allowance", type=float, default=1e-4)
    parser.add_argument("--candidate-map", type=Path, help="Optional 64×64 .npy radial acceptance block")
    parser.add_argument("--joint-exact", action="store_true", help="Also replay the exact joint face-grid evolution")
    args = parser.parse_args()
    if not 0 < args.operator_tolerance < 1 or not 0 <= args.implementation_allowance < 1:
        parser.error("Expected 0 < operator tolerance < 1 and 0 ≤ implementation allowance < 1")
    if args.field_target <= 0:
        parser.error("Expected positive field target")

    with ZipFile(args.source) as archive:
        config = json.loads(archive.read(DATA + "new_qsvt_report.json"))
        published = read_npz(archive, "qsvt_transport.npz")
        reference = read_npz(archive, "unified_reference.npz")
        physical = read_npz(archive, "physical_figure_data.npz")
        archived_field = read_npz(archive, "archived_final_field.npz")["reference"]
    radius = published["r"]
    np.testing.assert_array_equal(radius, reference["r"])
    np.testing.assert_allclose(published["initial"], reference["initial_amplitude"], atol=1e-14, rtol=0)
    substeps = config["radial_substeps"]
    duration = config["physical_duration"]
    normalized_time = config["configuration"]["mu"] * duration
    radial_step = normalized_time / substeps
    probe_phase = config["configuration"]["gamma_normalized"] * duration / substeps
    generators = radial_generators(radius)
    maps = {"archived_qsvt": published["B"]}
    maps.update({key: expm(radial_step * value) for key, value in generators.items()})
    if args.joint_exact:
        maps["joint_exact_face"] = None
    if args.candidate_map:
        candidate_map = np.load(args.candidate_map, allow_pickle=False)
        if candidate_map.shape != published["B"].shape:
            parser.error("Candidate map must have the archived radial map shape")
        maps["candidate"] = candidate_map

    indices = published["selected_indices"]
    values = published["values"]
    target = reference["continuum_moments"][-1]
    field_metric = reference["M"].T @ reference["G"] @ reference["M"] / float(reference["norm"])**2
    ref_coefficients = field_coefficients(reference["continuum_moments"], reference)
    np.testing.assert_allclose(ref_coefficients, reference["reference_coefficients"], atol=2e-13, rtol=0)
    ref_fields = physical["base"] + np.einsum("ni,ijkm->njkm", ref_coefficients, physical["modes"])
    np.testing.assert_allclose(ref_fields[-1], archived_field, atol=2e-13, rtol=0)
    field_norms = np.linalg.norm(ref_fields.reshape(len(indices), -1), axis=1)
    np.testing.assert_allclose(field_norms, reference["reference_field_norm"], atol=2e-12, rtol=0)
    implementation_allowance = args.implementation_allowance
    epsilon = args.operator_tolerance
    success_lower = (1 - epsilon)**2
    if success_lower <= implementation_allowance:
        parser.error("Requested tolerances do not certify a positive accepted fraction")
    polar_target = float(config["phase_certificate"]["target"])
    exact_polar = polar(published["A"])[0]

    summary = {
        "physical_window": config["physical_window"],
        "radial_cells": len(radius),
        "substeps": substeps,
        "normalized_time": normalized_time,
        "radial_step": radial_step,
        "physical_time_step": duration / substeps,
        "normalized_probe_phase_rate": config["configuration"]["gamma_normalized"],
        "expected_relative_rms_field_target": args.field_target,
        "generator_2_norm": {key: float(np.linalg.norm(value, 2)) for key, value in generators.items()},
        "generator_difference_2_norm": float(
            np.linalg.norm(generators["exact_face"] - generators["exact_cell_centered"], 2)
        ),
        "exponential_approximation_budget": {
            "aggregate_operator_2_norm": epsilon,
            "step_operator_2_norm": epsilon / substeps,
            "accepted_probability_lower_bound": success_lower,
            "conditioned_measurement_tv_bound": epsilon,
        },
        "published_polar_target": {
            "scaled_target": polar_target,
            "max_singular_deviation_from_scaled_target": float(
                np.max(abs(np.linalg.svd(published["B"], compute_uv=False) - polar_target))
            ),
            "step_operator_2_norm_to_polar_of_signal": float(np.linalg.norm(published["B"] - exact_polar, 2)),
            "step_polar_to_face_exponential_2_norm": float(
                np.linalg.norm(exact_polar - maps["exact_face"], 2)
            ),
        },
        "probe_radial_strang_splitting_error": {
            key: splitting_errors(generator, maps[key], radius, substeps, radial_step, probe_phase)
            for key, generator in generators.items()
        },
        "methods": {},
    }
    arrays = {
        "reference": ref_fields,
        "times": reference["times"],
        "grid_x": physical["grid_x"],
        "grid_y": physical["grid_y"],
        "fluid_mask": physical["fluid_mask"],
        "outcome_values": values,
    }
    for key, radial_map in maps.items():
        if key == "joint_exact_face":
            elapsed = np.arange(substeps + 1) * radial_step
            np.testing.assert_allclose(
                elapsed[indices], config["configuration"]["mu"] * reference["elapsed"], atol=2e-14, rtol=0
            )
            history, probabilities, survival = replay_joint_exact(
                generators["exact_face"], published["initial"], radius, elapsed, probe_phase / radial_step,
            )
            norm = 1.
        else:
            history, probabilities, survival = replay(
                radial_map, published["initial"], radius, substeps, probe_phase,
            )
            norm = float(np.linalg.norm(radial_map, 2))
        moments = probabilities @ values
        coefficients = field_coefficients(moments[indices], reference)
        fields = physical["base"] + np.einsum("ni,ijkm->njkm", coefficients, physical["modes"])
        field_bias = np.linalg.norm((fields - ref_fields).reshape(len(indices), -1), axis=1) / field_norms
        success = min(1., float(survival[-1]))
        if norm > 1 + 1e-12 or success <= implementation_allowance:
            parser.error(f"{key} fails the contraction or success assumptions")
        method = {
            "substep_contraction_norm": norm,
            "success_probability": success,
            "final_field_coefficients": coefficients[-1].tolist(),
            "max_relative_field_bias_across_saved_times": float(field_bias.max()),
        }
        final = probabilities[-1]
        method["ideal"] = output_metrics(final, values, target, field_metric, args.field_target, 0.)
        np.testing.assert_allclose(field_bias[-1], method["ideal"]["relative_field_bias"], atol=1e-14, rtol=0)
        if key == "archived_qsvt":
            np.testing.assert_allclose(history, published["state_history"], atol=2e-14, rtol=0)
            np.testing.assert_allclose(probabilities, published["joint_probabilities"], atol=2e-14, rtol=0)
            np.testing.assert_allclose(survival, published["survival"], atol=2e-14, rtol=0)
            np.testing.assert_allclose(field_bias * 100, published["field_bias_pct"], atol=2e-12, rtol=0)
            method["archived_implementation_allowance"] = output_metrics(
                final, values, target, field_metric, args.field_target,
                2 * implementation_allowance / (success - implementation_allowance),
            )
        elif key.startswith("exact_") or key == "joint_exact_face":
            method["with_approximation_budget"] = output_metrics(
                final, values, target, field_metric, args.field_target, epsilon,
            )
            method["with_approximation_and_implementation_budgets"] = output_metrics(
                final, values, target, field_metric, args.field_target,
                epsilon + 2 * implementation_allowance / (success_lower - implementation_allowance),
            )
        else:
            method["step_operator_2_norm_to_exact_cell_centered"] = float(
                np.linalg.norm(radial_map - maps["exact_cell_centered"], 2)
            )
            method["step_operator_2_norm_to_exact_face"] = float(
                np.linalg.norm(radial_map - maps["exact_face"], 2)
            )
            method["terminal_outcome_tv_to_exact_cell_centered"] = float(
                0.5 * np.sum(abs(final - arrays["exact_cell_centered_probabilities"][-1]))
            )
            method["terminal_outcome_tv_to_exact_face"] = float(
                0.5 * np.sum(abs(final - arrays["exact_face_probabilities"][-1]))
            )
            method["with_implementation_budget"] = output_metrics(
                final, values, target, field_metric, args.field_target,
                2 * implementation_allowance / (success - implementation_allowance),
            )
        summary["methods"][key] = method
        arrays[f"{key}_fields"] = fields
        arrays[f"{key}_moments"] = moments[indices]
        arrays[f"{key}_field_bias"] = field_bias
        arrays[f"{key}_probabilities"] = probabilities[indices]
        arrays[f"{key}_survival"] = survival[indices]

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "matched_output.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    np.savez_compressed(args.output / "matched_output.npz", **arrays)


if __name__ == "__main__":
    main()
