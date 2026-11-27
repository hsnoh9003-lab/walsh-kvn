"""Check the Novikau–Joseph 128-cell sine-drift example at dimensionless time 1.

Run with Python, NumPy and SciPy. To compare with the archived amplitudes, add
--qucf-dir followed by the QuCF simulations/LCHS/KvN directory and install h5py.

The spatial stencil, Fourier nodes and time splitting are Eqs. (25), (43),
(58) and (96) of https://arxiv.org/abs/2410.03985. The finite comparison uses
the direct weights and three weight-amplification rounds of QuCF revision
f34f89c, with exact matrix evolutions for the individual Hamiltonians.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import eigh, expm, norm, svdvals


def sine_generator():
    positions = np.linspace(-1.0, 1.0, 128)
    spacing = 2.0 / (len(positions) - 1)
    sine = np.sin(positions)
    drift = -sine * (sine**2 - 0.25)
    signs = np.sign(drift).astype(int)
    rows = np.arange(len(positions))
    neighbors = rows - signs
    valid = (neighbors >= 0) & (neighbors < len(positions))
    rows, neighbors = rows[valid], neighbors[valid]
    generator = np.diag(np.abs(drift) / spacing)
    generator[rows, neighbors] = (
        -signs[rows] * (drift[rows] + drift[neighbors]) / (2.0 * spacing)
    )
    return generator


def state_distance(exact, output):
    reference = exact / norm(exact)
    state = output / norm(output)
    relative_phase = np.angle(np.vdot(reference, state))
    return float(norm(np.exp(-1j * relative_phase) * state - reference))


def finite_circuit_model(generator, hermitian_part, initial, exact, shift):
    step_count = 10
    step_size = 1.0 / step_count
    cutoff = 40.967741935484
    nodes = np.linspace(-cutoff, cutoff, 128)
    raw_weights = 1.0 / (1.0 + nodes**2)
    weights = raw_weights / raw_weights.sum()
    preparation_rounds = 3
    angle = np.arcsin(np.sqrt(raw_weights.mean()))
    prepared_probability = np.sin((2 * preparation_rounds + 1) * angle) ** 2

    hamiltonian = (generator - generator.T) / (2j)
    half_step = expm(-0.5j * step_size * hamiltonian)
    eigenvalues, eigenvectors = eigh(hermitian_part)
    into_basis = eigenvectors.T @ half_step
    out_of_basis = half_step @ eigenvectors
    middle_step = into_basis @ out_of_basis
    node_phases = np.exp(-1j * step_size * eigenvalues[:, None] * nodes)

    # Each column evolves one Fourier branch for ten symmetric time steps.
    states = node_phases * (into_basis @ initial)[:, None]
    for _ in range(1, step_count):
        states = node_phases * (middle_step @ states)
    branches = out_of_basis @ states

    # Preparing and unpreparing the weights each supplies one success amplitude.
    results = {}
    for label, value in (("unshifted", 0.0), ("shifted", shift)):
        # exp(-i k s t) is exact for a scalar generator shift; t = 1.
        output = branches @ (weights * np.exp(-1j * nodes * value))
        results[label] = {
            "success_probability": float(prepared_probability**2 * norm(output) ** 2),
            "normalized_state_distance": state_distance(exact, output),
        }
    return results


def check_archive(qucf_dir, hermitian_part, exact):
    import h5py

    block_path = qucf_dir / "BE-3/case-SIN-LCU/Bk_OUTPUT.hdf5"
    with h5py.File(block_path) as archive:
        entries = archive["matrix/real"][()] + 1j * archive["matrix/imag"][()]
        block = entries.reshape(hermitian_part.shape).T
    scale = float(
        np.vdot(hermitian_part, block).real
        / np.vdot(hermitian_part, hermitian_part).real
    )

    with h5py.File(qucf_dir / "dt1/kvn_OUTPUT.hdf5") as archive:
        entries = archive["states/output-zero-anc-amplitudes-0"][()]
        amplitudes = entries["real"] + 1j * entries["imag"]
        bits = archive["states/output-zero-anc-states-0"][()].astype(np.int64)
        labels = (2 ** np.arange(bits.shape[0] - 1, -1, -1)) @ bits
        if not np.array_equal(np.sort(labels), np.arange(len(exact))):
            raise ValueError("Unexpected state labels in the ten-step output")
        amplitudes = amplitudes[np.argsort(labels)]
    block_spectrum = eigh((block + block.conj().T) / 2, eigvals_only=True)
    overlap = np.vdot(exact / norm(exact), amplitudes / norm(amplitudes))
    return {
        "positive_block_scale": scale,
        "minimum_hermitian_block_eigenvalue": float(block_spectrum[0]),
        "maximum_entry_discrepancy": float(
            np.max(np.abs(block - scale * hermitian_part))
        ),
        "unshifted_success_probability": float(norm(amplitudes) ** 2),
        "unshifted_normalized_state_distance": state_distance(exact, amplitudes),
        "unshifted_conditional_fidelity": float(abs(overlap) ** 2),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qucf-dir", type=Path, help="path to QuCF simulations/LCHS/KvN"
    )
    qucf_dir = parser.parse_args().qucf_dir
    generator = sine_generator()
    hermitian_part = (generator + generator.T) / 2
    spectrum = eigh(hermitian_part, eigvals_only=True)
    initial = np.ones(len(hermitian_part)) / np.sqrt(len(hermitian_part))
    propagator = expm(-generator)
    exact = propagator @ initial
    gain = float(svdvals(propagator)[0])
    shift = 0.005351
    results = {
        "negative_eigenvalues_of_hermitian_part": spectrum[spectrum < 0].tolist(),
        "operator_norm_of_propagator": gain,
        "original_cauchy_integral_absolute_operator_error_lower_bound": max(
            0.0, gain - 1.0
        ),
        "uniform_input_output_norm": float(norm(exact)),
        "shift": shift,
        "shifted_minimum_hermitian_eigenvalue": float(spectrum[0] + shift),
        "absolute_error_recovery_factor": float(np.exp(shift)),
        "finite_circuit_model": finite_circuit_model(
            generator, hermitian_part, initial, exact, shift
        ),
    }
    if qucf_dir is not None:
        results["qucf_archive"] = check_archive(qucf_dir, hermitian_part, exact)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
