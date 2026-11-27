"""Compile a full-window joint block retaining the Guseynov coordinate loader."""

# Direct-file execution must locate Codes/ independently of the working directory.
if __package__ in (None, ""):
    import sys as _path_sys
    from pathlib import Path as _Path
    _codes = next(p for p in _Path(__file__).resolve().parents
                  if (p / "Source/data_locations.py").is_file())
    _path_sys.path.insert(0, str(_codes))



from concurrent.futures import ProcessPoolExecutor
import json
from math import atan2, pi, sqrt
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "literal"))
from Source.data_locations import generated_outputs_root

from compile_loader import (
    amplitude_oracle,
    compiler,
    controlled,
    inverse,
    make_circuit,
    op,
    simulate,
)


N = 64
N_POSITION = 6
MU = 0.09248186783176017
GAMMA_NORMALIZED = 0.30173365805150404
PHASE_RATE = GAMMA_NORMALIZED / MU


def build():
    radial, _, _, alpha_radial, radial_width = make_circuit(
        N_POSITION, literal_scale=True
    )
    n = N_POSITION
    address, work = list(range(n)), list(range(n, 2 * n))
    internal_probe, signal, real, occupation, which = range(2 * n, 2 * n + 5)
    scale = radial_width - 1
    probe_a, probe_b, selector = range(radial_width, radial_width + 3)
    radius = (np.arange(N) + 0.5) / N

    # Apply the same Corollary 1 coordinate oracle to the probe polynomial.
    # Its successful signal is z=r/(sqrt(2)*||r||_2).
    coordinate = amplitude_oracle(
        address, work, internal_probe, signal, scale, radius, literal_scale=True
    )
    projection = ((signal, 0), (scale, 0))
    theta = -pi / 4
    square_inner = (
        coordinate
        + [op("rz", real, angle=2 * theta), op("rz", real, projection, -4 * theta)]
        + inverse(coordinate)
        + [op("rz", real, angle=-pi / 2)]
    )
    alpha_square = 2.0 * float(radius @ radius)

    # The probe-only LCU selects -I twice and then Z_a and Z_b once each.
    # Its successful block is -(n_a+n_b)/2, using the radial shift workspace.
    occ_prep = [op("h", occupation), op("h", which)]
    occ_select = [
        op("global", -1, ((occupation, 0),), pi),
        op("p", probe_a, ((occupation, 1), (which, 0)), pi),
        op("p", probe_b, ((occupation, 1), (which, 1)), pi),
    ]
    alpha_diagonal = 2 * PHASE_RATE * alpha_square
    branch_prep = [
        op("ry", selector, angle=2 * atan2(sqrt(alpha_diagonal), sqrt(alpha_radial)))
    ]
    width = selector + 1

    def assemble(outer_control=()):
        diagonal_control = ((selector, 1),) + outer_control
        radial_control = ((selector, 0),) + outer_control
        return (
            branch_prep
            + occ_prep
            + controlled(occ_select, *diagonal_control)
            + inverse(occ_prep)
            + [op("h", real)]
            + controlled(square_inner, *diagonal_control)
            + [op("h", real)]
            + controlled(radial, *radial_control)
            + inverse(branch_prep)
        )

    return (
        assemble(),
        assemble(((width, 1),)),
        width,
        (probe_a, probe_b),
        alpha_radial,
        alpha_diagonal,
        alpha_radial + alpha_diagonal,
    )


def encode(column, probes):
    radial_cell = column % N
    occupation = column // N
    return radial_cell | sum(
        ((occupation >> k) & 1) << bit for k, bit in enumerate(probes)
    )


def verify_column(arguments):
    circuit, width, probes, alpha, column, outer = arguments
    initial = encode(column, probes)
    if outer == "on":
        initial |= 1 << width
    state = simulate(circuit, initial)
    if outer == "off":
        return max(
            abs(state.get(i, 0.0) - int(i == initial)) for i in {*state, initial}
        )

    system_mask = (N - 1) | sum(1 << bit for bit in probes)
    if outer == "on":
        system_mask |= 1 << width
    radius_index = column % N
    occ = (column // N).bit_count()
    radius = (radius_index + 0.5) / N
    expected = {initial: -PHASE_RATE * occ * radius**2}
    if radius_index < N - 1:
        face = (radius_index + 1) / N
        expected[initial + 1] = 0.5j * N * (face - face**3)
    if radius_index > 0:
        face = radius_index / N
        expected[initial - 1] = -0.5j * N * (face - face**3)
    projected = {i: alpha * value for i, value in state.items() if i & system_mask == i}
    return max(
        abs(projected.get(i, 0.0) - expected.get(i, 0.0))
        for i in projected.keys() | expected.keys()
    )


def full_verification(circuit, controlled_circuit, width, probes, alpha):
    checks = [(circuit, width, probes, alpha, column, None) for column in range(4 * N)]
    active_spots = [0, 63, 64, 87, 128, 177, 192, 255]
    checks += [
        (controlled_circuit, width, probes, alpha, column, "on")
        for column in active_spots
    ]
    checks += [
        (controlled_circuit, width, probes, alpha, column, "off")
        for column in active_spots
    ]
    with ProcessPoolExecutor(max_workers=4) as pool:
        errors = list(pool.map(verify_column, checks, chunksize=8))
    assert max(errors) < 2e-9
    return {
        "all_projected_columns": 4 * N,
        "max_physical_entry_error": max(errors[: 4 * N]),
        "controlled_on_columns": len(active_spots),
        "controlled_on_max_physical_entry_error": max(
            errors[4 * N : 4 * N + len(active_spots)]
        ),
        "controlled_off_columns": len(active_spots),
        "controlled_off_max_amplitude_error": max(errors[4 * N + len(active_spots) :]),
    }


def main():
    circuit, control, width, probes, alpha_radial, alpha_diagonal, alpha = build()
    forward, forward_width = compiler(circuit, width)
    controlled_forward, controlled_width = compiler(control, width + 1)
    report = {
        "variant": "Guseynov Corollary 1 coordinate loader for radial and probe polynomials",
        "normalization_alpha": alpha,
        "radial_normalization": alpha_radial,
        "diagonal_normalization": alpha_diagonal,
        "system_qubits": N_POSITION + len(probes),
        "projector_qubits": width - N_POSITION - len(probes),
        "macro_width": width,
        "ordinary": {**forward, "logical_qubits": forward_width},
        "controlled": {**controlled_forward, "logical_qubits": controlled_width},
    }
    print(json.dumps(report, indent=2), flush=True)
    report["verification"] = full_verification(circuit, control, width, probes, alpha)
    output = generated_outputs_root() / "BE_KvN/literal_joint_counts.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["verification"], indent=2), flush=True)


if __name__ == "__main__":
    main()
