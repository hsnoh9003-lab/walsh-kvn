"""Compile the Guseynov-style amplitude-loader using the LCHS one-clean gates.

The signed-shift and order selectors specialize its sparse/LCU construction.
The same one-clean primitive decompositions as KvN–LCHS are reused; there is
no adjacent-gate cancellation pass or multiple-clean-ancilla ladder.
"""

from collections import Counter
from dataclasses import dataclass
from itertools import product
import json
from math import acos, atan2, pi, sqrt
from pathlib import Path
import sys

import numpy as np

# Reuse the exact, verified one-clean STMC lowering used by KvN–LCHS.
# This module is intentionally not copied or reimplemented in the BE compiler.
PACKAGE_ROOT = Path(__file__).resolve().parents[3]
LCHS_COMPILER = PACKAGE_ROOT / "Comparisons/KvN_LCHS"
if str(LCHS_COMPILER) not in sys.path:
    sys.path.insert(0, str(LCHS_COMPILER))
from compile_stmc_one_clean import controlled_source, lower_to_basis


@dataclass(frozen=True)
class Gate:
    name: str
    target: int
    controls: tuple = ()
    angle: float = 0.0


def op(name, target, controls=(), angle=0.0):
    return Gate(name, target, tuple(controls), angle)


def controlled(gates, *extra):
    return [op(g.name, g.target, g.controls + extra, g.angle) for g in gates]


def inverse(gates):
    return [op(g.name, g.target, g.controls, -g.angle) for g in reversed(gates)]


def zero_phase(qubits, angle, extra=()):
    *rest, target = qubits
    return [
        op("x", target),
        op("p", target, extra + tuple((q, 0) for q in rest), angle),
        op("x", target),
    ]


def amplitude_state(work, x):
    n = len(work)
    N = 1 << n
    amplitudes = x / np.linalg.norm(x)
    zero = np.sum(amplitudes) / sqrt(N)
    coefficients = [
        np.dot(amplitudes, 1 - 2 * ((np.arange(N) >> j) & 1)) / sqrt(N)
        for j in range(n)
    ]
    gates = []
    for j, coefficient in enumerate(coefficients):
        remain = sqrt(zero * zero + sum(v * v for v in coefficients[j + 1 :]))
        theta = 2 * atan2(coefficient, remain)
        gates.append(op("ry", work[j], tuple((q, 0) for q in work[:j]), theta))
    return gates + [op("h", q) for q in work]


def amplitude_oracle(address, work, probe, signal, scale, x, literal_scale):
    # Lemma 2 with p=0. W†GW = R D and W†G†W = D R for D=W† Z W.
    w = [op("h", probe)]
    w += controlled(amplitude_state(work, x), (probe, 0))
    w += [op("x", wj, ((probe, 1), (aj, 1))) for wj, aj in zip(work, address)]
    w += [op("h", probe)]
    result = [op("h", signal)]
    result += zero_phase(work + [probe], pi, ((signal, 1),))
    result += w + [op("p", probe, angle=pi)] + inverse(w)
    result += zero_phase(work + [probe], pi, ((signal, 0),))
    result += [
        op("h", signal),
        op("x", signal),
        op("p", signal, angle=pi),
        op("x", signal),
    ]
    if literal_scale:
        result.append(op("h", scale))
    return result


def polynomial_block(address, work, probe, signal, scale, real, x, h, literal_scale):
    # F=x-x³. The small correction makes (G_j+G_{j+1})/2 exactly F(midpoint).
    linear = 1.0 + 3.0 * h * h / 4.0
    stretch = np.linalg.norm(x) * (sqrt(2.0) if literal_scale else 1.0)
    maximum = max(stretch**3 - linear * stretch, 2 * linear**1.5 / (3 * sqrt(3.0)))
    # |a*z+b*z³|≤1 for z in [-1,1]. The real part of the three-query
    # alternating sequence is z[(2cos A−2cos D)z²+2cos D−cos A].
    a, b = linear * stretch / maximum, -(stretch**3) / maximum
    angle_sum, angle_diff = acos(np.clip(a + b, -1.0, 1.0)), acos(
        np.clip(a + b / 2.0, -1.0, 1.0)
    )
    angles = [(angle_sum + angle_diff) / 2.0, (angle_sum - angle_diff) / 2.0]
    signal_bits = [signal] + ([scale] if literal_scale else [])
    u = amplitude_oracle(address, work, probe, signal, scale, x, literal_scale)

    def real_phase(theta):
        # Rz on real gives opposite global phase in its two branches;
        # invert that phase on the block projector for the QSVT signal.
        return [
            op("rz", real, angle=2.0 * theta),
            op("rz", real, tuple((q, 0) for q in signal_bits), -4.0 * theta),
        ]

    out = [op("h", real)] + u
    out += real_phase(angles[0]) + inverse(u) + real_phase(angles[1]) + u
    out += [op("h", real)]
    return out, maximum


def increment(address, external, forward):
    carry = int(forward)
    return [
        op("x", address[j], tuple(external) + tuple((q, carry) for q in address[:j]))
        for j in range(len(address) - 1, -1, -1)
    ]


def momentum_block(address, direction, boundary):
    out = [
        op("h", direction),
        op("p", direction, angle=pi),
        op("global", -1, angle=pi / 2),
    ]
    out += [
        op("x", boundary, ((direction, 0),) + tuple((q, 1) for q in address)),
        op("x", boundary, ((direction, 1),) + tuple((q, 0) for q in address)),
    ]
    out += increment(address, [(direction, 0)], True)
    out += increment(address, [(direction, 1)], False)
    return out + [op("h", direction)]


def make_circuit(n, literal_scale):
    N = 1 << n
    h = 1 / N
    x = (np.arange(N) + 0.5) * h
    address, work = list(range(n)), list(range(n, 2 * n))
    probe, signal, real, direction, boundary, sym = range(2 * n, 2 * n + 6)
    scale = sym + 1 if literal_scale else None
    width = sym + 1 + int(literal_scale)
    f, f_alpha = polynomial_block(
        address, work, probe, signal, scale, real, x, h, literal_scale
    )
    p = momentum_block(address, direction, boundary)
    circuit = (
        [op("h", sym)]
        + controlled(p, (sym, 0))
        + f
        + controlled(p, (sym, 1))
        + [op("h", sym)]
    )
    return circuit, f, p, f_alpha / h, width


def compiler(gates, width, *, return_primitives=False):
    """Lower BE macros via the KvN–LCHS one-clean STMC decomposition.

    The algorithm wires occupy [0,width); *one* additional clean work bit is
    shared across all primitive gates and is restored after every operation.
    Positive and open controls are permuted into the LCHS control convention,
    then the resulting elementary wires are mapped back to the BE register.
    """
    work = width
    primitive = []
    snap_bound = 0.0
    names = {"x": "X", "h": "H", "ry": "Ry", "rz": "Rz", "p": "P"}
    for gate in gates:
        controls = list(gate.controls)
        target = gate.target
        if gate.name == "global":
            if not controls:
                continue  # Only a physically irrelevant global scalar.
            # A phase applied when *all* controls match is a controlled P/P0
            # on one of the controls; this preserves relative branch phases.
            target, active = controls[-1]
            controls = controls[:-1]
            source_name = "P" if active else "P0"
        else:
            source_name = names[gate.name]
        positives = [q for q, value in controls if value == 1]
        negatives = [q for q, value in controls if value == 0]
        physical = positives + negatives + [target, work]
        angles = (gate.angle,) if source_name in {"Ry", "Rz", "P", "P0"} else ()
        lowered, bound = lower_to_basis(
            controlled_source(source_name, len(positives), len(negatives), angles)
        )
        snap_bound += bound
        primitive.extend(
            (name, [physical[q] for q in wires], angle)
            for name, wires, angle in lowered
        )

    if return_primitives:
        # The sparse BE oracle simulator supports X/H/Ry/Rz/P. Represent
        # exact Clifford/T phases by the corresponding exact P angle.
        phases = {"z": pi, "s": pi / 2, "sdg": -pi / 2, "t": pi / 4, "tdg": -pi / 4}
        output = []
        for name, wires, angle in primitive:
            if name == "cx":
                output.append(op("x", wires[1], ((wires[0], 1),)))
            elif name in phases:
                output.append(op("p", wires[0], angle=phases[name]))
            else:
                output.append(op(name, wires[0], angle=angle or 0.0))
        return output

    counts = Counter(name for name, _, _ in primitive)
    return {
        "RZ": counts["rz"],
        "native_T": counts["t"] + counts["tdg"],
        "Clifford": sum(counts.values()) - counts["rz"] - counts["t"] - counts["tdg"],
        "source_angle_error_bound": snap_bound,
    }, width + 1


def simulate(gates, initial):
    """Sparse macro replay, keeping phases including controlled global phases."""
    state = {initial: 1.0 + 0j}
    for g in gates:
        mask = sum(1 << q for q, _ in g.controls)
        required = sum(v << q for q, v in g.controls)
        updated = {}
        if g.name == "ry":
            c, s = np.cos(g.angle / 2), np.sin(g.angle / 2)
        for i, a in state.items():
            if i & mask != required:
                updated[i] = updated.get(i, 0.0) + a
                continue
            if g.name == "global":
                updated[i] = updated.get(i, 0.0) + a * np.exp(1j * g.angle)
                continue
            bit = (i >> g.target) & 1
            flipped = i ^ (1 << g.target)
            if g.name == "x":
                updated[flipped] = updated.get(flipped, 0.0) + a
            elif g.name == "h":
                updated[i] = updated.get(i, 0.0) + (1 - 2 * bit) * a / sqrt(2.0)
                updated[flipped] = updated.get(flipped, 0.0) + a / sqrt(2.0)
            elif g.name == "ry":
                updated[i] = updated.get(i, 0.0) + c * a
                updated[flipped] = updated.get(flipped, 0.0) + (1 - 2 * bit) * s * a
            elif g.name == "rz":
                updated[i] = updated.get(i, 0.0) + a * np.exp(
                    1j * g.angle * (bit - 0.5)
                )
            elif g.name == "p":
                updated[i] = updated.get(i, 0.0) + a * np.exp(1j * g.angle * bit)
            else:
                raise ValueError(g.name)
        state = {i: a for i, a in updated.items() if abs(a) > 1e-14}
    return state


def verify_compiler():
    max_error = 0.0
    cases = 0
    for name, target, angle in [
        ("x", 0, 0.0),
        ("h", 0, 0.0),
        ("rz", 0, 1.17),
        ("ry", 0, 1.17),
        ("p", 0, 1.17),
        ("global", -1, 1.17),
    ]:
        for n in range(5):
            for signs in product((0, 1), repeat=n):
                width = n + 1
                controls = tuple(zip(range(1, width), signs))
                macro = [op(name, target, controls, angle)]
                elementary = compiler(macro, width, return_primitives=True)
                phase = None
                for column in range(1 << width):
                    a, b = simulate(macro, column), simulate(elementary, column)
                    if phase is None:
                        k = max(a, key=lambda i: abs(a[i]))
                        phase = b[k] / a[k]
                    error = max(
                        abs(b.get(k, 0.0) - phase * a.get(k, 0.0))
                        for k in a.keys() | b.keys()
                    )
                    max_error = max(max_error, error)
                    cases += 1
    assert max_error < 1e-12
    return dict(compiler_basis_cases=cases, compiler_max_error=max_error)


def verify(n, circuit, polynomial, momentum, alpha):
    N = 1 << n
    h = 1 / N
    faces = np.arange(1, N) * h
    vel = faces - faces**3
    K = np.diag(vel / (2 * h), -1) - np.diag(vel / (2 * h), 1)
    # Full block verification over all input basis columns.
    checked = list(range(N))
    largest_error = 0.0
    for j in checked:
        state = simulate(circuit, j)
        actual = np.array([state.get(i, 0.0) for i in range(N)])
        error = np.linalg.norm(alpha * actual - 1j * K[:, j])
        largest_error = max(largest_error, error)
    # Exhaustive diagonal coefficient check.
    x = (np.arange(N) + 0.5) * h
    target = x - x**3 + 3 * h**2 * x / 4
    diagonal = np.array([simulate(polynomial, j).get(j, 0.0) for j in range(N)])
    diagonal_error = np.max(np.abs(alpha * h * diagonal - target))
    momentum_columns = [simulate(momentum, j) for j in range(N)]
    p = np.array(
        [[column.get(i, 0.0) for i in range(N)] for column in momentum_columns]
    ).T
    p_error = np.max(np.abs(p - 0.5j * (np.eye(N, k=-1) - np.eye(N, k=1))))
    assert largest_error < 1e-9 and diagonal_error < 1e-10 and p_error < 1e-12
    return dict(
        full_matrix_columns_checked=len(checked),
        full_matrix_column_error=largest_error,
        all_coordinate_diagonal_error=diagonal_error,
        all_momentum_matrix_error=p_error,
        physical_norm=np.linalg.norm(K, ord=2),
    )


def main():
    n = 6
    primitive_verification = verify_compiler()
    results = []
    for literal_scale in (False, True):
        circ, f, p, alpha, width = make_circuit(n, literal_scale)
        gates, total_qubits = compiler(circ, width)
        controlled_gates, controlled_width = compiler(
            controlled(circ, (width, 1)), width + 1
        )
        actual = verify(n, circ, f, p, alpha)
        result = dict(
            variant="Corollary 1 scale" if literal_scale else "real-only Lemma 2",
            alpha=alpha,
            forward=gates,
            controlled_forward=controlled_gates,
            total_qubits=total_qubits,
            controlled_total_qubits=controlled_width,
            macro_width=width,
            macro_gates=len(circ),
            notes="64 midpoint faces exactly, no wrap, sparse signed-shift specialization; no cancellation, no arbitrary-angle synthesis",
            **primitive_verification,
            **actual,
        )
        results.append(result)
        print(json.dumps(result), flush=True)
    Path(__file__).with_name("literal_block_counts.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
