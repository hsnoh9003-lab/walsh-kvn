"""Compile the audited Novikau–Joseph STMC inventory with one clean work bit.

The source uses the first control bits for closed controls and the remaining
control bits for open controls in the primitive-level verification. Each
lowered primitive is exact up to a global phase and bounded angle snapping.
"""

import argparse
from collections import Counter
import csv
from functools import lru_cache
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Source.KvN.one_clean_synthesis import clean_mcx, cx, expand, inverse, x


SOURCE = Path(__file__).resolve().parent / "published_t1_source"
QUARTER_PI = math.pi / 4
SNAP_TOLERANCE = 1e-12
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.diag([1, -1]).astype(complex)
I = np.eye(2, dtype=complex)
PAULIS = (X, Y, Z)
H = (X + Z) / math.sqrt(2)
S = np.diag([1, 1j])
T = np.diag([1, np.exp(1j * QUARTER_PI)])
CONJUGATIONS = {
    "X": (),
    "Z": ("h",),
    "Y": ("s",),
    "H": ("t", "h", "s"),  # time-ordered basis change B = S H T
}
ROTATION_BASES = {"rx": ("h",), "ry": ("h", "s"), "rz": ()}


def one(name, target, angle=None):
    return [(name, [target], angle)]


def basis(names, target):
    return [(name, [target], None) for name in names]


def rotation(axis, angle):
    return math.cos(angle / 2) * I - 1j * math.sin(angle / 2) * axis


def source_matrix(gate, angles):
    if gate in {"X", "Y", "Z", "H"}:
        return {"X": X, "Y": Y, "Z": Z, "H": H}[gate]
    if gate in {"Rx", "Ry", "Rz"}:
        return rotation(PAULIS["xyz".index(gate[-1])], angles[0])
    if gate == "Rc":
        return rotation(Y, angles[1]) @ rotation(Z, angles[0])
    if gate == "P0":
        return np.diag([np.exp(1j * angles[0]), 1])
    if gate in {"P", "S", "T"}:
        phi = {"S": math.pi / 2, "T": QUARTER_PI}.get(gate, angles[0] if angles else 0)
        return np.diag([1, np.exp(1j * phi)])
    raise ValueError(gate)


def controlled_rotation(gate, angle, controls, target, work):
    """An echo with two exact MCX gates and opposite half angles."""
    if not controls:
        return one(gate, target, angle)
    change = basis(ROTATION_BASES[gate], target)
    mcx = clean_mcx(controls, target, work)
    return (
        inverse(change)
        + one("rz", target, angle / 2)
        + mcx
        + one("rz", target, -angle / 2)
        + mcx
        + change
    )


def zb_rccx(a, b, target):
    """The Hermitian relative-phase Toffoli in Zindorf–Bose Eq. (6)."""
    return (
        one("h", target)
        + one("tdg", target)
        + cx(a, target)
        + one("t", target)
        + cx(b, target)
        + one("tdg", target)
        + cx(a, target)
        + one("t", target)
        + one("h", target)
    )


def zb_i_z(a, b, target):
    """CC(iZ), using the four-phase parity cycle of Eq. (7)."""
    return (
        one("tdg", target)
        + cx(a, target)
        + one("t", target)
        + cx(b, target)
        + one("tdg", target)
        + cx(a, target)
        + one("t", target)
        + cx(b, target)
    )


def zb_diagonal_block(controls, other_controls, targets):
    """One (Delta * multi-target CZ) block of Eqs. (2) and (10).

    The opposite control group provides borrowed bits for the relative-phase
    ladder. The ladder is uncomputed before returning. For two targets, the
    parity conjugation in Eq. (24) extends the CZ without a second ladder.
    """
    target = targets[0]
    if len(controls) == 1:
        block = one("h", target) + cx(controls[0], target) + one("h", target)
    else:
        borrowed = list(other_controls[: len(controls) - 2]) + [target]
        assert len(borrowed) == len(controls) - 1
        ladder = []
        for index in reversed(range(2, len(controls))):
            ladder += zb_rccx(controls[index], borrowed[index - 1], borrowed[index - 2])
        block = ladder + zb_i_z(*controls[:2], borrowed[0]) + inverse(ladder)
    if len(targets) == 2:
        parity = cx(targets[1], target)
        block = parity + block + parity
    return block


def su2_axis_angle(unitary):
    """Separate the U(2) scalar phase and express the SU(2) part as a rotation."""
    phase = float(np.angle(np.linalg.det(unitary)) / 2)
    special = unitary * np.exp(-1j * phase)
    cosine = float(np.trace(special).real / 2)
    sine_axis = np.array([-np.trace(pauli @ special).imag / 2 for pauli in PAULIS])
    sine = float(np.linalg.norm(sine_axis))
    if sine < 4 * np.finfo(float).eps:
        return phase, np.array([1.0, 0.0, 0.0]), 0.0 if cosine >= 0 else 2 * math.pi
    return phase, sine_axis / sine, 2 * math.atan2(sine, cosine)


def zb_basis_to_x(axis, target):
    """Return A4 in Eq. (2), satisfying A4† X A4 = axis · Pauli."""
    vx, vy, vz = axis
    if vy == 0 and vz == 0:
        return [] if vx > 0 else one("z", target)
    if vx == 0 and vz == 0:
        return one("sdg" if vy > 0 else "s", target)
    if vx == 0 and vy == 0:
        return (x(target) if vz < 0 else []) + one("h", target)
    theta1 = math.atan2(vz, -vy)
    theta2 = math.atan2(math.hypot(vy, vz), vx)
    return one("rx", target, theta1) + one("rz", target, theta2)


def zb_mcu2(phase, axis, angle, controls, target, work):
    """Zindorf–Bose Sec. II C, Lemma 10 and Eq. (10).

    For zero or one control, use exact base cases of the same determinant
    split. Use the single-target Eq. (2) when the determinant phase is zero.
    """
    change = zb_basis_to_x(axis, target)
    if not controls:
        return change + one("rx", target, angle) + inverse(change)
    if len(controls) == 1:
        return (
            change
            + controlled_rotation("rx", angle, controls, target, work)
            + inverse(change)
            + one("rz", controls[0], phase)
        )

    groups = controls[: len(controls) // 2], controls[len(controls) // 2 :]
    use_phase = phase != 0
    use_target = angle != 0
    targets = ([target] if use_target else []) + ([work] if use_phase else [])
    if not targets:
        return []
    blocks = [zb_diagonal_block(groups[i], groups[1 - i], targets) for i in range(2)]
    gates = (change if use_target else []) + (one("h", work) if use_phase else [])
    for step, sign in enumerate((-1, 1, -1, 1)):
        block = blocks[step % 2]
        gates += block if step < 2 else inverse(block)
        if use_target:
            gates += one("rx", target, sign * angle / 4)
        if use_phase and step < 3:
            gates += one("rx", work, -sign * phase / 2)
    # The fourth ancilla quarter rotation is a scalar on its restored
    # |0> sector after conjugation through the final H.
    return (
        gates
        + (inverse(change) if use_target else [])
        + (one("h", work) if use_phase else [])
    )


def controlled_source(gate, npositive, nnegative, angles):
    controls = list(range(npositive + nnegative))
    target, work = len(controls), len(controls) + 1
    open_controls = sum((x(q) for q in controls[npositive:]), [])
    if gate in CONJUGATIONS:
        if controls:
            change = basis(CONJUGATIONS[gate], target)
            gates = inverse(change) + clean_mcx(controls, target, work) + change
        else:
            gates = one(gate.lower(), target)
    elif gate in {"Rx", "Ry", "Rz"}:
        gates = controlled_rotation(gate.lower(), angles[0], controls, target, work)
    elif gate in {"P", "P0", "S", "T"}:
        phi = {"S": math.pi / 2, "T": QUARTER_PI}.get(gate, angles[0] if angles else 0)
        theta = -phi if gate == "P0" else phi
        if controls:
            gates = zb_mcu2(phi / 2, (0, 0, 1), theta, controls, target, work)
        else:
            gates = one("rz", target, theta)
    elif gate == "Rc":
        # QuCF applies Rz first, then Ry. Both factors have determinant one.
        _, axis, angle = su2_axis_angle(source_matrix(gate, angles))
        gates = zb_mcu2(0.0, axis, angle, controls, target, work)
    else:
        raise ValueError(gate)
    return open_controls + gates + inverse(open_controls)


def lower_to_basis(gates):
    """Emit Clifford, T/T† and arbitrary Rz, with a norm bound for snapping."""
    compiled, snap_bound = [], 0.0
    exact_phases = (
        (),
        ("t",),
        ("s",),
        ("s", "t"),
        ("z",),
        ("z", "t"),
        ("sdg",),
        ("tdg",),
    )
    for name, qubits, angle in expand(gates):
        if name not in ROTATION_BASES:
            compiled.append((name, qubits, angle))
            continue
        (target,) = qubits
        multiple = round(angle / QUARTER_PI)
        discrepancy = abs(angle - multiple * QUARTER_PI)
        if discrepancy <= SNAP_TOLERANCE:
            core = basis(exact_phases[multiple % 8], target)
            snap_bound += discrepancy / 2
        else:
            core = one("rz", target, angle)
        if core:
            change = basis(ROTATION_BASES[name], target)
            compiled += inverse(change) + core + change
    return compiled, snap_bound


@lru_cache(maxsize=None)
def target_pairs(width, qubits):
    bits = np.arange(1 << width)
    target = qubits[-1]
    keep = ((bits >> target) & 1) == 0
    for control in qubits[:-1]:
        keep &= ((bits >> control) & 1) == 1
    lo = bits[keep]
    return lo, lo + (1 << target)


def evolve(gates, inputs):
    output = inputs.copy()
    width = output.shape[0].bit_length() - 1
    matrices = {
        "x": X,
        "y": Y,
        "z": Z,
        "h": H,
        "s": S,
        "sdg": S.conj(),
        "t": T,
        "tdg": T.conj(),
    }
    for name, qubits, angle in gates:
        lo, hi = target_pairs(width, tuple(qubits))
        if name == "cx":
            output[lo], output[hi] = output[hi].copy(), output[lo].copy()
        elif name == "rz":
            phase = np.exp(-0.5j * angle)
            output[lo] *= phase
            output[hi] *= phase.conjugate()
        else:
            matrix = matrices[name]
            a, b = output[lo].copy(), output[hi].copy()
            output[lo] = matrix[0, 0] * a + matrix[0, 1] * b
            output[hi] = matrix[1, 0] * a + matrix[1, 1] * b
    return output


def verify_primitive(compiled, unitary, npositive, nnegative, rng):
    """Check both target basis columns on coherent mixed-control inputs."""
    n = npositive + nnegative
    stride = 1 << n
    initial = np.zeros((4 * stride, 2), dtype=complex)
    for target in range(2):
        random = rng.normal(size=stride) + 1j * rng.normal(size=stride)
        initial[target * stride : (target + 1) * stride, target] = (
            random / np.linalg.norm(random)
        )
    selected = (1 << npositive) - 1
    positions = [selected, selected + stride]
    expected = initial.copy()
    expected[positions] = unitary @ initial[positions]
    actual = evolve(compiled, initial)
    overlap = np.vdot(expected, actual)
    error = float(np.linalg.norm(actual - expected * overlap / abs(overlap)))
    leakage = float(np.linalg.norm(actual[2 * stride :]))
    assert error < 3e-10 and leakage < 3e-10, (npositive, nnegative, error, leakage)
    return error, leakage


def verify_general_cases(rng):
    errors = []
    for n in range(13):
        # A general U(2) with both a nontrivial rotation and determinant phase.
        unitary = np.exp(0.317j) * rotation(Y, -0.846) @ rotation(Z, 1.136)
        phase, axis, angle = su2_axis_angle(unitary)
        elementary, _ = lower_to_basis(
            zb_mcu2(phase, axis, angle, list(range(n)), n, n + 1)
        )
        errors.append(verify_primitive(elementary, unitary, n, 0, rng)[0])
    for gate in ("S", "T", "Y", "Rx"):
        angles = (0.73,) if gate == "Rx" else ()
        for n in (0, 1, 2, 5, 8):
            p, z = n // 2, n - n // 2
            elementary, _ = lower_to_basis(controlled_source(gate, p, z, angles))
            errors.append(
                verify_primitive(elementary, source_matrix(gate, angles), p, z, rng)[0]
            )
    return max(errors)


def summarize_case(path, reference, cache, rng, verify):
    totals, by_gate, by_kind = Counter(), {}, Counter()
    snap_bound = 0.0
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            name = row["gate"]
            positive, negative = int(row["positive_controls"]), int(
                row["negative_controls"]
            )
            angles = tuple(json.loads(row["angles"]))
            key = name, positive, negative, angles
            if key not in cache:
                gates, error_bound = lower_to_basis(controlled_source(*key))
                inventory = Counter(name for name, _, _ in gates)
                err, leak = (
                    verify_primitive(
                        gates, source_matrix(name, angles), positive, negative, rng
                    )
                    if verify
                    else (None, None)
                )
                cache[key] = (inventory, error_bound, err, leak)
            inventory, error_bound, _, _ = cache[key]
            multiplicity = int(row["multiplicity"])
            assert multiplicity > 0
            weighted = {g: count * multiplicity for g, count in inventory.items()}
            totals.update(weighted)
            by_gate.setdefault(name, Counter()).update(weighted)
            by_kind[name] += multiplicity
            snap_bound += error_bound * multiplicity
    assert sum(by_kind.values()) == reference["STMC_total"]
    assert by_kind == reference["raw_gate_counts"]
    return {
        "simulation_time": reference["physical_simulation_time"],
        "time_steps": reference["time_steps"],
        "logical_qubits": reference["logical_qubits"] + 1,
        "algorithm_logical_qubits": reference["logical_qubits"],
        "reference_STMC_total": reference["STMC_total"],
        "RZ": totals["rz"],
        "direct_T": totals["t"] + totals["tdg"],
        "Clifford": sum(totals.values()) - totals["rz"] - totals["t"] - totals["tdg"],
        "CX": totals["cx"],
        "canonicalized_source_angle_operator_error_bound": snap_bound,
        "success_probability_archived": reference["success_probability_archived"],
        "by_basis_gate": dict(totals),
        "by_source_gate": {g: dict(counts) for g, counts in by_gate.items()},
        "source_histogram_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--count-only",
        action="store_true",
        help="Skip numerical checks for a quick recount.",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=SOURCE,
        help="Read the histograms and write the gate audit in this directory.",
    )
    args = parser.parse_args()
    references = json.loads((args.source_dir / "source_count_audit.json").read_text())[
        "cases"
    ]
    rng = np.random.default_rng(20261107)
    cache = {}
    cases = [
        summarize_case(
            args.source_dir / f"{ref['case']}_stmc_histogram.csv",
            ref,
            cache,
            rng,
            not args.count_only,
        )
        for ref in references
    ]
    report = {
        "compilation": "shared one-clean Khattar–Gidney MCX and Zindorf–Bose ATA MCU(2)",
        "extra_logical_qubits": 1,
        "numerical_verification_performed": not args.count_only,
        "clean_work_restoration": (
            "not checked (--count-only)"
            if args.count_only
            else "tested per distinct source primitive in the supplied workloads"
        ),
        "initial_hadamards": "seven, already included in each source histogram",
        "assembly": "multiplicity-weighted inventory with no cross-primitive gate cancellation",
        "angle_snap_tolerance_rad": SNAP_TOLERANCE,
        "primitive_types": len(cache),
        "max_state_l2_error": (
            max(v[2] for v in cache.values()) if not args.count_only else None
        ),
        "max_clean_ancilla_leakage_l2": (
            max(v[3] for v in cache.values()) if not args.count_only else None
        ),
        "max_supplemental_primitive_state_l2_error": (
            verify_general_cases(rng) if not args.count_only else None
        ),
        "cases": cases,
    }
    (args.source_dir / "one_clean_gate_audit.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    for r in cases:
        print(
            {
                k: r[k]
                for k in (
                    "simulation_time",
                    "logical_qubits",
                    "reference_STMC_total",
                    "RZ",
                    "direct_T",
                    "Clifford",
                    "CX",
                )
            }
        )
    print(
        "Max verification error:",
        report["max_state_l2_error"],
        "max work leakage:",
        report["max_clean_ancilla_leakage_l2"],
    )


if __name__ == "__main__":
    main()
