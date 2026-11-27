"""Compile finite-bond real MPS isometries into exact elementary gates.

Each isometry is reduced by real Givens rotations between adjacent Gray-code
basis states. A rotation is a pattern-controlled Ry, compiled with two Ry
rotations and two exact clean-work MCX operations from the archived compiler.
Work qubits are reused with the transport circuit. Counts include every gate;
no angle pruning or cancellation is used.
"""

# Direct-file execution must locate Codes/ independently of the working directory.
if __package__ in (None, ""):
    import sys as _path_sys
    from pathlib import Path as _Path
    _codes = next(p for p in _Path(__file__).resolve().parents
                  if (p / "Source/data_locations.py").is_file())
    _path_sys.path.insert(0, str(_codes))


from Source.data_locations import generated_outputs_root

from pathlib import Path
import os, sys, json, time, argparse, gzip
from collections import Counter

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import numpy as np

ROOT = generated_outputs_root() / "KvN"
from . import wake_gate_core as C


def ry_program(q, theta):
    """Express Ry(theta), with theta in radians, in the Rz/Clifford gate set."""
    return [
        ("sdg", [q], None),
        ("h", [q], None),
        ("rz", [q], float(theta)),
        ("h", [q], None),
        ("s", [q], None),
    ]


def controlled_rotation(w, a, b, theta, work_offset=None):
    """Rotate between two basis states differing in exactly one register bit."""
    target = (a ^ b).bit_length() - 1
    assert a ^ b == 1 << target
    controls = [q for q in range(w) if q != target]
    zeros = [q for q in controls if not (a >> q) & 1]
    angle = theta if not (a >> target) & 1 else -theta
    flags = list(
        range(
            w if work_offset is None else work_offset,
            (w if work_offset is None else work_offset)
            + min(1, max(0, len(controls) - 2)),
        )
    )
    x = [("x", [q], None) for q in zeros]
    mcx = C.mcx(controls, target, flags)
    return (
        x
        + ry_program(target, angle / 2)
        + mcx
        + ry_program(target, -angle / 2)
        + mcx
        + x
    )


def decompose_isometry(u):
    """Reduce a real (2**w, rank) isometry with adjacent Gray-order Givens rotations."""
    d, rank = u.shape
    w = d.bit_length() - 1
    assert 1 << w == d and rank <= d // 2
    gray = np.array([i ^ (i >> 1) for i in range(d)])
    r = u[gray].copy()
    rot = []
    for j in range(rank):
        for i in range(d - 1, j, -1):
            a, b = float(r[i - 1, j]), float(r[i, j])
            if b == 0.0:
                continue
            radius = np.hypot(a, b)
            c, s = a / radius, b / radius
            x, y = r[i - 1, j:].copy(), r[i, j:].copy()
            r[i - 1, j:] = c * x + s * y
            r[i, j:] = -s * x + c * y
            rot.append((int(gray[i - 1]), int(gray[i]), float(-2 * np.arctan2(s, c))))
        if r[j, j] < 0:
            r[j : j + 2] *= -1
            rot.append((int(gray[j]), int(gray[j + 1]), float(2 * np.pi)))
    expected = np.zeros_like(r)
    expected[:rank] = np.eye(rank)
    error = float(np.linalg.norm(r - expected))
    assert error < 2e-10, error
    # Reapply the recorded logical rotations with independent physical indexing.
    actual = u.copy()
    for a, b, theta in rot:
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        x, y = actual[a].copy(), actual[b].copy()
        actual[a] = c * x - s * y
        actual[b] = s * x + c * y
    actual = actual[gray]
    replay_error = float(np.linalg.norm(actual - expected))
    assert replay_error < 2e-10, replay_error
    # Inverse Gray map g -> binary, in temporal order.
    gray_inverse = [("cx", [q + 1, q], None) for q in reversed(range(w - 1))]
    return rot, gray_inverse, replay_error


def uniform_ry(angles, controls, target):
    """Compile a multiplexed Ry angle table without angle pruning."""
    coeff = C.walsh(angles)
    terms = []
    current = 0
    for i in range(len(coeff)):
        mask = i ^ (i >> 1)
        changed = current ^ mask
        for j, q in enumerate(controls):
            if (changed >> j) & 1:
                terms.append(("cx", [q, target], None))
        if coeff[mask] != 0.0:
            terms.append(("rz", [target], float(coeff[mask])))
        current = mask
    for j, q in enumerate(controls):
        if (current >> j) & 1:
            terms.append(("cx", [q, target], None))
    return (
        [("sdg", [target], None), ("h", [target], None)]
        + terms
        + [("h", [target], None), ("s", [target], None)]
    )


def base_preparation(amplitude):
    """Compile and verify preparation of the real MPS remainder state."""
    state = amplitude.copy()
    n = state.size.bit_length() - 1
    disentangle = []
    for target in range(n):
        stride = 1 << target
        x = state[:: 2 * stride].copy()
        y = state[stride :: 2 * stride].copy()
        angles = -2 * np.arctan2(y, x)
        disentangle += uniform_ry(angles, list(range(target + 1, n)), target)
        state[:: 2 * stride] = np.hypot(x, y)
        state[stride :: 2 * stride] = 0.0
    program = C.inverse(disentangle)
    initial = np.zeros(1 << n, complex)
    initial[0] = 1.0
    actual = C.apply_gates(initial, program, n)
    error = float(np.linalg.norm(actual - amplitude))
    assert error < 1e-10, error
    return program, error


def primitive_check(w):
    # The whole input sector is tested, with every work qubit initially zero.
    a = 0 if w < 7 else 0b01010100
    a &= (1 << w) - 1
    b = a ^ 1
    theta = 0.734221
    gates = controlled_rotation(w, a, b, theta)
    n = w + min(1, max(0, w - 3))
    dim = 1 << w
    state = np.zeros((1 << n, dim), complex)
    state[np.arange(dim), np.arange(dim)] = 1.0
    actual = C.apply_gates(state, gates, n)
    expected = state.copy()
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    x, y = state[a].copy(), state[b].copy()
    expected[a] = c * x - s * y
    expected[b] = s * x + c * y
    error = float(np.linalg.norm(actual - expected))
    leakage = float(np.linalg.norm(actual[dim:]))
    assert error < 2e-10 and leakage < 2e-10, (w, error, leakage)
    return dict(
        window_qubits=w,
        logical_input_columns=dim,
        work_qubits=n - w,
        frobenius_error=error,
        clean_work_leakage=leakage,
    )


def main(name, directory=None):
    out = Path(directory) if directory is not None else ROOT / "bond_sweep" / name
    selected = json.loads((out / "selected.json").read_text())
    arrays = np.load(out / "mps_blocks.npz")
    n = selected["qubits"]
    counts = Counter()
    records = []
    max_error = 0.0
    tick = time.monotonic()
    base, base_error = base_preparation(arrays["base"])
    counts.update(g[0] for g in base)
    for i, m in enumerate(selected["block_widths"]):
        u = arrays[f"U_{i}"]
        w = u.shape[0].bit_length() - 1
        rot, gray_inverse, error = decompose_isometry(u)
        max_error = max(max_error, error)
        local = Counter(g[0] for g in gray_inverse)
        mcx_counts = Counter(
            g[0]
            for g in C.mcx(
                list(range(w - 1)), w - 1, list(range(w, w + min(1, max(0, w - 3))))
            )
        )
        for a, b, theta in rot:
            target = (a ^ b).bit_length() - 1
            zeros = sum(not (a >> q) & 1 for q in range(w) if q != target)
            local.update(dict(x=2 * zeros, sdg=2, h=4, rz=2, s=2))
            local.update({g: 2 * k for g, k in mcx_counts.items()})
        counts.update(local)
        records.append(
            dict(
                remaining_qubits=m,
                window_qubits=w,
                window_offset=m - w,
                isometry_columns=u.shape[1],
                givens=rot,
                gray_inverse=gray_inverse,
                forward_preparation="inverse of recorded disentangling operations",
                by_gate=dict(local),
                subspace_frobenius_error=error,
            )
        )
        print(
            json.dumps(
                dict(
                    run=name, block=i + 1, window_qubits=w, givens=len(rot), error=error
                )
            ),
            flush=True,
        )
    work = max([0] + [min(1, max(0, x["window_qubits"] - 3)) for x in records])
    summary = dict(
        method="MPS sequential isometries; Gray-order real Givens decomposition",
        bond_dimension=selected["bond_dimension"],
        fidelity=selected["fidelity"],
        total_variation=selected["total_variation"],
        work_qubits=work,
        reusable_with_transport_work=True,
        position_qubits=n,
        RZ=counts["rz"],
        native_T=counts["t"] + counts["tdg"],
        Clifford=sum(v for k, v in counts.items() if k not in ["rz", "t", "tdg"]),
        by_gate=dict(counts),
        angle_omission_bound=0.0,
        maximum_isometry_subspace_error=max_error,
        base_gate_replay_error=base_error,
        primitive_checks=[
            primitive_check(w) for w in sorted({x["window_qubits"] for x in records})
        ],
        complete_statevector_gate_simulation=False,
        offline_seconds=time.monotonic() - tick,
    )
    (out / "compiled_inventory.json").write_text(json.dumps(summary, indent=2) + "\n")
    with gzip.open(out / "compiled_mps.json.gz", "wt") as f:
        json.dump(dict(base_program=base, isometries=records), f)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run", choices=["restart40", "long298", "late40"])
    args = ap.parse_args()
    main(args.run)
