"""Real bond-two MPS preparation from dense probability vectors.

Successive small singular-value decompositions truncate Schmidt rank to two.
Four-angle real two-qubit subspace rotations are compiled at fixed tolerance;
the remaining two-qubit state uses an analytic Schmidt preparation. The
algorithm records the gate-executed state, never substitutes target amplitudes.
"""

from pathlib import Path
import json, time
import numpy as np
from scipy.optimize import least_squares


def ry(a):
    """Return the real 2-by-2 Ry matrix for an angle in radians."""
    c, s = np.cos(a / 2), np.sin(a / 2)
    return np.array([[c, -s], [s, c]])


CX10 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], float)
CX01 = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]], float)


def local_matrix(a):
    """Build the four-angle two-qubit unitary with q0 as the surviving bond."""
    # Little-endian local q0 is the surviving bond, q1 is newly initialized.
    x = np.kron(ry(a[0]), np.eye(2))
    x = CX10 @ x
    x = np.kron(ry(a[2]), ry(a[1])) @ x
    x = CX01 @ x
    return np.kron(np.eye(2), ry(a[3])) @ x


def subspace_rotation(basis, seed=0):
    """Fit the projector onto a real two-dimensional subspace of four amplitudes."""
    target = basis @ basis.T
    rng = np.random.default_rng(seed)
    best = None
    for trial in range(12):
        guess = np.zeros(4) if trial == 0 else rng.uniform(-np.pi, np.pi, 4)

        def residual(a):
            u = local_matrix(a)[:, :2]
            return (u @ u.T - target).ravel()

        fit = least_squares(
            residual, guess, ftol=2e-13, xtol=2e-13, gtol=2e-13, max_nfev=400
        )
        error = float(np.linalg.norm(residual(fit.x)))
        if best is None or error < best[0]:
            best = (error, fit.x, trial, int(fit.nfev))
        if error < 3e-11:
            break
    if best[0] > 3e-10:
        raise RuntimeError(f"Subspace compilation failed: {best[0]}")
    return best[1], dict(projector_error=best[0], start=best[2], evaluations=best[3])


def local_program(angles, q0, q1):
    a, b, c, d = map(float, angles)
    return [
        ("ry", [q1], a),
        ("cx", [q1, q0], None),
        ("ry", [q0], b),
        ("ry", [q1], c),
        ("cx", [q0, q1], None),
        ("ry", [q0], d),
    ]


def execute(n, program, initial=None):
    """Apply the Ry/CX program to a little-endian n-qubit statevector."""
    if initial is None:
        state = np.zeros(1 << n, complex)
        state[0] = 1
    else:
        state = np.asarray(initial, complex).copy()
    idx = np.arange(1 << n)
    for name, qs, a in program:
        if name == "cx":
            state = state[idx ^ (((idx >> qs[0]) & 1) << qs[1])]
        elif name == "ry":
            lo = idx[(idx & (1 << qs[0])) == 0]
            hi = lo + (1 << qs[0])
            x, y = state[lo].copy(), state[hi].copy()
            c, s = np.cos(a / 2), np.sin(a / 2)
            state[lo] = c * x - s * y
            state[hi] = s * x + c * y
        else:
            raise ValueError(name)
    return state


def compile_rz(program):
    """Replace every Ry with an equivalent Rz/Clifford sequence."""
    result = []
    for name, qs, a in program:
        if name == "ry":
            result.extend(
                [
                    ("sdg", qs, None),
                    ("h", qs, None),
                    ("rz", qs, float(a)),
                    ("h", qs, None),
                    ("s", qs, None),
                ]
            )
        else:
            result.append((name, qs, a))
    return result


def prepare_amplitude(amplitude, seed=20261106):
    """Normalize and approximate a real amplitude vector with a bond-two MPS circuit."""
    t0 = time.perf_counter()
    target = np.asarray(amplitude, float)
    if (
        target.ndim != 1
        or len(target) < 4
        or len(target) & (len(target) - 1)
        or not np.all(np.isfinite(target))
    ):
        raise ValueError("Need a finite real power-of-two vector, size at least four.")
    target = target / np.linalg.norm(target)
    p = target * target
    n = int(np.log2(len(p)))
    state = target.copy()
    blocks = []
    discarded = []
    records = []
    for m in range(n, 2, -1):
        matrix = state.reshape(4, -1)
        u, s, vh = np.linalg.svd(matrix, full_matrices=False)
        basis = u[:, :2]
        angles, record = subspace_rotation(basis, seed + m)
        gate = local_matrix(angles)
        transformed = gate.T @ matrix
        discarded.append(float(np.sum(transformed[2:] ** 2)))
        state = transformed[:2].ravel()
        state /= np.linalg.norm(state)
        blocks.append(local_program(angles, m - 2, m - 1))
        records.append(record)
    # Make both Schmidt basis rotations proper; the smaller coefficient may sign.
    u, s, vh = np.linalg.svd(state.reshape(2, 2))
    v = vh.T
    du, dv = np.linalg.det(u), np.linalg.det(v)
    u = u @ np.diag([1, du])
    v = v @ np.diag([1, dv])
    theta = 2 * np.arctan2(s[1] * du * dv, s[0])
    program = [
        ("ry", [1], float(theta)),
        ("cx", [1, 0], None),
        ("ry", [1], float(2 * np.arctan2(u[1, 0], u[0, 0]))),
        ("ry", [0], float(2 * np.arctan2(v[1, 0], v[0, 0]))),
    ]
    for block in reversed(blocks):
        program.extend(block)
    amplitude = execute(n, program)
    if np.vdot(target, amplitude).real < 0:
        # A 2pi rotation changes the global sign without increasing gate count.
        name, qs, a = program[0]
        program[0] = (name, qs, a + 2 * np.pi)
        amplitude = execute(n, program)
    fidelity = float(abs(np.vdot(target, amplitude)) ** 2)
    ranks = [
        int(
            np.sum(
                np.linalg.svd(amplitude.real.reshape(1 << k, -1), compute_uv=False)
                > 1e-10
            )
        )
        for k in range(1, n)
    ]
    assert max(ranks) <= 2
    assert abs(np.linalg.norm(amplitude) - 1) < 1e-12
    metadata = dict(
        method="dense-vector real bond-two MPS; sequential truncated SVD and fixed-size real subspace compilation",
        n=n,
        cells=len(p),
        bond_dimension=2,
        Ry=4 * n - 5,
        CX=2 * n - 3,
        fidelity=fidelity,
        amplitude_l2=float(np.linalg.norm(amplitude - target)),
        total_variation=float(0.5 * np.sum(abs(abs(amplitude) ** 2 - p))),
        schmidt_ranks=ranks,
        discarded_norm_squared=discarded,
        local_compilation=records,
        offline_seconds=time.perf_counter() - t0,
        complexity="O(N) dense preprocessing for fixed bond, precision and bounded local-solver work; O(log N) quantum gates. No resolution-independent fidelity guarantee.",
        target_amplitude_injected=False,
    )
    return amplitude, program, metadata


def prepare(probabilities, seed=20261106):
    """Prepare the nonnegative square-root amplitudes of normalized probabilities."""
    p = np.asarray(probabilities, float)
    if np.any(p < 0) or p.sum() <= 0:
        raise ValueError("Invalid probabilities")
    return prepare_amplitude(np.sqrt(p / p.sum()), seed)


def prepare_layers(probabilities, min_fidelity=0.999999, max_layers=8, seed=20261106):
    """Add bond-two circuit layers until the target fidelity or layer limit is reached."""
    p = np.asarray(probabilities, float)
    p = p / p.sum()
    target = np.sqrt(p)
    residual = target.copy()
    layers = []
    records = []
    n = int(np.log2(len(p)))
    t0 = time.perf_counter()
    for index in range(max_layers):
        _, g, record = prepare_amplitude(residual, seed + 100 * index)
        layers.append(g)
        records.append(record)
        program = [op for layer in reversed(layers) for op in layer]
        actual = execute(n, program)
        fidelity = float(abs(np.vdot(target, actual)) ** 2)
        if fidelity >= min_fidelity:
            break
        inverse = [(name, qs, -a if name == "ry" else a) for name, qs, a in reversed(g)]
        residual = execute(n, inverse, residual).real
    else:
        raise RuntimeError(f"MPS layers did not reach declared fidelity: {fidelity}")
    return (
        actual,
        program,
        dict(
            method="successive real bond-two MPS layers",
            n=n,
            cells=len(p),
            bond_dimension=2,
            layers=len(layers),
            Ry=len(layers) * (4 * n - 5),
            CX=len(layers) * (2 * n - 3),
            fidelity=fidelity,
            amplitude_l2=float(np.linalg.norm(actual - target)),
            total_variation=float(0.5 * np.sum(abs(abs(actual) ** 2 - p))),
            offline_seconds=time.perf_counter() - t0,
            layer_records=records,
            min_fidelity=min_fidelity,
            max_layers=max_layers,
            target_amplitude_injected=False,
            complexity="O(N) preprocessing and O(log N) online gates at fixed layer/bond/precision budgets; layers required for accuracy are not uniformly bounded by this experiment.",
        ),
    )


def save(directory, probabilities):
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    amplitude, program, meta = prepare(probabilities)
    np.savez_compressed(
        out / "initial.npz",
        target=np.asarray(probabilities),
        amplitude=amplitude.real,
        angles=np.array([a for name, _, a in program if name == "ry"]),
        program_json=json.dumps(program),
    )
    (out / "preparation.json").write_text(
        json.dumps(
            dict(**meta, native_program=program, compiled_program=compile_rz(program)),
            indent=2,
        )
    )
    return meta


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("output", type=Path)
    a = ap.parse_args()
    data = np.load(a.input)
    print(json.dumps(save(a.output, data["target"]), indent=2))
