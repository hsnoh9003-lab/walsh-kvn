#!/usr/bin/env python3
"""Common one- and two-dimensional polynomial KvN transport and gate compilation.

Public callers include Source.KvN.grid_step_scaling,
Source.KvN.polynomial_benchmarks, and the Fig. S5 generator.
Saved MPS preparation fits are read from the retained Source_Data inputs, or
from an explicit PQC_PREPARATION_ROOT. Paths are resolved only when a fit is
read; importing the compiler does not access data or create outputs.
Independent validation lives in
Internal_Workspace.Validation.Checks.verify_polynomial_transport.

There are three distinct references. Explicit gates are checked against direct
pair rotations to test compilation. Exact finite-generator evolution isolates
time-splitting error. Continuum characteristics of the prepared cell density
isolate transport error, while a separate analytic Gaussian reference also
includes input projection and preparation errors. These references must not be
interchanged when interpreting the exported errors.

The helper's own default baseline is 16-by-16 for the coupled case; callers for
the published Fig. 5/S5 calculation explicitly use 32-by-32. Native gates are
simulated with exact arbitrary rotations; this is not a hardware noise model or
a protected physical-resource calculation.
"""
from __future__ import annotations

# Direct-file execution must locate Codes/ independently of the working directory.
if __package__ in (None, ""):
    import sys as _path_sys
    from pathlib import Path as _Path
    _codes = next(p for p in _Path(__file__).resolve().parents
                  if (p / "Source/data_locations.py").is_file())
    _path_sys.path.insert(0, str(_codes))

import os

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
from pathlib import Path
import json, math, time
from collections import Counter
import numpy as np
from numpy.polynomial import Polynomial
from scipy.special import ndtr
from scipy.integrate import solve_ivp
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import expm_multiply
from numpy.polynomial.legendre import leggauss
from Source.KvN.wake_gate_core import controlled_ry, increment, inverse, apply_gates, inventory
from Source.KvN.sparse_walsh import walsh
from Source.Data_Production.data_assets import fixed_asset

T_FINAL = 1.0
MU = [0.35, 0.65]
SIGMA = [0.12, 0.10]
SPATIAL_1D = [16, 32, 64, 128]
SPATIAL_2D = [8, 16, 32, 64]
TIME_1D = [16, 32, 64, 128]
TIME_2D = [8, 16, 32, 64]
BASE_N_1D = 64
BASE_N_2D = 16
BASE_STEPS = 64
COEFF_THRESHOLD = 1e-14


def max_abs_polynomial(p):
    """Find the maximum absolute drift on [0,1] using endpoints and interior stationary points."""
    real = [
        float(z.real)
        for z in p.deriv().roots()
        if abs(z.imag) < 1e-10 and 0 < z.real < 1
    ]
    candidates = [0.0, 1.0] + real
    values = [abs(float(p(x))) for x in candidates]
    ind = int(np.argmax(values))
    return values[ind], candidates[ind], candidates


def declared_cases():
    """Return the fixed polynomial families and their maximum-speed normalization."""
    x = Polynomial([0.0, 1.0])
    endpoint = x * (1 - x)
    raw = [
        ("quadratic", "Q2", endpoint, "x(1-x)", "endpoint-attracting growth"),
        (
            "cubic_positive",
            "C3",
            endpoint * (1 + 2 * x),
            "x(1-x)(1+2x)",
            "endpoint-attracting growth",
        ),
        (
            "cubic_interior",
            "C3i",
            endpoint * (0.5 - x),
            "x(1-x)(1/2-x)",
            "interior-attracting equilibrium at 1/2",
        ),
        ("quartic", "Q4", x - x**4, "x-x^4", "endpoint-attracting growth"),
        ("quintic", "Q5", x - x**5, "x-x^5", "endpoint-attracting growth"),
        (
            "quintic_multistable",
            "Q5m",
            endpoint * (x - 0.25) * (x - 0.5) * (x - 0.75),
            "x(1-x)(x-1/4)(x-1/2)(x-3/4)",
            "three attracting equilibria at 0, 1/2 and 1",
        ),
    ]
    cases = []
    for name, label, p, formula, behaviour in raw:
        maximum, argmax, candidates = max_abs_polynomial(p)
        scale = 0.25 / maximum
        cases.append(
            dict(
                name=name,
                label=label,
                D=1,
                K=p.degree(),
                raw_coefficients=p.coef.tolist(),
                raw_formula=formula,
                normalization_scale=scale,
                raw_max_abs=maximum,
                raw_abs_max_location=argmax,
                critical_candidates=candidates,
                velocity_max_abs=0.25,
                qualitative_behaviour=behaviour,
            )
        )
    cases.append(
        dict(
            name="coupled_cubic_2d",
            label="2D C3",
            D=2,
            K=3,
            raw_formula=["x(1-x)(1+2y)", "y(1-y)(-2+x)"],
            normalization_scale=[1 / 3, 1 / 2],
            raw_max_abs=[0.75, 0.5],
            velocity_max_abs=[0.25, 0.25],
            qualitative_behaviour="coupled coordinate-dependent drift toward x=1, y=0",
        )
    )
    return cases


def drift(case, a):
    if case["D"] == 1:
        return [
            case["normalization_scale"] * Polynomial(case["raw_coefficients"])(a[0])
        ]
    x, y = a
    return [x * (1 - x) * (1 + 2 * y) / 3, y * (1 - y) * (-2 + x) / 2]


def coordinates(N, D):
    """Decode cell-centre coordinates with axis zero in the lowest-order address bits."""
    basis = np.arange(N**D, dtype=np.int64)
    return [((basis // N**j) % N + 0.5) / N for j in range(D)]


def cell_probabilities(N, j):
    """Integrate the truncated Gaussian over cells before normalizing its total mass."""
    e = np.linspace(0, 1, N + 1)
    p = np.diff(ndtr((e - MU[j]) / SIGMA[j]))
    return p / p.sum()


PREPARATION_ROOT = None
_PREP_CACHE = {}


def preparation_root():
    """Resolve a selected JSON fit bundle only when it is needed."""
    explicit = PREPARATION_ROOT or os.environ.get("PQC_PREPARATION_ROOT")
    root = (Path(explicit).expanduser() if explicit else
            fixed_asset("inputs", "retained_experiments", "mps"))
    root = root.resolve()
    return root / "fits" if (root / "fits").is_dir() else root


def preparation_fit(N, j):
    """Replay stored preparation gates and compare their state with the target and saved amplitudes."""
    n = int(np.log2(N))
    family = "x" if j == 0 else "y"
    folder = preparation_root()
    key = (str(folder), n, j)
    if key not in _PREP_CACHE:
        file = folder / f"{family}_n{n}.json"
        if not file.is_file():
            raise FileNotFoundError(
                f"Missing saved MPS fit: {file}. Set PQC_PREPARATION_ROOT "
                "to a directory containing the required JSON gate records."
            )
        record = json.loads(file.read_text())
        gates = [
            (g["name"], g["qubits"], g["angle"]) for g in record["rz_clifford_gates"]
        ]
        state = np.zeros(N, complex)
        state[0] = 1
        state = apply_gates(state, gates, n)
        target = np.sqrt(cell_probabilities(N, j))
        overlap = np.vdot(target, state)
        aligned = state * np.conj(overlap / abs(overlap)) if abs(overlap) > 0 else state
        record_audit = dict(
            family=family,
            n_qubits=n,
            cells=N,
            architecture=record.get("architecture", "MPS"),
            Ry_count=sum(g["name"] == "ry" for g in record["ry_cx_gates"]),
            CX_count=sum(g["name"] == "cx" for g in record["ry_cx_gates"]),
            fidelity=float(abs(overlap) ** 2),
            target_state_l2_error=float(np.linalg.norm(aligned - target)),
            total_variation=float(0.5 * np.sum(abs(abs(state) ** 2 - target**2))),
            negative_aligned_real_weight=float(
                np.sum(abs(aligned[aligned.real < 0]) ** 2)
            ),
            aligned_imaginary_l2=float(np.linalg.norm(aligned.imag)),
            norm_defect=float(abs(np.vdot(state, state).real - 1)),
        )
        # The deposited fit JSON is the fixed input. A separate NPZ artifact
        # was not deposited; do not require or silently invent it.
        artifact = file.with_suffix(".npz")
        record_audit["gate_vs_fit_artifact_aligned_l2"] = None
        if artifact.is_file():
            with np.load(artifact) as saved:
                record_audit["gate_vs_fit_artifact_aligned_l2"] = aligned_error(
                    state, saved["prepared_amplitude"]
                )
        quality = record.get("quality", {})
        if "fidelity" in quality:
            record_audit["recorded_fidelity_abs_difference"] = abs(
                record_audit["fidelity"] - float(quality["fidelity"])
            )
        _PREP_CACHE[key] = (record, gates, state, record_audit)
    return _PREP_CACHE[key]


def prepared_cell_probabilities(N, j):
    return abs(preparation_fit(N, j)[2]) ** 2


def initial_amplitude(N, D):
    """Form the tensor-product prepared amplitude while retaining its complex phases."""
    amplitude = np.ones(N**D, dtype=complex)
    basis = np.arange(N**D)
    for j in range(D):
        amplitude *= preparation_fit(N, j)[2][(basis // N**j) % N]
    return amplitude


def observable_values(a):
    """Return x,x**2 in one dimension or x,y,x**2,y**2,x*y in two dimensions."""
    if len(a) == 1:
        return np.array([a[0], a[0] ** 2])
    x, y = a
    return np.array([x, y, x * x, y * y, x * y])


def observable_names(D):
    return ["x", "x2"] if D == 1 else ["x", "y", "x2", "y2", "xy"]


def moments(state, N, D):
    """Average the declared observables with the state-vector probabilities."""
    prob = np.abs(state) ** 2
    return observable_values(coordinates(N, D)) @ prob


def raw_density(a, D):
    """Evaluate the normalized truncated Gaussian density used only for the analytic-input reference."""
    value = np.ones_like(a[0], dtype=float)
    for j in range(D):
        norm = ndtr((1 - MU[j]) / SIGMA[j]) - ndtr(-MU[j] / SIGMA[j])
        value *= np.exp(-0.5 * ((a[j] - MU[j]) / SIGMA[j]) ** 2) / (
            SIGMA[j] * np.sqrt(2 * np.pi) * norm
        )
    return value


def continuum_reference(case, order, rtol, atol, times):
    """Integrate characteristics from the analytic Gaussian using whole-domain quadrature."""
    D = case["D"]
    x, w = leggauss(order)
    x = (x + 1) / 2
    w = w / 2
    if D == 1:
        a0 = np.array([x])
        weights = w
    else:
        xx, yy = np.meshgrid(x, x, indexing="ij")
        wx, wy = np.meshgrid(w, w, indexing="ij")
        a0 = np.array([xx.ravel(), yy.ravel()])
        weights = (wx * wy).ravel()
    weights = weights * raw_density(a0, D)

    # Quadrature normalization is reported, not silently re-normalized.
    def rhs(t, z):
        return np.asarray(drift(case, z.reshape(D, -1))).ravel()

    sol = solve_ivp(
        rhs,
        (0, T_FINAL),
        a0.ravel(),
        method="DOP853",
        t_eval=times,
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    hist = []
    for k in range(len(times)):
        av = sol.y[:, k].reshape(D, -1)
        hist.append(observable_values(av) @ weights)
    return np.asarray(hist), dict(
        order=order,
        rtol=rtol,
        atol=atol,
        nfev=sol.nfev,
        quadrature_mass=float(weights.sum()),
        min_coordinate=float(sol.y.min()),
        max_coordinate=float(sol.y.max()),
    )


def generator_and_pairs(case, N):
    """Build the skew-symmetric face generator and its coordinate/parity pair lists."""
    D = case["D"]
    h = 1 / N
    dim = N**D
    basis = np.arange(dim)
    a = coordinates(N, D)
    rows = []
    cols = []
    vals = []
    pairs = []
    for j in range(D):
        for parity in [0, 1]:
            own = (basis // N**j) % N
            lo = basis[(own % 2 == parity) & (own < N - 1)]
            hi = lo + N**j
            faces = [x[lo].copy() for x in a]
            faces[j] += 0.5 * h
            f = np.asarray(drift(case, faces)[j], dtype=float)
            # The face coefficient f/(2h) is the amplitude-generator coupling.
            # Its two-cell exponential uses cosine/sine angle dt*f/(2h); Ry uses
            # twice that angle, which is why angle_table returns dt*N*f.
            g = f / (2 * h)
            rows.extend(lo.tolist() + hi.tolist())
            cols.extend(hi.tolist() + lo.tolist())
            vals.extend((-g).tolist() + g.tolist())
            pairs.append(dict(axis=j, parity=parity, lo=lo, hi=hi, g=g))
    G = coo_matrix((vals, (rows, cols)), shape=(dim, dim)).tocsr()
    return G, pairs


def sequence(D):
    """Return the symmetric forward-half/full/reverse-half layer schedule."""
    size = 2 * D
    return (
        [(i, 0.5) for i in range(size - 1)]
        + [(size - 1, 1.0)]
        + [(i, 0.5) for i in reversed(range(size - 1))]
    )


def pair_step(state, pairs, D, delta):
    """Apply the symmetric pair-rotation split directly to amplitudes or matrix columns."""
    out = state.copy()
    for ind, weight in sequence(D):
        p = pairs[ind]
        lo, hi = p["lo"], p["hi"]
        phi = weight * delta * p["g"]
        c = np.cos(phi)
        s = np.sin(phi)
        a = out[lo].copy()
        b = out[hi].copy()
        if out.ndim == 2:
            c = c[:, None]
            s = s[:, None]
        out[lo] = c * a - s * b
        out[hi] = s * a + c * b
    return out


def pair_evolution(a, pairs, D, delta, steps, save_every=1):
    """Repeat pair steps and save only the requested diagnostic checkpoints."""
    state = a.astype(complex).copy()
    hist = [state.copy()]
    for k in range(steps):
        state = pair_step(state, pairs, D, delta)
        if (k + 1) % save_every == 0:
            hist.append(state.copy())
    return state, np.asarray(hist)


def remap(gates, mapping):
    return [(name, [mapping[q] for q in qs], angle) for name, qs, angle in gates]


def angle_table(case, N, j, parity, delta):
    """Evaluate the face drift and convert it to a target Ry angle for each control address."""
    D = case["D"]
    n = int(np.log2(N))
    target = j * n
    controls = [q for q in range(D * n) if q != target]
    z = np.arange(2 ** len(controls), dtype=np.int64)
    index = np.zeros_like(z)
    for k, q in enumerate(controls):
        index |= ((z >> k) & 1) << q
    a = [((index // N**k) % N + 0.5) / N for k in range(D)]
    a[j] = (((index // N**j) % N) + parity + 1) / N
    f = np.asarray(drift(case, a)[j], dtype=float)
    return delta * N * f, controls, target


def compile_step(case, N, delta):
    """Compile aligned even/odd layers and record the retained Walsh support."""
    D = case["D"]
    n = int(np.log2(N))
    work = list(range(D * n, D * n + min(1, max(0, n - 3))))
    nq = D * n + len(work)
    layers = []
    audits = []
    for j in range(D):
        mapping = {i: j * n + i for i in range(n)}
        mapping.update({q: q for q in work})
        inc = remap(increment(n, work), mapping)
        for parity in [0, 1]:
            theta, controls, target = angle_table(case, N, j, parity, delta)
            coeff = walsh(theta)
            nz = np.flatnonzero(np.abs(coeff) > COEFF_THRESHOLD)
            # A degree-K polynomial has binary-parity support on at most K bits.
            # Count possible subsets, then check the actual numerical Walsh support;
            # the floating-point coefficient cutoff is recorded separately.
            bound = sum(
                math.comb(len(controls), k)
                for k in range(min(case["K"], len(controls)) + 1)
            )
            max_weight = max([int(x).bit_count() for x in nz], default=0)
            audits.append(
                dict(
                    axis=j,
                    parity=parity,
                    control_bits=len(controls),
                    nonzero_terms=len(nz),
                    support_bound=bound,
                    max_Walsh_weight=max_weight,
                    maximum_discarded_angle=float(
                        np.max(
                            np.abs(coeff[np.abs(coeff) <= COEFF_THRESHOLD]), initial=0
                        )
                    ),
                    minimum_retained_angle=float(
                        np.min(np.abs(coeff[nz]), initial=np.inf)
                    ),
                    angle_threshold=COEFF_THRESHOLD,
                )
            )
            gates = controlled_ry(theta, controls, target)
            layers.append((inverse(inc) + gates + inc) if parity else gates)
    gates = []
    for ind, weight in sequence(D):
        j, parity = divmod(ind, 2)
        theta, controls, target = angle_table(case, N, j, parity, delta * weight)
        block = controlled_ry(theta, controls, target)
        if parity:
            mapping = {i: j * n + i for i in range(n)}
            mapping.update({q: q for q in work})
            inc = remap(increment(n, work), mapping)
            block = inverse(inc) + block + inc
        gates += block
    return gates, nq, audits


def compile_preparation(N, D):
    """Map each coordinate preparation circuit onto its own address register."""
    n = int(np.log2(N))
    g = []
    for j in range(D):
        g += remap(preparation_fit(N, j)[1], {i: j * n + i for i in range(n)})
    return g


class CachedPrimitiveExecutor:
    """Executes each actual primitive; caches indices/phases, never a whole layer."""

    def __init__(self, nq):
        self.nq = nq
        self.basis = np.arange(2**nq, dtype=np.int64)
        self.cache = {}

    def compiled(self, gates):
        out = []
        for name, qs, angle in gates:
            key = (name, tuple(qs), angle)
            if key not in self.cache:
                if name == "cx":
                    c, t = qs
                    self.cache[key] = (
                        "perm",
                        self.basis ^ (((self.basis >> c) & 1) << t),
                    )
                elif name == "x":
                    self.cache[key] = ("perm", self.basis ^ (1 << qs[0]))
                elif name == "h":
                    lo = self.basis[(self.basis & (1 << qs[0])) == 0]
                    self.cache[key] = ("h", lo, lo + (1 << qs[0]))
                else:
                    bit = (self.basis >> qs[0]) & 1
                    phase = (
                        np.exp(-0.5j * angle * (1 - 2 * bit))
                        if name == "rz"
                        else np.exp(
                            1j
                            * {
                                "t": np.pi / 4,
                                "tdg": -np.pi / 4,
                                "s": np.pi / 2,
                                "sdg": -np.pi / 2,
                            }[name]
                            * bit
                        )
                    )
                    self.cache[key] = ("phase", phase)
            out.append(self.cache[key])
        return out

    @staticmethod
    def run(state, primitives):
        state = state.copy()
        for op in primitives:
            if op[0] == "perm":
                state = state[op[1]]
            elif op[0] == "h":
                lo, hi = op[1:]
                a = state[lo].copy()
                b = state[hi].copy()
                state[lo] = (a + b) / np.sqrt(2)
                state[hi] = (a - b) / np.sqrt(2)
            else:
                state *= op[1] if state.ndim == 1 else op[1][:, None]
        return state


def aligned_error(a, b):
    """Remove the global phase before taking the state-vector Euclidean error."""
    z = np.vdot(b, a)
    phase = z / abs(z) if abs(z) > 0 else 1
    return float(np.linalg.norm(a - phase * b))


def gate_run(case, N, steps, return_history=False):
    """Replay preparation and transport gates against pair and exact-generator references."""
    D = case["D"]
    dim = N**D
    delta = T_FINAL / steps
    prep = compile_preparation(N, D)
    gates, nq, support = compile_step(case, N, delta)
    sim = CachedPrimitiveExecutor(nq)
    prep_ops = sim.compiled(prep)
    step_ops = sim.compiled(gates)
    state = np.zeros(2**nq, dtype=complex)
    state[0] = 1
    state = sim.run(state, prep_ops)
    a = initial_amplitude(N, D)
    prep_error = aligned_error(state[:dim], a)
    G, pairs = generator_and_pairs(case, N)
    expected = expm_multiply(T_FINAL * G, a)
    pair = a.astype(complex).copy()
    max_pair_error = 0.0
    max_leak = 0.0
    hist = [state[:dim].copy()]
    start = time.perf_counter()
    for k in range(steps):
        state = sim.run(state, step_ops)
        pair = pair_step(pair, pairs, D, delta)
        err = aligned_error(state[:dim], pair)
        leak = float(np.sum(np.abs(state[dim:]) ** 2))
        max_pair_error = max(max_pair_error, err)
        max_leak = max(max_leak, leak)
        if return_history:
            hist.append(state[:dim].copy())
    elapsed = time.perf_counter() - start
    final = state[:dim]
    mom = moments(final, N, D)
    emom = moments(expected, N, D)
    inv = inventory(gates)
    pinv = inventory(prep)
    total = {k: pinv[k] + steps * inv[k] for k in ["RZ", "direct_T", "Clifford"]}
    by = Counter(pinv["by_gate"])
    by.update({k: steps * v for k, v in inv["by_gate"].items()})
    row = dict(
        case=case["name"],
        label=case["label"],
        dimension=D,
        degree=case["K"],
        cells_per_coordinate=N,
        cells_total=dim,
        steps=steps,
        data_qubits=D * int(np.log2(N)),
        work_qubits=nq - D * int(np.log2(N)),
        logical_qubits=nq,
        preparation_state_l2_error=prep_error,
        preparation_target_state_l2_error=aligned_error(
            a,
            np.sqrt(
                np.prod(
                    [
                        cell_probabilities(N, j)[(np.arange(dim) // N**j) % N]
                        for j in range(D)
                    ],
                    axis=0,
                )
            ),
        ),
        preparation_target_probability_TV=float(
            0.5
            * np.sum(
                abs(
                    abs(a) ** 2
                    - np.prod(
                        [
                            cell_probabilities(N, j)[(np.arange(dim) // N**j) % N]
                            for j in range(D)
                        ],
                        axis=0,
                    )
                )
            )
        ),
        preparation_Ry=sum(preparation_fit(N, j)[3]["Ry_count"] for j in range(D)),
        maximum_gate_vs_pair_state_l2_error=max_pair_error,
        final_gate_vs_pair_raw_l2_error=float(np.linalg.norm(final - pair)),
        maximum_work_leakage=max_leak,
        norm_defect=float(abs(np.vdot(state, state).real - 1)),
        state_l2_splitting_error=aligned_error(final, expected),
        moment_rms_splitting_error=float(np.sqrt(np.mean((mom - emom) ** 2))),
        per_step_RZ=inv["RZ"],
        per_step_CX=inv["by_gate"].get("cx", 0),
        per_step_native_T=inv["direct_T"],
        preparation_RZ=pinv["RZ"],
        preparation_CX=pinv["by_gate"].get("cx", 0),
        complete_RZ=total["RZ"],
        complete_CX=by.get("cx", 0),
        complete_native_T=total["direct_T"],
        wall_seconds=elapsed,
        status="pass",
    )
    if prep_error > 1e-10 or max_pair_error > 1e-9 or max_leak > 1e-18:
        row["status"] = "FAIL"
    return (
        row,
        final,
        np.asarray(hist),
        support,
        dict(
            nqubits=nq,
            prep=prep,
            step=gates,
            steps=steps,
            native_basis="arbitrary RZ, H, S/Sdag, X, CX, T/Tdag; CCX decomposed; no arbitrary-rotation T synthesis",
        ),
    )

# Independent polynomial validation is in Internal_Workspace/Validation/Checks.
