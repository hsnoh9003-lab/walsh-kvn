#!/usr/bin/env python3
"""Compare centred and upwind quintic KvN generators at exact time.

Run: python -m Source.KvN.q5_upwind_comparison SOURCE_CSV OUTPUT_DIR
SOURCE_CSV is scalar_conservative_cell_masses.csv from the deposited grid
diagnostics. The signed prepared amplitudes come from the shared Q5 inputs.
The output directory receives q5_upwind_comparison.csv, distributions and a
summary. --input-convention positive_root selects a separate phase control.

This is a classical discretization comparison: expm_multiply evolves each
finite generator without a time-splitting approximation. It is not an LCHS
circuit or a fault-tolerant resource estimate. Finer grids subdivide the same
64-cell input; they do not refit it.
"""

# Direct-file execution must locate Codes/ independently of the working directory.
if __package__ in (None, ""):
    import sys as _path_sys
    from pathlib import Path as _Path
    _codes = next(p for p in _Path(__file__).resolve().parents
                  if (p / "Source/data_locations.py").is_file())
    _path_sys.path.insert(0, str(_codes))

import sys, csv, json, math, argparse, hashlib
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from Source.data_locations import shared_data
from scipy.sparse import lil_matrix, csc_matrix
from scipy.sparse.linalg import expm_multiply
from scipy.integrate import solve_ivp

CASE = "quintic_multistable__boundary_clipped"
C = 70.49560209450817  # Table S3 quintic scaled to maximum speed 0.25
T = 1.0
GRIDS = (64, 128, 256, 512, 1024, 2048, 4096, 8192)


def drift(x):
    return C * x * (1 - x) * (x - 0.25) * (x - 0.5) * (x - 0.75)


def flow(x0, t):
    """Characteristics dx/dt = F(x) for time t (t < 0 backtracks)."""
    sgn = 1.0 if t >= 0 else -1.0
    sol = solve_ivp(
        lambda s, x: sgn * drift(x),
        [0, abs(t)],
        np.asarray(x0, float),
        rtol=1e-11,
        atol=1e-14,
        method="DOP853",
    )
    return np.clip(sol.y[:, -1], 0.0, 1.0)


def centred_generator(n):
    """Assemble the skew-symmetric face stencil, omitting wrap-around couplings."""
    h = 1.0 / n
    g = drift(np.arange(1, n) / n) / (2 * h)  # face values
    G = lil_matrix((n, n))
    for k in range(n - 1):
        G[k + 1, k] = g[k]
        G[k, k + 1] = -g[k]
    return csc_matrix(G)


def upwind_generator(n):
    """Novikau-Joseph Eq. (25) in real form: d psi_r/dt = -|F_r| psi_r/h
    + xi_r (F_r + F_c) psi_c/(2h), c = r - xi_r, xi_r = sign(F_r); cell-centre drift.
    Couplings to cells outside the grid are omitted."""
    h = 1.0 / n
    F = drift((np.arange(n) + 0.5) / n)
    G = lil_matrix((n, n))
    for r in range(n):
        xi = 1 if F[r] > 0 else -1
        G[r, r] = -abs(F[r]) / h
        c = r - xi
        if 0 <= c < n:
            G[r, c] = xi * (F[r] + F[c]) / (2 * h)
    return csc_matrix(G)


def load_input(src, input_convention="mps"):
    """Load the circuit's actual amplitudes, or the auxiliary positive-root input."""
    rows = [
        r for r in csv.DictReader(open(src, encoding="utf-8-sig")) if r["id"] == CASE
    ]
    p64 = np.array([float(r["initial_prepared_mass"]) for r in rows])
    assert len(p64) == 64 and abs(p64.sum() - 1) < 1e-12
    amplitude_path = shared_data(
        "Radial_refinement", "Q5_Upwind_Comparison", "original_prepared_amplitude.npz"
    )
    with np.load(amplitude_path) as data:
        recorded = data["signed_mps_amplitude"]
    assert np.max(np.abs(np.imag(recorded))) < 1e-14
    recorded = np.real(recorded)
    assert np.max(np.abs(recorded**2 - p64)) < 2e-14
    assert abs(np.linalg.norm(recorded) - 1) < 1e-12
    if input_convention == "mps":
        initial = recorded
    elif input_convention == "positive_root":
        initial = np.sqrt(p64)
    else:
        raise ValueError("input_convention must be 'mps' or 'positive_root'")
    return p64, initial, amplitude_path


def main(src, out, input_convention="mps"):
    """Sweep spatial grids with matched characteristic references and write diagnostics."""
    p64, psi64, amplitude_path = load_input(src, input_convention)
    Path(out).mkdir(parents=True, exist_ok=True)
    e64 = np.arange(65) / 64
    cdf0 = np.concatenate([[0.0], np.cumsum(p64)])

    # continuum reference moments of the piecewise-uniform input (midpoint characteristics)
    K = 4000
    x0 = ((np.arange(64)[:, None] + (np.arange(K)[None, :] + 0.5) / K) / 64).ravel()
    w = np.repeat(p64, K) / K
    xT = flow(x0, T)
    m_ref = np.array([w @ xT, w @ xT**2])

    def reference_masses(n):
        back = flow(np.arange(n + 1) / n, -T)
        return np.diff(np.interp(back, e64, cdf0))

    records = []
    arrays = {}
    for n in GRIDS:
        m = n // 64
        # Low-order equal-superposition qubits preserve the recorded MPS signs.
        psi0 = (
            np.repeat(psi64 / np.sqrt(m), m)
            if input_convention == "mps"
            else np.repeat(np.sqrt(p64 / m), m)
        )
        xc = (np.arange(n) + 0.5) / n
        ref = reference_masses(n)
        for name, gen in (
            ("centred", centred_generator),
            ("upwind_NJ", upwind_generator),
        ):
            psi = expm_multiply(gen(n) * T, psi0)
            norm2 = float(psi @ psi)
            # The upwind amplitude generator need not conserve squared norm. Report
            # norm2 separately, and compare the normalized output probability with the
            # continuum density. norm2 is not a compiled LCHS success probability.
            q = psi**2 / norm2
            mom = np.array([q @ xc, q @ xc**2])
            # Report both native-cell total variation and mass aggregated back into
            # the original 64 bins; fine ripples and coarse mass errors differ.
            Q = q.reshape(64, m).sum(axis=1)
            R = ref.reshape(64, m).sum(axis=1)
            thr = 1e-9 * np.abs(psi).max()
            neg = psi < -thr
            dq = np.diff(q)
            dr = np.diff(ref)
            ext_q = int(
                np.sum(
                    (dq[:-1] * dq[1:] < 0)
                    & (np.maximum(q[1:-1], np.maximum(q[:-2], q[2:])) > 1e-6 * q.max())
                )
            )
            ext_r = int(
                np.sum(
                    (dr[:-1] * dr[1:] < 0)
                    & (
                        np.maximum(ref[1:-1], np.maximum(ref[:-2], ref[2:]))
                        > 1e-6 * ref.max()
                    )
                )
            )
            records.append(
                dict(
                    cells=n,
                    method=name,
                    squared_output_norm=norm2,
                    native_cell_TV=0.5 * float(np.abs(q - ref).sum()),
                    aggregated_64_bin_TV=0.5 * float(np.abs(Q - R).sum()),
                    relative_continuum_moment_error=float(
                        np.linalg.norm(mom - m_ref) / np.linalg.norm(m_ref)
                    ),
                    mean_x=float(mom[0]),
                    second_moment=float(mom[1]),
                    first_original_bin_mass=float(Q[0]),
                    first_original_bin_reference_mass=float(R[0]),
                    negative_amplitude_cells=int(neg.sum()),
                    negative_amplitude_probability=float(q[neg].sum()),
                    strict_interior_extrema=ext_q,
                    reference_strict_interior_extrema=ext_r,
                )
            )
            arrays[f"{name}_{n}_probability"] = q
        arrays[f"reference_{n}_mass"] = ref
    keys = list(records[0].keys())
    with open(
        out + "/q5_upwind_comparison.csv", "w", newline="", encoding="utf-8"
    ) as f:
        wri = csv.DictWriter(f, fieldnames=keys)
        wri.writeheader()
        wri.writerows(records)
    np.savez_compressed(out + "/q5_upwind_distributions.npz", **arrays)
    json.dump(
        dict(
            case=CASE,
            drift_scale=C,
            duration=T,
            reference_moments=m_ref.tolist(),
            input_amplitude_convention=input_convention,
            input_amplitude_source=(
                "original_prepared_amplitude.npz:signed_mps_amplitude"
                if input_convention == "mps"
                else "nonnegative square root of prepared cell masses"
            ),
            recorded_amplitude_file_sha256=hashlib.sha256(
                amplitude_path.read_bytes()
            ).hexdigest(),
            initial_negative_amplitude_cells=int(np.sum(psi64 < 0)),
            initial_negative_amplitude_probability=float(np.sum(psi64[psi64 < 0] ** 2)),
            refined_input=(
                "recorded six-qubit state tensor low-order equal-superposition qubits"
                if input_convention == "mps"
                else "uniform subdivision of masses with nonnegative square-root amplitudes"
            ),
            amplitude_sign_diagnostic="probability in cells with psi < -1e-9 * max(abs(psi)); not a density-ripple certificate",
            reference_moment_l2_norm=float(np.linalg.norm(m_ref)),
            reference_quadrature_points_per_original_cell=K,
            time_propagation="exact generator action (scipy.sparse.linalg.expm_multiply)",
            scope="classical discretization control; no LCHS circuit or resource estimate",
        ),
        open(out + "/summary.json", "w"),
        indent=1,
    )
    return records, m_ref


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_csv")
    parser.add_argument("output_dir")
    parser.add_argument(
        "--input-convention", choices=["mps", "positive_root"], default="mps"
    )
    args = parser.parse_args()
    recs, m_ref = main(args.source_csv, args.output_dir, args.input_convention)
    print("reference moments", m_ref)
    print(
        f"{'N':>5} {'method':>10} {'norm^2':>9} {'TV%':>8} {'TV64%':>8} {'mom%':>9} {'bin0':>6} {'neg':>5} {'negP':>9} {'ext':>4}/{'ref':<3}"
    )
    for r in recs:
        print(
            f"{r['cells']:5d} {r['method']:>10} {r['squared_output_norm']:9.5f} {100*r['native_cell_TV']:8.3f} {100*r['aggregated_64_bin_TV']:8.3f} "
            f"{100*r['relative_continuum_moment_error']:9.3f} {r['first_original_bin_mass']:6.3f} {r['negative_amplitude_cells']:5d} {r['negative_amplitude_probability']:9.2e} "
            f"{r['strict_interior_extrema']:4d}/{r['reference_strict_interior_extrema']:<3d}"
        )
