"""Compare radial wake stencils with the same input and coherent phase readout.

Run: python -m Source.KvN.wake_upwind_refinement --grids 64 128
Inputs: Shared_source_data/Radial_refinement/Radial128, including saved circuit
baselines. Outputs: Generated_Outputs/Wake_Upwind_Refinement/.

Centred face, centred Novikau-Joseph and positive-drift upwind stencils are
evolved with exact classical matrix-exponential actions. The upwind stencil
follows Eq. (24) of arXiv:2410.03985v2. This calculation does not compile LCHS
or estimate hardware runtime. The shot count excludes implementation error;
it is a nominal comparison under the same field metric and continuum input.
"""

# Direct-file execution must locate Codes/ independently of the working directory.
if __package__ in (None, ""):
    import sys as _path_sys
    from pathlib import Path as _Path
    _codes = next(p for p in _Path(__file__).resolve().parents
                  if (p / "Source/data_locations.py").is_file())
    _path_sys.path.insert(0, str(_codes))



import os

for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[key] = "1"
from pathlib import Path
import json, csv, math, time, hashlib
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.sparse import diags
from scipy.sparse.linalg import expm_multiply
from scipy.linalg import eigh_tridiagonal
import sys

PACKAGE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PACKAGE))
from Source.data_locations import generated_outputs_root
from Source.data_locations import shared_data

ROOT = generated_outputs_root() / "Wake_Upwind_Refinement"
INPUT = shared_data("Radial_refinement", "Radial128", "inputs")


def terminal_readout(state):
    """H on phase bit a; Sdag then H on phase bit b, little endian."""
    v = state.reshape(4, -1).copy()
    a = v.copy()
    v[0] = (a[0] + a[1]) / np.sqrt(2)
    v[1] = (a[0] - a[1]) / np.sqrt(2)
    v[2] = (a[2] + a[3]) / np.sqrt(2)
    v[3] = (a[2] - a[3]) / np.sqrt(2)
    v[2:] *= -1j
    a = v.copy()
    v[0] = (a[0] + a[2]) / np.sqrt(2)
    v[2] = (a[0] - a[2]) / np.sqrt(2)
    v[1] = (a[1] + a[3]) / np.sqrt(2)
    v[3] = (a[1] - a[3]) / np.sqrt(2)
    return v


def summary(probability, values, target, H):
    """Return field bias, same-shot covariance variance and nominal accepted shots.

    H is the normalized field metric. A bias at or above 1% leaves no sampling
    budget, represented by shots=None rather than a negative shot count."""
    mean = probability @ values
    covariance = (values.T * probability) @ values - np.outer(mean, mean)
    delta = mean - target
    bias2 = float(delta @ H @ delta)
    variance = float(np.sum(H * covariance))
    target_error = 0.01
    shots = None
    if bias2 < target_error**2:
        shots = max(1, math.ceil(variance / (target_error**2 - bias2)))
        assert bias2 + variance / shots <= target_error**2 + 1e-15
        assert shots == 1 or bias2 + variance / (shots - 1) > target_error**2 - 1e-15
    return dict(
        field_bias_pct=100 * math.sqrt(bias2),
        nominal_variance=variance,
        ideal_accepted_shots_for_1pct=shots,
        implementation_error_in_shot_calculation=0.0,
        mean_r_cos_phase=float(mean[0]),
        mean_r_sin_phase=float(mean[1]),
        mean_r2=float(mean[2]),
    )


def main(grids):
    """Evolve each stencil/grid, verify the continuum reference and export comparisons."""
    ROOT.mkdir(parents=True, exist_ok=True)
    tick = time.monotonic()
    cfg = json.loads((INPUT / "configuration.json").read_text())
    history = np.load(INPUT / "direct_quantum_history.npz")
    read = np.load(INPUT / "joint_readout_inputs.npz")
    amp = history["prepared_amplitude"].copy()
    amp /= np.linalg.norm(amp)
    mu = cfg["parameters"]["mu"]
    gamma = cfg["parameters"]["gamma_normalized"]
    duration = float(history["times"][-1] - history["times"][0])
    target = read["target"]
    H = read["M"].T @ read["G"] @ read["M"] / float(read["norm"]) ** 2
    # Independently regenerate the continuous reference using analytic radial/phase flow.
    refs = []
    widths = []
    for order in [32, 64]:
        nodes, w = leggauss(order)
        r0 = ((np.arange(64)[:, None] + (nodes + 1) / 2) / 64).ravel()
        weights = (abs(amp[:, None]) ** 2 * w[None, :] / 2).ravel()
        D = 1 + r0 * r0 * np.expm1(2 * mu * duration)
        end = r0 * np.exp(mu * duration) / np.sqrt(D)
        phase = gamma * np.log(D) / (2 * mu)
        refs.append(
            weights @ np.c_[end * np.cos(phase), end * np.sin(phase), end * end]
        )
        widths.append(
            [
                math.sqrt(weights @ (r0 * r0) - (weights @ r0) ** 2),
                math.sqrt(weights @ (end * end) - (weights @ end) ** 2),
            ]
        )
    reference_error = float(np.linalg.norm(refs[-1] - target))
    assert reference_error < 2e-11, reference_error
    rows = []
    baselines = []
    for N in grids:
        assert N >= 64 and N & (N - 1) == 0
        r = (np.arange(N) + 0.5) / N
        F = mu * r * (1 - r * r)
        initial = np.repeat(amp, N // 64) / np.sqrt(N // 64)
        values = np.c_[
            np.tile(r, 4) * np.repeat([1, -1, 1, -1], N),
            np.tile(r, 4) * np.repeat([1, 1, -1, -1], N),
            np.tile(r * r, 4),
        ]
        for method in ["centered_face", "centered_NJ", "upwind_KvN"]:
            if method == "centered_face":
                face = np.arange(1, N) / N
                g = mu * face * (1 - face * face) * N / 2
                G = diags([-g, g], [1, -1], format="csr")
                minimum_dissipation = 0.0
            elif method == "centered_NJ":
                g = (F[1:] + F[:-1]) * N / 4
                G = diags([-g, g], [1, -1], format="csr")
                minimum_dissipation = 0.0
            else:
                # Positive-drift upwind KvN amplitudes, not Liouville cell masses.
                # G_jj=-F_j/h, G_j,j-1=(F_j+F_j-1)/(2h); no out-of-grid coupling.
                G = diags([-F * N, (F[1:] + F[:-1]) * N / 2], [0, -1], format="csr")
                minimum_dissipation = float(
                    eigh_tridiagonal(
                        F * N,
                        -(F[1:] + F[:-1]) * N / 4,
                        select="i",
                        select_range=(0, 0),
                    )[0][0]
                )
                assert minimum_dissipation >= -2e-12
            unique = []
            # Only three distinct phase sectors are needed: the two middle sectors
            # have the same weight. Recombine all four before the terminal X/Y readout.
            for phase_weight in [0, 1, 2]:
                generator = G + diags(1j * gamma * phase_weight * r * r)
                unique.append(
                    expm_multiply(
                        generator * duration,
                        initial,
                        traceA=generator.diagonal().sum() * duration,
                    )
                )
            joint = np.array([unique[0], unique[1], unique[1], unique[2]]) / 2
            norm2 = float(np.sum(abs(joint) ** 2))
            assert 0 < norm2 <= 1 + 1e-11
            # Normalize the nonunitary semigroup output before forming observables.
            # Keep its norm separately; it is not a compiled circuit success rate.
            probability = abs(terminal_readout(joint)).ravel() ** 2 / norm2
            data = summary(probability, values, target, H)
            row = dict(
                cells=N,
                address_qubits=N.bit_length() - 1,
                coherent_phase_qubits=2,
                method=method,
                time_start=float(history["times"][0]),
                time_end=float(history["times"][-1]),
                **data,
                semigroup_output_norm_squared=norm2,
                minimum_dissipative_eigenvalue=minimum_dissipation,
                is_compiled_LCHS_success_probability=False,
                physical_resource_optimization=False,
            )
            rows.append(row)
            if method != "centered_NJ":
                np.savez_compressed(
                    ROOT / f"{method}_{N}.npz",
                    r=r,
                    initial_amplitude=initial,
                    final_unnormalized_joint_amplitudes=joint,
                    conditional_probability=probability,
                    observable_values=values,
                    reference=target,
                    H=H,
                )
            print(
                N,
                method,
                "bias%",
                round(data["field_bias_pct"], 8),
                "ideal shots",
                data["ideal_accepted_shots_for_1pct"],
                flush=True,
            )
        if N in [64, 128]:
            native = np.load(
                shared_data(
                    "Radial_refinement",
                    "Radial128",
                    f"grid{N}",
                    "states_and_readout.npz",
                )
            )
            baseline = dict(
                cells=N,
                method="compiled_Walsh_circuit",
                steps=33 * (N // 64),
                **summary(native["probabilities"], native["values"], target, H),
            )
            baselines.append(baseline)
            if N == 64:
                assert abs(baseline["field_bias_pct"] - 0.10720784802292105) < 1e-9
            if N == 128:
                assert abs(baseline["field_bias_pct"] - 0.048855915137143366) < 1e-9
    for filename, data in [
        ("discretization_comparison.csv", rows),
        ("existing_circuit_reference.csv", baselines),
    ]:
        with (ROOT / filename).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    goal = baselines[0]["field_bias_pct"]
    up = [r for r in rows if r["method"] == "upwind_KvN"]
    passing = [r for r in up if r["field_bias_pct"] <= goal]
    first = min(passing, key=lambda r: r["cells"]) if passing else None
    nominal = [r for r in up if r["ideal_accepted_shots_for_1pct"] is not None]
    qa = dict(
        status="PASS",
        same_prepared_continuum_input=True,
        same_drift=True,
        same_phase_readout=True,
        time_interval=[float(history["times"][0]), float(history["times"][-1])],
        reference_regeneration_error=reference_error,
        continuum_quadrature_refinement_error=float(np.linalg.norm(refs[0] - refs[1])),
        mu=mu,
        gamma_normalized=gamma,
        radial_interval=[0, 1],
        boundary_rule="Omit stencil couplings outside the finite cell-centred grid; no periodic wrap",
        continuum_initial_radial_std=widths[-1][0],
        continuum_terminal_radial_std=widths[-1][1],
        continuum_width_ratio=widths[-1][0] / widths[-1][1],
        primary_64_cell_circuit_bias_pct=goal,
        first_tested_upwind_grid_at_that_bias=None if first is None else first["cells"],
        first_tested_upwind_grid_with_bias_below_1pct=min(r["cells"] for r in nominal),
        thresholds_differ="Matching the existing deterministic bias is stricter than meeting the total1% RMS target by adjusting shots.",
        exact_time_semigroup=True,
        upwind_source="Novikau-Joseph arXiv:2410.03985v2 Eq.24",
        upwind_native_circuit_compiled=False,
        matched_end_to_end_resource_advantage_tested=False,
        input_sha256={
            name: hashlib.sha256((INPUT / name).read_bytes()).hexdigest()
            for name in [
                "configuration.json",
                "direct_quantum_history.npz",
                "joint_readout_inputs.npz",
            ]
        },
        elapsed_seconds=time.monotonic() - tick,
    )
    (ROOT / "validation.json").write_text(json.dumps(qa, indent=2) + "\n")
    print(json.dumps(qa, indent=2), flush=True)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument(
        "--grids", type=int, nargs="+", default=[64, 128, 256, 512, 1024, 2048, 4096]
    )
    main(p.parse_args().grids)
