"""Regenerate the four coupled-model trajectories in Supplementary Figure S5.

Inputs
------
The 32 by 32 grid model and retained matrix product state (MPS) preparation records.

Outputs
-------
Generated_Outputs/Regenerated/Fig_S5/, or --output, containing trajectories and time-step checks.

Run from the repository root
----------------------------
python -m Source.KvN.generate_fig_s5_trajectories

Scope
-----
Checks terminal probabilities and reference moments against retained records before exporting trajectories.
"""

import os
from Source.data_locations import generated_outputs_root

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
from pathlib import Path
import json, csv, argparse
import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse.linalg import expm_multiply
from scipy.special import ndtr
from numpy.polynomial.legendre import leggauss

code_root = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(
    description="Generate the 32 by 32 coupled trajectories and step-count check."
)
parser.add_argument(
    "--output", type=Path, default=generated_outputs_root() / "Regenerated/Fig_S5"
)
args = parser.parse_args()
out = args.output.resolve()
out.mkdir(parents=True, exist_ok=True)
os.environ["POLYNOMIAL_VALIDATION_OUTPUT"] = str(out / "model_work")
from Source.data_locations import source_data_root
from Source.Data_Production.data_assets import fixed_asset
from Source.KvN import polynomial_validation as model
from Source.KvN import radial_mps as mps

data = source_data_root()
inputs = fixed_asset("inputs")

# Rebuild the missing boundary preparation using the retained MPS algorithm.
# The resulting terminal probabilities are checked against the Fig. 5 record below.
interior_path = (
    data
    / "Shared_source_data/MPS_KvN_inputs/Figure5_Bias_Analysis/input/grid32/coupled_cubic_2d__interior.npz"
)
with np.load(interior_path) as saved:
    y_amplitude = saved["state_history"][0].reshape(32, 32).sum(axis=1)
y_amplitude /= np.linalg.norm(y_amplitude)
x_probability = np.diff(ndtr((np.linspace(0, 1, 33) - 0.02) / 0.1))
x_probability /= x_probability.sum()
x_amplitude, _, _ = mps.prepare_layers(
    x_probability, min_fidelity=0.9999, max_layers=16, seed=20261604
)
boundary_initial = np.outer(y_amplitude, x_amplitude).ravel()
case = next(c for c in model.declared_cases() if c["name"] == "coupled_cubic_2d")
specs = [
    (
        "interior",
        [0.5, 0.5],
        [0.1, 0.1],
        data
        / "Shared_source_data/MPS_KvN_inputs/Figure5_Bias_Analysis/input/grid32/coupled_cubic_2d__interior.npz",
    ),
    ("boundary_clipped", [0.02, 0.5], [0.1, 0.1], None),
    (
        "near_stable",
        [0.85, 0.15],
        [0.1, 0.1],
        inputs / "polynomial/coupled_cubic_2d__near_stable.npz",
    ),
    (
        "near_unstable",
        [0.02, 0.98],
        [0.04, 0.04],
        inputs / "polynomial/coupled_cubic_2d__near_unstable.npz",
    ),
]
# The published coupled trajectories use 32 cells on each axis and
# 64 steps over T=1. These saved plotting times do not prescribe the separate
# step-count sweep below.
times = np.arange(65) / 64
generator, pairs = model.generator_and_pairs(case, 32)
observables = model.observable_values(model.coordinates(32, 2)).T
widths = np.ptp(observables, axis=0)
squares = np.ptp(observables**2, axis=0)
nodes, w = leggauss(8)
points = ((np.arange(32)[:, None] + (nodes + 1) / 2) / 32).ravel()
weights = []
gaussians = []
states = []
initials = []
for name, mu, sigma, path in specs:
    if path is None:
        first = boundary_initial
    else:
        with np.load(path) as saved:
            first = saved["state_history"][0]
    initials.append(first)
    _, history = model.pair_evolution(first, pairs, 2, 1 / 64, 64)
    states.append(history)
    mass = abs(first.reshape(32, 32)) ** 2
    mass /= mass.sum()
    marginals = [mass.sum(axis=0), mass.sum(axis=1)]
    wp = [(p[:, None] * w / 2).ravel() for p in marginals]
    weights.append(np.outer(*wp).ravel())
    wg = []
    for m, s in zip(mu, sigma):
        norm = ndtr((1 - m) / s) - ndtr(-m / s)
        density = np.exp(-0.5 * ((points - m) / s) ** 2) / (
            s * np.sqrt(2 * np.pi) * norm
        )
        wg.append(np.tile(w / 64, 32) * density)
    gaussians.append(np.outer(*wg).ravel())
positions = np.array(np.meshgrid(points, points, indexing="ij")).reshape(2, -1)


def velocity(t, p):
    return np.asarray(model.drift(case, p.reshape(2, -1))).ravel()


solution = solve_ivp(
    velocity,
    (0, 1),
    positions.ravel(),
    t_eval=times,
    method="DOP853",
    rtol=4e-13,
    atol=4e-15,
)
assert solution.success
# Use the same characteristic trajectories with two initial weight sets:
# piecewise-uniform prepared masses, then analytic Gaussian density. This
# separates transport error from preparation/projection error.
# Columns contain four prepared-input and four analytic-Gaussian quadrature
# weight sets; all use the same characteristic trajectories.
all_weights = np.array(weights + gaussians).T
ref = np.array(
    [
        model.observable_values(solution.y[:, i].reshape(2, -1)) @ all_weights
        for i in range(65)
    ]
)
rows = []
diagnostics = []
sweep = []
for j, (name, mu, sigma, path) in enumerate(specs):
    first = initials[j]
    history = states[j]
    moments = abs(history) ** 2 @ observables
    reference = ref[:, :, j]
    key = "coupled_cubic_2d__" + name
    with np.load(
        inputs / "retained_experiments/budget/grid32" / (key + ".npz")
    ) as archived:
        np.testing.assert_allclose(
            abs(history[-1]) ** 2,
            archived["probability_history"][-1],
            atol=2e-11,
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            reference[-1],
            archived["terminal_reference_moments"],
            atol=3e-12,
            rtol=1e-10,
        )
    for t, a, b in zip(times, moments, reference):
        rows.append(
            dict(
                id=key,
                t=float(t),
                circuit_mean_x=float(a[0]),
                circuit_mean_y=float(a[1]),
                characteristic_mean_x=float(b[0]),
                characteristic_mean_y=float(b[1]),
                mean_error_norm=float(np.linalg.norm(a[:2] - b[:2])),
            )
        )
    target = np.sqrt(
        np.outer(
            *[
                np.diff(ndtr((np.linspace(0, 1, 33) - m) / s))
                / (ndtr((1 - m) / s) - ndtr(-m / s))
                for m, s in zip(mu[::-1], sigma[::-1])
            ]
        ).ravel()
    )
    np.savez_compressed(
        out / (key + ".npz"),
        times=times,
        initial_state=first,
        moment_history=moments,
        prepared_cell_reference_moments=reference,
        analytic_target_reference_moments=ref[:, :, 4 + j],
        preparation_fidelity=float(abs(np.vdot(target, first)) ** 2),
        reference_quadrature_order=8,
        cells_per_coordinate=32,
        steps=64,
    )
    diagnostics.append(
        dict(
            id=key,
            fidelity=float(abs(np.vdot(target, first)) ** 2),
            matched_absolute_RMS=float(
                np.linalg.norm(moments[-1] - reference[-1]) / np.sqrt(5)
            ),
            analytic_absolute_RMS=float(
                np.linalg.norm(moments[-1] - ref[-1, :, 4 + j]) / np.sqrt(5)
            ),
        )
    )
    exact = expm_multiply(generator, first)
    exact_moments = abs(exact) ** 2 @ observables
    norm = np.linalg.norm(reference[-1])
    for steps in [8, 16, 32, 64, 128, 256]:
        final, _ = model.pair_evolution(first, pairs, 2, 1 / steps, steps)
        probability = abs(final) ** 2
        probability /= probability.sum()
        mean = probability @ observables
        bias = mean - reference[-1]
        variance = probability @ ((observables - mean) ** 2)
        # The five moments use a Euclidean metric normalized by the continuum
        # moment-vector norm. Protect bias and variance with total-variation allowance
        # 1e-4, then allocate the remaining 1% squared-RMS budget to sampling.
        b2 = np.sum((abs(bias) + 1e-4 * widths) ** 2) / norm**2
        v = (
            variance.sum()
            + np.sum(1e-4 * (squares + 2 * abs(mean) * widths) + 1e-8 * widths**2)
        ) / norm**2
        shots = int(np.ceil(v / (0.01**2 - b2))) if b2 < 0.01**2 else None
        sweep.append(
            dict(
                id=key,
                cells_per_coordinate=32,
                steps=steps,
                temporal_relative_RMS=float(
                    np.linalg.norm(mean - exact_moments) / norm
                ),
                total_relative_bias=float(np.linalg.norm(bias) / norm),
                relative_bias_bound=float(np.sqrt(b2)),
                accepted_shots=shots,
                step_shot_product=steps * shots if shots else None,
            )
        )
    print(
        name,
        "bias %",
        100 * np.linalg.norm(moments[-1] - reference[-1]) / norm,
        flush=True,
    )
for filename, records in [
    ("coupled_mean_trajectories.csv", rows),
    ("coupled_time_convergence.csv", sweep),
    ("coupled_input_diagnostics.csv", diagnostics),
]:
    with (out / filename).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
print("TIME SWEEP")
for r in sweep:
    if r["id"].endswith("boundary_clipped"):
        print(r, flush=True)
