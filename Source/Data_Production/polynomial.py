"""Export polynomial-density diagnostics from the fixed prepared inputs.

The model protocol supplies polynomial coefficients; retained states supply
initial/final circuit probabilities. Backtracked characteristics integrate the
same piecewise-uniform initial masses to provide continuum cell references.
The generator writes comparison CSVs in --output; it does not redraw figures.
Use the csv_generators.rebuild or verify dispatcher for the full dependency order.
"""

# Direct-file execution must locate Codes/ independently of the working directory.
if __package__ in (None, ""):
    import sys as _path_sys
    from pathlib import Path as _Path
    _codes = next(p for p in _Path(__file__).resolve().parents
                  if (p / "Source/data_locations.py").is_file())
    _path_sys.path.insert(0, str(_codes))


import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
try:
    from Source.Data_Production.data_assets import fixed_asset
except ModuleNotFoundError:
    from data_assets import fixed_asset

HERE = Path(__file__).resolve().parent


def save(out, name, records):
    with (out / name).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def split_final(state, coeff):
    """Apply the 64-step symmetric pair split to one polynomial input."""
    size = state.size
    amp = state.copy()
    for _ in range(64):
        for parity, weight in ((0, 0.5), (1, 1.0), (0, 0.5)):
            left = np.arange(parity, size - 1, 2)
            angle = (
                np.polynomial.polynomial.polyval((left + 1) / size, coeff)
                * size
                * weight
                / (2 * 64)
            )
            a, b = amp[left].copy(), amp[left + 1].copy()
            amp[left] = np.cos(angle) * a - np.sin(angle) * b
            amp[left + 1] = np.sin(angle) * a + np.cos(angle) * b
    return amp


def endpoints(data, key, coeff):
    """Load the actual prepared input and its corresponding final circuit probability."""
    base = data / "Shared_source_data"
    if key.endswith("__boundary_clipped"):
        path = (
            base
            / "Radial_refinement/Grid_Oscillation_Diagnostics/inputs/quintic_multistable__boundary_clipped.npz"
        )
        with np.load(path) as z:
            first = z["state_history"][0]
            last = (
                z["state_history"][-1]
                if key.startswith("quintic_")
                else split_final(first, coeff)
            )
    else:
        if key == "coupled_cubic_2d__interior":
            path = (
                base
                / "MPS_KvN_inputs/Figure5_Bias_Analysis/input/grid32"
                / (key + ".npz")
            )
        else:
            from Source.Data_Production.data_assets import fixed_asset
            path = fixed_asset("inputs", "polynomial", key + ".npz")
        with np.load(path) as z:
            first, last = z["state_history"][[0, -1]]
    return abs(first) ** 2, abs(last) ** 2


def backtrack(coeff, p):
    """Integrate edges backward and difference the initial cumulative mass."""
    edges = np.linspace(0, 1, len(p) + 1)

    def drift(time, x):
        speed = np.polynomial.polynomial.polyval(x, coeff)
        speed[[0, -1]] = 0.0
        return speed

    def propagate(interval, x):
        sol = solve_ivp(drift, interval, x, method="DOP853", rtol=3e-13, atol=3e-15)
        if not sol.success:
            raise ValueError(sol.message)
        return sol

    reverse = propagate((1.0, 0.0), edges)
    back = reverse.y[:, -1]
    cdf = np.r_[0.0, np.cumsum(p)]
    # Conserve cell mass by evaluating the initial cumulative distribution
    # at backtracked final edges. The reference uses the same prepared masses.
    mass = np.diff(np.interp(np.clip(back, 0, 1), edges, cdf))
    qa = dict(
        backflow_nfev=reverse.nfev,
        minimum_final_cell_mass=mass.min(),
        mass_defect=mass.sum() - p.sum(),
        initial_CDF_recovery_max=np.max(abs(np.diff(cdf) - p)),
        minimum_adjacent_backflow_gap=np.diff(back).min(),
        inverse_forward_edge_residual=np.max(
            abs(propagate((0.0, 1.0), back).y[:, -1] - edges)
        ),
    )
    return back, mass, qa


def generate(data, out):
    """Write cell-mass and moment diagnostics for the declared displayed inputs."""
    protocol = json.loads(
        (
            data / "Figures/Fig_5/plot_inputs/panels/a_moments/accuracy_protocol.json"
        ).read_text()
    )
    from Source.Data_Production.data_assets import fixed_asset
    polys = json.loads(fixed_asset("model_protocol.json").read_text())
    coeff = {
        x["name"]: np.array(x["raw_coefficients"]) * x["normalization_scale"]
        for x in polys[:3]
    }
    models = dict(
        zip(
            ("quadratic", "cubic_interior", "quintic_multistable", "coupled_cubic_2d"),
            ("Q2", "C3", "Q5", "2D C3"),
        )
    )
    samples, density, metrics = [], [], []
    for rec in protocol["sources"]:
        key = rec["id"]
        model, ic = key.split("__")
        if ic == "boundary_clipped" and model == "coupled_cubic_2d":
            continue
        p0, p = endpoints(data, key, coeff.get(model))
        if model in coeff:
            n = len(p)
            back, ref_mass, qa = backtrack(coeff[model], p0)
            for j in range(n):
                density.append(
                    dict(
                        id=key,
                        cell=j,
                        left=j / n,
                        right=(j + 1) / n,
                        initial_prepared_mass=p0[j],
                        final_circuit_mass=p[j],
                        final_matched_characteristic_mass=ref_mass[j],
                        backflow_left=back[j],
                        backflow_right=back[j + 1],
                    )
                )
            metrics.append(
                dict(
                    id=key,
                    cell_mass_total_variation=0.5 * abs(p - ref_mass).sum(),
                    circuit_initial_mass=p0.sum(),
                    circuit_final_mass=p.sum(),
                    reference_final_mass=ref_mass.sum(),
                    **qa
                )
            )
            x = (np.arange(n) + 0.5) / n
            values = np.column_stack((x, x * x))
        else:
            n = round(np.sqrt(len(p)))
            axes = (np.arange(n) + 0.5) / n
            x, y = [a.ravel() for a in np.meshgrid(axes, axes, indexing="xy")]
            values = np.column_stack((x, y, x * x, y * y, x * y))
        if ic == "boundary_clipped":
            continue
        p = p / p.sum()
        mu = p @ values
        np.testing.assert_allclose(mu, rec["outcome_mean"], rtol=1e-10, atol=1e-12)
        ref = np.array(rec["reference"])
        norm = np.linalg.norm(ref)
        nshots = protocol["panel_a_shots"]
        repeat = protocol["repetitions"]
        seed = (
            int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")
            ^ 20261111
        ) % 2**63
        counts = np.random.default_rng(seed).multinomial(nshots, p, size=repeat)
        e = np.linalg.norm(counts @ values / nshots - ref, axis=1) / norm
        lo, mid, hi = np.quantile(e, [0.025, 0.5, 0.975])
        bias = np.linalg.norm(mu - ref) / norm
        variance = (p @ ((values - mu) ** 2)).sum() / norm**2
        # Expected squared relative moment error = squared bias +
        # trace of the single-shot covariance / (shots * reference norm**2).
        predicted = bias**2 + variance / nshots
        mse = e**2
        samples.append(
            dict(
                id=key,
                model=models[model],
                input=ic,
                cells_per_coordinate=n,
                components=len(ref),
                shots=nshots,
                repetitions=repeat,
                seed=seed,
                deterministic_relative_RMS=bias,
                sampled_relative_RMS_median=mid,
                sampled_relative_RMS_lower=lo,
                sampled_relative_RMS_upper=hi,
                analytic_expected_relative_RMS_squared=predicted,
                empirical_mean_relative_RMS_squared=mse.mean(),
                sampling_condition="fixed common benchmark shot count, not accuracy-derived",
                MSE_check_z=abs(mse.mean() - predicted)
                / (mse.std(ddof=1) / np.sqrt(repeat)),
            )
        )
    save(out, "sampling_errorbars.csv", samples)
    save(out, "scalar_conservative_cell_masses.csv", density)
    save(out, "scalar_density_metrics.csv", metrics)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    generate(args.source_data, args.output)
