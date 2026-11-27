"""Recalculate field, joint-sampling and CFD-coordinate summary tables.

Inputs are the deposited reduced-model fields, quantum histories and joint
readout arrays. The finite-shot experiment samples complete joint outcomes, so
the cross-covariance terms are retained. Output CSVs separate deterministic
field discrepancy from finite-shot fluctuations. Use --source-data and --output;
this exporter does not rerun the CFD solver or quantum hardware.
"""

# Direct-file execution must locate Codes/ independently of the working directory.
if __package__ in (None, ""):
    import sys as _path_sys
    from pathlib import Path as _Path
    _codes = next(p for p in _Path(__file__).resolve().parents
                  if (p / "Source/data_locations.py").is_file())
    _path_sys.path.insert(0, str(_codes))


import argparse
from io import BytesIO
from zipfile import ZipFile
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd


def table(output, name, records):
    pd.DataFrame(records).to_csv(output / name, index=False)


def fields(case, out):
    """Reconstruct and compare velocity fields from the archived modal moments."""
    config = json.loads((case / "configuration.json").read_text())
    ref = np.load(case / "unified_reference.npz")
    direct = np.load(case / "direct_quantum_history.npz")
    data_root = case.parents[4]
    with ZipFile(
        data_root / "Shared_source_data/RE_common/BE_inputs/wake_task_inputs.zip"
    ) as archive:
        packed = archive.read("Figure2/unified_run/cases/Re100/data/qsvt_transport.npz")
    with np.load(BytesIO(packed)) as polar:
        np.testing.assert_array_equal(polar["selected_times"], ref["times"])
        moments = {
            "reference": ref["continuum_moments"],
            "direct": direct["direct_moments"],
            "polar": polar["selected_moments"],
        }
    base = np.load(case / "new_model_inputs.npz")
    times = ref["times"]
    series = {"step": np.arange(len(times)), "tstar": times}
    velocity = {}
    for name, m in moments.items():
        xy = (m[:, 0] + 1j * m[:, 1]) * np.exp(1j * ref["known_phase"])
        series[name + "_x"], series[name + "_y"] = xy.real, xy.imag
        series[name + "_z"] = series[name + "_mean_r2"] = m[:, 2]
        series[name + "_squared_mean_radius"] = xy.real**2 + xy.imag**2
        coords = np.column_stack(
            (
                ref["R"] * xy.real,
                ref["R"] * xy.imag,
                ref["c0"] + ref["c2"] * ref["R"] ** 2 * m[:, 2],
            )
        )
        velocity[name] = base["base"] + coords @ base["Phi"]
    table(out, "ensemble_trajectory.csv", series)
    norm = np.linalg.norm(velocity["reference"], axis=1)
    bias = {
        name: 100
        * np.linalg.norm(velocity[name] - velocity["reference"], axis=1)
        / norm
        for name in ("direct", "polar")
    }
    table(
        out,
        "field_bias_by_time.csv",
        [
            dict(algorithm=name, step=i, tstar=t, bias_pct=y)
            for name, values in bias.items()
            for i, (t, y) in enumerate(zip(times, values))
        ],
    )
    inventory = config["gate_verification"]
    table(
        out,
        "direct_refinement.csv",
        [
            dict(
                positive_cells=len(direct["r"]),
                steps=len(times) - 1,
                dtstar=times[1] - times[0],
                bias_pct=bias["direct"][-1],
                max_bias_pct=max(bias["direct"]),
                logical_qubits=inventory["logical_qubits"],
                RZ=inventory["counts"]["RZ"],
                direct_T=inventory["counts"]["native_T"],
                scope="Current baseline only; no new refinement sweep.",
            )
        ],
    )
    with np.load(case / "reconstructed_fields.npz") as snapshots:
        metrics, cfd = [], []
        for name in velocity:
            for step, truth in zip(
                snapshots["steps"], snapshots["cfd_fields"], strict=True
            ):
                field = velocity[name][step]
                metrics.append(
                    dict(
                        method=name,
                        step=step,
                        tstar=times[step],
                        relative_full_velocity_field_bias_pct=100
                        * np.linalg.norm(field - velocity["reference"][step])
                        / norm[step],
                    )
                )
                cfd.append(
                    dict(
                        method=name,
                        tstar=times[step],
                        field_L2_vs_CFD_pct=100
                        * np.linalg.norm(field - truth.ravel())
                        / np.linalg.norm(truth),
                    )
                )
        table(out, "snapshot_metrics.csv", metrics)
        table(out, "snapshot_CFD_metrics.csv", cfd)
    return times, bias["direct"]


def sampling(case, out):
    """Sample joint outcomes and compare finite-shot field errors with covariance predictions."""
    conf = json.loads((case / "configuration.json").read_text())["sampling"]
    repetitions, seed = conf["independent_repetitions"], conf["seed"]
    rng = np.random.default_rng(seed)
    with np.load(case / "joint_readout_inputs.npz") as state:
        # H converts a three-component moment error into squared relative
        # velocity-field error; M maps moments and G supplies the field metric.
        H = state["M"].T @ state["G"] @ state["M"] / float(state["norm"]) ** 2
        draws = {"repeat": np.arange(repetitions)}
        summaries = []
        for name in ("direct", "polar"):
            n = conf["accepted_shots_per_estimate"][name]
            p, y = state[name + "_p"], state[name + "_values"]
            mean = p @ y
            delta = y - mean
            # Use one joint-outcome vector per shot, not independent moment draws.
            # The off-diagonal covariance contributes to the reconstructed field MSE.
            cov = (delta * p[:, None]).T @ delta
            bias = mean - state["target"]
            emse = bias @ H @ bias + np.trace(H @ cov) / n
            e = rng.multinomial(n, p, size=repetitions) @ y / n - state["target"]
            mse = np.maximum(0, np.einsum("ti,ij,tj->t", e, H, e))
            draws[name + "_field_error_pct"] = 100 * np.sqrt(mse)
            row = dict(
                algorithm=name,
                successful_shots=n,
                repetitions=repetitions,
                seed=seed,
                exact_field_bias_pct=100 * np.sqrt(bias @ H @ bias),
                expected_field_RMS_pct=100 * np.sqrt(emse),
                empirical_field_RMS_pct=100 * np.sqrt(mse.mean()),
                field_MSE_zscore=(mse.mean() - emse)
                / (mse.std(ddof=1) / np.sqrt(repetitions)),
            )
            for i, label in enumerate(("cos", "sin", "r2")):
                draws[name + "_" + label + "_error"] = e[:, i]
                row[label + "_bias"] = bias[i]
                row[label + "_sampling_sd"] = np.sqrt(cov[i, i] / n)
            summaries.append(row)
    table(out, "joint_readout_repetitions.csv", draws)
    table(out, "readout_summary.csv", summaries)


def cfd_tracks(case, out, times, bias):
    """Normalize CFD modal trajectories and export the displayed time windows."""
    with np.load(case / "new_model_inputs.npz") as model:
        par = json.loads((case / "configuration.json").read_text())["parameters"]
        R, c0, c2 = (par[k] for k in ("R", "c0", "c2"))

        def normalize(t, a):
            x, y, z = a.T
            return pd.DataFrame(
                dict(
                    tstar=t,
                    CFD_x=x / R,
                    CFD_y=y / R,
                    CFD_z=(z - c0) / (c2 * R**2),
                    CFD_r2=np.sum(a[:, :2] ** 2, axis=1) / R**2,
                )
            )

        training = normalize(model["training_times"], model["training_coordinates"])
        outside = (training.CFD_r2 < 0.5) | (training.CFD_z < 0.4)
        left = np.flatnonzero(outside)[-1] + 1 if outside.any() else 0
        training.to_csv(out / "bc_training_trajectory.csv", index=False)
        training.iloc[left:].loc[lambda df: df.tstar <= times[-1]].to_csv(
            out / "bc_displayed_training_trajectory.csv", index=False
        )
        tp = model["all_times"]
        window = np.r_[times[0], tp[(tp > times[0]) & (tp < times[-1])], times[-1]]
        coords = np.column_stack(
            [np.interp(window, tp, v) for v in model["all_projected_coordinates"].T]
        )
        physical = normalize(window, coords)
        physical[["CFD_a", "CFD_b", "CFD_s"]] = coords
        for name in ("physical_trajectory", "bc_test_trajectory"):
            physical.to_csv(out / (name + ".csv"), index=False)
    # Assemble the plotted errors from the two different normalization conventions.
    root = case.parents[4]
    from Source.Data_Production.data_assets import fixed_asset
    input_file = fixed_asset("inputs", "physical_model_inputs.npz")
    outcomes = pd.read_csv(
        root
        / "Shared_source_data/CFD_ROM/SINDy_Degree/model_comparison/figure4_comparison.csv"
    )
    with np.load(input_file) as model:
        modes, base = model["Phi"], model["base"]
        truth = model["evaluation_fields"].reshape(-1, len(base))
        observed, radial = model["evaluation_coordinates"], model["cubic_coordinates"]
        times_fem = model["evaluation_times"]
        norm = np.linalg.norm(truth, axis=1)

        def error(field):
            return 100 * np.linalg.norm(field, axis=1) / norm

        candidate = outcomes.loc[outcomes.model == "candidate"]
        np.testing.assert_allclose(candidate.time, times_fem, rtol=0, atol=1e-12)
        quadratic = candidate[["a1", "a2", "a3"]].to_numpy()
        table(
            out,
            "physical_error_by_time.csv",
            dict(
                tstar=times_fem,
                projection_pct=error(base + observed @ modes - truth),
                full_quadratic_pct=error((quadratic - observed) @ modes),
                reduced_dynamics_pct=error((radial - observed) @ modes),
                cubic_extra_pct=error((radial - quadratic) @ modes),
                cubic_total_pct=error(base + radial @ modes - truth),
                quadratic_total_pct=error(base + quadratic @ modes - truth),
            ),
        )
        # The archived composite curve uses the earlier deployed fit for its single-flow rows.
        deployed = model["quadratic_coordinates"]
        full, closure = error(base + deployed @ modes - truth), error(
            (radial - deployed) @ modes
        )
        records = []
        for i, t in enumerate(times_fem):
            for label, values, reference in (
                ("3-mode ROM vs CFD", full, "CFD single flow"),
                (
                    "Radial vs 3-mode ROM",
                    closure,
                    "Independently integrated three-variable ROM",
                ),
            ):
                records.append(
                    dict(
                        tstar=t,
                        component=label,
                        error_pct=values[i],
                        comparison=reference,
                        normalization="same-time CFD field norm",
                        target="single flow",
                    )
                )
    records.extend(
        dict(
            tstar=t,
            component="Walsh-KvN vs radial",
            error_pct=y,
            comparison="continuum ROM ensemble",
            normalization="same-time continuum ensemble mean field norm",
            target="ensemble mean",
        )
        for t, y in zip(times, bias)
    )
    table(out, "time_resolved_errors.csv", records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    case = args.source_data / "Shared_source_data/CFD_ROM/CFD_ROM/Re100/data"
    times, bias = fields(case, args.output)
    sampling(case, args.output)
    cfd_tracks(case, args.output, times, bias)
