"""Evaluate centred radial time-step candidates and their complete task costs.

Inputs
------
Prepared radial state, physical model, continuum reference and resource calibration records.

Outputs
-------
Generated_Outputs/Radial_Time_Selection/, or --output, with step candidates, checks and selected resources.

Run from the repository root
----------------------------
python -m Source.KvN.radial_time_selection

Scope
-----
Only tested counts are compared. The archived 33-step input supplies the state/model; it does not impose an upwind step criterion.
"""

from __future__ import annotations

# Direct-file execution must locate Codes/ independently of the working directory.
if __package__ in (None, ""):
    import sys as _path_sys
    from pathlib import Path as _Path
    _codes = next(p for p in _Path(__file__).resolve().parents
                  if (p / "Source/data_locations.py").is_file())
    _path_sys.path.insert(0, str(_codes))

from Source.data_locations import generated_outputs_root
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from Source.KvN.radial_refinement_flow import (
    pair_step,
    field_budget,
    run,
)
from Source.KvN.radial_step_comparison import exact_grid_probabilities
from Source.data_locations import shared_data
from Source.RE.preparation import load_workloads, make_config, write_csv
from Source.RE.geometry import raw_geometry
from Source.RE.optimization import select_work, select_designs
from Source.RE.uncapped import find_fastest

COUNTS = tuple(range(1, 65)) + (66, 80, 96, 128, 132, 256)


def evaluate(steps, history, observable, parameters):
    """Recompute final bias and protected shot budget at one tested step count."""
    size = len(history["r"])
    n = int(np.log2(size))
    duration = float(np.ptp(history["times"]))
    state = history["direct_history"][0].copy().ravel()
    for _ in range(steps):
        state = pair_step(
            state, n, duration / steps, parameters["mu"], parameters["gamma_normalized"]
        )
    hadamard = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    readout = np.kron(hadamard @ np.diag([1, -1j]), hadamard)
    probabilities = abs(readout @ state.reshape(4, size)) ** 2
    probabilities = probabilities.ravel() / probabilities.sum()
    r = history["r"]
    values = np.c_[
        np.tile(r, 4) * np.repeat([1, -1, 1, -1], size),
        np.tile(r, 4) * np.repeat([1, 1, -1, -1], size),
        np.tile(r * r, 4),
    ]
    metric = (
        observable["M"].T
        @ observable["G"]
        @ observable["M"]
        / float(observable["norm"]) ** 2
    )
    exact = exact_grid_probabilities(history, parameters)
    # Use the same finite-grid generator at exact time to separate spatial
    # and splitting contributions. These are vectors: their norms need not add,
    # because spatial and temporal errors can partially cancel.
    spatial = exact @ values - observable["target"]
    temporal = (probabilities - exact) @ values
    bias = probabilities @ values - observable["target"]
    covariance = (values.T * probabilities) @ values - np.outer(
        probabilities @ values, probabilities @ values
    )
    variance = float(np.trace(metric @ covariance))
    record = dict(
        steps=steps,
        dt=duration / steps,
        field_bias_percent=100 * np.sqrt(bias @ metric @ bias),
        spatial_bias_percent=100 * np.sqrt(spatial @ metric @ spatial),
        splitting_error_percent=100 * np.sqrt(temporal @ metric @ temporal),
        per_shot_relative_field_variance=variance,
        ideal_shots_for_one_percent=None,
        protected_shots=None,
        feasible=False,
    )
    # Sampling can use only the squared-error budget left after deterministic
    # bias. A smaller step count can therefore increase the required shots.
    room = 0.01**2 - float(bias @ metric @ bias)
    if room > 0:
        record["ideal_shots_for_one_percent"] = int(np.ceil(variance / room))
        # The implementation allowance is the same normalized diamond-distance
        # bound as in the original resource model. It also bounds output TV.
        try:
            budget = field_budget(probabilities, values, observable["target"], metric)
        except (AssertionError, ValueError, ZeroDivisionError):
            budget = None
        if budget is not None:
            record.update(
                feasible=True,
                protected_shots=budget["shots"],
                protected_bias_bound=budget["protected_bias_bound"],
                protected_variance_bound=budget["protected_variance_bound"],
                RMS_bound=budget["RMS_bound"],
                RMS_bound_at_one_fewer=budget["RMS_bound_at_one_fewer"],
            )
    return record


def main(output, counts=COUNTS, rates=range(2, 10), verify=True):
    """Optimize physical runtime for each tested count and retain the main 33-step benchmark."""
    output.mkdir(parents=True, exist_ok=True)
    inputs = shared_data("Radial_refinement", "Radial128", "inputs")
    archive = shared_data("RE_common", "RE")
    config = json.loads((inputs / "configuration.json").read_text())
    circuit = json.loads((inputs / "direct_64_33_circuit.json").read_text())
    history = np.load(inputs / "direct_quantum_history.npz")
    observable = np.load(inputs / "joint_readout_inputs.npz")
    numerical = [evaluate(n, history, observable, config["parameters"]) for n in counts]
    write_csv(output / "time_convergence.csv", numerical)
    workload = load_workloads(archive / "workloads.json")["direct"]
    # These fixed architectures are seeds only; the uncapped search uses lower
    # bounds to test all architectures that can beat each feasible seed.
    from Source.Data_Production.data_assets import fixed_asset
    seeds = json.loads(fixed_asset("catalogue", "radial_search_seeds.json").read_text())
    resources, certificates = [], {}
    for rate in rates:
        p = rate / 10000
        seed = next(r for r in seeds if float(r["p_phys"]) == p)
        prep = archive / f"prep_p{rate}.csv"
        for row in numerical:
            if not row["feasible"]:
                continue
            steps = row["steps"]
            task = dict(
                workload,
                RZ=19 + 78 * steps + 65 * (steps + 1),
                direct_T=224 * steps,
                shots=row["protected_shots"],
                steps=steps,
            )
            cfg = make_config(p, prep, seed["block"], task)
            geometry = raw_geometry(
                p,
                seed["block"],
                cfg,
                int(seed["d_core"]),
                tuple(map(int, seed["distances"].split(":"))),
                tuple(map(int, seed["u_levels"].split(":"))),
            )
            incumbent = select_work(geometry, "direct", task)
            if incumbent is None:
                _, best, _, _, certificate = select_designs("direct", p, prep, task)
            else:
                best, certificate = find_fastest("direct", task, incumbent, prep)
            resources.append(
                dict(best, steps=steps, task_hours=best["complete_task_seconds"] / 3600)
            )
            certificates[f"p{rate}_steps{steps}"] = certificate
        print("Completed physical error probability", p, flush=True)
    write_csv(output / "candidate_fastest.csv", resources)
    # Select by total task time, including accepted shots and optimized
    # hardware; this is the fastest of tested counts, not a proof over all Nt.
    winners = [
        min(
            (r for r in resources if r["p_phys"] == rate / 10000),
            key=lambda r: (r["complete_task_seconds"], r["physical_qubits"]),
        )
        for rate in rates
    ]
    write_csv(output / "fastest_tested_by_rate.csv", winners)
    reference = next(r for r in winners if r["p_phys"] == 0.0005)
    selected = reference["steps"]
    report = dict(
        selection="shortest total task time among tested counts at p=5e-4",
        tested_steps=list(counts),
        physical_error_probabilities=[r / 10000 for r in rates],
        selected_steps=selected,
        implementation_error_allowance=1e-4,
        final_relative_field_RMS_target=0.01,
        comparison="same signed MPS, 64 cells, model, duration and continuum reference",
        global_integer_optimality_claimed=False,
        scope="additional comparison; the main 33-step benchmark is retained",
        selected_numerics=next(r for r in numerical if r["steps"] == selected),
        reference_fastest=reference,
        certificates=certificates,
    )
    if verify:
        native = run(
            6,
            selected,
            config,
            circuit,
            history,
            observable,
            output_dir=output / "selected_gate_replay",
        )
        assert native["counts"]["RZ"] == reference["RZ"]
        assert native["counts"]["native_T"] == reference["native_T"]
        assert native["budget"]["shots"] == reference["successful_shots"]
        report["gate_verification"] = native
    selected_workload = dict(
        workload,
        RZ=reference["RZ"],
        direct_T=reference["native_T"],
        shots=reference["successful_shots"],
        steps=selected,
    )
    minimum, fastest, capped, _, full_certificate = select_designs(
        "radial8" if selected == 8 else f"radial{selected}",
        0.0005,
        archive / "prep_p5.csv",
        selected_workload,
    )
    assert (
        abs(fastest["complete_task_seconds"] - reference["complete_task_seconds"])
        < 1e-8
    )
    assert fastest["physical_qubits"] == reference["physical_qubits"]
    write_csv(output / "selected_designs_p5.csv", [minimum, fastest, *capped])
    report["full_reference_search_certificate"] = full_certificate
    (output / "selection.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        "Selected",
        selected,
        "steps;",
        reference["successful_shots"],
        "shots;",
        reference["task_hours"],
        "hours",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=generated_outputs_root() / "Radial_Time_Selection"
    )
    parser.add_argument("--steps", type=int, nargs="+", default=COUNTS)
    parser.add_argument("--rates", type=int, nargs="+", default=list(range(2, 10)))
    parser.add_argument("--skip-gate-verification", action="store_true")
    args = parser.parse_args()
    main(args.output, args.steps, args.rates, not args.skip_gate_verification)
