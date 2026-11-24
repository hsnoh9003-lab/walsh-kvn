"""Reproduce the 16- and 33-step, 64-cell wake comparison in Methods."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))

from Reproduction.auxiliary.radial128.reproduce_flow import run
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import expm_multiply
from Source.RE.geometry import raw_geometry
from Source.RE.optimization import select_work
from Source.RE.preparation import load_workloads, make_config
from Source.RE.uncapped import find_fastest
from Source.data_locations import shared_data, published_path


def exact_grid_probabilities(history, parameters):
    """Exact finite-grid transport, including the two phase readouts."""
    radii = history["r"]
    cells = len(radii)
    duration = float(np.ptp(history["times"]))
    faces = np.arange(1, cells) / cells
    couplings = parameters["mu"] * cells * faces * (1 - faces**2) / 2
    generator = diags((couplings, -couplings), (-1, 1), format="csr")
    initial = history["direct_history"][0].reshape(4, cells)
    phase_rate = 1j * parameters["gamma_normalized"] * radii**2
    exact_state = np.stack([
        expm_multiply(duration * (generator + diags(weight * phase_rate)), amplitudes)
        for weight, amplitudes in zip((0, 1, 1, 2), initial)
    ])
    hadamard = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    readout = np.kron(hadamard @ np.diag([1, -1j]), hadamard)
    return abs(readout @ exact_state).ravel()**2


def main(output):
    inputs = shared_data('Radial_refinement', 'Radial128', 'inputs')
    archive = shared_data('RE_common', 'RE')
    diagnostics = shared_data('RE_common', 'Numerical_Details', 'radial_step_tradeoff')
    config = json.loads((inputs / "configuration.json").read_text())
    circuit = json.loads((inputs / "direct_64_33_circuit.json").read_text())
    workload = load_workloads(archive / "workloads_v38.json")["direct"]
    physical_error = 0.0005
    prep_table = archive / "prep_p5.csv"

    with (archive / "v38_reference_selected.csv").open(newline="") as stream:
        seed = next(r for r in csv.DictReader(stream) if r["model"] == "direct"
                    and float(r["p_phys"]) == physical_error and r["selection"] == "fastest_uncapped")
    with (diagnostics / "step_shot_tradeoff.csv").open(newline="") as stream:
        ideal_shots = {int(r["steps"]): int(r["ideal_shots_for_one_percent"]) for r in csv.DictReader(stream)}

    results = []
    with TemporaryDirectory() as work, np.load(inputs / "direct_quantum_history.npz") as history, \
            np.load(inputs / "joint_readout_inputs.npz") as observable:
        exact = exact_grid_probabilities(history, config["parameters"])
        for steps in (33, 16):
            name = f"walsh_{steps}"
            out = Path(work) / name
            report = run(6, steps, config, circuit, history, observable, output_dir=out)
            with np.load(out / "states_and_readout.npz") as data:
                metric, values = data["H"], data["values"]
                time_difference = (exact - data["probabilities"]) @ values
                space_difference = exact @ values - data["reference"]
            splitting = float(100 * np.sqrt(time_difference @ metric @ time_difference))
            spatial = float(100 * np.sqrt(space_difference @ metric @ space_difference))
            budget = report["budget"]
            bias_percent = budget["noiseless_field_bias_pct"]
            ideal = int(np.ceil(budget["nominal_variance"] / (0.01**2 - (bias_percent / 100)**2)))
            assert ideal == ideal_shots[steps]
            rotations, native = report["counts"]["RZ"], report["counts"]["native_T"]
            assert rotations == 19 + 78 * steps + 65 * (steps + 1) and native == 224 * steps
            task = dict(workload, RZ=rotations, direct_T=native, shots=budget["shots"])

            # Start from the published 33-step architecture to bound the runtime.
            # The search then reoptimizes distances, factories, block and synthesis.
            config_re = make_config(physical_error, prep_table, seed["block"], task)
            geometry = raw_geometry(physical_error, seed["block"], config_re, int(seed["d_core"]),
                                    tuple(map(int, seed["distances"].split(":"))),
                                    tuple(map(int, seed["u_levels"].split(":"))))
            same_architecture = select_work(geometry, name, task)
            best, _ = find_fastest(name, task, same_architecture, prep_table)
            if steps == 33:
                assert abs(best["complete_task_seconds"] - float(seed["complete_task_seconds"])) < 1e-6
                assert best["physical_qubits"] == int(seed["physical_qubits"])
            results.append(dict(
                steps=steps, arbitrary_rotations=rotations, native_T=native, ideal_shots=ideal,
                protected_shots=budget["shots"], total_bias_percent=bias_percent,
                spatial_bias_percent=spatial, splitting_error_percent=splitting,
                per_attempt_error=best["E_total"], physical_qubits=best["physical_qubits"],
                T_per_rotation=best["N_T"], T_per_attempt=best["T_count"],
                task_hours=best["complete_task_seconds"] / 3600,
                fixed_33_architecture_hours=same_architecture["complete_task_seconds"] / 3600,
                fixed_33_architecture_qubits=same_architecture["physical_qubits"],
                core_distance=best["d_core"], distillation_distances=best["distances"],
                factory_units=best["u_levels"],
            ))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="Comparison CSV")
    main(parser.parse_args().output)
