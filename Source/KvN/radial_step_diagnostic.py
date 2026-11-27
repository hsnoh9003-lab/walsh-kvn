"""Validate the step/shot trade-off without changing the 33-step benchmark."""

# Direct-file execution must locate Codes/ independently of the working directory.
if __package__ in (None, ""):
    import sys as _path_sys
    from pathlib import Path as _Path
    _codes = next(p for p in _Path(__file__).resolve().parents
                  if (p / "Source/data_locations.py").is_file())
    _path_sys.path.insert(0, str(_codes))


from Source.data_locations import generated_outputs_root

from pathlib import Path
import json, csv, numpy as np

ROOT = generated_outputs_root() / "KvN"


def main(case):
    CASE = Path(case)
    d = np.load(CASE / "direct_quantum_history.npz")
    metric_data = np.load(CASE / "joint_readout_inputs.npz")
    cfg = json.loads((CASE / "configuration.json").read_text())["parameters"]
    r = d["r"]
    h = 1 / len(r)
    duration = float(d["times"][-1] - d["times"][0])
    initial = d["direct_history"][0].copy()
    metric = (
        metric_data["M"].T
        @ metric_data["G"]
        @ metric_data["M"]
        / float(metric_data["norm"]) ** 2
    )
    reference = metric_data["target"]
    rows = []
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    U = np.kron(H @ np.diag([1, -1j]), H)
    branch = np.arange(4)
    values = np.column_stack(
        [
            np.tile(r, 4) * np.repeat(1 - 2 * (branch & 1), len(r)),
            np.tile(r, 4) * np.repeat(1 - 2 * (branch >> 1), len(r)),
            np.tile(r * r, 4),
        ]
    )
    for steps in [8, 16, 33, 66, 132]:
        state = initial.copy()
        dt = duration / steps
        phase = np.exp(
            1j
            * np.array([0, 1, 1, 2])[:, None]
            * cfg["gamma_normalized"]
            * r[None, :] ** 2
            * dt
            / 2
        )
        for _ in range(steps):
            state *= phase
            for parity, fraction in [(0, 0.5), (1, 1.0), (0, 0.5)]:
                first = np.arange(parity, len(r) - 1, 2)
                face = (first + 1) * h
                angle = fraction * dt * cfg["mu"] * face * (1 - face * face) / (2 * h)
                left = state[:, first].copy()
                right = state[:, first + 1].copy()
                state[:, first] = np.cos(angle) * left - np.sin(angle) * right
                state[:, first + 1] = np.sin(angle) * left + np.cos(angle) * right
            state *= phase
        probability = abs(U @ state) ** 2
        probability = probability.ravel()
        mean = probability @ values
        cov = (values.T * probability) @ values - np.outer(mean, mean)
        delta = mean - reference
        bias2 = float(delta @ metric @ delta)
        variance = float(np.trace(metric @ cov))
        room = 0.01**2 - bias2
        shots = None if room <= 0 else int(np.ceil(variance / room))
        if steps == 33:
            assert np.max(abs(mean - d["direct_moments"][-1])) < 1e-12
            assert abs(100 * np.sqrt(bias2) - 0.10720784802311842) < 1e-9
            assert shots == 813
        rows.append(
            dict(
                steps=steps,
                field_bias_percent=100 * np.sqrt(bias2),
                per_shot_relative_field_variance=variance,
                ideal_shots_for_one_percent=shots,
                implementation_error_included=False,
            )
        )
    target = ROOT / "radial_step_check"
    target.mkdir(exist_ok=True)
    with (target / "step_shot_tradeoff.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (target / "README.txt").write_text(
        "Diagnostic only: same MPS input, interval and noiseless 1% terminal RMS target. Implementation error is not included in these shot counts. The reported 33-step benchmark uses the protected 815-shot count and is unchanged. No BE or hardware reoptimization is implied.\n"
    )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Noiseless radial transport step/shot diagnostic"
    )
    parser.add_argument(
        "--case", type=Path, required=True, help="F/F2 original Re100/data path"
    )
    args = parser.parse_args()
    main(args.case)
