#!/usr/bin/env python3
"""Render Figure 6 with the full polar Fourier-coordinate wake simulations."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
BASELINE = ROOT / "baseline"
sys.path.insert(0, str(BASELINE / "Figure6"))

from build_figure6_compact import (
    BLUE, HEIGHT, ORANGE, WIDTH, columns, coupled_resources, model_accuracy, plt,
    quintic_density, records, wake_means, wake_resources,
)

START = 29.8
RESTART = 40.0
END = 49.82906260206943
REFERENCE_TOLERANCE = 1e-8
REFERENCE_COLUMNS = [f"reference_mean_{axis}" for axis in "xyz"]
MEAN_COLUMNS = [f"walsh_mean_{axis}" for axis in "xyz"]


def validate_resources(path, terminal_percent, expected_shots):
    rows = records(path)
    expected = {(p, mode) for p in range(2, 10)
                for mode in ("minimum_qubits", "fastest_uncapped")}
    actual = set()
    for row in rows:
        rate = 1e4 * float(row["p_phys"])
        if not np.isclose(rate, round(rate), atol=1e-8, rtol=0):
            raise ValueError(f"Invalid physical error rate in {path}")
        actual.add((round(rate), row["selection"]))
    if len(rows) != len(expected) or actual != expected:
        raise ValueError(f"Incomplete or duplicate resources in {path}")
    for row in rows:
        bias = float(row["noiseless_deterministic_relative_field_bias"])
        bound = float(row["expected_relative_field_rms_bound"])
        shots = int(row["successful_shots"])
        if not np.isclose(terminal_percent / 100, bias, atol=1e-12, rtol=0):
            raise ValueError(f"Terminal error and resource bias differ in {path}")
        if not bias <= bound <= 0.01 or shots != expected_shots:
            raise ValueError(f"The archived 1% terminal field-error budget does not match {path}")
        qubits = int(row["physical_qubits"])
        days = float(row["complete_task_days"])
        if qubits <= 0 or not np.isfinite(days) or days <= 0:
            raise ValueError(f"Invalid resource count or task time in {path}")


def load_runs(manifest):
    folder = manifest.resolve().parent
    config = json.loads(manifest.read_text())
    shared_reference = columns(folder / config["reference"], ["time", *REFERENCE_COLUMNS]).T
    if not np.isfinite(shared_reference).all() or np.any(np.diff(shared_reference[:, 0]) <= 0):
        raise ValueError("The common continuum reference must have strictly increasing times")
    np.testing.assert_allclose(shared_reference[[0, -1], 0], [START, END],
                               atol=REFERENCE_TOLERANCE, rtol=0)
    if len(config["runs"]) != 2:
        raise ValueError("Figure 6 requires both archived wake runs")
    runs = []
    for record, (start, color, marker, action) in zip(config["runs"], (
        (START, BLUE, "o", "start"),
        (RESTART, ORANGE, "D", "restart"),
    )):
        steps = record["steps"]
        label = f"{start:g} {action}, {steps} steps"
        if not isinstance(steps, int) or steps <= 0:
            raise ValueError(f"Unexpected step count for {label}")
        data = columns(folder / record["trajectory"], [
            "time", *MEAN_COLUMNS, "field_relative_percent", *REFERENCE_COLUMNS,
        ]).T
        if not np.isfinite(data).all() or np.any(np.diff(data[:, 0]) <= 0):
            raise ValueError(f"Invalid trajectory for {label}")
        np.testing.assert_allclose(data[[0, -1], 0], [start, END],
                                   atol=REFERENCE_TOLERANCE, rtol=0)
        sampled_reference = data[:, [0, 5, 6, 7]]
        input_reference = columns(folder / record.get("reference", config["reference"]),
                                  ["time", *REFERENCE_COLUMNS]).T
        indices = np.searchsorted(input_reference[:, 0], sampled_reference[:, 0] - REFERENCE_TOLERANCE)
        np.testing.assert_allclose(sampled_reference, input_reference[indices],
                                   atol=REFERENCE_TOLERANCE, rtol=0)
        path = folder / record["resources"]
        validate_resources(path, data[-1, 4], record["shots"])
        runs.append((data[:, 0], data[:, 1:4].T, data[:, 4], color, marker, label, path))
    return shared_reference[:, 0], shared_reference[:, 1:].T, runs


def build(manifest, output):
    time, reference, runs = load_runs(manifest)
    fig = plt.figure(figsize=(WIDTH / 72, HEIGHT / 72), facecolor="white")
    model_accuracy(fig, BASELINE)
    quintic_density(fig, BASELINE)
    coupled_resources(fig, BASELINE)
    wake_means(fig, time, reference, runs)
    wake_resources(fig, time, runs)
    output.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"savefig.bbox": None}):
        fig.savefig(output / "Figure6.pdf", metadata={"CreationDate": None, "ModDate": None})
        fig.savefig(output / "Figure6.svg", metadata={"Date": None})
        fig.savefig(output / "Figure6.png", dpi=450)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "runs.json")
    parser.add_argument("--output", type=Path, default=ROOT / "out")
    args = parser.parse_args()
    build(args.manifest, args.output)
