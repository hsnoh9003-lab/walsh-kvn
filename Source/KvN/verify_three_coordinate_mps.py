"""Regenerate both full-size signed MPS inputs and replay their native gates.

All preparation gates act on the entire 17-/21-qubit state vector. The two
endpoint Fourier circuits are also replayed gate by gate. Transport still
uses the documented equivalent matching blocks; this does not claim a
gate-by-gate run of the complete transport circuit.
"""

# Direct-file execution must locate Codes/ independently of the working directory.
if __package__ in (None, ""):
    import sys as _path_sys
    from pathlib import Path as _Path
    _codes = next(p for p in _Path(__file__).resolve().parents
                  if (p / "Source/data_locations.py").is_file())
    _path_sys.path.insert(0, str(_codes))


from Source.data_locations import generated_outputs_root

from pathlib import Path
from types import SimpleNamespace
from collections import Counter
import argparse, gzip, json, time
import numpy as np
from Source.ROM.model import load
from Source.ROM.ensemble import make_input
from Source.ROM.field_budget import field_budget
from .mps import compress
from . import compile_mps, count_polar, transport
from .wake_gate_core import inverse, phase_aligned_error, apply_gates
from .native_replay import run as native

ROOT = Path(__file__).resolve().parents[2]


def complete_program(base, blocks):
    program, error = compile_mps.base_preparation(base)
    for block in reversed(blocks):
        u = block["basis"]
        w = u.shape[0].bit_length() - 1
        offset = block["m"] - w
        rotations, gray, error = compile_mps.decompose_isometry(u)
        # Invert the logical Givens rotations. MCX is exactly self-inverse,
        # so reuse its archived forward decomposition in both directions.
        forward = inverse(gray)
        for a, b, theta in reversed(rotations):
            forward += compile_mps.controlled_rotation(w, a, b, -theta)
        program += [
            (g, [offset + q for q in wires], angle) for g, wires, angle in forward
        ]
    return program


def verify(run, output=None, propagate=True):
    from Source.data_locations import shared_data, figure_data

    out = (
        Path(output)
        if output
        else generated_outputs_root() / "Three_Coordinate_MPS_Replay" / run
    )
    out.mkdir(parents=True, exist_ok=True)
    tick = time.monotonic()
    source = shared_data("MPS_KvN_inputs", "KvN", "inputs", run)
    config = json.loads((source / "protocol.json").read_text())
    target, normalization = make_input(config, order=12)
    for chi in [2, 4, 8, 16, 32, 64]:
        amplitude, blocks, base, preparation = compress(target.ravel(), chi)
        if preparation["fidelity"] >= 0.9999:
            break
    assert chi == 4, (run, chi, preparation)
    program = complete_program(base, blocks)
    inventory = json.loads((source / "compiled_inventory.json").read_text())
    actual_counts = Counter(g for g, q, a in program)
    assert actual_counts == Counter(inventory["by_gate"]), (
        run,
        actual_counts,
        inventory["by_gate"],
    )
    rng = np.random.default_rng(261116)
    test = rng.normal(size=32) + 1j * rng.normal(size=32)
    test /= np.linalg.norm(test)
    test_program = [
        ("h", [1], None),
        ("cx", [1, 4], None),
        ("t", [0], None),
        ("tdg", [3], None),
        ("s", [1], None),
        ("sdg", [2], None),
        ("rz", [2], 0.337),
        ("x", [4], None),
    ]
    np.testing.assert_allclose(
        native(test, test_program),
        apply_gates(test, test_program, 5),
        rtol=0,
        atol=3e-15,
    )
    initial = np.zeros(amplitude.size, complex)
    initial[0] = 1.0
    prepared = native(initial, program)
    error = float(np.linalg.norm(prepared - amplitude))
    assert error < 2e-11, error
    n = preparation["qubits"]
    nr, nt, nz = config["grid_r_theta_z"]
    regs, position, work, flags = count_polar.settings(
        config["grid_r_theta_z"], config["rrange"], config["zrange"]
    )
    qft = count_polar.centered_forward_qft(regs["k"])
    transformed = native(prepared, qft)
    expected = np.fft.fftshift(
        np.fft.fft(prepared.reshape(nz, nr, nt), axis=2, norm="ortho"), axes=2
    ).ravel()
    fourier_error = phase_aligned_error(transformed, expected)
    restored = native(transformed, inverse(qft))
    roundtrip = float(np.linalg.norm(restored - prepared))
    assert fourier_error < 2e-11 and roundtrip < 2e-11, (fourier_error, roundtrip)
    report = {
        "run": run,
        "grid_r_theta_z": config["grid_r_theta_z"],
        "position_qubits": n,
        "bond_dimension": chi,
        "fidelity": preparation["fidelity"],
        "total_variation": preparation["total_variation"],
        "native_preparation_gates": len(program),
        "gate_counts": dict(actual_counts),
        "full_preparation_statevector_error": error,
        "negative_amplitude_probability": float(np.sum(amplitude[amplitude < 0] ** 2)),
        "native_endpoint_QFT_gates_each": len(qft),
        "native_forward_QFT_vs_FFT_error_up_to_global_phase": fourier_error,
        "native_QFT_roundtrip_error": roundtrip,
        "preparation_count_matches_published": True,
        "scope": "Entire preparation and endpoint Fourier circuits replayed gate by gate; propagation uses matching blocks",
        "full_transport_elementary_gate_replay": False,
    }
    np.savez_compressed(
        out / "mps_blocks.npz",
        base=base,
        **{f"U_{i}": b["basis"] for i, b in enumerate(blocks)},
    )
    with gzip.open(out / "native_preparation_gates.json.gz", "wt") as f:
        json.dump(program, f)
    if propagate:
        model = load()
        ref = shared_data(
            "MPS_KvN_inputs", "MPS", "domain_selection", run, "reference.csv"
        )
        args = SimpleNamespace(
            output=out / "propagation",
            model=model,
            grid=config["grid_r_theta_z"],
            steps=config["steps"],
            rrange=config["rrange"],
            zrange=config["zrange"],
            stencil=config["stencil"],
            quadrature=12,
            restart=None,
            no_save_preparation=True,
            start=config["time_window"][0],
            initial_probability_integral=normalization,
            prepared_amplitude=prepared.reshape(nz, nr, nt),
            reference_csv=ref,
        )
        transport.replay(args)
        terminal = json.loads((args.output / "terminal_readout.json").read_text())
        budget = field_budget(terminal, 1e-4, 0.01)
        old = json.loads((source / "expected_field_budget.json").read_text())
        keys = [
            "noiseless_deterministic_relative_field_bias",
            "noiseless_relative_field_variance_per_shot",
        ]
        changes = {key: float(budget[key] - old[key]) for key in keys}
        assert max(abs(x) for x in changes.values()) < 2e-10, changes
        assert (
            budget["minimum_accepted_shots_under_rms_bound"]
            == old["minimum_accepted_shots_under_rms_bound"]
        )
        retained = (
            figure_data("F5")
            / "wake/data"
            / ("long298" if run == "long298" else "restart40")
            / "trajectory.csv"
        )
        full_change = None
        if retained.is_file():
            # An independently regenerated dataset does not yet contain its
            # published destination. The comparison is optional; numerical
            # budget and physical-propagation checks above remain mandatory.
            saved = np.genfromtxt(retained, delimiter=",", names=True)
            fresh = np.genfromtxt(
                args.output / "trajectory.csv", delimiter=",", names=True
            )
            assert saved.dtype.names == fresh.dtype.names and saved.shape == fresh.shape
            full_change = max(
                float(np.max(abs(saved[key] - fresh[key]))) for key in saved.dtype.names
            )
            assert full_change < 2e-9, full_change
        report.update(
            bias_percent=100 * budget[keys[0]],
            accepted_shots=budget["minimum_accepted_shots_under_rms_bound"],
            difference_from_published_budget=changes,
            maximum_saved_trajectory_column_change=full_change,
            propagated_input="Actual global elementary-gate MPS preparation output, including its phases",
        )
        (args.output / "field_budget.json").write_text(
            json.dumps(budget, indent=2) + "\n"
        )
    report["wall_seconds"] = time.monotonic() - tick
    (out / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run", choices=["late40", "long298", "both"])
    p.add_argument("--output", type=Path)
    p.add_argument("--preparation-only", action="store_true")
    a = p.parse_args()
    for name in ["late40", "long298"] if a.run == "both" else [a.run]:
        verify(name, a.output / name if a.output else None, not a.preparation_only)
