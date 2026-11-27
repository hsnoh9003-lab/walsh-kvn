"""Candidate Fourier-Walsh replay on fixed inputs, domains, grids and steps."""

# Direct-file execution must locate Codes/ independently of the working directory.
if __package__ in (None, ""):
    import sys as _path_sys
    from pathlib import Path as _Path
    _codes = next(p for p in _Path(__file__).resolve().parents
                  if (p / "Source/data_locations.py").is_file())
    _path_sys.path.insert(0, str(_codes))


from Source.data_locations import generated_outputs_root

import os

for key in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[key] = "1"
from pathlib import Path
from collections import Counter
import argparse, csv, json, time
import numpy as np
from scipy.integrate import solve_ivp
from Source.ROM.model import load
from Source.ROM.ensemble import make_input, reference
from Source.ROM.field_budget import field_budget
from .mps import compress
from . import transport
from Source.data_locations import shared_data

PACKAGE = Path(__file__).resolve().parents[2]
ROOT = shared_data("MPS_KvN_inputs", "KvN")


def count_native(model, config, prep):
    from . import count_polar as C

    regs, position, work, flags = C.settings(
        config["grid_r_theta_z"], config["rrange"], config["zrange"]
    )
    layers = C.schedule(config["stencil"], "rkz")
    symmetric = (
        [(x, 0.5) for x in layers[:-1]]
        + [(layers[-1], 1.0)]
        + [(x, 0.5) for x in layers[-2::-1]]
    )
    dt = (config["time_window"][1] - config["time_window"][0]) / config["steps"]
    program = [
        gate
        for layer, scale in symmetric
        for gate in C.compile_layer(
            layer,
            regs,
            model.coefficients,
            scale * dt,
            config["stencil"],
            "face",
            work,
            flags,
        )
    ]
    counts = Counter({g: n * config["steps"] for g, n in C.getcounts(program).items()})
    qft = C.centered_forward_qft(regs["k"])
    counts.update(C.getcounts(qft + C.core.inverse(qft)))
    counts.update(prep["by_gate"])
    logical = max(
        max(q for _, wires, _ in program for q in wires) + 1,
        position + prep["work_qubits"],
    )
    return dict(
        logical_qubits=logical,
        position_qubits=position,
        arbitrary_Rz=counts["rz"],
        native_T=counts["t"] + counts["tdg"],
        Clifford=sum(v for k, v in counts.items() if k not in ["rz", "t", "tdg"]),
        by_gate=dict(counts),
        steps=config["steps"],
        dt=dt,
        preparation=prep,
        arithmetic_clean_work_qubits=len(work),
        boundary_flag_qubits=len(flags),
        note="One-clean exact MCX decomposition shared with the radial circuit; two boundary flags retained. Same matching schedule, including identity-layer address operations. No extra cancellation or zero-layer elimination introduced.",
    )


def main(run, policy, model_name, data_root=None, output_root=None, model_dir=None):
    input_root = Path(data_root) if data_root is not None else ROOT / "inputs"
    destination = (
        Path(output_root)
        if output_root is not None
        else generated_outputs_root() / "KvN/ensemble"
    )
    model = load(model_name, data_dir=model_dir)
    config = json.loads((input_root / run / "protocol.json").read_text())
    out = destination / f"{model_name}_{run}_{policy}"
    out.mkdir(parents=True, exist_ok=True)
    if policy == "recentered":
        assert run == "late40"
        growth = json.loads((input_root / "long298/protocol.json").read_text())
        sol = solve_ivp(
            lambda t, a: model.drift(a),
            (29.8, 40.0),
            growth["centre_cartesian"],
            method="DOP853",
            rtol=2e-12,
            atol=2e-14,
        )
        assert sol.success
        config["centre_cartesian"] = sol.y[:, -1].tolist()
        target, norm = make_input(config)
        for chi in [2, 4, 8, 16, 32, 64]:
            amplitude, blocks, base, prep = compress(target.ravel(), chi)
            if prep["fidelity"] >= 0.9999:
                break
        assert prep["fidelity"] >= 0.9999
        prep.update(block_widths=[b["m"] for b in blocks], passed=True)
        (out / "selected.json").write_text(json.dumps(prep, indent=2) + "\n")
        np.savez_compressed(
            out / "mps_blocks.npz",
            base=base,
            **{f"U_{i}": b["basis"] for i, b in enumerate(blocks)},
        )
        from . import compile_mps as compiler

        compiler.main(run, directory=out)
        inventory = json.loads((out / "compiled_inventory.json").read_text())
        np.save(out / "prepared_amplitude.npy", amplitude)
    else:
        prepared_file = input_root / run / "prepared_amplitude.npy"
        if prepared_file.is_file():
            amplitude = np.load(prepared_file)
        else:
            # The public MPS replay preserves the complete Schmidt blocks.
            # Reconstruct the same prepared state without requiring an
            # unpublished intermediate prepared_amplitude.npy file.
            blocks_path = shared_data(
                "MPS_KvN_inputs", "Three_Coordinate_MPS_Replay", run, "mps_blocks.npz"
            )
            with np.load(blocks_path) as blocks:
                amplitude = blocks["base"].copy()
                ordered = sorted(
                    (key for key in blocks.files if key.startswith("U_")),
                    key=lambda key: int(key.removeprefix("U_")),
                    reverse=True,
                )
                for key in ordered:
                    basis = blocks[key]
                    amplitude = (basis @ amplitude.reshape(basis.shape[1], -1)).ravel()
            assert np.isclose(np.linalg.norm(amplitude), 1.0, rtol=0.0, atol=2e-10)
        original = json.loads((input_root / run / "summary.json").read_text())
        norm = original["initial_probability_integral"]
        inventory = json.loads(
            (input_root / run / "compiled_inventory.json").read_text()
        )
    (out / "protocol.json").write_text(json.dumps(config, indent=2) + "\n")
    reference_path = out / "reference.csv"
    audit_path = out / "reference_audit.json"
    if not audit_path.exists():
        full, audit = reference(model, config, (64, 128, 64), reference_path)
        lower, lower_audit = reference(model, config, (48, 96, 48))
        change = float(np.max(model.field_error(full, lower)))
        assert change < 1e-6, change
        audit.update(
            lower_orders=lower_audit["orders"], max_relative_field_change=change
        )
        audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    nr, nt, nz = config["grid_r_theta_z"]
    args = argparse.Namespace(
        output=out,
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
        initial_probability_integral=norm,
        prepared_amplitude=amplitude.reshape(nz, nr, nt),
        reference_csv=reference_path,
    )
    transport.replay(args)
    raw = json.loads((out / "terminal_readout.json").read_text())
    budget = field_budget(raw, 1e-4, 0.01)
    (out / "field_budget.json").write_text(json.dumps(budget, indent=2) + "\n")
    counted = count_native(model, config, inventory)
    (out / "inventory.json").write_text(json.dumps(counted, indent=2) + "\n")
    result = dict(
        model=model_name,
        run=run,
        input_policy=policy,
        field_bias_percent=100 * budget["noiseless_deterministic_relative_field_bias"],
        per_shot_field_variance=budget["noiseless_relative_field_variance_per_shot"],
        protected_shots=budget["minimum_accepted_shots_under_rms_bound"],
        logical_qubits=counted["logical_qubits"],
        arbitrary_Rz=counted["arbitrary_Rz"],
        native_T=counted["native_T"],
        initial_mean=config["centre_cartesian"],
        reference_audit=json.loads(audit_path.read_text()),
    )
    synthesis = int(np.ceil(3 * np.log2(counted["arbitrary_Rz"] / (0.5e-4))))
    result["common_precision_T_per_rotation"] = synthesis
    result["T_workload_proxy"] = (
        None
        if result["protected_shots"] is None
        else int(
            result["protected_shots"]
            * (counted["native_T"] + synthesis * counted["arbitrary_Rz"])
        )
    )
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print("RESULT", json.dumps(result), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run", choices=["long298", "late40"])
    ap.add_argument("--policy", choices=["common", "recentered"], default="common")
    ap.add_argument("--model", choices=["deployed", "candidate"], default="candidate")
    a = ap.parse_args()
    main(a.run, a.policy, a.model)
