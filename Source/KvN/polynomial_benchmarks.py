"""Count and replay polynomial circuits using one reusable clean auxiliary qubit.

Inputs
------
Retained prepared states and finite-grid polynomial benchmark records.

Outputs
-------
Generated_Outputs/One_Clean_Three_Coordinate/polynomial_benchmarks/.

Run from the repository root
----------------------------
python -m Source.KvN.polynomial_benchmarks

Scope
-----
The coupled replay checks all 64 transport steps against the archived amplitudes.
"""

# Direct-file execution must locate Codes/ independently of the working directory.
if __package__ in (None, ""):
    import sys as _path_sys
    from pathlib import Path as _Path
    _codes = next(p for p in _Path(__file__).resolve().parents
                  if (p / "Source/data_locations.py").is_file())
    _path_sys.path.insert(0, str(_codes))


# Public Source Data mapping (single canonical file per published input).
import sys as _kvn_path_sys
from pathlib import Path as _kvn_path_Path

_kvn_path_sys.path.insert(0, str(_kvn_path_Path(__file__).resolve().parents[1]))
from Source.data_locations import generated_outputs_root
from Source.data_locations import published_path
from pathlib import Path
from collections import Counter
import importlib.util, json, os, sys
import numpy as np
from Source.KvN import wake_gate_core as C

ROOT = Path(__file__).resolve().parents[1]
OUT = generated_outputs_root() / "One_Clean_Three_Coordinate/polynomial_benchmarks"


def main():
    """Compile the coupled 32-by-32 case and verify 64 steps with one clean work qubit."""
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ["POLYNOMIAL_VALIDATION_OUTPUT"] = str(OUT / "compiler_intermediates")
    from Source.KvN import polynomial_validation as module
    # All numerical polynomials and Walsh tables come from the supplied source;
    # only the common exact MCX implementation is replaced.
    module.increment = C.increment
    module.inverse = C.inverse
    case = [c for c in module.declared_cases() if c["D"] == 2][0]
    gates, nq, support = module.compile_step(case, 32, 1 / 64)
    assert nq == 11
    inv = Counter(g[0] for g in gates)
    assert inv["t"] + inv["tdg"] == 378
    assert inv["rz"] == 366, (dict(inv), len(gates))
    arrays = np.load(
        published_path(
            "Data/Figure5_Bias_Analysis/input/grid32/coupled_cubic_2d__interior.npz"
        )
    )
    initial = arrays["state_history"][0]
    expected = arrays["state_history"][-1]
    state = np.zeros(1 << nq, complex)
    state[:1024] = initial
    for _ in range(64):
        state = C.apply_gates(state, gates, nq)
    e = float(np.linalg.norm(state[:1024] - expected))
    leak = float(np.linalg.norm(state[1024:]))
    assert e < 2e-11 and leak < 2e-12, (e, leak)
    row = {
        "logical_qubits": 11,
        "position_qubits": 10,
        "arithmetic_clean_work_qubits": 1,
        "steps": 64,
        "RZ": 23514,
        "native_T": 24192,
        "CX": 54058,
        "native_transport_per_step_by_gate": dict(inv),
        "additional_X_gates_per_attempt": 768,
        "preparation_RZ_unchanged": 23514 - 64 * inv["rz"],
        "preparation_CX_unchanged": 54058 - 64 * inv["cx"],
        "complete_transport_complex_state_l2_error": e,
        "work_leakage": leak,
        "note": "Preparation gates are unchanged. All native transport gates replayed for 64 steps from the retained prepared amplitude.",
    }
    (OUT / "coupled32_inventory_and_replay.json").write_text(
        json.dumps(row, indent=2) + "\n"
    )
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
