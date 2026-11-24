"""Trace BE and KvN–LCHS circuits into the v100 EFTQC search inputs.

The BE inventory is recomputed from the full-window circuit/polynomial.
The LCHS circuit histogram can be reconstructed from the bundled, pinned QuCF
oracles, phase HDF5 data and archived output HDF5 by running
Comparisons/KvN_LCHS/verify.py. That command also recompiles the lowering
ledger. This script compares the saved LCHS tally with the resource inputs. Measured Gidney calibration CSVs
remain unmodified and are loaded by Source.RE for an actual search.

Run from the package root:
  python -m Reproduction.build_comparison_workloads
  python -m Source.RE.run --workloads Build/Comparisons/workloads_from_sources.json \
      --prep-dir ../Source_Data/Shared_source_data/RE_common/RE --output Build/RE_comparisons --models generator nj_lchs
"""
from __future__ import annotations

# Public Source Data mapping (single canonical file per published input).
import sys as _kvn_path_sys
from pathlib import Path as _kvn_path_Path
_kvn_path_sys.path.insert(0, str(_kvn_path_Path(__file__).resolve().parents[1]))
from Source.data_locations import published_path

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BE = ROOT / "Comparisons/BE_KvN"
LCHS = ROOT / "Comparisons/KvN_LCHS"
sys.path.insert(0, str(BE / "fullwindow"))
from fullwindow import resource_record  # noqa: E402
from compile_literal_joint import build as build_be_joint, compiler as compile_be_gates  # noqa: E402


def compiled_be_joint_ledger():
    """Count the actual BE macros with the shared LCHS one-clean compiler.

    Recompile here instead of trusting an old literal_joint_counts.json when
    constructing new RE inputs. The cached count and this fresh result must
    agree, so a source change cannot silently leave the manuscript ledger stale.
    """
    ordinary, controlled, width, probes, radial, diagonal, alpha = build_be_joint()
    plain_counts, plain_width = compile_be_gates(ordinary, width)
    ctrl_counts, ctrl_width = compile_be_gates(controlled, width + 1)
    fresh = {
        "normalization_alpha": alpha, "radial_normalization": radial,
        "diagonal_normalization": diagonal,
        "system_qubits": 6 + len(probes),
        "projector_qubits": width - 6 - len(probes),
        "macro_width": width,
        "ordinary": {**plain_counts, "logical_qubits": plain_width},
        "controlled": {**ctrl_counts, "logical_qubits": ctrl_width},
    }
    cached = json.loads(published_path('Comparisons/BE_KvN/fullwindow/literal_joint_counts.json').read_text())
    for key, value in fresh.items():
        assert cached[key] == value, ("Stale BE literal_joint_counts.json; "
                                      "rerun fullwindow/compile_literal_joint.py", key)
    return fresh


def workloads():
    old = json.loads((published_path('Data/RE/workloads_v38.json')).read_text())
    generator = resource_record(compiled_be_joint_ledger())
    compiled = json.loads(published_path('Comparisons/KvN_LCHS/published_t1_source/one_clean_gate_audit.json').read_text())
    source = json.loads(published_path('Comparisons/KvN_LCHS/published_t1_source/source_count_audit.json').read_text())
    lchs = next(c for c in compiled["cases"] if c["simulation_time"] == 1)
    source_lchs = next(c for c in source["cases"] if c["physical_simulation_time"] == 1)
    assert lchs["reference_STMC_total"] == source_lchs["STMC_total"]
    assert lchs["logical_qubits"] == 26
    old["generator"] = {
        "logical_qubits": generator["logical_qubits"], "position_qubits": 6,
        "RZ": generator["RZ_per_attempt"], "native_T": generator["native_T_per_attempt"],
        "Clifford": generator["Clifford_per_attempt"],
        "successful_shots": generator["accepted_shots_at_1pct_with_protected_allowance"],
        "success_probability": generator["nominal_success_probability"],
        "source_angle_error_bound": generator["source_angle_error_bound"],
    }
    old["nj_lchs"] = {
        "logical_qubits": lchs["logical_qubits"], "position_qubits": 7,
        "RZ": lchs["RZ"], "native_T": lchs["direct_T"],
        "Clifford": lchs["Clifford"], "successful_shots": 815,
        "success_probability": source_lchs["success_probability_archived"],
        "source_angle_error_bound": lchs["canonicalized_source_angle_operator_error_bound"],
    }
    return old


def main():
    current = json.loads((published_path('Data/RE/workloads_v38.json')).read_text())
    recomputed = workloads()
    checks = []
    for name in ("generator", "nj_lchs"):
        keys = ("logical_qubits", "position_qubits", "RZ", "native_T", "successful_shots",
                "success_probability", "source_angle_error_bound")
        if name == "generator":
            keys += ("Clifford",)
        for key in keys:
            got, wanted = recomputed[name][key], current[name].get(key, 0)
            if wanted in ("", None):
                wanted = 0
            assert math.isclose(float(got), float(wanted), rel_tol=1e-11, abs_tol=1e-10), (
                name, key, got, wanted)
            checks.append(f"{name}.{key}")
    out = ROOT / "Build/Comparisons"
    out.mkdir(parents=True, exist_ok=True)
    (out / "workloads_from_sources.json").write_text(json.dumps(recomputed, indent=2) + "\n")
    (out / "verification.json").write_text(json.dumps({"status": "PASS", "checks": checks,
        "resource_engine": "Source.RE", "Gidney_input": "supplied measurement CSVs",
        "LCHS_source_boundary": "Pinned QuCF oracle/HDF5 source bundled; block-matrix and "
            "algebraic histogram recomputed, original 25-qubit evolution not rerun"}, indent=2) + "\n")
    print(f"PASS: checked {len(checks)} source-derived workload fields; wrote {out}")


if __name__ == "__main__":
    main()
