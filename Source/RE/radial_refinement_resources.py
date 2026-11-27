"""Recompute physical resources for the archived 128-cell radial workload.

Run: python -m Source.RE.radial_refinement_resources
Read the deposited Radial128/grid128/flow_report.json and the cultivation table
at physical error probability 5e-4. Source.RE searches synthesis precision,
surface-code distances and magic-state supply. Results are written under
Generated_Outputs/Radial_Refinement/resources/.

This entry reads the archived gate counts and shot budget, not an arbitrary
local replay output. The final assertions compare the resulting selected
designs with the deposited numerical reference.
"""

# Direct-file execution must locate Codes/ independently of the working directory.
if __package__ in (None, ""):
    import sys as _path_sys
    from pathlib import Path as _Path
    _codes = next(p for p in _Path(__file__).resolve().parents
                  if (p / "Source/data_locations.py").is_file())
    _path_sys.path.insert(0, str(_codes))



from pathlib import Path
import sys, json, math

PACKAGE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PACKAGE))
from Source.data_locations import generated_outputs_root
from Source.data_locations import published_path, shared_data

ROOT = shared_data("Radial_refinement", "Radial128")
from Source.RE.run import sweep
from Source.RE.model import ModelConfig

if __name__ == "__main__":
    report = json.loads((ROOT / "grid128/flow_report.json").read_text())
    count = report["counts"]
    # Seven radial bits plus two phase bits and one clean work bit give ten
    # logical qubits. RZ counts arbitrary rotations; direct_T counts native T
    # gates before rotation synthesis. shots counts accepted circuit measurements.
    work = dict(
        logical_qubits=10,
        is_wake=True,
        n_position=7,
        RZ=count["RZ"],
        direct_T=count["native_T"],
        Clifford=count["Clifford"],
        shots=report["budget"]["shots"],
        success=1.0,
        canonicalization_error_bound=0.0,
        precision_min=45,
        precision_max=200,
        shot_basis="1% expected relative field RMS",
        workload_scope="same wake ensemble, refined radial grid",
    )
    assert ModelConfig().W_seconds == 1e-6
    expected = json.loads((ROOT / "comparison_summary.json").read_text())
    rows = sweep(
        {"radial128": work},
        {5: published_path("Data/RE/prep_p5.csv")},
        generated_outputs_root() / "Radial_Refinement/resources",
        rates=[5],
    )
    fastest = next(r for r in rows if r["selection"] == "fastest_uncapped")
    minimum = next(r for r in rows if r["selection"] == "minimum_qubits")
    assert fastest["physical_qubits"] == 561553 and fastest["T_count"] == 1193151
    assert math.isclose(fastest["complete_task_seconds"], 18339.924021, rel_tol=1e-12)
    assert minimum["physical_qubits"] == 60482
    print(
        json.dumps(
            {
                "fastest": {
                    k: fastest[k]
                    for k in ["physical_qubits", "T_count", "complete_task_seconds"]
                },
                "minimum_qubits": minimum["physical_qubits"],
            },
            indent=2,
        )
    )
