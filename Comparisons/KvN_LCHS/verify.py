"""Replay the published input files through the source tally and our one-clean compilation."""

# Direct-file execution must locate Codes/ independently of the working directory.
if __package__ in (None, ""):
    import sys as _path_sys
    from pathlib import Path as _Path
    _codes = next(p for p in _Path(__file__).resolve().parents
                  if (p / "Source/data_locations.py").is_file())
    _path_sys.path.insert(0, str(_codes))


from pathlib import Path
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Source.data_locations import published_path, source_data_root, generated_outputs_root

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "source_replay"
ARCHIVE = source_data_root() / "Tables/Table_S10/LCHS_source_counts"


def main():
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    manifest = json.loads((ARCHIVE / "upstream_input_manifest.json").read_text())
    for entry in manifest:
        if entry["path"].startswith("simulations/"):
            path = published_path(
                "Comparisons/KvN_LCHS/source_replay/QuCF/" + entry["path"]
            )
            assert (
                hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
            ), path
    subprocess.run(
        [sys.executable, "-B", str(SOURCE / "audit/count_circuit.py")],
        env=env,
        check=True,
    )
    newly_counted = generated_outputs_root() / "Comparisons/KvN_LCHS/source_replay/audit/results"
    for name in ("dt1_stmc_histogram.csv", "t2_stmc_histogram.csv"):
        assert (newly_counted / name).read_bytes() == (
            ARCHIVE / name
        ).read_bytes(), name
    with tempfile.TemporaryDirectory() as tmp:
        check = Path(tmp) / "published_t1_source"
        shutil.copytree(newly_counted, check)
        subprocess.run(
            [
                sys.executable,
                "-B",
                str(HERE / "compile_stmc_one_clean.py"),
                "--source-dir",
                str(check),
            ],
            env=env,
            check=True,
        )
        report = json.loads((check / "one_clean_gate_audit.json").read_text())
    source = json.loads((newly_counted / "source_count_audit.json").read_text())[
        "cases"
    ][0]
    case = report["cases"][0]
    saved = json.loads((ARCHIVE / "one_clean_gate_audit.json").read_text())["cases"][0]
    for name in (
        "logical_qubits",
        "reference_STMC_total",
        "RZ",
        "direct_T",
        "Clifford",
        "CX",
        "by_basis_gate",
        "by_source_gate",
    ):
        assert case[name] == saved[name], name
    workload = json.loads(published_path("Data/RE/workloads.json").read_text())[
        "nj_lchs"
    ]
    assert int(workload["native_T"]) == case["direct_T"]
    assert int(float(workload["RZ"])) == case["RZ"]
    assert int(workload["logical_qubits"]) == case["logical_qubits"]
    assert (
        int(workload["position_qubits"]) == 7
        and int(workload["successful_shots"]) == 815
    )
    assert math.isclose(
        float(workload["success_probability"]),
        source["success_probability_archived"],
        rel_tol=2e-15,
    )
    assert math.isclose(
        float(workload["source_angle_error_bound"]),
        case["canonicalized_source_angle_operator_error_bound"],
        rel_tol=1e-14,
    )
    with published_path("Data/RE/reference_selected.csv").open() as f:
        fastest = next(
            row
            for row in csv.DictReader(f)
            if row["model"] == "nj_lchs"
            and row["selection"] == "fastest_uncapped"
            and float(row["p_phys"]) == 5e-4
        )
    assert int(fastest["N_T"]) == 126
    assert (
        int(fastest["T_count"]) == case["direct_T"] + int(fastest["N_T"]) * case["RZ"]
    )
    total_time = (
        float(fastest["seconds_per_attempt"])
        * int(workload["successful_shots"])
        / float(workload["success_probability"])
    )
    assert math.isclose(
        total_time, float(fastest["complete_task_seconds"]), rel_tol=2e-15
    )
    print(
        "Source histograms, compiled ten-step gate inventory and current resource inputs agree."
    )


if __name__ == "__main__":
    main()
