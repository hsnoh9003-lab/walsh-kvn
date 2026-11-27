"""Build publication-facing error and gate-count tables from retained results.

The source arrays contain signed moment-error components, compiled gate counts
and model protocols. Printed labels and table formatting are applied only after
those records are selected. Input-projection, preparation and transport errors
are distinct; their vector sum is used before an RMS norm is taken. The script
writes CSVs to --output and does not perform a new resource optimization.
"""

# Direct-file execution must locate Codes/ independently of the working directory.
if __package__ in (None, ""):
    import sys as _path_sys
    from pathlib import Path as _Path
    _codes = next(p for p in _Path(__file__).resolve().parents
                  if (p / "Source/data_locations.py").is_file())
    _path_sys.path.insert(0, str(_codes))


import argparse
import csv
import json
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
try:
    from Source.Data_Production.data_assets import fixed_asset
except ModuleNotFoundError:
    from data_assets import fixed_asset

HERE = Path(__file__).resolve().parent
INPUT = fixed_asset("inputs", "publication")
SWEEP = fixed_asset("inputs", "full_sweep")


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def save(out, name, header, records):
    with (out / name).open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(records)


def seven_models(out):
    """Combine time, space and input-error records for the declared model cases."""
    time = [r for r in read(SWEEP / "time_refinement.csv") if int(r["steps"]) == 64]
    by_case = lambda rows: {(r["case"], r["cells_per_coordinate"]): r for r in rows}
    spatial = by_case(read(SWEEP / "spatial_refinement.csv"))
    projection = by_case(read(INPUT / "prediction_error_decomposition.csv"))
    budgets = []
    errors = []
    for r in time:
        key = r["case"], r["cells_per_coordinate"]
        s, analytic = spatial[key], projection[key]
        components = (
            ["x", "x2"] if int(r["dimension"]) == 1 else ["x", "y", "x2", "y2", "xy"]
        )
        # Ideal-cell projection is independent of the state-preparation circuit.
        grid_bias = np.array(
            [float(analytic["ideal_cell_projection_" + k]) for k in components]
        )
        preparation = np.array(
            [
                float(s["total_analytic_error_" + k]) - float(s["transport_error_" + k])
                for k in components
            ]
        )
        # Subtract signed component errors before taking their RMS;
        # subtracting the scalar RMS values would lose cancellation.
        fit = preparation - grid_bias
        rms = lambda v: np.sqrt(np.mean(v * v))
        terms = (
            r["moment_rms_splitting_error"],
            s["moment_rms_transport_error"],
            rms(fit),
            rms(grid_bias),
            s["moment_rms_total_analytic_error"],
            r["continuum_total_analytic_moment_rms_error"],
        )
        errors.append([r["label"], *(format(1000 * float(v), ".6g") for v in terms)])
        # The one-clean synchronization changes the workspace label of the scalar circuits.
        budgets.append(
            [
                r["label"],
                r["cells_per_coordinate"],
                f'{r["data_qubits"]} / 1',
                r["steps"],
                *(
                    f"{int(r[k]):,}"
                    for k in ("complete_RZ", "complete_CX", "complete_native_T")
                ),
            ]
        )
    save(
        out,
        "Supplementary_Data_4.csv",
        [
            "model",
            "temporal_RMS_x1e3",
            "transport_RMS_x1e3",
            "MPS_fit_RMS_x1e3",
            "cell_projection_RMS_x1e3",
            "finite_total_RMS_x1e3",
            "compiled_total_RMS_x1e3",
        ],
        errors,
    )
    save(
        out,
        "Supplementary_Data_5.csv",
        [
            "model",
            "cells_per_coordinate",
            "data_work_qubits",
            "steps",
            "RZ",
            "CX",
            "pre_synthesis_T",
        ],
        budgets,
    )


def table_diagnostics(out):
    """Read printed diagnostic labels and update coupled-case errors from retained arrays."""
    root = ET.parse(INPUT / "diagnostics_table.xml").getroot()
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    rows = [
        [
            "".join(t.text or "" for t in cell.findall(".//w:t", ns))
            for cell in row.findall("w:tc", ns)
        ]
        for row in root.findall("w:tr", ns)
    ]
    labels = {
        "Interior": "interior",
        "Near boundary": "boundary_clipped",
        "Near stable": "near_stable",
        "Near unstable": "near_unstable",
    }
    for row in rows[1:]:
        if row[0] == "2D C3":
            name = "coupled_cubic_2d__" + labels[row[1]] + ".npz"
            path = fixed_asset("inputs", "retained_experiments", "trajectories", name)
            with np.load(path) as run:
                mean = run["moment_history"][-1]
                matched = run["prepared_cell_reference_moments"][-1]
                analytic = run["analytic_target_reference_moments"][-1]
                row[4] = format(float(run["preparation_fidelity"]), ".6g")
                row[5] = format(float(np.sqrt(np.mean((mean - matched) ** 2))), ".3g")
                row[6] = format(float(np.sqrt(np.mean((mean - analytic) ** 2))), ".3g")
    save(
        out,
        "Table_S4_input_diagnostics.csv",
        [
            "Model",
            "Input",
            "Gaussian centre before truncation",
            "Gaussian width before truncation",
            "Preparation fidelity",
            "Matched absolute moment RMS error",
            "Analytic-target absolute moment RMS error",
        ],
        rows[1:],
    )


def wake_inventory(data, out):
    """Assemble per-circuit operation counts from compiled inventories and tabulated schedules."""
    by_circuit = {r["Circuit"]: r for r in read(INPUT / "Table_S6_circuit_counts.csv")}
    inv = json.loads((INPUT / "circuit_inventories.json").read_text())
    with (data / "Tables/Table_S2/Table_S2_as_printed.csv").open(newline="") as file:
        operations = list(csv.reader(file))[2:-1]
    rows = []
    for scheme, parity, label in (
        ("A", "rebuilt", "Walsh-KvN A reference"),
        ("A", "reused", "Walsh-KvN A parity reuse"),
        ("B", "reused", "Walsh-KvN B parity reuse"),
    ):
        r = by_circuit[label]
        if parity == "rebuilt":
            cx = sum(int(row[1]) * int(row[4]) for row in operations)
        else:
            cx = inv[scheme]["by_gate"]["cx"]
            assert (
                int(r["circuit_T"])
                == inv[scheme]["by_gate"]["t"] + inv[scheme]["by_gate"]["tdg"]
            )
        rows.append(
            [
                scheme,
                parity,
                r["logical_qubits"],
                r["steps"],
                r["RZ"],
                r["circuit_T"],
                cx,
                r["base_Clifford"],
            ]
        )
    save(
        out,
        "wake_gate_inventory.csv",
        [
            "address_shift",
            "walsh_parity",
            "logical_qubits",
            "radial_steps",
            "arbitrary_RZ",
            "native_T_including_T_dagger",
            "CNOT",
            "Clifford_including_CNOT",
        ],
        rows,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    seven_models(args.output)
    table_diagnostics(args.output)
    wake_inventory(args.source_data, args.output)
