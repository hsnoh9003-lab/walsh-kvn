"""Export force observables, drift curves and a small Walsh-parity example.

forces evaluates Pauli expectations from saved complex circuit amplitudes and
then applies the fitted drag/lift polynomials. Those are reduced-model force
predictions, not the CFD surface-stress integration that produced the FEM
reference. models evaluates the fixed drift coefficients; walsh_panel tabulates
the binary parity signs. Outputs are CSVs under the requested --output directory.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def write(output, name, header, rows):
    with (output / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def forces(data, out):
    """Evaluate the Pauli expectations analytically from the saved complex amplitudes."""
    source = data / "Shared_source_data/CFD_ROM/CFD_ROM/Re100/data"
    with np.load(source / "direct_quantum_history.npz") as history:
        times, radii, states = (
            history[key] for key in ("times", "r", "direct_history")
        )
    with np.load(source / "unified_reference.npz") as reference:
        phases = reference["known_phase"]
    with (
        data / "Figures/Fig_4/plot_inputs/plot_values/Re100/force_coefficients.csv"
    ).open() as f:
        fit = {
            (row["observable"], row["term"]): float(row["coefficient"])
            for row in csv.DictReader(f)
        }
    drag = {
        term: value for (observable, term), value in fit.items() if observable == "C_D"
    }
    lift = {
        term: value for (observable, term), value in fit.items() if observable == "C_L"
    }
    result = []
    for step, (t, theta, state) in enumerate(zip(times, phases, states, strict=True)):
        # Four phase-register sectors share the radial grid: state has
        # shape (4, radial_cells); summing sectors gives each cell mass.
        a, b, c, d = state
        probability = (abs(state) ** 2).sum(axis=0)
        np.testing.assert_allclose(probability.sum(), 1, atol=1e-11, rtol=0)
        # X on phase qubit two and Y on phase qubit one supply the first harmonic.
        first = (
            2
            * (
                (a.conj() * b + c.conj() * d).real
                + 1j * (a.conj() * c + b.conj() * d).imag
            )
            * np.exp(1j * theta)
        )
        # The complex second harmonic combines XX-YY with i*(XY+YX).
        # Multiplying by exp(2i*theta) restores the constant-rate phase. With
        # z=r**2, x**2-y**2=z*cos(2*phase) and 2*x*y=z*sin(2*phase), explaining
        # the factors of one half in the quadratic drag terms below.
        second = 4 * a.conj() * d * np.exp(2j * theta)
        z = radii**2
        cd = np.sum(
            probability * (drag["1"] + drag["z"] * z + drag["z^2"] * z * z)
            + z
            / 2
            * (
                (drag["x^2"] + drag["y^2"]) * probability
                + (drag["x^2"] - drag["y^2"]) * second.real
                + drag["xy"] * second.imag
            )
        )
        cl = np.sum(
            radii
            * (
                (lift["x"] + lift["xz"] * z) * first.real
                + (lift["y"] + lift["yz"] * z) * first.imag
            )
        )
        result.append((step, t, cd, cl))
    write(out, "fig4f_walsh_forces.csv", ["step", "time", "C_D", "C_L"], result)


def models(out):
    """Tabulate scalar drifts, equilibria and coupled-vector arrows from the fixed protocol."""
    protocol = json.loads(Path(__file__).with_name("model_protocol.json").read_text())
    x = np.linspace(0, 1, 1001)
    curves, markers = [], []
    for label, record in zip(("Q2", "C3", "Q5"), protocol[:3], strict=True):
        coeff = np.array(record["raw_coefficients"]) * record["normalization_scale"]
        curves.extend(
            (label, t, v) for t, v in zip(x, np.polynomial.polynomial.polyval(x, coeff))
        )
        for root in sorted(np.polynomial.polynomial.polyroots(coeff).real):
            location = round(float(root), 12)
            slope = np.polynomial.polynomial.polyval(
                location, np.polynomial.polynomial.polyder(coeff)
            )
            markers.append(
                (label, format(location, "g"), 0, "stable" if slope < 0 else "unstable")
            )
    markers.extend([("2D C3", 0, 1, "unstable"), ("2D C3", 1, 0, "stable")])
    nx, ny = protocol[3]["normalization_scale"]
    arrows = [
        (u, v, nx * u * (1 - u) * (1 + 2 * v), ny * v * (1 - v) * (-2 + u))
        for v in np.linspace(0.2, 0.8, 3)
        for u in np.linspace(0.2, 0.8, 3)
    ]
    write(out, "scalar_drift_curves.csv", ["model", "x", "dx_dt"], curves)
    write(out, "coupled_drift_arrows.csv", ["x", "y", "dx_dt", "dy_dt"], arrows)
    write(out, "marked_equilibria.csv", ["model", "x", "y", "marker"], markers)


def walsh_panel(out):
    """Enumerate three-bit pair indices and Pauli-Z parity signs for the diagram."""
    rows = [
        (
            q,
            format(q, "03b"),
            2 * q + 1,
            1,
            *(1 - 2 * ((q >> bit) & 1) for bit in range(3)),
        )
        for q in range(8)
    ]
    write(
        out,
        "Figure1_panel_c.csv",
        [
            "pair_index",
            "pair_bits",
            "linear_angle",
            "constant_sign",
            "b0_sign",
            "b1_sign",
            "b2_sign",
        ],
        rows,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    forces(args.source_data, args.output)
    models(args.output)
    walsh_panel(args.output)
