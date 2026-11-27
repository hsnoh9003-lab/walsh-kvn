"""Draw the scalar density comparison on the 64-cell grid after 64 steps."""

import sys as _figure_sys
from pathlib import Path as _FigurePath

_figure_sys.path.insert(0, str(_FigurePath(__file__).resolve().parents[3]))
from Figure_Plots.common.paths import output_dir
from pathlib import Path
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from Source.data_locations import figure_data, published_path

ROOT = Path(__file__).resolve().parents[1]
MODELS = [("quadratic", "Q2"), ("cubic_interior", "C3"), ("quintic_multistable", "Q5")]
INPUTS = [
    ("interior", "Interior"),
    ("boundary_clipped", "Near boundary"),
    ("near_stable", "Near stable"),
    ("near_unstable", "Near unstable"),
]


def draw(output):
    rows = pd.read_csv(
        figure_data("S04") / "plot_values/scalar_conservative_cell_masses.csv"
    )
    inputs = pd.read_csv(
        published_path("Data/Details/Table_S4_input_diagnostics.csv")
    ).set_index(["Model", "Input"])
    style = {
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "font.size": 7,
        "axes.linewidth": 0.6,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "svg.fonttype": "path",
        "svg.hashsalt": "kvn-s5-densities",
        "pdf.fonttype": 42,
        "figure.dpi": 150,
        "savefig.dpi": 320,
    }
    with plt.rc_context(style):
        fig, axs = plt.subplots(3, 4, figsize=(180 / 25.4, 125 / 25.4), sharex=True)
        fig.subplots_adjust(
            left=0.100, right=0.985, top=0.91, bottom=0.095, wspace=0.26, hspace=0.34
        )
        for r, (model, title) in enumerate(MODELS):
            for c, (case, label) in enumerate(INPUTS):
                ax = axs[r, c]
                profile = rows[rows.id.eq(model + "__" + case)].sort_values("cell")
                widths = profile.right.to_numpy() - profile.left.to_numpy()
                assert len(profile) == 64 and np.allclose(widths, 1 / 64)
                centres = (profile.left.to_numpy() + profile.right.to_numpy()) / 2
                fields = [
                    "initial_prepared_mass",
                    "final_circuit_mass",
                    "final_matched_characteristic_mass",
                ]
                mass = profile[fields].to_numpy()
                assert (
                    np.allclose(mass.sum(axis=0), 1, atol=1e-9)
                    and np.min(mass) >= -1e-12
                )
                initial, circuit, reference = (mass / widths[:, None]).T
                ax.stairs(
                    initial,
                    np.r_[profile.left.to_numpy(), profile.right.iloc[-1]],
                    fill=True,
                    color="#d7dee1",
                    lw=0.35,
                    label="Prepared input",
                )
                ax.plot(centres, circuit, color="#23688e", lw=1.05, label="Circuit")
                ax.plot(
                    centres,
                    reference,
                    color="#272727",
                    ls=(0, (2.15, 1.4)),
                    lw=0.67,
                    label="Reference",
                )
                diag = inputs.loc[(title, label)]
                centre = float(diag["Gaussian centre before truncation"])
                sigma = float(diag["Gaussian width before truncation"])
                ax.text(
                    0.055 if c == 2 else 0.960,
                    0.96,
                    rf"$x_0={centre:g},\;\sigma={sigma:g}$",
                    fontsize=6.2,
                    ha="left" if c == 2 else "right",
                    va="top",
                    transform=ax.transAxes,
                )
                ax.set_xlim(0, 1)
                ax.set_ylim(
                    0, 1.30 * max(initial.max(), circuit.max(), reference.max())
                )
                ax.yaxis.set_major_locator(
                    MaxNLocator(3, steps=[1, 2, 2.5, 5, 10], integer=True)
                )
                ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
                ax.tick_params(length=2.5, pad=1.8, labelsize=6.0)
                ax.spines[["top", "right"]].set_visible(False)
                if r == 0:
                    ax.set_title(label, fontsize=7.7, pad=4)
                if c == 0:
                    ax.set_ylabel("Density", fontsize=7.0, labelpad=2.1)
            pos = axs[r, 0].get_position()
            fig.text(
                0.013,
                (pos.y0 + pos.y1) / 2,
                title,
                ha="center",
                va="center",
                rotation=90,
                weight="bold",
                fontsize=8.1,
            )
        axs[2, 1].legend(
            loc="upper right",
            bbox_to_anchor=(0.98, 0.79),
            borderaxespad=0,
            handlelength=1.2,
            handletextpad=0.38,
            labelspacing=0.23,
            frameon=False,
            fontsize=6.2,
        )
        fig.text(0.547, 0.025, r"Position, $x$", ha="center", fontsize=7.3)
        output.mkdir(parents=True, exist_ok=True)
        for ext, meta in [
            ("svg", {"Date": None}),
            ("pdf", {"CreationDate": None}),
            ("png", None),
        ]:
            fig.savefig(output / f"Figure_S4.{ext}", metadata=meta, facecolor="white")
        plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("output", type=Path, nargs="?", default=output_dir("S04"))
    draw(ap.parse_args().output)
