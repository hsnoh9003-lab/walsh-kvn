"""Render current Supplementary Figure S8 directly from published Source Data."""

import sys as _figure_sys
from pathlib import Path as _FigurePath

_figure_sys.path.insert(0, str(_FigurePath(__file__).resolve().parents[3]))
from Figure_Plots.common.paths import output_dir
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from pathlib import Path
from Source.data_locations import figure_data

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

BLUE, GREEN, ORANGE = "#285f86", "#528b78", "#b45239"
PURPLE, TEAL = "#7866a6", "#237c83"
BLOCKS = ["compact", "intermediate", "fast"]
CAPS = [100_000, 1_000_000]
PLABEL = r"Physical error probability ($10^{-4}$)"


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=3, width=0.6, pad=2.6)
    ax.tick_params(which="minor", length=1.5, width=0.45)


def mark(fig, letter, ax):
    bbox = ax.get_position()
    fig.text(
        0.014,
        bbox.y1 + 0.006,
        letter,
        ha="left",
        va="bottom",
        fontsize=7.5,
        weight="bold",
    )


def sensitivity(folder):
    data = pd.read_csv(folder / "implementation_budget_sensitivity.csv")
    back = pd.read_csv(folder / "new_backend.csv")
    fig = plt.figure(figsize=(88 / 25.4, 82 / 25.4))
    footprint = fig.add_axes([0.159, 0.668, 0.285, 0.251])
    timing = fig.add_axes([0.684, 0.668, 0.285, 0.251])
    factories = fig.add_axes([0.176, 0.160, 0.764, 0.337])
    eta = np.sort(data.implementation_allowance.unique())
    assert np.allclose(eta, [1e-6, 1e-4, 1e-3, 1e-2])
    x = np.arange(len(eta))
    for i, (key, col, label) in enumerate(
        [
            ("minimum_footprint", GREEN, "Minimum-qubit"),
            ("fastest_under_cap", BLUE, "Fastest"),
        ]
    ):
        subset = data[data.selection.eq(key)].sort_values("implementation_allowance")
        assert len(subset) == len(eta)
        for ax, vals in [
            (footprint, subset.physical_qubits / 1e5),
            (timing, subset.complete_task_hours),
        ]:
            ax.bar(x + (i - 0.5) * 0.36, vals, width=0.345, color=col, label=label)
    for ax in (footprint, timing):
        ax.set_xticks(x, [rf"$10^{{{int(np.round(np.log10(v)))}}}$" for v in eta])
        ax.set_xlabel(r"Error budget, $E_{\mathrm{max}}$", fontsize=6.2, labelpad=3.2)
    footprint.set_ylabel(r"Qubits ($10^5$)", labelpad=1.5)
    timing.set_ylabel("Total task time (h)", labelpad=1.5)
    timing.set_yscale("log")
    timing.set_ylim(0.95, 150)
    for name, color, marker, label, face in [
        ("Original 33-step data", PURPLE, "s", "Silva [40]", PURPLE),
        ("A Game of Surface Codes", ORANGE, "^", "Litinski [39]", ORANGE),
        ("Not as Costly", TEAL, "o", "Litinski [47]", "white"),
    ]:
        sub = back[back.comparison_label.eq(name)].sort_values("p_phys")
        assert len(sub) == 8
        factories.plot(
            sub.physical_qubits / 1e5,
            sub.p_phys * 1e4,
            lw=0.75,
            color=color,
            marker=marker,
            mfc=face,
            mec=color,
            mew=0.55,
            ms=3,
            label=label,
        )
    factories.set(
        xlabel=r"Minimum physical qubits ($10^5$)",
        ylabel=r"$p_{\rm phys}$ ($10^{-4}$)",
        yticks=[2, 4, 6, 8, 9],
    )
    factories.set_xlim(0.24, 2.18)
    factories.set_xticks([0.5, 1, 1.5, 2])
    for ax in (footprint, timing, factories):
        style(ax)
        ax.tick_params(length=2, pad=1.6)
    footprint.legend(
        loc="upper right",
        frameon=False,
        fontsize=5.55,
        handlelength=0.90,
        handletextpad=0.34,
        borderpad=0.15,
        borderaxespad=0.16,
        labelspacing=0.18,
    )
    footprint.set_ylim(0, 10)
    factories.legend(
        loc="lower right",
        frameon=False,
        fontsize=5.65,
        handlelength=1.35,
        handletextpad=0.4,
        borderpad=0.18,
        borderaxespad=0.13,
        labelspacing=0.25,
    )
    fig.text(0.015, 0.952, "a", weight="bold", fontsize=7.5)
    fig.text(0.015, 0.504, "b", weight="bold", fontsize=7.5)
    return fig


def plot(output=None):
    root = Path(__file__).resolve().parents[1]
    out = Path(output) if output else output_dir("S08")
    out.mkdir(parents=True, exist_ok=True)
    options = {
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "font.size": 6.4,
        "axes.labelsize": 6.3,
        "xtick.labelsize": 5.9,
        "ytick.labelsize": 5.9,
        "axes.linewidth": 0.6,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "figure.dpi": 150,
        "savefig.dpi": 400,
        "svg.fonttype": "path",
        "pdf.fonttype": 42,
        "svg.hashsalt": {8: "kvn-supplement-10", 10: "kvn-supplement-9"}[8],
    }
    with plt.rc_context(options):
        pass
        fig = sensitivity(figure_data("S8") / "plot_values")
        for ext in ["svg", "pdf", "png"]:
            meta = (
                {"Date": None}
                if ext == "svg"
                else {"CreationDate": None} if ext == "pdf" else None
            )
            fig.savefig(out / f"Figure_S8.{ext}", facecolor="white", metadata=meta)
        plt.close(fig)


if __name__ == "__main__":
    plot(sys.argv[1] if len(sys.argv) > 1 else None)
