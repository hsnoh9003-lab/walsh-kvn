"""Render the resource figure with three horizontal legend groups."""

import sys as _figure_sys
from pathlib import Path as _FigurePath

_figure_sys.path.insert(0, str(_FigurePath(__file__).resolve().parents[3]))
from Figure_Plots.common.paths import output_dir
from pathlib import Path
import argparse
import sys

import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch
from matplotlib.transforms import Bbox

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT.parents[1]))
from Source.data_locations import figure_data
import base


def shared_legend(fig):
    def text(x, y, value, heading=False):
        return fig.text(
            x / 88,
            y / 180,
            value,
            fontsize=5.2 if heading else 5.0,
            weight="bold" if heading else "normal",
            va="center",
        )

    def marker(x, y, shape, color, filled=True):
        fig.add_artist(
            Line2D(
                [x / 88],
                [y / 180],
                transform=fig.transFigure,
                marker=shape,
                ms=3.2,
                ls="none",
                mfc=color if filled else "white",
                mec=color,
                mew=0.65,
                zorder=6,
            )
        )

    text(13, 77, "Reference [a]", True)
    marker(14, 73.5, "*", "black")
    text(17, 73.5, "Hardware (no QEC)")
    marker(14, 70, "D", "black", False)
    text(17, 70, "QEC estimate")

    text(38.5, 77, "Method [b,c]", True)
    for model, y in zip(base.ORDER, (73.5, 70, 66.5)):
        fig.add_artist(
            Line2D(
                [39 / 88, 42.3 / 88],
                [y / 180, y / 180],
                transform=fig.transFigure,
                color=base.COLORS[model],
                lw=0.7,
                ls="--" if model == "nj_lchs" else "-",
                zorder=5,
            )
        )
        marker(39, y, base.MARKERS[model], base.COLORS[model])
        marker(42.3, y, base.MARKERS[model], base.COLORS[model], False)
        task = "sine drift" if model == "nj_lchs" else "wake"
        text(43.5, y, f"{base.LABELS[model]} ({task})")

    text(64.5, 77, "Optimize type [b,c]", True)
    for filled, y, value in ((False, 73.5, "Min. qubits"), (True, 70, "Fastest")):
        for x, shape in zip((65, 67.6, 70.2), ("s", "o", "^")):
            marker(x, y, shape, "#62676B", filled)
        text(73.6, y, value)


def main():
    parser = argparse.ArgumentParser(
        description="Draw Fig. 3 from the archived and reoptimized data."
    )
    parser.add_argument("output", nargs="?", type=Path, default=output_dir("F3"))
    parser.add_argument(
        "--full-range",
        action="store_true",
        help="Show all selected designs up to ten million physical qubits.",
    )
    parser.add_argument(
        "--clip-to-window",
        action="store_true",
        help="Clip panel b at one million physical qubits.",
    )
    args = parser.parse_args()
    output = args.output
    if args.full_range and args.clip_to_window:
        parser.error("Choose one plotting range.")
    if args.full_range:
        base.PANEL_B_QUBIT_WINDOW = (1e4, 1e7)
    if args.clip_to_window:
        base.PANEL_B_QUBIT_WINDOW = (1e4, 1e6)
    output.mkdir(parents=True, exist_ok=True)
    data = figure_data("F3")
    base.style()
    fig = base.plt.figure(figsize=(88 * base.MM, 180 * base.MM), facecolor="white")
    a = base.panel_a(
        fig,
        pd.read_csv(data / "nisq_examples.csv"),
        pd.read_csv(data / "reference_points.csv"),
    )
    b, b_count = base.panel_b(fig, pd.read_csv(data / "panel_b_selected_designs.csv"))
    _, c_count = base.panel_c(fig, pd.read_csv(data / "panel_c_time.csv"))
    for label, x, y in (("a", 2, 175), ("b", 2, 121), ("c", 55, 121)):
        fig.text(
            x / 88, y / 180, label, fontsize=8.5, weight="bold", ha="left", va="center"
        )
    shared_legend(fig)
    fig.add_artist(
        ConnectionPatch(
            (1e4, 1e-3),
            (1e4, 1e-3),
            "data",
            "data",
            axesA=a,
            axesB=b,
            color="#858585",
            lw=0.8,
            ls=(0, (3, 1.3)),
            clip_on=False,
            zorder=0.5,
        )
    )
    fig.canvas.draw()
    start = fig.transFigure.inverted().transform(a.transData.transform((1e6, 1e-4)))
    stop = fig.transFigure.inverted().transform(b.transData.transform((1e6, 1e-3)))
    fig.add_artist(
        Line2D(
            [start[0], 52 / 88, stop[0]],
            [start[1], 124 / 180, stop[1]],
            transform=fig.transFigure,
            color="#858585",
            lw=0.8,
            ls=(0, (3, 1.3)),
            clip_on=False,
            zorder=0.5,
        )
    )
    crop = Bbox.from_extents(0, 62.5 / 25.4, 88 / 25.4, 180 / 25.4)
    for suffix in ("pdf", "svg", "png"):
        metadata = (
            {"CreationDate": None, "ModDate": None}
            if suffix == "pdf"
            else {"Date": None} if suffix == "svg" else None
        )
        fig.savefig(
            output / f"Figure3.{suffix}",
            dpi=450,
            bbox_inches=crop,
            pad_inches=0,
            facecolor="white",
            metadata=metadata,
        )
    base.plt.close(fig)
    print(
        {
            "panel_b_visible_points": b_count,
            "panel_c_endpoints": c_count,
            "legend_order": ["Reference [a]", "Method [b,c]", "Optimize type [b,c]"],
        }
    )


if __name__ == "__main__":
    main()
