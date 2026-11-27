"""Plot the two forecast windows and the six specified initial disturbances."""

import sys as _figure_sys
from pathlib import Path as _FigurePath

_figure_sys.path.insert(0, str(_FigurePath(__file__).resolve().parents[3]))
from Figure_Plots.common.paths import output_dir
from pathlib import Path
import sys
import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
CASES = tuple("ABCDEF")
WIDTH_MM, HEIGHT_MM = 180, 72
CHART_HEIGHT_MM = 50
MAX_ERROR = 22
TICKS = [0, 5, 10, 15, 20]
WINDOWS = [("short", "Short forecast"), ("extended", "Extended forecast")]
SERIES = [
    ("Cubic field error (%)", "Cubic model", "#216d8f"),
    ("Projection error (%)", "Three-mode projection", "#adbdc6"),
]


def draw_case_settings(fig, source):
    with (source / "case_settings.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert tuple(row["case"] for row in rows) == CASES
    parameters = [
        (r"Amplitude, $A_\psi$", "streamfunction_amplitude", "+.3f"),
        (r"Downstream of cylinder centre, $x_c$ (diameters)", "centre_x", ".1f"),
        (r"Height above wake centreline, $y_c$ (diameters)", "centre_y", ".2f"),
        (r"Width, $w$ (diameters)", "width", ".2f"),
    ]
    values = []
    for label, key, fmt in parameters:
        entries = []
        for row in rows:
            value = float(row[key])
            number = format(value, "+.2f" if key == "centre_y" and value != 0 else fmt)
            entries.append(number.replace("-", "−"))
        values.append([label, *entries])
    fig.text(0.087, 0.307, "c", fontsize=9, weight="bold", ha="left", va="center")
    fig.text(
        0.538,
        0.307,
        r"$\delta\psi(x,y)=A_\psi\,C(x)\,\exp\!\left[-\frac{(x-x_c)^2+(y-y_c)^2}{w^2}\right]$",
        fontsize=10.8,
        ha="center",
        va="center",
    )
    ax = fig.add_axes([0.087, 0.024, 0.902, 0.228])
    ax.set_axis_off()
    table = ax.table(
        cellText=values,
        colLabels=["Initial disturbance", *CASES],
        cellLoc="center",
        colWidths=[0.40, *([0.10] * len(CASES))],
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    for (row, column), cell in table.get_celld().items():
        cell.visible_edges = ""
        cell.PAD = 0.025
        if column == 0:
            cell.get_text().set_ha("left")
        if row == 0:
            cell.get_text().set_weight("bold")
            cell.visible_edges = "B"
            cell.set_edgecolor("#9aa8ad")
            cell.set_linewidth(0.45)


def main():
    from Source.data_locations import shared_data

    source = shared_data("CFD_ROM", "Physical_Forecast_Audit")
    style = {
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.linewidth": 0.6,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "svg.fonttype": "path",
        "svg.hashsalt": "kvn-s1-two-column",
        "pdf.fonttype": 42,
    }
    with plt.rc_context(style):
        fig, axes = plt.subplots(
            1, 2, sharey=True, figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4)
        )
        table_height = HEIGHT_MM - CHART_HEIGHT_MM
        fig.subplots_adjust(
            left=0.087,
            right=0.989,
            wspace=0.17,
            bottom=(table_height + 0.233 * CHART_HEIGHT_MM) / HEIGHT_MM,
            top=(table_height + 0.829 * CHART_HEIGHT_MM) / HEIGHT_MM,
        )
        x = np.arange(len(CASES))
        for idx, (ax, (name, title)) in enumerate(zip(axes, WINDOWS)):
            with (source / f"{name}_window.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            assert tuple(row["Case"] for row in rows) == CASES
            for (key, label, colour), shift in zip(SERIES, [-0.17, 0.17]):
                values = [float(row[key]) for row in rows]
                assert all(0 <= value < MAX_ERROR for value in values)
                ax.bar(
                    x + shift, values, width=0.32, color=colour, label=label, zorder=2
                )
            ax.set_xlim(-0.54, len(CASES) - 0.46)
            ax.set_xticks(x, CASES)
            ax.set_xlabel("Flow case", fontsize=8.0, labelpad=2.5)
            ax.set_ylim(0, MAX_ERROR)
            ax.set_yticks(TICKS)
            ax.grid(axis="y", color="#dce1e3", linewidth=0.40)
            ax.set_axisbelow(True)
            ax.tick_params(length=2.7, pad=2.2, labelsize=7.4, labelleft=True)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_title(title, pad=4.7, fontsize=8.7)
            ax.text(
                0.0,
                1.075,
                "ab"[idx],
                transform=ax.transAxes,
                weight="bold",
                fontsize=9,
                ha="left",
                va="bottom",
            )
        axes[0].set_ylabel("Field error vs FEM (%)", fontsize=7.9, labelpad=4.0)
        axes[0].legend(
            loc="upper left",
            frameon=False,
            fontsize=7.5,
            handlelength=1.25,
            handletextpad=0.45,
            labelspacing=0.28,
            borderaxespad=0.42,
            borderpad=0.06,
        )
        draw_case_settings(fig, source)
        output = output_dir("S01")
        output.mkdir(exist_ok=True)
        for ext, metadata in [
            ("pdf", {"CreationDate": None, "ModDate": None}),
            ("svg", {"Date": None}),
            ("png", None),
        ]:
            fig.savefig(
                output / f"Figure_S1.{ext}", metadata=metadata, facecolor="white"
            )
        plt.close(fig)
    audit = {
        "panels": 3,
        "forecast_panels": len(WINDOWS),
        "cases_per_panel": len(CASES),
        "source": "Data/Physical_Forecast_Audit",
        "numerical_generator": "python -m Source.CFD.physical_audit",
        "rendering": "Panels a and b show forecast errors. Panel c gives the streamfunction profile, signed centreline offset and all six perturbation settings.",
        "width_mm": WIDTH_MM,
        "height_mm": HEIGHT_MM,
        "shared_y_limits_percent": [0, MAX_ERROR],
        "shared_y_ticks_percent": TICKS,
        "axis_size_pt": 7.9,
        "legend_size_pt": 7.5,
        "panel_letter_size_pt": 9.0,
    }
    (output / "plot_audit.json").write_text(json.dumps(audit, indent=2) + "\n")


if __name__ == "__main__":
    main()
