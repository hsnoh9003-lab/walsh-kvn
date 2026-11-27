"""Reproduce Extended Data Figure 1 from archived IBM MPS data; no provider access."""

import sys as _figure_sys
from pathlib import Path as _FigurePath

_figure_sys.path.insert(0, str(_FigurePath(__file__).resolve().parents[3]))
from Figure_Plots.common.paths import output_dir

# Public Source Data mapping (single canonical file per published input).
import sys as _kvn_path_sys
from pathlib import Path as _kvn_path_Path

_kvn_path_sys.path.insert(0, str(_kvn_path_Path(__file__).resolve().parents[3]))
from Source.data_locations import published_path
from pathlib import Path
import argparse
import csv
import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
DATA = published_path("Hardware/IBM/data/ibm_mps_calibration_full_precision.csv").parent
plt.rcParams.update(
    {
        "font.family": "Liberation Sans",
        "mathtext.fontset": "dejavusans",
        "font.size": 6.5,
        "axes.linewidth": 0.6,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": None,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "svg.hashsalt": "walsh-kvn-ed1",
    }
)
BLUE, ORANGE, INK, GRAY = "#35658C", "#B95432", "#171717", "#888888"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, nargs="?", default=output_dir("ED1"))
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    source = DATA / "ibm_mps_calibration_full_precision.csv"
    rows = list(csv.DictReader(source.open()))
    fig, axes = plt.subplots(
        2, 2, figsize=(172.72 / 25.4, 118.745 / 25.4), sharex=True, sharey="col"
    )
    fig.subplots_adjust(
        left=0.095, right=0.99, bottom=0.10, top=0.93, wspace=0.24, hspace=0.30
    )
    for row_index, nt in enumerate((4, 6)):
        series = [r for r in rows if int(r["nt"]) == nt and int(r["cells"]) == 16]
        col = lambda key: np.array([float(r[key]) for r in series])
        times = col("t_star")
        for col_index, (key, label) in enumerate(
            (("P", r"$P(r\geq7/8)$"), ("r", r"$\langle r\rangle$"))
        ):
            ax = axes[row_index, col_index]
            ax.plot(times, col(key + "_ideal"), color=INK, linewidth=0.8, zorder=2)
            if key == "r":
                ax.plot(
                    times,
                    col("r_continuum"),
                    "--",
                    color=GRAY,
                    linewidth=0.8,
                    dashes=(4, 2),
                    zorder=1,
                )
                ax.axhline(
                    31 / 32,
                    color="#A9A9A9",
                    linewidth=0.55,
                    linestyle=(0, (1.5, 2.5)),
                    zorder=0,
                )
                ax.text(40.2, 31 / 32 + 0.006, "31/32", color="#777777", fontsize=6)
            # Keep the endpoints visible around small markers at publication size.
            # Cap width is in display points. All vertical lengths use the source errors.
            ax.errorbar(
                times,
                col(key + "_raw"),
                yerr=col(key + "_raw_SE"),
                fmt="none",
                ecolor=BLUE,
                elinewidth=0.68,
                capsize=3.1,
                capthick=0.62,
                zorder=3,
                gid=f"{row_index}-{key}-measured-errors",
            )
            ax.plot(
                times,
                col(key + "_raw"),
                "o",
                color=BLUE,
                markerfacecolor="white",
                markersize=3.25,
                markeredgewidth=0.72,
                zorder=4,
            )
            # At k=0 only the preparation was measured; there is no paired reference.
            good = [r for r in series if int(r["k"]) > 0]
            mitigated_t = [float(r["t_star"]) for r in good]
            mitigated_y = [float(r[key + "_refnorm"]) for r in good]
            ax.errorbar(
                mitigated_t,
                mitigated_y,
                yerr=[float(r[key + "_refnorm_err"]) for r in good],
                fmt="none",
                ecolor=ORANGE,
                elinewidth=0.72,
                capsize=3.1,
                capthick=0.62,
                zorder=3,
            )
            ax.plot(
                mitigated_t,
                mitigated_y,
                "s",
                color=ORANGE,
                markeredgecolor=ORANGE,
                markersize=3.0,
                markeredgewidth=0.45,
                zorder=4,
            )
            ax.set_ylabel(label, labelpad=4)
            ax.set_xlim(39.7, 50.1)
            ax.set_xticks([40, 42, 44, 46, 48, 50])
            ax.tick_params(length=3, width=0.6, labelsize=6)
            ax.spines[["top", "right"]].set_visible(False)
            ax.text(
                -0.15,
                1.065,
                "abcd"[2 * row_index + col_index],
                transform=ax.transAxes,
                fontsize=7,
                fontweight="bold",
                va="bottom",
            )
            if col_index == 0:
                ax.set_title(
                    f'{nt} steps, {int(series[0]["shots"]):,} shots per circuit',
                    loc="left",
                    pad=9,
                    fontsize=6.5,
                )
            if row_index == 1:
                ax.set_xlabel(r"$t^*$", labelpad=3)
            if key == "P":
                ax.set_ylim(0.25, 1.12)
                ax.set_yticks([0.4, 0.6, 0.8, 1.0])
            else:
                ax.set_ylim(0.60, 1.065)
                ax.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
    handles = [
        Line2D([], [], color=INK, lw=0.8, label="Ideal circuit"),
        Line2D(
            [],
            [],
            marker="o",
            color=BLUE,
            mfc="white",
            mew=0.72,
            ms=3.25,
            lw=0,
            label="Measured",
        ),
        Line2D(
            [],
            [],
            marker="s",
            color=ORANGE,
            mec=ORANGE,
            mew=0.45,
            ms=3.0,
            lw=0,
            label="Noise-mitigated",
        ),
        Line2D(
            [], [], color=GRAY, lw=0.8, linestyle="--", label="Continuum radial model"
        ),
    ]
    axes[0, 0].legend(
        handles=handles,
        loc="lower left",
        ncol=2,
        frameon=False,
        fontsize=6.5,
        handlelength=1.8,
        columnspacing=1.2,
        labelspacing=0.45,
        borderaxespad=0.7,
    )
    for suffix in ["png", "svg"]:
        fig.savefig(
            output / f"Extended_Data_Figure_1.{suffix}",
            bbox_inches=None,
            pad_inches=0,
            metadata={"Date": None} if suffix == "svg" else None,
        )
    fig.savefig(
        output / "Extended_Data_Figure_1.pdf",
        bbox_inches=None,
        pad_inches=0,
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(fig)
    print(
        "Extended Data Figure 1: vector PDF/SVG and 300-dpi PNG from published Source Data."
    )


if __name__ == "__main__":
    main()
