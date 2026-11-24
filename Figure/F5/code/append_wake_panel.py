"""Draw two wake propagations against one continuum ensemble and append panel f."""
import argparse
import csv
from pathlib import Path

import fitz
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, NullLocator
import numpy as np

HEIGHT = 70
EARLY = "#236A9A"
LATE = "#B65A22"
INK = "#30373B"
GRID = "#DBE1E5"
SELECTIONS = (
    ("minimum_qubits", (0, (3, 2)), "Min. qubits"),
    ("fastest_uncapped", "-", "Fastest"),
)

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "font.size": 6.2,
    "axes.linewidth": .6,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "text.color": INK,
})


def columns(path, names):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    return np.array([[float(row[name]) for row in rows] for name in names])


def trace(path):
    time, x, y, z, error = columns(
        path, ["time", *[f"walsh_mean_{axis}" for axis in "xyz"],
               "field_relative_percent"]
    )
    return time, np.array([x, y, z]), error


def axis(fig, left, bottom, width, height):
    ax = fig.add_axes([left, bottom / HEIGHT, width, height / HEIGHT])
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2, width=.6, pad=1.2, labelsize=5.5)
    ax.grid(axis="y", color=GRID, lw=.4)
    ax.set_axisbelow(True)
    return ax


def time_axis(ax, times):
    ax.set_xlim(times[0]-.7, times[-1]+.7)
    ax.set_xticks([times[0], 40, times[-1]], ["29.8", "40", "49.83"])


def read_resources(path, selection, column):
    with path.open(newline="") as stream:
        rows = {round(float(row["p_phys"])*1e4): row for row in csv.DictReader(stream)
                if row["selection"] == selection}
    return np.array([float(rows.get(rate, {}).get(column) or "nan")
                     for rate in range(2, 10)])


def plot(out, width_mm, reference, early_path, late_path, resources, early_steps, late_steps):
    times, *reference_xyz = columns(
        reference, ["time", *[f"reference_mean_{axis}" for axis in "xyz"]]
    )
    assert np.all(np.diff(times) > 0)
    assert np.isclose(times[0], 29.8) and np.any(np.isclose(times, 40))
    runs = (
        (trace(early_path), EARLY, "o", f"29.8 start, {early_steps} steps"),
        (trace(late_path), LATE, "D", f"40 start, {late_steps} steps"),
    )
    for ((t, _, _), _, _, _), start in zip(runs, (29.8, 40)):
        np.testing.assert_allclose([t[0], t[-1]], [start, times[-1]], atol=1e-8, rtol=0)

    fig = plt.figure(figsize=(width_mm / 25.4, HEIGHT / 25.4), facecolor="white")
    fig.text(.023, 68.6/HEIGHT, "e", fontsize=9, fontweight="bold", va="top")
    handles = [Line2D([], [], color=color, marker=marker, ms=2.75,
                      mfc="white", mew=.67, ls="none", label=label)
               for _, color, marker, label in runs]
    handles.append(Line2D([], [], color="black", lw=.8, ls=(0, (3, 2)),
                          label="ODE"))
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(.977, 68.8/HEIGHT),
               ncol=3, frameon=False, fontsize=5.55, handlelength=1.25,
               handletextpad=.35, columnspacing=.82, borderaxespad=0)

    for i, (left, coord) in enumerate(zip((.104, .413, .728), "xyz")):
        ax = axis(fig, left, 40.5, .233, 22.8)
        ax.plot(times, reference_xyz[i], color="black", lw=.8, ls=(0, (3, 2)))
        for (t, xyz, _), color, marker, _ in runs:
            take = np.unique(np.linspace(0, len(t)-1, min(17, len(t))).round().astype(int))
            ax.plot(t[take], xyz[i, take], color=color, marker=marker,
                    ms=2.35, mfc="white", mew=.68, ls="none")
        time_axis(ax, times)
        ax.set_ylabel(r"$\langle " + coord + r"\rangle$", fontsize=6.2, labelpad=1)
        if coord in "xy":
            ax.set_ylim(-1.08, 1.08)
            ax.set_yticks([-1, 0, 1])
        else:
            ax.set_yticks([.5, .75, 1], ["0.50", "0.75", "1.00"])
    fig.text(.533, 36/HEIGHT, r"$tU_\infty/D$", fontsize=6.2, ha="center", va="center")

    fig.text(.023, 33.6/HEIGHT, "f", fontsize=9, fontweight="bold", va="top")
    error = axis(fig, .102, 7.6, .23, 20)
    error.axhline(1, color="#838b92", lw=.65, ls=(0, (2, 2)), zorder=1)
    for (t, _, values), color, _, _ in runs:
        error.plot(t, values, color=color, lw=.85)
    error.set_ylim(bottom=0)
    error.yaxis.set_major_locator(MaxNLocator(4))
    time_axis(error, times)
    error.set_ylabel("Noiseless field error (%)", fontsize=6, labelpad=1)
    error.set_xlabel(r"$tU_\infty/D$", fontsize=6.1, labelpad=1)

    qubits = axis(fig, .445, 7.6, .2, 20)
    runtime = axis(fig, .774, 7.6, .197, 20)
    handles = [Line2D([], [], color=INK, ls=ls, lw=.8, marker="o", ms=2.6,
                      mfc="white" if key == "minimum_qubits" else INK, label=label)
               for key, ls, label in SELECTIONS]
    for path, (_, color, marker, _) in zip(resources, runs):
        for key, ls, _ in SELECTIONS:
            style = dict(color=color, marker=marker, ms=2.35,
                         mfc="white" if key == "minimum_qubits" else color,
                         mec=color, mew=.65, lw=.8, ls=ls)
            qubits.plot(np.arange(2, 10), read_resources(path, key, "physical_qubits"),
                        **style)
            runtime.plot(np.arange(2, 10), read_resources(path, key, "complete_task_days"),
                         **style)
    for ax in (qubits, runtime):
        ax.set_xlim(1.55, 9.45)
        ax.set_xticks([2, 5, 9])
        ax.set_xlabel(r"$p_{\mathrm{phys}}$ ($10^{-4}$)", fontsize=6, labelpad=1)
    qubits.set_yscale("log")
    qubits.yaxis.set_minor_locator(NullLocator())
    qubits.set_ylabel("Qubits", fontsize=6, labelpad=1)
    runtime.set_yscale("log")
    runtime.yaxis.set_minor_locator(NullLocator())
    runtime.set_ylabel("Total task time (d)", fontsize=6, labelpad=1)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.702, 28.6/HEIGHT),
               ncol=2, frameon=False, fontsize=5.4, handlelength=1.15,
               handletextpad=.35, columnspacing=.8, borderaxespad=0, borderpad=0)
    for suffix in ("pdf", "svg"):
        fig.savefig(out/f"Fig6e_wake.{suffix}", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("original", "figure", "reference", "long-trajectory", "late-trajectory",
                 "long-resources", "late-resources", "panel-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("long-steps", "late-steps"):
        parser.add_argument(f"--{name}", type=int, required=True)
    args = parser.parse_args()
    with fitz.open(args.original) as original:
        width, height = original[0].rect.br
        plot(args.panel_dir, width*25.4/72, args.reference,
             args.long_trajectory, args.late_trajectory,
             (args.long_resources, args.late_resources), args.long_steps, args.late_steps)
        with fitz.open(args.panel_dir/"Fig6e_wake.pdf") as panel, fitz.open() as doc:
            panel_height = panel[0].rect.height
            page = doc.new_page(width=width, height=height+panel_height+2.4)
            page.show_pdf_page(fitz.Rect(0, 0, width, height), original, 0)
            page.show_pdf_page(fitz.Rect(0, height+2.4, width, height+2.4+panel_height), panel, 0)
            doc.set_metadata({"title": "Figure 6 | Reuse across nonlinear dynamics and initial distributions"})
            doc.save(args.figure, garbage=4, deflate=True)


if __name__ == "__main__":
    main()
