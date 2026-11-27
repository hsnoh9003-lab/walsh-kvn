#!/usr/bin/env python3
"""Draw the five-panel Figure 5 from the archived numerical source tables."""

# Direct-file execution must locate Codes/ independently of the working directory.
if __package__ in (None, ""):
    import sys as _path_sys
    from pathlib import Path as _Path
    _codes = next(p for p in _Path(__file__).resolve().parents
                  if (p / "Source/data_locations.py").is_file())
    _path_sys.path.insert(0, str(_codes))

from Source.data_locations import generated_outputs_root
import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, NullLocator
import numpy as np

from append_wake_panel import columns, read_resources, trace


ROOT = Path(__file__).resolve().parents[1]
WIDTH, HEIGHT = 488, 492
INK = "#30373B"
BLUE = "#236A9A"
RUST = "#C06143"
PURPLE = "#8266A0"
ORANGE = "#B65A22"
DARK = "#123A63"
RED = "#B3261E"
GRID = "#D8DDE1"
GRAY = "#D6DCE0"
DASHED = (0, (3, 2))

plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.color": INK,
        "ytick.color": INK,
        "text.color": INK,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "svg.hashsalt": "Figure6-five-panels",
    }
)


def records(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def axes(fig, left, top, width, height):
    ax = fig.add_axes(
        [left / WIDTH, 1 - (top + height) / HEIGHT, width / WIDTH, height / HEIGHT]
    )
    ax.tick_params(length=2.2, width=0.6, pad=2, labelsize=7)
    return ax


def text(fig, x, y, label, **style):
    return fig.text(x / WIDTH, 1 - y / HEIGHT, label, **style)


def panel(fig, letter, top, left=7):
    text(fig, left, top, letter, fontsize=10, fontweight="bold", va="top")


def legend(fig, handles, x, y, ncol=2, size=7.3):
    fig.legend(
        handles=handles,
        loc="center",
        bbox_to_anchor=(x / WIDTH, 1 - y / HEIGHT),
        ncol=ncol,
        frameon=False,
        fontsize=size,
        handlelength=1.3,
        handletextpad=0.42,
        columnspacing=1.0,
        borderaxespad=0,
    )


def inside_legend(ax, handles, loc, size=6.8):
    ax.legend(
        handles=handles,
        loc=loc,
        frameon=False,
        fontsize=size,
        handlelength=1.25,
        handletextpad=0.4,
        labelspacing=0.28,
        borderaxespad=0.4,
        borderpad=0,
    )


def model_accuracy(fig, root):
    panel(fig, "a", 6)
    data = root / "mergeab/input"
    models = json.loads((data / "protocol.json").read_text())["models"]
    errors = {row["id"]: row for row in records(data / "selected_moment_errors.csv")}
    samples = {row["id"]: row for row in records(data / "sampling_errorbars.csv")}
    model_info = (
        ("Q2", [0, 1], [1], BLUE),
        ("C3", [0, 0.5, 1], [0.5], RUST),
        ("Q5", [0, 0.25, 0.5, 0.75, 1], [0, 0.5, 1], PURPLE),
        ("2D C3", [], [], INK),
    )
    centres = (34, 55, 76, 97)
    text(fig, 97, 14, "Model", ha="center", va="center")
    for boundary in (23.5, 44.5, 65.5, 86.5, 107.5):
        yy = 1 - boundary / HEIGHT
        fig.add_artist(
            Line2D(
                [7 / WIDTH, 481 / WIDTH],
                [yy, yy],
                transform=fig.transFigure,
                color=GRID,
                lw=0.45,
                zorder=0,
            )
        )
    for i, (name, roots, stable, color) in enumerate(model_info):
        text(fig, 28, centres[i], name, ha="center", va="center")
        if i == 3:
            continue
        ax = axes(fig, 56, centres[i] - 8, 77, 16)
        ax.set_facecolor("none")
        x = np.linspace(0, 1, 1001)
        coeff = (
            np.array(models[i]["raw_coefficients"]) * models[i]["normalization_scale"]
        )
        ax.axhline(0, color="#BDC5CA", lw=0.5, zorder=0)
        ax.plot(x, np.polynomial.polynomial.polyval(x, coeff), color=color, lw=0.8)
        for root in roots:
            ax.plot(
                root,
                0,
                "o",
                ms=3,
                mec=color,
                mfc=color if root in stable else "white",
                mew=0.65,
                zorder=4,
                clip_on=False,
            )
        ax.set(
            xlim=(-0.03, 1.03),
            ylim=(-0.33, 0.33),
            xticks=(0, 0.5, 1),
            yticks=(-0.25, 0, 0.25),
        )
        ax.tick_params(axis="both", labelsize=6.5, length=1.3, pad=0.7)
        ax.tick_params(axis="y", labelleft=i == 0)
        ax.tick_params(axis="x", labelbottom=i == 2)
        ax.set_xticklabels(("0", ".5", "1"))
        ax.set_yticklabels(("−.25", "0", ".25"))
    text(fig, 139, 87, r"$x$", ha="left", va="center", fontsize=7)

    coupled = axes(fig, 82, 89.5, 26, 16)
    coupled.set_facecolor("none")
    x, y = np.meshgrid(np.linspace(0.2, 0.8, 3), np.linspace(0.2, 0.8, 3))
    norm_x, norm_y = models[3]["normalization_scale"]
    coupled.quiver(
        x,
        y,
        norm_x * x * (1 - x) * (1 + 2 * y),
        norm_y * y * (1 - y) * (-2 + x),
        color="#8699A7",
        angles="xy",
        scale_units="xy",
        scale=0.85,
        width=0.029,
        headwidth=3.4,
    )
    coupled.plot(0, 1, "o", mfc="white", mec=INK, ms=2.8, mew=0.65, clip_on=False)
    coupled.plot(1, 0, "o", mfc=INK, mec=INK, ms=2.8, mew=0.65, clip_on=False)
    coupled.set(xlim=(-0.04, 1.04), ylim=(-0.04, 1.04), xticks=[], yticks=[])

    cases = (
        ("interior", "Interior"),
        ("near_stable", "Near stable"),
        ("near_unstable", "Near unstable"),
    )
    for left, (key, label) in zip((151, 263, 375), cases):
        text(fig, left + 53, 14, label, ha="center", va="center")
        ax = axes(fig, left, 23.5, 106, 84)
        ax.set_facecolor("none")
        ax.set(xscale="log", xlim=(3e-6, 0.03), ylim=(-0.5, 3.5), yticks=[])
        ax.grid(axis="x", color=GRID, lw=0.45, zorder=0)
        for i, model in enumerate(models):
            name = f'{model["name"]}__{key}'
            sample = samples[name]
            error = float(errors[name]["moment_RMS_error"])
            assert error == float(sample["deterministic_RMS"])
            median, lower, upper = (
                float(sample[name])
                for name in (
                    "sampled_RMS_median",
                    "sampled_RMS_lower",
                    "sampled_RMS_upper",
                )
            )
            ax.errorbar(
                median,
                3 - i + 0.13,
                xerr=[[median - lower], [upper - median]],
                fmt="s",
                color=DARK,
                mfc="white",
                mec=DARK,
                ms=3,
                mew=0.55,
                elinewidth=0.7,
                capsize=1.6,
                capthick=0.6,
            )
            ax.plot(error, 3 - i - 0.13, "D", mec=RED, mfc="white", ms=3, mew=0.75)
        ax.set_xticks((1e-4, 1e-2), labels=(r"$10^{-4}$", r"$10^{-2}$"))
        ax.minorticks_off()

    fixed_points = [
        Line2D(
            [],
            [],
            marker="o",
            mec=INK,
            mfc=fill,
            ms=3,
            mew=0.65,
            ls="none",
            label=f"{status} fixed point",
        )
        for status, fill in (("Stable", INK), ("Unstable", "white"))
    ]
    legend(fig, fixed_points, 67, 126, ncol=1)
    accuracy = [
        Line2D(
            [],
            [],
            marker="D",
            mec=RED,
            mfc="white",
            ms=3,
            mew=0.75,
            ls="none",
            label="Statevector",
        ),
        Line2D(
            [],
            [],
            marker="s",
            mec=DARK,
            mfc="white",
            ms=3,
            mew=0.55,
            ls="none",
            label="Sampled",
        ),
    ]
    legend(fig, accuracy, 315, 123)
    text(
        fig,
        315,
        134,
        r"Moment RMS error, $\varepsilon_{\mathrm{mom}}$",
        ha="center",
        va="center",
    )


def quintic_density(fig, root):
    panel(fig, "b", 148)
    cells = records(root / "upper_baseline/input/scalar_conservative_cell_masses.csv")
    cases = (
        ("interior", r"Interior, $x_0=0.35$"),
        ("near_unstable", r"Unstable centre, $x_0=0.25$"),
    )
    density_styles = [
        Line2D([], [], color=GRAY, lw=3, label="Initial"),
        Line2D([], [], color=BLUE, lw=0.85, label="Circuit"),
        Line2D([], [], color=INK, lw=0.8, ls=DASHED, label="Reference"),
    ]
    for i, ((key, label), left) in enumerate(zip(cases, (32, 182))):
        ax = axes(fig, left, 176, 125, 60)
        text(fig, left + 62.5, 167, label, ha="center", va="center")
        rows = sorted(
            (row for row in cells if row["id"] == f"quintic_multistable__{key}"),
            key=lambda row: int(row["cell"]),
        )
        assert len(rows) == 64
        edges = np.r_[[float(row["left"]) for row in rows], float(rows[-1]["right"])]
        initial, final, reference = (
            np.array([float(row[name]) for row in rows])
            for name in (
                "initial_prepared_mass",
                "final_circuit_mass",
                "final_matched_characteristic_mass",
            )
        )
        for root in (0.25, 0.75):
            ax.axvline(root, color="#BBC2C7", ls=(0, (1.2, 2)), lw=0.55, zorder=0)
        ax.stairs(
            64 * initial, edges, fill=True, color=GRAY, alpha=0.75, lw=0.5, zorder=1
        )
        ax.stairs(64 * final, edges, color=BLUE, lw=0.85, zorder=2)
        ax.stairs(64 * reference, edges, color=INK, ls=DASHED, lw=0.75, zorder=3)
        for root in (0, 0.25, 0.5, 0.75, 1):
            ax.plot(
                root,
                0,
                "o",
                mfc=INK if root in (0, 0.5, 1) else "white",
                mec=INK,
                ms=2.9,
                mew=0.6,
                zorder=4,
                clip_on=False,
            )
        ax.set(
            xlim=(-0.025, 1.025),
            ylim=(0, 24),
            xticks=(0, 0.25, 0.5, 0.75, 1),
            xticklabels=("0", ".25", ".5", ".75", "1"),
            yticks=(0, 10, 20),
            xlabel="$x$",
        )
        ax.xaxis.labelpad = 1.5
        if i == 0:
            ax.set_ylabel("Density", labelpad=2)
            inside_legend(ax, density_styles, "upper right")
        else:
            ax.tick_params(labelleft=False)


def coupled_resources(fig, root):
    panel(fig, "c", 148, left=334)
    rows = [
        row
        for row in records(
            root / "upper_baseline/resource_estimation/selected_resources.csv"
        )
        if row["model"] == "2d_interior"
    ]
    qubits = axes(fig, 372, 153, 104, 48)
    runtime = axes(fig, 372, 208, 104, 28)
    keys = (
        ("minimum_qubits", RED, "s", DASHED, "Min. qubits"),
        ("fastest_uncapped", DARK, "o", "-", "Fastest"),
    )
    handles = []
    for key, color, marker, line, label in keys:
        selected = sorted(
            (row for row in rows if row["selection"] == key),
            key=lambda row: float(row["p_phys"]),
        )
        assert len(selected) == 8
        rates = [float(row["p_phys"]) * 1e4 for row in selected]
        style = dict(
            color=color,
            marker=marker,
            ms=2.8,
            mfc="white",
            mec=color,
            mew=0.7,
            lw=0.8,
            ls=line,
            label=label,
        )
        handles.extend(
            qubits.plot(
                rates,
                [float(row["physical_qubits"]) / 1e5 for row in selected],
                **style,
            )
        )
        runtime.plot(
            rates, [float(row["complete_task_hours"]) for row in selected], **style
        )
    inside_legend(qubits, handles, "upper left")
    qubits.set(xlim=(1.6, 9.4), ylim=(0, 10), xticks=(2, 5, 9), yticks=(0, 5, 10))
    qubits.tick_params(labelbottom=False)
    qubits.set_ylabel(r"Qubits ($10^5$)", fontsize=7.2, labelpad=1)
    qubits.yaxis.set_label_coords(-0.17, 0.52)
    runtime.set(
        xlim=(1.6, 9.4),
        ylim=(20, 400),
        yscale="log",
        xticks=(2, 5, 9),
        yticks=(30, 100, 300),
        yticklabels=("30", "100", "300"),
    )
    runtime.yaxis.set_minor_locator(NullLocator())
    runtime.set_ylabel("Time (h)", fontsize=7.2, labelpad=1)
    runtime.yaxis.set_label_coords(-0.17, 0.43)
    runtime.set_xlabel(r"$p_{\mathrm{phys}}$ ($10^{-4}$)", labelpad=1.5)


def wake_data(root):
    times, *reference = columns(
        root / "long298/common_reference.csv",
        ["time", *[f"reference_mean_{axis}" for axis in "xyz"]],
    )
    runs = json.loads((root / "selected_runs.json").read_text())
    items = []
    for key, start, color, marker in (
        ("long", 29.8, BLUE, "o"),
        ("late", 40, ORANGE, "D"),
    ):
        row = runs[key]
        time, means, error = trace(root / row["trajectory"])
        np.testing.assert_allclose(
            [time[0], time[-1]], [start, times[-1]], atol=1e-8, rtol=0
        )
        items.append(
            (
                time,
                means,
                error,
                color,
                marker,
                f'{start:g} start, {row["steps"]} steps',
                root / row["resources"],
            )
        )
    return times, reference, items


def wake_axis(fig, left, top, width, height):
    ax = axes(fig, left, top, width, height)
    ax.grid(axis="y", color=GRID, lw=0.45)
    ax.set_axisbelow(True)
    return ax


def time_axis(ax, times):
    ax.set_xlim(times[0] - 0.7, times[-1] + 0.7)
    ax.set_xticks([times[0], 40, times[-1]], ["29.8", "40", "49.83"])


def wake_means(fig, times, reference, runs):
    panel(fig, "d", 263)
    handles = [
        Line2D(
            [],
            [],
            color=color,
            marker=marker,
            ms=3.1,
            mfc="white",
            mew=0.7,
            ls="none",
            label=label,
        )
        for _, _, _, color, marker, label, _ in runs
    ]
    handles.append(Line2D([], [], color="black", lw=0.8, ls=DASHED, label="ODE"))
    for i, (left, coord) in enumerate(zip((35, 201, 367), "xyz")):
        ax = wake_axis(fig, left, 274, 111, 76)
        ax.plot(times, reference[i], color="black", lw=0.8, ls=DASHED)
        for time, means, _, color, marker, _, _ in runs:
            take = np.unique(
                np.linspace(0, len(time) - 1, min(17, len(time))).round().astype(int)
            )
            ax.plot(
                time[take],
                means[i, take],
                color=color,
                marker=marker,
                ms=2.9,
                mfc="white",
                mew=0.7,
                ls="none",
            )
        time_axis(ax, times)
        ax.set_ylabel(r"$\langle " + coord + r"\rangle$", labelpad=2)
        if coord in "xy":
            ax.set(ylim=(-1.08, 1.08), yticks=(-1, 0, 1))
        else:
            ax.set_yticks((0.5, 0.75, 1), labels=("0.50", "0.75", "1.00"))
            inside_legend(ax, handles, "lower right")
    text(fig, 256.5, 368, r"$tU_\infty/D$", ha="center", va="center")


def wake_resources(fig, times, runs):
    panel(fig, "e", 375)
    selections = (
        ("minimum_qubits", DASHED, "Min. qubits"),
        ("fastest_uncapped", "-", "Fastest"),
    )
    handles = [
        Line2D(
            [],
            [],
            color=INK,
            ls=line,
            lw=0.8,
            marker="o",
            ms=3.1,
            mfc="white" if key == "minimum_qubits" else INK,
            label=label,
        )
        for key, line, label in selections
    ]
    error, qubits, runtime = (
        wake_axis(fig, left, 391, 111, 79) for left in (35, 201, 367)
    )
    inside_legend(qubits, handles, "upper left")
    error.axhline(1, color="#838b92", lw=0.65, ls=(0, (2, 2)), zorder=1)
    for time, _, bias, color, marker, _, path in runs:
        error.plot(time, bias, color=color, lw=0.85)
        for key, line, _ in selections:
            style = dict(
                color=color,
                marker=marker,
                ms=2.9,
                mfc="white" if key == "minimum_qubits" else color,
                mec=color,
                mew=0.65,
                lw=0.8,
                ls=line,
            )
            qubits.plot(
                np.arange(2, 10), read_resources(path, key, "physical_qubits"), **style
            )
            runtime.plot(
                np.arange(2, 10),
                read_resources(path, key, "complete_task_days"),
                **style,
            )
    error.set_ylim(bottom=0)
    error.yaxis.set_major_locator(MaxNLocator(4))
    time_axis(error, times)
    error.set_ylabel("Noiseless field error (%)", fontsize=7.3, labelpad=2)
    error.set_xlabel(r"$tU_\infty/D$", labelpad=1.5)
    for ax, label in ((qubits, "Qubits"), (runtime, "Total task time (d)")):
        ax.set(xlim=(1.55, 9.45), xticks=(2, 5, 9), yscale="log")
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_ylabel(label, labelpad=2)
        ax.set_xlabel(r"$p_{\mathrm{phys}}$ ($10^{-4}$)", labelpad=1.5)


def build(root, output):
    fig = plt.figure(figsize=(WIDTH / 72, HEIGHT / 72), facecolor="white")
    model_accuracy(fig, root)
    quintic_density(fig, root)
    coupled_resources(fig, root)
    times, reference, runs = wake_data(root)
    wake_means(fig, times, reference, runs)
    wake_resources(fig, times, runs)

    output.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"savefig.bbox": None}):
        for ext in ("pdf", "svg"):
            stamp = (
                {"CreationDate": None, "ModDate": None}
                if ext == "pdf"
                else {"Date": None}
            )
            fig.savefig(output / f"Figure5_compact.{ext}", metadata=stamp)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=generated_outputs_root() / "Figures/F5/variants/compact",
    )
    args = parser.parse_args()
    build(args.root, args.output)
