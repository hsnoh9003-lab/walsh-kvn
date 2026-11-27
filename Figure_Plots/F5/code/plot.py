"""One-column Figure 5, drawn from the archived figure's source data."""

import sys as _figure_sys
from pathlib import Path as _FigurePath

_figure_sys.path.insert(0, str(_FigurePath(__file__).resolve().parents[3]))
from Figure_Plots.common.paths import output_dir

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[2]))
from Source.data_locations import figure_data

FIGURE = figure_data("F5") / "wake"
DATA = figure_data("F5") / "panels"
sys.path.insert(0, str(ROOT))

from wake_data import load_runs, plt  # noqa: E402
from panels import BLUE, DARK, DASHED, GRAY, INK, RED  # noqa: E402
from append_wake_panel import read_resources, time_axis  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import NullLocator  # noqa: E402


WIDTH, HEIGHT = 89, 166
LEFT, RIGHT, TRACK = 10.4, 55.7, 31.7
RATES = np.arange(2, 10)
MODELS = ("Q2", "C3", "Q5", "2D C3")
STABLE_BLUE = "#266B98"
UNSTABLE_RED = "#B24D52"
CASES = (
    ("near_unstable", "Near unstable", UNSTABLE_RED),
    ("interior", "Interior", "#8064A2"),
    ("near_stable", "Near stable", STABLE_BLUE),
)
MODEL_INK = "#666666"
DATA_LINE_WIDTH = 0.7
LEGEND_CASES = (CASES[2], CASES[1], CASES[0])
MOMENT_MARKER_SIZE_PT = 2.6
DIAMOND_MARKER_SIZE_PT = 2.2

plt.rcParams.update(
    {
        "font.family": "Liberation Sans",
        "mathtext.fontset": "stix",
        "font.size": 6,
        "axes.labelsize": 6,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.color": INK,
        "ytick.color": INK,
        "pdf.fonttype": 42,
        "savefig.bbox": None,
        "savefig.dpi": 450,
        "svg.fonttype": "none",
    }
)


def label(fig, x, y, text, **kwargs):
    options = {"va": "center", "ha": "left", "fontsize": 6}
    options.update(kwargs)
    return fig.text(x / WIDTH, 1 - y / HEIGHT, text, **options)


def panel(fig, letter, x, y):
    label(fig, x, y, letter, va="top", fontsize=8, weight="bold")


def axis(fig, x, y, width, height):
    ax = fig.add_axes(
        [x / WIDTH, 1 - (y + height) / HEIGHT, width / WIDTH, height / HEIGHT]
    )
    ax.tick_params(labelsize=5.4, length=2, width=0.6, pad=1.35)
    return ax


def key(fig, handles, x, y, columns, size=5.35):
    return fig.legend(
        handles=handles,
        ncols=columns,
        loc="center",
        bbox_to_anchor=(x / WIDTH, 1 - y / HEIGHT),
        frameon=False,
        fontsize=size,
        handlelength=1.25,
        handletextpad=0.45,
        columnspacing=0.8,
        borderaxespad=0,
    )


def profiles(fig, edges):
    drift = pd.read_csv(DATA / "a_models/scalar_drift_curves.csv")
    fixed = pd.read_csv(DATA / "a_models/marked_equilibria.csv")
    vector = pd.read_csv(DATA / "a_models/coupled_drift_arrows.csv")
    assert len(vector) == 9
    label(fig, 21.8, 4.15, "Drift", ha="center", fontsize=6.4)

    for index, model in enumerate(MODELS):
        centre = (edges[index] + edges[index + 1]) / 2
        label(fig, 8.35, centre, model, ha="right", weight="semibold", fontsize=5.6)
        if model == "2D C3":
            ax = axis(fig, 16.7, centre - 6, 12, 12)
            ax.quiver(
                vector.x,
                vector.y,
                vector.dx_dt,
                vector.dy_dt,
                color=MODEL_INK,
                angles="xy",
                scale_units="xy",
                scale=0.85,
                width=0.022,
                headwidth=3.25,
            )
            ax.set(xlim=(-0.045, 1.045), ylim=(-0.045, 1.045), xticks=[], yticks=[])
        else:
            curve = drift[drift.model == model]
            assert len(curve) == 1001
            ax = axis(fig, 14.5, centre - (3.3 if model == "Q5" else 2.1), 16.4, 4.2)
            ax.axhline(0, color="#BEC5C9", lw=0.45, zorder=0)
            ax.plot(curve.x, curve.dx_dt, color=MODEL_INK, lw=DATA_LINE_WIDTH)
            ax.set(
                xlim=(-0.035, 1.035),
                ylim=(-0.34, 0.34),
                xticks=(0, 0.5, 1),
                yticks=(-0.25, 0, 0.25),
                xticklabels=("0", "0.5", "1"),
                yticklabels=("−0.25", "0", "0.25"),
            )
            ax.tick_params(axis="both", labelsize=5.15, length=1.25, pad=0.55)
            ax.tick_params(axis="x", labelbottom=index == 2)
            ax.tick_params(axis="y", labelleft=False)
            if index == 2:
                label(fig, 22.7, centre + 3.15, r"$x$", ha="center", fontsize=5.6)
        ax.set_facecolor("none")
        for point in fixed[fixed.model == model].itertuples():
            ax.plot(
                point.x,
                point.y,
                "o",
                ms=2.6,
                mec=MODEL_INK,
                mfc=MODEL_INK if point.marker == "stable" else "white",
                mew=DATA_LINE_WIDTH,
                zorder=4,
                clip_on=False,
            )

    key(
        fig,
        [
            Line2D(
                [],
                [],
                ls="none",
                marker="o",
                ms=2.6,
                mec=MODEL_INK,
                mfc=fill,
                mew=DATA_LINE_WIDTH,
                label=name,
            )
            for name, fill in (("Stable", MODEL_INK), ("Unstable", "white"))
        ],
        17.5,
        48.65,
        2,
    )
    label(fig, 17.5, 45.55, "Fixed points", ha="center", fontsize=5.4)


def model_accuracy(fig):
    panel(fig, "a", 1.6, 2.25)
    edges = 7.1 + np.r_[0, np.cumsum((8.0, 8.0, 8.0, 12.65))]
    for edge in edges[1:-1]:
        y = 1 - edge / HEIGHT
        fig.add_artist(
            Line2D(
                [2.1 / WIDTH, 87.25 / WIDTH],
                [y, y],
                transform=fig.transFigure,
                color="#D9DEE1",
                lw=0.45,
                zorder=0,
            )
        )
    profiles(fig, edges)
    key(
        fig,
        [
            Line2D([], [], color=color, lw=DATA_LINE_WIDTH, label=name)
            for _, name, color in LEGEND_CASES
        ],
        60.8,
        4.15,
        3,
        size=5.35,
    )

    samples = pd.read_csv(DATA / "a_moments/sampling_errorbars.csv")
    errors = pd.read_csv(DATA / "a_moments/selected_moment_errors.csv").set_index("id")
    assert len(samples) == 12 and not samples.duplicated(["model", "input"]).any()
    np.testing.assert_allclose(
        samples.deterministic_relative_RMS,
        errors.loc[samples.id].moment_relative_RMS_error,
        rtol=1e-13,
        atol=0,
    )
    samples = samples.set_index(["model", "input"])
    ax = axis(fig, 34.4, edges[0], 52.85, edges[-1] - edges[0])
    ax.set_facecolor("none")
    ax.set(
        xscale="log",
        xlim=(8e-4, 3),
        ylim=(edges[-1], edges[0]),
        yticks=[],
        xticks=(0.001, 0.01, 0.1, 1),
        xticklabels=("0.001", "0.01", "0.1", "1"),
    )
    ax.axvline(1, color=INK, lw=0.55, ls=(0, (2, 2)), zorder=0)
    ax.xaxis.set_minor_locator(NullLocator())
    for i, model in enumerate(MODELS):
        for j, (case, _, color) in enumerate(CASES):
            row = samples.loc[model, case]
            middle, low, high = (
                100
                * row[
                    [
                        "sampled_relative_RMS_median",
                        "sampled_relative_RMS_lower",
                        "sampled_relative_RMS_upper",
                    ]
                ]
            )
            assert 0 < low <= middle <= high
            y = edges[i] + (edges[i + 1] - edges[i]) * (j + 0.5) / len(CASES)
            ax.errorbar(
                middle,
                y - 0.69,
                xerr=[[middle - low], [high - middle]],
                fmt="s",
                ms=MOMENT_MARKER_SIZE_PT,
                mfc="white",
                mec=color,
                mew=DATA_LINE_WIDTH,
                color=color,
                capsize=1.0,
                capthick=DATA_LINE_WIDTH,
                elinewidth=DATA_LINE_WIDTH,
            )
            ax.plot(
                100 * row.deterministic_relative_RMS,
                y + 0.69,
                "D",
                ms=DIAMOND_MARKER_SIZE_PT,
                mec=color,
                mfc="white",
                mew=DATA_LINE_WIDTH,
            )

    ax.set_xlabel("Relative moment RMS error (%)", labelpad=1.8)
    key(
        fig,
        [
            Line2D(
                [],
                [],
                color=INK,
                marker="D",
                mfc="white",
                ms=DIAMOND_MARKER_SIZE_PT,
                mew=DATA_LINE_WIDTH,
                ls="none",
                label="Shot-noise-free",
            ),
            Line2D(
                [],
                [],
                color=INK,
                marker="s",
                mfc="white",
                ms=MOMENT_MARKER_SIZE_PT,
                mew=DATA_LINE_WIDTH,
                lw=DATA_LINE_WIDTH,
                label="Sampled, 95% range",
            ),
        ],
        60.8,
        50.25,
        2,
    )


def density(fig):
    panel(fig, "b", 1.6, 54.0)
    data = pd.read_csv(DATA / "b_densities/scalar_conservative_cell_masses.csv")
    cases = (
        ("interior", r"Interior, $x_0=0.35$"),
        ("near_unstable", r"Unstable centre, $x_0=0.25$"),
    )
    legend = [
        Line2D([], [], color=GRAY, lw=3.1, label="Initial"),
        Line2D([], [], color=BLUE, lw=0.85, label="Circuit"),
        Line2D([], [], color=INK, lw=0.8, ls=DASHED, label="Characteristics"),
    ]
    for i, ((case, title), top) in enumerate(zip(cases, (62.4, 85.0))):
        label(fig, LEFT + TRACK / 2, top - 2.35, title, ha="center", fontsize=6.1)
        ax = axis(fig, LEFT, top, TRACK, 15.2)
        rows = data[data.id == f"quintic_multistable__{case}"].sort_values("cell")
        np.testing.assert_array_equal(rows.cell, np.arange(64))
        np.testing.assert_allclose(rows.right - rows.left, 1 / 64)
        edges = np.r_[rows.left.to_numpy(), rows.right.iloc[-1]]
        for root in (0.25, 0.75):
            ax.axvline(root, color="#BBC2C7", ls=(0, (1.2, 2)), lw=0.55, zorder=0)
        ax.stairs(
            64 * rows.initial_prepared_mass,
            edges,
            fill=True,
            color=GRAY,
            alpha=0.75,
            lw=0.5,
            zorder=1,
        )
        ax.stairs(64 * rows.final_circuit_mass, edges, color=BLUE, lw=0.85, zorder=2)
        ax.stairs(
            64 * rows.final_matched_characteristic_mass,
            edges,
            color=INK,
            ls=DASHED,
            lw=0.75,
            zorder=3,
        )
        for root in (0, 0.25, 0.5, 0.75, 1):
            stable = root in (0, 0.5, 1)
            color = STABLE_BLUE if stable else UNSTABLE_RED
            ax.plot(
                root,
                0,
                "o",
                mfc=color if stable else "white",
                mec=color,
                ms=2.6,
                mew=DATA_LINE_WIDTH,
                zorder=4,
                clip_on=False,
            )
        ax.set(
            xlim=(-0.025, 1.025),
            ylim=(0, 24),
            xticks=(0, 0.25, 0.5, 0.75, 1),
            xticklabels=("0", "0.25", "0.5", "0.75", "1"),
            yticks=(0, 10, 20),
        )
        ax.tick_params(labelbottom=i == 1)
        if i == 0:
            ax.legend(
                handles=legend,
                loc="upper right",
                fontsize=5.1,
                facecolor="white",
                edgecolor="none",
                framealpha=1,
                handlelength=1.25,
                handletextpad=0.4,
                labelspacing=0.28,
                borderpad=0.15,
            )
        else:
            ax.set_xlabel(r"$x$", labelpad=1.2)
    label(fig, 2.7, 81.3, "Density", rotation=90, ha="center")


def resource_xaxis(ax, last):
    ax.set(xlim=(1.6, 9.4), xticks=(2, 5, 9))
    ax.tick_params(labelbottom=last)
    if last:
        ax.set_xlabel(r"$p_{\rm phys}$ ($10^{-4}$)", labelpad=1.45)


def coupled_resources(fig):
    panel(fig, "c", 45.0, 54.0)
    label(fig, RIGHT + TRACK / 2, 60.05, "2D C3, interior", ha="center", fontsize=6.1)
    all_rows = pd.read_csv(DATA / "c_coupled_resources/selected_resources.csv")
    rows = all_rows[all_rows.model == "2d_interior"]
    qubits = axis(fig, RIGHT, 62.4, TRACK, 17.1)
    runtime = axis(fig, RIGHT, 85.0, TRACK, 15.2)
    for selection, color, marker, line, name in (
        ("minimum_qubits", RED, "s", DASHED, "Min. qubits"),
        ("fastest_uncapped", DARK, "o", "-", "Fastest"),
    ):
        data = rows[rows.selection == selection].sort_values("p_phys")
        np.testing.assert_allclose(data.p_phys * 1e4, RATES, atol=1e-8, rtol=0)
        style = dict(
            color=color, marker=marker, ms=2.7, mfc="white", mew=0.65, lw=0.8, ls=line
        )
        qubits.plot(RATES, data.physical_qubits / 1e5, label=name, **style)
        runtime.plot(RATES, data.complete_task_hours, **style)
    qubits.legend(
        loc="upper left",
        fontsize=5.2,
        frameon=False,
        handlelength=1.35,
        handletextpad=0.4,
        labelspacing=0.3,
        borderpad=0.15,
    )
    qubits.set(ylim=(0, 10), yticks=(0, 5, 10), ylabel=r"Qubits ($10^5$)")
    runtime.set(
        yscale="log",
        ylim=(5, 180),
        yticks=(10, 30, 100),
        yticklabels=("10", "30", "100"),
        ylabel="Total task time (h)",
    )
    runtime.yaxis.set_minor_locator(NullLocator())
    for i, ax in enumerate((qubits, runtime)):
        ax.yaxis.labelpad = 1.1
        resource_xaxis(ax, last=i == 1)


def wake(fig):
    panel(fig, "d", 1.6, 109.6)
    panel(fig, "e", 45.0, 109.6)
    times, reference, runs = load_runs(FIGURE / "runs.json")
    names = ("29.8 start", "40 start")
    markers = [
        Line2D(
            [],
            [],
            marker=run[4],
            color=run[3],
            mfc="white",
            mew=0.65,
            ms=2.8,
            ls="none",
            label=name,
        )
        for run, name in zip(runs, names)
    ]
    markers.append(Line2D([], [], color=INK, ls=DASHED, lw=0.75, label="ODE refs."))
    key(fig, markers, 25.9, 113.35, 3, size=5.2)

    for i, (name, top) in enumerate(zip("xyz", (117.0, 131.6, 146.2))):
        ax = axis(fig, LEFT, top, TRACK, 12.45)
        for run in runs:
            ref = pd.read_csv(run[6].parent / "trajectory.csv")
            ax.plot(
                ref.time, ref["reference_mean_" + name], color=INK, lw=0.72, ls=DASHED
            )
        for time, means, _, color, marker, _, _ in runs:
            take = np.unique(
                np.linspace(0, len(time) - 1, min(17, len(time))).round().astype(int)
            )
            ax.plot(
                time[take],
                means[i, take],
                color=color,
                marker=marker,
                ms=2.55,
                mfc="white",
                mew=0.65,
                ls="none",
            )
        time_axis(ax, times)
        if name in "xy":
            ax.set(ylim=(-1.08, 1.08), yticks=(-1, 0, 1))
        else:
            ax.set(
                ylim=(0.44, 1.045),
                yticks=(0.5, 0.75, 1),
                yticklabels=("0.50", "0.75", "1.00"),
            )
            ax.set_xlabel(r"$tU_\infty/D$", labelpad=1.3)
        ax.tick_params(labelbottom=i == 2)
        ax.set_ylabel(r"$\langle " + name + r"\rangle$", labelpad=1.35)
        ax.yaxis.set_label_coords(-0.215, 0.5)

    selections = (
        ("minimum_qubits", DASHED, "Min. qubits"),
        ("fastest_uncapped", "-", "Fastest"),
    )
    key(
        fig,
        [
            Line2D(
                [],
                [],
                color=INK,
                ls=line,
                lw=0.8,
                marker="o",
                ms=2.8,
                mfc="white" if i == 0 else INK,
                label=name,
            )
            for i, (_, line, name) in enumerate(selections)
        ],
        71.55,
        113.35,
        2,
    )
    qubits = axis(fig, RIGHT, 117.0, TRACK, 17.35)
    runtime = axis(fig, RIGHT, 141.0, TRACK, 17.65)
    for _, _, _, color, marker, _, path in runs:
        for i, (selection, line, _) in enumerate(selections):
            style = dict(
                color=color,
                marker=marker,
                ms=2.7,
                mfc="white" if i == 0 else color,
                mec=color,
                mew=0.65,
                lw=0.8,
                ls=line,
            )
            qubits.plot(
                RATES, read_resources(path, selection, "physical_qubits") / 1e5, **style
            )
            runtime.plot(
                RATES, read_resources(path, selection, "complete_task_days"), **style
            )
    qubits.axhspan(10, 150, facecolor="#777777", alpha=0.15, edgecolor="none", zorder=0)
    qubits.axhline(10, color=INK, ls=(0, (2, 2)), lw=0.45, zorder=1)
    qubits.set(
        yscale="log",
        ylim=(0.5, 150),
        yticks=(1, 10, 100),
        yticklabels=("1", "10", "100"),
        ylabel=r"Qubits ($10^5$)",
    )
    runtime.set(
        yscale="log",
        ylim=(0.1, 800),
        yticks=(0.1, 1, 10, 100),
        ylabel="Total task time (d)",
    )
    for i, ax in enumerate((qubits, runtime)):
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.labelpad = 1.1
        resource_xaxis(ax, last=i == 1)


def main():
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else output_dir("F5")
    output.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(WIDTH / 25.4, HEIGHT / 25.4), facecolor="white", dpi=180)
    model_accuracy(fig)
    density(fig)
    coupled_resources(fig)
    wake(fig)
    fig.canvas.draw()
    fig.savefig(
        output / "Figure5_updated.pdf",
        metadata={
            "Title": "Figure 5 | Nonlinear benchmarks and three-coordinate wake simulation",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    fig.savefig(output / "Figure5_updated.svg", metadata={"Date": None})
    fig.savefig(output / "Figure5_updated.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
