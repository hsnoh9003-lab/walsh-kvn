"""Distance and error sweeps from the supplied single-column factory figure."""

import numpy as np
import matplotlib.pyplot as plt

BLUE = "#2A6EA6"
RED = "#C65B3D"
INK = "#111111"
GRID = "#d9d9d9"
FILL_BLUE = "#8BB6D8"
FILL_RED = "#E7B5A8"
STAR = "#FFC000"
STAR_EDGE = "#000000"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "mathtext.fontset": "stix",
        "font.size": 5.0,
        "axes.labelsize": 6.0,
        "xtick.labelsize": 5.0,
        "ytick.labelsize": 5.0,
        "legend.fontsize": 5.0,
        "axes.linewidth": 0.55,
        "xtick.major.width": 0.55,
        "ytick.major.width": 0.55,
        "xtick.major.size": 2.1,
        "ytick.major.size": 2.1,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "path",
        "svg.hashsalt": "kvn-s11-original",
        "axes.unicode_minus": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "xtick.direction": "in",
        "ytick.direction": "in",
    }
)


def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="in", length=2.1, width=0.55, colors=INK, pad=1.3)
    ax.yaxis.grid(True, color=GRID, linewidth=0.4, alpha=0.65)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(INK)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def shade_vs_supply_optimum(ax, xx, supply, one, ref):
    """Shade only the regions where the two policies differ.

    Direction is determined by ONE relative to the global supply-enforced
    optimum, not by ONE relative to the supply policy at that distance.
    Equal-policy segments, including all distances beyond the optimum where
    both traces coincide, remain unshaded regardless of their height.
    """
    for i in range(len(xx) - 1):
        s0, s1 = supply[i : i + 2]
        o0, o1 = one[i : i + 2]
        if not np.isfinite([s0, s1, o0, o1]).all():
            continue
        # If both policies coincide at both endpoints, there is no comparison.
        eps = 1e-10 * max(1.0, abs(s0), abs(s1), abs(o0), abs(o1))
        if abs(s0 - o0) <= eps and abs(s1 - o1) <= eps:
            continue
        # Piecewise-linear ONE curve, with a horizontal supply-optimum line.
        # Split at crossings so better and worse never mix in one polygon.
        xs = [float(xx[i]), float(xx[i + 1])]
        yy = [float(o0), float(o1)]
        v0, v1 = o0 - ref, o1 - ref
        if v0 * v1 < 0:
            f = (ref - o0) / (o1 - o0)
            xs = [
                float(xx[i]),
                float(xx[i] + f * (xx[i + 1] - xx[i])),
                float(xx[i + 1]),
            ]
            yy = [float(o0), float(ref), float(o1)]
        for j in range(len(xs) - 1):
            ym = 0.5 * (yy[j] + yy[j + 1])
            if abs(ym - ref) <= eps:
                continue
            color = FILL_BLUE if ym < ref else FILL_RED
            ax.fill_between(
                xs[j : j + 2],
                yy[j : j + 2],
                ref,
                facecolor=color,
                alpha=0.31,
                linewidth=0,
                zorder=1,
            )


def draw_policy(ax, subset, summary, mode, panel_letter=None, p_note=None):
    style(ax)
    xx = subset.d_core.to_numpy(dtype=float)
    unit_counts = subset.strict_u.to_numpy(dtype=int)
    if mode == "qubits":
        supply = subset.strict_q.to_numpy(dtype=float)
        one = subset.one_q.to_numpy(dtype=float)
        ref = float(summary.strict_opt_Q)
        ylabel = "Physical qubits"
        # Unambiguous scientific axis factor at the axis end.
        ax.ticklabel_format(axis="y", style="sci", scilimits=(4, 4), useMathText=True)
        ax.yaxis.get_offset_text().set_fontsize(5)
    else:
        supply = subset.strict_t.to_numpy(dtype=float)
        one = subset.one_t.to_numpy(dtype=float)
        ref = float(summary.strict_opt_time)
        ylabel = "Time per attempt (s)"

    shade_vs_supply_optimum(ax, xx, supply, one, ref)
    valid_one = np.isfinite(one)
    distinct = valid_one & ~np.isclose(one, supply, rtol=1e-10, atol=1e-10)

    # One-MSF line is behind the supply-enforced line everywhere. At equal
    # points, omit the blue square; a wider red line hides any overlap.
    # Draw ONE only on segments containing genuinely different policy values.
    # This prevents underlying blue from peeking out on coincident intervals.
    for idx in range(len(xx) - 1):
        if not (valid_one[idx] and valid_one[idx + 1]):
            continue
        if not (distinct[idx] or distinct[idx + 1]):
            continue
        ax.plot(xx[idx : idx + 2], one[idx : idx + 2], lw=0.90, color=BLUE, zorder=3)

    if distinct.any():
        ax.scatter(
            xx[distinct],
            one[distinct],
            marker="s",
            s=10.0,
            facecolor=BLUE,
            edgecolor=BLUE,
            linewidth=0.4,
            zorder=4,
        )

    valid_supply = np.isfinite(supply)
    ax.plot(xx[valid_supply], supply[valid_supply], lw=1.18, color=RED, zorder=6)
    normal = valid_supply & (unit_counts <= 1)
    more_units = valid_supply & (unit_counts > 1)
    ax.scatter(
        xx[normal],
        supply[normal],
        marker="o",
        s=11.2,
        facecolor=RED,
        edgecolor=RED,
        linewidth=0.3,
        zorder=7,
    )
    ax.scatter(
        xx[more_units],
        supply[more_units],
        marker="D",
        s=12.5,
        facecolor="white",
        edgecolor=RED,
        linewidth=0.8,
        zorder=7,
    )

    # The optimum horizontal guide is not part of the shaded difference.
    ax.axhline(ref, ls=":", color=RED, lw=0.75, zorder=2)
    ax.scatter(
        [summary.strict_opt_d],
        [ref],
        marker="*",
        s=38,
        facecolor=STAR,
        edgecolor=STAR_EDGE,
        linewidth=0.75,
        zorder=9,
    )

    ax.set_xlim(xx.min() - 0.75, xx.max() + 0.75)
    ax.set_xticks(xx.astype(int))
    if len(xx) > 6:  # Avoid excessive ticks at 5 pt in the half-width panels.
        ax.set_xticks(xx.astype(int)[::2])
    ax.set_xlabel(r"Core code distance, $d_{\mathrm{core}}$", labelpad=1.9)
    ax.set_ylabel(ylabel, labelpad=2.0)

    vals = np.r_[supply[valid_supply], one[valid_one], [ref]]
    low, high = float(np.min(vals)), float(np.max(vals))
    pad = 0.10 * (high - low) if high > low else 1.0
    ax.set_ylim(low - pad, high + pad)

    if panel_letter is not None:
        ax.text(
            -0.08,
            1.13,
            panel_letter,
            transform=ax.transAxes,
            fontsize=7,
            fontweight="bold",
            ha="left",
            va="bottom",
            color=INK,
        )
    if p_note is not None:
        ax.text(
            0.99,
            1.035,
            p_note,
            transform=ax.transAxes,
            fontsize=5,
            ha="right",
            va="bottom",
            color=INK,
        )


def draw_error_bars(ax, opt, letter):
    style(ax)
    xx = opt.p_phys.to_numpy(dtype=float)
    q = opt.physical_qubit_change_pct.to_numpy(dtype=float)
    t = opt.runtime_change_pct.to_numpy(dtype=float)
    width = 0.345e-4  # 1.5 times the previous width (0.23e-4)
    ax.bar(xx - width / 2, q, width, color=BLUE, zorder=3)
    ax.bar(xx + width / 2, t, width, color=RED, zorder=3)
    ax.axhline(0, color=INK, linewidth=0.65)
    ax.set_ylabel("Relative change (%)", labelpad=2)
    ax.set_xlabel(r"Physical error, $p_{\mathrm{phys}}$", labelpad=2)
    ax.set_xlim(1.5e-4, 9.5e-4)
    ax.set_ylim(min(q.min(), t.min()) - 6, max(q.max(), t.max()) + 6)
    ax.set_xticks(np.arange(2, 10) * 1e-4)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(-4, -4), useMathText=True)
    ax.xaxis.get_offset_text().set_fontsize(5)
    ax.text(
        -0.035,
        1.08,
        letter,
        transform=ax.transAxes,
        fontsize=7,
        fontweight="bold",
        ha="left",
        va="bottom",
        color=INK,
    )
