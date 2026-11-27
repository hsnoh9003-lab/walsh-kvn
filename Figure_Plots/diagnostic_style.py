"""Isolated, readable styling for diagnostic plots (not publication artwork)."""

from contextlib import contextmanager

import matplotlib as mpl
from cycler import cycler

COLORS = ("#263c50", "#d87923", "#23857d")
TIME_LABEL = r"$t^* = tU_\infty/D$"


@contextmanager
def diagnostic_style():
    """Use a consistent style and restore the caller's Matplotlib settings."""
    with mpl.rc_context():
        mpl.rcdefaults()
        mpl.rcParams.update(
            {
                "font.family": "serif",
                "font.serif": ["DejaVu Serif"],
                "mathtext.fontset": "stix",
                "font.size": 10,
                "axes.titlesize": 10,
                "axes.labelsize": 10,
                "legend.fontsize": 9,
                "axes.prop_cycle": cycler(color=COLORS),
                "axes.linewidth": 0.7,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.grid": True,
                "axes.axisbelow": True,
                "grid.alpha": 0.18,
                "grid.linewidth": 0.6,
                "xtick.direction": "in",
                "ytick.direction": "in",
                "lines.linewidth": 1.4,
                "lines.markersize": 4,
                "figure.figsize": (8, 3.2),
                "figure.dpi": 120,
                "figure.facecolor": "white",
                "axes.facecolor": "white",
                "savefig.facecolor": "white",
                "savefig.transparent": False,
                "savefig.dpi": 200,
                "savefig.bbox": "tight",
                "legend.frameon": False,
                "svg.fonttype": "none",
                "pdf.fonttype": 42,
            }
        )
        yield
