"""Compare full-range and clipped views of the same Fig. 3b data."""


from pathlib import Path
import sys
import pandas as pd
from matplotlib.lines import Line2D
import base

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1]))
from Source.data_locations import generated_outputs_root
from Source.data_locations import figure_data

out = (
    Path(sys.argv[1])
    if len(sys.argv) > 1
    else generated_outputs_root() / "Figures/F3/variants"
)
out.mkdir(parents=True, exist_ok=True)
data = pd.read_csv(figure_data("F3") / "panel_b_selected_designs.csv")
base.style()
fig = base.plt.figure(figsize=(7, 3.2), facecolor="white")
for j, (limit, title) in enumerate(
    [((1e4, 1e7), "Full range"), ((1e4, 1e6), "Clipped at one million qubits")]
):
    base.PANEL_B_QUBIT_WINDOW = limit
    ax, _ = base.panel_b(fig, data)
    ax.set_position([0.085 + j * 0.49, 0.29, 0.36, 0.60])
    ax.set_title(title, fontsize=9, pad=15)
    ax.set_xlabel("Physical qubits", fontsize=9)
    ax.set_ylabel("Physical error" if j == 0 else "", fontsize=9)
    ax.tick_params(labelsize=8)
    for text in ax.texts:
        text.set_fontsize(8)
legend = [
    Line2D(
        [],
        [],
        marker=base.MARKERS[m],
        color=base.COLORS[m],
        ms=4,
        lw=0.8,
        ls="--" if m == "nj_lchs" else "-",
        label=base.LABELS[m],
    )
    for m in base.ORDER
]
fig.legend(
    handles=legend,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.005),
    ncol=3,
    frameon=False,
    fontsize=8,
    handlelength=2,
    columnspacing=2,
)
fig.text(0.5, 0.13, "Open: minimum qubits     Filled: fastest", ha="center", fontsize=8)
for suffix in ["png", "pdf", "svg"]:
    fig.savefig(
        out / f"Figure3b_comparison.{suffix}",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.08,
    )
base.plt.close(fig)
