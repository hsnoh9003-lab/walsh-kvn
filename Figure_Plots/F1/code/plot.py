"""Export the current approved SVG master (artwork-only replay)."""

import sys as _figure_sys
from pathlib import Path as _FigurePath

_figure_sys.path.insert(0, str(_FigurePath(__file__).resolve().parents[3]))
from Figure_Plots.common.paths import output_dir
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

from Figure_Plots.common.render_figure_master import run

if __name__ == "__main__":
    run(ROOT)
