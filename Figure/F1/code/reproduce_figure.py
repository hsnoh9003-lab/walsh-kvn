"""Export the current approved SVG master (artwork-only replay)."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parents[1]/'T'))
from render_figure_master import run
if __name__=='__main__':run(ROOT)
