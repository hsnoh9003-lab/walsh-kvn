from pathlib import Path
import sys
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root.parents[1]/'T'))
from render_figure_master import run
if __name__=='__main__': run(root)
