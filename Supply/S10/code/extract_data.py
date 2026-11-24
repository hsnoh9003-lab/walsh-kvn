"""Export DOI-hosted Supplementary Figure S10 source files into Build/."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parents[1]/'T'))
from export_published_figure_inputs import export
if __name__=='__main__':export(ROOT,Path(sys.argv[1]) if len(sys.argv)>1 else None)
