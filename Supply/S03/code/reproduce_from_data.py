from pathlib import Path
import sys
from reproduce_figure import FIGURE, plot

if __name__ == "__main__":
    plot(Path(sys.argv[1]) if len(sys.argv)>1 else FIGURE/"reproduced")
