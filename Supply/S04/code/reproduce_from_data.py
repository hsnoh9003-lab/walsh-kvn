"""Render the archived scalar cell masses with the main figure entry point."""
from pathlib import Path
import runpy
if __name__=='__main__':
    runpy.run_path(str(Path(__file__).with_name('reproduce_figure.py')),run_name='__main__')
