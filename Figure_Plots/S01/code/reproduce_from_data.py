"""Render the archived forecast data with the main figure entry point."""

from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("plot.py")), run_name="__main__")
