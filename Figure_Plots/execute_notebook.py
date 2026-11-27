"""Execute notebook code cells sequentially without launching a Jupyter server.

Inputs
------
A notebook path and the environment/data required by its cells.

Outputs
-------
An executed notebook at the required --output path, with captured text logs.

Run from the repository root
----------------------------
python -m Figure_Plots.execute_notebook Reproduce_Figures.ipynb --output ../Source_Data/Generated_Outputs/Figures_executed.ipynb

Scope
-----
Execution uses the same plotting and scientific functions as the notebook; the output notebook records text streams.
"""

from __future__ import annotations

import argparse
from contextlib import chdir, redirect_stdout, redirect_stderr
import io
from pathlib import Path
import os


def main() -> None:
    """Execute code cells in one namespace and save captured stdout/stderr.

    This runner records text streams, not rich display/image outputs. A cell
    error stops execution before the destination notebook is written. The
    destination is saved only after every code cell succeeds.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    os.environ.setdefault("MPLBACKEND", "Agg")
    import nbformat
    from IPython.core.interactiveshell import InteractiveShell

    notebook_path = args.notebook.resolve()
    output_path = args.output.resolve()
    book = nbformat.read(notebook_path, as_version=4)

    class HeadlessShell(InteractiveShell):
        """Support inline plots without installing a GUI event loop."""

        def enable_gui(self, gui=None):
            if gui not in (None, "inline", "agg"):
                raise ValueError(f"Headless execution cannot start the {gui!r} GUI")

    shell = HeadlessShell.instance()
    executed = 0
    with chdir(notebook_path.parent):
        for index, cell in enumerate(book.cells, start=1):
            if cell.cell_type != "code" or not cell.source.strip():
                continue
            stdout = io.StringIO()
            stderr = io.StringIO()
            print(f"Executing notebook cell {index}/{len(book.cells)}", flush=True)
            with redirect_stdout(stdout), redirect_stderr(stderr):
                result = shell.run_cell(cell.source, store_history=False)
            cell.execution_count = executed + 1
            cell.outputs = []
            for name, output in [
                ("stdout", stdout.getvalue()),
                ("stderr", stderr.getvalue()),
            ]:
                if output:
                    cell.outputs.append(
                        nbformat.v4.new_output("stream", name=name, text=output)
                    )
                    print(output[-1800:], flush=True)
            if result.error_before_exec or result.error_in_exec:
                raise RuntimeError(f"{args.notebook}: cell {index} failed") from (
                    result.error_before_exec or result.error_in_exec
                )
            executed += 1
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(book, output_path)
    print(f"Notebook completed: {executed} executed cells; {output_path}", flush=True)


if __name__ == "__main__":
    main()
