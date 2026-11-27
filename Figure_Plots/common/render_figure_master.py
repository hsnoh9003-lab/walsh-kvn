"""Export the packaged figure master without changing its scientific values.

This exports only the current approved master artwork. Historical source
versions were removed. The command does not rerun numerical simulations.
Source_Data records unavailable inputs and artwork paths.
"""

# Direct-file execution must locate Codes/ independently of the working directory.
if __package__ in (None, ""):
    import sys as _path_sys
    from pathlib import Path as _Path
    _codes = next(p for p in _Path(__file__).resolve().parents
                  if (p / "Source/data_locations.py").is_file())
    _path_sys.path.insert(0, str(_codes))


from pathlib import Path
from shutil import copy2, which
import argparse, json, subprocess
import fitz
from Source.data_locations import generated_outputs_root, source_data_root


def run(root, output=None):
    root = Path(root).resolve()
    out = (
        Path(output).resolve()
        if output
        else generated_outputs_root() / "Figures" / root.name
    )
    out.mkdir(parents=True, exist_ok=True)
    status = json.loads((source_data_root() / "Additional_Source_Data/Figure_Artwork_Status" / (root.name + ".json")).read_text())
    source = root / status["master"]
    name = status["figure"]
    target = out / (name + ".pdf")
    if not source.is_file():
        # Current approved editable artwork is released with each Figure's Source Data.
        import sys

        sys.path.insert(0, str(root.parents[1]))

        fig_key = (
            ("Fig_S" + str(int(root.name[1:])))
            if root.name.startswith("S")
            else ("Fig_" + root.name[1:])
        )
        source = source_data_root() / "Figures" / fig_key / "editable_figure.svg"
        if not source.is_file():
            raise FileNotFoundError(source)
    if source.suffix.lower() == ".svg":
        exe = which("inkscape")
        if exe is None:
            raise RuntimeError("Inkscape is required to export the editable SVG.")
        command = [
            exe,
            str(source),
            "--export-area-page",
            "--export-filename=" + str(target),
        ]
        if status.get("pdf_text_as_paths", True):
            command.append("--export-text-to-path")
        subprocess.run(command, check=True)
    else:
        copy2(source, target)
    with fitz.open(target) as doc:
        dpi = status.get("raster_dpi", 300)
        doc[0].get_pixmap(dpi=dpi, alpha=False).save(out / (name + ".png"))
    print(target)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("figure_root", type=Path)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    run(a.figure_root, a.output)
