"""Render one or all public figures from archived data.

Examples: python -m Figure_Plots.run F3; python -m Figure_Plots.run all
Numerical regeneration is provided separately by Source.Data_Production.end_to_end.
"""

from __future__ import annotations

# Direct-file execution must locate Codes/ independently of the working directory.
if __package__ in (None, ""):
    import sys as _path_sys
    from pathlib import Path as _Path
    _codes = next(p for p in _Path(__file__).resolve().parents
                  if (p / "Source/data_locations.py").is_file())
    _path_sys.path.insert(0, str(_codes))


import argparse
import json
import os
import subprocess
import sys
import time

from .common.paths import (
    AVAILABLE,
    GROUPS,
    REPOSITORY,
    canonical_name,
    figure_root,
    output_dir,
)
from Source.data_locations import generated_outputs_root, source_label


def run(name: str, *, timeout: float = 120) -> dict:
    name = canonical_name(name)
    script = figure_root(name) / "code" / "plot.py"
    if not script.is_file():
        raise FileNotFoundError(script)
    output = output_dir(name)
    command = [sys.executable, str(script)]
    if name == "F4":
        command.extend(["--package-root", str(REPOSITORY)])
    elif name != "F1":
        command.append(str(output))
    if name == "F3":
        command.append("--clip-to-window")
    start = time.perf_counter()
    env=dict(os.environ, PYTHONPATH=str(REPOSITORY)+os.pathsep+os.environ.get('PYTHONPATH',''))
    subprocess.run(command, cwd=REPOSITORY, env=env, check=True, timeout=timeout)
    # Other figures set their final typography within their own plotter.
    if name == "F1":
        from .common.ncs_style import normalize_folder

        normalize_folder(output, name)
    files = sorted(p.name for p in output.iterdir() if p.is_file())
    if not any(p.endswith(".pdf") for p in files) or not any(
        p.endswith(".png") for p in files
    ):
        raise RuntimeError(f"Expected PDF and PNG outputs in {output}")
    return {
        "figure": name,
        "status": "PASS",
        "files": files,
        "output": source_label(output),
        "elapsed_seconds": round(time.perf_counter() - start, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("figure", help="F1, F3–F5, S01/S03–S05/S08–S10, ED1, or all")
    parser.add_argument(
        "--group", choices=tuple(GROUPS), help="Filter an all-figures run"
    )
    parser.add_argument(
        "--timeout", type=float, default=120, help="Seconds allowed per plotter"
    )
    args = parser.parse_args()
    if args.group and args.figure.lower() != "all":
        parser.error("--group is used with all")
    try:
        names = (
            (GROUPS[args.group] if args.group else AVAILABLE)
            if args.figure.lower() == "all"
            else (canonical_name(args.figure),)
        )
    except ValueError as exc:
        parser.error(str(exc))
    results = []
    for name in names:
        try:
            result = run(name, timeout=args.timeout)
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
            result = {"figure": name, "status": "FAIL", "error": str(exc)}
        results.append(result)
        print(name, result["status"], flush=True)
    report = generated_outputs_root() / "Figure_Reproduction_Results.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(results, indent=2) + "\n")
    print("Results:", report)
    return int(any(row["status"] != "PASS" for row in results))


if __name__ == "__main__":
    raise SystemExit(main())
