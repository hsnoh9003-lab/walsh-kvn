"""Run the Navier--Stokes benchmark with shared published or newly generated inputs."""

# Direct-file execution must locate Codes/ independently of the working directory.
if __package__ in (None, ""):
    import sys as _path_sys
    from pathlib import Path as _Path
    _codes = next(p for p in _Path(__file__).resolve().parents
                  if (p / "Source/data_locations.py").is_file())
    _path_sys.path.insert(0, str(_codes))


import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[2]))
from Source.CFD.inputs import steady_input
from Source.data_locations import set_source_data_root
from Source.data_locations import source_data_root
from Source.data_locations import generated_outputs_root


def validate_run_settings(dt, end, reynolds):
    """Require finite positive settings and a whole number of fixed time steps."""
    for name, value in (("dt", dt), ("end", end), ("reynolds", reynolds)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"--{name} must be a finite number greater than zero.")
    if end < dt:
        raise ValueError("--end must be at least --dt to run one full time step.")
    steps = end / dt
    if not math.isfinite(steps) or not math.isclose(
        steps, round(steps), rel_tol=0.0, abs_tol=1e-8
    ):
        raise ValueError(
            "--end must be an integer multiple of --dt for fixed-step integration."
        )


def main():
    """Validate the requested batch, then launch the steady and transient solvers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reynolds", type=int, nargs="+", default=[100])
    parser.add_argument(
        "--dt",
        type=float,
        default=None,
        help="Override the recorded stable integration step",
    )
    parser.add_argument("--end", type=float, default=90.0)
    parser.add_argument(
        "--reference-replay",
        action="store_true",
        help="Use the published Re100 mesh/base input",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Source_Data directory (default: local copy, KVN_SOURCE_DATA or the Zenodo DOI cache)",
    )
    args = parser.parse_args()
    # Validate the whole batch before resolving data or launching a steady solve.
    if any(reynolds <= 0 for reynolds in args.reynolds):
        parser.error("--reynolds must be greater than zero.")
    try:
        if args.reference_replay:
            runs = [(100, 0.02 if args.dt is None else args.dt)]
        else:
            dataset = Path(
                args.data_root or os.environ.get("KVN_SOURCE_DATA", ROOT.parents[3] / "Source_Data")
            )
            settings_path = dataset / "Additional_Source_Data/CFD_Records/fem/verified_timesteps.json"
            steps = (
                json.loads(settings_path.read_text())
                if settings_path.exists()
                else {"100": 0.02, "200": 0.01, "300": 0.01}
            )
            runs = []
            for reynolds in args.reynolds:
                dt = args.dt if args.dt is not None else steps.get(str(reynolds))
                if dt is None:
                    raise ValueError(
                        f"No recorded time step for Re={reynolds}; supply --dt explicitly."
                    )
                runs.append((reynolds, dt))
        for reynolds, dt in runs:
            validate_run_settings(dt, args.end, reynolds)
    except (TypeError, ValueError) as error:
        parser.error(str(error))

    if args.data_root is not None:
        try:
            set_source_data_root(args.data_root)
        except FileNotFoundError as error:
            parser.error(str(error))
    if args.reference_replay:
        try:
            initial = steady_input()
        except (FileNotFoundError, KeyError) as error:
            parser.error(
                f"{error}\nCheck the published DOI dataset or pass --data-root /path/to/Source_Data."
            )
        # The solver receives its data path explicitly; no duplicate input tree is made.
        subprocess.run(
            [
                sys.executable,
                str(ROOT.parent / "replay_fem/cfd_benchmark.py"),
                "--input",
                str(initial),
                "--mesh",
                "coarse",
                "--dt",
                str(runs[0][1]),
                "--end",
                str(args.end),
                "--case",
                "train",
                "--output-root",
                str(generated_outputs_root() / "CFD/fem/reference_replay"),
            ],
            check=True,
        )
        return

    for reynolds, dt in runs:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "steady.py"),
                "--name",
                f"Re{reynolds}_coarse",
                "--reynolds",
                str(reynolds),
                "--xmin",
                "-10",
                "--xmax",
                "20",
                "--height",
                "10",
                "--mesh-factor",
                "1",
            ],
            check=True,
        )
        # Generated fields are run outputs, not additions to the published dataset.
        initial = generated_outputs_root() / "CFD/fem/results" / f"Re{reynolds}_coarse/finite_element_solution.npz"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "simulate.py"),
                "--reynolds",
                str(reynolds),
                "--input",
                str(initial),
                "--mesh",
                "coarse",
                "--dt",
                str(dt),
                "--end",
                str(args.end),
                "--case",
                "train",
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
