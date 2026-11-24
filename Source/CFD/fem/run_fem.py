"""Run the Navier--Stokes benchmark with shared published or newly generated inputs."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[2]))
from Source.CFD.inputs import steady_input
from Source.data_locations import set_source_data_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reynolds', type=int, nargs='+', default=[100])
    parser.add_argument('--dt', type=float, default=None, help='Override the recorded stable integration step')
    parser.add_argument('--end', type=float, default=90.)
    parser.add_argument('--reference-replay', action='store_true', help='Use the published Re100 mesh/base input')
    parser.add_argument(
        '--data-root', type=Path,
        help='Source_Data directory (default: local copy, KVN_SOURCE_DATA or the Zenodo DOI cache)',
    )
    args = parser.parse_args()
    if args.reference_replay:
        try:
            if args.data_root is not None:
                set_source_data_root(args.data_root)
            initial = steady_input()
        except (FileNotFoundError, KeyError) as error:
            parser.error(f'{error}\nCheck the published DOI dataset or pass --data-root /path/to/Source_Data.')
        # The solver receives its data path explicitly; no duplicate input tree is made.
        subprocess.run([
            sys.executable, str(ROOT.parent/'replay_fem/cfd_benchmark.py'),
            '--input', str(initial), '--mesh', 'coarse', '--dt', str(args.dt or .02),
            '--end', str(args.end), '--case', 'train',
            '--output-root', str(ROOT/'runs/reference_replay/cfd'),
        ], check=True)
        return

    settings_path = ROOT/'verified_timesteps.json'
    steps = json.loads(settings_path.read_text()) if settings_path.exists() else {'100': .02, '200': .01, '300': .01}
    for reynolds in args.reynolds:
        subprocess.run([
            sys.executable, str(ROOT/'steady.py'), '--name', f'Re{reynolds}_coarse',
            '--reynolds', str(reynolds), '--xmin', '-10', '--xmax', '20',
            '--height', '10', '--mesh-factor', '1',
        ], check=True)
        # Generated fields are run outputs, not additions to the published dataset.
        initial = ROOT/'results'/f'Re{reynolds}_coarse/finite_element_solution.npz'
        dt = args.dt if args.dt is not None else steps[str(reynolds)]
        subprocess.run([
            sys.executable, str(ROOT/'simulate.py'), '--reynolds', str(reynolds),
            '--input', str(initial), '--mesh', 'coarse', '--dt', str(dt),
            '--end', str(args.end), '--case', 'train',
        ], check=True)


if __name__ == '__main__':
    main()
