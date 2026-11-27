"""Compare the published t<=90 FEM reference with four refinement runs.

Usage: python -m Source.CFD.refinement
All generated CFD data go to Source_Data/Generated/CFD_Refinement by default.
The published reference and Zenodo downloader are read-only inputs.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'

from Source.data_locations import source_data_root, source_file
from Source.CFD.inputs import steady_input
from Source.CFD.refinement_analysis import METRICS, late_window_statistics, relative_changes
from Source.CFD.refinement_metrics import write_json

ROOT = Path(__file__).resolve().parents[2]
REFERENCE_KEY = 'Cylinder_Forces_Re100_20261110/archived_replay/forces.csv'
PLAN = [('dt_0p01', 'time', 1., .01), ('mesh_0p8', 'space', .8, .02),
        ('dt_0p005', 'time', 1., .005), ('mesh_0p64', 'space', .64, .02)]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def generated_data_root():
    return source_data_root() / 'Generated/CFD_Refinement'


def protocol():
    return dict(reynolds=100, tolerance=.005, tolerance_type='relative_to_finer',
                stationarity_tolerance=.001, max_end=90., comparison_window=[60., 90.],
                max_wall_hours=8., mesh_factors=[1., .8, .64],
                time_steps=[.02, .01, .005], spatial_sweep_dt=.02,
                temporal_sweep_mesh_factor=1., metrics=list(METRICS),
                geometry=[-10., 20., -10., 10.],
                note='Late-window sensitivity using the published reference. '
                     'Cycle stationarity is diagnostic, not an acceptance requirement. '
                     'No asymptotic periodic-convergence claim.')


def settings(factor):
    return dict(reynolds=100., inlet_speed=1., diameter=1., viscosity=.01,
                xmin=-10., xmax=20., height=10., mesh_factor=factor,
                quadrature_order=5, domain_status='Fixed reference FEM domain')


def study_provenance():
    sources = [Path(__file__), ROOT/'Source/CFD/refinement_solver.py',
               ROOT/'Source/CFD/refinement_analysis.py', ROOT/'Source/CFD/refinement_metrics.py',
               ROOT/'Source/CFD/fem/steady.py',
               ROOT/'Source/CFD/force_replay/cfd_force_replay.py',
               ROOT/'Source/CFD/force_replay/source/solve_steady.py']
    value = dict(protocol=protocol(),
                 sources={str(p.relative_to(ROOT)): digest(p) for p in sources},
                 reference_input_sha256=digest(steady_input()),
                 published_force_sha256=digest(source_file(REFERENCE_KEY)),
                 python=sys.version, platform=platform.platform(),
                 packages={p: importlib.metadata.version(p) for p in
                           ('numpy','scipy','scikit-fem','triangle','pandas')})
    value['signature'] = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    return value


def worker(args):
    from Source.CFD.refinement_solver import run
    from Source.CFD.fem.steady import solve_case
    if args.factor == 1.:
        initial = steady_input()
    else:
        initial = args.output / 'steady/finite_element_solution.npz'
        if not initial.exists():
            solve_case(settings(args.factor), initial.parent)
    # The diagnostic is first checked at the final time; no extension beyond t=90.
    run(f'h{args.factor:g}', args.dt, 90., 'train', args.output, initial,
        min_end=90., stationarity_tol=.001)


def result_row(name, axis, factor, dt, force_path, summary):
    import pandas as pd
    frame = pd.read_csv(force_path)
    if abs(float(frame.time.max())-90.) > 1e-9:
        raise ValueError(f'{name}: force history must end at t=90')
    metrics = late_window_statistics(frame)
    return dict(name=name, axis=axis, factor=factor, dt=dt,
                force_file=force_path.resolve().relative_to(source_data_root()).as_posix(),
                triangles=summary['triangles'], velocity_dofs=summary['velocity_dofs'],
                pressure_dofs=summary['pressure_dofs'], actual_end=90., metrics=metrics)


def reference_row():
    force_path = source_file(REFERENCE_KEY)
    configuration = json.loads((force_path.parent/'config.json').read_text())
    return result_row('reference', 'reference', 1., .02, force_path, configuration)


def finite_window_verdict(rows):
    """Accept only when both adjacent pairs and reference-to-finest meet 0.5%."""
    if len(rows) != 3:
        return dict(status='incomplete')
    if not all(math.isfinite(row['metrics'][key]) for row in rows for key in METRICS):
        raise ValueError('Refinement statistics must be finite for every run and quantity.')
    first = relative_changes(rows[0]['metrics'], rows[1]['metrics'])
    adjacent = relative_changes(rows[1]['metrics'], rows[2]['metrics'])
    baseline = relative_changes(rows[0]['metrics'], rows[2]['metrics'])
    within_tolerance = max(*first.values(), *adjacent.values(), *baseline.values()) <= .005
    return dict(status='within_tolerance' if within_tolerance else 'needs_further_refinement',
                first_pair_relative_change=first,
                last_pair_relative_change=adjacent,
                reference_to_finest_relative_change=baseline,
                reference_within_tolerance=max(baseline.values()) <= .005,
                periodic_stationarity_established=all(r['metrics']['stationary'] for r in rows))


def report(output, rows, state, current=None):
    def sweep(axis):
        return [r for r in rows if r['axis'] in ('reference', axis)]
    value = dict(state=state, current_case=current, protocol=protocol(), runs=rows,
                 spatial=finite_window_verdict(sweep('space')),
                 temporal=finite_window_verdict(sweep('time')))
    write_json(output/'report.json', value)
    return value


def main(args):
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    provenance = study_provenance()
    prior = output/'provenance.json'
    if prior.exists() and json.loads(prior.read_text())['signature'] != provenance['signature']:
        raise RuntimeError('Protocol/code/environment changed: use a new output directory within Source_Data')
    write_json(prior, provenance)
    rows = [reference_row()]
    deadline = time.monotonic() + protocol()['max_wall_hours']*3600
    for name, axis, factor, dt in PLAN:
        case = output/name
        case.mkdir(exist_ok=True)
        summary_path = case/'summary.json'
        if not summary_path.exists():
            if time.monotonic() >= deadline:
                report(output, rows, 'wall_time_limit')
                return
            if (case/'forces.csv').exists() or (case/'run.log').exists():
                archived = output/'interrupted'
                archived.mkdir(exist_ok=True)
                case.rename(archived/f'{name}_{time.time_ns()}')
                case.mkdir()
            report(output, rows, 'running', name)
            command = [sys.executable, '-u', '-m', 'Source.CFD.refinement', '--worker',
                       '--output', str(case), '--factor', str(factor), '--dt', str(dt)]
            print('START_CASE', name, 'mesh spacing factor', factor, 'dt', dt, flush=True)
            with (case/'run.log').open('w') as log:
                try:
                    subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   check=True, timeout=max(1., deadline-time.monotonic()))
                except subprocess.TimeoutExpired:
                    report(output, rows, 'wall_time_limit', name)
                    return
                except subprocess.CalledProcessError:
                    report(output, rows, 'failed', name)
                    raise
        summary = json.loads(summary_path.read_text())
        if not summary.get('completed') or summary['dt'] != dt or summary['actual_end'] != 90.:
            raise RuntimeError(f'Invalid completed result: {name}')
        row = result_row(name, axis, factor, dt, case/'forces.csv', summary)
        write_json(case/'cycle_metrics.json', row['metrics'])
        rows.append(row)
        report(output, rows, 'running')
        print('DONE_CASE', name, {k: row['metrics'][k] for k in METRICS}, flush=True)
    report(output, rows, 'completed')
    print('STUDY_COMPLETE: finite-window comparisons saved in Source_Data', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Output directory inside Source_Data')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--factor', type=float, choices=(1., .8, .64), default=1.)
    parser.add_argument('--dt', type=float, choices=(.02, .01, .005), default=.02)
    options = parser.parse_args()
    options.output = (options.output or generated_data_root()).expanduser().resolve()
    if not options.output.is_relative_to(source_data_root()):
        parser.error('--output must be inside Source_Data')
    if options.worker:
        worker(options)
    else:
        main(options)
