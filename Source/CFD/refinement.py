"""Run a bounded, sequential Re100 FEM mesh/time-step sensitivity study.

Usage: python -m Source.CFD.refinement --output Build/CFD_Refinement
Run this command in tmux for overnight execution. Reports go to the output directory.
No raw dataset or archived replay implementation is changed.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'

from Source.CFD.inputs import steady_input
from Source.CFD.refinement_metrics import cycle_metrics, sweep_verdict, write_json

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protocol():
    """Fixed controls; spacing factor is not the 128x64 observation grid."""
    return dict(reynolds=100, tolerance=.01, tolerance_type='relative_to_finer',
                stationarity_tolerance=.001, min_end=90., max_end=150.,
                max_wall_hours=8., mesh_factors=[1., .8, .64],
                time_steps=[.02, .01, .005], spatial_sweep_dt=.02,
                temporal_sweep_mesh_factor=1.,
                metrics=['mean_CD', 'rms_CL', 'St'],
                geometry=[-10., 20., -10., 10.],
                note='Three levels per axis. Non-nested meshes; polygonal cylinder also refined. '
                     'No domain-size test, Richardson extrapolation or combined h/dt error certificate.')


def settings(factor):
    return dict(reynolds=100., inlet_speed=1., diameter=1., viscosity=.01,
                xmin=-10., xmax=20., height=10., mesh_factor=factor,
                quadrature_order=5, domain_status='Fixed reference FEM domain')


def worker(args):
    """One isolated process owns the steady solve and transient sparse factors."""
    from Source.CFD.refinement_solver import run
    from Source.CFD.fem.steady import solve_case
    if args.factor == 1.:
        initial = steady_input()
    else:
        initial = args.output / 'steady' / 'finite_element_solution.npz'
        if not initial.exists():
            solve_case(settings(args.factor), initial.parent)
    run(f'h{args.factor:g}', args.dt, args.end, 'train', args.output, initial,
        min_end=args.min_end, stationarity_tol=.001)


def report(output, configuration, rows, state, current=None):
    baseline = [r for r in rows if r['name'] == 'reference']
    def sweep(axis):
        return baseline + [r for r in rows if r['axis'] == axis]
    value = dict(state=state, current_case=current, protocol=configuration,
                 runs=rows, spatial=sweep_verdict(sweep('space')),
                 temporal=sweep_verdict(sweep('time')))
    write_json(output / 'report.json', value)
    return value


def main(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    configuration = protocol()
    sources = [Path(__file__), ROOT/'Source/CFD/refinement_solver.py',
               ROOT/'Source/CFD/refinement_metrics.py', ROOT/'Source/CFD/fem/steady.py',
               ROOT/'Source/CFD/force_replay/cfd_force_replay.py',
               ROOT/'Source/CFD/force_replay/source/solve_steady.py']
    provenance = dict(protocol=configuration,
                      sources={str(p.relative_to(ROOT)): digest(p) for p in sources},
                      reference_input_sha256=digest(steady_input()),
                      python=sys.version, platform=platform.platform(),
                      packages={p: importlib.metadata.version(p) for p in ('numpy','scipy','scikit-fem','triangle','pandas')})
    signature = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    provenance['signature'] = signature
    prior = output / 'provenance.json'
    if prior.exists() and json.loads(prior.read_text())['signature'] != signature:
        raise RuntimeError('Protocol/code/environment changed: use a new output directory')
    write_json(prior, provenance)
    # Interleave sweeps so a bounded overnight run produces evidence on both axes.
    plan = [('reference', 'reference', 1., .02),
            ('dt_0p01', 'time', 1., .01), ('mesh_0p8', 'space', .8, .02),
            ('dt_0p005', 'time', 1., .005), ('mesh_0p64', 'space', .64, .02)]
    rows = []
    deadline = time.monotonic() + configuration['max_wall_hours']*3600
    for name, axis, factor, dt in plan:
        case = output / name
        case.mkdir(exist_ok=True)
        summary_path = case / 'summary.json'
        if not summary_path.exists():
            if time.monotonic() >= deadline:
                report(output, configuration, rows, 'wall_time_limit')
                return
            # Retain interrupted attempts; only completed cases are reused.
            if (case / 'forces.csv').exists() or (case / 'run.log').exists():
                archived = output / 'interrupted'
                archived.mkdir(exist_ok=True)
                case.rename(archived / f'{name}_{time.time_ns()}')
                case.mkdir()
            report(output, configuration, rows, 'running', name)
            command = [sys.executable, '-u', '-m', 'Source.CFD.refinement',
                       '--worker', '--output', str(case), '--factor', str(factor),
                       '--dt', str(dt), '--end', str(configuration['max_end']),
                       '--min-end', str(configuration['min_end'])]
            print('START_CASE', name, 'mesh_factor', factor, 'dt', dt, flush=True)
            with (case / 'run.log').open('w') as log:
                try:
                    subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   check=True, timeout=max(1., deadline-time.monotonic()))
                except subprocess.TimeoutExpired:
                    report(output, configuration, rows, 'wall_time_limit', name)
                    print('STOP: eight-hour limit; partial files preserved', flush=True)
                    return
                except subprocess.CalledProcessError:
                    report(output, configuration, rows, 'failed', name)
                    raise
        summary = json.loads(summary_path.read_text())
        if not summary.get('completed') or summary['dt'] != dt:
            raise RuntimeError(f'Invalid completed result: {name}')
        metrics = json.loads((case / 'cycle_metrics.json').read_text())
        rows.append(dict(name=name, axis=axis, factor=factor, dt=dt,
                         triangles=summary['triangles'], velocity_dofs=summary['velocity_dofs'],
                         pressure_dofs=summary['pressure_dofs'], actual_end=summary['actual_end'],
                         elapsed_seconds=summary['elapsed_seconds'], metrics=metrics))
        report(output, configuration, rows, 'running')
        print('DONE_CASE', name, 'stationary', metrics['stationary'],
              {k: metrics.get(k) for k in configuration['metrics']}, flush=True)
    final = report(output, configuration, rows, 'completed')
    print('STUDY_COMPLETE', json.dumps({k: final[k] for k in ('spatial','temporal')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'Build/CFD_Refinement')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--factor', type=float, default=1.)
    parser.add_argument('--dt', type=float, default=.02)
    parser.add_argument('--end', type=float, default=150.)
    parser.add_argument('--min-end', type=float, default=90.)
    options = parser.parse_args()
    if options.worker:
        if options.factor not in (1., .8, .64):
            parser.error('Supported mesh factors: 1, .8, .64')
        worker(options)
    else:
        main(options)
