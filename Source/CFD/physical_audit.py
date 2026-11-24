"""Regenerate both panels of Fig. S1 from frozen-basis CFD statistics.

Use --from-cfd DIR to extract the small sufficient statistics from six raw
velocity series, or --full-cfd to rerun their finite-element simulations first.
Fresh statistics default to <output>/inputs; archived Source_Data stays read-only.
The default run regenerates numerical CSVs; it does not read plotted values.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Set numerical-library thread defaults before importing NumPy/SciPy.
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_key, '1')

import numpy as np
from scipy.integrate import solve_ivp

# ROOT: code-package root; also supports direct execution of this file.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from Source.data_locations import published_path, shared_data, source_data_root
from Source.CFD.inputs import steady_input

# CASES/ROLES: figure labels, solver case names and historical data roles.
CASES = {
    'A': 'train', 'B': 'test_a', 'C': 'test_b',
    'D': 'holdout_d', 'E': 'holdout_e', 'F': 'holdout_f',
}
ROLES = {
    'A': 'Training', 'B': 'Previously viewed', 'C': 'Previously viewed',
    'D': 'New holdout', 'E': 'New holdout', 'F': 'New holdout',
}
# WINDOWS: nominal evaluation intervals; PARAMETERS: frozen radial-model keys.
WINDOWS = {'short': (40., 49.82906260206943), 'extended': (20., 90.)}
PARAMETERS = ('mu', 'beta', 'omega', 'gamma_raw', 'c0', 'c2')


def array_hash(a):
    # Include the array dtype and shape so the hash identifies its representation.
    a = np.ascontiguousarray(a)
    return hashlib.sha256(
        str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()
    ).hexdigest()


def audit_data():
    return shared_data('CFD_ROM', 'Physical_Forecast_Audit')


def __getattr__(name):
    # Retain old path attributes without resolving data during module import.
    if name == 'DATA':
        return audit_data()
    if name == 'MODEL':
        return shared_data('CFD_ROM', 'CFD_ROM', 'Re100', 'data')
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def frozen_model():
    model_data = shared_data('CFD_ROM', 'CFD_ROM', 'Re100', 'data')
    # phi: spatial modes stored by row; base: reference velocity field.
    with np.load(model_data / 'new_model_inputs.npz') as f:
        phi, base = f['Phi'], f['base']
    # p: fixed radius/phase dynamics and third-coordinate closure coefficients.
    p = json.loads((model_data / 'model_configuration.json').read_text())['parameters']
    p = {k: float(p[k]) for k in PARAMETERS}
    return phi, base, p


def extract(cfd_root: Path, destination: Path):
    """Compute projection coordinates and residual norms from actual CFD fields."""
    phi, base, parameters = frozen_model()
    # gram = Phi*Phi.T: mode inner products, including their stored normalization.
    gram = phi @ phi.T
    # arrays: compact inputs for error evaluation; records: per-case provenance.
    arrays = {'gram': gram}
    records = []
    for label, name in CASES.items():
        folder = cfd_root / f'coarse_dt0.02_{name}'
        path = folder / 'snapshots.npz'
        with np.load(path) as raw:
            # times: saved times; fields: one flattened two-component velocity per row.
            times = raw['times'].copy()
            fields = raw['fields'].astype(np.float64).reshape(len(times), -1)
        if len(times) != 376 or not np.allclose(
            times, np.arange(376)*.24, rtol=0, atol=2e-13
        ):
            raise ValueError(
                f'{label}: expected all saved times from 0 to 90 in increments of 0.24'
            )
        # Least-squares projection: gram*a = Phi*(u-base), independently at each time.
        # coordinates stores a by row; residual is the velocity outside the basis.
        coordinates = np.linalg.solve(gram, ((fields-base) @ phi.T).T).T
        residual = fields-base-coordinates @ phi
        # Squared spatial norms at each time, used without retaining full fields.
        projection_squared = np.einsum('ti,ti->t', residual, residual)
        field_squared = np.einsum('ti,ti->t', fields, fields)
        # Verify the sufficient-statistics identity with a deliberately nonzero
        # coordinate perturbation, before discarding the full velocity arrays.
        trial = coordinates + np.array([.013, -.017, .019])
        # Orthogonality: ||base+a_trial*Phi-u||^2 = ||residual||^2 + delta*gram*delta.T.
        direct = np.einsum(
            'ti,ti->t', base+trial @ phi-fields, base+trial @ phi-fields
        )
        delta = trial-coordinates
        reconstructed = projection_squared+np.einsum('ti,ij,tj->t', delta, gram, delta)
        # identity_error: largest discrepancy, normalized by the largest field norm squared.
        identity_error = float(np.max(np.abs(direct-reconstructed))/np.max(field_squared))
        if identity_error > 1e-12:
            raise AssertionError(f'{label}: projection norm identity failed: {identity_error}')
        arrays.update({
            f'{label}_times': times,
            f'{label}_coordinates': coordinates,
            f'{label}_projection_squared': projection_squared,
            f'{label}_field_squared': field_squared,
        })
        # config: original CFD settings; the record also identifies the source snapshots.
        config = json.loads((folder/'config.json').read_text())
        records.append({
            'case': label,
            'solver_case': name,
            'snapshots_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'snapshots_bytes': path.stat().st_size,
            'projection_identity_relative_error': identity_error,
            'simulation': config,
        })
        print('EXTRACTED', label, 'projection identity error', identity_error, flush=True)
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination/'audit_inputs.npz', **arrays)
    # provenance: identity of the frozen model and the six CFD inputs.
    provenance = {
        'source': 'Fresh full finite-element reruns of the recovered six-case solver',
        'Phi_sha256': array_hash(phi),
        'base_sha256': array_hash(base),
        'parameters': parameters,
        'cases': records,
        'stored_arrays': 'Projected coordinates, squared projection residual norms, squared full-field norms, times, and Gram matrix; no forecast errors are stored as inputs.',
    }
    (destination/'input_provenance.json').write_text(
        json.dumps(provenance, indent=2)+'\n'
    )


def forecast(initial, times, p):
    # rho0/theta0: initial radius and phase of the first two modal coordinates.
    rho0 = np.linalg.norm(initial[:2])
    theta0 = np.arctan2(initial[1], initial[0])

    def rhs(_, y):
        # y=(rho,theta): d(rho)/dt=mu*rho-beta*rho^3; d(theta)/dt=omega+gamma_raw*rho^2.
        return [
            p['mu']*y[0]-p['beta']*y[0]**3,
            p['omega']+p['gamma_raw']*y[0]**2,
        ]

    # sol: numerical radius/phase trajectory at the retained CFD times.
    sol = solve_ivp(
        rhs, (times[0], times[-1]), [rho0, theta0], t_eval=times,
        method='DOP853', rtol=2e-8, atol=2e-10,
    )
    if not sol.success or sol.y.shape[1] != len(times):
        raise RuntimeError(sol.message)
    rho, theta = sol.y
    # Modal reconstruction: a1=rho*cos(theta), a2=rho*sin(theta), a3=c0+c2*rho^2.
    predicted = np.c_[rho*np.cos(theta), rho*np.sin(theta), p['c0']+p['c2']*rho*rho]
    # An independent closed-form solution validates both the cubic radius and
    # its integrated phase, using the same physical (unscaled) modal units.
    elapsed = times-times[0]
    # q=rho^2=rho0^2*exp(2*mu*tau)/D, D=1+(beta*rho0^2/mu)*(exp(2*mu*tau)-1).
    denominator = 1 + p['beta']*rho0*rho0/p['mu']*np.expm1(2*p['mu']*elapsed)
    q = rho0*rho0*np.exp(2*p['mu']*elapsed)/denominator
    # theta=theta0+omega*tau+(gamma_raw/(2*beta))*log(D).
    angle = theta0+p['omega']*elapsed+p['gamma_raw']/(2*p['beta'])*np.log(denominator)
    # exact: modal trajectory from the analytic radius and phase.
    exact = np.c_[np.sqrt(q)*np.cos(angle), np.sqrt(q)*np.sin(angle), p['c0']+p['c2']*q]
    return predicted, exact, sol.nfev


def write_csv(path, rows, columns=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, columns or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def calculate(inputs: Path, output: Path, supplementary_csv: Path | None = None):
    phi, base, p = frozen_model()
    # Reject audit inputs extracted with a different basis, reference field or model.
    provenance = json.loads((inputs/'input_provenance.json').read_text())
    assert provenance['Phi_sha256'] == array_hash(phi), 'Frozen spatial modes changed'
    assert provenance['base_sha256'] == array_hash(base), 'Frozen steady field changed'
    assert provenance['parameters'] == p, 'Frozen radial model changed'
    # rows: full metrics; plot_rows: figure tables; predictions: trajectories for NPZ output.
    rows, plot_rows, predictions = [], {'short': [], 'extended': []}, {}
    # Maximum ODE-versus-analytic relative field difference over all cases/windows.
    max_analytic_error = 0.
    with np.load(inputs/'audit_inputs.npz') as data:
        gram = data['gram']
        np.testing.assert_allclose(gram, phi @ phi.T, rtol=1e-14, atol=1e-14)
        for label in CASES:
            # t: this case's saved times; use selects one nominal evaluation window.
            t = data[f'{label}_times']
            for window, (lo, hi) in WINDOWS.items():
                use = (t >= lo) & (t <= hi+1e-9)
                # observed: CFD projection coordinates; predicted/exact: model trajectories.
                times, observed = t[use], data[f'{label}_coordinates'][use]
                predicted, exact, nfev = forecast(observed[0], times, p)
                # projection/denominator: time-summed squared projection residual / CFD norm.
                projection = float(np.sum(data[f'{label}_projection_squared'][use]))
                denominator = float(np.sum(data[f'{label}_field_squared'][use]))
                # d: coordinate prediction error; extra=sum_t d(t)*gram*d(t).T.
                d = predicted-observed
                extra = max(0., float(np.einsum('ti,ij,tj->', d, gram, d)))
                # Relative field error (%) = 100*sqrt((projection+extra)/denominator).
                field_error = 100*np.sqrt((projection+extra)/denominator)
                projection_error = 100*np.sqrt(projection/denominator)
                # da: ODE-minus-analytic coordinate difference; analytic_error is not percent.
                da = predicted-exact
                analytic_error = float(np.sqrt(
                    max(0., np.einsum('ti,ij,tj->', da, gram, da))/denominator
                ))
                max_analytic_error = max(max_analytic_error, analytic_error)
                if analytic_error > 1e-7:
                    raise AssertionError(
                        f'{label}/{window}: ODE/analytic field difference {analytic_error}'
                    )
                rows.append({
                    'case': label,
                    'data_role': ROLES[label],
                    'window': window,
                    'nominal_start': lo,
                    'nominal_end': hi,
                    'saved_start': float(times[0]),
                    'saved_end': float(times[-1]),
                    'saved_states': len(times),
                    'field_error_pct': field_error,
                    'projection_error_pct': projection_error,
                    'within_basis_error_pct': 100*np.sqrt(extra/denominator),
                    'analytic_relative_field_difference': analytic_error,
                    'ode_evaluations': nfev,
                })
                plot_rows[window].append({
                    'Case': label,
                    'Data role': ROLES[label],
                    'Cubic field error (%)': f'{field_error:.6f}',
                    'Projection error (%)': f'{projection_error:.6f}',
                })
                predictions[f'{label}_{window}_coordinates'] = predicted
                predictions[f'{label}_{window}_times'] = times
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output/'physical_forecast_metrics.csv', rows)
    write_csv(output/'short_window.csv', plot_rows['short'])
    write_csv(output/'extended_window.csv', plot_rows['extended'])
    if supplementary_csv is not None:
        write_csv(supplementary_csv, plot_rows['short'])
    np.savez_compressed(output/'forecast_coordinates.npz', **predictions)
    # validation: completed checks, input identity and the recorded evaluation protocol.
    validation = {
        'cases': 6,
        'windows_per_case': 2,
        'model_refitted': False,
        'all_ode_solves_completed': True,
        'maximum_analytic_relative_field_difference': max_analytic_error,
        'input_sha256': hashlib.sha256((inputs/'audit_inputs.npz').read_bytes()).hexdigest(),
        'metric': '100 * sqrt(sum of squared full two-component velocity residuals / sum of squared full CFD velocities)',
        'initialization': 'First retained CFD state within each nominal window (40.08 or 20.16); no later state resets the prediction.',
        'holdout_scope': 'Historical D–F holdouts; this rerun evaluates an already frozen model and performs no new model selection.',
    }
    (output/'validation.json').write_text(json.dumps(validation, indent=2)+'\n')
    print(json.dumps(validation, indent=2))
    return rows


def run_cfd(directory, jobs):
    # solver: archived six-case velocity solver; directory: destination for raw CFD runs.
    solver = ROOT/'Source/CFD/replay_fem/cfd_benchmark.py'
    initial = steady_input()

    def run(name):
        subprocess.run([
            sys.executable, str(solver),
            '--input', str(initial),
            '--case', name,
            '--dt', '.02',
            '--end', '90',
            '--output-root', str(directory),
        ], check=True)

    # jobs limits the number of simultaneous CFD processes.
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        list(pool.map(run, CASES.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--full-cfd', action='store_true',
        help='Reintegrate all six full Navier–Stokes cases before extracting inputs',
    )
    parser.add_argument(
        '--from-cfd', type=Path,
        help='Directory containing coarse_dt0.02_train etc.; extract fresh sufficient statistics',
    )
    parser.add_argument('--cfd-output', type=Path, default=ROOT/'Build/Raw_CFD_Audit')
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument(
        '--inputs', type=Path,
        help='Statistics directory: archived inputs by default, or <output>/inputs when extracting CFD',
    )
    parser.add_argument('--output', type=Path, default=ROOT/'Build/Replayed_Physical_Audit')
    parser.add_argument('--update-supplementary-data', action='store_true')
    parser.add_argument(
        '--check', action='store_true',
        help='Compare regenerated presentation CSVs with the published tables',
    )
    args = parser.parse_args()

    # Fresh statistics belong to the run, not to the published reference dataset.
    extracting = args.full_cfd or args.from_cfd is not None
    inputs = args.inputs if args.inputs is not None else (
        args.output/'inputs' if extracting else audit_data()
    )
    if extracting and inputs.resolve().is_relative_to(source_data_root().resolve()):
        parser.error('Fresh CFD statistics cannot overwrite Source_Data; choose --inputs outside it.')

    # Optional CFD generation/extraction precedes the frozen-model error calculation.
    if args.full_cfd:
        run_cfd(args.cfd_output.resolve(), args.jobs)
        args.from_cfd = args.cfd_output
    if args.from_cfd:
        extract(args.from_cfd, inputs)
    calculate(
        inputs, args.output,
        published_path('Data/Supplementary_Data_1.csv') if args.update_supplementary_data else None,
    )
    # Compare the regenerated, rounded presentation tables with the published files.
    if args.check:
        for name in ('short_window.csv', 'extended_window.csv'):
            assert (args.output/name).read_bytes() == (audit_data()/name).read_bytes(), name
        assert (
            (args.output/'short_window.csv').read_bytes()
            == (published_path('Data/Supplementary_Data_1.csv')).read_bytes()
        )
        print('PASS: both regenerated numerical CSVs match the published tables')


if __name__ == '__main__':
    main()
