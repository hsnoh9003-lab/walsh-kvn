"""Reproduce cell-mass and moment checks without changing either simulation.

Run this file from any directory. Inputs are the archived MPS-prepared state
histories and the existing conservative scalar cell-mass export beside it.
Requires numpy and scipy. All TV distances are normalized probabilities (not %).
"""
from pathlib import Path
import csv
import json
import hashlib
import numpy as np
import sys
PACKAGE=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(PACKAGE))
from Source.data_locations import shared_data
from numpy.polynomial import Polynomial
from numpy.polynomial.legendre import leggauss
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.signal import find_peaks

ROOT = PACKAGE/'Build/Grid_Oscillation_Diagnostics'
INP = shared_data('Radial_refinement', 'Grid_Oscillation_Diagnostics', 'inputs')


def save_csv(name, rows):
    ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT / name).open('w', newline='') as stream:
        out = csv.DictWriter(stream, fieldnames=list(rows[0]))
        out.writeheader()
        out.writerows(rows)


def cdf_at(x, probabilities, edges):
    return np.interp(x, edges, np.r_[0.0, np.cumsum(probabilities)])


def extrema(p):
    """Strict interior extrema; no significance cutoff or smoothing."""
    maxima = (p[1:-1] > p[:-2]) & (p[1:-1] > p[2:])
    minima = (p[1:-1] < p[:-2]) & (p[1:-1] < p[2:])
    return int(maxima.sum()), int(minima.sum())


def main():
    config = json.loads((INP / 'configuration.json').read_text())
    radial = np.load(INP / 'direct_quantum_history.npz')
    times, centres = radial['times'], radial['r']
    p = np.sum(np.abs(radial['direct_history']) ** 2, axis=1)
    p /= p.sum(axis=1)[:, None]
    initial = radial['initial_probability'].copy()
    initial /= initial.sum()
    h = float(centres[1] - centres[0])
    edges = np.linspace(0, 1, len(centres) + 1)
    mu = config['parameters']['mu']
    inverse_check = solve_ivp(lambda t, y: -mu*y*(1-y*y),
                              (0, float(times[-1]-times[0])), edges,
                              t_eval=times-times[0], method='DOP853',
                              rtol=2e-13, atol=2e-15)
    assert inverse_check.success
    inverse_residual = 0.0
    rows, cells = [], []
    for step, (t, nominal) in enumerate(zip(times, p)):
        elapsed = float(t - times[0])
        q = np.exp(-2 * mu * elapsed)
        backflow = edges * np.sqrt(q) / np.sqrt(1 - (1 - q) * edges ** 2)
        backflow[0], backflow[-1] = 0.0, 1.0
        inverse_residual=max(inverse_residual,float(max(abs(backflow-inverse_check.y[:,step]))))
        reference = np.diff(cdf_at(backflow, initial, edges))
        mean = float(nominal @ centres)
        sigma = float(np.sqrt(nominal @ (centres - mean) ** 2))
        peaks, troughs = extrema(nominal)
        ref_peaks, ref_troughs = extrema(reference)
        prominent_peaks = len(find_peaks(nominal, prominence=.01*nominal.max())[0])
        prominent_troughs = len(find_peaks(-nominal, prominence=.01*nominal.max())[0])
        rows.append(dict(step=step, time=float(t),
                         total_variation=float(0.5 * np.abs(nominal-reference).sum()),
                         cell_centre_std=sigma, std_in_cells=sigma/h,
                         last_cell_probability=float(nominal[-1]),
                         strict_interior_maxima=peaks, strict_interior_minima=troughs,
                         reference_interior_maxima=ref_peaks,
                         reference_interior_minima=ref_troughs,
                         prominent_interior_maxima_1pct=prominent_peaks,
                         prominent_interior_minima_1pct=prominent_troughs,
                         mass_in_exact_zero_reference_cells=float(nominal[reference == 0].sum()),
                         field_bias_pct=float(radial['field_bias_pct'][step])))
        for j in range(len(centres)):
            cells.append(dict(step=step, time=float(t), cell=j,
                              circuit_mass=float(nominal[j]),
                              matched_characteristic_mass=float(reference[j])))
    save_csv('radial_history_diagnostics.csv', rows)
    save_csv('radial_cell_masses.csv', cells)

    q5 = np.load(INP / 'quintic_multistable__boundary_clipped.npz')
    p0 = q5['probability_history'][0].copy()
    p0 /= p0.sum()
    pq = q5['probability_history'][-1].copy()
    pq /= pq.sum()
    x = Polynomial([0.0, 1.0])
    raw = x * (1-x) * (x-.25) * (x-.5) * (x-.75)
    roots = raw.deriv().roots()
    candidates = [0.0, 1.0] + [float(r.real) for r in roots if abs(r.imag)<1e-12 and 0<r.real<1]
    scale = .25/max(abs(raw(np.array(candidates))))
    drift = scale * raw
    terminal_time = float(q5['times'][-1])
    inverse = solve_ivp(lambda t, y: -drift(y), (0, terminal_time), edges,
                        method='DOP853', rtol=2e-12, atol=2e-14)
    assert inverse.success
    reference_q5 = np.diff(cdf_at(inverse.y[:, -1], p0, edges))
    cell_records = [dict(cell=j, circuit_mass=float(pq[j]),
                         matched_characteristic_mass=float(reference_q5[j])) for j in range(64)]
    save_csv('q5_boundary_cell_masses.csv', cell_records)

    def characteristic_moments(order):
        nodes, weights = leggauss(order)
        positions = (edges[:-1, None] + h*(nodes[None, :]+1)/2).ravel()
        masses = (p0[:, None]*weights[None, :]/2).ravel()
        flow = solve_ivp(lambda t, y: drift(y), (0, terminal_time), positions,
                         method='DOP853', rtol=2e-12, atol=2e-14)
        assert flow.success
        y = flow.y[:, -1]
        return np.array([masses @ y, masses @ y**2])
    mref = characteristic_moments(64)
    refinement = float(np.max(abs(mref-characteristic_moments(32))))
    mq = np.array([pq @ centres, pq @ centres**2])
    g = np.zeros((64,64))
    for j, velocity in enumerate(drift(edges[1:-1])):
        g[j, j+1] = -velocity/(2*h)
        g[j+1, j] = velocity/(2*h)
    exact = expm(g * terminal_time) @ q5['state_history'][0]
    pe = abs(exact)**2
    me = np.array([pe @ centres, pe @ centres**2])
    with (INP / 'scalar_conservative_cell_masses.csv').open() as stream:
        archived = [r for r in csv.DictReader(stream) if r['id']=='quintic_multistable__boundary_clipped']
    archived_p = np.array([float(r['final_matched_characteristic_mass']) for r in archived])
    summary = dict(
        definitions=dict(total_variation='0.5 * sum_j |p_circuit[j] - p_characteristic[j]|',
                         absolute_moment_RMS='||m_circuit-m_reference||_2 / sqrt(2)',
                         relative_moment_RMS='||m_circuit-m_reference||_2 / ||m_reference||_2',
                         extrema='Strict interior cell-mass extrema; no smoothing or significance threshold.',
                         radial_width='Cell-centre standard deviation divided by h=1/64; the within-cell uniform variance is excluded.'),
        radial=dict(saved_states=len(times), cells=64, steps=33,
                    mu=mu, time_start=float(times[0]), time_end=float(times[-1]),
                    max_TV=float(max(r['total_variation'] for r in rows)),
                    min_TV_excluding_initial=float(min(r['total_variation'] for r in rows[1:])),
                    min_std_in_cells=float(min(r['std_in_cells'] for r in rows)),
                    initial_std_in_cells=rows[0]['std_in_cells'],
                    max_strict_interior_maxima=max(r['strict_interior_maxima'] for r in rows),
                    max_strict_interior_minima=max(r['strict_interior_minima'] for r in rows),
                    prominence_threshold='1% of each saved circuit marginal maximum cell probability',
                    max_prominent_interior_maxima=max(r['prominent_interior_maxima_1pct'] for r in rows),
                    max_prominent_interior_minima=max(r['prominent_interior_minima_1pct'] for r in rows),
                    final_TV=rows[-1]['total_variation'],
                    final_mass_in_exact_zero_reference_cells=rows[-1]['mass_in_exact_zero_reference_cells'],
                    final_last_cell_probability=rows[-1]['last_cell_probability']),
        q5_boundary=dict(normalization_scale=float(scale), total_variation=float(.5*abs(pq-reference_q5).sum()),
                         reference_first_cell_probability=float(reference_q5[0]), circuit_first_cell_probability=float(pq[0]),
                         absolute_moment_RMS=float(np.linalg.norm(mq-mref)/np.sqrt(2)),
                         relative_moment_RMS=float(np.linalg.norm(mq-mref)/np.linalg.norm(mref)),
                         exact_finite_generator_RMS=float(np.linalg.norm(me-mref)/np.sqrt(2)),
                         compiled_vs_finite_RMS=float(np.linalg.norm(mq-me)/np.sqrt(2)),
                         reference_moments=mref.tolist(), circuit_moments=mq.tolist(),
                         characteristic_quadrature_32_vs_64_max=refinement,
                         inverse_cell_mass_vs_archived_max_abs=float(max(abs(reference_q5-archived_p)))),
        scope='Only the archived 33-step radial wake and the scalar Q5 near-boundary control are assessed. No statement is made about the three-coordinate runs or later times.',
        inputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(INP.iterdir()) if p.is_file()})
    summary['radial']['analytic_inverse_vs_DOP853_max_abs'] = inverse_residual
    assert inverse_residual < 1e-10
    assert refinement < 1e-10
    assert summary['q5_boundary']['inverse_cell_mass_vs_archived_max_abs'] < 1e-10
    (ROOT / 'diagnostics_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k in ['radial','q5_boundary']}, indent=2))


if __name__ == '__main__':
    main()
