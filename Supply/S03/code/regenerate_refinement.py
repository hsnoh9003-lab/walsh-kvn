"""Reproduce Fig. S3b,c from the prepared wake input and the two phase readouts.

Run with --native to also recompile and simulate the 128-cell, 33-step circuit.
All field differences use the same continuum reference-field normalization.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import solve_ivp
from scipy.sparse import diags
from scipy.sparse.linalg import expm_multiply

FIGURE = Path(__file__).resolve().parents[1]
PACKAGE = FIGURE.parents[1]
sys.path.insert(0, str(PACKAGE))
from Reproduction.auxiliary.radial128.reproduce_flow import pair_step, run
from Reproduction.auxiliary.wake_upwind.reproduce import terminal_readout
from Source.data_locations import shared_data, published_path, figure_data
RADIAL = shared_data('Radial_refinement', 'Radial128')


def continuum_moments(amplitude, order, duration, mu, gamma):
    nodes, weights = leggauss(order)
    radii = ((np.arange(len(amplitude))[:, None] + (nodes + 1) / 2) / len(amplitude)).ravel()
    mass = (abs(amplitude[:, None]) ** 2 * weights / 2).ravel()
    initial = np.r_[radii, np.zeros_like(radii)]

    def rhs(t, y):
        r = y[:radii.size]
        return np.r_[mu*r*(1-r*r), gamma*r*r]

    solution = solve_ivp(rhs, (0, duration), initial, method='DOP853',
                         rtol=2e-12, atol=2e-14, t_eval=[duration])
    if not solution.success:
        raise RuntimeError(solution.message)
    r, phase = np.split(solution.y[:, -1], 2)
    return mass @ np.c_[r*np.cos(phase), r*np.sin(phase), r*r]


def calculate(native):
    inputs = RADIAL / 'inputs'
    config = json.loads((inputs / 'configuration.json').read_text())
    mu, gamma = (config['parameters'][key] for key in ('mu', 'gamma_normalized'))
    with np.load(inputs / 'direct_quantum_history.npz') as history, np.load(inputs / 'joint_readout_inputs.npz') as readout:
        amplitude = history['prepared_amplitude'].copy()
        amplitude /= np.linalg.norm(amplitude)
        duration = float(np.ptp(history['times']))
        continuum = readout['target']
        metric = readout['M'].T @ readout['G'] @ readout['M'] / float(readout['norm'])**2
        if native:
            circuit = json.loads((inputs / 'direct_64_33_circuit.json').read_text())
            run(7, 33, config, circuit, history, readout)
    references = {n: continuum_moments(amplitude, n, duration, mu, gamma) for n in (16, 32)}
    delta_quad = float(np.max(abs(references[32] - references[16])))
    delta_archive = float(np.max(abs(references[32] - continuum)))
    assert max(delta_quad, delta_archive) < 2e-11
    field_percent = lambda delta: 100 * np.sqrt(delta @ metric @ delta)
    rows = []
    for cells in (64, 128):
        bins = cells.bit_length()-1
        r = (np.arange(cells) + .5) / cells
        face = np.arange(1, cells) / cells
        coupling = (mu * cells/2) * face * (1-face**2)
        generator = diags((coupling, -coupling), (-1, 1), format='csr')
        refined = np.repeat(amplitude, cells//64) / np.sqrt(cells//64)
        joint = np.tile(refined, 4) / 2
        operators = [generator + diags(1j*gamma*w*r*r) for w in (0, 1, 1, 2)]
        exact = np.stack([expm_multiply(duration*g, refined/2) for g in operators])
        values = np.c_[np.tile(r,4)*np.repeat([1,-1,1,-1],cells),
                       np.tile(r,4)*np.repeat([1,1,-1,-1],cells), np.tile(r*r,4)]
        def means(state):
            prob = abs(terminal_readout(state)).ravel()**2
            assert abs(prob.sum()-1) < 1e-11
            return prob @ values
        exact_mean = means(exact)
        spatial = field_percent(exact_mean-continuum)
        for steps in (33, 66, 132):
            state = joint.copy()
            for _ in range(steps):
                state = pair_step(state, bins, duration/steps, mu, gamma)
            propagated = means(state)
            bias = field_percent(propagated-continuum)
            temporal = field_percent(propagated-exact_mean)
            native_dir = {(64,33): 'grid64', (128,33): 'grid128_steps33', (128,66): 'grid128'}.get((cells,steps))
            scope = 'exact finite generator and independent split pair rotations'
            if native_dir:
                source = json.loads((RADIAL/native_dir/'flow_report.json').read_text())
                assert abs(source['budget']['noiseless_field_bias_pct']-bias)<1e-9
                assert source['max_full_evolution_state_error']<1e-10
                scope += f'; native state-vector check in Data/Radial128/{native_dir}'
            rows.append(dict(positive_cells=cells, steps=steps, bias_pct=bias,
                             finite_generator_bias_pct=spatial, time_error_pct=temporal,
                             input='original 64-cell prepared density and signed amplitudes, uniformly subdivided at 128 cells',
                             validation=scope))
    directory = figure_data('S03') / 'plot_values'
    def table(path, fields, body):
        with path.open('w', newline='') as out:
            writer = csv.DictWriter(out, fieldnames=fields)
            writer.writeheader()
            writer.writerows(body)
    table(directory/'wake_refinement.csv', rows[0].keys(), rows)
    fields = ('positive_cells','steps','total_field_bias_percent','finite_generator_bias_percent','temporal_error_percent')
    summary = [dict(zip(fields, [r['positive_cells'],r['steps']]+[f'{r[k]:.8f}' for k in ('bias_pct','finite_generator_bias_pct','time_error_pct')])) for r in rows]
    table(published_path('Data/Supplementary_Data_2.csv'), fields, summary)
    report = dict(original_cells=64, quadrature_orders=[16,32], duration=duration,
                  integrator='DOP853',rtol=2e-12,atol=2e-14, terminal_moments_16=references[16].tolist(),
                  terminal_moments_32=references[32].tolist(), max_absolute_16_vs_32=delta_quad,
                  max_absolute_32_vs_archived_reference=delta_archive,
                  native_128_33='Data/Radial128/grid128_steps33/flow_report.json')
    (figure_data('S03') / 'numerical_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
    print('Maximum quadrature differences:', delta_quad, delta_archive)

if __name__ == '__main__':
    arguments = argparse.ArgumentParser(description=__doc__)
    arguments.add_argument('--native', action='store_true')
    calculate(arguments.parse_args().native)
