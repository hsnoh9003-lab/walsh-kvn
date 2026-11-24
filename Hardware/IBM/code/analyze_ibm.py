"""Offline IBM circuit validation and reference calibration from archived counts."""
from pathlib import Path
import ast
import csv
import json
import math
import re
import platform
from importlib.metadata import version
from collections import Counter
import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import norm, chi2

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data'
N = 16
X = (np.arange(N) + .5) / N
edges = np.linspace(0, 1, N + 1)
def masses(loc):
    p = np.diff(norm.cdf(edges, loc, .25))
    return p / p.sum()
P0 = masses(brentq(lambda loc: masses(loc) @ X - .9017, .5, 3))
uniform = np.ones(N) / N

def number(expr):
    def calc(node):
        if isinstance(node, ast.Expression): return calc(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (float, int)): return node.value
        if isinstance(node, ast.Name) and node.id == 'pi': return math.pi
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            return (-1 if isinstance(node.op, ast.USub) else 1) * calc(node.operand)
        if isinstance(node, ast.BinOp):
            a, b = calc(node.left), calc(node.right)
            for typ, fun in [(ast.Add, lambda: a+b), (ast.Sub, lambda: a-b),
                             (ast.Mult, lambda: a*b), (ast.Div, lambda: a/b),
                             (ast.Pow, lambda: a**b)]:
                if isinstance(node.op, typ): return fun()
        raise ValueError(expr)
    return calc(ast.parse(expr, mode='eval'))

def single(v, gate, q):
    for j in range(len(v)):
        if (j >> q) & 1: continue
        jj = j | (1 << q)
        v[[j, jj]] = gate @ v[[j, jj]]

def simulate_qasm(path):
    source = path.read_text()
    physical = '$' in source
    active = sorted(set(map(int, re.findall(r'\$(\d+)', source)))) if physical else list(range(4))
    positions = {q: i for i, q in enumerate(active)}
    v = np.zeros(2**len(active), complex); v[0] = 1
    counts = Counter(); measurements = {}
    for line in source.splitlines():
        line = line.strip()
        if not line or line.startswith(('OPENQASM', 'include', 'bit[', 'qubit[', 'barrier', '//', 'gphase')): continue
        if 'measure' in line:
            b = int(re.search(r'meas\[(\d+)\]', line)[1])
            q = int(re.search(r'\$(\d+)', line)[1]) if physical else int(re.search(r'q\[(\d+)\]', line)[1])
            measurements[b] = positions[q]; continue
        match = re.fullmatch(r'([\w]+)(?:\((.*?)\))?\s+(.*);', line)
        if not match: raise ValueError(line)
        name, args, target = match.groups()
        params = [number(x.strip()) for x in args.split(',')] if args else []
        qs = list(map(int, re.findall(r'\$(\d+)', target) if physical else re.findall(r'q\[(\d+)\]', target)))
        qs = [positions[q] for q in qs]; counts[name] += 1
        if name in ('cx', 'cz'):
            c, t = qs
            if name == 'cx':
                for j in range(len(v)):
                    if (j >> c) & 1 and not (j >> t) & 1:
                        jj = j | (1 << t); v[j], v[jj] = v[jj], v[j]
            else:
                for j in range(len(v)):
                    if (j >> c) & 1 and (j >> t) & 1: v[j] *= -1
        else:
            if name.lower() == 'u':
                theta, phi, lam = params; c, s = math.cos(theta/2), math.sin(theta/2)
                g = np.array([[c, -np.exp(1j*lam)*s], [np.exp(1j*phi)*s, np.exp(1j*(phi+lam))*c]])
            elif name == 'rz': g = np.diag(np.exp(np.array([-1j, 1j])*params[0]/2))
            elif name == 'sx': g = np.array([[1+1j, 1-1j], [1-1j, 1+1j]])/2
            elif name == 'x': g = np.array([[0, 1], [1, 0]])
            else: raise ValueError(name)
            single(v, g, qs[0])
    probs = np.zeros(N)
    for j, p in enumerate(abs(v)**2):
        index = sum(((j >> q) & 1) << b for b, q in measurements.items())
        probs[index] += p
    return probs, dict(counts)

def pair_state(nt, k):
    v = np.sqrt(P0).copy(); dt = (49.8291-40)/nt
    def layer(odd, duration):
        for j in range(1 if odd else 0, N-1, 2):
            face = (j+1)/N
            angle = .0924819*(face-face**3)*duration*N
            c, s = math.cos(angle/2), math.sin(angle/2)
            v[j:j+2] = np.array([[c, -s], [s, c]]) @ v[j:j+2]
    for _ in range(k): layer(False, dt/2); layer(True, dt); layer(False, dt/2)
    return v**2


def read_counts(data, name):
    counts = np.zeros(N)
    for key, value in data[name].items():
        counts[int(key, 2)] += value
    return counts/counts.sum(), int(counts.sum())


def calibrate(pk, pr):
    a = P0-uniform
    contrast = float(a @ (pr-uniform)/(a @ a))
    corrected = P0+(pk-pr)/contrast
    return contrast, corrected


def continuum_mean(elapsed):
    # Exact characteristics integrated against the matched piecewise-constant input.
    nodes, weights = np.polynomial.legendre.leggauss(64)
    initial = edges[:-1, None]+(nodes[None, :]+1)/(2*N)
    exponential = np.exp(.0924819*np.asarray(elapsed))
    return np.array([np.sum(P0[:, None]*weights[None, :]/2 *
        initial*e/np.sqrt(1+initial**2*(e**2-1))) for e in exponential])


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def main():
    archived = {}
    for row in csv.DictReader((OUT/'ideal_probabilities.csv').open()):
        archived[(int(row['set'][-1]), row['circuit'])] = np.array([float(row[f'cell_{j}']) for j in range(N)])
    errors = {'qasm_vs_archived': 0., 'logical_vs_ISA': 0.,
              'logical_vs_pair_rotation': 0., 'reference_vs_initial': 0.}
    gate_records = []
    for nt in (4, 6):
        for path in sorted((ROOT/f'circuits/logical/qasm3_dT{nt}').glob('*.qasm')):
            logical, lops = simulate_qasm(path)
            isa, iops = simulate_qasm(ROOT/f'circuits/ibm_cleveland_isa/qasm3_dT{nt}'/path.name)
            errors['qasm_vs_archived'] = max(errors['qasm_vs_archived'], float(abs(logical-archived[(nt, path.stem)]).max()))
            errors['logical_vs_ISA'] = max(errors['logical_vs_ISA'], float(abs(logical-isa).max()))
            k = int(path.stem.split('k')[-1])
            key = 'reference_vs_initial' if path.stem.startswith('ref') else 'logical_vs_pair_rotation'
            expected = P0 if path.stem.startswith('ref') else pair_state(nt, k)
            errors[key] = max(errors[key], float(abs(logical-expected).max()))
            gate_records.append({'nt': nt, 'circuit': path.stem, 'CX_logical': lops.get('cx', 0),
                                 **{f'ISA_{name}': count for name, count in iops.items()}})
    assert errors['logical_vs_ISA'] < 2e-12
    assert errors['logical_vs_pair_rotation'] < 2e-12
    assert errors['reference_vs_initial'] < 2e-12
    rows, diagnostics = [], []
    comparison_max = 0.
    bootstrap_relative_difference = 0.
    for nt in (4, 6):
        data = json.loads((OUT/f'ibm_sampler_dT{nt}_counts.json').read_text())
        stored = {int(r['k']): r for r in csv.DictReader((OUT/f'results_ibm_sampler_dT{nt}.csv').open())}
        rng = np.random.default_rng(0)
        for k in range(nt+1):
            pk, shots = read_counts(data, f'kvn_k{k}')
            ideal = pair_state(nt, k)
            row = {'nt': nt, 'k': k, 't_star': 40+k*(49.8291-40)/nt,
                   'shots': shots, 'P_ideal': float(ideal[14:].sum()), 'r_ideal': float(ideal@X),
                   'r_continuum': float(continuum_mean([k*(49.8291-40)/nt])[0]),
                   'P_raw': float(pk[14:].sum()), 'r_raw': float(pk@X),
                   'P_raw_SE': float(np.sqrt(pk[14:].sum()*(1-pk[14:].sum())/shots)),
                   'r_raw_SE': float(np.sqrt((pk@X**2-(pk@X)**2)/shots))}
            if k == 0:
                # No reference at k=0: omit forced-known calibrated values from plot data.
                row.update(F_ref='', P_refnorm='', P_refnorm_err='', r_refnorm='', r_refnorm_err='')
            else:
                pr, reference_shots = read_counts(data, f'ref_k{k}')
                contrast, corrected = calibrate(pk, pr)
                bootstrap = []
                for _ in range(1000):
                    _, sampled = calibrate(rng.multinomial(shots, pk)/shots,
                                           rng.multinomial(reference_shots, pr)/reference_shots)
                    bootstrap.append((sampled[14:].sum(), sampled@X))
                bootstrap = np.asarray(bootstrap)
                row.update(F_ref=contrast, P_ref_raw=float(pr[14:].sum()),
                    P_refnorm=float(corrected[14:].sum()), P_refnorm_err=float(bootstrap[:, 0].std()),
                    r_refnorm=float(corrected@X), r_refnorm_err=float(bootstrap[:, 1].std()))
                model = contrast*P0+(1-contrast)*uniform
                diagnostics.append({'nt': nt, 'k': k, 'F_ref': contrast,
                    'reference_depolarizing_TVD': float(abs(pr-model).sum()/2),
                    'corrected_min_probability': float(corrected.min()),
                    'corrected_negative_mass': float(-np.minimum(corrected, 0).sum()),
                    'corrected_r2': float(corrected@X**2),
                    'corrected_variance': float(corrected@X**2-(corrected@X)**2),
                    'r_bias_over_bootstrap_SE': float((corrected-ideal)@X/row['r_refnorm_err'])})
                for key in ['F_ref', 'P_refnorm', 'P_refnorm_err', 'r_refnorm', 'r_refnorm_err']:
                    difference = abs(row[key]-float(stored[k][key]))
                    if key.endswith('_err'):
                        bootstrap_relative_difference = max(bootstrap_relative_difference,
                            difference/float(stored[k][key]))
                        assert difference/float(stored[k][key]) < .02, (nt, k, key)
                    else:
                        comparison_max = max(comparison_max, difference)
                        assert difference < 5.1e-6, (nt, k, key)
            rows.append(row)
    write_csv(OUT/'ibm_reference_calibration_full_precision.csv', rows)
    write_csv(OUT/'ibm_reference_diagnostics.csv', diagnostics)
    write_csv(OUT/'ibm_gate_counts.csv', gate_records)
    report = {'provenance_scope': 'Offline checks of supplied files; no provider job retrieval.',
        'analysis_versions': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': version('scipy')},
        'initial_probabilities': P0.tolist(), 'initial_mean': float(P0@X),
        'bootstrap': {'resamples': 1000, 'seed_per_job': 0, 'standard_deviation_ddof': 0,
                      'scope': 'Independent multinomial count resampling; excludes systematic calibration bias.'},
        'simulation_checks': errors, 'QASM_circuits_checked': len(gate_records),
        'original_CSV_max_rounding_difference': comparison_max,
        'original_CSV_bootstrap_max_relative_difference': bootstrap_relative_difference,
        'bootstrap_note': 'Figure standard errors are recomputed from archived counts. The original rounded tables are retained; their bootstrap errors differ slightly from the recomputed Monte Carlo estimates.',
        'continuum': 'Analytical characteristics with 64-point Gaussian quadrature within each initial cell.',
        'hardware_scope': 'Four-qubit radial transport with general StatePreparation; no phase or field readout.'}
    (OUT/'offline_validation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'circuits': len(gate_records), 'simulation_checks': errors,
                      'original_CSV_max_difference': comparison_max}, indent=2))


if __name__ == '__main__':
    main()

