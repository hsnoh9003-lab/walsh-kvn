"""Tables and figures for the MPS-prepared runs: ../data/results_ibm_mps_*.csv, ../figures/figH5, figH6."""
import os, json, csv
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from qiskit.quantum_info import Statevector
import kvn_mps as M, kvn_circuits as K
H = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(H, '..', 'data'); F = os.path.join(H, '..', 'figures')
def run(counts_file, sets, seed=0):
    C = json.load(open(os.path.join(D, counts_file))); rng = np.random.default_rng(seed); rows = []
    for n, nt, kmax in sets:
        Nx = 2 ** n; xc = (np.arange(Nx) + .5) / Nx; top = xc >= 7 / 8; u = np.ones(Nx) / Nx
        ideal = {c.name: Statevector(c.remove_final_measurements(inplace=False)).probabilities() for c in M.mps_circuit_set(n, nt, kmax)}
        p0 = ideal[f'n{n}_kvn_k0']
        def probs(nm):
            p = np.zeros(Nx)
            for b, v in C[nm].items(): p[int(b, 2)] += v
            S = p.sum(); return p / S, int(round(S))
        def cor(pk, pr):
            a = p0 - u; Fr = float(a @ (pr - u) / (a @ a)); return Fr, p0[top].sum() + (pk[top].sum() - pr[top].sum()) / Fr, p0 @ xc + (pk @ xc - pr @ xc) / Fr
        for k in range(kmax + 1):
            pk, S = probs(f'n{n}_kvn_k{k}'); pid = ideal[f'n{n}_kvn_k{k}']
            r = dict(cells=Nx, nt=nt, k=k, t_star=40 + k * K.T / nt, P_ideal=pid[top].sum(), r_ideal=pid @ xc, P_raw=pk[top].sum(), r_raw=pk @ xc, shots=S)
            if k == 0: r.update(F_ref=1.0, P_refnorm=p0[top].sum(), P_refnorm_err=0.0, r_refnorm=p0 @ xc, r_refnorm_err=0.0)
            else:
                pr, Sr = probs(f'n{n}_ref_k{k}'); Fr, Pc, Rc = cor(pk, pr)
                bs = np.array([cor(rng.multinomial(S, pk) / S, rng.multinomial(Sr, pr) / Sr)[1:] for _ in range(1000)])
                r.update(F_ref=Fr, P_refnorm=Pc, P_refnorm_err=bs[:, 0].std(), r_refnorm=Rc, r_refnorm_err=bs[:, 1].std())
            rows.append(r)
    return rows
def write(rows, fn):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(D, fn), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader()
        for r in rows: w.writerow({k: (f'{v:.6g}' if isinstance(v, float) else v) for k, v in r.items()})
m4 = run('ibm_mps_dT4_n4n5_counts.json', [(4, 4, 4), (5, 4, 1)]); m6 = run('ibm_mps_dT6_n4_counts.json', [(4, 6, 6)])
write(m4, 'results_ibm_mps_dT4_n4n5.csv'); write(m6, 'results_ibm_mps_dT6_n4.csv')
for r in m4 + m6: print(f"{r['cells']} cells NT={r['nt']} k={r['k']} F={r['F_ref']:.3f} P {r['P_ideal']:.3f} -> {r['P_refnorm']:.3f}±{r['P_refnorm_err']:.3f}  <r> {r['r_ideal']:.4f} -> {r['r_refnorm']:.4f}±{r['r_refnorm_err']:.4f}")
prev4 = list(csv.DictReader(open(os.path.join(D, 'results_ibm_sampler_dT4.csv')))); prev6 = list(csv.DictReader(open(os.path.join(D, 'results_ibm_sampler_dT6.csv'))))
BLUE, ORANGE, AQUA, INK, INK2 = '#2a78d6', '#eb6834', '#1baf7a', '#0b0b0b', '#52514e'
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2}); mk = dict(ms=8, mec='white', mew=2, capsize=3)
g = lambda rows, key: np.array([float(r[key]) for r in rows])
r4 = [r for r in m4 if r['cells'] == 16]; r5 = [r for r in m4 if r['cells'] == 32]; t = g(r4, 't_star')
fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.8))
for a, key, yl, ttl in [(ax[0], 'P', 'P(r ≥ 7/8)', 'a  Concentration (reference-normalized), δ = T/4'), (ax[1], 'r', '⟨r⟩', 'b  Ensemble mean radius (reference-normalized)')]:
    a.plot(t, g(r4, key + '_ideal'), '-', color=INK, lw=2, label='Ideal, 16 cells')
    a.plot(t[:2], g(r5, key + '_ideal'), '--', color=INK2, lw=2, label='Ideal, 32 cells')
    a.errorbar(t - 0.1, g(prev4, key + '_refnorm'), yerr=g(prev4, key + '_refnorm_err'), fmt='o', color=BLUE, label='16 cells, exact state preparation', **mk)
    a.errorbar(t + 0.1, g(r4, key + '_refnorm'), yerr=g(r4, key + '_refnorm_err'), fmt='s', color=ORANGE, label='16 cells, bond-2 MPS preparation', **mk)
    a.errorbar(t[:2] + 0.25, g(r5, key + '_refnorm'), yerr=g(r5, key + '_refnorm_err'), fmt='D', color=AQUA, label='32 cells, bond-2 MPS preparation', **mk)
    a.set_ylabel(yl); a.set_title(ttl, loc='left', fontsize=10); a.set_xlabel('t*'); a.grid(alpha=0.25, lw=0.6); a.spines[['top', 'right']].set_visible(False); a.legend(frameon=False, fontsize=7.5, loc='lower right')
fig.tight_layout(); fig.savefig(os.path.join(F, 'figH5_ibm_mps_dT4.png'), dpi=200); plt.close(fig)
t = g(m6, 't_star'); fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.8))
a = ax[0]; a.plot(t, g(m6, 'P_ideal'), '-', color=INK, lw=2, label='Ideal, 16 cells')
a.errorbar(t - 0.08, g(prev6, 'P_refnorm'), yerr=g(prev6, 'P_refnorm_err'), fmt='o', color=BLUE, label='Exact state preparation', **mk)
a.errorbar(t + 0.08, g(m6, 'P_refnorm'), yerr=g(m6, 'P_refnorm_err'), fmt='s', color=ORANGE, label='Bond-2 MPS preparation', **mk)
a.set_ylabel('P(r ≥ 7/8)'); a.set_xlabel('t*'); a.set_title('a  Concentration (reference-normalized), δ = T/6', loc='left', fontsize=10)
b = ax[1]; cz_exact = [94, 175, 262, 345, 429, 513]; cz_mps = [91, 172, 256, 342, 427, 510]
b.plot(cz_exact, g(prev6, 'F_ref')[1:], '-o', color=BLUE, lw=1.5, ms=7, mec='white', mew=2, label='Exact state preparation')
b.plot(cz_mps, g(m6, 'F_ref')[1:], '-s', color=ORANGE, lw=1.5, ms=7, mec='white', mew=2, label='Bond-2 MPS preparation')
b.set_xlabel('CZ gates'); b.set_ylabel('Reference fidelity F'); b.set_title('b  Reference-circuit fidelity vs depth', loc='left', fontsize=10)
for x in ax: x.grid(alpha=0.25, lw=0.6); x.spines[['top', 'right']].set_visible(False)
a.legend(frameon=False, fontsize=8, loc='lower right'); b.legend(frameon=False, fontsize=8, loc='upper right')
fig.tight_layout(); fig.savefig(os.path.join(F, 'figH6_ibm_mps_dT6.png'), dpi=200); plt.close(fig)
