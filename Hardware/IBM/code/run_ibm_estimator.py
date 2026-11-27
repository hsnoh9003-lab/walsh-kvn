"""Estimator runs on ibm_cleveland: <r>, <r^2>, P(r >= 7/8) with TREX + ZNE.
--mode folding : resilience level 2, gate-folding ZNE (1,3,5), observables r, r^2   (paper: --nt 4)
--mode pea     : server-side Estimator, PEA ZNE (1,1.5,2,3), observables r, r^2, P  (paper: --nt 4 and --nt 6)

This executable submits live provider jobs and saves results in the current
working directory. Importing it also executes its top-level workflow.
"""

import os, json, argparse
from qiskit import transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import QiskitRuntimeService
import kvn_circuits as K

ap = argparse.ArgumentParser()
ap.add_argument("--nt", type=int, nargs="+", default=[4, 6])
ap.add_argument("--mode", choices=["folding", "pea"], default="pea")
a = ap.parse_args()
n, Nx = K.N_QUBITS, K.NX
w = [-(2**i) / 2 / Nx for i in range(n)]
c0 = (0.5 + sum(2**i / 2 for i in range(n))) / Nx


def Z(*idx):
    s = ["I"] * n
    for i in idx:
        s[n - 1 - i] = "Z"
    return "".join(s)


R = SparsePauliOp.from_list(
    [(Z(), c0)] + [(Z(i), w[i]) for i in range(n)]
)  # r = c0 + sum_i w_i Z_i
R2 = SparsePauliOp.from_list(
    [(Z(), c0**2 + sum(x * x for x in w))]
    + [(Z(i), 2 * c0 * w[i]) for i in range(n)]
    + [(Z(i, j), 2 * w[i] * w[j]) for i in range(n) for j in range(i + 1, n)]
).simplify()
P = SparsePauliOp.from_list([("IIII", 1.0)])  # projector q1=q2=q3=1
for q in (1, 2, 3):
    P = P.compose(SparsePauliOp.from_list([("IIII", 0.5), (Z(q), -0.5)]))
P = P.simplify()
obs = [R, R2] if a.mode == "folding" else [R, R2, P]
svc = QiskitRuntimeService(
    channel="ibm_cloud",
    token=os.environ["IBM_TOKEN"],
    instance=os.environ["IBM_INSTANCE"],
)
bk = svc.backend("ibm_cleveland")
pubs, meta = [], []
for nt in a.nt:
    seed = 11 if nt == 4 else 5
    circs = [K.kvn_circuit(k, nt, measure=False) for k in range(nt + 1)]
    for k, t in enumerate(
        transpile(circs, bk, optimization_level=3, seed_transpiler=seed)
    ):
        pubs.append((t, [o.apply_layout(t.layout) for o in obs]))
        meta.append((nt, k))
if a.mode == "folding":
    from qiskit_ibm_runtime import EstimatorV2 as Est

    e = Est(mode=bk)
    o = e.options
    o.resilience_level = 2
    o.resilience.zne.noise_factors = (1, 3, 5)
    o.twirling.enable_gates = True
else:
    from qiskit_ibm_runtime.server_side import Estimator as Est

    e = Est(mode=bk)
    o = e.options
    o.resilience_level = 2
    o.resilience.measure_mitigation = True
    o.resilience.zne_mitigation = True
    o.resilience.zne.amplifier = "pea"
    o.resilience.zne.noise_factors = (1, 1.5, 2, 3)
    o.twirling.enable_gates = True
    o.twirling.enable_measure = True
o.resilience.zne.extrapolator = ("exponential", "linear")
o.dynamical_decoupling.enable = True
o.default_shots = 16000
job = e.run(pubs)
print("job", job.job_id())
res = job.result()
out = [
    dict(
        nt=m[0],
        k=m[1],
        observables=["r", "r2", "P_top"][: len(obs)],
        ev=res[i].data.evs.tolist(),
        std=res[i].data.stds.tolist(),
    )
    for i, m in enumerate(meta)
]
json.dump(out, open(f"ibm_estimator_zne_{a.mode}.json", "w"), indent=1)
