"""MPS-prepared runs on ibm_cleveland (SamplerV2, twirling + DD).
Paper settings: --sets 4,4,4 5,4,1 --shots 16000;
                --sets 4,6,6 --shots 32000.

The CLI encodes each (n, nt, kmax) triple in --sets. This executable submits
live provider jobs and writes counts to --out; importing it starts the workflow.
"""

import os, json, argparse
import numpy as np
from qiskit import transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
import kvn_mps as M

ap = argparse.ArgumentParser()
ap.add_argument("--sets", nargs="+", default=["4,4,4", "5,4,1"])
ap.add_argument("--shots", type=int, default=16000)
ap.add_argument("--out", default="ibm_mps_counts.json")
a = ap.parse_args()
svc = QiskitRuntimeService(
    channel="ibm_cloud",
    token=os.environ["IBM_TOKEN"],
    instance=os.environ["IBM_INSTANCE"],
)
bk = svc.backend("ibm_cleveland")
tg = bk.target


def score(t):
    e = 0
    for ins in t.data:
        q = tuple(t.find_bit(x).index for x in ins.qubits)
        nm = ins.operation.name
        if nm in tg and q in tg[nm] and tg[nm][q] is not None and tg[nm][q].error:
            e += -np.log(1 - tg[nm][q].error)
    return e


circs = []
for s in a.sets:
    n, nt, kmax = map(int, s.split(","))
    circs += M.mps_circuit_set(n, nt, kmax)
isa = [
    min(
        (transpile(c, bk, optimization_level=3, seed_transpiler=sd) for sd in range(6)),
        key=score,
    )
    for c in circs
]
smp = SamplerV2(mode=bk)
o = smp.options
o.default_shots = a.shots
o.twirling.enable_gates = True
o.twirling.enable_measure = True
o.twirling.num_randomizations = 32
o.dynamical_decoupling.enable = True
o.dynamical_decoupling.sequence_type = "XpXm"
job = smp.run(isa)
print("job", job.job_id())
res = job.result()
json.dump(
    {c.name: res[i].data.meas.get_counts() for i, c in enumerate(circs)},
    open(a.out, "w"),
)
