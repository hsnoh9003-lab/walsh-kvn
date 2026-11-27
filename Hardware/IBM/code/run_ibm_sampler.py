"""Run the 16-cell Walsh-KvN circuit set on ibm_cleveland with SamplerV2.
Usage: IBM_TOKEN=... IBM_INSTANCE=... python run_ibm_sampler.py --nt 4 --shots 8000
(paper runs: --nt 4 --shots 8000 --seed 11 ; --nt 6 --shots 32000 --seed 5)

This executable submits live provider jobs and saves counts in the current
working directory. Importing it also executes its top-level workflow.
"""

import os, json, argparse
from qiskit import transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
import kvn_circuits as K

ap = argparse.ArgumentParser()
ap.add_argument("--nt", type=int, default=4)
ap.add_argument("--shots", type=int, default=8000)
ap.add_argument("--seed", type=int, default=11)
ap.add_argument("--backend", default="ibm_cleveland")
a = ap.parse_args()
svc = QiskitRuntimeService(
    channel="ibm_cloud",
    token=os.environ["IBM_TOKEN"],
    instance=os.environ["IBM_INSTANCE"],
)
bk = svc.backend(a.backend)
circs = K.circuit_set(a.nt)
isa = transpile(circs, bk, optimization_level=3, seed_transpiler=a.seed)
s = SamplerV2(mode=bk)
s.options.default_shots = a.shots
s.options.twirling.enable_gates = True
s.options.twirling.enable_measure = True
s.options.twirling.num_randomizations = 32
s.options.dynamical_decoupling.enable = True
s.options.dynamical_decoupling.sequence_type = "XpXm"
job = s.run(isa)
print("job", job.job_id())
res = job.result()
counts = {c.name: res[i].data.meas.get_counts() for i, c in enumerate(circs)}
json.dump(counts, open(f"ibm_sampler_dT{a.nt}_counts.json", "w"))
