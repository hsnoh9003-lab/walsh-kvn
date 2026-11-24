"""Bond-dimension-2 MPS preparation of the truncated-Gaussian cell masses (as in SI Note 5),
followed by the Walsh-KvN Strang steps of kvn_circuits.py, for n = 4 (16 cells) or n = 5 (32 cells).
Target masses: CDF differences of N(loc, 0.25) on [0,1], loc chosen so the cell-centre mean is 0.9017043.
The MPS is obtained by left-to-right SVD truncation to chi = 2 and is prepared by a staircase of
nearest-neighbour two-qubit unitaries (right-canonical form, most significant bit first)."""
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.linalg import null_space
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Statevector
import kvn_circuits as K

def target_masses(n, sd=0.25, mean=0.9017043):
    Nx = 2 ** n; e = np.linspace(0, 1, Nx + 1); xc = (np.arange(Nx) + .5) / Nx
    m = lambda L: (lambda p: p / p.sum())(np.diff(norm.cdf(e, L, sd)))
    return m(brentq(lambda L: m(L) @ xc - mean, 0.5, 3))

def mps_chi2(psi, n, chi=2):
    A = []; M = psi.reshape(1, -1); r = 1
    for k in range(n - 1):
        M = M.reshape(r * 2, -1); U, S, Vt = np.linalg.svd(M, full_matrices=False)
        r2 = min(chi, len(S)); A.append(U[:, :r2].reshape(r, 2, r2)); M = np.diag(S[:r2]) @ Vt[:r2]; r = r2
    A.append(M.reshape(r, 2, 1)); v = A[0]
    for a in A[1:]: v = np.tensordot(v, a, axes=([-1], [0]))
    v = v.reshape(-1); return v / np.linalg.norm(v)

def _complete(cols):
    C = np.array(cols).T; return np.hstack([C, null_space(C.conj().T)])

def mps_prep(psi, n):
    v = mps_chi2(psi, n); T = v.reshape([2] * n)
    B = [None] * n; U, S, Vt = np.linalg.svd(T.reshape(-1, 2), full_matrices=False)
    r = min(2, len(S)); B[n - 1] = Vt[:r]; M = U[:, :r] * S[:r]
    for k in range(n - 2, 0, -1):
        rr = M.shape[1]; U, S, Vt = np.linalg.svd(M.reshape(-1, 2 * rr), full_matrices=False)
        r = min(2, len(S)); B[k] = Vt[:r].reshape(r, 2, rr); M = U[:, :r] * S[:r]
    B0 = M.reshape(2, -1); qc = QuantumCircuit(n); q = lambda k: n - 1 - k
    vec = np.zeros(4)
    for s in range(2):
        for b in range(B0.shape[1]): vec[s + 2 * b] = B0[s, b]
    qc.append(UnitaryGate(_complete([vec / np.linalg.norm(vec)])), [q(0), q(1)])
    for k in range(1, n - 1):
        cols = []
        for b in range(B[k].shape[0]):
            c = np.zeros(4)
            for s in range(2):
                for b2 in range(B[k].shape[2]): c[s + 2 * b2] = B[k][b, s, b2]
            cols.append(c)
        qc.append(UnitaryGate(_complete(cols)), [q(k), q(k + 1)])
    qc.append(UnitaryGate(_complete([B[n - 1][b] for b in range(B[n - 1].shape[0])])), [q(n - 1)])
    return qc

def mps_kvn_circuit(n, NT, k, zero=False, measure=True):
    K.N_QUBITS = n; K.NX = 2 ** n; K.XC = (np.arange(2 ** n) + .5) / 2 ** n
    qc = QuantumCircuit(n, name=f"n{n}_{'ref' if zero else 'kvn'}_k{k}")
    qc.compose(mps_prep(np.sqrt(target_masses(n)), n), inplace=True); qc.barrier()
    for _ in range(k): K.strang_step(qc, K.T / NT, zero)
    if measure: qc.measure_all()
    return qc

def mps_circuit_set(n, NT, kmax):
    return [mps_kvn_circuit(n, NT, k) for k in range(kmax + 1)] + [mps_kvn_circuit(n, NT, k, True) for k in range(1, kmax + 1)]
