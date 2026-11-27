"""Walsh-KvN circuits for the radial Stuart-Landau ensemble on a 16-cell grid.

Model: dr/dt* = f(r) = mu (r - r^3), mu = 0.0924819, on r in [0, 1].
Interval: t* = 40 -> 49.8291 (T = 9.8291), split into NT Strang steps.
Register: n = 4 qubits (Nx = 16 cells, cell j <-> integer j = sum_i 2^i b_i, qubit i = bit i).
One Strang step = even layer (dt/2) . odd layer (dt) . even layer (dt/2).
Each layer is a multiplexed RY on qubit 0 (pair target) controlled by the
pair-address qubits 1..3, synthesized from its Walsh coefficients with
Gray-ordered CX parity reuse. The odd layer is aligned to the even-layer
pairing by a cyclic decrement/increment of the cell register.
Reference circuits ("ref") keep every gate but set all Walsh angles to zero;
barriers prevent the compiler from cancelling the CX pairs.
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.integrate import solve_ivp
from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation

MU = 0.0924819
T = 49.8291 - 40.0
N_QUBITS = 4
NX = 2**N_QUBITS
XC = (np.arange(NX) + 0.5) / NX  # cell centres
R_MEAN0 = 0.9017  # initial ensemble mean radius
SIGMA0 = 0.25  # Gaussian width before truncation to [0, 1]


def f(r):
    return MU * (r - r**3)


def initial_distribution():
    """Cell masses of a truncated Gaussian on [0,1] whose discrete mean is R_MEAN0."""
    edges = np.linspace(0, 1, NX + 1)

    def masses(loc):
        p = np.diff(norm.cdf(edges, loc, SIGMA0))
        return p / p.sum()

    loc = brentq(lambda L: masses(L) @ XC - R_MEAN0, 0.5, 3)
    return masses(loc)


def walsh_coeffs(angles, m):
    N = 2**m
    c = {}
    for S in range(N):
        s = sum(angles[q] * (-1) ** bin(q & S).count("1") for q in range(N)) / N
        if abs(s) > 1e-12:
            c[S] = s
    return c


def increment(qc, reg):
    n = len(reg)
    for k in range(n - 1, 0, -1):
        qc.mcx(reg[:k], reg[k])
    qc.x(reg[0])


def mux_ry(qc, tgt, addr, angles, zero=False):
    """Uniformly controlled RY via Walsh coefficients, Gray-ordered parity reuse."""
    m = len(addr)
    c = walsh_coeffs(angles, m)
    gray = [g ^ (g >> 1) for g in range(2**m)]
    order = [S for S in gray if S in c]
    cur = 0
    for S in order:
        d = cur ^ S
        for i in range(m):
            if d >> i & 1:
                qc.cx(addr[i], tgt)
        qc.ry(0.0 if zero else c[S], tgt)
        qc.barrier()
        cur = S
    for i in range(m):
        if cur >> i & 1:
            qc.cx(addr[i], tgt)
    qc.barrier()


def strang_step(qc, dt, zero=False):
    reg = list(range(N_QUBITS))
    ev = [
        f((2 * q + 1) / NX) * dt * NX for q in range(NX // 2)
    ]  # faces of pairs (2q, 2q+1)
    od = [
        f((2 * q + 2) / NX) * dt * NX for q in range(NX // 2)
    ]  # faces of pairs (2q+1, 2q+2)
    inc = QuantumCircuit(N_QUBITS)
    increment(inc, reg)
    mux_ry(qc, 0, reg[1:], [a / 2 for a in ev], zero)
    qc.compose(inc.inverse(), reg, inplace=True)
    qc.barrier()  # decrement
    mux_ry(qc, 0, reg[1:], od, zero)
    qc.compose(inc, reg, inplace=True)
    qc.barrier()  # increment
    mux_ry(qc, 0, reg[1:], [a / 2 for a in ev], zero)


def kvn_circuit(k, NT, zero=False, measure=True):
    """k Strang steps of length T/NT. zero=True gives the matched reference circuit."""
    p0 = initial_distribution()
    qc = QuantumCircuit(N_QUBITS, name=f"{'ref' if zero else 'kvn'}_k{k}")
    qc.append(StatePreparation(np.sqrt(p0)), range(N_QUBITS))
    qc.barrier()
    for _ in range(k):
        strang_step(qc, T / NT, zero)
    if measure:
        qc.measure_all()
    return qc


def circuit_set(NT):
    """Circuits run on hardware: kvn_k0..kvn_kNT, then ref_k1..ref_kNT."""
    return [kvn_circuit(k, NT) for k in range(NT + 1)] + [
        kvn_circuit(k, NT, zero=True) for k in range(1, NT + 1)
    ]


def continuum_moments(t_list, nsub=200):
    """<r>, <r^2> of the continuum ensemble (piecewise-uniform initial density, characteristics)."""
    p0 = initial_distribution()
    edges = np.linspace(0, 1, NX + 1)
    out = []
    for tt in t_list:
        m1 = m2 = 0.0
        for j in range(NX):
            r0 = np.linspace(edges[j], edges[j + 1], nsub + 2)[1:-1]
            rT = solve_ivp(
                lambda t, y: f(y), [0, tt + 1e-12], r0, rtol=1e-11, atol=1e-13
            ).y[:, -1]
            m1 += p0[j] * rT.mean()
            m2 += p0[j] * (rT**2).mean()
        out.append((m1, m2))
    return out
