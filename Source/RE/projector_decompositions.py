"""Executable projector gate lists shared by circuit construction and RE.

No quantum SDK is required to count these gates. Operations are tuples
``(name, qubit_indices, parameters)`` in execution order. Rz uses
diag(exp(-i*theta/2), exp(i*theta/2)).

The default implements the same projector phase as the archived exact
C5X--Rz(2*phi)--C5X block. It factors the five controls into groups of
three and two. Four controlled flips and four quarter-angle rotations
implement C5Rz(-4*phi); its first rotation merges with Rz(2*phi).
The borrowed position bit is restored for every input, including entangled
inputs. The common reflection is kept exactly as in the archived circuit.
"""
from collections import Counter

COUNT_MODEL_VERSION = "projector-merged4rz-rccx5clifford-20261020-v1"


def _op(name, qubits, *params):
    return (name, tuple(qubits), tuple(params))


def dirty_mcx_operations(controls, target, borrowed):
    controls, borrowed = list(controls), list(borrowed)
    k = len(controls)
    if k < 1:
        raise ValueError("At least one control is required")
    if k == 1:
        return [_op("cx", [controls[0], target])]
    if k == 2:
        return [_op("ccx", [*controls, target])]
    a = borrowed[:k-2]
    if len(a) != k-2 or len(set(controls + [target] + a)) != 2*k-1:
        raise ValueError("Distinct controls, target, and k-2 borrowed bits required")
    ladder = [(controls[0], controls[1], a[0])]
    ladder += [(controls[j+1], a[j-1], a[j]) for j in range(1, k-2)]
    ops = []
    for subset in (ladder, ladder[1:]):
        ops += [_op("ccx", gate) for gate in subset]
        ops.append(_op("ccx", [controls[-1], a[-1], target]))
        ops += [_op("ccx", gate) for gate in reversed(subset)]
    return ops


def inverse_operations(ops):
    inverse_names = {"t": "tdg", "tdg": "t", "s": "sdg", "sdg": "s"}
    return [(inverse_names.get(name, name), qs,
             tuple(-p for p in ps) if name in ("rz", "ry") else ps)
            for name, qs, ps in reversed(ops)]


def relative_phase_toffoli_operations(a, b, target, *, inverse=False):
    """RCCX: four T/Tdg and five Clifford gates (2 H + 3 CX)."""
    ops = [_op("h", [target]), _op("t", [target]),
           _op("cx", [b, target]), _op("tdg", [target]),
           _op("cx", [a, target]), _op("t", [target]),
           _op("cx", [b, target]), _op("tdg", [target]),
           _op("h", [target])]
    return inverse_operations(ops) if inverse else ops


def projector_operations(flags, real, phi, common, position, *,
                         mode="ancilla_free", clean_ancillas=(), merge=True):
    """Five all-zero flags projector; optional common phase is 0 or -pi/2.

    ``merge=False`` retains the five-Rz reference for the default construction.
    The ancilla-assisted construction requires four extra clean |0> bits;
    relative phases cancel through the explicit compute/uncompute pair.
    """
    import math
    flags, position, clean = list(flags), list(position), list(clean_ancillas)
    if len(flags) != 5 or len(set(flags + [real])) != 6:
        raise ValueError("Exactly five distinct flags and a separate target required")
    if common and not math.isclose(float(common), -math.pi/2, abs_tol=1e-13):
        raise ValueError("The common reflection must be 0 or -pi/2")
    ops = [_op("x", [f]) for f in flags]
    if mode == "ancilla_free":
        # a=AND(flags[:3]), b=AND(flags[3:]); a,b are Boolean predicates.
        # R(theta/4),Xa,R(-theta/4),Xb,R(theta/4),Xa,R(-theta/4),Xb
        # has net angle theta*a*b. Here theta=-4*phi; prepend R(2*phi).
        if not merge:
            ops += [_op("rz", [real], 2*phi), _op("rz", [real], -phi)]
        else:
            ops.append(_op("rz", [real], phi))
        ops += dirty_mcx_operations(flags[:3], real, position)
        ops.append(_op("rz", [real], phi))
        ops.append(_op("ccx", [*flags[3:], real]))
        ops.append(_op("rz", [real], -phi))
        ops += dirty_mcx_operations(flags[:3], real, position)
        ops.append(_op("rz", [real], phi))
        ops.append(_op("ccx", [*flags[3:], real]))
    elif mode == "ancilla_assisted":
        if len(clean) != 4 or len(set(flags + [real] + clean)) != 10:
            raise ValueError("Four distinct clean ancillas required")
        compute = relative_phase_toffoli_operations(flags[0], flags[1], clean[0])
        for j in range(1, 4):
            compute += relative_phase_toffoli_operations(flags[j+1], clean[j-1], clean[j])
        ops += compute
        # Rz(2phi) + CRz(-4phi); retain this mode's declared three-Rz cost.
        ops += [_op("rz", [real], 2*phi), _op("rz", [real], -2*phi),
                _op("cx", [clean[-1], real]), _op("rz", [real], 2*phi),
                _op("cx", [clean[-1], real])]
        ops += inverse_operations(compute)
    elif mode == "legacy_exact":
        # Reference only: the shipped one-Rz / 24-exact-CCX implementation.
        ops += dirty_mcx_operations(flags, real, position)
        ops.append(_op("rz", [real], 2*phi))
        ops += dirty_mcx_operations(flags, real, position)
    else:
        raise ValueError(f"Unknown projector mode {mode!r}")
    if common:
        ops.append(_op("h", [flags[-1]]))
        ops += dirty_mcx_operations(flags[:-1], flags[-1], position)
        ops.append(_op("h", [flags[-1]]))
    ops += [_op("x", [f]) for f in reversed(flags)]
    return ops


def operation_cost(ops):
    """Declared Clifford+T+arbitrary-Rz basis; exact CCX=7T+8 Clifford."""
    counts = Counter(name for name, _, _ in ops)
    allowed = {"rz", "ccx", "t", "tdg", "x", "h", "cx", "cz", "s", "sdg"}
    if set(counts) - allowed:
        raise ValueError(f"Unsupported operations {set(counts)-allowed}")
    return {"n_RZ": counts["rz"],
            "n_direct_T": 7*counts["ccx"] + counts["t"] + counts["tdg"],
            "n_base_Clifford": 8*counts["ccx"] + sum(counts[k] for k in
                                 ("x", "h", "cx", "cz", "s", "sdg"))}


def projector_cost(mode="ancilla_free"):
    return operation_cost(projector_operations(range(5), 5, 0.371, 0,
                          [6, 7, 8], mode=mode, clean_ancillas=[9, 10, 11, 12]))

# Direct-real realization: the five good-branch flags themselves receive the
# phase. No extra realization target is present. An exact no-ancilla 5-qubit
# phase polynomial is used so that its RE cost is independently auditable.
# This is a conservative decomposition, NOT an optimal C^4P synthesis.
DIRECT_PHASE_COUNT_MODEL_VERSION = 'direct-real-5flag-phasepoly-31rz-98cx-v1'


def direct_projector_operations(flags, phi, common, position):
    """Implement exp(i*phi*(2|00000><00000|-I)) on five flags.

    Fourier identity: |0^5><0^5| = 2^-5 sum_{S subset [5]} Z_S.
    The S=empty term and exp(-i phi) are applied as GLOBAL PHASE by
    physical_circuits.projector_direct; this returns the remaining phase
    gadgets. Each nonempty Z_S is implemented by parity CNOTs, Rz(-phi/8),
    and uncomputation. No ancilla, 31 Rz + 98 CX. A separate optional
    common (-R) reflection is retained for the original Wx-QSVT convention.
    """
    from itertools import combinations
    import math
    flags=list(flags); position=list(position)
    if len(flags)!=5 or len(set(flags))!=5:
        raise ValueError('Exactly five distinct good-branch flags required')
    if common and not math.isclose(float(common),-math.pi/2,abs_tol=1e-12):
        raise ValueError('The common reflection must be 0 or -pi/2')
    ops=[]
    for n in range(1,6):
        for subset in combinations(flags,n):
            target=subset[-1]
            for control in subset[:-1]:ops.append(_op('cx',[control,target]))
            ops.append(_op('rz',[target],-float(phi)/8.))
            for control in reversed(subset[:-1]):ops.append(_op('cx',[control,target]))
    if common:
        ops.extend(_op('x',[qb]) for qb in flags)
        ops.append(_op('h',[flags[-1]]))
        ops += dirty_mcx_operations(flags[:-1],flags[-1],position)
        ops.append(_op('h',[flags[-1]]))
        ops.extend(_op('x',[qb]) for qb in reversed(flags))
    return ops


def direct_projector_cost():
    """Cost of one ordinary projector phase WITHOUT the common reflection."""
    return operation_cost(direct_projector_operations(range(5),0.317,0,range(5,8)))
