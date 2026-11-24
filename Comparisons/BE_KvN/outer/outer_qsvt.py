"""QSVT exponential from Chebyshev cosine/sine and one OAA round.

Signal model follows Gilyén et al., arXiv:1806.01838, §5.1 and Theorem 58.
The elementary inventory is for the explicit template described in
outer_inventory() and excludes the compiled oracle and common phase kicks.
"""
from collections import Counter
from dataclasses import asdict, dataclass
from functools import lru_cache
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
from numpy.polynomial.chebyshev import chebval
from scipy.optimize import least_squares
from scipy.special import jv

# Exactly the same one-clean controlled-U(2) decomposition as KvN–LCHS.
LCHS_COMPILER = Path(__file__).resolve().parents[2] / "KvN_LCHS"
if str(LCHS_COMPILER) not in sys.path:
    sys.path.insert(0, str(LCHS_COMPILER))
from compile_stmc_one_clean import controlled_source, lower_to_basis


@dataclass
class PolynomialDesign:
    alpha: float
    step: float
    budget: float
    degree: int
    bessel_tail_bound: float
    pre_amplification_scale: float
    full_interval_bound: float
    paper_degree_bound: int


def tail_bound(z, degree):
    """Bound separate even/odd tails using actual J_n and an analytic remainder."""
    cutoff = max(degree + 16, math.ceil(2 * abs(z)) + 16)
    orders = np.arange(degree + 1, cutoff + 1)
    terms = 2 * np.abs(jv(orders, z))
    even = terms[orders % 2 == 0].sum()
    odd = terms[orders % 2 == 1].sum()
    log_term = (cutoff + 1) * math.log(abs(z) / 2) - math.lgamma(cutoff + 2)
    remainder = 2 * math.exp(log_term) / (1 - abs(z) / (2 * (cutoff + 2)))
    return float(max(even, odd) + remainder)


def coefficients(alpha, step, budget):
    z = alpha * step
    degree = 1
    while True:
        tail = tail_bound(z, degree)
        # Each component loses at most another tail when scaled to norm <= 1.
        # Conservative scalar bound for the final amplification via |dp| < 1.
        full_interval_bound = 6 * math.sqrt(2) * (tail + 1e-12)
        if full_interval_bound <= budget:
            break
        degree += 1
    pre_scale = 1 / (1 + tail + 1e-12)
    orders = np.arange(degree + 1)
    values = 2 * (-1.0) ** (orders // 2) * jv(orders, z) * pre_scale
    values[0] *= 0.5
    cosine = values.copy()
    cosine[orders % 2 == 1] = 0
    sine = values.copy()
    sine[orders % 2 == 0] = 0
    argument = math.e * abs(z) / 2
    paper_degree = max(1, math.ceil(argument))
    while paper_degree * math.log(argument / paper_degree) > math.log(budget / 6):
        paper_degree += 1
    design = PolynomialDesign(alpha, step, budget, degree, tail, pre_scale,
                              full_interval_bound, paper_degree)
    return design, np.trim_zeros(cosine, "b"), np.trim_zeros(sine, "b")


def qsp_values(phases, x):
    """Upper-left of exp(i phi_d Z) W(x)...W(x) exp(i phi_0 Z)."""
    transition = 1j * np.sqrt(np.maximum(0, 1 - x * x))
    top = np.full_like(x, np.exp(1j * phases[0]), dtype=complex)
    bottom = np.zeros_like(top)
    for phase in phases[1:]:
        top, bottom = ((x * top + transition * bottom) * np.exp(1j * phase),
                       (transition * top + x * bottom) * np.exp(-1j * phase))
    return top


def qsp_jacobian(phases, x):
    transition = 1j * np.sqrt(1 - x * x)
    factors = np.exp(1j * phases)
    forwards = np.empty((len(phases), 2, len(x)), dtype=complex)
    forwards[0, 0] = factors[0]
    forwards[0, 1] = 0
    for index, factor in enumerate(factors[1:], start=1):
        top, bottom = forwards[index - 1]
        forwards[index] = ((x * top + transition * bottom) * factor,
                           (transition * top + x * bottom) * factor.conjugate())
    jacobian = np.empty((len(x), len(phases)))
    left = np.ones_like(x, dtype=complex)
    right = np.zeros_like(left)
    for index in reversed(range(len(phases))):
        top, bottom = forwards[index]
        jacobian[:, index] = (1j * (left * top - right * bottom)).real
        left, right = (left * factors[index] * x + right * factors[index].conjugate() * transition,
                       left * factors[index] * transition + right * factors[index].conjugate() * x)
    return jacobian


def fit_phases(coefs):
    degree = len(coefs) - 1
    n_independent = (degree + 2) // 2
    x = np.cos(np.pi * (np.arange(n_independent) + 0.5) / (2 * n_independent))
    target = chebval(x, coefs)

    def mirrored(parameters):
        return np.r_[parameters, parameters[:(degree + 1) // 2][::-1]]

    def jacobian(parameters):
        full = qsp_jacobian(mirrored(parameters), x)
        result = full[:, :n_independent].copy()
        reflected = (degree + 1) // 2
        result[:, :reflected] += full[:, -reflected:][:, ::-1]
        return result

    start = np.zeros(n_independent)
    start[0] = np.pi / 4
    rng = np.random.default_rng(46719)
    scales = (1,) if degree < 24 else (0.25, 0.5, 0.75, 0.9, 0.96, 0.99, 0.999, 1)
    for index in range(4):
        guess = start if index == 0 else start + rng.normal(0, 0.15, len(start))
        for scale in scales:
            def residual(parameters):
                return qsp_values(mirrored(parameters), x).real - scale * target
            result = least_squares(residual, guess, jac=jacobian, xtol=2e-15,
                                   ftol=2e-15, gtol=2e-15, max_nfev=180)
            guess = result.x
        if max(np.abs(result.fun)) < 5e-13:
            phases = mirrored(result.x)
            # Recover the realized polynomial at all Chebyshev nodes, then
            # bound its uniform difference by the sum of coefficient errors.
            theta = np.pi * (np.arange(degree + 1) + 0.5) / (degree + 1)
            nodes = np.cos(theta)
            amplitudes = qsp_values(phases, nodes).real
            vandermonde = np.cos(np.outer(theta, np.arange(degree + 1)))
            recovered = (2 / (degree + 1)) * (vandermonde.T @ amplitudes)
            recovered[0] *= 0.5
            bound = float(np.sum(np.abs(recovered - coefs)))
            if bound < 5e-11:
                return phases.tolist(), bound
    raise RuntimeError(f"Phase solve failed at degree {degree}")


def evaluate_success(cosine, sine, x):
    """Successful block of one short step after one oblivious AA round."""
    amplitude = (chebval(x, cosine) - 1j * chebval(x, sine)) / 2
    return amplitude * (3 - 4 * np.abs(amplitude) ** 2)


def projected_angle_table(cosine_phases, sine_phases):
    degree = max(len(cosine_phases), len(sine_phases)) - 1
    angles = np.zeros((degree + 1, 4))
    for parity, phases in enumerate((cosine_phases, sine_phases)):
        shifts = np.full(len(phases), -np.pi / 2)
        shifts[[0, -1]] = -np.pi / 4
        angles[:len(phases), 2 * parity] = shifts + phases
        angles[:len(phases), 2 * parity + 1] = shifts - phases
    return angles


def compiled_scalar_block(cosine_phases, sine_phases, x):
    """Evaluate the two-selector circuit in the signed two-dimensional signal planes."""
    angles = projected_angle_table(cosine_phases, sine_phases)
    y = np.sqrt(np.maximum(0, 1 - x * x))
    branch_outputs = []
    for branch in range(4):
        degree = len((cosine_phases, sine_phases)[branch // 2]) - 1
        top = np.full_like(x, np.exp(1j * angles[0, branch]), dtype=complex)
        bottom = np.zeros_like(top)
        for index in range(1, degree + 1):
            factor = np.exp(1j * angles[index, branch])
            top, bottom = ((x * top + y * bottom) * factor,
                           (y * top - x * bottom) * factor.conjugate())
        branch_outputs.append(top * 1j**degree)
    base = (branch_outputs[0] + branch_outputs[1]
            - 1j * (branch_outputs[2] + branch_outputs[3])) / 4
    return base * (3 - 4 * np.abs(base)**2)


def _basis_inventory(gates):
    """Use the LCHS basis convention, without assuming a multi-clean ladder."""
    c = Counter(name for name, _, _ in gates)
    return Counter(rz=c["rz"], t=c["t"] + c["tdg"], cx=c["cx"],
                   clifford_other=sum(c.values()) - c["rz"] - c["t"]
                                  - c["tdg"] - c["cx"])


@lru_cache(maxsize=None)
def _projected_branch(projector_bits, branch, angle):
    """Phase on projector=|0...0>, selected by the two outer QSVT bits.

    A P0 on one projector bit, controlled by the remaining projector bits
    and the two selectors, applies exactly that conditional scalar phase.
    This avoids an additional clean flag while preserving relative phases.
    """
    if angle == 0.:
        return Counter()
    positive = int(bool(branch & 1)) + int(bool(branch & 2))
    negative = projector_bits - 1 + 2 - positive
    gates, _ = lower_to_basis(controlled_source("P0", positive, negative, (angle,)))
    return _basis_inventory(gates)


def projector_phase(projector_bits, angles=None):
    """Count an explicit four-branch projected phase with one clean work bit.

    Without actual fitted QSP angles, treat retained outer phase angles as
    arbitrary.  This is a conservative compiled template; no unverified
    exact-angle cancellations are credited to the resource calculation.
    """
    if angles is None:
        angles = (.371, .473, .619, .827)  # Generic non-Clifford sentinels.
    if len(angles) != 4:
        raise ValueError("A two-selector projected phase requires four angles")
    counts = Counter()
    for branch, angle in enumerate(angles):
        counts.update(_projected_branch(projector_bits, branch, float(angle)))
    return counts


@lru_cache(maxsize=None)
def _reflection_inventory(bits):
    """Reflection about |0...0> using one-clean MCZ and target X echoes."""
    gates, _ = lower_to_basis(controlled_source("Z", 0, bits - 1, ()))
    result = _basis_inventory(gates)
    result["clifford_other"] += 2  # Conjugate target Z by X for zero state.
    return result


def outer_inventory(degree, projector_bits, angle_table=None):
    # Cosine and sine alternate the same U/Udag schedule except for the last
    # query, which is controlled by parity. Sign-conjugate phase schedules
    # extract each real part using the second selector.
    # One OAA round executes the base circuit three times and reflects
    # twice on its a+2 postselected ancillas. Each reflection uses a
    # multi-controlled X, conjugated with H on its target and X on all flags.
    if projector_bits < 1 or degree < 1:
        raise ValueError("Invalid projector width or QSP degree")
    phase_counts = Counter()
    if angle_table is None:
        # The last parity step is absent in one of the cosine/sine branches.
        cos_steps = degree + 1 - (degree % 2)
        sin_steps = degree + (degree % 2)
        generic = (.371, .473, .619, .827)
        for branch, angle in enumerate(generic):
            multiplicity = cos_steps if branch < 2 else sin_steps
            phase_counts.update({key: multiplicity * value for key, value in
                                 _projected_branch(projector_bits, branch, angle).items()})
    else:
        table = np.asarray(angle_table, dtype=float)
        if table.shape != (degree + 1, 4):
            raise ValueError("Expected a (degree+1, 4) phase-angle table")
        for angles in table:
            phase_counts.update(projector_phase(projector_bits, tuple(angles)))
    counts = Counter({key: 3 * value for key, value in phase_counts.items()})
    reflection_bits = projector_bits + 2
    counts.update({key: 2 * value for key, value in
                   _reflection_inventory(reflection_bits).items()})
    counts["clifford_other"] += 3 * (4 + int(degree % 2 == 0))
    counts.update(oracle=3 * (degree - 1), oracle_controlled=3)
    # Projected phases and OAA reflections share a single, restored work bit.
    return dict(counts), 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("alpha", type=float)
    parser.add_argument("--step", type=float, default=0.9090100684746815 / 66)
    parser.add_argument("--budget", type=float, default=1e-4 / 66)
    parser.add_argument("--projector-bits", type=int, default=6)
    parser.add_argument("--fit", action="store_true")
    args = parser.parse_args()
    design, cosine, sine = coefficients(args.alpha, args.step, args.budget)
    x = np.linspace(-1, 1, 100001)
    simulated = evaluate_success(cosine, sine, x)
    gates, scratch = outer_inventory(design.degree, args.projector_bits)
    output = {**asdict(design),
              "error_sampled": float(np.max(np.abs(simulated - np.exp(-1j * args.alpha * args.step * x)))),
              "min_success_per_step_sampled": float(np.min(np.abs(simulated) ** 2)),
              "cosine_coefficients": cosine.tolist(), "sine_coefficients": sine.tolist(),
              "outer_gate_inventory_per_step": gates, "outer_scratch_qubits": scratch}
    if args.fit:
        output["cosine_phases"], output["cosine_phase_fit_uniform_error"] = fit_phases(cosine)
        output["sine_phases"], output["sine_phase_fit_uniform_error"] = fit_phases(sine)
        phases = np.asarray(output["cosine_phases"]), np.asarray(output["sine_phases"])
        compiled = compiled_scalar_block(*phases, x)
        output["scalar_circuit_check"] = float(np.max(np.abs(compiled - simulated)))
        assert output["scalar_circuit_check"] < 1e-10
        output["outer_gate_inventory_per_step"], _ = outer_inventory(
            design.degree, args.projector_bits, projected_angle_table(*phases))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
