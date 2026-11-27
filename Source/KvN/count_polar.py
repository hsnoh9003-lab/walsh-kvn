"""Count native gates for the 12-coefficient Fourier-Walsh transport circuit.

Sparse mask-to-coefficient dictionaries encode Walsh polynomials on the
radial, angular Fourier-mode and shift-coordinate registers."""

from collections import Counter
from dataclasses import dataclass
import numpy as np
from . import wake_gate_core as core


def multiply_walsh(left, right):
    """Multiply Walsh polynomials using mask XOR for character products."""
    result = {}
    for a, ca in left.items():
        for b, cb in right.items():
            result[a ^ b] = result.get(a ^ b, 0.0) + ca * cb
    return result


def add(*terms):
    """Sum scaled mask-to-coefficient dictionaries and drop exact zeros."""
    result = {}
    for scalar, poly in terms:
        for mask, value in poly.items():
            result[mask] = result.get(mask, 0.0) + scalar * value
    return {mask: value for mask, value in result.items() if value != 0.0}


def product(left, right):
    """Multiply sparse Walsh polynomials and drop exact zeros."""
    return {
        mask: value
        for mask, value in multiply_walsh(left, right).items()
        if value != 0.0
    }


def constant(value):
    return {0: float(value)} if value else {}


def restrict(poly, bits):
    """Fix the selected register bits to one and remove them from parity masks."""
    return add(
        *[
            ((-1.0) ** ((mask & bits).bit_count()), {mask & ~bits: val})
            for mask, val in poly.items()
        ]
    )


@dataclass(frozen=True)
class Register:
    """Uniform coordinate grid encoded by little-endian bits at a register offset."""

    # size: grid points; offset: least-significant bit in the joint register.
    size: int
    offset: int
    # start/spacing use the units of this coordinate (or Fourier-mode index).
    start: float
    spacing: float

    @property
    def n(self):
        return self.size.bit_length() - 1

    def coord(self, anchor=0.0, omitted=None):
        terms = constant(self.start + anchor * self.spacing)
        for bit in range(self.n):
            if bit == omitted:
                continue
            half = self.spacing * (1 << bit) / 2
            terms = add((1, terms), (1, {0: half, 1 << (self.offset + bit): -half}))
        return terms


@dataclass(frozen=True)
class Layer:
    """One axis, matching parity and Fourier harmonic in the transport schedule."""

    axis: str
    axis_span: int = 0
    axis_parity: int = 0
    harmonic: str = "dc"
    k_parity: int = 0


def schedule(stencil, order):
    """Build the second-order spatial and Fourier-mode layer schedule."""
    if stencil != 2:
        raise ValueError("Three-coordinate transport uses stencil=2")
    result = []
    for axis in order:
        distances = (0,) if axis == "k" else (1,)
        for distance in distances:
            for parity in (0,) if axis == "k" else (0, 1):
                result.append(Layer(axis, distance, parity))
                result.extend(
                    Layer(axis, distance, parity, harmonic, k_parity)
                    for harmonic in ("cos", "sin")
                    for k_parity in (0, 1)
                )
    return result


def polar_polynomials(axis, radius, z, c):
    """Return the constant, cos(2 theta) and sin(2 theta) drift polynomials."""
    if axis == "z":
        g, h, j = (c[4, 2] + c[7, 2]) / 2, (c[4, 2] - c[7, 2]) / 2, c[5, 2] / 2
        r2 = product(radius, radius)
        return (
            add((c[3, 2], z), (c[9, 2], product(z, z)), (g, r2)),
            add((h, r2)),
            add((j, r2)),
        )
    a = add((1, constant(c[1, 0])), (c[6, 0], z))
    b = add((1, constant(c[2, 0])), (c[8, 0], z))
    e = add((1, constant(c[1, 1])), (c[6, 1], z))
    d = add((1, constant(c[2, 1])), (c[8, 1], z))
    mean, anisotropy = add((0.5, a), (0.5, d)), add((0.5, a), (-0.5, d))
    symmetric, skew = add((0.5, b), (0.5, e)), add((0.5, e), (-0.5, b))
    if axis == "r":
        return tuple(product(radius, p) for p in (mean, anisotropy, symmetric))
    return skew, symmetric, add((-1, anisotropy))


def coefficient(layer, regs, c, stencil, span1):
    if stencil != 2:
        raise ValueError("Three-coordinate transport uses stencil=2")
    harmonic_index = ("dc", "cos", "sin").index(layer.harmonic)
    r = regs["r"].coord()
    z = regs["z"].coord()
    if layer.axis == "k":
        momentum = regs["k"].coord()
        if harmonic_index:
            momentum = regs["k"].coord(anchor=2 * layer.k_parity + 1, omitted=1)
        p = product(polar_polynomials("k", r, z, c)[harmonic_index], momentum)
    else:
        span = layer.axis_span
        omitted = span.bit_length() - 1
        step = regs[layer.axis].spacing

        def at(shift):
            point = regs[layer.axis].coord(
                anchor=layer.axis_parity * span + shift, omitted=omitted
            )
            return polar_polynomials(
                layer.axis,
                point if layer.axis == "r" else r,
                point if layer.axis == "z" else z,
                c,
            )[harmonic_index]

        if span1 == "face":
            p = add((1 / (2 * step), at(span / 2)))
        else:
            d = 0.5
            p = add((d / (2 * step), at(0)), (d / (2 * step), at(span)))
    # The Fourier sine multiplier on the ordered (k,k+2) pair is -Y/2.
    return add(((1.0, 0.5, -0.5)[harmonic_index], p))


def gray_rank(mask):
    rank = mask
    while mask:
        mask >>= 1
        rank ^= mask
    return rank


def diagonal(poly, dt, target=None):
    groups = {}
    for mask, value in poly.items():
        if target is None:
            if mask == 0:
                continue  # Common global phase.
            wire = mask.bit_length() - 1
            mask ^= 1 << wire
        else:
            wire = target
            assert not mask & (1 << wire)
        groups.setdefault(wire, {})[mask] = value
    gates = []
    for wire, terms in groups.items():
        previous = 0
        for mask in sorted(terms, key=gray_rank):
            changed = previous ^ mask
            while changed:
                bit = changed & -changed
                gates.append(("cx", [bit.bit_length() - 1, wire], None))
                changed ^= bit
            gates.append(("rz", [wire], 2 * dt * terms[mask]))
            previous = mask
        while previous:
            bit = previous & -previous
            gates.append(("cx", [bit.bit_length() - 1, wire], None))
            previous ^= bit
    return gates


def pauli_rotation(poly, word, dt):
    if not word:
        return diagonal(poly, dt)
    basis = []
    for wire, operator in word:
        if operator == "Y":
            basis.append(("sdg", [wire], None))
        basis.append(("h", [wire], None))
    target = word[-1][0]
    basis.extend(("cx", [wire, target], None) for wire, _ in word[:-1])
    return basis + diagonal(poly, dt, target) + core.inverse(basis)


def matching_guard(reg, span, work, flag):
    bit = span.bit_length() - 1
    width = reg.n - bit
    wires = list(range(reg.offset + bit, reg.offset + reg.n))
    shift = [
        (name, [wires[q] if q < width else q for q in qs], angle)
        for name, qs, angle in core.increment(width, work)
    ]
    high = wires[1:]
    flag_gates = core.mcx(high, flag, work)
    return core.inverse(shift), flag_gates, sum(1 << q for q in high)


def compile_layer(layer, regs, c, dt, stencil, span1, work, flags):
    p = coefficient(layer, regs, c, stencil, span1)
    addresses, word = [], []
    if layer.axis != "k":
        reg = regs[layer.axis]
        if layer.axis_span * 2 == reg.size and layer.axis_parity:
            return []
        addresses.append((reg, layer.axis_span, layer.axis_parity))
        word.append((reg.offset + layer.axis_span.bit_length() - 1, "Y"))
    if layer.harmonic != "dc":
        reg = regs["k"]
        if reg.size == 4 and layer.k_parity:
            return []
        addresses.append((reg, 2, layer.k_parity))
        word.append((reg.offset + 1, "X" if layer.harmonic == "cos" else "Y"))
    permute, compute = [], []
    for (reg, span, _), flag in zip([a for a in addresses if a[2]], flags):
        shift, guard, bits = matching_guard(reg, span, work, flag)
        permute.extend(shift)
        compute.extend(guard)
        valid_delta = {0: -0.5, 1 << flag: 0.5}
        p = add((1, p), (1, product(valid_delta, restrict(p, bits))))
    return (
        permute
        + compute
        + pauli_rotation(p, word, dt)
        + core.inverse(compute)
        + core.inverse(permute)
    )


def rz_gate(wire, angle):
    if abs(angle) == np.pi / 4:
        return [("t" if angle > 0 else "tdg", [wire], None)]
    return [("rz", [wire], float(angle))]


def centered_forward_qft(reg):
    """Map samples at theta_j=2*pi*j/N to Fourier amplitudes labelled k=j-N/2."""
    plus = []
    for target in reversed(range(reg.n)):
        t = reg.offset + target
        plus.append(("h", [t], None))
        for control in reversed(range(target)):
            c = reg.offset + control
            angle = np.pi / (1 << (target - control + 1))
            plus += rz_gate(t, angle) + rz_gate(c, angle)
            plus += [("cx", [c, t], None)] + rz_gate(t, -angle) + [("cx", [c, t], None)]
    for lower in range(reg.n // 2):
        a, b = reg.offset + lower, reg.offset + reg.n - lower - 1
        plus += [("cx", [a, b], None), ("cx", [b, a], None), ("cx", [a, b], None)]
    return core.inverse(plus) + [("x", [reg.offset + reg.n - 1], None)]


def settings(sizes, rrange, zrange):
    nr, nt, nz = sizes
    bits = [n.bit_length() - 1 for n in sizes]
    assert all(n == 1 << b and b >= 2 for n, b in zip(sizes, bits))
    regs = {
        "k": Register(nt, 0, -nt / 2, 1.0),
        "r": Register(
            nr,
            bits[1],
            rrange[0] + (rrange[1] - rrange[0]) / (2 * nr),
            (rrange[1] - rrange[0]) / nr,
        ),
        "z": Register(
            nz,
            bits[1] + bits[0],
            zrange[0] + (zrange[1] - zrange[0]) / (2 * nz),
            (zrange[1] - zrange[0]) / nz,
        ),
    }
    n = sum(bits)
    # Match the radial compiler's exact one-clean MCX decomposition.  The two
    # matching-validity flags remain distinct from the reusable arithmetic bit.
    work = list(range(n, n + min(1, max(0, max(bits) - 3))))
    flags = list(range(n + len(work), n + len(work) + 2))
    return regs, n, work, flags


def getcounts(gates):
    return Counter(gate[0] for gate in gates)
