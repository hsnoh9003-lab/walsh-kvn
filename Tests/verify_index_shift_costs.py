"""Reproduce the per-shift Toffoli counts in Supplementary Note 3.

Run from the package root with python -m Tests.verify_index_shift_costs.
The even-width comparison uses the published six-bit gate sequence at six
qubits and the same equal split with borrowed-bit ladders at larger widths.
"""

from collections import Counter

import numpy as np

from Source.KvN import one_clean_synthesis as gates


def borrowed_flip(controls, target, borrowed):
    """Flip the target on all-one controls, restoring the borrowed bits."""
    if len(controls) < 3 or len(borrowed) < len(controls) - 2:
        raise ValueError("The ladder needs at least three controls and k-2 borrowed bits")
    seed = gates.tof(controls[0], controls[1], borrowed[0])
    ladder = []
    for i in range(2, len(controls) - 1):
        ladder += gates.tof(borrowed[i-2], controls[i], borrowed[i-1])
    end = gates.tof(borrowed[len(controls)-3], controls[-1], target)
    sweep = ladder + end + gates.inverse(ladder)
    return seed + sweep + seed + sweep


def split_shift(width):
    """Extend the six-bit B circuit using equal lower and upper halves."""
    if width < 6 or width % 2:
        raise ValueError("This comparison uses even widths of at least six bits")
    half = width // 2
    low = list(range(half))
    high = list(range(half, width))
    work = width
    carry = borrowed_flip(low, work, high)
    upper = [work, *high[:-1]]
    return (carry + borrowed_flip(upper, high[-1], low)
            + gates.borrowed_increment(upper, low) + gates.x(work)
            + carry + gates.borrowed_increment(low, high))


def check_permutation(sequence, width, step):
    """Check every input cell and the final value of the clean work qubit."""
    cells = np.arange(1 << width, dtype=np.uint64)
    values = cells.copy()
    for gate, wires, _ in sequence:
        if gate not in {"x", "cx", "ccx"}:
            raise ValueError(f"Unexpected reversible gate: {gate}")
        active = np.ones_like(cells)
        for control in wires[:-1]:
            active &= (values >> control) & 1
        values ^= active << wires[-1]
    expected = (cells + (1 if step == 1 else (1 << width) - 1)) % (1 << width)
    np.testing.assert_array_equal(values, expected)


if __name__ == "__main__":
    assert split_shift(6) == gates.increment_B(range(6), 6)
    print("register_qubits,shift,Toffoli,T_or_inverse_T,CNOT")
    for width in (6, 8, 10, 12):
        for label, sequence in (("A", gates.increment_A(range(width), width)),
                                ("B", split_shift(width))):
            check_permutation(sequence, width, 1)
            check_permutation(gates.inverse(sequence), width, -1)
            toffolis = Counter(gate for gate, _, _ in sequence)["ccx"]
            elementary = gates.inventory(gates.expand(sequence))
            assert elementary["native_T"] == 7 * toffolis
            print(f"{width},{label},{toffolis},{elementary['native_T']},{elementary['CX']}")
