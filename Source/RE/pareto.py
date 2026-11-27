"""Compute the four-objective architecture Pareto frontier.

Minimize physical-qubit count, effective core error per consumed T state,
factory-output error, and T-state consumption interval.

Error objectives are represented by negative log-survival probabilities,
which preserve their ordering and provide numerical stability.
Candidates with supply waiting remain eligible for Pareto filtering.
"""

import numpy as np


def pareto_indices_4d(Q, neg_log_s_core, neg_log_s_MSF, tau_T):
    """Exact four-objective nondomination, O(n log^2 n), no tolerance.

    Sort ascending by (Q, -log(s_core), -log(1-e_MSF), tau_T), then maintain the smallest
    seen tau over the 2D prefix (core <= x, magic <= y) via a sparse,
    offline Fenwick-of-Fenwick tree. Sorting guarantees Q <= current Q.
    The earliest exact tie is retained (deterministic source-row order).
    """
    # All four objectives are minimized.
    # Sorting first by Q ensures that previously processed candidates have
    # qubit counts no greater than the current candidate.
    # The tree query then checks the remaining two error coordinates and tau_T.
    arrays = [
        np.asarray(v, dtype=np.float64)
        for v in (Q, neg_log_s_core, neg_log_s_MSF, tau_T)
    ]
    n = len(arrays[0])
    if any(len(v) != n for v in arrays):
        raise ValueError("Four Pareto objective lengths differ")
    if any(not np.isfinite(v).all() for v in arrays):
        raise ValueError("Nonfinite Pareto objective")
    if not n:
        return np.zeros(0, dtype=np.int64)
    Q, neg_log_s_core, neg_log_s_MSF, tau_T = arrays
    order = np.lexsort((np.arange(n), tau_T, neg_log_s_MSF, neg_log_s_core, Q))
    xs = np.unique(neg_log_s_core)
    ys = np.unique(neg_log_s_MSF)
    ix = np.searchsorted(xs, neg_log_s_core).astype(np.int32) + 1
    iy = np.searchsorted(ys, neg_log_s_MSF).astype(np.int32) + 1
    # Precompute the secondary Fenwick coordinate list at each x tree node.
    nodes = [[] for _ in range(len(xs) + 1)]
    for a, b in zip(ix, iy):
        j = int(a)
        while j < len(nodes):
            nodes[j].append(int(b))
            j += j & -j
    # Replace error values with their sorted ranks for tree indexing.
    # Coordinate compression preserves exact ordering; it does not round errors
    # or merge approximately equal values.
    ranks = [None] * (len(xs) + 1)
    tree = [None] * (len(xs) + 1)
    for j in range(1, len(nodes)):
        ranks[j] = np.unique(np.asarray(nodes[j], dtype=np.int32))
        tree[j] = np.full(len(ranks[j]) + 1, np.inf, dtype=np.float64)
    accepted = []
    # Query the smallest previously observed tau_T among candidates whose core
    # and MSF error coordinates are both no greater than the current values.
    # If that interval is also no greater, the current row is dominated or
    # exactly duplicated.
    # For exact duplicates, retain the earliest source row.
    for i in order:
        a = int(ix[i])
        b = int(iy[i])
        t = tau_T[i]
        best = np.inf
        j = a
        while j > 0:
            k = int(np.searchsorted(ranks[j], b, side="right"))
            ar = tree[j]
            while k > 0:
                if ar[k] < best:
                    best = ar[k]
                k -= k & -k
            j -= j & -j
        if best <= t:
            continue
        accepted.append(int(i))
        j = a
        while j < len(nodes):
            k = int(np.searchsorted(ranks[j], b, side="left")) + 1
            ar = tree[j]
            while k < len(ar):
                if t < ar[k]:
                    ar[k] = t
                k += k & -k
            j += j & -j
    return np.asarray(sorted(accepted), dtype=np.int64)


def architecture_frontier(frame):
    """Return positional indices of nondominated rows in the supplied frame.

    Any physical-qubit cap must already have been applied by the caller.
    The returned indices are row positions for iloc, not DataFrame index labels.
    """
    if frame.empty:
        raise ValueError("No architecture meets qubit budget")
    neg_log_s_core = -frame.log_core_survival_per_T.to_numpy(dtype=float)
    neg_log_s_MSF = -frame.log_magic_survival_per_T.to_numpy(dtype=float)
    if np.any(neg_log_s_core < 0) or np.any(neg_log_s_MSF < 0):
        raise ValueError("Negative stochastic-error exponent")
    return pareto_indices_4d(
        frame.physical_qubits.to_numpy(dtype=float),
        neg_log_s_core,
        neg_log_s_MSF,
        frame.service_period_s.to_numpy(dtype=float),
    )
