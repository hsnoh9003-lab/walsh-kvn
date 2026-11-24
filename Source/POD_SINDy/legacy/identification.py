"""Retained generic SINDy regression and presentation helpers.

The current parity-constrained fit is implemented in ..fit.
"""

import numpy as np
from scipy.linalg import lstsq

from .. import identification as ident


def labels(p, names):
    return [
        ' '.join(
            name+(f'^{e}' if e > 1 else '')
            for name, e in zip(names, row) if e
        ) or '1'
        for row in p
    ]


def least_squares_scaled(A, b):
    # Column equilibration improves conditioning; returned coefficients retain
    # original feature units, which are also the units used for thresholding.
    scale = np.linalg.norm(A, axis=0)
    scale[scale == 0] = 1
    return lstsq(A/scale, b, cond=1e-12, lapack_driver='gelsy')[0]/scale[:, None]


def stlsq(A, b, threshold, iterations=10):
    """Sequential thresholded least squares (SINDy), no ridge penalty."""
    if b.ndim == 1:
        b = b[:, None]
    # C: feature coefficients, one column per derivative component.
    C = least_squares_scaled(A, b)
    for _ in range(iterations):
        small = np.abs(C) < threshold
        C[small] = 0
        for j in range(b.shape[1]):
            active = ~small[:, j]
            if active.any():
                C[active, j] = least_squares_scaled(A[:, active], b[:, j:j+1])[:, 0]
    return C


def learn(x, t, degree, threshold, kind='fd4'):
    xx, tt, dx = ident.differentiate(x, t, kind)
    p = ident.powers(x.shape[1], degree)
    A = ident.features(xx, p)
    C = stlsq(A, dx, threshold)
    cscale = np.linalg.norm(A, axis=0)
    cscale[cscale == 0] = 1
    return dict(
        p=p, C=C, degree=degree, threshold=threshold, derivative=kind,
        nonzero=int(np.count_nonzero(C)), condition=float(np.linalg.cond(A/cscale)),
        derivative_error_pct=float(100*np.linalg.norm(A@C-dx)/np.linalg.norm(dx)),
    )


def simulate(model, x0, t):
    """Backward-compatible prediction API; use simulate_diagnostic for reasons."""
    return ident.simulate_diagnostic(model, x0, t)[0]


def equations(model, names):
    terms = labels(model['p'], names)
    return [
        f'd{name}/dt = '+' '.join(
            f'{c:+.8g} {term}' for c, term in zip(model['C'][:, j], terms) if c != 0
        )
        for j, name in enumerate(names)
    ]


def serial_model(model):
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in model.items()}
