"""Fit the reflection-constrained SINDy model in training-RMS coordinates."""

import json

import numpy as np

from Source.data_locations import shared_data, source_file
from . import identification as ident

# RIDGE: penalty after feature-column scaling.
RIDGE = 1e-4


def fit(raw, t, degree, threshold, observed=None, derivative_mask=None):
    # observed selects states for RMS scaling; derivative_mask selects FD4 rows.
    if observed is None:
        observed = np.ones(len(raw), bool)
    # u_j = a_j / scale_j, with scale_j = sqrt(mean(a_j^2)) on observed states.
    scale = np.sqrt(np.mean(raw[observed]**2, axis=0))
    assert np.all(scale > 0)
    u, tt, du = ident.differentiate(raw/scale, t)
    if derivative_mask is not None:
        u, du = u[derivative_mask], du[derivative_mask]

    # p: monomial exponents; A: polynomial library evaluated at retained states.
    p = ident.powers(3, degree)
    A = ident.features(u, p)
    # Reflection sends (u1, u2, u3) to (-u1, -u2, u3).
    signs = np.array([-1, -1, 1])
    term_sign = np.prod(signs[None, :]**p, axis=1)
    allowed = (term_sign[:, None] == signs[None, :])
    allowed[0] = False
    # active: retained term/equation pairs; C: coefficients in RMS coordinates.
    active = allowed.copy()
    C = np.zeros_like(active, dtype=float)
    for iteration in range(20):
        C[:] = 0
        for j in range(3):
            mask = active[:, j]
            if not mask.any():
                continue
            # aa: active library with unit column norms; s restores its units.
            a = A[:, mask]
            s = np.linalg.norm(a, axis=0)
            s[s == 0] = 1
            aa = a/s
            # min ||aa*q-du_j||^2 + RIDGE*||q||^2; C_j = q/s.
            C[mask, j] = np.linalg.lstsq(
                np.vstack([aa, np.sqrt(RIDGE)*np.eye(aa.shape[1])]),
                np.r_[du[:, j], np.zeros(aa.shape[1])],
                rcond=1e-12,
            )[0]/s
        new = active & (np.abs(C) >= threshold)
        if np.array_equal(new, active):
            break
        active = new

    # da_j/dt = sum_k C_kj * scale_j/prod_i(scale_i^p_ki) * a^p_k.
    physical = C*scale[None, :]/np.prod(scale[None, :]**p, axis=1)[:, None]
    used = np.any(C != 0, axis=1)
    maxdegree = int(p[used].sum(axis=1).max()) if used.any() else 0
    higher = int(np.count_nonzero(C[p.sum(axis=1) > 2]))
    return dict(
        p=p, C=C, physical_C=physical, scale=scale,
        degree=degree, threshold=threshold,
        allowed_terms=int(allowed.sum()), active_terms=int(np.count_nonzero(C)),
        active_degree=maxdegree, active_above_quadratic=higher,
    )


def __getattr__(name):
    # Compatibility for callers that used the former eagerly resolved ROOT.
    if name == 'ROOT':
        return shared_data('CFD_ROM', 'Model')
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == '__main__':
    d = np.load(source_file('Data/Model/baseline_inputs.npz'))
    model = fit(d['training_coordinates'], d['training_times'], 2, .03)
    selected = np.load(source_file('Data/Model/candidate_K2.npz'))
    error = float(np.max(np.abs(model['physical_C']-selected['physical_coefficients'])))
    assert error < 1e-12 and model['active_terms'] == 12
    print(json.dumps(dict(
        retained_coefficients=12, maximum_coefficient_difference=error, threshold=.03,
    )))
