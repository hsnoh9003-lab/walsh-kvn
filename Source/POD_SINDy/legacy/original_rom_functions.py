"""Retained historical basis construction and weighted model-fitting variants."""

import numpy as np
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d

from .. import identification as ident
from ..pod import project, closure
from . import shift_pipeline as pipe


def physical(m, cl):
    # J: linearized three-mode drift at the origin; ev: its stability eigenvalues.
    J = m['C'][1:4].T
    ev = np.linalg.eigvals(J)
    cubic = pipe.averaged_cubic(m, cl)
    okay = (
        any(z.real > 0 and abs(z.imag) > .05 for z in ev)
        and cubic['normalization_applicable']
    )
    return bool(okay), cubic, ev


def make_basis(data):
    # Reflection-restricted POD is our implementation choice, not a mandatory
    # step specified by the PNAS three-mode construction. Matched evidence and
    # full-field controls are in antisymmetry_ablation_20261101/REPORT_KO.txt.
    t = data['times']
    fields = data['fields'].astype(float)
    mask = (t >= 20) & (t <= 80)
    pair = pipe.fit_oscillatory_pair(t[mask], fields[mask], scale_mode='training_rms')
    _, anti = pipe.decompose(fields)
    # xy: normalized oscillatory coordinates; phase: unwrapped orbit angle.
    xy = anti @ pair['pod_modes'] @ pair['W']/pair['legacy_limit']
    phase = np.unwrap(np.arctan2(xy[:, 1], xy[:, 0]))
    late = t >= 65
    # Complete cycles in settled training flow. Pick orientation as above.
    ph = phase[late]
    tl = t[late]
    cs = 2*np.pi*np.arange(np.ceil(ph[0]/(2*np.pi)), np.floor(ph[-1]/(2*np.pi))+1)
    crossings = np.interp(cs, ph, tl)
    bounds = crossings[(crossings >= 70) & (crossings <= 90)]
    assert len(bounds) >= 3
    tt = np.r_[bounds[0], t[(t > bounds[0]) & (t < bounds[-1])], bounds[-1]]
    # Mean field: integral u(t) dt divided by the complete-cycle time interval.
    periodic_mean = trapezoid(
        interp1d(t, fields, axis=0)(tt), tt, axis=0,
    )/(tt[-1]-tt[0])
    base = data['steady'].astype(float).ravel()
    delta = periodic_mean.ravel()-base
    # Unit shift direction: remove the oscillatory POD projection from mean-base.
    shift = delta-pair['pod_modes'] @ (pair['pod_modes'].T @ delta)
    shift /= np.linalg.norm(shift)
    b = {'reference': base, 'Phi': np.vstack([pair['pair'].T, shift])}
    x = project(b, fields)
    return b, x, dict(
        basis_training_case='coarse_dt0.02_train', pair_training_interval=[20, 80],
        mean_interval=[float(bounds[0]), float(bounds[-1])], mean_cycles=len(bounds)-1,
        shift_definition='Orthogonalized periodic mean minus independently solved steady flow on SAME FEM mesh and BC',
        mean_cycle_bounds=bounds.tolist(), initial_radius=float(np.linalg.norm(x[0, :2])),
        phase_frequency=float(1/np.mean(np.diff(bounds))),
    )


def fit_weighted(x, t, ridge, threshold, parity, power):
    # scale: per-coordinate training RMS; xx/dx: normalized states and derivatives.
    scale = np.sqrt(np.mean(x*x, axis=0))
    xx, _, dx = ident.differentiate(x/scale, t)
    p = ident.powers(x.shape[1], 2)
    A = ident.features(xx, p)
    # Residual multiplier: w = max(radius, 0.05)^(-power), normalized to unit RMS.
    weights = 1/np.maximum(np.linalg.norm(xx[:, :2], axis=1), .05)**power
    weights /= np.sqrt(np.mean(weights**2))
    A = A*weights[:, None]
    dx = dx*weights[:, None]
    active = np.ones((len(p), x.shape[1]), bool)
    active[0] = False
    if parity is not False:
        signs = (
            np.array([-1, -1]+[1]*(x.shape[1]-2))
            if parity is True else np.asarray(parity)
        )
        term_signs = np.prod(signs[None, :]**p, axis=1)
        active &= term_signs[:, None] == signs[None, :]
    C = np.zeros_like(active, dtype=float)
    for _ in range(20):
        C[:] = 0
        for j in range(x.shape[1]):
            mask = active[:, j]
            if not mask.any():
                continue
            a = A[:, mask]
            s = np.linalg.norm(a, axis=0)
            s[s == 0] = 1
            aa = a/s
            # min ||aa*q-dx_j||^2 + ridge*||q||^2, followed by coefficient thresholding.
            C[mask, j] = np.linalg.lstsq(
                np.vstack([aa, np.sqrt(ridge)*np.eye(aa.shape[1])]),
                np.r_[dx[:, j], np.zeros(aa.shape[1])], rcond=1e-12,
            )[0]/s
        new = active & (abs(C) >= threshold)
        if np.array_equal(new, active):
            break
        active = new
    # Convert normalized coefficients back to physical modal-coordinate units.
    return dict(
        p=p, C=C*scale[None, :]/np.prod(scale[None, :]**p, axis=1)[:, None],
        canonical_C=C, training_rms_scale=scale, degree=2,
        threshold=threshold, ridge=ridge, reflection_parity=parity,
        early_weight_power=power, nonzero=int(np.count_nonzero(C)),
    )
