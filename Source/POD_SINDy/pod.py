"""Spatial POD pair, modal projection, and radial shift closure used by the replay."""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares


def reflect(fields):
    # fields: (time, velocity component, y, x); reflection reverses y and v.
    reflected = fields[:, :, ::-1, :].copy()
    reflected[:, 1] *= -1
    return reflected


def decompose(fields):
    reflected = reflect(fields)
    shape = (len(fields), -1)
    # Symmetric/antisymmetric parts: (u+Ru)/2 and (u-Ru)/2.
    return (
        ((fields + reflected) / 2).reshape(shape),
        ((fields - reflected) / 2).reshape(shape),
    )


def fit_oscillatory_pair(times, fields, scale_mode='legacy_cubic_fit'):
    """Keep the existing fixed linear coordinates to isolate the shift change.

    The retained preliminary cubic-trajectory fit determines a constant scale;
    it is an existing physical assumption, not an unconstrained discovery.
    """
    _, antisymmetric = decompose(fields)
    _, singular, vt = np.linalg.svd(antisymmetric, full_matrices=False)
    # modes: first two antisymmetric spatial POD directions; pod: their coordinates.
    modes = vt[:2].T
    pod = antisymmetric @ modes
    eig, vectors = np.linalg.eigh(pod.T @ pod / len(pod))
    assert eig.min() > 0
    # Whitening gives mean(ab*ab.T) = I/2 before the final constant rescaling.
    W = vectors @ np.diag(1 / np.sqrt(2 * eig)) @ vectors.T
    ab = pod @ W
    if np.mean(np.diff(np.unwrap(np.arctan2(ab[:, 1], ab[:, 0])))) < 0:
        W[:, 1] *= -1
        ab = pod @ W
    # Transform the spatial pair inversely so ab*pair.T reconstructs the POD field.
    pair = modes @ np.linalg.inv(W).T
    np.testing.assert_allclose(ab @ pair.T, pod @ modes.T, atol=1e-12)
    rho = np.linalg.norm(ab, axis=1)

    def residual(logs):
        # Optional preliminary scale fit: drho/dt = rate*rho*(1-(rho/limit)^2).
        rate, limit = np.exp(logs)
        solution = solve_ivp(
            lambda _, y: rate*y*(1-(y/limit)**2),
            (times[0], times[-1]), [rho[0]], t_eval=times,
            method='DOP853', rtol=1e-10, atol=1e-12,
        )
        if not solution.success or solution.y.shape[1] != len(times):
            raise RuntimeError('Constant-scale fit failed')
        return solution.y[0] - rho

    if scale_mode == 'legacy_cubic_fit':
        fit = least_squares(
            residual, np.log([.1, 1.]), max_nfev=500,
            xtol=1e-12, ftol=1e-12, gtol=1e-12,
        )
        assert fit.success
        rate, limit = np.exp(fit.x)
        rate, fit_rmse = float(rate), float(np.sqrt(np.mean(fit.fun**2)))
    elif scale_mode == 'training_rms':
        limit = np.sqrt(np.mean(rho**2))
        rate, fit_rmse = None, None
    else:
        raise ValueError(scale_mode)
    return dict(
        pod_modes=modes, W=W, legacy_limit=float(limit), xy=ab/limit, pair=pair*limit,
        meta=dict(
            scale_mode=scale_mode, preliminary_constant_scale=float(limit),
            preliminary_rate=rate, preliminary_fit_rmse=fit_rmse,
            two_mode_energy_fraction=float(np.sum(singular[:2]**2)/np.sum(singular**2)),
        ),
    )


def project(b, fields):
    # Gram*a = Phi*(u-reference): project onto the possibly non-unit spatial modes.
    return np.linalg.solve(
        b['Phi'] @ b['Phi'].T,
        ((fields.reshape(len(fields), -1)-b['reference']) @ b['Phi'].T).T,
    ).T


def closure(x):
    # Fit a3 = c2*(a1^2+a2^2), constraining the intercept to zero.
    rho2 = np.sum(x[:, :2]**2, axis=1)
    return np.array([0., float(rho2@x[:, 2]/(rho2@rho2))])
