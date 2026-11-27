"""Training-only shift extraction and a separate, optional radial closure.

The independent shift extractor takes only velocity snapshots: neither radius
nor a quadratic radius fit determines its spatial direction. This is a centered
symmetric POD direction, not the Noack mean-minus-unstable-steady shift.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from Source.POD_SINDy import identification as ident


def reflect(fields):
    reflected = fields[:, :, ::-1, :].copy()
    reflected[:, 1] *= -1
    return reflected


def decompose(fields):
    reflected = reflect(fields)
    shape = (len(fields), -1)
    return ((fields + reflected) / 2).reshape(shape), ((fields - reflected) / 2).reshape(shape)


def fit_oscillatory_pair(times, fields, scale_mode='legacy_cubic_fit'):
    """Keep the existing fixed linear coordinates to isolate the shift change.

    The retained preliminary cubic-trajectory fit determines a constant scale;
    it is an existing physical assumption, not an unconstrained discovery.
    """
    _, antisymmetric = decompose(fields)
    _, singular, vt = np.linalg.svd(antisymmetric, full_matrices=False)
    modes = vt[:2].T
    pod = antisymmetric @ modes
    eig, vectors = np.linalg.eigh(pod.T @ pod / len(pod))
    assert eig.min() > 0
    W = vectors @ np.diag(1 / np.sqrt(2 * eig)) @ vectors.T
    ab = pod @ W
    if np.mean(np.diff(np.unwrap(np.arctan2(ab[:, 1], ab[:, 0])))) < 0:
        W[:, 1] *= -1
        ab = pod @ W
    pair = modes @ np.linalg.inv(W).T
    np.testing.assert_allclose(ab @ pair.T, pod @ modes.T, atol=1e-12)
    rho = np.linalg.norm(ab, axis=1)

    def residual(logs):
        rate, limit = np.exp(logs)
        solution = solve_ivp(lambda _, y: rate*y*(1-(y/limit)**2),
                             (times[0], times[-1]), [rho[0]], t_eval=times,
                             method='DOP853', rtol=1e-10, atol=1e-12)
        if not solution.success or solution.y.shape[1] != len(times):
            raise RuntimeError('Constant-scale fit failed')
        return solution.y[0] - rho

    if scale_mode == 'legacy_cubic_fit':
        fit = least_squares(residual, np.log([.1, 1.]), max_nfev=500,
                            xtol=1e-12, ftol=1e-12, gtol=1e-12)
        assert fit.success
        rate, limit = np.exp(fit.x)
        rate, fit_rmse = float(rate), float(np.sqrt(np.mean(fit.fun**2)))
    elif scale_mode == 'training_rms':
        limit = np.sqrt(np.mean(rho**2))
        rate, fit_rmse = None, None
    else:
        raise ValueError(scale_mode)
    return dict(pod_modes=modes, W=W, legacy_limit=float(limit),
                xy=ab/limit, pair=pair*limit,
                meta=dict(scale_mode=scale_mode,preliminary_constant_scale=float(limit),
                          preliminary_rate=rate,
                          preliminary_fit_rmse=fit_rmse,
                          two_mode_energy_fraction=float(np.sum(singular[:2]**2)/np.sum(singular**2))))


def fit_independent_shift(fields):
    """Leading POD of centered symmetric velocity; no oscillation coordinates.

    The sign convention is the largest-magnitude spatial entry being positive.
    The temporal coefficient has zero training mean and unit training RMS.
    """
    symmetric, _ = decompose(fields)
    reference = symmetric.mean(axis=0)
    centered = symmetric-reference
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    direction = vt[0].copy()
    if direction[np.argmax(np.abs(direction))] < 0:
        direction *= -1
    coefficient = centered @ direction
    scale = np.sqrt(np.mean(coefficient**2))
    if scale <= 0:
        raise ValueError('No symmetric fluctuation is available for a shift mode')
    shift = direction*scale
    s = coefficient/scale
    np.testing.assert_allclose(s.mean(), 0, atol=1e-12)
    np.testing.assert_allclose(np.mean(s*s), 1, atol=1e-12)
    return reference, shift, s, dict(
        method='Leading POD of centered symmetric velocity snapshots',
        uses_radius_for_direction=False,
        sign_convention='Largest absolute spatial entry is positive',
        coefficient_scale='Zero training mean and unit training RMS',
        spatial_scale=float(scale),
        centered_symmetric_energy_fraction=float(singular[0]**2/np.sum(singular**2)),
        centered_symmetric_first_five_energy_fractions=(singular[:5]**2/np.sum(singular**2)).tolist())


def make_frozen(pair, reference, shift, coordinate, fields):
    Phi = np.vstack([pair['pair'].T, shift])
    # Reflection symmetry makes the shift orthogonal to both oscillatory modes.
    cosines = Phi[:2] @ shift / (np.linalg.norm(Phi[:2], axis=1)*np.linalg.norm(shift))
    assert np.max(np.abs(cosines)) < 1e-12
    return dict(reference=reference, Phi=Phi, pod_modes=pair['pod_modes'],
                W=pair['W'], legacy_limit=np.array(pair['legacy_limit']),
                training_coordinates=np.column_stack([pair['xy'], coordinate]),
                training_mean=fields.reshape(len(fields), -1).mean(axis=0))


def project(frozen, fields):
    symmetric, antisymmetric = decompose(fields)
    xy = antisymmetric @ frozen['pod_modes'] @ frozen['W'] / frozen['legacy_limit']
    shift = frozen['Phi'][2]
    s = (symmetric-frozen['reference']) @ shift / (shift @ shift)
    return np.column_stack([xy, s])


def reconstruct(frozen, coordinates):
    return frozen['reference'] + coordinates @ frozen['Phi']


def fit_affine_closure(coordinates):
    radius_squared = np.sum(coordinates[:, :2]**2, axis=1)
    return np.linalg.lstsq(np.column_stack([np.ones(len(coordinates)), radius_squared]),
                          coordinates[:, 2], rcond=None)[0]


def averaged_cubic(model, closure):
    """Uniform angular average AFTER imposing s=c0+c2*rho^2.

    This is a reduction/closure assumption; it does not assert that the learned
    full quadratic ODE is exactly rotationally invariant or has this manifold.
    """
    c0, c2 = closure
    def coefficient(p, j):
        return model['C'][np.flatnonzero((model['p'] == p).all(axis=1))[0], j]
    L = (coefficient([1,0,0],0)+coefficient([0,1,0],1))/2
    Q = (coefficient([1,0,1],0)+coefficient([0,1,1],1))/2
    W = (coefficient([1,0,0],1)-coefficient([0,1,0],0))/2
    G = (coefficient([1,0,1],1)-coefficient([0,1,1],0))/2
    mu, beta, omega, gamma = L+c0*Q, -c2*Q, W+c0*G, c2*G
    angles = 2*np.pi*np.arange(64)/64
    for rho in [.5, .7, 1.]:
        circle = np.column_stack([rho*np.cos(angles), rho*np.sin(angles),
                                  np.full(len(angles), c0+c2*rho*rho)])
        velocity = ident.features(circle, model['p']) @ model['C']
        radial = np.sum(circle[:,:2]*velocity[:,:2],axis=1)/rho
        angular = (circle[:,0]*velocity[:,1]-circle[:,1]*velocity[:,0])/rho**2
        np.testing.assert_allclose(radial.mean(), mu*rho-beta*rho**3, atol=2e-12)
        np.testing.assert_allclose(angular.mean(), omega+gamma*rho**2, atol=2e-12)
    applicable = bool(mu > 0 and beta > 0)
    return dict(mu=float(mu), beta=float(beta), omega=float(omega),
                gamma_raw=float(gamma), normalization_applicable=applicable,
                R=float(np.sqrt(mu/beta)) if applicable else None,
                gamma_normalized=float(gamma*mu/beta) if applicable else None)


def predict_reduced(model, cubic, closure, initial, times):
    c0, c2 = closure
    def constrained_rhs(_, ab):
        state = np.array([*ab, c0+c2*(ab@ab)])
        return (ident.features(state, model['p']) @ model['C'])[:2]
    reduced = solve_ivp(constrained_rhs, (times[0],times[-1]), initial[:2],
                        t_eval=times, method='DOP853', rtol=1e-10, atol=1e-12)
    results = {}
    if reduced.success and reduced.y.shape[1] == len(times):
        ab = reduced.y.T
        results['closure_2d'] = np.column_stack([ab, c0+c2*np.sum(ab*ab,axis=1)])
    if cubic['normalization_applicable']:
        rho0 = np.linalg.norm(initial[:2])
        theta0 = np.arctan2(initial[1],initial[0])
        mu,beta,omega,gamma = [cubic[k] for k in ['mu','beta','omega','gamma_raw']]
        solution = solve_ivp(lambda _, v: [mu*v[0]-beta*v[0]**3,omega+gamma*v[0]**2],
                             (times[0],times[-1]), [rho0,theta0], t_eval=times,
                             method='DOP853', rtol=1e-11, atol=1e-13)
        if solution.success and solution.y.shape[1] == len(times):
            rho, theta = solution.y
            results['cubic_radius'] = np.column_stack([rho*np.cos(theta),rho*np.sin(theta),c0+c2*rho*rho])
    return results


def closure_diagnostics(frozen, model, closure, observed, fields, times):
    c0, c2 = closure
    r2 = np.sum(observed[:,:2]**2,axis=1)
    residual = observed[:,2] - c0 - c2*r2
    phi_norm = np.linalg.norm(frozen['Phi'][2])
    q = fields.reshape(len(fields), -1)
    state_on_closure = np.column_stack([observed[:,:2],c0+c2*r2])
    velocity = ident.features(state_on_closure,model['p']) @ model['C']
    tangency = velocity[:,2]-2*c2*np.sum(state_on_closure[:,:2]*velocity[:,:2],axis=1)
    _,_,dq = ident.differentiate(q,times)
    equivalent_z = (observed[:,2]-c0)/c2
    diagnostics = dict(
        coefficient_rmse=float(np.sqrt(np.mean(residual**2))),
        physical_closure_residual_relative_to_field_pct=float(100*phi_norm*np.linalg.norm(residual)/np.linalg.norm(q)),
        radial_equivalent_closure_relative_error_pct=float(100*np.linalg.norm(equivalent_z-r2)/np.linalg.norm(equivalent_z)),
        manifold_tangency_defect_physical_rms=float(phi_norm*np.sqrt(np.mean(tangency**2))),
        manifold_tangency_defect_relative_to_field_derivative_pct=float(100*phi_norm*np.linalg.norm(tangency[2:-2])/np.linalg.norm(dq)),
        note='Tangency is evaluated on the proposed manifold at observed a,b, not proof of an invariant manifold.')
    return diagnostics, residual, tangency
