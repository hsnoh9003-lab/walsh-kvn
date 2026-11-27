"""Shared polynomial library, finite differences, and guarded ODE prediction."""

import itertools

import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import savgol_filter


def powers(n, degree):
    """Return monomial exponents with shape (terms, n), ordered by total degree."""
    # Each row stores exponents of one monomial, starting with the constant term.
    out = [np.zeros(n, dtype=int)]
    for d in range(1, degree + 1):
        for combo in itertools.combinations_with_replacement(range(n), d):
            out.append(np.bincount(combo, minlength=n))
    return np.array(out)


def features(x, p):
    """Evaluate monomials: (..., coordinates) becomes (..., terms)."""
    # Theta_k(x) = product_j x_j^p_kj; the final axis indexes monomials.
    return np.prod(np.asarray(x)[..., None, :] ** p, axis=-1)


def differentiate(x, t, kind="fd4", paper_endpoint=False):
    """Differentiate finite samples on a strictly increasing, uniform time grid."""
    minimum_samples = {"fd2": 3, "fd4": 5, "fd6": 7, "fd8": 9, "sg9": 9}
    if kind not in minimum_samples:
        raise ValueError(f"Unknown derivative method: {kind!r}")
    x = np.asarray(x)
    t = np.asarray(t, dtype=float)
    if t.ndim != 1 or x.ndim == 0 or x.shape[0] != len(t):
        raise ValueError("t must be one-dimensional and match the first axis of x.")
    required = minimum_samples[kind] + int(kind == "fd4" and paper_endpoint)
    if len(t) < required:
        raise ValueError(f"{kind} requires at least {required} samples.")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(x)):
        raise ValueError("x and t must contain only finite values.")
    steps = np.diff(t)
    if np.any(steps <= 0):
        raise ValueError("t must be strictly increasing.")
    # Allow floating-point roundoff, but reject grids requiring nonuniform stencils.
    dt = float(steps.mean())
    if not np.allclose(steps, dt, rtol=1e-7, atol=0.0):
        raise ValueError(
            "t must be uniformly spaced for finite-difference and Savitzky-Golay derivatives."
        )
    if kind == "fd4":
        # x'(t_i) = (-x_(i+2)+8*x_(i+1)-8*x_(i-1)+x_(i-2))/(12*dt).
        xx = x[2:-2]
        tt = t[2:-2]
        dx = (-x[4:] + 8 * x[3:-1] - 8 * x[1:-3] + x[:-4]) / (12 * dt)
        if paper_endpoint:
            xx, tt, dx = xx[:-1], tt[:-1], dx[:-1]
    elif kind in ["fd6", "fd8"]:
        # w satisfies sum_j w_j*node_j^m = delta_(m,1)/dt.
        k = int(kind[2:]) // 2
        nodes = np.arange(-k, k + 1, dtype=float)
        rhs = np.zeros(2 * k + 1)
        rhs[1] = 1
        w = (
            np.linalg.solve(
                np.vstack([nodes**j for j in range(2 * k + 1)]),
                rhs,
            )
            / dt
        )
        xx = x[k:-k]
        tt = t[k:-k]
        dx = sum(wj * x[j : len(x) - 2 * k + j] for j, wj in enumerate(w))
    elif kind == "fd2":
        xx, tt, dx = x[1:-1], t[1:-1], (x[2:] - x[:-2]) / (2 * dt)
    elif kind == "sg9":
        # Nine-point cubic Savitzky-Golay derivative; discard boundary estimates.
        xx = x[4:-4]
        tt = t[4:-4]
        dx = savgol_filter(x, 9, 3, deriv=1, delta=dt, axis=0)[4:-4]
    else:
        raise ValueError(kind)
    return xx, tt, dx


def simulate_diagnostic(model, x0, t):
    """Numerical ODE with explicit termination reasons.

    t must contain at least two finite, strictly increasing times. Invalid
    input returns (None, info) with reason='invalid_input' before solver dispatch.
    A large initial-to-final growth factor is not an instability criterion.
    The remaining distant guard is a compute safeguard in training-RMS units,
    never a proof of model divergence. No trained coefficients are modified.
    """
    try:
        if any(np.iscomplexobj(value) for value in (x0, t, model["p"], model["C"])):
            raise ValueError("State, times and polynomial model must be real.")
        t = np.asarray(t, dtype=float)
        x0 = np.asarray(x0, dtype=float)
        p, C = np.asarray(model["p"]), np.asarray(model["C"])
        if t.ndim != 1 or len(t) < 2 or not np.all(np.isfinite(t)):
            raise ValueError(
                "t must contain at least two finite times in one dimension."
            )
        if np.any(np.diff(t) <= 0):
            raise ValueError("t must be strictly increasing.")
        if x0.ndim != 1 or not x0.size or not np.all(np.isfinite(x0)):
            raise ValueError("x0 must be a nonempty finite one-dimensional state.")
        if p.ndim != 2 or p.shape[1] != x0.size or C.shape != p.shape:
            raise ValueError(
                "p and C must have matching (monomials, state dimensions) shapes."
            )
        if not np.all(np.isfinite(p)) or not np.all(np.isfinite(C)):
            raise ValueError("Polynomial exponents and coefficients must be finite.")
        if np.any(p < 0) or np.any(p != np.floor(p)):
            raise ValueError("Polynomial exponents must be nonnegative integers.")
        raw_scale = model.get("training_rms_scale", np.maximum(np.abs(x0), 1.0))
        if np.iscomplexobj(raw_scale):
            raise ValueError("training_rms_scale must be real.")
        scale = np.asarray(raw_scale, dtype=float)
        if scale.shape != x0.shape or not np.all(np.isfinite(scale)):
            raise ValueError("training_rms_scale must be finite and match x0.")
    except (KeyError, TypeError, ValueError) as exc:
        return None, dict(
            completed=False, reason="invalid_input", message=str(exc), nfev=0
        )
    active = np.any(C != 0, axis=1)
    p, C = p[active], C[active]
    # scale: physical state per training-RMS unit, used only by the magnitude guard.
    scale = np.maximum(np.abs(scale), 1e-14)
    calls = [0]

    def rhs(_, x):
        calls[0] += 1
        if calls[0] > 300000:
            raise RuntimeError("rhs_evaluation_budget")
        if not np.all(np.isfinite(x)):
            raise RuntimeError("nonfinite_state")
        # dx/dt = Theta(x)*C, restricted to monomials used by any equation.
        return features(x, p) @ C

    def safeguard(_, x):
        return 1e4 - np.linalg.norm(x / scale)

    safeguard.terminal = True
    safeguard.direction = -1
    try:
        with np.errstate(over="raise", invalid="raise"):
            sol = solve_ivp(
                rhs,
                (float(t[0]), float(t[-1])),
                x0,
                t_eval=t,
                method="DOP853",
                rtol=2e-8,
                atol=2e-10,
                events=safeguard,
            )
        complete = bool(sol.success and sol.y.shape[1] == len(t))
        reason = (
            "completed"
            if complete
            else "magnitude_safeguard" if sol.status == 1 else "solver_failure"
        )
        info = dict(
            completed=complete,
            reason=reason,
            status=int(sol.status),
            message=sol.message,
            nfev=sol.nfev,
            last_stored_time=float(sol.t[-1]),
            event_time=float(sol.t_events[0][0]) if len(sol.t_events[0]) else None,
        )
        return sol.y.T if complete else None, info
    except (RuntimeError, FloatingPointError, ValueError) as exc:
        return None, dict(
            completed=False,
            reason=type(exc).__name__,
            message=str(exc),
            nfev=calls[0],
        )


__all__ = ["powers", "features", "differentiate", "simulate_diagnostic"]
