"""Prescribed Cartesian-Gaussian inputs and continuum ODE characteristic references."""

import csv
import time

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import solve_ivp
from scipy.special import ndtr


def normal(x, mu, sigma):
    """Evaluate a one-dimensional Gaussian density with mean mu and width sigma."""
    # Normal density: exp(-(x-mu)^2/(2*sigma^2)) / (sqrt(2*pi)*sigma).
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (np.sqrt(2 * np.pi) * sigma)


def _finite_vector(value, length, name):
    """Read a fixed-length real configuration vector before quadrature."""
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must contain {length} finite real values.")
    values = np.asarray(value, dtype=float)
    if values.shape != (length,) or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain {length} finite real values.")
    return values


def _positive_integers(value, length, name):
    values = np.asarray(value, dtype=object)
    if values.shape != (length,) or any(
        isinstance(v, (bool, np.bool_))
        or not isinstance(v, (int, np.integer))
        or v <= 0
        for v in values
    ):
        raise ValueError(f"{name} must contain {length} positive integers.")
    return tuple(int(v) for v in values)


def _gaussian_parameters(config):
    """Validate the shared Gaussian and polar-domain parameters."""
    centre = _finite_vector(config["centre_cartesian"], 3, "centre_cartesian")
    sigma = _finite_vector(config["sigma_cartesian"], 3, "sigma_cartesian")
    if np.any(sigma <= 0):
        raise ValueError("sigma_cartesian must be strictly positive.")
    rrange = _finite_vector(config["rrange"], 2, "rrange")
    zrange = _finite_vector(config["zrange"], 2, "zrange")
    if rrange[0] < 0 or rrange[1] <= rrange[0]:
        raise ValueError("rrange must be increasing with a nonnegative lower bound.")
    if zrange[1] <= zrange[0]:
        raise ValueError("zrange must be strictly increasing.")
    return centre, sigma, rrange, zrange


def _positive_normalization(probabilities):
    norm = float(probabilities.sum())
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError(
            "Gaussian probability normalization must be finite and positive."
        )
    return norm


def make_input(config, order=12):
    """Integrate a valid Gaussian over positive integer polar-grid cell counts."""
    # nr/nt/nz: polar-grid cell counts; centre/sigma: Cartesian Gaussian parameters.
    centre, sigma, rrange, zrange = _gaussian_parameters(config)
    nr, nt, nz = _positive_integers(config["grid_r_theta_z"], 3, "grid_r_theta_z")
    (order,) = _positive_integers([order], 1, "order")
    redges = np.linspace(*rrange, nr + 1)
    delta = 2 * np.pi / nt
    tedges = delta * np.arange(nt + 1) - delta / 2
    zedges = np.linspace(*zrange, nz + 1)
    # rs/ts: Gauss-Legendre quadrature nodes within each radial/angular cell.
    nodes, weights = leggauss(order)
    rs = (redges[:-1, None] + redges[1:, None]) / 2 + np.diff(redges)[
        :, None
    ] * nodes / 2
    ts = (tedges[:-1, None] + tedges[1:, None]) / 2 + delta * nodes / 2
    rt = np.zeros((nr, nt))
    for i in range(order):
        r = rs[:, i, None]
        for j in range(order):
            theta = ts[None, :, j]
            # dx*dy = r*dr*dtheta: include the polar Jacobian in cell probability.
            rt += (
                weights[i]
                * weights[j]
                * r
                * normal(r * np.cos(theta), centre[0], sigma[0])
                * normal(r * np.sin(theta), centre[1], sigma[1])
            )
    rt *= np.diff(redges)[:, None] * delta / 4
    # zmass: exact Gaussian CDF differences; p is ordered (z, r, theta).
    zmass = np.diff(ndtr((zedges - centre[2]) / sigma[2]))
    p = zmass[:, None, None] * rt[None, :, :]
    norm = _positive_normalization(p)
    return np.sqrt(p / norm), norm


def reference(model, config, orders, output=None):
    """Propagate Gaussian quadrature nodes over an increasing finite time window."""
    centre, sigma, (r0, r1), (z0, z1) = _gaussian_parameters(config)
    orders = _positive_integers(orders, 3, "orders")
    window = _finite_vector(config["time_window"], 2, "time_window")
    if window[1] <= window[0]:
        raise ValueError("time_window must be strictly increasing.")

    def quad(n, a, b):
        x, w = leggauss(n)
        return (a + b) / 2 + (b - a) * x / 2, (b - a) * w / 2

    # Quadrature approximates the initial Cartesian Gaussian over the polar domain.
    r, wr = quad(orders[0], r0, r1)
    theta, wt = quad(orders[1], 0.0, 2 * np.pi)
    z, wz = quad(orders[2], z0, z1)
    zz, rr, tt = np.meshgrid(z, r, theta, indexing="ij")
    x = rr * np.cos(tt)
    y = rr * np.sin(tt)
    weight = wz[:, None, None] * wr[None, :, None] * wt[None, None, :] * rr
    weight *= (
        normal(x, centre[0], sigma[0])
        * normal(y, centre[1], sigma[1])
        * normal(zz, centre[2], sigma[2])
    )
    norm = _positive_normalization(weight)
    weight = weight.ravel() / norm
    coordinates = np.array([x.ravel(), y.ravel(), zz.ravel()])
    del zz, rr, tt, x, y
    # means/r2: E[(x,y,z)] and E[x^2+y^2] at 67 evaluation times.
    times = np.linspace(*config["time_window"], 67)
    means = np.zeros((67, 3))
    r2 = np.zeros(67)
    path_outside_weight = 0.0
    tick = time.monotonic()
    for begin in range(0, len(weight), 4096):
        end = min(begin + 4096, len(weight))
        initial = coordinates[:, begin:end]
        shape = initial.shape
        sol = solve_ivp(
            lambda t, a: model.drift(a.reshape(shape)).ravel(),
            config["time_window"],
            initial.ravel(),
            method="DOP853",
            rtol=2e-10,
            atol=2e-12,
            t_eval=times,
        )
        assert sol.success
        # values: (coordinate, initial quadrature node, time); local: initial weights.
        values = sol.y.reshape(3, end - begin, 67)
        local = weight[begin:end]
        means += np.einsum("cqt,q->tc", values, local)
        radius2 = values[0] ** 2 + values[1] ** 2
        r2 += np.einsum("qt,q->t", radius2, local)
        # Count crossing either radial boundary or either z boundary at sampled times.
        outside = np.any(
            (radius2 < r0 * r0)
            | (radius2 > r1 * r1)
            | (values[2] < z0)
            | (values[2] > z1),
            axis=1,
        )
        path_outside_weight += float(local[outside].sum())
        if begin % (4096 * 32) == 0:
            print(
                "REFERENCE",
                orders,
                end,
                len(weight),
                round(time.monotonic() - tick, 2),
                flush=True,
            )
    audit = dict(
        orders=list(orders),
        normalization=norm,
        seconds=time.monotonic() - tick,
        sampled_path_outside_domain_weight=path_outside_weight,
        terminal_means=means[-1].tolist(),
    )
    if output:
        with output.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "time",
                    "reference_mean_x",
                    "reference_mean_y",
                    "reference_mean_z",
                    "reference_mean_z_minus_r2",
                ]
            )
            w.writerows(
                [t, *mean, mean[2] - squared]
                for t, mean, squared in zip(times, means, r2)
            )
    return means, audit
