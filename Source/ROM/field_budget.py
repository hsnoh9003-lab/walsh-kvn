"""Terminal ensemble error and expected RMS shot budget."""

import itertools, math
import numpy as np


def finite_observable_ranges(centres, polar=True):
    """Return value and pair-product ranges on a discrete joint support."""
    axes = [np.asarray(axis, dtype=float) for axis in centres]
    assert len(axes) == 3
    assert all(
        axis.ndim == 1 and len(axis) and np.all(np.isfinite(axis)) for axis in axes
    )
    if polar:
        r, theta, z = axes
        assert min(r) >= 0.0
        x = (r[:, None] * np.cos(theta)[None, :]).ravel()
        y = (r[:, None] * np.sin(theta)[None, :]).ravel()
        axes = [x, y, z]
    minima = np.array([axis.min() for axis in axes])
    maxima = np.array([axis.max() for axis in axes])
    product_widths = np.empty((3, 3))
    for i, axis in enumerate(axes):
        for j in range(i, 3):
            if i == j:
                products = axis**2
            elif polar and i == 0 and j == 1:
                products = axes[0] * axes[1]
            else:
                products = np.outer([minima[i], maxima[i]], [minima[j], maxima[j]])
            product_widths[i, j] = product_widths[j, i] = np.ptp(products)
    return maxima - minima, product_widths


def field_budget(raw, implementation_allowance, target_rms, polar=True):
    """Conservative expected squared relative field error of a sample mean."""
    eta = implementation_allowance
    assert 0 <= eta <= 1 and 0 < target_rms <= 1
    widths, product_widths = finite_observable_ranges(raw["grid_centres"], polar=polar)
    mean = np.asarray(raw["outcome_mean"], dtype=float)
    reference_mean = np.asarray(raw["reference_outcome_mean"], dtype=float)
    bias = mean - reference_mean
    covariance = np.asarray(raw["covariance"], dtype=float)
    metric = np.asarray(raw["field_metric"], dtype=float)
    assert mean.shape == reference_mean.shape == (3,)
    for matrix in (covariance, metric):
        assert matrix.shape == (3, 3)
        assert np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-14)
        assert np.linalg.eigvalsh(matrix).min() >= -1e-13
    assert math.isclose(raw.get("probability_sum", 1.0), 1.0, abs_tol=1e-9, rel_tol=0.0)

    # TV bounds the change in an expectation by TV times its range. The
    # convex squared field norm is maximized at a vertex of the bias box.
    bias_bound_sq = max(
        float(displaced @ metric @ displaced)
        for signs in itertools.product((-1, 1), repeat=3)
        for displaced in [bias + eta * widths * np.array(signs)]
    )
    covariance_change = (
        product_widths + np.outer(np.abs(mean), widths) + np.outer(widths, np.abs(mean))
    ) * eta + np.outer(widths, widths) * eta**2
    bias_sq = float(bias @ metric @ bias)
    variance = float(np.trace(metric @ covariance))
    assert bias_sq >= -1e-15 and variance >= -1e-15
    bias_sq = max(0.0, bias_sq)
    variance = max(0.0, variance)
    variance_bound = variance + float(np.sum(np.abs(metric) * covariance_change))
    reported = raw.get(
        "ideal_relative_field_bias", raw.get("deterministic_relative_field_bias")
    )
    if reported is not None:
        assert math.isclose(math.sqrt(bias_sq), reported, rel_tol=1e-8, abs_tol=1e-12)
    reported_variance = raw.get("ideal_per_shot_field_variance")
    if reported_variance is not None:
        assert math.isclose(variance, reported_variance, rel_tol=1e-10, abs_tol=1e-12)
    # Expected squared error <= bias_bound_sq + variance_bound / accepted_shots.
    room = target_rms**2 - bias_bound_sq
    shots = max(1, math.ceil(variance_bound / room)) if room > 0 else None
    if shots is not None:
        assert bias_bound_sq + variance_bound / shots <= target_rms**2 + 1e-16
        if shots > 1:
            assert bias_bound_sq + variance_bound / (shots - 1) > target_rms**2
    result = {
        "field_error_definition": "relative fluid-mask L2 error of the mean velocity field against the common matched ODE ensemble",
        "target_relative_field_rms": target_rms,
        "implementation_TV_allowance": eta,
        "noiseless_deterministic_relative_field_bias": math.sqrt(bias_sq),
        "worst_case_relative_field_bias_bound": math.sqrt(bias_bound_sq),
        "noiseless_relative_field_variance_per_shot": variance,
        "relative_field_variance_per_shot_bound": variance_bound,
        "minimum_accepted_shots_under_rms_bound": shots,
        "noiseless_expected_field_rms_at_selected_shots": (
            math.sqrt(bias_sq + variance / shots) if shots is not None else None
        ),
        "expected_relative_field_rms_bound": (
            math.sqrt(bias_bound_sq + variance_bound / shots)
            if shots is not None
            else None
        ),
        "noiseless_10000_shot_expected_relative_field_rms": math.sqrt(
            bias_sq + variance / 10000
        ),
        "10000_shot_expected_relative_field_rms_bound": math.sqrt(
            bias_bound_sq + variance_bound / 10000
        ),
    }
    if polar:
        result.update(
            readout_observables=["r*cos(theta)", "r*sin(theta)", "z"],
            finite_grid_observable_ranges=widths.tolist(),
            finite_grid_observable_product_ranges=product_widths.tolist(),
        )
    return result
