"""Phase-independent force statistics over complete shedding cycles."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.polynomial.legendre import leggauss
from scipy.interpolate import CubicSpline

METRICS = ('mean_CD', 'rms_CL', 'St')


def write_json(path, value):
    """Atomically publish a small report so readers never see partial JSON."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def relative_changes(previous, current):
    """Relative change with the newer (finer) result as denominator."""
    return {key: abs(current[key] - previous[key]) / max(abs(current[key]), 1e-12)
            for key in METRICS}


def cycle_metrics(records, stationarity_tol=.001):
    """Use upward CL=0 crossings and six complete cycles, excluding invalid t=0.

    Cubic interpolation removes sample-index quantization from periods. Exact
    spline integrals give mean CD; four-point Gauss quadrature integrates the
    squared cubic CL on each time interval exactly. No signal smoothing is used.
    Stability requires both adjacent three-cycle blocks and the individual last
    three cycles to agree within the relative stationarity tolerance.
    """
    if not np.isfinite(stationarity_tol) or stationarity_tol <= 0:
        raise ValueError('stationarity_tol must be positive and finite')
    frame = pd.DataFrame(records).copy()
    if 'pressure_valid' in frame:
        frame = frame[frame.pressure_valid.astype(str).str.lower().eq('true')]
    frame = frame.dropna(subset=['time', 'C_L', 'C_D'])
    result = dict(stationary=False, reason='Need six complete shedding cycles',
                  stationarity_tolerance=stationarity_tol, cycles=[])
    if len(frame) < 8:
        return result
    t, cl, cd = (frame[key].to_numpy(dtype=float) for key in ('time', 'C_L', 'C_D'))
    if not all(np.isfinite(v).all() for v in (t, cl, cd)) or np.any(np.diff(t) <= 0):
        raise ValueError('Forces must be finite with strictly increasing times')
    lift, drag = CubicSpline(t, cl), CubicSpline(t, cd)
    roots = lift.roots(extrapolate=False)
    roots = roots[(roots >= t[0]) & (roots <= t[-1])]
    roots = roots[lift(roots, 1) > 1e-6]
    nodes, weights = leggauss(4)
    cycles = []
    for left, right in zip(roots[-7:-1], roots[-6:]):
        period = right - left
        # Only plausible developed Re100 shedding, not small startup crossings.
        if not 3 < period < 12:
            continue
        breaks = np.r_[left, t[(t > left) & (t < right)], right]
        mid, half = (breaks[:-1] + breaks[1:])/2, np.diff(breaks)/2
        points = mid[:, None] + half[:, None]*nodes
        rms = np.sqrt(np.sum(half*(lift(points)**2 @ weights))/period)
        extrema = lift.derivative().roots(extrapolate=False)
        extrema = extrema[(extrema >= left) & (extrema <= right)]
        vals = lift(np.r_[left, extrema, right])
        cycles.append(dict(start=float(left), end=float(right),
                           mean_CD=float(drag.integrate(left, right)/period),
                           mean_CL=float(lift.integrate(left, right)/period),
                           rms_CL=float(rms), St=float(1/period),
                           amplitude_CL=float((vals.max()-vals.min())/2)))
    result['cycles'] = cycles
    if len(cycles) < 6:
        return result
    old = {key: float(np.mean([r[key] for r in cycles[:3]])) for key in METRICS}
    new = {key: float(np.mean([r[key] for r in cycles[-3:]])) for key in METRICS}
    drift = relative_changes(old, new)
    spread = {key: float(np.ptp([r[key] for r in cycles[-3:]]) / max(abs(new[key]), 1e-12))
              for key in METRICS}
    result.update(new)
    result.update(window=[cycles[-3]['start'], cycles[-1]['end']],
                  previous_block=old, block_relative_change=drift,
                  last_cycles_relative_range=spread,
                  mean_CL=float(np.mean([r['mean_CL'] for r in cycles[-3:]])),
                  amplitude_CL=float(np.mean([r['amplitude_CL'] for r in cycles[-3:]])))
    result['stationary'] = max(*drift.values(), *spread.values()) <= stationarity_tol
    result['reason'] = 'Settled cycle statistics' if result['stationary'] else 'Cycle statistics still evolving'
    return result


def sweep_verdict(rows, tolerance=.01):
    """Keep adjacent convergence distinct from adequacy of the original reference."""
    result = dict(status='incomplete', reference_within_tolerance=False)
    if len(rows) < 3:
        return result
    if not all(r['metrics']['stationary'] for r in rows):
        result['status'] = 'unsettled_periodic_statistics'
        return result
    adjacent = relative_changes(rows[-2]['metrics'], rows[-1]['metrics'])
    baseline = relative_changes(rows[0]['metrics'], rows[-1]['metrics'])
    result.update(status='within_tolerance' if max(adjacent.values()) <= tolerance else 'needs_further_refinement',
                  last_pair_relative_change=adjacent,
                  reference_to_finest_relative_change=baseline,
                  reference_within_tolerance=max(baseline.values()) <= tolerance)
    return result
