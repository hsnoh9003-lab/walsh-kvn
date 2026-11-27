"""Four-quantity assessment of existing FEM force histories.

Keep the original three-quantity run protocol and its provenance unchanged.
Recompute statistics from saved forces for the revised 0.5% assessment.
"""
import numpy as np
import pandas as pd
from numpy.polynomial.legendre import leggauss
from scipy.interpolate import CubicSpline

from Source.CFD.refinement_metrics import cycle_metrics

METRICS = ('mean_CD', 'rms_CD_fluct', 'rms_CL', 'St')


def relative_changes(previous, current):
    """Return relative differences, normalized by the finer result."""
    return {key: abs(current[key] - previous[key]) / max(abs(current[key]), 1e-12)
            for key in METRICS}


def cycle_statistics(records, stationarity_tol=.001):
    """Add mean-subtracted drag RMS and check stationarity of all four quantities.

    Retain the original CL cycle boundaries. Subtract each cycle's mean drag
    before integrating its squared fluctuation. Four-point Gauss quadrature
    integrates the squared cubic interpolant exactly on each sample interval.
    Report the average of the last three cycle RMS values, not a pooled RMS.
    """
    result = cycle_metrics(records, stationarity_tol)
    if not result['cycles']:
        return result
    frame = pd.DataFrame(records).copy()
    if 'pressure_valid' in frame:
        frame = frame[frame.pressure_valid.astype(str).str.lower().eq('true')]
    frame = frame.dropna(subset=['time', 'C_D', 'C_L'])
    times = frame.time.to_numpy(dtype=float)
    drag = CubicSpline(times, frame.C_D.to_numpy(dtype=float))
    nodes, weights = leggauss(4)
    for cycle in result['cycles']:
        start, end = cycle['start'], cycle['end']
        edges = np.r_[start, times[(times > start) & (times < end)], end]
        half_width = np.diff(edges)/2
        midpoint = (edges[:-1] + edges[1:])/2
        points = midpoint[:, None] + half_width[:, None]*nodes
        fluctuation = drag(points) - cycle['mean_CD']
        variance = np.sum(half_width*(fluctuation**2 @ weights))/(end-start)
        cycle['rms_CD_fluct'] = float(np.sqrt(variance))
    if len(result['cycles']) < 6:
        return result
    previous = {key: float(np.mean([c[key] for c in result['cycles'][:3]]))
                for key in METRICS}
    current = {key: float(np.mean([c[key] for c in result['cycles'][-3:]]))
               for key in METRICS}
    drift = relative_changes(previous, current)
    spread = {key: float(np.ptp([c[key] for c in result['cycles'][-3:]])
                        / max(abs(current[key]), 1e-12)) for key in METRICS}
    result.update(current)
    result.update(previous_block=previous, block_relative_change=drift,
                  last_cycles_relative_range=spread)
    result['stationary'] = max(*drift.values(), *spread.values()) <= stationarity_tol
    result['reason'] = 'Settled cycle statistics' if result['stationary'] else 'Cycle statistics still evolving'
    return result


def late_window_statistics(records, start=60., end=90., stationarity_tol=.001):
    """Compare the last three complete cycles inside the published late window.

    Data beyond end are excluded before interpolation. Earlier cycles are used
    only for the stationarity diagnostic, which is not a finite-window pass gate.
    """
    if not np.isfinite(start) or not np.isfinite(end) or not 0 <= start < end:
        raise ValueError('Require finite 0 <= start < end')
    frame = pd.DataFrame(records)
    frame = frame[frame.time <= end].copy()
    statistics = cycle_statistics(frame, stationarity_tol)
    if 'window' not in statistics or statistics['window'][0] < start:
        raise ValueError('Need three complete shedding cycles in the comparison window')
    statistics['comparison_window'] = [float(start), float(end)]
    return statistics
