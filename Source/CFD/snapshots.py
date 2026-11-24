"""Time-linear FEM snapshot interpolation used in the current Figure 4 source."""
import numpy as np


def interpolate_snapshots(cfd_times, cfd_fields, target_times):
    """Interpolate on saved times. No extrapolation; returns (fields, audit)."""
    cfd_times = np.asarray(cfd_times)
    cfd_inputs = np.asarray(cfd_fields)
    cfd_targets = np.asarray(target_times)
    fields, interpolation = [], []
    for target_time in cfd_targets:
        # The first saved time needs no left-hand neighbor or interpolation.
        if target_time == cfd_times[0]:
            fields.append(cfd_inputs[0].astype(float))
            interpolation.append({
                'left': float(cfd_times[0]),
                'target': float(target_time),
                'right': float(cfd_times[0]),
                'right_weight': 0.0,
            })
            continue
        right = int(np.searchsorted(cfd_times, target_time)); left = right - 1
        assert 0 <= left < right < len(cfd_times)
        weight = float((target_time-cfd_times[left])/(cfd_times[right]-cfd_times[left]))
        assert 0 <= weight <= 1
        fields.append((1-weight)*cfd_inputs[left].astype(float)+weight*cfd_inputs[right].astype(float))
        interpolation.append({'left':float(cfd_times[left]), 'target':float(target_time),
                              'right':float(cfd_times[right]), 'right_weight':weight})
    return np.stack(fields), interpolation
