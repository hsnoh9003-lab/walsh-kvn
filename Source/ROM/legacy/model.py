"""Optional conversion from normalized drift to physical-coordinate drift."""

import numpy as np


def raw_drift(model, state):
    # Chain rule: da/dt = scales * f(a/scales); columns are independent states.
    return (
        model.drift(np.asarray(state)/model.scales[:, None])*model.scales[:, None]
        if np.asarray(state).ndim == 2
        else model.drift(np.asarray(state)/model.scales)*model.scales
    )
