"""Compatibility imports for historical callers; no numerical code is duplicated."""

from Source.POD_SINDy.pod import reflect, decompose, fit_oscillatory_pair
from Source.POD_SINDy.legacy.shift_pipeline import (
    fit_independent_shift, make_frozen, project, reconstruct, fit_affine_closure,
    averaged_cubic, predict_reduced, closure_diagnostics,
)
