"""Frozen spatial representation with either the deployed or candidate drift."""

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path

import numpy as np

from Source.data_locations import shared_data



@dataclass
class Model:
    # coefficients/powers: normalized drift and its monomial exponent table.
    coefficients: np.ndarray
    powers: np.ndarray
    # scales: physical coordinate per normalized unit; modes includes this scale.
    scales: np.ndarray
    modes: np.ndarray
    # base/mask: steady velocity and the flattened two-component fluid mask.
    base: np.ndarray
    mask: np.ndarray
    # times: stored evaluation window; physical_coefficients acts on raw coordinates.
    times: np.ndarray
    physical_coefficients: np.ndarray

    def drift(self, state):
        # state has shape (3,) or (3, N), in normalized coordinates.
        x, y, z = state
        c = self.coefficients
        # Library order: 1, x, y, z, x^2, xy, xz, y^2, yz, z^2.
        # Reflection parity restricts dx,dy to x,y,xz,yz and dz to z,x^2,xy,y^2,z^2.
        return np.array([
            (c[1, 0]+c[6, 0]*z)*x + (c[2, 0]+c[8, 0]*z)*y,
            (c[1, 1]+c[6, 1]*z)*x + (c[2, 1]+c[8, 1]*z)*y,
            (c[3, 2]+c[9, 2]*z)*z + (c[4, 2]*x+c[5, 2]*y)*x + c[7, 2]*y*y,
        ])

    def field_error(self, estimate, reference):
        # Relative field error: ||(estimate-reference)*Phi|| / ||base+reference*Phi||.
        ref = self.base[self.mask] + np.asarray(reference) @ self.modes[:, self.mask]
        delta = (np.asarray(estimate)-reference) @ self.modes[:, self.mask]
        return np.linalg.norm(delta, axis=-1)/np.linalg.norm(ref, axis=-1)


def load(name='candidate', data_dir=None):
    if name not in ('candidate', 'deployed'):
        raise ValueError(f'Unknown model {name!r}; choose "candidate" or "deployed".')
    source = shared_data('CFD_ROM', 'Model') if data_dir is None else Path(data_dir).expanduser().resolve()
    d = np.load(source/'reported_model_scaling.npz')
    physical = d['coefficients'].copy()
    if name == 'candidate':
        candidate = np.load(source/'candidate_K2.npz')
        assert np.array_equal(candidate['powers'], d['powers'])
        physical = candidate['physical_coefficients'].copy()
        # The first two equations are unchanged apart from fitting roundoff.
        assert np.max(abs(physical[:, :2]-d['coefficients'][:, :2])) < 1e-14
    scales = d['scales']
    powers = d['powers']
    # a_j = scales_j*x_j: convert physical polynomial coefficients to dx_j/dt.
    coefficients = physical*np.prod(scales[None, :]**powers, axis=1)[:, None]/scales
    return Model(
        coefficients, powers, scales, scales[:, None]*d['modes'], d['base_field'],
        np.tile(d['fluid_mask'].ravel(), 2), d['window'], physical,
    )


def __getattr__(name):
    if name == 'ROOT':
        return shared_data('CFD_ROM', 'Model')
    # Keep external imports of raw_drift valid without loading legacy helpers by default.
    if name == 'raw_drift':
        return import_module('.legacy.model', __package__).raw_drift
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
