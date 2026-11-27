"""Frozen spatial representation with either the deployed or candidate drift."""
from pathlib import Path
from dataclasses import dataclass
import numpy as np
ROOT=Path(__file__).resolve().parent

@dataclass
class Model:
    coefficients:np.ndarray
    powers:np.ndarray
    scales:np.ndarray
    modes:np.ndarray
    base:np.ndarray
    mask:np.ndarray
    times:np.ndarray
    physical_coefficients:np.ndarray

    def drift(self,state):
        x,y,z=state;c=self.coefficients
        return np.array([(c[1,0]+c[6,0]*z)*x+(c[2,0]+c[8,0]*z)*y,
                         (c[1,1]+c[6,1]*z)*x+(c[2,1]+c[8,1]*z)*y,
                         (c[3,2]+c[9,2]*z)*z+(c[4,2]*x+c[5,2]*y)*x+c[7,2]*y*y])

    def field_error(self,estimate,reference):
        ref=self.base[self.mask]+np.asarray(reference)@self.modes[:,self.mask]
        delta=(np.asarray(estimate)-reference)@self.modes[:,self.mask]
        return np.linalg.norm(delta,axis=-1)/np.linalg.norm(ref,axis=-1)

def load(name='deployed'):
    d=np.load(ROOT/'inputs/reported_model_scaling.npz');physical=d['coefficients'].copy()
    if name=='candidate':
        candidate=np.load(ROOT/'inputs/candidate_K2.npz')
        assert np.array_equal(candidate['powers'],d['powers'])
        physical=candidate['physical_coefficients'].copy()
        # The first two equations are unchanged apart from fitting roundoff.
        assert np.max(abs(physical[:,:2]-d['coefficients'][:,:2]))<1e-14
    scales=d['scales'];powers=d['powers']
    coefficients=physical*np.prod(scales[None,:]**powers,axis=1)[:,None]/scales
    return Model(coefficients,powers,scales,scales[:,None]*d['modes'],d['base_field'],
                 np.tile(d['fluid_mask'].ravel(),2),d['window'],physical)

def raw_drift(model,state):
    return model.drift(np.asarray(state)/model.scales[:,None])*model.scales[:,None] if np.asarray(state).ndim==2 else model.drift(np.asarray(state)/model.scales)*model.scales
