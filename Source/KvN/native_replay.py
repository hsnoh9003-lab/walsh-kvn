"""Portable in-place C++ replay of elementary Clifford, T and Rz gates."""
from pathlib import Path
import ctypes, subprocess
import numpy as np

def run(state,program):
    here=Path(__file__).resolve().parent;lib=here.parents[1]/'Build/KvN/native_replay.so'
    lib.parent.mkdir(parents=True,exist_ok=True)
    if not lib.exists():
        subprocess.run(['g++','-O3','-std=c++17','-shared','-fPIC',str(here/'native_replay.cpp'),'-o',str(lib)],check=True)
    ptr=ctypes.CDLL(str(lib)).replay_native
    ptr.argtypes=[np.ctypeslib.ndpointer(np.complex128,flags='C_CONTIGUOUS'),ctypes.c_size_t]+[
        np.ctypeslib.ndpointer(dtype,flags='C_CONTIGUOUS') for dtype in [np.int32,np.int32,np.int32,np.float64]]+[ctypes.c_int]
    ptr.restype=None
    code={'h':0,'x':1,'cx':2,'rz':3,'s':4,'sdg':4,'t':4,'tdg':4}
    phase={'s':np.pi/2,'sdg':-np.pi/2,'t':np.pi/4,'tdg':-np.pi/4}
    gates=np.array([code[g] for g,q,a in program],np.int32)
    q0=np.array([q[0] for g,q,a in program],np.int32)
    q1=np.array([q[-1] for g,q,a in program],np.int32)
    angles=np.array([a if g=='rz' else phase.get(g,0.) for g,q,a in program],np.float64)
    actual=np.ascontiguousarray(state,dtype=np.complex128).copy()
    ptr(actual,actual.size,gates,q0,q1,angles,len(program));return actual
