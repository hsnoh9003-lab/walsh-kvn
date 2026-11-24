"""Sweep the MPS bond dimension at the unchanged polar-grid bit ordering."""
from pathlib import Path
import os,json,time,argparse
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
from scipy.linalg import svd
from Source.data_locations import shared_data
PACKAGE=Path(__file__).resolve().parents[2]
ROOT=shared_data('MPS_KvN_inputs','KvN')

def compress(target,chi):
    n=target.size.bit_length()-1;b=chi.bit_length()-1
    assert chi==1<<b and n>b
    state=target.copy();blocks=[];discarded=[]
    for m in range(n,b+1,-1):
        matrix=state.reshape(2*chi,-1)
        u,s,vh=svd(matrix,full_matrices=False,check_finite=False,lapack_driver='gesdd')
        rank=min(chi,len(s));discarded.append(float(np.dot(s[rank:],s[rank:])))
        blocks.append(dict(m=m,basis=u[:,:rank]))
        state=(s[:rank,None]*vh[:rank,:]).ravel()
        # Near the end the right-hand dimension can be smaller than chi.
        # The short remainder is stored directly as the base state.
        if rank<chi:break
    retained=float(np.dot(state,state));state/=np.sqrt(retained)
    base=state.copy()
    for block in reversed(blocks):
        state=(block['basis']@state.reshape(block['basis'].shape[1],-1)).ravel()
    fidelity=float(abs(np.dot(target,state))**2)
    assert abs(fidelity-retained)<2e-11,(fidelity,retained)
    return state,blocks,base,dict(fidelity=fidelity,total_variation=float(.5*np.sum(abs(state**2-target**2))),
       amplitude_l2=float(np.linalg.norm(state-target)),discarded_norm_squared=discarded,
       bond_dimension=chi,qubits=n,retained_norm_squared=retained,
       method='sequential Schmidt truncation with bond chi; contiguous polar bit ordering')

def main(name):
    source=ROOT/'fresh_late/target_amplitude.npy' if name=='late40' else ROOT/'output_fast'/name/'target_amplitude.npy'
    target=np.load(source);out=PACKAGE/'Build/KvN/bond_sweep'/name;out.mkdir(parents=True,exist_ok=True)
    records=[];tick=time.monotonic()
    for chi in [2,4,8,16,32,64,128,256,512]:
        state,blocks,base,meta=compress(target,chi)
        meta.update(run=name,passed=meta['fidelity']>=.9999,seconds=time.monotonic()-tick)
        records.append(meta);print(json.dumps(meta),flush=True)
        (out/'sweep.json').write_text(json.dumps(records,indent=2)+'\n')
        if meta['passed']:
            np.save(out/'prepared_amplitude.npy',state)
            np.savez_compressed(out/'mps_blocks.npz',base=base,**{f'U_{i}':x['basis'] for i,x in enumerate(blocks)})
            meta['block_widths']=[x['m'] for x in blocks]
            (out/'selected.json').write_text(json.dumps(meta,indent=2)+'\n')
            break

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('run',choices=['restart40','long298','late40']);args=ap.parse_args();main(args.run)
