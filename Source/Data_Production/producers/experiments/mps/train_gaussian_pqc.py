#!/usr/bin/env python3
"""Fit and gate-execute the unchanged primary WakeAnsatz(n,2) structure.

Training targets are positive square roots of analytic truncated-normal cell
masses. The delivered initial states are obtained by executing the compiled
ordered RZ/Clifford gate list on zero, not by injecting the target amplitudes.
Each (family,n) fit is independent of and reused across all subsequent drifts.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import json, time, sys, hashlib, importlib.util
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from scipy.optimize import least_squares
from scipy.special import ndtr

OUT = Path(__file__).resolve().parent
OUT_SOURCE=OUT
SOURCE = OUT/'source'/'Gaussian_ansatz_original.py'
spec = importlib.util.spec_from_file_location('primary_Gaussian_ansatz', SOURCE)
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)
WakeAnsatz = original.WakeAnsatz
SEED_BASE = 20261104
STARTS = 16
MAX_NFEV = 2000
TOLS = dict(ftol=1e-11, xtol=1e-11, gtol=1e-11)
FAMILIES = {'x':dict(mu=.35,sigma=.12,ns=list(range(3,8)),seed_offset=0),
            'y':dict(mu=.65,sigma=.10,ns=list(range(3,7)),seed_offset=10000)}

def write_json_atomic(path, value):
    path=Path(path);temp=path.with_name(path.name+'.ready')
    temp.write_text(json.dumps(value,indent=2)+'\n');os.replace(temp,path)

def write_npz_atomic(path, **arrays):
    path=Path(path);temp=path.with_name(path.name+'.ready')
    with temp.open('wb') as stream:np.savez_compressed(stream,**arrays)
    os.replace(temp,path)

def gate(name, qs, angle=None):
    return dict(name=name,qubits=list(qs),angle=None if angle is None else float(angle))

def gate_program(ans, angles):
    native=[]; compiled=[]
    angles=np.asarray(angles).reshape(ans.stages,ans.n)
    for stage in range(ans.stages):
        for q in range(ans.n):
            a=float(angles[stage,q]);native.append(gate('ry',[q],a))
            compiled.extend([gate('sdg',[q]),gate('h',[q]),gate('rz',[q],a),gate('h',[q]),gate('s',[q])])
        for c,t in ans.edges(stage):
            op=gate('cx',[c,t]);native.append(op);compiled.append(op.copy())
    return native,compiled

def execute_zero(n, gates):
    state=np.zeros(2**n,dtype=complex);state[0]=1
    idx=np.arange(2**n)
    for op in gates:
        name,qs,a=op['name'],op['qubits'],op['angle']
        if name=='cx':
            c,t=qs;state=state[idx^(((idx>>c)&1)<<t)]
        elif name=='h':
            lo=idx[((idx>>qs[0])&1)==0];hi=lo+(1<<qs[0])
            u,v=state[lo].copy(),state[hi].copy()
            state[lo]=(u+v)/np.sqrt(2);state[hi]=(u-v)/np.sqrt(2)
        elif name=='rz':state*=np.exp(-.5j*a*(1-2*((idx>>qs[0])&1)))
        elif name in ('s','sdg'):
            state*=np.exp((1j if name=='s' else -1j)*np.pi/2*((idx>>qs[0])&1))
        elif name=='ry':
            lo=idx[((idx>>qs[0])&1)==0];hi=lo+(1<<qs[0]);u,v=state[lo].copy(),state[hi].copy()
            co,si=np.cos(a/2),np.sin(a/2)
            state[lo]=co*u-si*v;state[hi]=si*u+co*v
        else:raise ValueError(name)
    return state

def counts(gates):
    ans={}
    for op in gates:ans[op['name']]=ans.get(op['name'],0)+1
    return ans

def target_distribution(n,mu,sigma):
    edges=np.linspace(0,1,2**n+1)
    lower=float(ndtr(-mu/sigma));upper=float(ndtr((mu-1)/sigma))
    mass=np.diff(ndtr((edges-mu)/sigma));mass/=mass.sum()
    return edges,mass,dict(lower=lower,upper=upper,total=lower+upper)

def fit_one(family,n):
    t0=time.perf_counter();cfg=FAMILIES[family];mu,sigma=cfg['mu'],cfg['sigma']
    edges,p,tail=target_distribution(n,mu,sigma);target=np.sqrt(p);centers=(edges[1:]+edges[:-1])/2
    seed=SEED_BASE+n*100+cfg['seed_offset']
    # The canonical KvN implementation is shared with the live experiment.
    from Source.KvN import radial_mps as mps
    native_state,program,metadata=mps.prepare_layers(p,min_fidelity=.9999,max_layers=16,seed=seed)
    native=[gate(name,qs,a)for name,qs,a in program]
    gates=[gate(name,qs,a)for name,qs,a in mps.compile_rz(program)]
    actual=execute_zero(n,gates);native_actual=execute_zero(n,native)
    replay_error=float(np.linalg.norm(actual-native_actual));class_error=float(np.linalg.norm(actual-native_state))
    assert replay_error<3e-12 and class_error<3e-12
    angles=np.array([a for name,_,a in program if name=='ry']);winner=0;trials=[]
    err=float(np.linalg.norm(actual-target))
    overlap=np.vdot(target,actual);phase=overlap/abs(overlap)
    aligned=actual/phase;prob=abs(actual)**2
    negative=aligned.real<0;neg_tol=aligned.real < -1e-12
    c2=(edges[:-1]**2+edges[:-1]*edges[1:]+edges[1:]**2)/3
    target_mean=float(p@centers);target_second=float(p@c2)
    prep_mean=float(prob@centers);prep_second=float(prob@c2)
    alpha,beta=(0-mu)/sigma,(1-mu)/sigma;Z=ndtr(beta)-ndtr(alpha)
    phi=lambda x:np.exp(-x*x/2)/np.sqrt(2*np.pi)
    zmean=(phi(alpha)-phi(beta))/Z
    mean_exact=mu+sigma*zmean
    zsecond=1+(alpha*phi(alpha)-beta*phi(beta))/Z
    second_exact=mu*mu+2*mu*sigma*zmean+sigma*sigma*zsecond
    qc=dict(target_phase_convention='positive real cell amplitudes',norm=float(np.vdot(actual,actual).real),
        amplitude_l2_error=float(np.linalg.norm(aligned-target)),max_abs_amplitude_error=float(np.max(abs(aligned-target))),
        raw_amplitude_l2_error=float(np.linalg.norm(actual-target)),fidelity=float(abs(overlap)**2),
        probability_l1_error=float(np.sum(abs(prob-p))),max_probability_error=float(np.max(abs(prob-p))),
        min_real_amplitude=float(np.min(aligned.real)),max_abs_imaginary_amplitude=float(np.max(abs(actual.imag))),
        max_abs_imaginary_amplitude_after_alignment=float(np.max(abs(aligned.imag))),
        negative_amplitude_probability_mass=float(np.sum(prob[negative])),negative_amplitude_count=int(np.sum(negative)),
        negative_amplitude_probability_mass_below_minus_1e_12=float(np.sum(prob[neg_tol])),
        most_negative_amplitude=float(min(0.,np.min(aligned.real))),
        target_underlying_normal_excluded_mass=tail,
        target_cell_uniform_moments={'mean':target_mean,'second':target_second},
        prepared_cell_uniform_moments={'mean':prep_mean,'second':prep_second},
        preparation_moment_bias={'mean':prep_mean-target_mean,'second':prep_second-target_second},
        analytic_truncated_normal_moments={'mean':float(mean_exact),'second':float(second_exact)},
        cell_binning_moment_bias={'mean':target_mean-mean_exact,'second':target_second-second_exact},
        total_prepared_cell_vs_analytic_normal_bias={'mean':prep_mean-mean_exact,'second':prep_second-second_exact},
        replay_vs_native_ry_state_error=replay_error,replay_vs_MPS_state_error=class_error,
        global_phase_to_positive_target={'real':float(phase.real),'imag':float(phase.imag),'argument_rad':float(np.angle(phase))})
    report=dict(family=family,n_qubits=n,picture_blocks=metadata["layers"],stages=0,architecture="MPS",
        target=dict(mu=mu,sigma=sigma,domain=[0,1],type='normalized analytic truncated-normal cell masses'),
        qubit_order='little-endian; q0 is least significant bit',initial_state='all zero',
        preparation_clean_work_qubits=0,angles=angles.tolist(),ry_cx_gates=native,rz_clifford_gates=gates,
        counts=dict(native=counts(native),compiled=counts(gates),arbitrary_rotations=metadata["Ry"],CX=metadata["CX"],native_T=0,clean_work_qubits=0),
        quality=qc,training=dict(loss='SVD truncation plus local real subspace compilation',method='sequential real MPS layers',
            source_sha256=hashlib.sha256(Path(mps.__file__).read_bytes()).hexdigest(),seed=seed,
            starts=1,max_nfev_per_start=0,tolerances={'minimum_fidelity':.9999,'maximum_layers':16},
            selected_start=0,total_nfev=0,total_njev=0,seconds=time.perf_counter()-t0,trials=[],MPS=metadata),
        reuse='One shared fit per family and resolution; independent of all drift polynomials and output errors.')
    folder=OUT/'fits';folder.mkdir(exist_ok=True);stem=f'{family}_n{n}'
    write_npz_atomic(folder/(stem+'.npz'),target_probability=p,target_amplitude=target,
        prepared_amplitude=actual,prepared_probability=prob,cell_edges=edges,cell_centers=centers,angles=angles)
    write_json_atomic(folder/(stem+'.json'),report)
    return dict(family=family,n_qubits=n,npz=f'fits/{stem}.npz',json=f'fits/{stem}.json',quality=qc,
        counts=report['counts'],training_seconds=report['training']['seconds'],selected_start=winner,
        total_nfev=report['training']['total_nfev'])

def main():
    begin=time.perf_counter();jobs=[(f,n) for f,cfg in FAMILIES.items() for n in cfg['ns']]
    results=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        futs={pool.submit(fit_one,f,n):(f,n) for f,n in jobs}
        for future in as_completed(futs):
            record=future.result();results.append(record)
            results.sort(key=lambda r:(r['family'],r['n_qubits']))
            index=dict(architecture='Real MPS layers; minimum fidelity .9999, maximum16 layers; recorded actual counts per input',
                families=FAMILIES,starts=STARTS,max_nfev=MAX_NFEV,fits=results,complete=len(results)==len(jobs))
            write_json_atomic(OUT/'index.json',index)
            print(json.dumps({'saved':record['json'],'L2':record['quality']['amplitude_l2_error'],
                'fidelity':record['quality']['fidelity'],'negative_mass':record['quality']['negative_amplitude_probability_mass'],
                'seconds':record['training_seconds']}),flush=True)
    print('COMPLETE',len(results),'fits; wall seconds',time.perf_counter()-begin,flush=True)

if __name__=='__main__':main()
