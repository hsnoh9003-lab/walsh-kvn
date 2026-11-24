#!/usr/bin/env python3
"""Predeclared polynomial KvN tests with real compiled elementary-gate execution.

This is a numerical validation of transport, not a protected-resource estimate.
No original wake parameters, circuits, resource records or manuscript are changed.
"""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import argparse, csv, hashlib, json, math, shutil, time, traceback
from collections import Counter
import numpy as np
from numpy.polynomial import Polynomial
from scipy.special import ndtr
from scipy.integrate import solve_ivp
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import expm_multiply
from numpy.polynomial.legendre import leggauss
from wake_gate_core import controlled_ry, increment, inverse, apply_gates, inventory
from gray_synthesis import walsh

ROOT=Path(__file__).resolve().parent
OUT=Path(os.environ.get('POLYNOMIAL_VALIDATION_OUTPUT',str(ROOT.parents[1]/'rebuilt/validation'))).resolve()
OUT.mkdir(parents=True,exist_ok=True)
(OUT/'data').mkdir(exist_ok=True)
(OUT/'circuits').mkdir(exist_ok=True)

T_FINAL=1.0
MU=[0.35,0.65]
SIGMA=[0.12,0.10]
SPATIAL_1D=[16,32,64,128]
SPATIAL_2D=[8,16,32,64]
TIME_1D=[16,32,64,128]
TIME_2D=[8,16,32,64]
BASE_N_1D=64
BASE_N_2D=16
BASE_STEPS=64
COEFF_THRESHOLD=1e-14

def clean(v):
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return [clean(x) for x in v]
    return v

def save_json(name,obj):
    (OUT/name).write_text(json.dumps(clean(obj),indent=2,allow_nan=False)+'\n')

def save_csv(name,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(clean(rows))

def max_abs_polynomial(p):
    real=[float(z.real) for z in p.deriv().roots() if abs(z.imag)<1e-10 and 0<z.real<1]
    candidates=[0.0,1.0]+real
    values=[abs(float(p(x))) for x in candidates]
    ind=int(np.argmax(values))
    return values[ind],candidates[ind],candidates

def declared_cases():
    x=Polynomial([0.,1.]);endpoint=x*(1-x)
    raw=[
      ('quadratic','Q2',endpoint,'x(1-x)','endpoint-attracting growth'),
      ('cubic_positive','C3',endpoint*(1+2*x),'x(1-x)(1+2x)','endpoint-attracting growth'),
      ('cubic_interior','C3i',endpoint*(.5-x),'x(1-x)(1/2-x)','interior-attracting equilibrium at 1/2'),
      ('quartic','Q4',x-x**4,'x-x^4','endpoint-attracting growth'),
      ('quintic','Q5',x-x**5,'x-x^5','endpoint-attracting growth'),
      ('quintic_multistable','Q5m',endpoint*(x-.25)*(x-.5)*(x-.75),
       'x(1-x)(x-1/4)(x-1/2)(x-3/4)','three attracting equilibria at 0, 1/2 and 1'),
    ]
    cases=[]
    for name,label,p,formula,behaviour in raw:
        maximum,argmax,candidates=max_abs_polynomial(p);scale=.25/maximum
        cases.append(dict(name=name,label=label,D=1,K=p.degree(),raw_coefficients=p.coef.tolist(),
             raw_formula=formula,normalization_scale=scale,raw_max_abs=maximum,
             raw_abs_max_location=argmax,critical_candidates=candidates,
             velocity_max_abs=.25,qualitative_behaviour=behaviour))
    cases.append(dict(name='coupled_cubic_2d',label='2D C3',D=2,K=3,
       raw_formula=['x(1-x)(1+2y)','y(1-y)(-2+x)'],normalization_scale=[1/3,1/2],
       raw_max_abs=[.75,.5],velocity_max_abs=[.25,.25],
       qualitative_behaviour='coupled coordinate-dependent drift toward x=1, y=0'))
    return cases

def drift(case,a):
    if case['D']==1:
        return [case['normalization_scale']*Polynomial(case['raw_coefficients'])(a[0])]
    x,y=a
    return [x*(1-x)*(1+2*y)/3, y*(1-y)*(-2+x)/2]

def coordinates(N,D):
    basis=np.arange(N**D,dtype=np.int64)
    return [((basis//N**j)%N+.5)/N for j in range(D)]

def cell_probabilities(N,j):
    e=np.linspace(0,1,N+1)
    p=np.diff(ndtr((e-MU[j])/SIGMA[j]))
    return p/p.sum()

PREPARATION_ROOT=Path(os.environ.get('PQC_PREPARATION_ROOT',str(ROOT/'preparation' if (ROOT/'preparation').exists() else ROOT.parents[1]/'output/pqc_gaussian_preparation_20261104'))).resolve()
_PREP_CACHE={}

def preparation_fit(N,j):
    n=int(np.log2(N));family='x' if j==0 else 'y';key=(n,j)
    if key not in _PREP_CACHE:
        file=PREPARATION_ROOT/'fits'/f'{family}_n{n}.json'
        record=json.loads(file.read_text())
        gates=[(g['name'],g['qubits'],g['angle']) for g in record['rz_clifford_gates']]
        state=np.zeros(N,complex);state[0]=1;state=apply_gates(state,gates,n)
        target=np.sqrt(cell_probabilities(N,j));overlap=np.vdot(target,state)
        aligned=state*np.conj(overlap/abs(overlap)) if abs(overlap)>0 else state
        record_audit=dict(family=family,n_qubits=n,cells=N,architecture=record.get('architecture','MPS'),
             Ry_count=sum(g['name']=='ry' for g in record['ry_cx_gates']),
             CX_count=sum(g['name']=='cx' for g in record['ry_cx_gates']),
             fidelity=float(abs(overlap)**2),target_state_l2_error=float(np.linalg.norm(aligned-target)),
             total_variation=float(.5*np.sum(abs(abs(state)**2-target**2))),
             negative_aligned_real_weight=float(np.sum(abs(aligned[aligned.real<0])**2)),
             aligned_imaginary_l2=float(np.linalg.norm(aligned.imag)),
             norm_defect=float(abs(np.vdot(state,state).real-1)))
        artifact=np.load(file.with_suffix('.npz'))
        record_audit['gate_vs_fit_artifact_aligned_l2']=aligned_error(state,artifact['prepared_amplitude'])
        _PREP_CACHE[key]=(record,gates,state,record_audit)
    return _PREP_CACHE[key]

def prepared_cell_probabilities(N,j):
    return abs(preparation_fit(N,j)[2])**2

def initial_amplitude(N,D):
    amplitude=np.ones(N**D,dtype=complex);basis=np.arange(N**D)
    for j in range(D):amplitude*=preparation_fit(N,j)[2][(basis//N**j)%N]
    return amplitude

def observable_values(a):
    if len(a)==1:return np.array([a[0],a[0]**2])
    x,y=a
    return np.array([x,y,x*x,y*y,x*y])

def observable_names(D):return ['x','x2'] if D==1 else ['x','y','x2','y2','xy']

def moments(state,N,D):
    prob=np.abs(state)**2
    return observable_values(coordinates(N,D))@prob

def raw_density(a,D):
    value=np.ones_like(a[0],dtype=float)
    for j in range(D):
        norm=ndtr((1-MU[j])/SIGMA[j])-ndtr(-MU[j]/SIGMA[j])
        value*=np.exp(-.5*((a[j]-MU[j])/SIGMA[j])**2)/(SIGMA[j]*np.sqrt(2*np.pi)*norm)
    return value

def continuum_reference(case,order,rtol,atol,times):
    D=case['D'];x,w=leggauss(order);x=(x+1)/2;w=w/2
    if D==1:a0=np.array([x]);weights=w
    else:
        xx,yy=np.meshgrid(x,x,indexing='ij');wx,wy=np.meshgrid(w,w,indexing='ij')
        a0=np.array([xx.ravel(),yy.ravel()]);weights=(wx*wy).ravel()
    weights=weights*raw_density(a0,D)
    # Quadrature normalization is reported, not silently re-normalized.
    def rhs(t,z):return np.asarray(drift(case,z.reshape(D,-1))).ravel()
    sol=solve_ivp(rhs,(0,T_FINAL),a0.ravel(),method='DOP853',t_eval=times,rtol=rtol,atol=atol)
    if not sol.success:raise RuntimeError(sol.message)
    hist=[]
    for k in range(len(times)):
        av=sol.y[:,k].reshape(D,-1);hist.append(observable_values(av)@weights)
    return np.asarray(hist),dict(order=order,rtol=rtol,atol=atol,nfev=sol.nfev,
        quadrature_mass=float(weights.sum()),min_coordinate=float(sol.y.min()),
        max_coordinate=float(sol.y.max()))

def generator_and_pairs(case,N):
    D=case['D'];h=1/N;dim=N**D;basis=np.arange(dim);a=coordinates(N,D)
    rows=[];cols=[];vals=[];pairs=[]
    for j in range(D):
        for parity in [0,1]:
            own=(basis//N**j)%N
            lo=basis[(own%2==parity)&(own<N-1)]
            hi=lo+N**j
            faces=[x[lo].copy() for x in a];faces[j]+=.5*h
            f=np.asarray(drift(case,faces)[j],dtype=float)
            g=f/(2*h)
            rows.extend(lo.tolist()+hi.tolist());cols.extend(hi.tolist()+lo.tolist())
            vals.extend((-g).tolist()+g.tolist())
            pairs.append(dict(axis=j,parity=parity,lo=lo,hi=hi,g=g))
    G=coo_matrix((vals,(rows,cols)),shape=(dim,dim)).tocsr()
    return G,pairs

def sequence(D):
    size=2*D
    return [(i,.5) for i in range(size-1)]+[(size-1,1.)]+[(i,.5) for i in reversed(range(size-1))]

def pair_step(state,pairs,D,delta):
    out=state.copy()
    for ind,weight in sequence(D):
        p=pairs[ind];lo,hi=p['lo'],p['hi'];phi=weight*delta*p['g']
        c=np.cos(phi);s=np.sin(phi);a=out[lo].copy();b=out[hi].copy()
        if out.ndim==2:c=c[:,None];s=s[:,None]
        out[lo]=c*a-s*b;out[hi]=s*a+c*b
    return out

def pair_evolution(a,pairs,D,delta,steps,save_every=1):
    state=a.astype(complex).copy();hist=[state.copy()]
    for k in range(steps):
        state=pair_step(state,pairs,D,delta)
        if (k+1)%save_every==0:hist.append(state.copy())
    return state,np.asarray(hist)

def remap(gates,mapping):return [(name,[mapping[q] for q in qs],angle) for name,qs,angle in gates]

def angle_table(case,N,j,parity,delta):
    D=case['D'];n=int(np.log2(N));target=j*n
    controls=[q for q in range(D*n) if q!=target]
    z=np.arange(2**len(controls),dtype=np.int64);index=np.zeros_like(z)
    for k,q in enumerate(controls):index|=((z>>k)&1)<<q
    a=[((index//N**k)%N+.5)/N for k in range(D)]
    a[j]=(((index//N**j)%N)+parity+1)/N
    f=np.asarray(drift(case,a)[j],dtype=float)
    return delta*N*f,controls,target

def compile_step(case,N,delta):
    D=case['D'];n=int(np.log2(N));work=list(range(D*n,D*n+min(1,max(0,n-3))))
    nq=D*n+len(work);layers=[];audits=[]
    for j in range(D):
        mapping={i:j*n+i for i in range(n)};mapping.update({q:q for q in work})
        inc=remap(increment(n,work),mapping)
        for parity in [0,1]:
            theta,controls,target=angle_table(case,N,j,parity,delta)
            coeff=walsh(theta);nz=np.flatnonzero(np.abs(coeff)>COEFF_THRESHOLD)
            bound=sum(math.comb(len(controls),k) for k in range(min(case['K'],len(controls))+1))
            max_weight=max([int(x).bit_count() for x in nz],default=0)
            audits.append(dict(axis=j,parity=parity,control_bits=len(controls),
                 nonzero_terms=len(nz),support_bound=bound,max_Walsh_weight=max_weight,
                 maximum_discarded_angle=float(np.max(np.abs(coeff[np.abs(coeff)<=COEFF_THRESHOLD]),initial=0)),
                 minimum_retained_angle=float(np.min(np.abs(coeff[nz]),initial=np.inf)),
                 angle_threshold=COEFF_THRESHOLD))
            gates=controlled_ry(theta,controls,target)
            layers.append((inverse(inc)+gates+inc) if parity else gates)
    gates=[]
    for ind,weight in sequence(D):
        j,parity=divmod(ind,2);theta,controls,target=angle_table(case,N,j,parity,delta*weight)
        block=controlled_ry(theta,controls,target)
        if parity:
            mapping={i:j*n+i for i in range(n)};mapping.update({q:q for q in work})
            inc=remap(increment(n,work),mapping);block=inverse(inc)+block+inc
        gates+=block
    return gates,nq,audits

def compile_preparation(N,D):
    n=int(np.log2(N));g=[]
    for j in range(D):g+=remap(preparation_fit(N,j)[1],{i:j*n+i for i in range(n)})
    return g

class CachedPrimitiveExecutor:
    """Executes each actual primitive; caches indices/phases, never a whole layer."""
    def __init__(self,nq):
        self.nq=nq;self.basis=np.arange(2**nq,dtype=np.int64);self.cache={}
    def compiled(self,gates):
        out=[]
        for name,qs,angle in gates:
            key=(name,tuple(qs),angle)
            if key not in self.cache:
                if name=='cx':
                    c,t=qs;self.cache[key]=('perm',self.basis^(((self.basis>>c)&1)<<t))
                elif name=='x':self.cache[key]=('perm',self.basis^(1<<qs[0]))
                elif name=='h':
                    lo=self.basis[(self.basis&(1<<qs[0]))==0];self.cache[key]=('h',lo,lo+(1<<qs[0]))
                else:
                    bit=(self.basis>>qs[0])&1
                    phase=np.exp(-.5j*angle*(1-2*bit)) if name=='rz' else np.exp(1j*{'t':np.pi/4,'tdg':-np.pi/4,'s':np.pi/2,'sdg':-np.pi/2}[name]*bit)
                    self.cache[key]=('phase',phase)
            out.append(self.cache[key])
        return out
    @staticmethod
    def run(state,primitives):
        state=state.copy()
        for op in primitives:
            if op[0]=='perm':state=state[op[1]]
            elif op[0]=='h':
                lo,hi=op[1:];a=state[lo].copy();b=state[hi].copy()
                state[lo]=(a+b)/np.sqrt(2);state[hi]=(a-b)/np.sqrt(2)
            else:state*=op[1] if state.ndim==1 else op[1][:,None]
        return state

def aligned_error(a,b):
    z=np.vdot(b,a);phase=z/abs(z) if abs(z)>0 else 1
    return float(np.linalg.norm(a-phase*b))

def gate_run(case,N,steps,return_history=False):
    D=case['D'];dim=N**D;delta=T_FINAL/steps
    prep=compile_preparation(N,D);gates,nq,support=compile_step(case,N,delta)
    sim=CachedPrimitiveExecutor(nq);prep_ops=sim.compiled(prep);step_ops=sim.compiled(gates)
    state=np.zeros(2**nq,dtype=complex);state[0]=1
    state=sim.run(state,prep_ops);a=initial_amplitude(N,D)
    prep_error=aligned_error(state[:dim],a)
    G,pairs=generator_and_pairs(case,N);expected=expm_multiply(T_FINAL*G,a)
    pair=a.astype(complex).copy();max_pair_error=0.;max_leak=0.;hist=[state[:dim].copy()]
    start=time.perf_counter()
    for k in range(steps):
        state=sim.run(state,step_ops);pair=pair_step(pair,pairs,D,delta)
        err=aligned_error(state[:dim],pair);leak=float(np.sum(np.abs(state[dim:])**2))
        max_pair_error=max(max_pair_error,err);max_leak=max(max_leak,leak)
        if return_history:hist.append(state[:dim].copy())
    elapsed=time.perf_counter()-start
    final=state[:dim];mom=moments(final,N,D);emom=moments(expected,N,D)
    inv=inventory(gates);pinv=inventory(prep);total={k:pinv[k]+steps*inv[k] for k in ['RZ','direct_T','Clifford']}
    by=Counter(pinv['by_gate']);by.update({k:steps*v for k,v in inv['by_gate'].items()})
    row=dict(case=case['name'],label=case['label'],dimension=D,degree=case['K'],cells_per_coordinate=N,
        cells_total=dim,steps=steps,data_qubits=D*int(np.log2(N)),work_qubits=nq-D*int(np.log2(N)),
        logical_qubits=nq,preparation_state_l2_error=prep_error,
        preparation_target_state_l2_error=aligned_error(a,np.sqrt(np.prod([cell_probabilities(N,j)[(np.arange(dim)//N**j)%N] for j in range(D)],axis=0))),
        preparation_target_probability_TV=float(.5*np.sum(abs(abs(a)**2-np.prod([cell_probabilities(N,j)[(np.arange(dim)//N**j)%N] for j in range(D)],axis=0)))),
        preparation_Ry=sum(preparation_fit(N,j)[3]['Ry_count'] for j in range(D)),
        maximum_gate_vs_pair_state_l2_error=max_pair_error,final_gate_vs_pair_raw_l2_error=float(np.linalg.norm(final-pair)),
        maximum_work_leakage=max_leak,norm_defect=float(abs(np.vdot(state,state).real-1)),
        state_l2_splitting_error=aligned_error(final,expected),
        moment_rms_splitting_error=float(np.sqrt(np.mean((mom-emom)**2))),
        per_step_RZ=inv['RZ'],per_step_CX=inv['by_gate'].get('cx',0),per_step_native_T=inv['direct_T'],
        preparation_RZ=pinv['RZ'],preparation_CX=pinv['by_gate'].get('cx',0),
        complete_RZ=total['RZ'],complete_CX=by.get('cx',0),complete_native_T=total['direct_T'],
        wall_seconds=elapsed,status='pass')
    if prep_error>1e-10 or max_pair_error>1e-9 or max_leak>1e-18:row['status']='FAIL'
    return row,final,np.asarray(hist),support,dict(nqubits=nq,prep=prep,step=gates,steps=steps,
          native_basis='arbitrary RZ, H, S/Sdag, X, CX, T/Tdag; CCX decomposed; no arbitrary-rotation T synthesis')

def operator_check(case):
    N=16 if case['D']==1 else 8;D=case['D'];dim=N**D;delta=T_FINAL/64
    gates,nq,support=compile_step(case,N,delta)
    inp=np.zeros((2**nq,dim),complex);inp[np.arange(dim),np.arange(dim)]=1
    sim=CachedPrimitiveExecutor(nq);actual=sim.run(inp,sim.compiled(gates))
    ideal=pair_step(np.eye(dim,dtype=complex),generator_and_pairs(case,N)[1],D,delta)
    # Check the optimized executor against the archived per-gate executor too.
    seed=np.random.default_rng(61104);v=seed.normal(size=2**nq)+1j*seed.normal(size=2**nq);v/=np.linalg.norm(v)
    a=sim.run(v,sim.compiled(gates));b=apply_gates(v,gates,nq)
    matrix_error=float(np.max(np.abs(actual[:dim]-ideal)))
    leak=float(np.max(np.sum(np.abs(actual[dim:])**2,axis=0),initial=0))
    return dict(case=case['name'],dimension=D,cells_per_coordinate=N,logical_qubits=nq,
         clean_work_operator_max_entry_error=matrix_error,maximum_work_leakage=leak,
         cached_vs_archived_executor_l2=float(np.linalg.norm(a-b)),
         maximum_Walsh_weight=max(s['max_Walsh_weight'] for s in support),
         support_bound_pass=all(s['nonzero_terms']<=s['support_bound'] and s['max_Walsh_weight']<=case['K'] for s in support),
         status='pass' if matrix_error<1e-11 and leak<1e-20 and np.linalg.norm(a-b)<1e-11 else 'FAIL')

def cell_references(case,N,order,times):
    """Independent characteristics, with two separately weighted cell inputs."""
    D=case['D'];z,w=leggauss(order);z=(z+1)/2;w=w/2
    nodes=[(np.arange(N)[:,None]+z[None,:]).ravel()/N for j in range(D)]
    prepared_weights=[(prepared_cell_probabilities(N,j)[:,None]*w[None,:]).ravel() for j in range(D)]
    gaussian_weights=[(cell_probabilities(N,j)[:,None]*w[None,:]).ravel() for j in range(D)]
    if D==1:a0=np.array([nodes[0]]);pw=prepared_weights[0];gw=gaussian_weights[0]
    else:
        x,y=np.meshgrid(*nodes,indexing='ij');a0=np.array([x.ravel(),y.ravel()])
        p0,p1=np.meshgrid(*prepared_weights,indexing='ij');pw=(p0*p1).ravel()
        g0,g1=np.meshgrid(*gaussian_weights,indexing='ij');gw=(g0*g1).ravel()
    def rhs(t,z):return np.asarray(drift(case,z.reshape(D,-1))).ravel()
    sol=solve_ivp(rhs,(0,T_FINAL),a0.ravel(),method='DOP853',t_eval=times,rtol=4e-13,atol=4e-15)
    if not sol.success:raise RuntimeError(sol.message)
    ph=[];gh=[]
    for k in range(len(times)):
        values=observable_values(sol.y[:,k].reshape(D,-1));ph.append(values@pw);gh.append(values@gw)
    return np.array(ph),np.array(gh),dict(quadrature_order_per_cell=order,characteristic_count=a0.shape[1],
       nfev=sol.nfev,prepared_input_mass=float(pw.sum()),gaussian_cell_input_mass=float(gw.sum()))

def rms(a):return float(np.sqrt(np.mean(np.asarray(a)**2)))

def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    cases=declared_cases();times=np.linspace(0,T_FINAL,65)
    protocol=dict(version='2026-11-06-MPS',cases=cases,T_FINAL=T_FINAL,mu=MU,sigma=SIGMA,
      physical_domain='[0,1]^D; every tested drift component vanishes on its own boundary faces',
      grid='uniform cells; MPS circuit compiled to analytic truncated-normal cell masses at each resolution',
      spatial_grids_1d=SPATIAL_1D,spatial_grids_2d=SPATIAL_2D,
      time_steps_1d=TIME_1D,time_steps_2d=TIME_2D,
      fixed_time_refinement_grid_1d=BASE_N_1D,fixed_time_refinement_grid_2d=BASE_N_2D,
      final_time=T_FINAL,coefficient_threshold=COEFF_THRESHOLD,
      splitting='symmetric palindrome over the 2D directional even/odd generators',
      preparation='MPS compilation with individually recorded layer count per coordinate; all gates executed from zero; no generic exact loading',
      preparation_sharing='one fit per coordinate family and n; no per-polynomial fits; four Ry/CX stages fixed at all n',
      input_comparisons='same actual MPS-prepared amplitude for split/expm; matched prepared-cell-density characteristics for transport; fixed analytic Gaussian for total error',
      cell_reference_quadrature_1d=[8,16],cell_reference_quadrature_2d=[2,4],
      observables_1d=observable_names(1),observables_2d=observable_names(2),
      primary_wake_changed=False,protected_resource_estimation=False,
      selection='all declared cases and all planned refinements are reported; no accuracy-based model/grid/time retuning')
    save_json('protocol.json',protocol)
    shutil.copyfile(ROOT/'provenance.json',OUT/'provenance.json')
    operator_rows=[];time_rows=[];space_rows=[];support_rows=[];continuum_rows=[];failures=[];baseline_rows=[];error_rows=[];cell_reference_rows=[]
    for case in cases:
      name=case['name'];D=case['D'];print('CASE',name,flush=True)
      try:
        row=operator_check(case);operator_rows.append(row);print('OPERATOR',row,flush=True)
        ref_order=128 if D==1 else 48
        reference,diag=continuum_reference(case,ref_order,2e-12,2e-14,times)
        reference_fine,fine_diag=continuum_reference(case,ref_order*2,4e-13,4e-15,times)
        continuum_rows.append(dict(case=name,reference_order=ref_order,refined_reference_order=ref_order*2,
           reference_refinement_max_abs_moment_difference=float(np.max(np.abs(reference-reference_fine))),
           **{f'baseline_{k}':v for k,v in diag.items()},**{f'fine_{k}':v for k,v in fine_diag.items()}))
        reference=reference_fine
        np.savez_compressed(OUT/'data'/f'{name}_continuum.npz',times=times,moments=reference,
              observable_names=np.asarray(observable_names(D)))
        ref_final=reference[-1];base_references=None
        for N in SPATIAL_1D if D==1 else SPATIAL_2D:
            a=initial_amplitude(N,D);G,pairs=generator_and_pairs(case,N)
            final=expm_multiply(T_FINAL*G,a);m=moments(final,N,D)
            Nbase=BASE_N_1D if D==1 else BASE_N_2D
            ct=times if N==Nbase else np.array([0.,T_FINAL]);q=8 if D==1 else 2
            cp_low,cg_low,cl=cell_references(case,N,q,ct)
            cp,cg,ch=cell_references(case,N,q*2,ct)
            cell_reference_rows.append(dict(case=name,cells_per_coordinate=N,dimension=D,
               prepared_characteristic_refinement_max_change=float(np.max(abs(cp-cp_low))),
               gaussian_cell_characteristic_refinement_max_change=float(np.max(abs(cg-cg_low))),
               **ch))
            if N==Nbase:base_references=(cp,cg)
            transport=m-cp[-1];fit=cp[-1]-cg[-1];projection=cg[-1]-ref_final;total=m-ref_final
            input_bias=cp[-1]-ref_final
            budget=dict(case=name,label=case['label'],dimension=D,degree=case['K'],cells_per_coordinate=N,
                transport_matched_input_RMS=rms(transport),PQC_fit_RMS_bias=rms(fit),
                ideal_cell_projection_RMS_bias=rms(projection),prepared_input_RMS_bias=rms(input_bias),
                total_analytic_Gaussian_RMS_error=rms(total),
                initial_PQC_fit_moment_RMS=rms(cp[0]-cg[0]),initial_cell_projection_moment_RMS=rms(cg[0]-reference[0]),
                vector_decomposition_max_residual=float(np.max(abs(total-transport-fit-projection))))
            for tag,vec in [('transport',transport),('PQC_fit',fit),('ideal_cell_projection',projection),('prepared_input',input_bias),('total_analytic',total)]:
                for key,value in zip(observable_names(D),vec):budget[tag+'_'+key]=value
            error_rows.append(budget)
            Gskew=G+G.T;skew=float(np.max(np.abs(Gskew.data),initial=0))
            row=dict(case=name,label=case['label'],dimension=D,degree=case['K'],
               cells_per_coordinate=N,cells_total=N**D,
               reference='exact finite generator versus characteristics of same actual MPS-prepared cell masses; input varies with resolution',
               moment_rms_transport_error=rms(transport),moment_rms_total_analytic_error=rms(total),
               moment_max_transport_error=float(np.max(abs(transport))),skew_defect=skew,
               norm_defect=float(abs(np.vdot(final,final).real-1)),
               preparation_target_probability_TV=float(.5*np.sum(abs(abs(a)**2-np.prod([cell_probabilities(N,j)[(np.arange(N**D)//N**j)%N] for j in range(D)],axis=0)))),
               spatial_convergence='not_assumed; resolution-dependent fitted input; transport and total errors both reported',
               status='pass' if skew<1e-13 and np.all(np.isfinite(m)) and abs(np.vdot(final,final).real-1)<1e-10 else 'FAIL')
            for k,v in zip(observable_names(D),m):row['moment_'+k]=v
            for k,v in zip(observable_names(D),transport):row['transport_error_'+k]=v
            for k,v in zip(observable_names(D),total):row['total_analytic_error_'+k]=v
            space_rows.append(row)
            gates,nq,support=compile_step(case,N,T_FINAL/BASE_STEPS)
            for support_item in support:support_rows.append(dict(case=name,label=case['label'],dimension=D,degree=case['K'],cells_per_coordinate=N,
                 full_step_RZ=inventory(gates)['RZ'],full_step_CX=inventory(gates)['by_gate'].get('cx',0),
                 full_step_native_T=inventory(gates)['direct_T'],logical_qubits=nq,**support_item))
            np.savez_compressed(OUT/'data'/f'{name}_grid{N}.npz',initial_amplitude=a,semidiscrete_final=final,
                 moments=m,analytic_Gaussian_moments=ref_final,prepared_cell_moments=cp,Gaussian_cell_moments=cg,
                 cell_reference_times=ct,grid_coordinates=np.array(coordinates(N,D)))
            print('GRID',name,N,'transport',rms(transport),'fit',rms(fit),'total',rms(total),flush=True)
        N=BASE_N_1D if D==1 else BASE_N_2D
        for steps in TIME_1D if D==1 else TIME_2D:
            row,final,hist,support,circuit=gate_run(case,N,steps,return_history=(steps==BASE_STEPS))
            time_rows.append(row);print('GATE',name,N,steps,row['state_l2_splitting_error'],row['maximum_gate_vs_pair_state_l2_error'],row['wall_seconds'],flush=True)
            if steps==BASE_STEPS:
                m=moments(final,N,D);cp,cg=base_references
                row.update(continuum_matched_input_moment_rms_error=rms(m-cp[-1]),continuum_total_analytic_moment_rms_error=rms(m-ref_final))
                baseline_rows.append(dict(row))
                mh=np.array([moments(st,N,D) for st in hist]);np.savez_compressed(OUT/'data'/f'{name}_compiled_baseline.npz',
                     times=times,state_history=hist,moment_history=mh,final_state=final,
                     prepared_cell_reference_moments=cp,Gaussian_cell_reference_moments=cg,analytic_Gaussian_reference_moments=reference)
                save_json('circuits/'+name+'.json',circuit)
        for name_file,rows in [('operator_checks.csv',operator_rows),('time_refinement.csv',time_rows),
            ('spatial_refinement.csv',space_rows),('Walsh_support_and_gate_counts.csv',support_rows),
            ('continuum_reference_checks.csv',continuum_rows),('compiled_baselines.csv',baseline_rows),
            ('prediction_error_decomposition.csv',error_rows),('matched_cell_reference_checks.csv',cell_reference_rows)]:save_csv(name_file,rows)
      except Exception as exc:
        failures.append(dict(case=name,error=repr(exc),traceback=traceback.format_exc()));print('FAILURE',failures[-1],flush=True)
        save_json('failures.json',failures)
    preparation_rows=[v[3] for k,v in sorted(_PREP_CACHE.items())];save_csv('preparation_quality_checks.csv',preparation_rows)
    save_json('failures.json',failures)
    result=dict(declared_cases=len(cases),operator_checks=len(operator_rows),time_refinement_runs=len(time_rows),
       spatial_refinement_runs=len(space_rows),baseline_runs=len(baseline_rows),unique_preparation_fits=len(preparation_rows),
       unhandled_case_failures=failures,checks_failed=[r for r in operator_rows+time_rows+space_rows if r.get('status')!='pass'],
       max_gate_vs_pair_error=max([r['maximum_gate_vs_pair_state_l2_error'] for r in time_rows],default=None),
       max_work_leakage=max([r['maximum_work_leakage'] for r in time_rows],default=None),
       max_operator_entry_error=max([r['clean_work_operator_max_entry_error'] for r in operator_rows],default=None),
       max_continuum_reference_change=max([r['reference_refinement_max_abs_moment_difference'] for r in continuum_rows],default=None),
       max_prepared_cell_reference_change=max([r['prepared_characteristic_refinement_max_change'] for r in cell_reference_rows],default=None),
       max_preparation_negative_weight=max([r['negative_aligned_real_weight'] for r in preparation_rows],default=None),
       max_preparation_imaginary_l2=max([r['aligned_imaginary_l2'] for r in preparation_rows],default=None))
    save_json('validation_summary.json',result);print('SUMMARY',json.dumps(clean(result)),flush=True)
    source=OUT/'source';source.mkdir(exist_ok=True)
    for p in ROOT.glob('*.py'):shutil.copyfile(p,source/p.name)
    if (ROOT/'provenance').exists():shutil.copytree(ROOT/'provenance',source/'provenance',dirs_exist_ok=True)
    if (ROOT/'provenance.json').exists():shutil.copyfile(ROOT/'provenance.json',source/'provenance.json')
    shutil.copytree(PREPARATION_ROOT,OUT/'preparation',dirs_exist_ok=True)
    shutil.copytree(PREPARATION_ROOT,source/'preparation',dirs_exist_ok=True)
    if (ROOT/'fonts').exists():shutil.copytree(ROOT/'fonts',source/'fonts',dirs_exist_ok=True)
    if (ROOT/'requirements.txt').exists():shutil.copyfile(ROOT/'requirements.txt',source/'requirements.txt')
    if failures or result['checks_failed']:raise SystemExit(2)

if __name__=='__main__':main()
