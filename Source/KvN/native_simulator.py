"""Reproduce gate-level and operator checks for both 9-qubit wake circuits.

Run with --configuration and --initial to certify a specific frozen wake case.
Without those arguments it tests the current manuscript's rounded parameters,
an explicitly labelled arbitrary complex input, and the complete input basis.
"""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import argparse,json,hashlib
import numpy as np
from .one_clean_synthesis import *
from .sparse_walsh import controlled_ry,phase_layer,walsh

ROOT=Path(__file__).resolve().parent


class Simulator:
    def __init__(self,n):
        self.n=n;self.basis=np.arange(1<<n);self.perms={};self.lo={};self.hi={};self.sign={}
        for q in range(n):
            self.lo[q]=self.basis[(self.basis&(1<<q))==0]
            self.hi[q]=self.lo[q]+(1<<q)
            self.sign[q]=1-2*((self.basis>>q)&1)
    def run(self,state,gates):
        state=np.asarray(state,dtype=complex).copy()
        for name,qs,angle in gates:
            if name in ['cx','x','ccx']:
                key=(name,*qs)
                if key not in self.perms:
                    if name=='x':mask=1<<qs[0]
                    elif name=='cx':mask=(((self.basis>>qs[0])&1)<<qs[1])
                    else:mask=((((self.basis>>qs[0])&1)&((self.basis>>qs[1])&1))<<qs[2])
                    self.perms[key]=self.basis^mask
                state=state[self.perms[key]]
            elif name=='h':
                lo,hi=self.lo[qs[0]],self.hi[qs[0]]
                a,b=state[lo].copy(),state[hi].copy()
                state[lo]=(a+b)/np.sqrt(2);state[hi]=(a-b)/np.sqrt(2)
            elif name=='rz':
                phase=np.exp(-.5j*angle*self.sign[qs[0]])
                state*=phase if state.ndim==1 else phase[:,None]
            else:
                phase=np.exp(1j*{'t':np.pi/4,'tdg':-np.pi/4,'s':np.pi/2,'sdg':-np.pi/2}[name])
                state[self.hi[qs[0]]]*=phase
        return state


def aligned_error(actual,ideal):
    overlap=np.vdot(ideal,actual)
    phase=overlap/abs(overlap)
    return float(np.linalg.norm(actual-phase*ideal)),complex(phase)


def ry(t,angle):
    return [('sdg',[t],None),('h',[t],None),('rz',[t],float(angle)),('h',[t],None),('s',[t],None)]


def mps_prep(angles,n=6):
    a=list(map(float,angles));assert len(a)==4*n-5
    out=ry(1,a[0])+[('cx',[1,0],None)]+ry(1,a[1])+ry(0,a[2])
    index=3
    for q0 in range(1,n-1):
        q1=q0+1;b,c,d,e=a[index:index+4];index+=4
        out+=ry(q1,b)+[('cx',[q1,q0],None)]+ry(q0,c)+ry(q1,d)+[('cx',[q0,q1],None)]+ry(q0,e)
    return out+[('h',[6],None),('h',[7],None)]


def pqc_prep(angles,n=6):
    if len(angles)==4*n-5:return mps_prep(angles,n)
    theta=np.asarray(angles,dtype=float).reshape(4,n);out=[]
    for stage in range(4):
        for bit in range(n):out+=ry(bit,theta[stage,bit])
        out+=sum((cx(q,q+1) if stage%2==0 else cx(q+1,q) for q in range(n-1)),[])
    return out+[('h',[6],None),('h',[7],None)]


def rotate_pairs_matrix(state,angles):
    # Radius is the least-significant six-bit subregister in every probe branch.
    s=state.reshape((4,64)+state.shape[1:]).copy()
    a,b=s[:,0::2].copy(),s[:,1::2].copy()
    shape=(1,32)+(1,)*(s.ndim-2)
    co=np.cos(angles).reshape(shape);si=np.sin(angles).reshape(shape)
    s[:,0::2]=co*a-si*b;s[:,1::2]=si*a+co*b
    return s.reshape(state.shape)


def ideal_step(v,r,dt,mu,gamma):
    h=1/64;w=np.repeat([0,1,1,2],64)
    phase=np.exp(.5j*gamma*dt*w*np.tile(r*r,4))
    shape=(-1,)+(1,)*(v.ndim-1)
    v=v*phase.reshape(shape)
    even=(2*np.arange(32)+1)/64;odd=(2*np.arange(32)+2)/64
    v=rotate_pairs_matrix(v,mu*even*(1-even**2)*dt/(4*h))
    v=np.roll(v.reshape((4,64)+v.shape[1:]),-1,axis=1).reshape(v.shape)
    v=rotate_pairs_matrix(v,mu*odd*(1-odd**2)*dt/(2*h))
    v=np.roll(v.reshape((4,64)+v.shape[1:]),1,axis=1).reshape(v.shape)
    v=rotate_pairs_matrix(v,mu*even*(1-even**2)*dt/(4*h))
    return v*phase.reshape(shape)


def rz_string(qs,angle):
    if not qs:return []
    t=qs[-1]
    return sum((cx(q,t) for q in qs[:-1]),[])+[('rz',[t],float(angle))]+sum((cx(q,t) for q in reversed(qs[:-1])),[])


def reference_ry(values,controls,target):
    terms=walsh(values);g=[('sdg',[target],None),('h',[target],None)]
    for mask,a in enumerate(terms):
        if abs(a)>1e-14:g+=rz_string([q for j,q in enumerate(controls) if mask>>j&1]+[target],a)
    return g+[('h',[target],None),('s',[target],None)]


def reference_phase(r,lam):
    g=[]
    for mask,value in enumerate(walsh(r*r)):
        if abs(value)<1e-14:continue
        inds=[j for j in range(6) if mask>>j&1]
        if inds:g+=rz_string(inds,-2*lam*value)
        for probe in [6,7]:g+=rz_string(inds+[probe],lam*value)
    return g


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--configuration');ap.add_argument('--initial');ap.add_argument('--label',default='universal_current_rounded');args=ap.parse_args()
    out=ROOT/'results'/args.label;out.mkdir(parents=True,exist_ok=True)
    # The parameter-free INC/DEC tests and full input-basis tests establish the
    # reduced-work implementation independently of a particular initial ensemble.
    cfg={'mu':.0924819,'gamma_normalized':.301734,'dtstar':9.8291/33,'start_tstar':40.,'steps':33}
    params_source='current manuscript rounded coefficients; this is not a new physical fit'
    if args.configuration:
        cfg.update(json.loads(Path(args.configuration).read_text()));params_source=str(Path(args.configuration).resolve())
    dt=float(cfg['dtstar']);mu=float(cfg['mu']);gamma=float(cfg['gamma_normalized'])
    r=(np.arange(64)+.5)/64
    # This warm-up provides exactly four Ry stages and 20 CX; the actual frozen
    # preparation angles replace it when --initial is supplied.
    rng=np.random.default_rng(20261104)
    angles=rng.normal(size=24)
    input_source='declared fixed-seed four-stage PQC; universal operator verification is independent of it'
    target_amplitude=None
    if args.initial:
        data=np.load(args.initial);angles=data['angles'];input_source=str(Path(args.initial).resolve())
        if 'amplitude' in data:target_amplitude=data['amplitude']
    prep=pqc_prep(angles);readout=[('h',[6],None),('sdg',[7],None),('h',[7],None)]
    sim=Simulator(9);zero=np.zeros(512,complex);zero[0]=1
    prepared=sim.run(zero,prep)
    prep_match=None
    if target_amplitude is not None:
        prep_match=aligned_error(prepared[:256],np.tile(target_amplitude,4)/2)[0]
        assert prep_match<1e-11,(prep_match,input_source)
    ef=(2*np.arange(32)+1)/64;of=(2*np.arange(32)+2)/64;h=1/64
    ea=mu*ef*(1-ef**2)*dt/(2*h);oa=mu*of*(1-of**2)*dt/h
    even=controlled_ry(ea,list(range(1,6)),0);odd=controlled_ry(oa,list(range(1,6)),0)
    ph=phase_layer(r,gamma*dt/2,6,(6,7));pf=phase_layer(r,gamma*dt,6,(6,7))
    even_ref=reference_ry(ea,list(range(1,6)),0);odd_ref=reference_ry(oa,list(range(1,6)),0);pf_ref=reference_phase(r,gamma*dt)
    basis=np.zeros((512,256),complex);basis[:256]=np.eye(256)
    v=rng.normal(size=256)+1j*rng.normal(size=256);v/=np.linalg.norm(v)
    report={'logical_qubits':9,'radius_qubits':6,'phase_probes':2,'clean_work':1,'steps':33,'configuration':cfg,'configuration_source':params_source,'initial_source':input_source,'prep_actual_amplitude_match_error':prep_match,'all_basis_columns':256,'methods':{}}
    report['parity_reference_counts']={'even':inventory(even_ref),'odd':inventory(odd_ref),'phase':inventory(pf_ref)}
    report['parity_reused_counts']={'even':inventory(even),'odd':inventory(odd),'phase':inventory(pf)}
    for name,fn in [('A',increment_A),('B',increment_B)]:
        high=fn(list(range(6)),8);inc=expand(high);dec=inverse(inc)
        permcheck=sim.run(basis,inc)
        permideal=np.zeros_like(basis)
        inds=np.arange(256);permideal[(inds&~63)|((inds+1)&63),inds]=1
        inc_error=float(np.linalg.norm(permcheck-permideal,ord=2))
        decref=np.zeros_like(basis);decref[(inds&~63)|((inds-1)&63),inds]=1
        dec_error=float(np.linalg.norm(sim.run(basis,dec)-decref,ord=2))
        assert max(inc_error,dec_error)<1e-12
        radial=even+dec+odd+inc+even
        fullstep=ph+radial+ph
        actual=sim.run(basis,fullstep);target=ideal_step(np.eye(256,dtype=complex),r,dt,mu,gamma)
        err,global_phase=aligned_error(actual[:256],target)
        op_error=float(np.linalg.norm(actual[:256]-global_phase*target,ord=2))
        leakage=float(np.linalg.norm(actual[256:],ord=2))
        assert op_error<1e-11 and leakage<1e-12
        histories=[]
        for input_name,initial in [('arbitrary_complex_joint',v),('prepared_pqc_joint',prepared[:256])]:
            state=np.zeros(512,complex);state[:256]=initial;ideal=initial.copy();state=sim.run(state,ph)
            errors=[];leaks=[]
            for step in range(33):
                state=sim.run(state,radial+(ph if step==32 else pf))
                ideal=ideal_step(ideal,r,dt,mu,gamma)
                snap=state if step==32 else sim.run(state,inverse(ph))
                errors.append(aligned_error(snap[:256],ideal)[0]);leaks.append(float(np.sum(abs(snap[256:])**2)))
            assert max(errors)<1e-10 and max(leaks)<1e-20
            measured=sim.run(state,readout)
            measured_ideal=np.zeros(512,complex);measured_ideal[:256]=ideal;measured_ideal=sim.run(measured_ideal,readout)
            probdiff=float(np.max(np.abs(np.abs(measured)**2-np.abs(measured_ideal)**2)))
            histories.append({'input':input_name,'max_33step_joint_state_l2_error':max(errors),'final_error':errors[-1],'max_work_leakage_probability':max(leaks),'joint_readout_max_probability_error':probdiff,'per_step_state_errors':errors})
        allgates=prep+ph+sum((radial+(ph if j==32 else pf) for j in range(33)),[])+readout
        counts=inventory(allgates);counts['elementary_depth']=depth(allgates,9)
        init=inventory(prep+pf+readout);step=inventory(radial+pf)
        result={'counts':counts,'initial_plus_one_phase_and_readout':init,'per_radial_step_plus_phase':step,'increment_high_level':inventory(high),'increment_elementary':inventory(inc),'radial_step':inventory(radial),'prep':inventory(prep),'phase_layer':inventory(pf),'increment_operator_norm_error_all_256_inputs':inc_error,'decrement_operator_norm_error_all_256_inputs':dec_error,'one_step_operator_norm_error_all_256_inputs':op_error,'one_step_work_leakage_operator_norm':leakage,'simulations':histories}
        assert counts['RZ']==4784+len(angles) and counts['native_T']=={'A':7392,'B':12936}[name]
        report['methods'][name]=result
        gates={'nqubits':9,'radius_qubits':6,'phase_probes':[6,7],'work':[8],'steps':33,'prep':prep,'inc':inc,'radial':radial,'phase_half':ph,'phase_full':pf,'readout':readout}
        (out/f'method_{name}_gates.json').write_text(json.dumps(gates,separators=(',',':')))
        (out/f'method_{name}_address_high_level.json').write_text(json.dumps(high,separators=(',',':')))
        print(name,counts,'maxerr',max(z['max_33step_joint_state_l2_error'] for z in histories),flush=True)
    report['limitations']=['Arbitrary RZ gates are simulated exactly; approximate Clifford+T rotation strings are not certified here.','Only native address Toffolis are expanded into exact Clifford+T.','Universal operator checks apply to any initial amplitude but do not establish physical ROM accuracy.','B uses Cuccaro MAJ/UMA subtraction widgets inside the Gidney borrowed-register construction; its CX counts replace the older analytical widget estimate.']
    (out/'verification.json').write_text(json.dumps(report,indent=2))
    print(out/'verification.json',flush=True)


if __name__=='__main__':main()
