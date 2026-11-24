"""Compile and gate-simulate 128-cell refinement of the same prepared input.

The original six-bit MPS is tensored with a low-order |+> bit, preserving the
same piecewise-uniform continuum density exactly. No new fit or input model.
Uses the archived one-clean method A and sparse Walsh compiler unchanged.
"""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
from pathlib import Path
import sys,json,csv,itertools,math,time
from collections import Counter
import numpy as np
from numpy.polynomial.legendre import leggauss
PACKAGE=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(PACKAGE))
from Source.data_locations import shared_data
ROOT=shared_data('Radial_refinement', 'Radial128')
from Source.KvN.native_simulator import Simulator,aligned_error
from Source.KvN.one_clean_synthesis import increment_A,expand,inverse,inventory
from Source.KvN.sparse_walsh import controlled_ry,phase_layer


def pair_step(v,n,dt,mu,gamma):
    size=1<<n;h=1/size;r=(np.arange(size)+.5)*h
    state=np.array(v,complex,copy=True).reshape(4,size)
    phase=np.exp(.5j*gamma*dt*np.array([0,1,1,2])[:,None]*r*r)
    state*=phase
    for parity,duration in [(0,dt/2),(1,dt),(0,dt/2)]:
        starts=np.arange(parity,size-1,2)
        faces=(starts+1)*h
        theta=mu*faces*(1-faces*faces)*duration/(2*h)
        a,b=state[:,starts].copy(),state[:,starts+1].copy()
        co,si=np.cos(theta),np.sin(theta)
        state[:,starts]=co*a-si*b
        state[:,starts+1]=si*a+co*b
    return (state*phase).reshape(-1)


def field_budget(probabilities,values,reference,H,eta=1e-4,target=.01):
    mean=probabilities@values
    cov=(values.T*probabilities)@values-np.outer(mean,mean)
    delta=mean-reference
    b2=float(delta@H@delta);variance=float(np.sum(H*cov))
    ranges=np.ptp(values,axis=0)
    product_ranges=np.ptp(values[:,:,None]*values[:,None,:],axis=0)
    change=(product_ranges+abs(mean)[:,None]*ranges[None,:]+abs(mean)[None,:]*ranges[:,None])*eta+np.outer(ranges,ranges)*eta**2
    effective_b2=max(float((delta+np.asarray(signs)*ranges*eta)@H@(delta+np.asarray(signs)*ranges*eta)) for signs in itertools.product((-1,1),repeat=3))
    effective_v=variance+float(np.sum(abs(H)*change))
    shots=max(1,math.ceil(effective_v/(target**2-effective_b2)))
    assert effective_b2+effective_v/shots<=target**2
    assert shots==1 or effective_b2+effective_v/(shots-1)>target**2
    return dict(noiseless_field_bias_pct=100*math.sqrt(b2),nominal_variance=variance,
                implementation_error_budget=eta,field_error_target=target,
                protected_bias_bound=math.sqrt(effective_b2),protected_variance_bound=effective_v,
                shots=shots,RMS_bound=math.sqrt(effective_b2+effective_v/shots),
                RMS_bound_at_one_fewer=math.sqrt(effective_b2+effective_v/(shots-1)))


def run(n,steps,cfg,original,hist,readout,output_dir=None):
    tick=time.monotonic();size=1<<n;width=n+3;dt=(float(hist['times'][-1])-40)/steps
    mu=cfg['parameters']['mu'];gamma=cfg['parameters']['gamma_normalized']
    r=(np.arange(size)+.5)/size
    shift=n-6
    assert shift in [0,1]
    prep=[(g,[q+shift for q in qs],a) for g,qs,a in original['prep']]
    if shift:prep=[('h',[0],None)]+prep
    ef=(2*np.arange(size//2)+1)/size;of=(2*np.arange(size//2)+2)/size
    even=controlled_ry(mu*ef*(1-ef**2)*dt*size/2,list(range(1,n)),0)
    odd=controlled_ry(mu*of*(1-of**2)*dt*size,list(range(1,n)),0)
    inc=expand(increment_A(list(range(n)),n+2));dec=inverse(inc)
    radial=even+dec+odd+inc+even
    half=phase_layer(r,gamma*dt/2,n,(n,n+1));full=phase_layer(r,gamma*dt,n,(n,n+1))
    terminal=[('h',[n],None),('sdg',[n+1],None),('h',[n+1],None)]
    gates=prep+half+sum((radial+(half if j==steps-1 else full) for j in range(steps)),[])+terminal
    counts=inventory(gates)
    if (n,steps)==(6,33):
        assert counts['RZ']==4803 and counts['native_T']==7392 and counts['CX']==12577
    sim=Simulator(width);state=np.zeros(1<<width,complex);state[0]=1
    state=sim.run(state,prep)
    amp=hist['prepared_amplitude'] if shift==0 else np.repeat(hist['prepared_amplitude'],2)/np.sqrt(2)
    expected=np.tile(amp,4)/2
    prep_error=aligned_error(state[:4*size],expected)[0]
    assert prep_error<1e-12
    # Verify the expanded increment on every clean-work input basis column.
    basis=np.zeros((1<<width,4*size),complex);basis[:4*size]=np.eye(4*size)
    moved=sim.run(basis,inc);expected_perm=np.zeros_like(basis)
    indices=np.arange(4*size);expected_perm[(indices&~(size-1))|((indices+1)&(size-1)),indices]=1
    increment_error=float(np.linalg.norm(moved-expected_perm))
    assert increment_error<1e-11
    # One-step Frobenius error bounds the operator norm on this full sector.
    full_step=half+radial+half
    actual=sim.run(basis,full_step)
    exact=np.column_stack([pair_step(np.eye(4*size,dtype=complex)[:,i],n,dt,mu,gamma) for i in range(4*size)])
    operator_error=aligned_error(actual[:4*size],exact)[0]
    operator_leakage=float(np.linalg.norm(actual[4*size:]))
    assert operator_error<1e-10 and operator_leakage<1e-11
    direct=expected.copy();history=[direct.copy()];errors=[];leakage=[]
    for j in range(steps):
        state=sim.run(state,full_step)
        direct=pair_step(direct,n,dt,mu,gamma)
        errors.append(aligned_error(state[:4*size],direct)[0]);leakage.append(float(np.sum(abs(state[4*size:])**2)))
        history.append(state[:4*size].copy())
    assert max(errors)<1e-10 and max(leakage)<1e-20
    measured=sim.run(state,terminal)
    # Check the merged-phase gate program independently, including every gate.
    zero=np.zeros(1<<width,complex);zero[0]=1
    merged=sim.run(zero,gates)
    merged_error=aligned_error(merged,measured)[0]
    assert merged_error<1e-10
    probabilities=abs(merged[:4*size])**2;probabilities/=probabilities.sum()
    values=np.c_[np.tile(r,4)*np.repeat([1,-1,1,-1],size),
                 np.tile(r,4)*np.repeat([1,1,-1,-1],size),np.tile(r*r,4)]
    H=readout['M'].T@readout['G']@readout['M']/float(readout['norm'])**2
    budget=field_budget(probabilities,values,readout['target'],H)
    if (n,steps)==(6,33):
        assert budget['shots']==815
        assert abs(budget['noiseless_field_bias_pct']-.10720784802292105)<1e-9
    if (n,steps)==(7,66):
        assert abs(budget['noiseless_field_bias_pct']-.048855915137143366)<1e-9
    if output_dir is None:
        output_dir = PACKAGE/'Build/Radial128'/(f'grid{size}' if (n,steps) in {(6,33),(7,66)} else f'grid{size}_steps{steps}')
    out=Path(output_dir)
    out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'states_and_readout.npz',times=np.linspace(40,float(hist['times'][-1]),steps+1),
                        history=np.asarray(history),r=r,initial_amplitude=amp,probabilities=probabilities,
                        values=values,reference=readout['target'],H=H)
    (out/'gate_program.json').write_text(json.dumps(dict(nqubits=width,steps=steps,gates=gates),separators=(',',':')))
    report=dict(cells=size,address_qubits=n,logical_qubits=width,steps=steps,
                start_time=40.,end_time=float(hist['times'][-1]),input='same actual six-bit MPS tensored with a low-order |+> refinement bit' if shift else 'original six-bit MPS',
                MPS_bond_dimension=2,nominal_success_probability=1.,counts=counts,budget=budget,
                preparation_error=prep_error,increment_Frobenius_error=increment_error,
                full_input_sector_one_step_Frobenius_error=operator_error,
                full_input_sector_one_step_leakage_Frobenius=operator_leakage,
                max_full_evolution_state_error=max(errors),max_work_leakage=max(leakage),
                merged_program_state_error=merged_error,arbitrary_rotations_simulated_exactly=True,
                approximate_Clifford_T_rotation_strings_not_simulated=True,elapsed_seconds=time.monotonic()-tick)
    (out/'flow_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['cells','logical_qubits','steps','counts','budget','elapsed_seconds']}),flush=True)
    return report


if __name__=='__main__':
    cfg=json.loads((ROOT/'inputs/configuration.json').read_text())
    original=json.loads((ROOT/'inputs/direct_64_33_circuit.json').read_text())
    hist=np.load(ROOT/'inputs/direct_quantum_history.npz');readout=np.load(ROOT/'inputs/joint_readout_inputs.npz')
    reports=[run(n,steps,cfg,original,hist,readout) for n,steps in [(6,33),(7,66)]]
    rows=[dict(cells=r['cells'],address_qubits=r['address_qubits'],logical_qubits=r['logical_qubits'],
               steps=r['steps'],RZ=r['counts']['RZ'],native_T=r['counts']['native_T'],
               noiseless_field_bias_pct=r['budget']['noiseless_field_bias_pct'],shots=r['budget']['shots']) for r in reports]
    output_table=PACKAGE/'Build/Radial128/grid_comparison.csv'
    output_table.parent.mkdir(parents=True, exist_ok=True)
    with output_table.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
