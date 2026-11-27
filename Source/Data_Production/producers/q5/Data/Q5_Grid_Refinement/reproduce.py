"""Refine the SAME prepared Q5 input; distinguish fine-cell and coarse-bin errors.

Full evolutions use exact sparse-generator action and the implemented pair
product. Selected one-clean native gate programs are compiled and independently
verified for one step. This is not an upwind/LCHS matched resource comparison.
"""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import sys,json,csv,math,gzip,time,itertools,hashlib
import numpy as np
from numpy.polynomial import Polynomial
from numpy.polynomial.legendre import leggauss
from scipy.integrate import solve_ivp
from scipy.sparse import diags
from scipy.sparse.linalg import expm_multiply
ROOT=Path(__file__).resolve().parent
PACKAGE=ROOT.parents[1]
sys.path.insert(0,str(PACKAGE))
from Source.KvN.radial_mps import prepare_amplitude,compile_rz
from Source.KvN.sparse_walsh import controlled_ry,walsh
from Source.KvN.one_clean_synthesis import increment_A,expand,inverse,inventory
from Source.KvN.native_simulator import Simulator,aligned_error
INPUT=PACKAGE/'Data/Grid_Oscillation_Diagnostics/inputs/quintic_multistable__boundary_clipped.npz'

def save_csv(name,rows):
    with (ROOT/name).open('w',newline='') as f:
        keys=list(dict.fromkeys(k for r in rows for k in r))
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def split_step(v,N,dt,f):
    v=v.copy()
    for parity,weight in [(0,.5),(1,1.),(0,.5)]:
        lo=np.arange(parity,N-1,2);angle=f((lo+1)/N)*N*dt*weight/2
        c,s=np.cos(angle),np.sin(angle);a,b=v[lo].copy(),v[lo+1].copy()
        shape=(-1,)+(1,)*(v.ndim-1);c=c.reshape(shape);s=s.reshape(shape)
        v[lo]=c*a-s*b;v[lo+1]=s*a+c*b
    return v

def compile_selected(N,steps,f,amp0,program6,expected):
    n=N.bit_length()-1;k=n-6;dt=1/steps;work=n
    prep=[('h',[q],None) for q in range(k)]+[(g,[q+k for q in qs],a) for g,qs,a in program6]
    faces_e=(2*np.arange(N//2)+1)/N;faces_o=(2*np.arange(N//2)+2)/N
    even=controlled_ry(dt*N*f(faces_e)/2,list(range(1,n)),0)
    odd=controlled_ry(dt*N*f(faces_o),list(range(1,n)),0)
    inc=expand(increment_A(list(range(n)),work));dec=inverse(inc)
    step=even+dec+odd+inc+even
    a=inventory(prep);b=inventory(step)
    counts={key:int(a[key]+steps*b[key]) for key in ['RZ','native_T','CX','Clifford']}
    sim=Simulator(n+1);z=np.zeros(2*N,complex);z[0]=1
    prepared=sim.run(z,prep)
    amp=np.repeat(amp0,N//64)/np.sqrt(N//64)
    prep_error=aligned_error(prepared[:N],amp)[0]
    assert prep_error<3e-10,prep_error
    # Complete classical truth table for the unexpanded reversible increment.
    bits=np.arange(2*N,dtype=np.int64)
    for name,qs,angle in increment_A(list(range(n)),work):
        if name=='x':bits^=1<<qs[0]
        elif name=='cx':bits^=((bits>>qs[0])&1)<<qs[1]
        elif name=='ccx':bits^=(((bits>>qs[0])&1)&((bits>>qs[1])&1))<<qs[2]
        else:raise ValueError(name)
    assert np.array_equal(bits[:N],(np.arange(N)+1)%N)
    assert (bits[:N]<N).all()
    # Independent native one-step checks on the prepared and random complex inputs.
    rng=np.random.default_rng(20261114+N)
    v=np.c_[amp,rng.normal(size=N)+1j*rng.normal(size=N),rng.normal(size=N)+1j*rng.normal(size=N)]
    v/=np.linalg.norm(v,axis=0)
    start=np.vstack([v,np.zeros_like(v)])
    actual=sim.run(start,step);target=split_step(v,N,dt,f)
    discrepancy=aligned_error(actual[:N],target)[0];leakage=float(np.linalg.norm(actual[N:]))
    assert discrepancy<3e-10 and leakage<3e-11,(discrepancy,leakage)
    # Full native run for the archived 64-cell baseline, as a transport regression.
    full_native_error=None
    if N==64:
        state=prepared
        for _ in range(steps):state=sim.run(state,step)
        full_native_error=aligned_error(state[:N],expected)[0]
        assert full_native_error<3e-10
    synthesis_budget=5e-5
    per_rotation=int(math.ceil(3*math.log2(counts['RZ']/synthesis_budget)))
    t_count=counts['native_T']+counts['RZ']*per_rotation
    error_sum=counts['RZ']*2.**(-per_rotation/3)
    assert error_sum<=synthesis_budget
    config=dict(cells=N,steps=steps,position_qubits=n,work_qubits=1,logical_qubits=n+1,
        preparation='original bond-two MPS with low-order |+> subdivision bits',
        inventory=counts,T_per_rotation=per_rotation,rotation_synthesis_error_budget=synthesis_budget,
        rotation_synthesis_model='error per RZ <= 2^(-T_per_rotation/3)',
        synthesized_T_count_proxy=t_count,rotation_error_sum_bound=error_sum,
        preparation_gate_error=prep_error,one_step_test_columns=3,
        one_step_native_vs_pair_error=discrepancy,clean_work_leakage=leakage,
        full_64_cell_native_run_error=full_native_error,
        full_refined_gate_by_gate_run=False,
        physical_resource_optimization=False)
    folder=ROOT/f'compiled_{N}';folder.mkdir(exist_ok=True)
    (folder/'inventory.json').write_text(json.dumps(config,indent=2)+'\n')
    with gzip.open(folder/'gate_program.json.gz','wt') as out:
        json.dump(dict(preparation=prep,repeated_transport_step=step,repetitions=steps,
                       gate_convention='RZ(theta)=exp(-i theta Z/2)',qubits=n+1),out)
    return config

def main(grids):
    tick=time.monotonic();data=np.load(INPUT);amp=data['state_history'][0].copy()
    amp/=np.linalg.norm(amp);p0=abs(amp)**2;old_edges=np.linspace(0,1,65)
    x=Polynomial([0.,1.]);raw=x*(1-x)*(x-.25)*(x-.5)*(x-.75)
    points=[0.,1.]+[r.real for r in raw.deriv().roots() if abs(r.imag)<1e-12 and 0<r.real<1]
    scale=.25/max(abs(raw(np.array(points))));f=scale*raw
    refs=[]
    for order in [32,64,128]:
        nodes,w=leggauss(order);positions=((np.arange(64)[:,None]+(nodes+1)/2)/64).ravel()
        weights=(p0[:,None]*w[None,:]/2).ravel()
        sol=solve_ivp(lambda t,y:f(y),(0,1),positions,method='DOP853',rtol=1e-12,atol=1e-14)
        assert sol.success
        y=sol.y[:,-1];refs.append(np.array([weights@y,weights@y**2]))
    mref=refs[-1];quad_error=max(float(np.linalg.norm(m-mref)) for m in refs[:-1])
    assert quad_error<3e-11,quad_error
    prepared,ry_program,prepmeta=prepare_amplitude(amp.real)
    assert prepmeta['amplitude_l2']<3e-10,prepmeta
    program6=compile_rz(ry_program)
    (ROOT/'mps_recompilation.json').write_text(json.dumps(dict(**prepmeta,program=program6),indent=2)+'\n')
    rows=[];compilations=[]
    for N in grids:
        assert N>=64 and N&(N-1)==0
        centres=(np.arange(N)+.5)/N;sub=N//64;initial=np.repeat(amp,sub)/np.sqrt(sub)
        faces=np.linspace(0,1,N+1);speed=f(faces[1:-1])*N/2
        G=diags([-speed,speed],[1,-1],format='csr')
        exact=expm_multiply(G,initial)
        split=initial.copy()
        for j in range(N):split=split_step(split,N,1/N,f)
        inv=solve_ivp(lambda t,y:-f(y),(0,1),faces,method='DOP853',rtol=1e-12,atol=1e-14)
        assert inv.success
        cdf=np.interp(np.clip(inv.y[:,-1],0,1),old_edges,np.r_[0,np.cumsum(p0)])
        refp=np.diff(cdf);coarse_ref=refp.reshape(64,sub).sum(axis=1)
        assert refp.min()>-1e-10 and abs(refp.sum()-1)<1e-11
        metrics=[]
        for label,state in [('exact_generator',exact),('pair_product',split)]:
            pq=abs(state)**2;mom=np.array([pq@centres,pq@(centres**2)])
            coarse=pq.reshape(64,sub).sum(axis=1);coarse_x=(np.arange(64)+.5)/64
            coarse_m=np.array([coarse@coarse_x,coarse@(coarse_x**2)])
            coarse_mref=np.array([coarse_ref@coarse_x,coarse_ref@(coarse_x**2)])
            delta=mom-mref;values=np.c_[centres,centres**2];ranges=np.ptp(values,axis=0)
            eta=1e-4;worst=max(float(np.linalg.norm(delta+eta*ranges*np.array(s))) for s in itertools.product((-1,1),repeat=2))/np.linalg.norm(mref)
            metrics.append(dict(cells=N,steps=N if label=='pair_product' else 0,method=label,
                initial_mean_before_propagation=float(p0@((np.arange(64)+.5)/64)),
                native_cell_TV=float(.5*abs(pq-refp).sum()),
                aggregated_64_bin_TV=float(.5*abs(coarse-coarse_ref).sum()),
                relative_continuum_moment_error=float(np.linalg.norm(delta)/np.linalg.norm(mref)),
                relative_rebinned_moment_error=float(np.linalg.norm(coarse_m-coarse_mref)/np.linalg.norm(coarse_mref)),
                first_fine_cell_mass=float(pq[0]),first_fine_cell_reference_mass=float(refp[0]),
                first_original_bin_mass=float(coarse[0]),first_original_bin_reference_mass=float(coarse_ref[0]),
                mean_x=float(mom[0]),second_moment=float(mom[1]),reference_mean_x=float(mref[0]),reference_second_moment=float(mref[1]),
                protected_relative_bias_bound_eta_1e_4=worst,
                passes_noiseless_1pct_moment_check=bool(np.linalg.norm(delta)/np.linalg.norm(mref)<.01),
                passes_1pct_bias_check_at_eta_1e_4=bool(worst<.01),
                norm_defect=float(abs(np.vdot(state,state).real-1))))
        if N==64:
            archive_error=aligned_error(split,data['state_history'][-1])[0]
            assert archive_error<3e-10,archive_error
            assert abs(metrics[1]['relative_continuum_moment_error']-1.9885713)<2e-7
        np.savez_compressed(ROOT/f'grid_{N}.npz',centres=centres,initial=initial,
            exact_generator=exact,pair_product=split,matched_characteristic_cell_mass=refp,
            reference_moments=mref)
        rows+=metrics;save_csv('grid_refinement.csv',rows)
        print(N,'fine TV',metrics[1]['native_cell_TV'],'64bin TV',metrics[1]['aggregated_64_bin_TV'],
            'continuum moment',metrics[1]['relative_continuum_moment_error'],
            'rebinned moment',metrics[1]['relative_rebinned_moment_error'],flush=True)
        if N in [64,512,2048]:
            c=compile_selected(N,N,f,amp,program6,split);compilations.append(c)
            print('compiled',N,c['inventory'],c['synthesized_T_count_proxy'],flush=True)
    save_csv('compiled_inventory.csv',[dict(cells=c['cells'],steps=c['steps'],logical_qubits=c['logical_qubits'],
        **c['inventory'],T_per_rotation=c['T_per_rotation'],synthesis_error_sum_bound=c['rotation_error_sum_bound'],
        synthesized_T_count_proxy=c['synthesized_T_count_proxy'],native_vs_pair_step_error=c['one_step_native_vs_pair_error']) for c in compilations])
    (ROOT/'validation.json').write_text(json.dumps(dict(status='PASS',input_sha256=hashlib.sha256(INPUT.read_bytes()).hexdigest(),
        input_source=str(INPUT.relative_to(PACKAGE)),input_mu=.02,input_sigma=.1,
        same_prepared_continuum_input=True,same_time_interval=[0,1],same_drift=True,drift_scale=float(scale),
        quadrature_orders=[32,64,128],maximum_quadrature_moment_difference=quad_error,
        fine_grid_definitions='Native TV uses each refined grid, not aggregation into 64 bins. Moments use continuous characteristic positions as the reference.',
        compiled_scope='MPS prep plus repeated one-clean native Clifford/T/RZ gates; arbitrary rotations are not individually synthesized. One step checked on three complex inputs; full native run checked only at N=64.',
        full_refined_evolution='Explicit even-half / odd-full / even-half pair product, N steps, same operator as compiled one-step circuit',
        physical_resource_optimization=False,matched_LCHS_comparison=False,elapsed_seconds=time.monotonic()-tick),indent=2)+'\n')

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--grids',type=int,nargs='+',default=[64,128,256,512,1024,2048,4096,8192]);a=ap.parse_args()
    main(a.grids)
