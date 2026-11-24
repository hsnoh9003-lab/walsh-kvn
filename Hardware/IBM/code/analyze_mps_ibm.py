"""Independent offline replay of IBM MPS preparation and reference calibration."""
from pathlib import Path
from collections import Counter
from importlib.metadata import version
import csv
import json
import math
import platform
import re
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm
from analyze_ibm import number, single

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'


def target_masses(n):
    size=2**n;edges=np.linspace(0,1,size+1);centres=(np.arange(size)+.5)/size
    def masses(loc):
        p=np.diff(norm.cdf(edges,loc,.25));return p/p.sum()
    return masses(brentq(lambda loc:masses(loc)@centres-.9017043,.5,3))


def mps_state(target,n):
    # Sequential Schmidt truncation to bond dimension two, in the supplied bit order.
    blocks=[];matrix=target.reshape(1,-1);rank=1
    for _ in range(n-1):
        matrix=matrix.reshape(rank*2,-1)
        u,s,vh=np.linalg.svd(matrix,full_matrices=False)
        rank=min(2,len(s));blocks.append(u[:,:rank].reshape(-1,2,rank))
        matrix=s[:rank,None]*vh[:rank]
    blocks.append(matrix.reshape(rank,2,1));state=blocks[0]
    for block in blocks[1:]:state=np.tensordot(state,block,axes=([-1],[0]))
    state=state.ravel();return state/np.linalg.norm(state)


def simulate_qasm(path):
    source=path.read_text();n=int(re.search(r'bit\[(\d+)\] meas',source)[1]);size=2**n
    physical='$' in source
    active=sorted(set(map(int,re.findall(r'\$(\d+)',source)))) if physical else list(range(n))
    assert len(active)==n, 'This archive uses only the measured position qubits.'
    positions={q:i for i,q in enumerate(active)};state=np.zeros(2**len(active),complex);state[0]=1
    counts=Counter();measurements={}
    for line in source.splitlines():
        line=line.strip()
        if not line or line.startswith(('OPENQASM','include','bit[','qubit[','barrier','//','gphase')):continue
        if 'measure' in line:
            bit=int(re.search(r'meas\[(\d+)\]',line)[1])
            q=int(re.search(r'\$(\d+)',line)[1]) if physical else int(re.search(r'q\[(\d+)\]',line)[1])
            measurements[bit]=positions[q];continue
        match=re.fullmatch(r'([\w]+)(?:\((.*?)\))?\s+(.*);',line)
        if not match:raise ValueError(line)
        name,args,target=match.groups();params=[number(s.strip()) for s in args.split(',')] if args else []
        qs=[positions[int(q)] for q in re.findall(r'\$(\d+)' if physical else r'q\[(\d+)\]',target)]
        counts[name]+=1
        if name in ('cx','cz'):
            control,target=qs
            for j in range(len(state)):
                if (j>>control)&1:
                    if name=='cz' and (j>>target)&1:state[j]*=-1
                    elif name=='cx' and not (j>>target)&1:
                        jj=j|(1<<target);state[j],state[jj]=state[jj],state[j]
        else:
            if name.lower()=='u':
                theta,phi,lam=params;c,s=math.cos(theta/2),math.sin(theta/2)
                gate=np.array([[c,-np.exp(1j*lam)*s],[np.exp(1j*phi)*s,np.exp(1j*(phi+lam))*c]])
            elif name=='rz':gate=np.diag(np.exp(np.array([-1j,1j])*params[0]/2))
            elif name=='sx':gate=np.array([[1+1j,1-1j],[1-1j,1+1j]])/2
            elif name=='x':gate=np.array([[0,1],[1,0]])
            else:raise ValueError(name)
            single(state,gate,qs[0])
    ordered=np.zeros(size,complex)
    for j,value in enumerate(state):
        index=sum(((j>>q)&1)<<bit for bit,q in measurements.items());ordered[index]+=value
    return ordered,dict(counts)


def propagate(initial,nt,k):
    state=initial.copy();size=len(state);delta=(49.8291-40)/nt
    def layer(odd,duration):
        for j in range(1 if odd else 0,size-1,2):
            face=(j+1)/size;angle=.0924819*(face-face**3)*duration*size
            c,s=math.cos(angle/2),math.sin(angle/2)
            state[j:j+2]=np.array([[c,-s],[s,c]])@state[j:j+2]
    for _ in range(k):layer(False,delta/2);layer(True,delta);layer(False,delta/2)
    return state


def csv_write(path,rows):
    keys=list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def continuum_mean(p0,times):
    size=len(p0);nodes,weights=np.polynomial.legendre.leggauss(64)
    initial=(np.arange(size)[:,None]+(nodes[None,:]+1)/2)/size
    answer=[]
    for time in times:
        e=np.exp(.0924819*time);radial=initial*e/np.sqrt(1+initial**2*(e**2-1))
        answer.append(float(np.sum(p0[:,None]*weights[None,:]*radial/2)))
    return answer


def main():
    initial={n:mps_state(np.sqrt(target_masses(n)),n) for n in [4,5,6]}
    preparation=[]
    for n,state in initial.items():
        target=np.sqrt(target_masses(n));size=2**n;centre=(np.arange(size)+.5)/size
        preparation.append({'qubits':n,'cells':size,'bond_dimension':2,
            'target_mean':float(target**2@centre),'prepared_mean':float(state**2@centre),
            'fidelity':float(abs(target@state)**2),'target_TVD':float(abs(state**2-target**2).sum()/2)})
    errors={'logical_vs_device_probabilities':0.,'QASM_vs_independent_pair_rotations':0.,
            'reference_vs_MPS_initial':0.,'logical_loader_infidelity':0.,
            'logical_device_state_infidelity':0.,'logical_pair_state_infidelity':0.}
    gates=[];ideal={};checked=0
    groups=[('dT4_n4n5',[(4,4,4),(5,4,1)]),('dT6_n4',[(4,6,6)])]
    for tag,sets in groups:
        for path in sorted((ROOT/f'circuits/logical/qasm3_mps_{tag}').glob('*.qasm')):
            n=int(path.stem[1]);k=int(path.stem.split('_k')[-1]);nt=sets[0][1]
            state,logical=simulate_qasm(path)
            device,native=simulate_qasm(ROOT/f'circuits/ibm_cleveland_isa/qasm3_mps_{tag}'/path.name)
            probability=abs(state)**2;expected=initial[n] if '_ref_' in path.stem else propagate(initial[n],nt,k)
            errors['logical_device_state_infidelity']=max(errors['logical_device_state_infidelity'],float(abs(1-abs(np.vdot(state,device))**2)))
            errors['logical_pair_state_infidelity']=max(errors['logical_pair_state_infidelity'],float(abs(1-abs(np.vdot(expected,state))**2)))
            errors['logical_vs_device_probabilities']=max(errors['logical_vs_device_probabilities'],float(abs(probability-abs(device)**2).max()))
            key='reference_vs_MPS_initial' if '_ref_' in path.stem else 'QASM_vs_independent_pair_rotations'
            errors[key]=max(errors[key],float(abs(probability-abs(expected)**2).max()))
            if k==0:errors['logical_loader_infidelity']=max(errors['logical_loader_infidelity'],float(abs(1-abs(np.vdot(initial[n],state))**2)))
            ideal[(tag,path.stem)]=probability;checked+=1
            gates.append({'set':tag,'qubits':n,'cells':2**n,'nt':nt,'circuit':path.stem,
                          'logical_CX':logical.get('cx',0),'CZ':native.get('cz',0)})
    assert errors['logical_vs_device_probabilities']<3e-12
    assert errors['QASM_vs_independent_pair_rotations']<3e-12
    assert errors['reference_vs_MPS_initial']<3e-12
    assert errors['logical_loader_infidelity']<3e-12
    assert errors['logical_device_state_infidelity']<3e-12 and errors['logical_pair_state_infidelity']<3e-12
    rows=[];diagnostics=[];central_difference=0.;bootstrap_difference=0.
    for tag,sets in groups:
        counts=json.loads((DATA/f'ibm_mps_{tag}_counts.json').read_text());rng=np.random.default_rng(0)
        stored={(int(r['cells']),int(r['k'])):r for r in csv.DictReader((DATA/f'results_ibm_mps_{tag}.csv').open())}
        for n,nt,kmax in sets:
            size=2**n;centres=(np.arange(size)+.5)/size;top=centres>=7/8
            p0=abs(initial[n])**2;uniform=np.ones(size)/size
            def measured(name):
                result=np.zeros(size)
                for key,value in counts[name].items():result[int(key,2)]+=value
                shots=int(result.sum());return result/shots,shots
            def corrected(pk,pr):
                a=p0-uniform;contrast=float(a@(pr-uniform)/(a@a));return contrast,p0+(pk-pr)/contrast
            for k in range(kmax+1):
                name=f'n{n}_kvn_k{k}';pk,shots=measured(name);pid=ideal[(tag,name)]
                row={'cells':size,'qubits':n,'nt':nt,'k':k,'t_star':40+k*(49.8291-40)/nt,'shots':shots,
                     'P_ideal':float(pid[top].sum()),'r_ideal':float(pid@centres),
                     'r_continuum':continuum_mean(p0,[k*(49.8291-40)/nt])[0],
                     'P_raw':float(pk[top].sum()),'r_raw':float(pk@centres),
                     'P_raw_SE':float(np.sqrt(pk[top].sum()*(1-pk[top].sum())/shots)),
                     'r_raw_SE':float(np.sqrt((pk@centres**2-(pk@centres)**2)/shots))}
                if k==0:
                    row.update(F_ref='',P_refnorm='',P_refnorm_err='',r_refnorm='',r_refnorm_err='')
                    preparation[n-4][f'k0_raw_TVD_{tag}']=float(abs(pk-p0).sum()/2)
                else:
                    pr,refshots=measured(f'n{n}_ref_k{k}');contrast,pc=corrected(pk,pr);boot=[]
                    for _ in range(1000):
                        _,sample=corrected(rng.multinomial(shots,pk)/shots,rng.multinomial(refshots,pr)/refshots)
                        boot.append([sample[top].sum(),sample@centres])
                    boot=np.asarray(boot)
                    row.update(F_ref=contrast,P_refnorm=float(pc[top].sum()),r_refnorm=float(pc@centres),
                               P_refnorm_err=float(boot[:,0].std()),r_refnorm_err=float(boot[:,1].std()))
                    diagnostics.append({'cells':size,'nt':nt,'k':k,'F_ref':contrast,
                        'reference_depolarizing_TVD':float(abs(pr-(contrast*p0+(1-contrast)*uniform)).sum()/2),
                        'corrected_mean':float(pc@centres),'largest_cell_centre':float(centres[-1]),
                        'corrected_r2':float(pc@centres**2),'corrected_variance':float(pc@centres**2-(pc@centres)**2),
                        'corrected_negative_mass':float(-np.minimum(pc,0).sum()),
                        'P_bias_over_bootstrap_SE':float((pc[top].sum()-pid[top].sum())/row['P_refnorm_err']),
                        'r_bias_over_bootstrap_SE':float((pc@centres-pid@centres)/row['r_refnorm_err'])})
                    for key in ['F_ref','P_refnorm','r_refnorm','P_refnorm_err','r_refnorm_err']:
                        difference=abs(row[key]-float(stored[(size,k)][key]))
                        if key.endswith('_err'):
                            relative=difference/float(stored[(size,k)][key]);bootstrap_difference=max(bootstrap_difference,relative)
                            assert relative<.03,(size,nt,k,key,relative)
                        else:
                            central_difference=max(central_difference,difference);assert difference<5.1e-6,(size,nt,k,key)
                rows.append(row)
    from Source.data_locations import published_path
    comparison=published_path('Data/Details/input/direct_quantum_history.npz')
    old=np.load(comparison)['prepared_amplitude'];fidelity=float(abs(np.vdot(old,initial[6]))**2)
    assert fidelity>1-1e-10
    support={}
    for n in [4,5,6]:
        size=2**n;count=size//2;result={}
        for parity in [0,1]:
            faces=(2*np.arange(count)+1+parity)/size;angles=.0924819*(faces-faces**3)*size
            coeff=np.array([sum(angles[q]*(-1)**((q&mask).bit_count()) for q in range(count))/count for mask in range(count)])
            result['even' if parity==0 else 'odd']=int(np.count_nonzero(abs(coeff)>1e-12))
        support[str(size)]=result
    assert support=={'16':{'even':8,'odd':8},'32':{'even':15,'odd':15},'64':{'even':26,'odd':26}}
    report={'analysis_versions':{'python':platform.python_version(),'numpy':np.__version__,'scipy':version('scipy')},
        'MPS_logical_device_circuit_pairs':checked,'checks':errors,'preparation':preparation,
        'Walsh_terms_per_layer':support,
        '64_cell_overlap_fidelity_with_manuscript_prepared_amplitude':fidelity,
        'target_mean_rounding': 'New hardware target mean 0.9017043; original field target 0.9017043047820993.',
        'original_CSV_max_central_rounding_difference':central_difference,
        'original_CSV_max_bootstrap_relative_difference':bootstrap_difference,
        'bootstrap':{'resamples':1000,'seed_per_job':0,'scope':'Multinomial count resampling only; excludes systematic calibration mismatch.'},
        '64_cell_scope':'Input approximation checked offline; no 64-cell hardware transport is reported.'}
    csv_write(DATA/'ibm_mps_calibration_full_precision.csv',rows)
    csv_write(DATA/'ibm_mps_reference_diagnostics.csv',diagnostics)
    csv_write(DATA/'ibm_mps_gate_counts.csv',gates)
    csv_write(DATA/'ibm_mps_preparation_validation.csv',preparation)
    (DATA/'ibm_mps_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
