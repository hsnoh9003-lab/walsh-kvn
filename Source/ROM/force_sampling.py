"""Joint 815-shot drag/lift estimates from the actual signed-MPS wake circuit.

Four terminal Pauli settings isolate the second phase coherence needed by
drag. Each time point uses 815 shots TOTAL, shared by both force estimators,
in addition to the separately budgeted velocity-field readout. No ODE or
classical phase samples replace the saved quantum state.
"""

# Public Source Data mapping (single canonical file per published input).
import sys as _kvn_path_sys
from pathlib import Path as _kvn_path_Path
_kvn_path_sys.path.insert(0, str(_kvn_path_Path(__file__).resolve().parents[2]))
from Source.data_locations import published_path, source_label, shared_data
from pathlib import Path
import argparse, csv, hashlib, json
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SETTINGS = ('XX', 'XY', 'YX', 'YY')  # high phase qubit, low phase qubit
COUNTS = np.array([204, 204, 204, 203])

def write_csv(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)

def evaluate(output=None, seed=20261116, repetitions=2000):
    out = Path(output) if output else ROOT/'Build/Force_Sampling'
    out.mkdir(parents=True, exist_ok=True)
    source = shared_data('CFD_ROM', 'CFD_ROM', 'Re100', 'data')
    with np.load(source/'direct_quantum_history.npz') as a:
        state, times, r = a['direct_history'], a['times'], a['r']
        initial = a['prepared_amplitude']
    with np.load(source/'unified_reference.npz') as a:
        phase = a['known_phase']
    np.testing.assert_allclose(state[0], np.tile(initial, (4, 1))/2, atol=2e-13, rtol=0)
    coeff_file = published_path('Figure/F4/data/plot_values/Re100/force_coefficients.csv')
    coefficients = list(csv.DictReader(coeff_file.open()))
    cd = {x['term']:float(x['coefficient']) for x in coefficients if x['observable']=='C_D'}
    cl = {x['term']:float(x['coefficient']) for x in coefficients if x['observable']=='C_L'}
    fem = np.genfromtxt(published_path('Figure/F4/data/force_output/fig4f_fem_forces.csv'), delimiter=',', names=True)
    exact_file = published_path('Figure/F4/data/force_output/fig4f_walsh_forces.csv')
    exact = np.genfromtxt(exact_file, delimiter=',', names=True)
    np.testing.assert_allclose(times, exact['time'], rtol=0, atol=1e-13)
    x = np.array([[0,1],[1,0]], complex); y = np.array([[0,-1j],[1j,0]], complex)
    h = np.array([[1,1],[1,-1]], complex)/np.sqrt(2)
    rot = {'X':h, 'Y':h @ np.diag([1,-1j])}
    signs = 1-2*((np.arange(4)[:,None]>>np.arange(2))&1)
    r2 = r*r; base = cd['1']+(cd['z']+(cd['x^2']+cd['y^2'])/2)*r2+cd['z^2']*r2*r2
    rng = np.random.default_rng(seed); draws=[]; trials=np.zeros((repetitions,len(times),2))
    rows=[]; checks=[]; distributions=[]; value_arrays=[]
    for t,(psi,theta) in enumerate(zip(state,phase)):
        # The four Pauli combinations isolate 4*conj(psi_00)*psi_11.
        u = r2/2*((cd['x^2']-cd['y^2'])*np.cos(2*theta)+cd['xy']*np.sin(2*theta))
        v = r2/2*(-(cd['x^2']-cd['y^2'])*np.sin(2*theta)+cd['xy']*np.cos(2*theta))
        a = r*(cl['x']+cl['xz']*r2); b = r*(cl['y']+cl['yz']*r2)
        alpha = a*np.cos(theta)+b*np.sin(theta)
        beta = -a*np.sin(theta)+b*np.cos(theta)
        means=[]; covariances=[]; sampled=[]; lower=[]; upper=[]
        for setting,count in zip(SETTINGS,COUNTS):
            basis = np.kron(rot[setting[0]],rot[setting[1]])
            probability = abs(basis @ psi)**2
            probability /= probability.sum()
            product = signs[:,0]*signs[:,1]
            weight = {'XX':u,'YY':-u,'XY':v,'YX':v}[setting]
            d = base[None,:]+4*product[:,None]*weight[None,:]
            l = np.zeros_like(d)
            if setting[1]=='X': l += 2*signs[:,0,None]*alpha
            if setting[0]=='Y': l += 2*signs[:,1,None]*beta
            values = np.stack([d.ravel(),l.ravel()],axis=1)
            p = probability.ravel(); mean = p @ values
            covariance = values.T @ (p[:,None]*values)-np.outer(mean,mean)
            counts = rng.multinomial(int(count),p)
            sampled.append(counts @ values/count); means.append(mean); covariances.append(covariance/count)
            lower.append(values.min(axis=0));upper.append(values.max(axis=0))
            for j,n in enumerate(counts):
                if n: draws.append({'step':t,'setting':setting,'outcome':j,'count':int(n)})
            if repetitions:
                repeated = rng.multinomial(int(count),p,size=repetitions)
                trials[:,t,:] += repeated @ values/(4*count)
            distributions.append(p); value_arrays.append(values)
        mean=np.mean(means,axis=0); cov=np.sum(covariances,axis=0)/16
        estimate=np.mean(sampled,axis=0)
        target=np.array([exact['C_D'][t],exact['C_L'][t]])
        np.testing.assert_allclose(mean,target,atol=2e-13,rtol=0)
        # Independently check Pauli identities, including complex coherences.
        for operator,component in [(np.kron(x,x)-np.kron(y,y),'real'),
                                   (np.kron(x,y)+np.kron(y,x),'imag')]:
            actual=np.einsum('ir,ij,jr->r',psi.conj(),operator,psi).real
            expected=getattr(4*psi[0].conj()*psi[3],component)
            np.testing.assert_allclose(actual,expected,atol=2e-15,rtol=0)
        ref=np.array([np.interp(times[t],fem['time'],fem[k]) for k in ['C_D','C_L']])
        row={'step':t,'time':float(times[t]),'shots':int(sum(COUNTS))}
        for i,name in enumerate(['C_D','C_L']):
            row.update({name+'_FEM':float(ref[i]),name+'_exact':float(mean[i]),
                name+'_815_shot':float(estimate[i]),name+'_standard_error':float(np.sqrt(max(0,cov[i,i]))),
                name+'_implementation_mean_bound_eta_1e-4':float(1e-4*np.mean(np.array(upper)[:,i]-np.array(lower)[:,i]))})
        row['estimator_covariance']=float(cov[0,1]); rows.append(row)
        checks.append(float(np.max(abs(mean-target))))
    write_csv(out/'force_815_shots.csv',rows)
    write_csv(out/'sampled_outcome_counts.csv',draws)
    summary=[]
    error_scales=[]
    for i,name in enumerate(['C_D','C_L']):
        reference=np.array([q[name+'_FEM'] for q in rows]); norm=np.mean(reference**2)
        ideal=np.array([q[name+'_exact'] for q in rows]); sample=np.array([q[name+'_815_shot'] for q in rows])
        variance=np.array([q[name+'_standard_error']**2 for q in rows])
        expected_mse=np.mean((ideal-reference)**2+variance)
        observed_mse=np.mean((trials[:,:,i]-reference)**2) if repetitions else None
        error_scales.append({'observable':name,'saved_times':len(rows),
            'FEM_RMS_normalizer':float(np.sqrt(norm)),
            'noiseless_absolute_RMS_residual':float(np.sqrt(np.mean((ideal-reference)**2))),
            'noiseless_relative_RMS_percent':float(100*np.sqrt(np.mean((ideal-reference)**2)/norm))})
        summary.append({'observable':name,'shots_per_time':815,'noiseless_FEM_relative_RMS_percent':float(100*np.sqrt(np.mean((ideal-reference)**2)/norm)),
            'one_seed_FEM_relative_RMS_percent':float(100*np.sqrt(np.mean((sample-reference)**2)/norm)),
            'expected_FEM_relative_RMS_percent':float(100*np.sqrt(expected_mse/norm)),
            'sampling_only_relative_RMS_percent':float(100*np.sqrt(np.mean(variance)/norm)),
            'repeated_sampling_MSE_over_exact_MSE':float(observed_mse/expected_mse) if repetitions else None})
    write_csv(out/'summary.csv',summary)
    write_csv(out/'force_error_scales.csv',error_scales)
    np.savez_compressed(out/'readout_distributions.npz',probabilities=np.array(distributions).reshape(len(times),4,-1),
                        estimator_values=np.array(value_arrays).reshape(len(times),4,-1,2),times=times)
    metadata={'shots_total_per_time':815,'setting_order_high_low':list(SETTINGS),
        'shots_per_setting':COUNTS.tolist(),'joint_force_estimators':True,'velocity_field_shots_separate':True,
        'terminal_basis_change':'H for X; S-dagger then H for Y; no new T gates or qubits',
        'T_work_per_attempt':'Same preparation and propagation as the radial circuit',
        'extra_force_attempts_all_34_states':815*len(times),'seed':seed,'repeated_sampling_batches':repetitions,
        'sampling_scope':'Independent ideal-distribution draws; not hardware-noise experiments or a 1% force guarantee',
        'maximum_expectation_discrepancy':max(checks),'input_convention':'Actual signed prepared MPS amplitudes, with the saved two coherent phase qubits',
        'plot_error_bars':'One theoretical standard deviation of each 815-shot estimator',
        'sources':{source_label(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [source/'direct_quantum_history.npz',source/'unified_reference.npz',coeff_file,exact_file]}}
    (out/'protocol.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps({'summary':summary,'protocol':metadata},indent=2))
    return summary,metadata

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path);parser.add_argument('--repetitions',type=int,default=2000)
    args=parser.parse_args();evaluate(args.output,repetitions=args.repetitions)
