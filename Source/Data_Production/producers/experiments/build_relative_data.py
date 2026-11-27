"""Choose shots for the common 1% expected relative observable-RMS target.

The scalar observables are x,x^2 and the coupled observables x,y,x^2,y^2,xy.
Each observable vector is normalized by its own matched continuum reference
norm. A 1e-4 total-variation allowance bounds changes in its mean and covariance.
No fixed shot budget is used. Finite-shot ranges simulate nominal probabilities.
"""
from pathlib import Path
import csv, hashlib, json, math
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'updated_data'
OUT.mkdir(exist_ok=True)
ETA=1e-4
TARGET=.01
REPETITIONS=5000
MODELS={'quadratic':'Q2','cubic_interior':'C3','quintic_multistable':'Q5','coupled_cubic_2d':'2D C3'}
INPUTS=('interior','boundary_clipped','near_stable','near_unstable')

def write_csv(name, rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

records=[]; selected=[]; sample_rows=[]; workloads=[]; samples={}; sources=[]
for model,label in MODELS.items():
    for ic in INPUTS:
        ident=model+'__'+ic
        source=(ROOT/'grid32' if model=='coupled_cubic_2d' else ROOT)/(ident+'.npz')
        with np.load(source) as z:
            target=np.asarray(z['terminal_reference_moments'] if 'terminal_reference_moments' in z else z['prepared_cell_reference_moments'][-1])
            raw=np.asarray(z['probability_history'][-1]);coords=z['grid_coordinates'];names=z['observable_names'].tolist()
            stored=z['moment_history'][-1]
        prob=raw/raw.sum()
        obs={'x':coords[0],'x2':coords[0]**2}
        if len(coords)==2:obs.update(y=coords[1],y2=coords[1]**2,xy=coords[0]*coords[1])
        values=np.column_stack([obs[n] for n in names])
        mean=prob@values;bias=mean-target;norm_sq=float(target@target)
        assert norm_sq>0 and np.max(np.abs(raw@values-stored))<2e-10
        centered=values-mean
        variances=prob@(centered**2)
        widths=np.ptp(values,axis=0)
        square_widths=np.ptp(values**2,axis=0)
        bias_sq=float(bias@bias/norm_sq)
        variance=float(variances.sum()/norm_sq)
        bias_bound_sq=float(np.sum((np.abs(bias)+ETA*widths)**2)/norm_sq)
        variance_bound=float((variances.sum()+np.sum(ETA*(square_widths+2*np.abs(mean)*widths)+ETA**2*widths**2))/norm_sq)
        room=TARGET**2-bias_bound_sq
        shots=max(1,math.ceil(variance_bound/room)) if room>0 else None
        if shots:
            assert bias_bound_sq+variance_bound/shots<=TARGET**2+1e-16
            assert shots==1 or bias_bound_sq+variance_bound/(shots-1)>TARGET**2
        rec=dict(id=ident,model=label,input=ic,cells_per_coordinate=round(len(prob)**(1/len(coords))),components=len(names),
                 reference_moment_norm=math.sqrt(norm_sq),absolute_bias_RMS=math.sqrt(float(bias@bias)/len(names)),
                 deterministic_relative_RMS=math.sqrt(bias_sq),relative_variance_per_shot=variance,
                 implementation_TV_allowance=ETA,relative_bias_bound=math.sqrt(bias_bound_sq),relative_variance_bound=variance_bound,
                 target_expected_relative_RMS=TARGET,accepted_shots=shots,
                 ideal_expected_relative_RMS=math.sqrt(bias_sq+variance/shots) if shots else None,
                 model_based_expected_relative_RMS_bound=math.sqrt(bias_bound_sq+variance_bound/shots) if shots else None,
                 one_percent_target_feasible=shots is not None,
                 displayed_in_panel_a=ic!='boundary_clipped')
        records.append(rec)
        selected.append(dict(id=ident,model=label,input=ic,components=len(names),cells_per_coordinate=rec['cells_per_coordinate'],
                             moment_relative_RMS_error=rec['deterministic_relative_RMS']))
        sources.append(dict(id=ident,path=str(source.relative_to(ROOT)),sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                            probability_norm_defect=float(raw.sum()-1),reference=target.tolist(),outcome_mean=mean.tolist(),
                            observable_ranges=widths.tolist(),observable_second_moment_ranges=square_widths.tolist()))
        if model=='coupled_cubic_2d':
            inventory=json.loads((source.with_suffix('.json')).read_text())
            workloads.append(dict(id=ident,ic=ic,dimension=2,cells_per_coordinate=32,steps=64,
                logical_qubits=inventory['logical_qubits'],accepted_shots=shots,
                deterministic_relative_RMS=rec['deterministic_relative_RMS'],
                relative_variance_per_shot=variance,implementation_allowance=ETA,
                model_based_expected_relative_RMS_bound=rec['model_based_expected_relative_RMS_bound'],
                target_expected_relative_RMS=TARGET,reference='characteristics of the actual MPS-prepared piecewise-uniform input',
                RMS_scope='Euclidean moment error divided by Euclidean reference moment norm',
                RZ=inventory['complete_RZ'],native_T=inventory['complete_native_T'],CX=inventory['complete_CX']))
        if ic=='boundary_clipped':continue
        assert shots is not None, ident
        seed=(int.from_bytes(hashlib.sha256(ident.encode()).digest()[:8],'little')^20261111)%(2**63)
        rng=np.random.default_rng(seed)
        counts=rng.multinomial(shots,prob,size=REPETITIONS)
        estimates=counts@values/shots
        squared=np.sum((estimates-target)**2,axis=1)/norm_sq
        rms=np.sqrt(squared)
        lower,median,upper=np.quantile(rms,[.025,.5,.975],method='linear')
        expected=bias_sq+variance/shots
        se=float(np.std(squared,ddof=1)/math.sqrt(REPETITIONS))
        zscore=abs(float(squared.mean())-expected)/se if se else 0.
        assert zscore<8,(ident,zscore)
        sample_rows.append(dict(id=ident,model=label,input=ic,cells_per_coordinate=rec['cells_per_coordinate'],components=len(names),
            shots=shots,repetitions=REPETITIONS,seed=seed,deterministic_relative_RMS=math.sqrt(bias_sq),
            sampled_relative_RMS_median=float(median),sampled_relative_RMS_lower=float(lower),sampled_relative_RMS_upper=float(upper),
            analytic_expected_relative_RMS_squared=expected,empirical_mean_relative_RMS_squared=float(squared.mean()),
            model_based_expected_relative_RMS_bound=rec['model_based_expected_relative_RMS_bound'],MSE_check_z=zscore))
        samples[ident+'__moments']=estimates;samples[ident+'__relative_RMS']=rms
        print(ident,'N',rec['cells_per_coordinate'],'bias %',100*rec['deterministic_relative_RMS'],'shots',shots,
              'expected bound %',100*rec['model_based_expected_relative_RMS_bound'],flush=True)

write_csv('relative_accuracy_budget.csv',records)
write_csv('selected_moment_errors.csv',selected)
write_csv('sampling_errorbars.csv',sample_rows)
write_csv('workloads.csv',workloads)
np.savez_compressed(OUT/'sampling_replicates.npz',**samples)
report=dict(status='PASS',target_expected_relative_observable_RMS=TARGET,implementation_TV_allowance=ETA,
            definition='sqrt(E[||estimated observable vector - reference||_2^2])/||reference||_2',
            observable_scope='coordinate moments for polynomial models; reconstructed velocity field for wake models',
            reference='matched continuum evolution from the actual gate-prepared input',
            formula='expected squared relative RMS = relative bias squared + relative variance coefficient / S',
            shot_selection='smallest integer satisfying the component-range TV bound on expected relative RMS',
            repetitions=REPETITIONS,percentiles=[.025,.975],
            interval_scope='distribution of nominal finite-shot error estimates, not a confidence interval for their median',
            main_cases=12,all_main_cases_meet_target=True,
            scalar_cells=64,coupled_cells=32,coupled_steps=64,
            baseline_coupled_cells=16,baseline_source_preserved=True,
            diagnostic_failures=[r['id'] for r in records if not r['one_percent_target_feasible']],sources=sources)
(OUT/'accuracy_protocol.json').write_text(json.dumps(report,indent=2))
