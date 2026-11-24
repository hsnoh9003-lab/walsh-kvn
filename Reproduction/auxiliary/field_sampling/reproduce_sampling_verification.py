"""Repeat field-MSE checks from the frozen current joint circuit distributions."""
from pathlib import Path
import csv,json,hashlib,math
import numpy as np
import sys
PACKAGE=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(PACKAGE))
from Source.data_locations import source_data_root
ROOT=PACKAGE/'Build/Field_Sampling_Replay'
INPUT=source_data_root()/'Supplementary_Data/Finite_Shot_Replay/sampling_inputs'
SEED=20261107;REPEATS=3000

def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    h=np.load(INPUT/'direct_quantum_history.npz');j=np.load(INPUT/'joint_readout_inputs.npz')
    cfg=json.loads((INPUT/'configuration.json').read_text())
    reference=json.loads((INPUT/'observable_diagnostics_reference.json').read_text())['history']
    omega=float(cfg['parameters']['omega']);times=j['physical_times'];Mfinal=j['M'];G=j['G'];norm=float(j['norm']);values=h['joint_values']
    rng=np.random.default_rng(SEED);rows=[];replicates={};norm_drift=[]
    for step in [0,11,22,33]:
        p=h['joint_probabilities'][step].copy();norm_drift.append(abs(float(p.sum())-1));p/=p.sum()
        angle=omega*(times[step]-times[-1]);c,s=np.cos(angle),np.sin(angle)
        rot=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
        fieldmap=rot@Mfinal;mean=p@values;cov=(values.T*p)@values-np.outer(mean,mean)
        H=fieldmap.T@G@fieldmap/norm**2;variance=float(np.trace(H@cov))
        assert abs(100*np.sqrt(variance/1000)-reference[step]['sampling_RMS_1000_pct'])<2e-8
        for shots in [1000,8000]:
            counts=rng.multinomial(shots,p,size=REPEATS)
            moments=counts@values/shots;delta=moments-mean
            mse=np.einsum('bi,ij,bj->b',delta,H,delta)
            predicted=variance/shots;empirical=float(mse.mean());se=float(mse.std(ddof=1)/np.sqrt(REPEATS));score=(empirical-predicted)/se
            assert abs(score)<6
            rows.append(dict(step=step,shots=shots,repeats=REPEATS,seed=SEED,predicted_RMS_percent=100*np.sqrt(predicted),sampled_RMS_percent=100*np.sqrt(empirical),MSE_z_score=score,reference='exact finite-circuit mean',normalization='one fixed final continuum-field norm',state='current MPS-prepared Walsh circuit'))
            replicates[f'step{step}_shots{shots}_moments']=moments
            replicates[f'step{step}_shots{shots}_MSE']=mse
    with (ROOT/'Table_S13_sampling_verification.csv').open('w',newline='')as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    np.savez_compressed(ROOT/'sampling_MSE_replicates.npz',**replicates)
    qa=dict(status='PASS',seed=SEED,repeats=REPEATS,steps=[0,11,22,33],shot_counts=[1000,8000],rows=len(rows),maximum_abs_MSE_z_score=max(abs(x['MSE_z_score'])for x in rows),expected_variances_match_current_saved_diagnostics=True,max_probability_norm_drift=max(norm_drift),new_dynamics_or_resource_estimates=False,numpy_version=np.__version__,input_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in INPUT.iterdir()})
    (ROOT/'SAMPLING_VERIFICATION_QA.json').write_text(json.dumps(qa,indent=2));print(json.dumps(qa,indent=2))

if __name__=='__main__':main()
