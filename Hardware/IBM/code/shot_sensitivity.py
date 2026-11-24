"""Conditional, terminal-probability shot estimates with equal transport/reference shots.

This diagnostic is distinct from the manuscript's 815-shot velocity-field task.
It treats archived measured histograms as stationary generating distributions.
The ratio-based reference estimator uses a first-order variance approximation;
the 50,000-resample calculation checks that same empirical model, not new hardware.
"""
from pathlib import Path
import csv,json,math
import numpy as np
import analyze_ibm as exact
import analyze_mps_ibm as mps

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data'


def main():
    elapsed=49.8291-40;mu=.0924819;cutoff=.875
    threshold=cutoff/math.sqrt(math.exp(2*mu*elapsed)*(1-cutoff**2)+cutoff**2)
    rng=np.random.default_rng(20261117);results=[]
    cases=[('exact',4,'ibm_sampler_dT4_counts.json'),('MPS',4,'ibm_mps_dT4_n4n5_counts.json'),
           ('exact',6,'ibm_sampler_dT6_counts.json'),('MPS',6,'ibm_mps_dT6_n4_counts.json')]
    for mode,steps,filename in cases:
        p0=exact.P0 if mode=='exact' else mps.mps_state(np.sqrt(mps.target_masses(4)),4)**2
        counts=json.loads((DATA/filename).read_text());prefix='n4_' if mode=='MPS' else ''
        def measured(name):
            c=np.zeros(16)
            for key,value in counts[name].items():c[int(key,2)]=value
            return c/c.sum(),int(c.sum())
        pk,shots=measured(prefix+f'kvn_k{steps}');pr,refshots=measured(prefix+f'ref_k{steps}')
        h=(np.arange(16)>=14).astype(float);uniform=np.ones(16)/16
        a=(p0-uniform)/np.sum((p0-uniform)**2)
        contrast=float(a@(pr-uniform));difference=float(h@(pk-pr))
        corrected=float(h@p0+difference/contrast)
        cell=int(threshold*16)
        reference=float(p0[cell+1:].sum()+p0[cell]*((cell+1)/16-threshold)*16)
        covariance_k=np.diag(pk)-np.outer(pk,pk);covariance_r=np.diag(pr)-np.outer(pr,pr)
        gradient_k=h/contrast
        gradient_r=-h/contrast-difference*a/contrast**2
        variance=float((gradient_k@covariance_k@gradient_k+gradient_r@covariance_r@gradient_r)/reference**2)
        signed_bias=(corrected-reference)/reference;bias=abs(signed_bias)
        finite_bias=float((h@covariance_r@a/contrast**2+difference*(a@covariance_r@a)/contrast**3)/reference)
        row={'input':mode,'steps':steps,'existing_shots_each':shots,'P_continuum':reference,'P_corrected':corrected,
             'signed_relative_bias':signed_bias,'bias_pct':100*bias,'v_relative':variance,'finite_sample_bias_coefficient':finite_bias}
        for epsilon in [.01,.05]:
            label=str(int(epsilon*100))+'pct'
            if bias>=epsilon:row[label]=None;continue
            estimate=math.ceil(variance/(epsilon**2-bias**2))
            row[label]={'shots_each':estimate,'shots_total_pair':2*estimate}
            if epsilon==.05:
                sampled_k=rng.multinomial(estimate,pk,size=50000)/estimate
                sampled_r=rng.multinomial(estimate,pr,size=50000)/estimate
                sampled_contrasts=(sampled_r-uniform)@a
                sampled=h@p0+((sampled_k-sampled_r)@h)/sampled_contrasts
                relative=(sampled-reference)/reference
                row[label].update(resampling_relative_RMS=float(np.sqrt(np.mean(relative**2))),
                    resampling_signed_bias=float(relative.mean()),resampling_relative_SD=float(relative.std()),
                    min_contrast=float(sampled_contrasts.min()))
        results.append(row)
    report={'threshold':threshold,'results':results,'resamples':50000,'seed':20261117,
        'observable':'Terminal P(r >= 7/8), matched piecewise-uniform continuum input, 16 cells.',
        'bias_scope':'Plug-in difference of corrected empirical means and the continuum reference; not an independently determined hardware bias.',
        'variance_scope':'First-order delta method including reference contrast variance and its covariance with the reference observable.',
        'shot_scope':'Equal independent shots per terminal transport and reference circuit; total pair cost is twice shots_each.',
        'resampling_scope':'Internal consistency within fitted empirical histograms; does not establish a hardware accuracy guarantee or full-time-series field accuracy.'}
    (DATA/'terminal_shot_sensitivity.json').write_text(json.dumps(report,indent=2)+'\n')
    with (DATA/'terminal_shot_sensitivity.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['input','steps','P_continuum','P_corrected','bias_pct','v_relative','epsilon',
            'estimated_shots_each','estimated_shots_pair','resampling_relative_RMS'])
        writer.writeheader()
        for row in results:
            for epsilon,label in [(.01,'1pct'),(.05,'5pct')]:
                value={k:row[k] for k in ['input','steps','P_continuum','P_corrected','bias_pct','v_relative']};value['epsilon']=epsilon
                target=row[label]
                value.update(estimated_shots_each=target['shots_each'] if target else '',
                    estimated_shots_pair=target['shots_total_pair'] if target else '',
                    resampling_relative_RMS=target.get('resampling_relative_RMS','') if target else '')
                writer.writerow(value)
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
