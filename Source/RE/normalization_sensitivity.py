"""Fixed-architecture sensitivity to BE normalization; not a compiled circuit.

Recalculate a Jacobi–Anger truncation degree at the recorded spectral norm,
matching the absolute Chebyshev-tail bound of the archived outer degree.
Only the effective call-linear workload is rescaled. No alternative oracle,
gate program, success probability, field bias or architecture is claimed.
"""

# Public Source Data mapping (single canonical file per published input).
import sys as _kvn_path_sys
from pathlib import Path as _kvn_path_Path
_kvn_path_sys.path.insert(0, str(_kvn_path_Path(__file__).resolve().parents[2]))
from Source.data_locations import published_path
from pathlib import Path
import csv,json,math
import numpy as np
from scipy.special import jv,gammaln

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'Build/Normalization_Sensitivity'

def tail_array(argument):
    limit=max(100,int(math.ceil(3*argument+1000)))
    k=np.arange(limit+1)
    terms=2*np.abs(jv(k,argument))
    tails=np.cumsum(terms[::-1])[::-1]
    # |J_k(a)| <= (a/2)^k / k! * exp(a^2/[4(k+1)]).
    # Beyond limit the resulting bound decreases geometrically.
    first=limit+1
    log_first=math.log(2)+first*math.log(argument/2)-float(gammaln(first+1))+argument**2/(4*(first+1))
    ratio=argument/(2*(first+1))
    log_remainder=log_first-math.log1p(-ratio)
    return tails,log_remainder

def main():
    norm_rows=list(csv.DictReader((published_path('Data/Details/generator_norm.csv')).open()))
    norms={r['quantity']:float(r['value']) for r in norm_rows}
    params=json.loads((published_path('Data/CFD_ROM/Re100/data/model_configuration.json')).read_text())['parameters']
    normalized_duration=params['mu']*9.82906260206943
    alpha=norms['block_encoding_alpha'];spectral_norm=norms['joint_generator_spectral_norm']
    degree_reference=16197;calls_reference=3*degree_reference
    old_tail,old_remainder=tail_array(alpha*normalized_duration)
    epsilon_tail=float(old_tail[degree_reference+1])
    new_tail,new_remainder=tail_array(spectral_norm*normalized_duration)
    degree_control=int(np.flatnonzero(new_tail[1:]<=epsilon_tail)[0]);calls_control=3*degree_control
    assert epsilon_tail<1e-4 and new_tail[degree_control+1]<=epsilon_tail
    assert old_remainder < -700 and new_remainder < -700
    records=list(csv.DictReader((published_path('Data/RE/v38_reference_selected.csv')).open()))
    original=next(r for r in records if r['model']=='generator' and r['selection']=='fastest_uncapped' and float(r['p_phys'])==.0005)
    ratio=calls_control/calls_reference
    result={'original_alpha':alpha,'generator_spectral_norm':spectral_norm,'normalization_ratio':alpha/spectral_norm,
        'normalized_duration':normalized_duration,'original_outer_degree':degree_reference,'original_block_calls':calls_reference,
        'original_absolute_Chebyshev_tail_bound':epsilon_tail,'sensitivity_outer_degree':degree_control,'sensitivity_block_calls':calls_control,
        'sensitivity_absolute_Chebyshev_tail_bound':float(new_tail[degree_control+1]),
        'old_log_tail_remainder_bound':old_remainder,'new_log_tail_remainder_bound':new_remainder,
        'call_ratio':ratio,'effective_T_count_at_fixed_precision':float(original['T_count'])*ratio,
        'task_hours_at_fixed_architecture':float(original['complete_task_seconds'])/3600*ratio,
        'original_task_hours':float(original['complete_task_seconds'])/3600,
        'fixed_physical_qubits':int(original['physical_qubits']),'fixed_synthesis_T_per_rotation':int(original['N_T']),
        'fixed_accepted_shots':int(original['successful_shots']),
        'assumptions':'Keep encoded width, effective original gate inventory per block call, selected architecture, synthesis precision, shot count and success probability fixed; rescale the call-linear workload only.',
        'scope':'Normalization sensitivity, not a compiled alternative or a lower bound. No new generator encoding, gate list, field-error or acceptance calculation is supplied. Fixed preparation/readout are included in the effective per-call average, rather than recompiled separately.'}
    OUT.mkdir(exist_ok=True)
    (OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    scalar={k:v for k,v in result.items() if isinstance(v,(int,float))}
    with (OUT/'result.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['quantity','value']);w.writerows(scalar.items())
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
