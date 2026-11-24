"""Change the field RMS target with the published physical design held fixed."""

# Public Source Data mapping (single canonical file per published input).
import sys as _kvn_path_sys
from pathlib import Path as _kvn_path_Path
_kvn_path_sys.path.insert(0, str(_kvn_path_Path(__file__).resolve().parents[2]))
from Source.data_locations import published_path, source_label
from pathlib import Path
import csv,json,math
ROOT=Path(__file__).resolve().parents[2]

def main():
    source=published_path('Data/Details/implementation_allowance_sensitivity.csv')
    rows=list(csv.DictReader(source.open()));output=[]
    for design in ['fastest_under_cap','minimum_footprint']:
        original=next(r for r in rows if r['selection']==design and float(r['implementation_allowance'])==1e-4)
        bias=float(original['bias_eff']);variance=float(original['variance_eff'])
        for target in [.01,.05]:
            shots=math.ceil(variance/(target**2-bias**2))
            output.append({'design':design,'relative_RMS_target':target,'accepted_shots':shots,
                'physical_qubits':int(original['physical_qubits']),'seconds_per_attempt':float(original['seconds_per_attempt']),
                'complete_task_minutes':shots*float(original['seconds_per_attempt'])/60,
                'implementation_allowance':1e-4,'protected_bias':bias,'protected_variance':variance,
                'RMS_bound':math.sqrt(bias**2+variance/shots),'new_architecture_optimization':False})
    out=ROOT/'Build/Accuracy_Target_Sensitivity';out.mkdir(parents=True,exist_ok=True)
    with (out/'fixed_design_targets.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,list(output[0]));w.writeheader();w.writerows(output)
    (out/'protocol.json').write_text(json.dumps({'source':source_label(source),
        'calculation':'ceil(variance/(target squared minus bias squared)); per-attempt design and implementation error unchanged',
        'scope':'Terminal velocity-field task, not force-coefficient target',
        'reoptimized':False},indent=2)+'\n')
    print(json.dumps(output,indent=2))
if __name__=='__main__':main()
