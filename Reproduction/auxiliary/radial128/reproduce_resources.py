"""Recompute the published 128-cell resource row using Source.RE only."""
from pathlib import Path
import sys,json,math
PACKAGE=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(PACKAGE))
from Source.data_locations import published_path, shared_data
ROOT=shared_data('Radial_refinement', 'Radial128')
from Source.RE.run import sweep
from Source.RE.model import ModelConfig
if __name__=='__main__':
    report=json.loads((ROOT/'grid128/flow_report.json').read_text())
    count=report['counts']
    work=dict(logical_qubits=10,is_wake=True,n_position=7,RZ=count['RZ'],direct_T=count['native_T'],
        Clifford=count['Clifford'],shots=report['budget']['shots'],success=1.,
        canonicalization_error_bound=0.,precision_min=45,precision_max=200,
        shot_basis='1% expected relative field RMS',
        workload_scope='same wake ensemble, refined radial grid')
    assert ModelConfig().W_seconds==1e-6
    expected=json.loads((ROOT/'comparison_summary.json').read_text())
    rows=sweep({'radial128':work},{5:published_path('Data/RE/prep_p5.csv')},PACKAGE/'Build/Radial128/resources',rates=[5])
    fastest=next(r for r in rows if r['selection']=='fastest_uncapped')
    minimum=next(r for r in rows if r['selection']=='minimum_qubits')
    assert fastest['physical_qubits']==561553 and fastest['T_count']==1193151
    assert math.isclose(fastest['complete_task_seconds'],18339.924021,rel_tol=1e-12)
    assert minimum['physical_qubits']==60482
    print(json.dumps({'fastest':{k:fastest[k] for k in ['physical_qubits','T_count','complete_task_seconds']},
        'minimum_qubits':minimum['physical_qubits']},indent=2))
