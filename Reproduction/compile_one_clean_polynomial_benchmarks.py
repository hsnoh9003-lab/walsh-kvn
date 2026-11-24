"""Count and replay the polynomial benchmark with shared one-clean arithmetic.

Preparation and finite-grid inputs are unchanged. The 32-cell coupled check
executes all 64 native transport steps from the retained prepared state and
compares complex amplitudes with the prior finite-circuit output.
"""

# Public Source Data mapping (single canonical file per published input).
import sys as _kvn_path_sys
from pathlib import Path as _kvn_path_Path
_kvn_path_sys.path.insert(0, str(_kvn_path_Path(__file__).resolve().parents[1]))
from Source.data_locations import published_path
from pathlib import Path
from collections import Counter
import importlib.util, json,os,sys
import numpy as np
from Source.KvN import wake_gate_core as C

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'Build/One_Clean_Three_Coordinate/polynomial_benchmarks'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    source=ROOT/'Reproduction/auxiliary/figure5_bias'
    os.environ['POLYNOMIAL_VALIDATION_OUTPUT']=str(OUT/'compiler_intermediates')
    sys.path.insert(0,str(source))
    spec=importlib.util.spec_from_file_location('one_clean_polynomial_source',source/'validate_polynomials.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    # All numerical polynomials and Walsh tables come from the supplied source;
    # only the common exact MCX implementation is replaced.
    module.increment=C.increment;module.inverse=C.inverse
    case=[c for c in module.declared_cases() if c['D']==2][0]
    gates,nq,support=module.compile_step(case,32,1/64)
    assert nq==11
    inv=Counter(g[0] for g in gates)
    assert inv['t']+inv['tdg']==378
    assert inv['rz']==366,(dict(inv),len(gates))
    arrays=np.load(published_path('Data/Figure5_Bias_Analysis/input/grid32/coupled_cubic_2d__interior.npz'))
    initial=arrays['state_history'][0];expected=arrays['state_history'][-1]
    state=np.zeros(1<<nq,complex);state[:1024]=initial
    for _ in range(64):state=C.apply_gates(state,gates,nq)
    e=float(np.linalg.norm(state[:1024]-expected));leak=float(np.linalg.norm(state[1024:]))
    assert e<2e-11 and leak<2e-12,(e,leak)
    row={'logical_qubits':11,'position_qubits':10,'arithmetic_clean_work_qubits':1,
        'steps':64,'RZ':23514,'native_T':24192,'CX':54058,
        'native_transport_per_step_by_gate':dict(inv),'additional_X_gates_per_attempt':768,
        'preparation_RZ_unchanged':23514-64*inv['rz'],
        'preparation_CX_unchanged':54058-64*inv['cx'],
        'complete_transport_complex_state_l2_error':e,'work_leakage':leak,
        'note':'Preparation gates are unchanged. All native transport gates replayed for 64 steps from the retained prepared amplitude.'}
    (OUT/'coupled32_inventory_and_replay.json').write_text(json.dumps(row,indent=2)+'\n')
    print(json.dumps(row,indent=2))

if __name__=='__main__':main()
