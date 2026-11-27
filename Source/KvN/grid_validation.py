"""Test one-axis half grids using the adopted model and production Gaussians.

Read the current protocol from KvN/inputs. Each run retains its current
spatial stencil and time-step count while rebuilding the input on the
modified grid. Results include both the bias and the sampling/circuit cost.
"""

# Direct-file execution must locate Codes/ independently of the working directory.
if __package__ in (None, ""):
    import sys as _path_sys
    from pathlib import Path as _Path
    _codes = next(p for p in _Path(__file__).resolve().parents
                  if (p / "Source/data_locations.py").is_file())
    _path_sys.path.insert(0, str(_codes))


from Source.data_locations import generated_outputs_root
from pathlib import Path
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import json,argparse,numpy as np
from scipy.integrate import solve_ivp
from . import ensemble as E
from Source.data_locations import shared_data
PACKAGE=Path(__file__).resolve().parents[2]
ROOT=shared_data('MPS_KvN_inputs','KvN')

def main(run,axis,grid_override=None):
    config=json.loads((ROOT/'inputs'/run/'protocol.json').read_text())
    # Change the selected grid axis while retaining the archived physical domain, transport window, and model protocol.
    if grid_override is None:config['grid_r_theta_z'][axis]//=2
    else:config['grid_r_theta_z']=list(grid_override)
    key=f'{run}_half_{axis}' if grid_override is None else run+'_n'+'_'.join(map(str,grid_override))
    out=generated_outputs_root() / 'KvN/grid_checks'/key;out.mkdir(parents=True,exist_ok=True)
    (out/'protocol.json').write_text(json.dumps(config,indent=2)+'\n')
    # Rebuild the physical Gaussian on the modified grid and compile a new MPS, because the original grid amplitudes cannot be reused unchanged.
    target,norm=E.make_input(config)
    for chi in [2,4,8,16,32,64]:
        amp,blocks,base,prep=E.compress(target.ravel(),chi)
        if prep['fidelity']>=.9999:break
    assert prep['fidelity']>=.9999
    prep.update(block_widths=[x['m'] for x in blocks],passed=True)
    (out/'selected.json').write_text(json.dumps(prep,indent=2)+'\n')
    np.savez_compressed(out/'mps_blocks.npz',base=base,**{f'U_{i}':b['basis'] for i,b in enumerate(blocks)})
    from . import compile_mps as C
    C.main(run,directory=out)
    inventory=json.loads((out/'compiled_inventory.json').read_text())
    nr,nt,nz=config['grid_r_theta_z'];model=E.load('candidate')
    args=argparse.Namespace(output=out,model=model,grid=config['grid_r_theta_z'],steps=config['steps'],rrange=config['rrange'],zrange=config['zrange'],
        stencil=config['stencil'],quadrature=12,restart=None,no_save_preparation=True,start=config['time_window'][0],
        initial_probability_integral=norm,prepared_amplitude=amp.reshape(nz,nr,nt),reference_csv=shared_data('MPS_KvN_inputs','MPS','domain_selection',run,'reference.csv'))
    # Replay transport from the independently prepared coarse-grid state against the archived reference field.
    E.transport.replay(args)
    raw=json.loads((out/'terminal_readout.json').read_text());budget=E.field_budget(raw,1e-4,.01)
    (out/'field_budget.json').write_text(json.dumps(budget,indent=2)+'\n')
    counts=E.count_native(model,config,inventory);counts['axis_cells']=config['grid_r_theta_z'];counts['input_protocol']='Independent Cartesian Gaussian centred on its ODE characteristic'
    (out/'inventory.json').write_text(json.dumps(counts,indent=2)+'\n')
    synthesis=int(np.ceil(3*np.log2(counts['arbitrary_Rz']/.5e-4)))
    shots=budget['minimum_accepted_shots_under_rms_bound']
    result=dict(run=run,half_axis=axis,grid=config['grid_r_theta_z'],bias_percent=100*budget['noiseless_deterministic_relative_field_bias'],
        peak_bias_percent=json.loads((out/'summary.json').read_text())['peak_sampled_field_error_percent'],shots=shots,
        arbitrary_Rz=counts['arbitrary_Rz'],native_T=counts['native_T'],logical_qubits=counts['logical_qubits'],
        T_workload_proxy=None if shots is None else int(shots*(counts['native_T']+synthesis*counts['arbitrary_Rz'])))
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print('GRID_CHECK',json.dumps(result),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('run');ap.add_argument('axis',type=int);ap.add_argument('--grid',type=int,nargs=3);a=ap.parse_args();main(a.run,a.axis,a.grid)
