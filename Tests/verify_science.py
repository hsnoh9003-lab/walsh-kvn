"""Portable v38 scientific-source smoke and archived-inventory regression tests.

Run from the distribution root: python -m Tests.verify_science
Requires numpy, scipy, pandas and g++ for the matching kernel.
This test does NOT reproduce the full FEM/Navier–Stokes run or the large-grid
transport; those require external CFD arrays and MPS amplitudes.
"""

# Public Source Data mapping (single canonical file per published input).
import sys as _kvn_path_sys
from pathlib import Path as _kvn_path_Path
_kvn_path_sys.path.insert(0, str(_kvn_path_Path(__file__).resolve().parents[1]))
from Source.data_locations import published_path, shared_data
from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import numpy as np

from Source.CFD.snapshots import interpolate_snapshots
from Source.POD_SINDy.fit import fit
from Source.ROM.model import load
from Source.ROM.ensemble import make_input, reference
from Source.ROM.field_budget import field_budget
from Source.KvN.ensemble import count_native
from Source.KvN.mps import compress
from Source.KvN.transport import replay
from Source.KvN.radial_mps import execute, prepare_amplitude

ROOT = Path(__file__).resolve().parents[1]


def main():
    tests = {}
    params = np.load(published_path('Data/Model/baseline_inputs.npz'))
    candidate = np.load(published_path('Data/Model/candidate_K2.npz'))
    fitted = fit(params['training_coordinates'], params['training_times'], 2, .03)
    assert fitted['active_terms'] == 12
    np.testing.assert_allclose(fitted['physical_C'], candidate['physical_coefficients'], rtol=0, atol=5e-14)
    tests['POD_SINDy_refit_12'] = 'PASS'

    model = load('candidate')
    assert int(np.count_nonzero(model.physical_coefficients)) == 12
    assert model.coefficients.shape == (10, 3)
    tests['ROM_frozen_model'] = 'PASS'

    targets = np.linspace(.1, 1., 256); targets /= np.linalg.norm(targets)
    state, blocks, base, meta = compress(targets, 4)
    assert meta['bond_dimension'] == 4 and meta['fidelity'] >= 0.99
    np.testing.assert_allclose(np.linalg.norm(state), 1, rtol=0, atol=1e-13)
    radial_target = np.array([.1, .5, .2, .8]); radial_target /= np.linalg.norm(radial_target)
    _, program, report = prepare_amplitude(radial_target)
    actual = execute(2, program)
    assert report['fidelity'] > .999999999
    np.testing.assert_allclose(np.abs(actual), np.abs(radial_target), atol=1e-12, rtol=0)
    tests['KvN_MPS_and_radial_loader'] = 'PASS'

    for case in ('late40', 'long298'):
        folder = shared_data('MPS_KvN_inputs', 'KvN', 'inputs', case)
        conf = json.loads((folder / 'protocol.json').read_text())
        inventory = json.loads((folder / 'compiled_inventory.json').read_text())
        expected = json.loads((folder / 'expected_gate_inventory.json').read_text())
        counted = count_native(model, conf, inventory)
        for key in ('logical_qubits', 'position_qubits', 'arbitrary_Rz', 'native_T', 'Clifford'):
            assert counted[key] == expected[key], (case, key, counted[key], expected[key])
        readout = json.loads((folder / 'terminal_readout.json').read_text())
        budget = field_budget(readout, 1e-4, .01)
        expected_budget = json.loads((folder / 'expected_field_budget.json').read_text())
        for key in ('minimum_accepted_shots_under_rms_bound', 'noiseless_deterministic_relative_field_bias',
                    'noiseless_relative_field_variance_per_shot'):
            assert budget[key] == expected_budget[key], (case,key)
    tests['KvN_both_current_gates_and_shots'] = 'PASS'

    t = np.array([0., 1., 3., 5.]); fields = np.arange(24).reshape(4, 3, 2)
    out, audit = interpolate_snapshots(t, fields, [.5, 1.6, 4.2])
    np.testing.assert_array_equal(out[0], (fields[0]+fields[1])/2)
    assert len(audit) == 3
    tests['CFD_saved_snapshot_interpolation'] = 'PASS'

    with tempfile.TemporaryDirectory(prefix='stage2_science_test_') as tmp:
        tmp = Path(tmp)
        conf = json.loads((published_path('Data/KvN/inputs/late40/protocol.json')).read_text())
        conf['grid_r_theta_z'] = [8, 4, 8]
        conf['steps'] = 66
        conf['stencil'] = 2
        conf['centre_cartesian'] = [.45, .24, .5]
        conf['sigma_cartesian'] = [.12, .12, .12]
        amp, norm = make_input(conf, order=4)
        ref, audit = reference(model, conf, (4, 4, 4), tmp / 'reference.csv')
        cfg = SimpleNamespace(output=tmp/'result', model=model, grid=conf['grid_r_theta_z'], steps=66,
                              rrange=conf['rrange'], zrange=conf['zrange'], stencil=2, quadrature=4,
                              restart=None, no_save_preparation=True, start=40.,
                              initial_probability_integral=norm, prepared_amplitude=amp,
                              reference_csv=tmp/'reference.csv')
        replay(cfg)
        summary = json.loads((cfg.output / 'summary.json').read_text())
        assert summary['matching_layers'] == 25
        assert summary['max_norm_drift'] < 1e-11
        np.testing.assert_allclose(summary['terminal_means'],
            [-0.4558585561416604, 0.0692485020286184, 0.5956541044286389],
            rtol=0, atol=2e-12)
    tests['KvN_compiled_Cpp_matching_and_66_steps'] = 'PASS'
    print(json.dumps({'status':'PASS','tests':tests},indent=2))

if __name__ == '__main__':
    main()
