"""Verify canonical modules, retained algorithms, and historical import compatibility."""

import importlib
import subprocess
import sys
import unittest

import numpy as np

from Source.data_locations import published_path
from Source.POD_SINDy import identification, pod
from Source.POD_SINDy.fit import fit


class ModuleLayoutTests(unittest.TestCase):
    def test_main_imports_do_not_load_historical_modules(self):
        script = '''
import sys
from Source.POD_SINDy import replay_frozen_basis, fit, identification, pod
from Source.ROM import model
assert not any(name.startswith((
    'Source.POD_SINDy.recovered_original', 'Source.POD_SINDy.legacy', 'Source.ROM.legacy'
)) for name in sys.modules)
assert 'original_rom_functions' not in sys.modules
assert 'shift_pipeline' not in sys.modules
'''
        subprocess.run([sys.executable, '-B', '-c', script], check=True)

    def test_old_paths_resolve_to_single_implementations(self):
        old_ident = importlib.import_module('Source.POD_SINDy.recovered_original.identification')
        old_rom = importlib.import_module('Source.POD_SINDy.recovered_original.original_rom_functions')
        old_shift = importlib.import_module('Source.POD_SINDy.recovered_original.shift_pipeline')
        generic = importlib.import_module('Source.POD_SINDy.legacy.identification')
        self.assertIs(old_ident.differentiate, identification.differentiate)
        self.assertIs(old_ident.simulate_diagnostic, identification.simulate_diagnostic)
        self.assertIs(old_ident.stlsq, generic.stlsq)
        self.assertIs(identification.learn, generic.learn)
        self.assertIs(old_rom.project, pod.project)
        self.assertIs(old_rom.closure, pod.closure)
        self.assertIs(old_shift.fit_oscillatory_pair, pod.fit_oscillatory_pair)
        model = importlib.import_module('Source.ROM.model')
        legacy_model = importlib.import_module('Source.ROM.legacy.model')
        force = importlib.import_module('Source.ROM.force_observables')
        legacy_force = importlib.import_module('Source.ROM.legacy.force_observables')
        self.assertIs(model.raw_drift, legacy_model.raw_drift)
        self.assertIs(force.force_curves, legacy_force.force_curves)

    def test_legacy_regression_still_recovers_sparse_coefficients(self):
        generic = importlib.import_module('Source.POD_SINDy.legacy.identification')
        rng = np.random.default_rng(481)
        powers = identification.powers(3, 2)
        library = identification.features(rng.normal(size=(600, 3)), powers)
        exact = np.zeros((len(powers), 3))
        exact[1, 0], exact[6, 0], exact[2, 1], exact[8, 1] = -.5, .8, .3, -.2
        learned = generic.stlsq(library, library@exact, .01)
        np.testing.assert_allclose(learned, exact, rtol=0, atol=1e-12)

    def test_retained_weighted_fit_matches_current_unweighted_case(self):
        legacy = importlib.import_module('Source.POD_SINDy.legacy.original_rom_functions')
        with np.load(published_path('Data/Model/baseline_inputs.npz')) as data:
            coordinates = data['training_coordinates']
            times = data['training_times']
        current = fit(coordinates, times, 2, .03)
        historical = legacy.fit_weighted(coordinates, times, 1e-4, .03, True, 0)
        np.testing.assert_allclose(historical['C'], current['physical_C'], rtol=0, atol=5e-14)


if __name__ == '__main__':
    unittest.main()
