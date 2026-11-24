"""Core regression checks: python -m Tests.verify_core_guards."""

import contextlib
import importlib
import io
import unittest
from unittest.mock import patch

import numpy as np

from Source.data_locations import published_path
from Source.POD_SINDy import identification
from Source.ROM.ensemble import reference
from Source.ROM.model import load


class CoreRegressionTests(unittest.TestCase):
    def test_model_typo_is_rejected_before_loading_files(self):
        with patch('Source.ROM.model.np.load') as read:
            with self.assertRaisesRegex(ValueError, 'Unknown model'):
                load('canddiate')
            read.assert_not_called()
        for name in ('candidate', 'deployed'):
            with self.subTest(name=name):
                self.assertTrue(np.all(np.isfinite(load(name).coefficients)))

    def test_derivatives_match_polynomials(self):
        t = np.linspace(-1., 1., 41)
        for module in (identification,):
            for kind, degree in [('fd2', 2), ('fd4', 4), ('fd6', 6), ('fd8', 8), ('sg9', 3)]:
                with self.subTest(module=module.__name__, kind=kind):
                    x = np.column_stack([t**n for n in range(1, degree+1)])
                    _, times, derivative = module.differentiate(x, t, kind)
                    exact = np.column_stack([n*times**(n-1) for n in range(1, degree+1)])
                    np.testing.assert_allclose(derivative, exact, rtol=0, atol=2e-12)

    def test_bad_time_grids_are_rejected(self):
        irregular = np.array([0., .1, .3, .6, 1., 1.5, 2.1, 2.8, 3.6])
        for module in (identification,):
            for kind in ('fd2', 'fd4', 'fd6', 'fd8', 'sg9'):
                with self.subTest(module=module.__name__, kind=kind):
                    with self.assertRaisesRegex(ValueError, 'uniformly spaced'):
                        module.differentiate(irregular[:, None]**2, irregular, kind)
            for times in ([0., 1., 1., 2., 3.], [4., 3., 2., 1., 0.]):
                with self.assertRaisesRegex(ValueError, 'strictly increasing'):
                    module.differentiate(np.zeros((5, 3)), times)
            with self.assertRaisesRegex(ValueError, 'finite'):
                module.differentiate(np.zeros((5, 3)), [0., 1., np.nan, 3., 4.])
            with self.assertRaisesRegex(ValueError, 'at least 5'):
                module.differentiate(np.zeros((4, 3)), np.arange(4.))
            with self.assertRaisesRegex(ValueError, 'first axis'):
                module.differentiate(np.zeros((6, 3)), np.arange(5.))

    def test_annular_inner_boundary_and_zero_radius_control(self):
        class Damping:
            def drift(self, state):
                return np.array([-state[0], -state[1], np.zeros_like(state[2])])

        config = dict(
            centre_cartesian=[.7, 0., .5], sigma_cartesian=[.15, .15, .1],
            rrange=[.5, 1.], zrange=[0., 1.], time_window=[0., 2.],
        )
        for inner, expected_exit in ((.5, 1.), (0., 0.)):
            with self.subTest(inner_radius=inner):
                config['rrange'] = [inner, 1.]
                with contextlib.redirect_stdout(io.StringIO()):
                    means, audit = reference(Damping(), config, (8, 16, 8))
                # Every radius is multiplied by exp(-2), so all annular paths exit.
                self.assertAlmostEqual(audit['sampled_path_outside_domain_weight'], expected_exit)
                np.testing.assert_allclose(
                    means[-1, :2], means[0, :2]*np.exp(-2), rtol=1e-8, atol=1e-10,
                )

    def test_historical_basis_construction(self):
        helpers = importlib.import_module('Source.POD_SINDy.legacy.original_rom_functions')
        with np.load(published_path('Data/CFD_ROM/Re100/data/source_snapshots.npz')) as archive:
            data = {key: archive[key] for key in archive.files}
        basis, coordinates, _ = helpers.make_basis(data)
        self.assertEqual(coordinates.shape, (len(data['times']), 3))
        self.assertTrue(np.all(np.isfinite(coordinates)))
        self.assertAlmostEqual(np.linalg.norm(basis['Phi'][2]), 1.)
        # Least-squares projection leaves a residual orthogonal to all basis rows.
        fields = data['fields'].astype(float).reshape(len(data['times']), -1)
        residual = fields-basis['reference']-coordinates@basis['Phi']
        np.testing.assert_allclose(residual@basis['Phi'].T, 0., rtol=0, atol=1e-10)


if __name__ == '__main__':
    unittest.main()
