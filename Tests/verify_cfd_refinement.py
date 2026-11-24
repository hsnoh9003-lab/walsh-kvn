"""Analytic checks for the force-statistics and reference-adequacy decisions."""
import unittest
import numpy as np
from Source.CFD.refinement_metrics import cycle_metrics, sweep_verdict


class RefinementTests(unittest.TestCase):
    def signal(self, dt=.02, phase=.0, growing=False):
        t = np.arange(0, 100 + dt/2, dt)
        amplitude = .3*(1 + .01*t) if growing else .3
        return dict(time=t, C_L=amplitude*np.sin(2*np.pi*.17*t + phase),
                    C_D=1.4 + .01*np.cos(4*np.pi*.17*t + 2*phase))

    def test_periodic_statistics_are_phase_independent(self):
        for dt in (.02, .013, .005):
            for phase in (0., .8):
                result = cycle_metrics(self.signal(dt, phase))
                self.assertTrue(result['stationary'])
                self.assertAlmostEqual(result['mean_CD'], 1.4, places=8)
                self.assertAlmostEqual(result['rms_CL'], .3/np.sqrt(2), places=8)
                self.assertAlmostEqual(result['St'], .17, places=8)

    def test_growing_signal_not_stationary(self):
        self.assertFalse(cycle_metrics(self.signal(growing=True))['stationary'])

    def test_constant_signal_not_periodic(self):
        signal = self.signal()
        signal['C_L'][:] = 0
        self.assertFalse(cycle_metrics(signal)['stationary'])

    def test_invalid_time_order(self):
        signal = self.signal()
        signal['time'][10] = signal['time'][9]
        with self.assertRaises(ValueError):
            cycle_metrics(signal)

    def test_refinement_pass_does_not_certify_reference(self):
        rows = [dict(metrics=dict(stationary=True, mean_CD=x, rms_CL=.2, St=.17))
                for x in (1., 1.2, 1.201)]
        verdict = sweep_verdict(rows)
        self.assertEqual(verdict['status'], 'within_tolerance')
        self.assertFalse(verdict['reference_within_tolerance'])
        rows[1]['metrics']['stationary'] = False
        self.assertEqual(sweep_verdict(rows)['status'], 'unsettled_periodic_statistics')


if __name__ == '__main__':
    unittest.main()
