import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from uwbench.trajectory import evaluate


def circle(t):
    """Gravity-level loop with varying radius, so a time shift is not a rigid rotation of the path."""
    w = 2*np.pi/20
    r = 0.5 + 0.2*np.sin(0.37*t)
    p = np.column_stack([r*np.cos(w*t), r*np.sin(w*t), 0.1*np.sin(0.3*t)])
    R = Rotation.from_euler('z', w*t + np.pi/2).as_matrix()
    return R, p


class EvalTrajTest(unittest.TestCase):
    def setUp(self):
        self.ref_t = np.arange(0, 60, 0.005)
        self.ref_R, self.ref_p = circle(self.ref_t)
        self.est_t = np.arange(1, 59, 0.05)
        self.est_R, self.est_p = circle(self.est_t)

    def run_eval(self, est_t=None, est_R=None, est_p=None, **config):
        return evaluate(self.ref_t, self.ref_R, self.ref_p,
                        self.est_t if est_t is None else est_t,
                        self.est_R if est_R is None else est_R,
                        self.est_p if est_p is None else est_p, config=config)[0]

    def test_identical_is_zero_with_full_coverage(self):
        r = self.run_eval()
        self.assertLess(r['public']['ape_translation_m']['max'], 1e-9)
        self.assertEqual(r['coverage']['failure_count'], 0)
        self.assertAlmostEqual(r['initialization_delay_s'], 1.0)
        self.assertEqual(r['risk']['axes']['x']['exceedance']['exceed_s'], 0)

    def test_known_world_rotation_and_translation_are_aligned_away(self):
        G = Rotation.from_euler('z', 30, degrees=True).as_matrix()
        r = self.run_eval(est_R=G @ self.est_R, est_p=self.est_p @ G.T + [3, -2, 1])
        self.assertLess(r['public']['ape_translation_m']['max'], 1e-6)
        self.assertAlmostEqual(r['risk']['alignment_fit']['yaw_deg'], -30, places=6)
        self.assertLess(r['risk']['axes']['yaw']['abs_error_max'], 1e-6)

    def test_late_drift_is_seen_in_risk_mode_on_body_axes(self):
        drift = np.clip(self.est_t - 30, 0, None)[:, None]*np.array([[0.02, 0, 0]])  # 2 cm/s along world x after 30 s
        r = self.run_eval(est_p=self.est_p + drift)
        x, y = (r['risk']['axes'][a]['exceedance']['exceed_s'] for a in 'xy')
        self.assertGreater(x + y, 10)  # world-x drift projects onto both body axes on a loop
        self.assertLess(r['risk']['axes']['z']['abs_error_max'], 1e-9)
        self.assertLess(r['public']['ape_translation_m']['max'], np.abs(drift).max())  # whole-path fit hides part of it

    def test_time_offset_is_detected_and_correctable(self):
        shifted = self.est_t + 0.5
        biased = self.run_eval(est_t=shifted)
        fixed = self.run_eval(est_t=shifted, timestamp_offset_s=-0.5)
        self.assertGreater(biased['public']['ape_translation_m']['rmse'], 0.01)
        self.assertLess(fixed['public']['ape_translation_m']['rmse'], 1e-9)

    def test_missing_frames_are_lost_time_not_bridged(self):
        keep = (self.est_t < 20) | (self.est_t >= 22)
        r = self.run_eval(est_t=self.est_t[keep], est_R=self.est_R[keep], est_p=self.est_p[keep])
        gaps = [i for i in r['coverage']['lost_intervals'] if i['kind'] == 'gap']
        self.assertEqual(len(gaps), 1)
        self.assertAlmostEqual(gaps[0]['duration_s'], 2.05, places=6)

    def test_reference_invalid_is_counted_separately(self):
        r = self.run_eval(reference_valid=[[0, 30]])
        self.assertGreater(r['counts']['outside_reference_valid'], 0)
        self.assertEqual(r['counts']['unmatched_inside_valid'], 0)

    def test_empty_or_single_estimate_fails(self):
        for n in (0, 1):
            with self.assertRaises(ValueError):
                self.run_eval(est_t=self.est_t[:n], est_R=self.est_R[:n], est_p=self.est_p[:n])


if __name__ == '__main__':
    unittest.main()
