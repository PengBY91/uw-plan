import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from uwbench.metrics.axes import align_yaw_translation, body_axis_errors, body_axis_sigma, exceedance


def yaw(deg, n=1):
    return np.repeat(Rotation.from_euler('z', deg, degrees=True).as_matrix()[None], n, axis=0)


class AxisMetricsTest(unittest.TestCase):
    def test_world_offset_is_reported_on_body_axes(self):
        R = yaw(90)
        error = body_axis_errors(R, np.zeros((1, 3)), R, np.array([[0.2, 0, 0]]))
        np.testing.assert_allclose(error, [[0, -0.2, 0, 0]], atol=1e-12)

    def test_known_heading_error(self):
        error = body_axis_errors(yaw(10), np.zeros((1, 3)), yaw(13), np.zeros((1, 3)))
        np.testing.assert_allclose(error[0, 3], 3.0, atol=1e-9)

    def test_alignment_recovers_frame_without_path_excitation(self):
        n = 5
        ref_R, ref_p = yaw(30, n), np.tile([1.0, 2.0, 3.0], (n, 1))
        est_R, est_p = yaw(-15, n), np.tile([0.5, 0.0, 3.0], (n, 1))
        R, t, info = align_yaw_translation(ref_R, ref_p, est_R, est_p, np.ones(n, bool))
        np.testing.assert_allclose(R@est_R, ref_R, atol=1e-12)
        np.testing.assert_allclose(est_p@R.T + t, ref_p, atol=1e-12)
        self.assertAlmostEqual(info['yaw_deg'], 45.0)
        with self.assertRaises(ValueError):
            align_yaw_translation(ref_R, ref_p, est_R, est_p, np.zeros(n, bool))

    def test_covariance_rotates_to_body_and_heading(self):
        covariance = np.zeros((1, 6, 6))
        covariance[0, 0, 0], covariance[0, 1, 1] = 0.04, 0.01
        covariance[0, 5, 5] = np.radians(2)**2  # local z is world z when level
        sigma = body_axis_sigma(yaw(90), covariance)
        np.testing.assert_allclose(sigma, [[0.1, 0.2, 0, 2]], atol=1e-12)

    def test_time_is_integrated_and_gaps_are_not_bridged(self):
        result = exceedance([0, 1, 2, 3, 10, 11], [0, 0.2, 0.2, 0.2, 0, 0], 0.15, 2.0)
        self.assertEqual(result['exceed_s'], 2.0)
        self.assertEqual(result['unavailable_s'], 7.0)
        self.assertEqual(result['within_s'], 2.0)
        self.assertEqual(result['exceed_intervals'], [{'start': 1.0, 'end': 3.0, 'duration_s': 2.0}])

    def test_empty_or_zero_tolerance_is_error(self):
        with self.assertRaises(ValueError):
            exceedance([], [], 0.15, 1)
        with self.assertRaises(ValueError):
            exceedance([0, 1], [0, 0], 0, 1)


if __name__ == '__main__':
    unittest.main()
