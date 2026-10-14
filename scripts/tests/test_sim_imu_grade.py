import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from sim_imu_grade import check_reconstruction, synthesize, true_body_rate

RATE = 200.0


def trajectory(seconds=60.0, sigma=0.002, seed=0):
    """Known body rate (slow yaw plus a pitch wobble) -> gt poses and a simulator-like imu array with white noise."""
    t = np.arange(0, seconds, 1/RATE)
    w = np.column_stack([np.zeros_like(t), 0.2*np.cos(0.5*t), 0.05 + 0*t])
    rot = [Rotation.identity()]
    for k in range(len(t) - 1):  # exact midpoint integration so the central difference sees w at the sample
        rot.append(rot[-1]*Rotation.from_rotvec((w[k] + w[k + 1])/2/RATE))
    T = np.tile(np.eye(4), (len(t), 1, 1))
    T[:, :3, :3] = Rotation.concatenate(rot).as_matrix()
    gt = np.column_stack([t, T.reshape(len(t), 16)])
    rng = np.random.default_rng(seed)
    bias = np.array([1e-4, -2e-4, 3e-4])
    imu = np.zeros((len(t), 13))
    imu[:, 0] = t
    imu[:, 1:4] = [0, 0, -9.8]
    imu[:, 4:7] = w + bias + rng.standard_normal((len(t), 3))*sigma
    imu[:, 7:10] = 0.01
    imu[:, 10:13] = bias
    return imu, gt, w


class SimImuGradeTest(unittest.TestCase):
    def test_true_rate_recovers_known_rate(self):
        imu, gt, w = trajectory()
        np.testing.assert_allclose(true_body_rate(gt)[1:-1], w[1:-1], atol=1e-4)

    def test_reconstruction_check_accepts_matching_and_rejects_wrong_sigma(self):
        imu, gt, _ = trajectory()
        stats = check_reconstruction(imu, gt, 0.002)
        np.testing.assert_allclose(stats['residual_std_rad_s'], 0.002, rtol=0.05)
        with self.assertRaises(SystemExit):
            check_reconstruction(imu, gt, 0.004)

    def test_synthesized_noise_matches_requested_grade(self):
        imu, gt, w = trajectory(seconds=300.0)
        N, K = 1e-4, 1e-5
        new = synthesize(imu, gt, N, K, seed=3)
        np.testing.assert_array_equal(new[:, :4], imu[:, :4])          # time and accel untouched
        np.testing.assert_array_equal(new[:, 7:10], imu[:, 7:10])
        np.testing.assert_array_equal(new[0, 10:13], imu[0, 10:13])    # starts at the simulator's true bias
        white = new[:, 4:7] - new[:, 10:13] - true_body_rate(gt)
        np.testing.assert_allclose(white[1:-1].std(0), N*np.sqrt(RATE), rtol=0.05)
        steps = np.diff(new[:, 10:13], axis=0)
        np.testing.assert_allclose(steps.std(0), K/np.sqrt(RATE), rtol=0.05)

    def test_seed_reproducible(self):
        imu, gt, _ = trajectory(seconds=5.0)
        np.testing.assert_array_equal(synthesize(imu, gt, 1e-4, 1e-5, 7), synthesize(imu, gt, 1e-4, 1e-5, 7))


if __name__ == '__main__':
    unittest.main()
