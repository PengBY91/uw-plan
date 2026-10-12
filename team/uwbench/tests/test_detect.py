import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from uwbench.detect import (ATMOSPHERE_PA, G, RHO, drift_budget_deg, fit_line, information_gap, parse_diag, sonar_conflict,
                            sonar_first_returns, statistics, threshold)


def synthetic(seed=0, yaw_drift=0.0):
    """20 Hz estimate with a slow yaw turn; 200 Hz gyro consistent with it; optional extra estimated yaw drift (deg/s)."""
    rng = np.random.default_rng(seed)
    imu_t = np.arange(0, 40, 0.005)
    rate = np.radians(2.0)  # true yaw rate about body z
    gyro = np.column_stack([imu_t, rng.normal(0, 1e-4, len(imu_t)), rng.normal(0, 1e-4, len(imu_t)), np.full(len(imu_t), rate)])
    est_t = np.arange(1, 40, 0.05)
    yaw = rate*est_t + np.radians(yaw_drift)*np.clip(est_t - 20, 0, None)
    R = Rotation.from_euler('z', yaw).as_matrix()
    p = np.column_stack([0.1*est_t, np.zeros_like(est_t), -5 + 0.01*est_t])
    cov = np.tile(np.diag([1e-3, 4e-3, 1e-4, 1e-6, 1e-6, 1e-4]), (len(est_t), 1, 1))
    odom = np.column_stack([imu_t[imu_t >= 1], gyro[imu_t >= 1, 1:]])  # zero estimated bias
    pressure_t = np.arange(0, 40, 0.02)
    depth = 5 - 0.01*pressure_t
    pressure = np.column_stack([pressure_t, ATMOSPHERE_PA + RHO*G*depth])
    upd_t = est_t[::2]
    rows = np.column_stack([upd_t, np.ones(len(upd_t)), np.full(len(upd_t), 30), np.zeros(len(upd_t)),
                            rng.uniform(40, 80, len(upd_t)), np.full(len(upd_t), 60), rng.uniform(40, 80, len(upd_t)),
                            np.full(len(upd_t), 60)])
    gains = np.array([np.diag(rng.uniform(1, 10, 6)) for _ in upd_t])
    return est_t, (est_t, R, p, cov), (rows, gains), gyro, odom, pressure


def truncate(data, t_cut):
    t, est, (rows, gains), imu, odom, pressure = data
    k = est[0] <= t_cut
    u = rows[:, 0] <= t_cut
    return (t[t <= t_cut], tuple(x[k] for x in est), (rows[u], gains[u]), imu[imu[:, 0] <= t_cut],
            odom[odom[:, 0] <= t_cut], pressure[pressure[:, 0] <= t_cut])


class DetectTest(unittest.TestCase):
    def test_statistics_are_causal(self):
        full = synthetic(yaw_drift=0.5)
        s_full = statistics(*full)
        for t_cut in (12.0, 25.0):
            cut = truncate(full, t_cut)
            s_cut = statistics(*cut)
            n = len(cut[0])
            for key in s_cut:
                np.testing.assert_allclose(np.nan_to_num(s_full[key][:n]), np.nan_to_num(s_cut[key]), atol=1e-9, err_msg=key)

    def test_heading_conflict_sees_drift_that_the_gyro_does_not_explain(self):
        quiet = statistics(*synthetic(yaw_drift=0.0))['conflict'][:, 3]
        drift = statistics(*synthetic(yaw_drift=0.5))['conflict'][:, 3]
        self.assertLess(quiet.max(), 0.1)
        self.assertGreater(drift[-1], 4.0)  # 0.5 deg/s over a 10 s window
        self.assertLess(drift[:300].max(), 0.1)  # before the drift starts at 20 s

    def test_cumulative_heading_conflict_keeps_drift_after_it_stops_growing(self):
        stats = statistics(*synthetic(yaw_drift=0.5))
        t = synthetic()[0]
        self.assertAlmostEqual(stats['conflict_cum'][-1, 3], 0.5*(t[-1] - 20), delta=0.2)
        self.assertLess(stats['conflict_cum'][t < 20, 3].max(), 0.1)

    def test_frozen_bias_keeps_drift_that_the_online_bias_absorbed(self):
        t, est, diag, imu, odom, pressure = synthetic(yaw_drift=0.5)
        absorbed = odom.copy()  # online bias moves to explain the estimated drift: corrected gyro = gyro + drift rate
        absorbed[absorbed[:, 0] > 20, 3] += np.radians(0.5)
        stats = statistics(t, est, diag, imu, absorbed, pressure, gyro_noise=(1e-4, 1e-5))
        self.assertLess(stats['conflict_cum'][:, 3].max(), 0.1)
        self.assertAlmostEqual(stats['conflict_cum_frozen'][-1, 3], 0.5*(t[-1] - 20), delta=0.2)
        np.testing.assert_allclose(stats['conflict_cum_frozen_excess'][:, 3],
                                   stats['conflict_cum_frozen'][:, 3] - stats['budget_yaw_deg'])
        true_bias = np.column_stack([imu[:, 0], np.zeros((len(imu), 3))])
        oracle = statistics(t, est, diag, imu, absorbed, pressure, true_bias=true_bias)['conflict_cum_true'][:, 3]
        np.testing.assert_allclose(oracle, stats['conflict_cum_frozen'][:, 3], atol=1e-6)  # frozen bias is the true one here

    def test_drift_budget_grows_with_white_noise_and_bias_walk(self):
        self.assertAlmostEqual(drift_budget_deg(100.0, 1e-3, 0.0, 1.0), np.degrees(1e-2))
        self.assertAlmostEqual(drift_budget_deg(30.0, 0.0, 1e-4, 3.0), np.degrees(3e-4*np.sqrt(30.0**3/3)))
        self.assertEqual(drift_budget_deg(-1.0, 1e-3, 1e-4, 3.0), 0.0)

    def test_sonar_first_returns_follow_imaging_sonar_layout(self):
        geometry = {'RangeMin': 0.0, 'RangeMax': 10.0, 'Azimuth': 120.0}
        image = np.zeros((100, 3))
        image[20, 1] = image[50, 1] = 0.8  # straight ahead: first return at 2.05 m, the later one is ignored
        image[30, 2] = 0.8  # column 2 is +40 deg, to the left (y > 0)
        pts = sonar_first_returns(image, geometry, 0.25)
        np.testing.assert_allclose(pts[0], [2.05, 0.0], atol=1e-9)
        np.testing.assert_allclose(pts[1], 3.05*np.array([np.cos(np.radians(40)), np.sin(np.radians(40))]), atol=1e-9)

    def test_fit_line_finds_wall_despite_outliers(self):
        rng = np.random.default_rng(1)
        y = np.linspace(-3, 3, 60)
        wall = np.column_stack([np.full(60, 2.5) + rng.normal(0, 0.01, 60), y])
        pts = np.vstack([wall, rng.uniform(0.5, 8, (10, 2))])
        m, d = fit_line(pts, 0.05, 20, rng)
        np.testing.assert_allclose(m, [1, 0], atol=0.02)
        self.assertAlmostEqual(d, 2.5, delta=0.01)

    def test_sonar_conflict_measures_estimated_motion_toward_a_wall_the_sonar_does_not_see(self):
        est_t = np.arange(0, 20, 0.05)
        R = np.tile(np.eye(3), (len(est_t), 1, 1))
        p = np.zeros((len(est_t), 3))
        p[:, 0] = 0.02*np.clip(est_t - 5, 0, None)  # estimate drifts 2 cm/s toward the wall after 5 s
        sonar_t = np.arange(0, 20, 0.1)
        lines = np.column_stack([sonar_t, np.ones(len(sonar_t)), np.zeros(len(sonar_t)), np.full(len(sonar_t), 2.5)])
        lines[50:60, 1:3] = [0, 1]  # another structure in view: outside the gate, skipped
        out = sonar_conflict(est_t, (est_t, R, p), lines, np.eye(4), 15.0)
        self.assertAlmostEqual(out[-1, 0], 0.02*(est_t[-1] - 5), delta=0.01)
        self.assertLess(out[est_t < 5, 0].max(), 1e-9)
        np.testing.assert_allclose(out[:, 1], 0.0, atol=1e-9)

    def test_sonar_reference_plane_averages_noisy_normals(self):
        rng = np.random.default_rng(2)
        est_t = np.arange(0, 120, 0.05)
        R = np.tile(np.eye(3), (len(est_t), 1, 1))
        p = np.column_stack([np.zeros_like(est_t), 0.2*est_t, np.zeros_like(est_t)])  # 24 m along the wall, no error
        sonar_t = np.arange(0, 120, 0.1)
        angle = rng.normal(0, np.radians(0.5), len(sonar_t))  # per-frame normal noise of the line fit
        lines = np.column_stack([sonar_t, np.cos(angle), np.sin(angle), np.full(len(sonar_t), 2.0)])
        single = sonar_conflict(est_t, (est_t, R, p), lines, np.eye(4), 15.0, 0.0)[:, 0]
        averaged = sonar_conflict(est_t, (est_t, R, p), lines, np.eye(4), 15.0, 5.0)
        self.assertLess(averaged[:, 0].max(), 0.1)
        self.assertLess(averaged[:, 0].max(), single.max())
        np.testing.assert_allclose(averaged[est_t < 5, 0], 0.0)  # nothing before the plane is fixed

    def test_depth_conflict_is_zero_when_estimate_follows_gauge(self):
        conflict_z = statistics(*synthetic())['conflict'][:, 2]
        self.assertLess(conflict_z.max(), 1e-3)  # only the 50 Hz sample hold of the latest gauge reading

    def test_information_gap_counts_time_since_last_strong_window(self):
        t = np.arange(0, 10, 1.0)
        lam = np.column_stack([[5, 5, 0, 0, 0, 5, 0, 0, 0, 0]])
        np.testing.assert_allclose(information_gap(t, lam, [1.0])[:, 0], [0, 0, 1, 2, 3, 0, 1, 2, 3, 4])

    def test_threshold_meets_target_on_training_normals(self):
        values = np.arange(100.0)
        self.assertEqual(threshold(values, 0.0), 99.0)
        self.assertLessEqual(np.mean(values > threshold(values, 0.05)), 0.05)

    def test_parse_diag_rebuilds_symmetric_gain(self):
        upper = ' '.join(str(float(i)) for i in range(21))
        with tempfile.TemporaryDirectory() as d:
            log = Path(d)/'estimator.log'
            log.write_text('noise\n[run-1] \x1b[0m[DIAG] SLAM 12.5 25 1 2.5 48 9.5 50 ' + upper + '\n')
            rows, gains = parse_diag(log)
        self.assertEqual(rows.shape, (1, 8))
        self.assertEqual(rows[0, 1], 1)
        np.testing.assert_allclose(gains[0], gains[0].T)
        self.assertEqual(gains[0][0, 5], 5.0)
        self.assertEqual(gains[0][5, 0], 5.0)
        with tempfile.TemporaryDirectory() as d:
            (Path(d)/'estimator.log').write_text('no diagnostics\n')
            with self.assertRaises(ValueError):
                parse_diag(Path(d)/'estimator.log')


if __name__ == '__main__':
    unittest.main()
