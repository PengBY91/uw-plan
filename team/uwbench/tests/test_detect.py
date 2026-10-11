import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from uwbench.detect import ATMOSPHERE_PA, G, RHO, information_gap, parse_diag, statistics, threshold


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
