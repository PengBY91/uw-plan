import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from uwbench.detect import (ATMOSPHERE_PA, CONFIG, G, RHO, drift_budget_deg, fit_line, information_gap, parse_diag, sonar_conflict,
                            sonar_conflict_map, sonar_conflict_multi,
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

    @staticmethod
    def wall_scene(drift, surfaces):
        """Vehicle facing +x, moving along y at 0.2 m/s; estimate drifts toward the wall by drift(t); sonar sees surfaces(t)."""
        est_t = np.arange(0, 100, 0.05)
        R = np.tile(np.eye(3), (len(est_t), 1, 1))
        p = np.column_stack([drift(est_t), 0.2*est_t, np.zeros_like(est_t)])
        sonar_t = np.arange(0, 100, 0.1)
        rows = []
        for ts in sonar_t:
            angle_deg, wall_x = surfaces(ts)  # true wall normal heading and the wall's x at the true vehicle position
            a = np.radians(angle_deg)
            rows.append([ts, np.cos(a), np.sin(a), wall_x*np.cos(a)])  # true vehicle at x = 0
        return est_t, (est_t, R, p), np.array(rows)

    def test_multi_plane_conflict_ignores_switching_between_parallel_surfaces(self):
        # wall at 2.0 m and a buttress face at 1.4 m alternate in view; the estimate has no error
        est_t, est, lines = self.wall_scene(lambda t: 0*t, lambda ts: (0.0, 2.0 if int(ts) % 3 else 1.4))
        single = sonar_conflict(est_t, est, lines, np.eye(4), 15.0)[:, 0]
        multi, n_planes = sonar_conflict_multi(est_t, est, lines, np.eye(4), 3.0, 0.15)
        self.assertGreater(single.max(), 0.5)  # one reference plane reads the buttress as 0.6 m of conflict
        self.assertLess(multi[:, 0].max(), 1e-6)
        self.assertEqual(n_planes, 2)

    def test_multi_plane_conflict_carries_drift_across_a_new_surface(self):
        # estimate drifts 1 cm/s toward the structure; the sonar sees a new surface 0.6 m nearer from t = 50 s on
        est_t, est, lines = self.wall_scene(lambda t: 0.01*t, lambda ts: (0.0, 2.0 if ts < 50 else 1.4))
        multi, n_planes = sonar_conflict_multi(est_t, est, lines, np.eye(4), 3.0, 0.15)
        self.assertAlmostEqual(multi[-1, 0], 0.01*est_t[-1], delta=0.02)
        self.assertEqual(n_planes, 2)

    def test_multi_plane_conflict_follows_a_curved_wall(self):
        # arc wall of radius 100 m, 2 m away at y = 0: over 20 m along y the normal turns 11 deg
        est_t = np.arange(0, 100, 0.05)
        est = (est_t, np.tile(np.eye(3), (len(est_t), 1, 1)), np.column_stack([0*est_t, 0.2*est_t, 0*est_t]))
        centre, radius = np.array([102.0, 0.0]), 100.0
        rows = []
        for ts in np.arange(0, 100, 0.1):
            to_centre = centre - np.array([0.0, 0.2*ts])
            dist = np.linalg.norm(to_centre)
            rows.append([ts, *(to_centre/dist), dist - radius])  # tangent line at the closest wall point
        single = sonar_conflict(est_t, est, np.array(rows), np.eye(4), 15.0)[:, 0]
        self.assertGreater(single.max(), 0.5)
        # each local plane leaves the arc by about radius * angle^2 / 2 before it is replaced, and the hand-over carries
        # that bias on: the false conflict scales with the match angle (a known limit of chaining planes)
        bias = {}
        for match_deg in (0.5, 3.0):
            multi, n_planes = sonar_conflict_multi(est_t, est, np.array(rows), np.eye(4), match_deg, 0.15)
            bias[match_deg] = multi[:, 0].max()
            self.assertGreater(n_planes, 2)
        self.assertLess(bias[0.5], 0.1)
        self.assertLess(bias[0.5], bias[3.0])

    @staticmethod
    def scan_scene(segments, est_offset, est_yaw_drift, duration=100.0):
        """Vehicle at x = 0 facing +x, moving along +y at 0.2 m/s past 2-D wall segments; 128-beam first returns
        quantised to 3.8 cm. The estimate is offset by est_offset(t) (2-vector) and rotated by est_yaw_drift(t) rad."""
        est_t = np.arange(0, duration, 0.05)
        true_p = np.column_stack([0*est_t, 0.2*est_t])
        yaw = est_yaw_drift(est_t)
        R = Rotation.from_euler('z', yaw).as_matrix()
        # the estimate believes the vehicle moved along its own (rotated) heading, plus an extra offset
        steps = np.diff(true_p, axis=0, prepend=true_p[:1])
        p_est = np.cumsum(np.einsum('nij,nj->ni', R[:, :2, :2], steps), axis=0) + est_offset(est_t)
        est = (est_t, R, np.column_stack([p_est, 0*est_t]))
        az = np.radians(np.linspace(-60, 60, 128))
        d = np.column_stack([np.cos(az), np.sin(az)])
        a0 = np.array([seg[0] for seg in segments], float)
        a1 = np.array([seg[1] for seg in segments], float)
        scans = []
        for ts in np.arange(0, duration, 0.1):
            o = np.array([0.0, 0.2*ts])
            # solve o + r d = a0 + u (a1 - a0) for every beam and segment at once
            e = a1 - a0
            den = d[:, None, 0]*(-e[None, :, 1]) - d[:, None, 1]*(-e[None, :, 0])
            w = a0[None, :, :] - o
            with np.errstate(divide='ignore', invalid='ignore'):
                r = (w[..., 0]*(-e[None, :, 1]) - w[..., 1]*(-e[None, :, 0]))/den
                u = (d[:, None, 0]*w[..., 1] - d[:, None, 1]*w[..., 0])/den
            r = np.where((np.abs(den) > 1e-12) & (r > 0.3) & (r < 10) & (u >= 0) & (u <= 1), r, np.inf).min(axis=1)
            hit = np.isfinite(r)
            r = (np.floor(r[hit]/0.038) + 0.5)*0.038
            scans.append((ts, r[:, None]*d[hit]))
        return est_t, est, scans

    GROW = {**CONFIG, 'map_grow': True, 'map_rotation': True}

    def test_fixed_map_conflict_is_flat_along_a_wall_without_error(self):
        est_t, est, scans = self.scan_scene([((2, -10), (2, 40))], lambda t: np.zeros((len(t), 2)), lambda t: 0*t)
        out, n_map = sonar_conflict_map(est_t, est, scans, np.eye(4), CONFIG)
        # 20 m along the wall, points past the map follow the tangent of its edge; the edge normal from one quantised
        # scan is off by about 0.1 deg, which over 20 m is 4 cm
        self.assertLess(out[:, 0].max(), 0.06)
        self.assertLess(out[:, 1].max(), 0.03)  # and the registration does not slide along the wall
        self.assertLess(n_map, 200)  # one scan, not grown

    def test_fixed_map_conflict_keeps_estimated_drift_toward_the_wall(self):
        est_t, est, scans = self.scan_scene([((2, -10), (2, 40))], lambda t: np.column_stack([0.01*t, 0*t]),
                                            lambda t: 0*t)
        out, _ = sonar_conflict_map(est_t, est, scans, np.eye(4), CONFIG)
        self.assertAlmostEqual(out[-1, 0], 0.01*est_t[-1], delta=0.05)

    def test_growing_map_measures_heading_drift(self):
        # estimate heading drifts 0.05 deg/s; it walks into the wall in its own frame, which only a fixed frame shows
        drift = lambda t: np.radians(0.05)*t
        est_t, est, scans = self.scan_scene([((2, -10), (2, 40))], lambda t: np.zeros((len(t), 2)), drift)
        out, _ = sonar_conflict_map(est_t, est, scans, np.eye(4), self.GROW)
        expected = np.abs(np.sin(drift(est_t))*0.2*est_t/2)  # lateral offset of a path turning at a constant rate
        self.assertAlmostEqual(out[-1, 2], np.degrees(drift(est_t[-1])), delta=0.5)
        self.assertLess(np.abs(out[-1, 0] - expected[-1]), 0.15)

    def test_growing_map_bends_along_a_featureless_wall(self):
        # known limit: rotation from the overlap at the scan's trailing edge is weak, so the grown map bends slowly
        est_t, est, scans = self.scan_scene([((2, -10), (2, 40))], lambda t: np.zeros((len(t), 2)), lambda t: 0*t)
        out, _ = sonar_conflict_map(est_t, est, scans, np.eye(4), self.GROW)
        self.assertLess(out[:, 1].max(), 0.05)  # the degenerate direction is frozen: no slide along the wall
        self.assertGreater(out[:, 0].max(), 0.03)  # but 20 m of growth bends the wall toward the vehicle
        self.assertLess(out[:, 0].max(), 0.2)

    def test_growing_map_sees_along_wall_error_at_pilasters(self):
        # pilasters every 2 m, closer than the 7 m the fan covers at 2 m, so a mapped one is always still in view
        ys = np.arange(-8, 40, 2.0)
        wall = [((2, -10), (2, 40))] + [((1.7, y), (2, y)) for y in ys] + [((1.7, y), (1.7, y + 0.4)) for y in ys] + \
               [((1.7, y + 0.4), (2, y + 0.4)) for y in ys]
        est_t, est, scans = self.scan_scene(wall, lambda t: np.column_stack([0*t, 0.01*t]), lambda t: 0*t)
        out, _ = sonar_conflict_map(est_t, est, scans, np.eye(4), self.GROW)
        self.assertAlmostEqual(out[-1, 1], 0.01*est_t[-1], delta=0.15)
        self.assertLess(out[:, 0].max(), 0.15)

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
