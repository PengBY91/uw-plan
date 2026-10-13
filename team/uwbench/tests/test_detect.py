import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from uwbench.detect import (ATMOSPHERE_PA, CONFIG, G, RHO, drift_budget_deg, fit_line, information_gap, parse_diag, sonar_conflict,
                            sonar_conflict_map, sonar_conflict_multi,
                            sonar_echo_thickness, sonar_first_returns, sonar_lines, statistics, threshold, pooled, AXES, footprint_correct,
                            rereference, union_thresholds, union_false_alarm, comparable, evaluate, budget_horizon_s)


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
        t = full[0]
        sonar_t = np.arange(0, 40, 0.1)
        lines = np.column_stack([sonar_t, np.ones(len(sonar_t)), np.zeros(len(sonar_t)), np.full(len(sonar_t), 3.0)])
        true_bias = np.column_stack([full[3][:, 0], np.zeros((len(full[3]), 3))])
        window = t < 11.0  # the label alignment window: first 10 s after the first estimate at t = 1
        extras = dict(gyro_noise=(1e-4, 1e-5))
        s_full = statistics(*full, sonar=(lines, np.eye(4)), true_bias=true_bias, window=window, **extras)
        for t_cut in (6.0, 12.0, 25.0):  # inside the window, just after it, late
            cut = truncate(full, t_cut)
            n = len(cut[0])
            s_cut = statistics(*cut, sonar=(lines[sonar_t <= t_cut], np.eye(4)), true_bias=true_bias[true_bias[:, 0] <= t_cut],
                               window=window[:n], **extras)
            for key in s_cut:
                if np.ndim(s_cut[key]) == 0:
                    continue  # counts such as sonar_planes
                np.testing.assert_allclose(np.nan_to_num(s_full[key][:n]), np.nan_to_num(s_cut[key]), atol=1e-9, err_msg=key)

    def test_rereference_subtracts_the_causal_window_mean(self):
        signed = np.array([1.0, 3.0, np.nan, 5.0, 10.0, 20.0])
        window = np.array([True, True, True, True, False, False])
        out = rereference(signed, window)
        np.testing.assert_allclose(out[:2], [0.0, 1.0])  # running mean inside the window: 1, then 2
        self.assertTrue(np.isnan(out[2]))
        np.testing.assert_allclose(out[3:], [5 - 3, 10 - 3, 20 - 3])  # full window mean (1, 3, 5) afterwards
        np.testing.assert_allclose(rereference(signed, None), signed)
        np.testing.assert_allclose(rereference(signed, np.zeros(6, bool)), signed)  # no window sample: t_0 reference
        two = rereference(np.column_stack([signed, -signed]), window)
        np.testing.assert_allclose(two[:, 1], -out)

    def test_window_reference_matches_what_the_labels_measure(self):
        # estimated z drifts 1 cm/s away from the gauge during the first 10 s, then stays: the label (aligned over that
        # window) reads 0.1 - mean(drift over the window) = 0.05 m, the t_0 reference reads 0.1 m
        t, est, diag, imu, odom, pressure = synthetic()
        est_t, R, p, cov = est
        p = p.copy()
        p[:, 2] += 0.01*np.clip(est_t - 1, 0, 10)
        window = est_t < 11.0
        legacy = statistics(t, (est_t, R, p, cov), diag, imu, odom, pressure, cfg={**CONFIG, 'conflict_reference': 't0'}, window=window)
        aligned = statistics(t, (est_t, R, p, cov), diag, imu, odom, pressure, window=window)
        self.assertAlmostEqual(legacy['conflict_cum'][-1, 2], 0.10, delta=0.003)
        self.assertAlmostEqual(aligned['conflict_cum'][-1, 2], 0.10 - np.mean(0.01*np.clip(est_t[window] - 1, 0, 10)), delta=0.003)
        np.testing.assert_allclose(aligned['conflict'], legacy['conflict'])  # the windowed form is untouched
        # a wall at world x = 100 m with normal +x; the vehicle yaws at 2 deg/s, so the sonar-frame normal turns the other way
        sonar_t = np.arange(0, 40, 0.1)
        yaw = np.radians(2.0)*sonar_t
        lines = np.column_stack([sonar_t, np.cos(yaw), -np.sin(yaw), 100 - 0.1*sonar_t])
        p2 = est[2].copy()
        p2[:, 0] += 0.02*np.clip(est_t - 1, 0, 5)  # 0.1 m toward the wall inside the window, mean over the window 0.075
        s = statistics(t, (est_t, R, p2, cov), diag, imu, odom, pressure, sonar=(lines, np.eye(4)), window=window)
        # the vehicle has yawed 80 deg by the end, so the conflict along the wall normal is split over body x and y
        self.assertAlmostEqual(np.hypot(*s['conflict_cum_sonar'][-1, :2]), 0.1 - np.mean(0.02*np.clip(est_t[window] - 1, 0, 5)), delta=0.005)

    def test_scalar_baseline_folds_the_main_detector_sources(self):
        t, est, diag, imu, odom, pressure = synthetic(yaw_drift=0.5)
        sonar_t = np.arange(0, 40, 0.1)
        lines = np.column_stack([sonar_t, np.ones(len(sonar_t)), np.zeros(len(sonar_t)), np.full(len(sonar_t), 3.0)])
        tol = np.array([0.15, 0.15, 0.15, 3.0])
        s = statistics(t, est, diag, imu, odom, pressure, cfg={**CONFIG, 'dir_yaw_source': 'online'}, sonar=(lines, np.eye(4)), tol=tol)
        norm = s['conflict_dir']/tol
        expect = np.nanmax(norm, axis=1)
        np.testing.assert_allclose(s['DET-SCALAR'], expect)
        np.testing.assert_allclose(s['DET-SCALAR-l2'], np.sqrt(np.nansum(np.maximum(norm, 0)**2, axis=1)))
        self.assertGreater(s['DET-SCALAR'][-1], 3.0)  # 10 deg of heading drift is 3.3 tolerances
        # with the frozen-bias heading minus its budget the heading column goes negative while the gyro drifts less
        # than the budget allows; that must not count as evidence in the l2 form
        frozen = statistics(t, est, diag, imu, odom, pressure, gyro_noise=(1e-2, 1e-2), sonar=(lines, np.eye(4)), tol=tol)
        neg = frozen['conflict_dir'][:, 3] < 0
        self.assertTrue(neg.any())
        xz = np.nanmax(np.abs(frozen['conflict_dir'][neg][:, [0, 2]])/tol[[0, 2]], axis=1)
        self.assertTrue((frozen['DET-SCALAR-l2'][neg] <= np.sqrt(2)*xz + 1e-12).all())
        without = statistics(t, est, diag, imu, odom, pressure, tol=tol)  # no sonar: DET-DIR and DET-SCALAR still exist
        self.assertTrue(np.isnan(without['conflict_dir'][:, 0]).all())
        np.testing.assert_allclose(without['conflict_dir'][:, 2], without['conflict_cum'][:, 2])
        self.assertTrue(np.isfinite(without['DET-SCALAR']).all())

    def test_union_thresholds_spend_one_budget_whatever_the_channel_count(self):
        rng = np.random.default_rng(3)
        cols = [rng.normal(size=4000) for _ in range(4)]
        cols[1][::7] = np.nan  # an unscorable share never alarms
        for target in (0.0, 0.01, 0.05):
            thr = union_thresholds(cols, target)
            fa = union_false_alarm(cols, thr)
            self.assertLessEqual(fa, target)
            if target > 0:  # and not needlessly strict: a slightly larger common level would overshoot
                looser = [threshold(c, target/4*1.5) for c in cols]
                self.assertGreater(union_false_alarm(cols, looser), fa)
        fixed = union_thresholds(cols, 0.01, 'fixed', [0.01/4]*4)
        np.testing.assert_allclose(fixed, [threshold(c, 0.01/4) for c in cols])
        self.assertEqual(union_thresholds([cols[0]], 0.01), [threshold(cols[0], 0.01)])
        self.assertTrue(comparable(0.012, 0.01, 0.5))
        self.assertFalse(comparable(0.02, 0.01, 0.5))
        self.assertTrue(comparable(0.0, 0.0, 0.5))
        self.assertFalse(comparable(0.001, 0.0, 0.5))
        self.assertIsNone(comparable(None, 0.01, 0.5))

    @staticmethod
    def toy_runs(n_runs=3, n=600, seed=0):
        """Runs with precomputed statistics: an exceedance on x and yaw in the second half, a NaN detector column on x
        for one run (the sonar sees nothing), y never sourced."""
        rng = np.random.default_rng(seed)
        runs = []
        for i in range(n_runs):
            t = 1700000000 + np.arange(n)*0.05
            errors = np.zeros((n, 4))
            errors[n//2:, 0] = 0.3
            errors[n//2:, 1] = 0.2  # y exceeds too; no detector has a y source
            errors[n//2:, 3] = 5.0
            ramp = np.clip((np.arange(n) - n//2)/50.0, 0, None)
            stats = {'DET-NIS': rng.uniform(0, 1, n), 'DET-FEAT': -rng.integers(20, 40, n).astype(float),  # blind to the failure
                     'DET-COV': rng.uniform(0.1, 0.3, (n, 4)), 'lambda': rng.uniform(1, 10, (n, 4)),
                     'conflict': np.column_stack([np.zeros(n), np.zeros(n), rng.uniform(0, 0.01, n), rng.uniform(0, 0.1, n)])}
            cum = np.column_stack([np.zeros(n), np.zeros(n), rng.uniform(0, 0.01, n), 0.05*rng.uniform(0, 1, n) + 4.0*ramp])
            stats['conflict_cum'] = cum
            stats['conflict_cum_frozen'] = cum.copy()
            stats['conflict_cum_frozen_excess'] = cum.copy()
            x = 0.01*rng.uniform(0, 1, n) + 0.25*ramp
            if i == 0:
                x[n//2:] = np.nan  # no measurement during the exceedance in this run
            stats['conflict_dir'] = np.column_stack([x, np.full(n, np.nan), cum[:, 2], cum[:, 3]])
            tol = np.array([0.15, 0.15, 0.15, 3.0])
            norm = stats['conflict_dir']/tol
            stats['DET-SCALAR'] = np.nanmax(norm, axis=1)
            stats['DET-SCALAR-l2'] = np.sqrt(np.nansum(norm**2, axis=1))
            runs.append({'name': f'r{i}', 't': t, 'scored': np.r_[np.zeros(40, bool), np.ones(n - 40, bool)],
                         'errors': errors, 'tol': tol, 'stats': stats})
        return runs

    def test_evaluate_scores_every_detector_on_the_same_samples(self):
        runs = self.toy_runs()
        cfg = {**CONFIG, 'false_alarm_targets': [0.01], 'early_warning_s': [0.0]}
        rows = evaluate(runs, cfg)
        by = {(r['detector'], r['run']): r for r in rows}
        detectors = {r['detector'] for r in rows}
        self.assertTrue({'DET-NIS', 'DET-COV', 'DET-SCALAR', 'DET-SCALAR-l2', 'DET-DIR', 'DIR-cumconflict'} <= detectors)
        for run in ('r0', 'r1', 'r2'):
            exceed = {d: tuple(by[(d, run)][f'exceed_{a}'] for a in AXES) for d in detectors}
            self.assertEqual(len(set(exceed.values())), 1)  # identical denominators
            for d in detectors:
                row = by[(d, run)]
                for a in AXES:  # missed never exceeds exceed, and includes the unscorable share
                    self.assertLessEqual(row[f'unscorable_{a}'], row[f'missed_{a}'])
                    self.assertLessEqual(row[f'missed_{a}'], row[f'exceed_{a}'])
                self.assertIn('comparable', row)
        dir0 = by[('DET-DIR', 'r0')]
        self.assertEqual(dir0['unscorable_x'], dir0['exceed_x'])  # the sonar saw nothing: all x exceedances unscorable
        self.assertEqual(dir0['missed_x'], dir0['exceed_x'])  # and therefore counted as missed, not dropped
        self.assertEqual(dir0['missed_y'], dir0['exceed_y'])
        dir1 = by[('DET-DIR', 'r1')]
        self.assertLess(dir1['missed_x'], 0.2*dir1['exceed_x'])  # with a measurement the x exceedance is caught
        self.assertLess(dir1['missed_yaw'], 0.2*dir1['exceed_yaw'])
        self.assertGreater(by[('DET-NIS', 'r1')]['missed_x'], 0.5*dir1['exceed_x'])
        table = {e['detector']: e for e in pooled(rows)}
        # x, y and yaw fail together while z stays within tolerance: a global detector alarms z too (misattribution),
        # the per-axis detector does not
        self.assertGreater(table['DET-SCALAR']['r_offaxis_z'], 0.8)
        self.assertLess(table['DET-DIR']['r_offaxis_z'], 0.05)
        self.assertAlmostEqual(table['DET-DIR']['r_miss_y'], 1.0)
        self.assertIsNone(table['DET-DIR']['r_miss_scorable_y'])
        self.assertGreater(table['DET-DIR']['r_miss_x'], table['DET-DIR']['r_miss_scorable_x'])
        for e in table.values():  # the union calibration keeps the training false alarms at or under the target
            self.assertLessEqual(max(r['train_false_alarm_fraction'] for r in rows if r['detector'] == e['detector']), 0.01 + 1e-12)
        legacy = {e['detector']: e for e in pooled(evaluate(runs, {**cfg, 'threshold_allocation': 'fixed'}))}
        self.assertEqual(set(legacy), set(table))

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

    def test_main_detector_takes_one_source_per_axis(self):
        t, est, diag, imu, odom, pressure = synthetic(yaw_drift=0.05)
        sonar_t = np.arange(0, 40, 0.1)
        lines = np.column_stack([sonar_t, np.ones(len(sonar_t)), np.zeros(len(sonar_t)), np.full(len(sonar_t), 3.0)])
        s = statistics(t, est, diag, imu, odom, pressure, gyro_noise=(1e-4, 1e-5), sonar=(lines, np.eye(4)))
        d = s['conflict_dir']
        np.testing.assert_array_equal(d[:, 0], s['conflict_cum_sonar'][:, 0])  # x: single-plane sonar (NaN kept)
        self.assertTrue(np.isnan(d[:, 1]).all())  # y: no source
        np.testing.assert_array_equal(d[:, 2], s['conflict_cum'][:, 2])  # z: depth gauge
        np.testing.assert_allclose(d[:, 3], s['conflict_cum_frozen'][:, 3] - s['budget_yaw_deg'])  # heading
        self.assertGreater(s['budget_yaw_deg'][-1], 0.0)
        online = statistics(t, est, diag, imu, odom, pressure, cfg={**CONFIG, 'dir_yaw_source': 'online'},
                            gyro_noise=(1e-4, 1e-5), sonar=(lines, np.eye(4)))
        np.testing.assert_array_equal(online['conflict_dir'][:, 3], online['conflict_cum'][:, 3])

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

    def test_echo_thickness_separates_vertical_from_inclined_faces(self):
        geometry = {'RangeMin': 0.0, 'RangeMax': 10.0, 'Azimuth': 120.0}
        vertical, inclined = np.zeros((100, 30)), np.zeros((100, 30))
        vertical[20:22] = 0.8  # 2.0-2.2 m: a vertical face spans D to D/cos(10 deg) plus a bin
        inclined[20:28] = 0.8  # 2.0-2.8 m: the elevation fan slides down a tilted face
        inclined[29, :] = 0.8  # a one-row dropout does not end the echo
        inclined[60, :] = 0.8  # a separate later return (after a gap) is not part of it
        self.assertAlmostEqual(sonar_echo_thickness(vertical, geometry, 0.25), 2/20.5)
        self.assertAlmostEqual(sonar_echo_thickness(inclined, geometry, 0.25), 10/20.5)
        self.assertTrue(np.isnan(sonar_echo_thickness(np.zeros((100, 30)), geometry, 0.25)))

    def test_sonar_lines_skip_frames_without_a_vertical_face(self):
        geometry = {'RangeMin': 0.0, 'RangeMax': 10.0, 'Azimuth': 60.0}
        az = np.radians(-30 + (np.arange(64) + 0.5)*60/64)
        vertical = np.zeros((200, 64))
        vertical[np.round(2.5/np.cos(az)/0.05 - 0.5).astype(int), np.arange(64)] = 0.8  # wall at x = 2.5 m
        inclined = vertical.copy()
        for k in range(1, 12):
            inclined[np.round(2.5/np.cos(az)/0.05 - 0.5).astype(int) + k, np.arange(64)] = 0.8
        lines = sonar_lines(np.array([0.0, 0.1]), [vertical, inclined], geometry)
        np.testing.assert_allclose(lines[0, 1:], [1, 0, 2.5], atol=0.03)
        self.assertTrue(np.isnan(lines[1, 1:]).all())

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
        self.assertLess(np.nanmax(averaged[:, 0]), 0.1)
        self.assertLess(np.nanmax(averaged[:, 0]), single.max())
        self.assertTrue(np.isnan(averaged[est_t < 5, 0]).all())  # not scorable before the plane is fixed

    def test_sonar_conflict_is_not_scorable_while_frames_are_missing(self):
        est_t = np.arange(0, 20, 0.05)
        R = np.tile(np.eye(3), (len(est_t), 1, 1))
        p = np.zeros((len(est_t), 3))
        p[:, 0] = 0.02*est_t  # estimate drifts 2 cm/s toward the wall
        sonar_t = np.arange(0, 20, 0.1)
        lines = np.column_stack([sonar_t, np.ones(len(sonar_t)), np.zeros(len(sonar_t)), np.full(len(sonar_t), 2.5)])
        lines[(sonar_t >= 5) & (sonar_t < 12), 1:] = np.nan  # gated (inclined face) or no wall: no measurement
        held = sonar_conflict(est_t, (est_t, R, p), lines, np.eye(4), 15.0)
        out = sonar_conflict(est_t, (est_t, R, p), lines, np.eye(4), 15.0, hold_s=1.0)
        gap = (est_t >= 5.95) & (est_t < 12)  # more than 1 s after the last frame at 4.9 s
        other = (est_t < 5.85) | (est_t >= 12)
        self.assertTrue(np.isnan(out[gap]).all())
        self.assertAlmostEqual(held[gap, 0].max(), 0.02*4.9, delta=1e-6)  # the old hold kept a stale value
        np.testing.assert_allclose(out[other], held[other])  # elsewhere identical
        self.assertAlmostEqual(out[est_t >= 12, 0][0], 0.02*12, delta=1e-6)  # the cumulative value resumes in full
        multi, _ = sonar_conflict_multi(est_t, (est_t, R, p), lines, np.eye(4), 3.0, 0.15, hold_s=1.0)
        self.assertTrue(np.isnan(multi[gap]).all())

    def test_threshold_and_scoring_leave_out_unscorable_samples(self):
        self.assertEqual(threshold([np.nan, 1.0, 2.0, 3.0], 0.0), 3.0)
        self.assertEqual(threshold([np.nan, np.nan], 0.01), np.inf)
        rows = [{'early_warning_s': 0.0, 'target': 0.01, 'detector': 'D', 'run': 'r', 'normal_samples': 10,
                 'false_alarm_fraction': 0.1, 'normal_unscorable': 2, 'comparable': False,
                 **{f'exceed_{a}': 10 for a in AXES}, **{f'unscorable_{a}': 0 for a in AXES}, **{f'missed_{a}': 0 for a in AXES}}]
        rows[0].update(unscorable_x=4, missed_x=7)  # 4 unscorable (missed by definition) + 3 missed with a measurement
        x = pooled(rows)[0]
        self.assertAlmostEqual(x['r_miss_x'], 7/10)  # same denominator as every other detector
        self.assertAlmostEqual(x['r_miss_scorable_x'], 3/6)  # the pilot's definition
        self.assertAlmostEqual(x['r_unscorable_x'], 4/10)
        self.assertAlmostEqual(x['normal_unscorable_fraction'], 0.2)
        self.assertFalse(x['comparable'])  # 10 % realised against a 1 % target
        self.assertEqual(x['runs_comparable'], '0/1')
        self.assertIsNone(budget_horizon_s((0.0, 0.0), 3.0, 3.0))
        self.assertAlmostEqual(budget_horizon_s((0.0, 2.83e-4), 3.0, 3.0), 22.5, delta=0.5)  # the pilot's sim gyro

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
    def scan_scene(segments, est_offset, est_yaw_drift, duration=100.0, rays_per_beam=1):
        """Vehicle at x = 0 facing +x, moving along +y at 0.2 m/s past 2-D wall segments; 128-beam first returns
        quantised to 3.8 cm. The estimate is offset by est_offset(t) (2-vector) and rotated by est_yaw_drift(t) rad.
        rays_per_beam > 1 spreads rays across each beam's width and keeps the nearest, as the raycast sonar does."""
        est_t = np.arange(0, duration, 0.05)
        true_p = np.column_stack([0*est_t, 0.2*est_t])
        yaw = est_yaw_drift(est_t)
        R = Rotation.from_euler('z', yaw).as_matrix()
        # the estimate believes the vehicle moved along its own (rotated) heading, plus an extra offset
        steps = np.diff(true_p, axis=0, prepend=true_p[:1])
        p_est = np.cumsum(np.einsum('nij,nj->ni', R[:, :2, :2], steps), axis=0) + est_offset(est_t)
        est = (est_t, R, np.column_stack([p_est, 0*est_t]))
        az = np.radians(np.linspace(-60, 60, 128))
        width = az[1] - az[0]
        sub = az[:, None] + width*((np.arange(rays_per_beam) + 0.5)/rays_per_beam - 0.5 if rays_per_beam > 1 else np.zeros(1))
        d = np.column_stack([np.cos(sub.ravel()), np.sin(sub.ravel())])
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
            r = r.reshape(len(az), -1).min(axis=1)  # nearest ray of each beam, placed at the beam centre
            hit = np.isfinite(r)
            r = (np.floor(r[hit]/0.038) + 0.5)*0.038
            scans.append((ts, r[:, None]*np.column_stack([np.cos(az[hit]), np.sin(az[hit])])))
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

    def test_footprint_correction_puts_oblique_returns_on_the_wall(self):
        # the nearest ray of a 0.94 deg beam at 60 deg incidence is closer than the beam centre: edge points sit in front
        # of the wall; moving them to the bin edge nearest the wall puts them back on it
        _, _, scans = self.scan_scene([((2, -10), (2, 40))], lambda t: np.zeros((len(t), 2)), lambda t: 0*t,
                                      duration=0.1, rays_per_beam=16)
        pts = scans[0][1]
        edge = np.abs(np.degrees(np.arctan2(pts[:, 1], pts[:, 0]))) > 45
        half_bin = np.radians(120/127)/2
        fixed = footprint_correct(pts, half_bin)
        self.assertLess(np.mean(pts[edge, 0] - 2), -0.01)
        self.assertLess(abs(np.mean(fixed[edge, 0] - 2)), 0.004)

    def test_growing_map_bends_less_with_footprint_correction(self):
        # error-free estimate: with beam width the grown map bends toward the vehicle; the correction removes the bend
        est_t, est, scans = self.scan_scene([((2, -10), (2, 40))], lambda t: np.zeros((len(t), 2)), lambda t: 0*t,
                                            rays_per_beam=16)
        raw, _ = sonar_conflict_map(est_t, est, scans, np.eye(4), self.GROW)
        half_bin = np.radians(120/127)/2
        fixed, _ = sonar_conflict_map(est_t, est, [(ts, footprint_correct(p, half_bin)) for ts, p in scans], np.eye(4), self.GROW)
        self.assertGreater(np.nanmax(raw[:, 0]), 0.3)  # 20 m of growth: about 0.9 m and 5 deg
        self.assertGreater(np.nanmax(raw[:, 2]), 2.0)
        self.assertLess(np.nanmax(fixed[:, 0]), 0.05)
        self.assertLess(np.nanmax(fixed[:, 2]), 0.5)

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
