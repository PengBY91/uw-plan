"""M3.3 state machine: hysteresis, holding times, hard faults, epochs, causality."""
import unittest

import numpy as np

from uwbench.a import health

CFG = {'t_bad_s': 1.0, 't_good_s': 3.0, 't_fail_s': 5.0, 'obs_timeout_s': 1.0, 'recover_hold_s': 2.0}
TOL = np.array([0.15, 0.15, 0.15, 3.0])
THR = np.array([0.1, np.inf, 0.1, 2.0])


def series(seconds=30.0, rate=20.0):
    t = np.arange(0, seconds, 1/rate)
    stats = np.zeros((len(t), 4))
    stats[:, 1] = np.nan
    return t, stats


def simulate(t, stats, obs_t=None, finite=None, cfg=CFG):
    alarm, status = health.latch(stats, THR, TOL, 0.1)
    obs_t = t.copy() if obs_t is None else obs_t
    finite = np.ones(len(t), bool) if finite is None else finite
    return health.run(t, alarm, status, np.sort(obs_t), finite, cfg), status


def state_at(res, t, when):
    return res['state'][int(np.searchsorted(t, when))]


class LatchTests(unittest.TestCase):
    def test_hysteresis_holds_until_release_band(self):
        s = np.array([[0, np.nan, 0, 2.5], [0, np.nan, 0, 1.8], [0, np.nan, 0, 1.75], [0, np.nan, 0, 1.6]])
        alarm, status = health.latch(s, THR, TOL, 0.1)   # release at 2.0 - 0.1*3 = 1.7
        self.assertEqual(list(alarm[:, 3]), [True, True, True, False])
        self.assertEqual(list(status[:, 1]), ['no_source']*4)

    def test_nan_is_unscorable_not_ok(self):
        s = np.array([[np.nan, np.nan, 0.0, 0.0]])
        _, status = health.latch(s, THR, TOL, 0.1)
        self.assertEqual(status[0, 0], 'unscorable')
        self.assertEqual(status[0, 2], 'ok')


class StateTests(unittest.TestCase):
    def test_initialises_through_recovering(self):
        t, stats = series()
        res, _ = simulate(t, stats)
        self.assertEqual(res['state'][0], 'RECOVERING')
        self.assertEqual(state_at(res, t, 3.0), 'NORMAL')
        self.assertEqual([tr['to'] for tr in res['transitions']], ['RECOVERING', 'NORMAL'])

    def test_short_wobble_stays_normal(self):
        t, stats = series()
        stats[(t > 10) & (t < 10.5), 3] = 5.0
        res, _ = simulate(t, stats)
        self.assertNotIn('DEGRADED', set(res['state']))

    def test_sustained_axis_alarm_degrades_then_clears(self):
        t, stats = series()
        stats[(t > 10) & (t < 15), 3] = 5.0
        res, status = simulate(t, stats)
        self.assertEqual(state_at(res, t, 10.6), 'NORMAL')
        self.assertEqual(state_at(res, t, 11.2), 'DEGRADED')
        self.assertEqual(status[int(np.searchsorted(t, 12)), 3], 'alarm')
        self.assertEqual(state_at(res, t, 17.5), 'DEGRADED')      # needs t_good = 3 s of calm
        self.assertEqual(state_at(res, t, 18.2), 'NORMAL')
        self.assertIn('alarm:yaw', [tr['reason'] for tr in res['transitions']])

    def test_observation_timeout_loses_then_recovers(self):
        t, stats = series()
        obs = t[(t < 10) | (t > 14)]
        res, _ = simulate(t, stats, obs_t=obs)
        self.assertEqual(state_at(res, t, 10.5), 'NORMAL')        # age 0.5 s < 1 s timeout
        self.assertEqual(state_at(res, t, 11.5), 'LOST')
        self.assertEqual(state_at(res, t, 14.2), 'RECOVERING')
        self.assertEqual(state_at(res, t, 16.5), 'NORMAL')

    def test_input_integrity_lost_is_a_latched_hard_fault(self):
        t, stats = series()
        alarm, status = health.latch(stats, THR, TOL, 0.1)
        res = health.run(t, alarm, status, t.copy(), np.ones(len(t), bool), CFG, input_lost=(12.0, 'imu_outage'))
        self.assertEqual(state_at(res, t, 11.5), 'NORMAL')
        self.assertTrue(all(s == 'LOST' for s in res['state'][t >= 12.0]))
        self.assertEqual(res['transitions'][-1]['reason'], 'input_imu_outage')

    def test_time_regression_opens_new_epoch(self):
        t, stats = series(10.0)
        t = np.r_[t[:100], t[100:] - 3.0]                         # clock jumps back 3 s at sample 100
        res, _ = simulate(t, stats, obs_t=t)
        self.assertEqual(res['state'][100], 'LOST')
        self.assertEqual(res['epoch_id'][99], 0)
        self.assertEqual(res['epoch_id'][100], 1)
        self.assertIn('time_regression', [tr['reason'] for tr in res['transitions']])

    def test_all_position_axes_alarm_is_lost_but_unscorable_x_is_not(self):
        t, stats = series()
        stats[t > 5, 0] = 1.0
        stats[t > 5, 2] = 1.0
        res, _ = simulate(t, stats)
        self.assertEqual(state_at(res, t, 6.5), 'DEGRADED')
        self.assertEqual(state_at(res, t, 11.5), 'LOST')
        self.assertEqual(state_at(res, t, 25.0), 'LOST')          # stays LOST while both position axes still alarm
        stats[:, 0] = np.nan                                      # no sonar: x unscorable, z alone alarms
        res, _ = simulate(t, stats)
        self.assertEqual(state_at(res, t, 20.0), 'DEGRADED')

    def test_missing_sonar_alone_never_forces_lost(self):
        t, stats = series()
        stats[:, 0] = np.nan
        res, _ = simulate(t, stats)
        self.assertNotIn('LOST', set(res['state'][10:]))

    def test_nonfinite_pose_is_hard_fault(self):
        t, stats = series()
        finite = np.ones(len(t), bool)
        finite[(t > 10) & (t < 11)] = False
        res, _ = simulate(t, stats, finite=finite)
        self.assertEqual(state_at(res, t, 10.5), 'LOST')
        self.assertIn('pose_nonfinite', [tr['reason'] for tr in res['transitions']])

    def test_causal_prefix_unchanged_by_truncation(self):
        t, stats = series()
        stats[(t > 8) & (t < 20), 3] = 5.0
        full, _ = simulate(t, stats)
        cut = 300
        part, _ = simulate(t[:cut], stats[:cut], obs_t=t[:cut])
        self.assertEqual(list(full['state'][:cut]), list(part['state']))

    def test_durations_cover_the_run(self):
        t, stats = series()
        res, _ = simulate(t, stats)
        d = health.durations(t, res['state'])
        self.assertAlmostEqual(sum(d.values()), t[-1] - t[0], places=6)


if __name__ == '__main__':
    unittest.main()
