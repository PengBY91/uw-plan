import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from uwbench.b.deferred import (DEPENDS, DeferredCommit, HealthTrack, body_errors, health_decision, oracle_decision,
                                prefix_decision)
from uwbench.b.voxel_map import VoxelGrid


def track():
    t = [0.0, 1.0, 2.0, 3.0, 4.0]
    state = ['RECOVERING', 'NORMAL', 'NORMAL', 'DEGRADED', 'NORMAL']
    axes = [['ok', 'no_source', 'ok', 'ok'],
            ['ok', 'no_source', 'ok', 'ok'],
            ['unscorable', 'no_source', 'alarm', 'ok'],
            ['alarm', 'no_source', 'ok', 'ok'],
            ['ok', 'no_source', 'ok', 'alarm']]
    return HealthTrack(t, state, axes)


def pose(x=0.0, yaw_deg=0.0):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler('z', yaw_deg, degrees=True).as_matrix()
    T[0, 3] = x
    return T


def build(plan, poses_by_source):
    """Tiny map: each event integrates one ray from the pose along +y, 1 m long."""
    g = VoxelGrid(origin=(-1, -1, -1), shape=(30, 30, 20), resolution=0.1)
    for t, kind, i, source in plan:
        T = poses_by_source[source](t)
        g.integrate_rays(T[:3, 3], [T[:3, 3] + T[:3, :3] @ np.array([0, 1.0, 0])], t)
    return g


class HealthDecisionTest(unittest.TestCase):
    def test_rules(self):
        d = health_decision(track())
        self.assertEqual(d(-0.5, 'depth'), (False, 'no_health'))
        self.assertEqual(d(0.5, 'sonar'), (False, 'state:RECOVERING'))
        self.assertEqual(d(1.5, 'depth'), (True, 'commit'))           # y 'no_source' never holds stereo back
        self.assertEqual(d(2.5, 'depth'), (False, 'axis:z'))          # stereo depends on z
        self.assertEqual(d(2.5, 'sonar'), (True, 'commit'))           # sonar does not; x unscorable is not an alarm
        self.assertEqual(d(3.5, 'sonar'), (False, 'state:DEGRADED'))
        self.assertEqual(d(4.5, 'sonar'), (False, 'axis:yaw'))

    def test_strict_block_on_unscorable(self):
        d = health_decision(track(), block=('alarm', 'unscorable'))
        self.assertEqual(d(2.5, 'sonar'), (False, 'axis:x'))

    def test_causal(self):
        """Decisions at t use only health samples at or before t: truncating the future changes nothing."""
        full, h = track(), track()
        cut = HealthTrack(h.t[:3], h.state[:3], h.axis_status[:3])
        for t in np.arange(-0.5, 2.9, 0.25):
            for kind in DEPENDS:
                self.assertEqual(health_decision(full)(t, kind), health_decision(cut)(t, kind))

    def test_unordered_rejected(self):
        with self.assertRaises(ValueError):
            HealthTrack([0.0, 2.0, 1.0], ['NORMAL']*3, [['ok']*4]*3)


class BufferTest(unittest.TestCase):
    def events(self):
        return [(t, kind, i) for i, t in enumerate(np.arange(0, 5, 0.25)) for kind in ('depth', 'sonar')]

    def test_never_drops(self):
        dc = DeferredCommit(health_decision(track()))
        ev = self.events()
        for e in ev:
            dc.offer(*e)
        s = dc.summary()
        self.assertEqual(s['committed'] + s['still_buffered'], len(ev))
        self.assertGreater(s['still_buffered'], 0)
        moved = dc.correct(2.0)
        s2 = dc.summary()
        self.assertEqual(s2['committed'] + s2['still_buffered'], len(ev))
        self.assertEqual(s2['recommitted'], moved)
        self.assertTrue(all(b[0] > 2.0 for b in dc.buffer))

    def test_commit_all_equals_ungated(self):
        ev = self.events()
        dc = DeferredCommit(lambda t, kind: (True, 'commit'))
        for e in ev:
            dc.offer(*e)
        est = lambda t: pose(x=0.1*t)
        a = build(dc.replay_order(), {'estimate': est})
        b = build([e + ('estimate',) for e in sorted(ev)], {'estimate': est})
        np.testing.assert_array_equal(a.log_odds, b.log_odds)
        np.testing.assert_array_equal(a.observed, b.observed)

    def test_recommit_equals_replay_with_corrected_poses(self):
        """Everything buffered, then one correction: the map equals a replay from scratch with the corrected poses."""
        ev = self.events()
        dc = DeferredCommit(lambda t, kind: (False, 'state:LOST'))
        for e in ev:
            dc.offer(*e)
        dc.correct(10.0)
        est, corr = (lambda t: pose(x=0.3*t, yaw_deg=5*t)), (lambda t: pose(x=0.1*t))
        a = build(dc.replay_order(), {'estimate': est, 'corrected': corr})
        b = build([e + ('corrected',) for e in sorted(ev)], {'corrected': corr})
        np.testing.assert_array_equal(a.log_odds, b.log_odds)

    def test_mixed_replay_is_time_ordered(self):
        dc = DeferredCommit(health_decision(track()))
        for e in self.events():
            dc.offer(*e)
        dc.correct(10.0)
        order = dc.replay_order()
        self.assertEqual([e[:3] for e in order], sorted(e[:3] for e in order))
        self.assertEqual({e[3] for e in order}, {'estimate', 'corrected'})


class PrefixTest(unittest.TestCase):
    def test_prefix(self):
        d = prefix_decision(2.0)
        self.assertEqual(d(2.0, 'depth'), (True, 'commit'))
        self.assertEqual(d(2.01, 'sonar'), (False, 'after_prefix'))


class OracleTest(unittest.TestCase):
    def test_body_errors(self):
        T_true = pose(yaw_deg=90)
        T_est = pose(yaw_deg=92)
        T_est[:3, 3] = [0.0, 0.2, 0.0]   # world +y is body +x after a 90 deg yaw
        e = body_errors(T_est, T_true)
        np.testing.assert_allclose(e, [0.2, 0.0, 0.0, 2.0], atol=1e-9)

    def test_gate(self):
        true = lambda t: pose()
        est = lambda t: None if t < 0 else pose(x=0.1*t)
        d = oracle_decision(est, true)
        self.assertEqual(d(-1, 'depth'), (False, 'no_pose'))
        self.assertEqual(d(1.0, 'sonar'), (True, 'commit'))
        self.assertEqual(d(2.0, 'sonar'), (False, 'error:x'))


if __name__ == '__main__':
    unittest.main()
