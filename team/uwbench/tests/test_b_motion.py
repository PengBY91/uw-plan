import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from uwbench.b.motion import DEFAULTS, analyse, cells, in_front_of_first_echo, predicted_survivors, relative_motion, runs_count, summarise
from uwbench.sonar import range_bearing

GEOM = {'RangeBins': 256, 'AzimuthBins': 128, 'RangeMin': 0.3, 'RangeMax': 10.0, 'Elevation': 20.0, 'Azimuth': 120.0}


def render(points_world, T_world_sonar):
    """Boolean image of world points seen from a sonar pose (floor binning as the raycast generator; no occlusion)."""
    p = (points_world - T_world_sonar[:3, 3]) @ T_world_sonar[:3, :3]
    rho, th = range_bearing(p)
    ph = np.degrees(np.arctan2(p[:, 2], np.hypot(p[:, 0], p[:, 1])))
    ok = (rho > GEOM['RangeMin']) & (rho < GEOM['RangeMax']) & (np.abs(np.degrees(th)) < 60) & (np.abs(ph) <= 10)
    r, a = cells(rho[ok], th[ok], GEOM, (256, 128))
    img = np.zeros((256, 128), bool)
    img[r, a] = True
    return img


class FakeDataset:
    """Sonar at the origin of a vehicle frame, x forward, y left, z up; poses given per frame."""
    def __init__(self, poses, points_world):
        self.poses = poses
        self.sonar_t = np.arange(len(poses))*0.1
        self.T_imu_sonar = np.eye(4)
        self.sonar_geometry = GEOM
        self.points = points_world
        self.route = {'name': 'fake'}

    def gt_pose(self, t):
        return self.poses[int(round(t/0.1))]

    def sonar_points_path(self):
        return None

    def image(self, i):
        return render(self.points, self.poses[i]).astype(float)


def pose(x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler('xyz', [roll, pitch, yaw], degrees=True).as_matrix()
    T[:3, 3] = [x, y, z]
    return T


class MotionTests(unittest.TestCase):
    # a thin horizontal bar 3.5 m ahead at +0.4 m (elevation about 6.5 deg), 2 m wide; a vertical post beside it
    BAR = np.column_stack([np.full(41, 3.5), np.linspace(-1, 1, 41), np.full(41, 0.4)])
    POST = np.column_stack([np.full(31, 3.5), np.full(31, 1.5), np.linspace(-0.6, 0.6, 31)])

    def run_primitive(self, poses, cfg_over=None):
        ds = FakeDataset(poses, np.vstack([self.BAR, self.POST]))
        cfg = {**DEFAULTS, 'reference_t': 0.0, 'view_stride': 1, 'candidates': 41, **(cfg_over or {})}
        return analyse(ds, ds.image, cfg)

    @staticmethod
    def bar(rows):
        last = max(r['n_views'] for r in rows)
        return [r for r in rows if r['n_views'] == last and abs(r['theta_deg']) < 20]  # the post sits at +23 deg

    @staticmethod
    def med(rows, key):
        return float(np.median([r[key] for r in rows]))

    def test_heave_sweep_shrinks_a_thin_bar_only_under_the_aggressive_rule(self):
        # sonar sinks 0.6 m then rises 0.6 m. Under the conservative rule a candidate is cleared only when it ends up
        # more than a bin nearer than the bar's own echo while the bar is still in the fan; with a 20 deg aperture the
        # range separation is bounded by d tan^2(alpha/2) / 2 (5 cm at 3.5 m, about one 3.8 cm bin), so nothing goes.
        # The aggressive rule reads the bar leaving the fan as evidence and clears the candidates left inside it.
        poses = [pose(z=dz) for dz in list(np.linspace(0, -0.6, 7)) + list(np.linspace(-0.4, 0.6, 6))]
        rows, motions, _ = self.run_primitive(poses)
        bar = self.bar(rows)
        self.assertTrue(bar)
        self.assertGreater(self.med(bar, 'conservative_fraction_actual'), 0.8)
        self.assertLess(self.med(bar, 'aggressive_fraction_actual'), 0.4)
        for r in bar:  # the elimination never empties the set (the bar's own echo supports its candidates)
            self.assertGreater(r['conservative_fraction_actual'], 0.0)
            self.assertGreater(r['aggressive_fraction_actual'], 0.0)

    def test_sway_keeps_the_candidates(self):
        poses = [pose(y=dy) for dy in np.linspace(0, 1.0, 7)]
        rows, motions, _ = self.run_primitive(poses)
        bar = self.bar(rows)
        self.assertGreater(self.med(bar, 'conservative_fraction_actual'), 0.8)
        self.assertGreater(self.med(bar, 'aggressive_fraction_actual'), 0.6)
        self.assertGreater(self.med(bar, 'conservative_fraction_predicted'), 0.8)
        self.assertAlmostEqual(motions[-1]['dy_m'], 1.0, places=6)

    def test_rotation_about_the_sonar_gives_nothing_under_the_conservative_rule(self):
        pitch_rows, _, _ = self.run_primitive([pose(pitch=p) for p in np.linspace(0, 9, 7)])  # positive pitch about +y is nose down
        yaw_rows, _, _ = self.run_primitive([pose(yaw=y) for y in np.linspace(0, 20, 7)])
        self.assertAlmostEqual(self.med(self.bar(pitch_rows), 'conservative_fraction_actual'), 1.0)
        self.assertAlmostEqual(self.med(self.bar(yaw_rows), 'conservative_fraction_actual'), 1.0)
        # the aggressive rule reads the bar leaving the fan as information; yaw keeps the bar in view
        self.assertLess(self.med(self.bar(pitch_rows), 'aggressive_fraction_actual'), self.med(self.bar(yaw_rows), 'aggressive_fraction_actual'))

    def test_occlusion_does_not_clear_under_the_conservative_rule(self):
        # a plate appears 1 m in front of the bar after the first view: the bar's columns now have a nearer first echo
        plate = np.column_stack([np.full(41*5, 2.5), np.tile(np.linspace(-1, 1, 41), 5), np.repeat(np.linspace(-0.4, 0.4, 5), 41)])
        ds0 = FakeDataset([pose(), pose(y=0.05)], np.vstack([self.BAR, self.POST]))
        ds1 = FakeDataset([pose(), pose(y=0.05)], np.vstack([self.POST, plate]))  # the plate hides the bar (render has no occlusion itself)
        images = {0: ds0.image(0), 1: ds1.image(1)}
        cfg = {**DEFAULTS, 'reference_t': 0.0, 'view_stride': 1, 'candidates': 41}
        rows, _, _ = analyse(ds1, lambda i: images[i], cfg)
        bar = self.bar(rows)
        self.assertAlmostEqual(self.med(bar, 'conservative_fraction_actual'), 1.0)  # occluded: untestable, kept
        self.assertLess(self.med(bar, 'aggressive_fraction_actual'), 0.5)  # the carving rule clears the occluded bar

    def test_prediction_orders_the_primitives_like_the_measurement(self):
        results = {}
        for name, poses in [('heave', [pose(z=dz) for dz in list(np.linspace(0, -0.6, 7)) + list(np.linspace(-0.4, 0.6, 6))]),
                            ('sway', [pose(y=dy) for dy in np.linspace(0, 1.0, 13)]),
                            ('surge', [pose(x=dx) for dx in np.linspace(0, 1.0, 13)])]:
            rows, motions, _ = self.run_primitive(poses)
            results[name] = summarise(rows, motions)
        for rule in ('conservative', 'aggressive'):
            self.assertLessEqual(results['heave'][rule]['fraction_predicted_median'], results['sway'][rule]['fraction_predicted_median'])
            self.assertLessEqual(results['heave'][rule]['fraction_actual_median'], results['sway'][rule]['fraction_actual_median'])
        self.assertLess(results['heave']['aggressive']['fraction_actual_median'], results['sway']['aggressive']['fraction_actual_median'])

    def test_rule_primitives(self):
        first = np.array([-1, 10, 10, -1])  # columns 1 and 2 have their first echo at row 10
        elim, test = in_front_of_first_echo(first, np.array([5, 9, 10, 12, 5]), np.array([1, 1, 2, 2, 3]), 1)
        np.testing.assert_array_equal(test, [True, True, True, True, True])  # column 3 borrows column 2's echo
        np.testing.assert_array_equal(elim, [True, False, False, False, True])
        elim0, test0 = in_front_of_first_echo(np.array([-1, -1]), np.array([5]), np.array([0]), 1)
        self.assertFalse(test0[0])
        self.assertFalse(elim0[0])
        sig = np.array([[[10, 5]], [[10, 6]], [[20, 5]]])
        testable = np.ones((3, 1), bool)
        np.testing.assert_array_equal(predicted_survivors(sig, testable, 'aggressive'), [2, 2, 1])
        np.testing.assert_array_equal(predicted_survivors(sig, testable, 'conservative'), [3, 3, 1])  # the far one survives behind the near ones
        testable[2, 0] = False
        np.testing.assert_array_equal(predicted_survivors(sig, testable, 'aggressive'), [3, 3, 1])
        self.assertEqual(runs_count([True, True, False, True]), 2)
        self.assertEqual(runs_count([False, False]), 0)
        m = relative_motion(pose(), pose(x=1, z=0.5, pitch=10))
        self.assertAlmostEqual(m['dx_m'], 1.0)
        self.assertAlmostEqual(m['dz_m'], 0.5)
        self.assertAlmostEqual(m['pitch_deg'], 10.0, places=6)


if __name__ == '__main__':
    unittest.main()
