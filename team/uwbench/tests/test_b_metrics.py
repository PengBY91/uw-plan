import unittest

import numpy as np

from uwbench.b.metrics import occupied_placement, route_facing_surface, shadowed, signed_depth, surface_recall, visible_roi


class SurfaceMetricTests(unittest.TestCase):
    # a wall slab from y = 0 to 0.5, x in [-2, 2], z in [-1, 1]; the vehicle drives at y = -2.5
    BOX = [(np.array([-2.0, 0.0, -1.0]), np.array([2.0, 0.5, 1.0]))]
    Y_VEHICLE = -2.5

    def grid(self, res=0.1):
        xs = np.arange(-3, 3, res) + res/2
        ys = np.arange(-3, 2, res) + res/2
        zs = np.arange(-1.5, 1.5, res) + res/2
        return np.stack(np.meshgrid(xs, ys, zs, indexing='ij'), axis=-1).reshape(-1, 3)

    def test_signed_depth_is_positive_inside_and_minus_distance_outside(self):
        d = signed_depth(np.array([[0.0, 0.25, 0.0], [0.0, -1.0, 0.0], [0.0, 0.05, 0.0]]), self.BOX)
        self.assertAlmostEqual(d[0], 0.25)
        self.assertAlmostEqual(d[1], -1.0)
        self.assertAlmostEqual(d[2], 0.05)

    def test_route_facing_surface_is_the_near_face_band(self):
        pts = self.grid()
        s = route_facing_surface(pts, self.BOX, 0.1, self.Y_VEHICLE)
        y = pts[s, 1]
        self.assertTrue(((y > -0.11) & (y < 0.11)).all() or (y.max() < 0.3))  # the near face at y = 0, not the far one at 0.5
        self.assertFalse(s[(np.abs(pts[:, 1] - 0.5) <= 0.1) & (np.abs(pts[:, 0]) < 1.5)].any())  # far face excluded
        self.assertTrue(s[(np.abs(pts[:, 1] - 0.0) <= 0.1) & (np.abs(pts[:, 0]) < 1.5) & (np.abs(pts[:, 2]) < 0.8)].all())

    def test_shadow_and_visible_roi(self):
        pts = self.grid()
        sh = shadowed(pts, self.BOX, self.Y_VEHICLE)
        behind = (pts[:, 1] > 0.5) & (np.abs(pts[:, 0]) < 2) & (np.abs(pts[:, 2]) < 1)
        beside = (pts[:, 1] > 0.5) & (np.abs(pts[:, 0]) > 2.2)
        front = pts[:, 1] < 0
        self.assertTrue(sh[behind].all())
        self.assertFalse(sh[beside].any())
        self.assertFalse(sh[front].any())
        vis = visible_roi(pts, self.BOX, 0.1, self.Y_VEHICLE)
        inside_deep = (pts[:, 1] > 0.15) & (pts[:, 1] < 0.35) & (np.abs(pts[:, 0]) < 1.5) & (np.abs(pts[:, 2]) < 0.8)
        self.assertFalse(vis[inside_deep].any())
        self.assertTrue(vis[front].all())

    def test_placement_and_surface_recall(self):
        pts = self.grid()
        surface = route_facing_surface(pts, self.BOX, 0.1, self.Y_VEHICLE)
        d = signed_depth(pts, self.BOX)
        occ = surface.copy()
        occ[(np.abs(pts[:, 1] - 0.3) < 0.05) & (np.abs(pts[:, 0]) < 1.5) & (np.abs(pts[:, 2]) < 0.8)] = True  # plus a smear deep inside the wall
        place = occupied_placement(d, occ, 0.1)
        self.assertGreater(place['interior_beyond_band'], 0.0)
        self.assertAlmostEqual(place['outside_band'], 0.0)
        self.assertAlmostEqual(surface_recall(occ, surface), 1.0)
        half = surface.copy()
        half[pts[:, 0] > 0] = False
        self.assertAlmostEqual(surface_recall(half, surface), 0.5, delta=0.05)
        self.assertIsNone(surface_recall(occ, np.zeros(len(pts), bool)))
        self.assertEqual(occupied_placement(d, np.zeros(len(pts), bool), 0.1)['predicted_occupied'], 0)


if __name__ == '__main__':
    unittest.main()
