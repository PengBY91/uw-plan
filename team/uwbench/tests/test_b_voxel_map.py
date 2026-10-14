import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from uwbench.b.voxel_map import (FREE, OCCUPIED, R_BODY_OPTICAL, UNKNOWN, VoxelGrid, predicted_labels, truth_from_boxes)
from uwbench.b.occupancy import map_ratios

K = np.array([[40.0, 0, 32], [0, 40.0, 24], [0, 0, 1]])  # 64x48, about 77 x 62 deg field of view
T_BODY_CAMERA = np.eye(4)
T_BODY_CAMERA[:3, :3] = R_BODY_OPTICAL


def pose(xyz=(0, 0, 0), pitch_up_deg=0.0):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler('y', -pitch_up_deg, degrees=True).as_matrix()
    T[:3, 3] = xyz
    return T


def render(T_world_body, boxes, shape=(48, 64)):
    """Optical-z depth of the first box hit per pixel; NaN where nothing is hit."""
    T = T_world_body @ T_BODY_CAMERA
    v, u = np.mgrid[0:shape[0], 0:shape[1]]
    rays_optical = np.stack([(u + 0.5 - K[0, 2])/K[0, 0], (v + 0.5 - K[1, 2])/K[1, 1], np.ones(shape)], axis=-1)
    directions = rays_optical @ T[:3, :3].T
    depth = np.full(shape, np.nan)
    for lo, hi in boxes:
        with np.errstate(divide='ignore', invalid='ignore'):
            t1, t2 = (np.asarray(lo) - T[:3, 3])/directions, (np.asarray(hi) - T[:3, 3])/directions
        near, far = np.nanmax(np.minimum(t1, t2), axis=-1), np.nanmin(np.maximum(t1, t2), axis=-1)
        hit = (near <= far) & (near > 0)
        depth = np.where(hit & ~(depth <= near), near, depth)  # scale is optical z because ray z component is 1
    return depth


def grid():
    return VoxelGrid(origin=(-0.5, -2.0, -1.0), shape=(40, 40, 30), resolution=0.10)


def cell(g, point):
    idx, inside = g.index(np.asarray([point]))
    assert inside[0]
    return tuple(idx[0])


WALL = ((2.0, -3.0, -2.0), (2.2, 3.0, 2.0))


class B0GeometryTest(unittest.TestCase):
    def integrate(self, g, T, boxes, valid=None, max_range=5.0, scale=1.0):
        depth = render(T, boxes)*scale
        mask = np.isfinite(depth) if valid is None else valid & np.isfinite(depth)
        g.integrate_depth(T, T_BODY_CAMERA, K, np.nan_to_num(depth), mask, stamp=1.0, max_range=max_range)
        return depth

    def test_plane_occupied_free_in_front_unknown_behind(self):
        g = grid()
        self.integrate(g, pose(), [WALL])
        s = g.state()
        self.assertEqual(s[cell(g, (2.05, 0, 0))], OCCUPIED)
        self.assertEqual(s[cell(g, (1.0, 0, 0))], FREE)
        self.assertEqual(s[cell(g, (2.8, 0, 0))], UNKNOWN)
        self.assertEqual(s[cell(g, (1.0, 0, 1.4))], UNKNOWN)  # outside the vertical field of view
        self.assertTrue(g.sources[cell(g, (2.05, 0, 0))] & 1)
        self.assertEqual(g.last_update[cell(g, (1.0, 0, 0))], 1.0)

    def test_camera_axes_follow_body_frame(self):
        g = grid()
        T = pose()
        T[:3, :3] = Rotation.from_euler('z', 90, degrees=True).as_matrix()  # body x now points to world +y
        self.integrate(g, T, [((-1, 1.5, -1), (1, 1.7, 1))])
        self.assertEqual(g.state()[cell(g, (0, 1.55, 0))], OCCUPIED)
        self.assertEqual(g.state()[cell(g, (1.5, 0, 0))], UNKNOWN)

    def test_invalid_depth_never_clears(self):
        g = grid()
        valid = np.zeros((48, 64), bool)
        valid[:, 32:] = True  # only the right half (world -y) is trusted
        self.integrate(g, pose(), [WALL], valid=valid)
        s = g.state()
        self.assertEqual(s[cell(g, (1.0, 0.4, 0))], UNKNOWN)
        self.assertEqual(s[cell(g, (1.0, -0.4, 0))], FREE)

    def test_out_of_range_depth_is_not_a_free_ray(self):
        g = grid()
        self.integrate(g, pose(), [WALL], max_range=1.5)
        self.assertFalse((g.state() == FREE).any())

    def test_millimetre_depth_is_rejected(self):
        with self.assertRaises(ValueError):
            self.integrate(grid(), pose(), [WALL], scale=1000.0)

    def test_cube_occludes_and_cavity_interior_stays_unknown(self):
        g = grid()
        shell = [((1.0, -0.3, -0.3), (1.1, 0.3, 0.3)),   # front face of a hollow box
                 ((1.6, -0.3, -0.3), (1.7, 0.3, 0.3))]   # back face
        self.integrate(g, pose(), shell)
        s = g.state()
        self.assertEqual(s[cell(g, (1.05, 0, 0))], OCCUPIED)
        self.assertEqual(s[cell(g, (1.35, 0, 0))], UNKNOWN)  # cavity interior is occluded, not free
        self.assertEqual(s[cell(g, (1.65, 0, 0))], UNKNOWN)

    def test_overhead_beam_is_unknown_until_seen_and_keeps_its_layer(self):
        g = grid()
        beam = ((1.4, -1.0, 0.9), (1.6, 1.0, 1.1))
        self.integrate(g, pose(), [beam, WALL])
        self.assertNotEqual(g.state()[cell(g, (1.5, 0, 1.0))], FREE)
        self.integrate(g, pose(pitch_up_deg=35), [beam, WALL])
        s, c = g.state(), g.centers()
        near_beam = (np.abs(c[..., 0] - 1.5) < 0.2) & (np.abs(c[..., 1]) < 0.1) & (np.abs(c[..., 2] - 1.0) < 0.2)
        self.assertTrue((s[near_beam] == OCCUPIED).any())  # surface sits on a voxel boundary, so test the band
        self.assertEqual(s[cell(g, (1.45, 0, 0.0))], FREE)  # same column, lower layer stays passable

    def test_map_ratios_on_analytic_truth(self):
        g = grid()
        thin = map_ratios(truth_from_boxes(g, [WALL], band=0.1), predicted_labels(g))
        self.assertEqual(thin['counts']['true_occ'], 0)  # band wider than half the wall erases it from scoring
        self.assertIsNone(thin['false_free_rate'])       # which is undefined, not a perfect zero
        truth = truth_from_boxes(g, [WALL], band=0.04)
        empty = map_ratios(truth, predicted_labels(g))
        self.assertEqual(empty['effective_coverage'], 0)
        self.assertIsNone(empty['free_hazard_fraction'])
        self.integrate(g, pose(), [WALL])
        r = map_ratios(truth, predicted_labels(g))
        self.assertEqual(r['false_free_rate'], 0)
        self.assertGreater(r['effective_coverage'], 0)
        self.assertEqual(r['false_occupied_rate'], 0)


if __name__ == '__main__':
    unittest.main()
