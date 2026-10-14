"""B0 depth and B1/B2a/B2c sonar update semantics on synthetic inputs."""
import unittest

import numpy as np

from uwbench.mapping import FREE, OCCUPIED, UNKNOWN, VoxelGrid
from uwbench.b.sonar_map import SonarUpdater, cell_geometry

try:
    import cv2
    from uwbench.b.depth import depth_from_pair, render_box_depth
except ImportError:  # .venv-eval has no OpenCV
    cv2 = None

GEOM = {'RangeMin': 0.3, 'RangeMax': 10.0, 'RangeBins': 256, 'AzimuthBins': 128, 'Azimuth': 120, 'Elevation': 20}


def grid():
    # sonar at the origin looking along +x; 6 m x 4 m x 3 m
    return VoxelGrid(origin=(-0.5, -2.0, -1.5), shape=(60, 40, 30), resolution=0.10)


def echo_image(range_m, columns):
    img = np.zeros((GEOM['RangeBins'], GEOM['AzimuthBins']), np.float32)
    row = int((range_m - GEOM['RangeMin'])/((GEOM['RangeMax'] - GEOM['RangeMin'])/GEOM['RangeBins']))
    img[row, columns] = 1.0
    return img, row


def state_at(g, p):
    idx, inside = g.index(np.asarray([p]))
    assert inside[0]
    return g.state()[tuple(idx[0])]


class SonarUpdateTests(unittest.TestCase):
    def test_b1_puts_echo_at_zero_elevation_only(self):
        g = grid()
        SonarUpdater(g, GEOM, np.radians(20), 0.5, 'b1').update(echo_image(3.0, [64])[0], np.eye(4), 0.0)
        self.assertEqual(state_at(g, (3.0, 0.0, 0.0)), OCCUPIED)
        self.assertEqual(state_at(g, (3.0, 0.0, 0.4)), UNKNOWN)

    def test_b2a_marks_whole_arc_and_never_clears(self):
        g = grid()
        u = SonarUpdater(g, GEOM, np.radians(20), 0.5, 'b2a')
        u.update(echo_image(3.0, [64])[0], np.eye(4), 0.0)
        half = 3.0*np.sin(np.radians(9.5))
        for z in (-half, 0.0, half):
            self.assertEqual(state_at(g, (3.0*np.cos(np.arcsin(z/3.0)), 0.0, z)), OCCUPIED)
        self.assertEqual(state_at(g, (3.0, 0.0, 1.0)), UNKNOWN)       # outside the 20 deg fan
        self.assertEqual(state_at(g, (1.5, 0.0, 0.0)), UNKNOWN)       # in front of the echo: not cleared
        before = g.log_odds.copy()
        u.update(np.zeros((256, 128), np.float32), np.eye(4), 1.0)    # no echo at all
        np.testing.assert_array_equal(g.log_odds, before)
        self.assertEqual(u.stats['free_voxel_updates'], 0)

    def test_veto_blocks_stereo_free_through_sonar_candidates_only(self):
        """V: a stereo ray through a voxel the sonar arc marked keeps that voxel from turning free; the
        stereo hit itself is never vetoed (V off against the stored B2a maps is checked on the sim runs)."""
        img, _ = echo_image(3.0, [64])
        ray_end = np.array([[5.0, 0.0, 0.0]])            # stereo surface behind the sonar echo, ray along +x
        maps = {}
        for veto in (False, True):
            g = grid()
            if veto:
                g.veto = np.zeros(g.shape, bool)
            SonarUpdater(g, GEOM, np.radians(20), 0.5, 'b2a').update(img, np.eye(4), 0.0)
            for k in range(3):                            # three stereo misses (-0.4 each) outweigh one sonar hit (+0.85)
                g.integrate_rays(np.zeros(3), ray_end, 1.0 + k)
            maps[veto] = g
        self.assertEqual(state_at(maps[False], (3.0, 0.0, 0.0)), FREE)       # B2a: stereo clears the sonar arc
        self.assertEqual(state_at(maps[True], (3.0, 0.0, 0.0)), OCCUPIED)
        self.assertEqual(state_at(maps[True], (1.5, 0.0, 0.0)), FREE)        # off the arc: stereo free as before
        self.assertEqual(state_at(maps[True], (5.0, 0.0, 0.0)), OCCUPIED)
        self.assertGreater(maps[True].vetoed_free_updates, 0)

    def test_b2c_region_free_front_occluded_back(self):
        g = grid()
        u = SonarUpdater(g, GEOM, np.radians(20), 0.5, 'b2c')
        img, _ = echo_image(3.0, list(range(56, 72)))
        u.update(img, np.eye(4), 0.0)
        self.assertEqual(state_at(g, (3.0, 0.0, 0.3)), OCCUPIED)       # whole reflection region is a candidate
        self.assertEqual(state_at(g, (1.5, 0.0, 0.0)), FREE)           # in front of the first echo, inside the beam
        self.assertEqual(state_at(g, (4.5, 0.0, 0.0)), UNKNOWN)        # behind: occluded, never updated
        idx, _ = g.index(np.array([[4.5, 0.0, 0.0]]))
        self.assertGreater(u.occluded[tuple(idx[0])], 0)
        self.assertEqual(state_at(g, (1.5, 1.8, 0.0)), UNKNOWN)        # columns without echo are not cleared

    def test_aperture_must_be_given(self):
        with self.assertRaises(ValueError):
            SonarUpdater(grid(), GEOM, 0.0, 0.5, 'b2a')
        with self.assertRaises(ValueError):
            SonarUpdater(grid(), GEOM, np.radians(20), 0.5, 'b9')

    def test_cell_geometry_layout(self):
        rho, theta, dr, da = cell_geometry(GEOM, (256, 128))
        self.assertAlmostEqual(float(rho(0)), 0.3 + dr/2)
        self.assertAlmostEqual(float(theta(0)), -np.radians(60) + da/2)   # column 0 is the right edge (negative bearing)


@unittest.skipIf(cv2 is None, 'OpenCV not installed (use .venv-a)')
class DepthTests(unittest.TestCase):
    def test_fronto_parallel_plane_depth_and_masks(self):
        rng = np.random.default_rng(0)
        h, w, f, b, z = 120, 200, 100.0, 0.2, 2.0
        texture = (rng.random((h, w + 40))*255).astype(np.uint8)
        texture = cv2.GaussianBlur(texture, (3, 3), 0)
        d = int(f*b/z)                                               # 10 px disparity
        left = texture[:, :w].copy()                                 # a point at left x appears at x - d on the right
        right = texture[:, d:d + w].copy()
        depth, valid, masks, _ = depth_from_pair(left, right, f, b, {'num_disparities': 32})
        inner = valid[:, 40:-10]
        self.assertGreater(inner.mean(), 0.8)
        np.testing.assert_allclose(np.nanmedian(depth[:, 40:-10]), z, rtol=0.01)
        self.assertTrue(np.isnan(depth[~valid]).all())               # invalid pixels never carry a depth

    def test_textureless_region_is_invalid(self):
        flat = np.full((60, 120), 128, np.uint8)
        depth, valid, masks, _ = depth_from_pair(flat, flat, 100.0, 0.2, {'num_disparities': 32})
        self.assertFalse(valid.any())

    def test_render_box_depth(self):
        T = np.eye(4)
        T[:3, :3] = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=float)   # optical z along world +x
        d = render_box_depth(T, (50.0, 50.0, 20.0, 15.0), [((3.0, -5, -5), (4.0, 5, 5))], (30, 40))
        np.testing.assert_allclose(d, 3.0, atol=1e-9)


if __name__ == '__main__':
    unittest.main()
