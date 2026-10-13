"""A1 sonar factor checks (W3-02): truth residual, perturbation direction, Jacobians, units, kernels, elevation search.

Needs gtsam (run with .venv-a); skipped in .venv-eval.
"""
import unittest

import numpy as np

try:
    import gtsam
    from uwbench.a1 import factors
except ImportError:  # .venv-eval has no gtsam
    gtsam = None

from uwbench.sonar import range_bearing, range_bearing_jacobian, wrap_angle

# Tolerances (declared by magnitude): residuals of exact data are ~1e-15 m/rad, so 1e-9 is "zero"; forward
# differences with h = 1e-6 on O(1) m geometry carry ~1e-6 truncation error, so Jacobians agree to 1e-5.
ZERO, JAC_TOL, H = 1e-9, 1e-5, 1e-6


def pose(yaw=0.0, pitch=0.0, roll=0.0, t=(0, 0, 0)):
    return gtsam.Pose3(gtsam.Rot3.Ypr(yaw, pitch, roll), gtsam.Point3(*t))


@unittest.skipIf(gtsam is None, 'gtsam not installed (use .venv-a)')
class SonarFactorTests(unittest.TestCase):
    def setUp(self):
        self.T_bs = gtsam.Pose3(np.array([[1, 0, 0, 0.517], [0, -1, 0, 0.15], [0, 0, -1, 0.11], [0, 0, 0, 1.0]]))
        self.X = pose(0.4, 0.05, -0.03, (1.0, -2.0, -20.0))
        # a world point 3 m ahead of the sonar, 0.3 m off-axis and 0.2 m out of the horizontal plane (inside 20 deg)
        self.p_s_true = np.array([3.0, 0.3, 0.2])
        self.p_w = self.X.compose(self.T_bs).transformFrom(self.p_s_true)
        self.z = np.array(range_bearing(self.p_s_true))

    def test_truth_residual_is_zero(self):
        e, _, _ = factors.residual(self.X, self.T_bs, self.p_w, self.z)
        self.assertLess(np.abs(e).max(), ZERO)

    def test_elevation_is_not_assumed_zero(self):
        # the same (rho, theta) placed at elevation 0 would be a different point; the factor must use the true one
        flat = self.X.compose(self.T_bs).transformFrom(np.array([self.z[0]*np.cos(self.z[1]), self.z[0]*np.sin(self.z[1]), 0.0]))
        e, _, _ = factors.residual(self.X, self.T_bs, flat, self.z)
        self.assertLess(np.abs(e).max(), ZERO)  # both lie on the measured arc: rho/theta cannot tell them apart
        self.assertGreater(np.linalg.norm(flat - self.p_w), 0.15)

    def test_translation_toward_target_shortens_range(self):
        forward = self.X.compose(self.T_bs).rotation().matrix()[:, 0]      # sonar x axis in world
        moved = gtsam.Pose3(self.X.rotation(), self.X.translation() + 0.1*forward)
        e, _, _ = factors.residual(moved, self.T_bs, self.p_w, self.z)
        self.assertLess(e[0], -0.09)
        self.assertLess(abs(e[1]) , 0.02)

    def test_rotation_about_sonar_z_shifts_bearing(self):
        d = np.radians(2.0)
        moved = self.X.compose(self.T_bs).compose(pose(d)).compose(self.T_bs.inverse())
        e, _, _ = factors.residual(moved, self.T_bs, self.p_w, self.z)
        self.assertAlmostEqual(e[1], -d, places=9)      # target appears d further to the right
        self.assertLess(abs(e[0]), ZERO)                  # pure rotation about the sonar origin keeps range

    def test_units_metres_and_radians(self):
        z = self.z + np.array([0.1, np.radians(1.0)])
        e, _, _ = factors.residual(self.X, self.T_bs, self.p_w, z)
        np.testing.assert_allclose(e, [-0.1, -np.radians(1.0)], atol=1e-12)

    def test_bearing_residual_wraps(self):
        p_s = np.array([-3.0, 1e-3, 0.0])           # behind, bearing ~ +pi
        p_w = self.X.compose(self.T_bs).transformFrom(p_s)
        e, _, _ = factors.residual(self.X, self.T_bs, p_w, [3.0, -np.pi + 1e-3])
        self.assertLess(abs(e[1]), 2e-3)

    def test_analytic_jacobians_match_numeric(self):
        _, Hx, Hp = factors.residual(self.X, self.T_bs, self.p_w, self.z)
        e0 = factors.residual(self.X, self.T_bs, self.p_w, self.z)[0]
        Nx = np.column_stack([(factors.residual(self.X.retract(H*np.eye(6)[i]), self.T_bs, self.p_w, self.z)[0] - e0)/H for i in range(6)])
        Np = np.column_stack([(factors.residual(self.X, self.T_bs, self.p_w + H*np.eye(3)[i], self.z)[0] - e0)/H for i in range(3)])
        np.testing.assert_allclose(Hx, Nx, atol=JAC_TOL)
        np.testing.assert_allclose(Hp, Np, atol=JAC_TOL)

    def test_factor_linearisation_uses_the_jacobians(self):
        model = factors.noise_model(0.05, np.radians(1.0))
        f = factors.sonar_factor(1, 2, self.z + [0.02, 0.01], self.T_bs, model)
        v = gtsam.Values()
        v.insert(1, self.X)
        v.insert(2, self.p_w)
        A, b = f.linearize(v).jacobian()
        _, Hx, Hp = factors.residual(self.X, self.T_bs, self.p_w, self.z)
        W = np.diag([1/0.05, 1/np.radians(1.0)])
        np.testing.assert_allclose(A, W @ np.hstack([Hx, Hp]), atol=1e-8)
        e = factors.residual(self.X, self.T_bs, self.p_w, self.z + [0.02, 0.01])[0]
        np.testing.assert_allclose(b, -W @ e, atol=1e-9)

    def test_range_bearing_jacobian_on_axis_raises(self):
        with self.assertRaises(ValueError):
            range_bearing_jacobian([0.0, 0.0, 1.0])

    def test_robust_kernels_downweight_outliers(self):
        e_big = np.array([1.0, 0.0])                    # 20 sigma range outlier
        base = factors.noise_model(0.05, np.radians(1.0))
        f = lambda m: factors.sonar_factor(1, 2, self.z + e_big, self.T_bs, m)
        v = gtsam.Values()
        v.insert(1, self.X)
        v.insert(2, self.p_w)
        gauss = f(base).error(v)
        huber = f(factors.noise_model(0.05, np.radians(1.0), 'huber', 1.345)).error(v)
        cauchy = f(factors.noise_model(0.05, np.radians(1.0), 'cauchy', 1.0)).error(v)
        self.assertAlmostEqual(gauss, 0.5*20**2, places=6)
        self.assertLess(huber, gauss/5)      # k|e| - k^2/2 = 26 vs 200 at 20 sigma
        self.assertLess(cauchy, huber)
        self.assertAlmostEqual(factors.whitened_norm(f(factors.noise_model(0.05, np.radians(1.0), 'cauchy', 1.0)), v), 20.0, places=6)
        with self.assertRaises(ValueError):
            factors.noise_model(0.05, 0.01, 'tukey')
        with self.assertRaises(ValueError):
            factors.noise_model(0.0, 0.01)

    def test_elevation_search_recovers_height_with_vertical_motion(self):
        poses = [pose(0.4, 0, 0, (1.0 + 0.2*i, -2.0, -20.0 + 0.15*i)) for i in range(6)]
        zs = [np.array(range_bearing(T.compose(self.T_bs).transformTo(self.p_w))) for T in poses]
        p, phi, cost = factors.elevation_search(poses[0], self.T_bs, zs[0], list(zip(poses[1:], zs[1:])),
                                                np.radians(20), np.array([0.05, np.radians(1)]), steps=201)
        self.assertLess(np.linalg.norm(p - self.p_w), 0.05)
        self.assertLess(cost, 0.1)

    def test_elevation_search_without_support_returns_none(self):
        self.assertIsNone(factors.elevation_search(self.X, self.T_bs, self.z, [], np.radians(20), np.array([0.05, 0.02])))
        with self.assertRaises(ValueError):
            factors.elevation_search(self.X, self.T_bs, self.z, [(self.X, self.z)], 0.0, np.array([0.05, 0.02]))

    def test_wrap_angle(self):
        np.testing.assert_allclose(wrap_angle([np.pi + 0.1, -np.pi - 0.1, 0.3]), [-np.pi + 0.1, np.pi - 0.1, 0.3])


if __name__ == '__main__':
    unittest.main()
