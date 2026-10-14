"""diagnose-a (M1): landmark elimination, normalisation, axis information, read-only hook and E-A1 scoring."""
import unittest

import numpy as np

try:
    import gtsam
    from gtsam.symbol_shorthand import L, X
    from uwbench.a.backend import diagnose
    from test_a_smoother import Scene, run
except ImportError:
    gtsam = None


def rotz(a):
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


@unittest.skipIf(gtsam is None, 'gtsam not installed')
class EliminationTests(unittest.TestCase):
    def test_information_explained_by_the_landmark_vanishes(self):
        rng = np.random.default_rng(0)
        An = rng.standard_normal((9, 3))
        Ax = An @ rng.standard_normal((3, 6))   # every row's pose dependence can be absorbed by moving the landmark
        np.testing.assert_allclose(diagnose.eliminate(Ax, An), 0, atol=1e-10)

    def test_without_landmark_columns_it_is_plain_normal_matrix(self):
        Ax = np.random.default_rng(1).standard_normal((5, 6))
        np.testing.assert_allclose(diagnose.eliminate(Ax, np.zeros((5, 0))), Ax.T @ Ax)

    def test_rank_deficient_landmark_projects_only_its_determined_subspace(self):
        An = np.zeros((4, 3))
        An[:, 0] = 1.0                           # only the landmark's first coordinate is observed
        Ax = np.zeros((4, 6))
        Ax[:, 0], Ax[:, 1] = 1.0, [1, -1, 1, -1]
        H = diagnose.eliminate(Ax, An)
        self.assertAlmostEqual(H[0, 0], 0.0)     # aliased with the landmark coordinate: removed
        self.assertAlmostEqual(H[1, 1], 4.0)     # orthogonal to it: kept in full

    def test_matches_gtsam_partial_elimination(self):
        """The per-landmark Schur complement equals GTSAM eliminating the landmark from the same linearised rows."""
        K = gtsam.Cal3_S2Stereo(300, 300, 0, 320, 240, 0.2)
        model = gtsam.noiseModel.Isotropic.Sigma(3, 1.0)
        poses = [gtsam.Pose3(gtsam.Rot3.Yaw(0.0), gtsam.Point3(0, 0, 0)), gtsam.Pose3(gtsam.Rot3.Yaw(0.05), gtsam.Point3(0.3, 0.1, 0))]
        p = gtsam.Point3(0.4, -0.3, 4.0)
        values = gtsam.Values()
        values.insert(X(0), poses[0])
        values.insert(X(1), poses[1])
        values.insert(L(7), p)
        graph = gtsam.NonlinearFactorGraph()
        for k, pose in enumerate(poses):
            z = gtsam.StereoCamera(pose, K).project(p)
            graph.add(gtsam.GenericStereoFactor3D(gtsam.StereoPoint2(z.uL() + 0.3, z.uR() - 0.2, z.v() + 0.1), model, X(k), L(7), K))
        Ax, An = [], []
        for i in range(graph.size()):
            A, _ = graph.at(i).linearize(values).jacobian()
            keys = list(graph.at(i).keys())
            Ax.append(A[:, :6] if keys[0] == X(1) else np.zeros((3, 6)))
            An.append(A[:, 6:9])
        ours = diagnose.eliminate(np.vstack(Ax), np.vstack(An))
        lin = graph.linearize(values)
        bn, rest = lin.eliminatePartialSequential(gtsam.Ordering([L(7)]))
        ordering = gtsam.Ordering([X(0), X(1)])
        info = rest.hessian(ordering)[0] if isinstance(rest.hessian(ordering), tuple) else rest.hessian(ordering)
        np.testing.assert_allclose(ours, info[6:12, 6:12], rtol=1e-6, atol=1e-9)


@unittest.skipIf(gtsam is None, 'gtsam not installed')
class AxisTests(unittest.TestCase):
    def test_length_scales_translation_information_by_L_squared(self):
        H = np.diag([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        for length in (1.0, 2.0):
            Hbar = diagnose.scaling(length) @ H @ diagnose.scaling(length)
            np.testing.assert_allclose(diagnose.axis_information(Hbar, np.eye(3))[:3], np.array([1, 2, 3])*length**2)

    def test_yaw_is_world_z_in_the_body_frame(self):
        H = np.zeros((6, 6))
        H[5, 5] = 7.0                            # information on body-z rotation only
        self.assertAlmostEqual(diagnose.axis_information(H, np.eye(3))[3], 7.0)
        R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)   # body z is now horizontal
        self.assertAlmostEqual(diagnose.axis_information(H, R)[3], 0.0)

    def test_correspondence_finds_error_in_the_weak_direction(self):
        H = np.diag([100.0, 0.01, 100.0, 100.0, 100.0, 100.0])         # body y barely constrained
        T = [np.eye(4) for _ in range(11)]
        ref = [np.eye(4) for _ in range(11)]
        T[10] = np.eye(4)
        T[10][1, 3] = 0.3                                               # the estimate drifted along y
        rows = diagnose.correspondence([H]*10, [np.eye(3)]*10, T, ref, 2.0, 10)
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]['norm_cos_weak'], 1.0, places=6)
        self.assertAlmostEqual(rows[0]['norm_frac_weak2'], 1.0, places=6)
        self.assertAlmostEqual(rows[0]['norm_cos_strong'], 0.0, places=6)


@unittest.skipIf(gtsam is None, 'gtsam not installed')
class HookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene = Scene(n=25)
        cls.plain = run(cls.scene, {'enabled': True})
        cls.diag = diagnose.Diagnoser(2.0)
        cls.hooked = run_hooked(cls.scene, cls.diag)

    def test_hook_does_not_change_the_estimate(self):
        for (t0, P0), (t1, P1) in zip(self.plain.poses_causal, self.hooked.poses_causal):
            self.assertEqual(t0, t1)
            np.testing.assert_array_equal(P0, P1)

    def test_one_row_per_pair_and_modality_with_finite_information(self):
        self.assertEqual(len(self.diag.H), self.scene.n - 1)
        self.assertEqual(len(self.diag.rows), 3*(self.scene.n - 1))
        stereo = [r for r in self.diag.rows if r['modality'] == 'stereo']
        self.assertTrue(all(r['landmarks'] > 0 for r in stereo))
        self.assertTrue(all(r[f'lambda_{a}'] > 0 for r in stereo for a in diagnose.AXES))
        for H in self.diag.H:
            self.assertGreaterEqual(np.linalg.eigvalsh((H + H.T)/2).min(), -1e-6*np.abs(H).max())

    def test_total_is_sum_of_modalities(self):
        by = {}
        for r in self.diag.rows:
            by.setdefault(r['keyframe'], {})[r['modality']] = r
        for rows in by.values():
            for a in diagnose.AXES:
                self.assertAlmostEqual(rows['total'][f'lambda_{a}'], rows['stereo'][f'lambda_{a}'] + rows['sonar'][f'lambda_{a}'],
                                       delta=1e-6*max(1.0, abs(rows['total'][f'lambda_{a}'])))


def run_hooked(scene, diag):
    from uwbench.a.backend import smoother as backend
    from test_a_smoother import BASE, DT, K, NOISE, T
    cfg = backend.merge(backend.DEFAULTS, {'sonar': {'enabled': True}})
    sm = backend.Smoother(cfg, K, BASE, scene.T_bc, scene.T_bs, NOISE, np.radians(20))
    for k in range(scene.n):
        t = k*DT
        seg = scene.imu((k - 1)*DT, t) if k else np.zeros((0, 7))
        init = (T(scene.R, scene.p(0)), scene.v(0)) if k == 0 else None
        tracks = scene.stereo(t)
        sm.add_keyframe(t, seg, tracks, scene.sonar_obs(t), init)
        diag(sm, k, t, tracks)
    return sm


if __name__ == '__main__':
    unittest.main()
