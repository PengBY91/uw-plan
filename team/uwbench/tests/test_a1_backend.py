"""A1 backend behaviour on a synthetic straight run with exact IMU, stereo and sonar data (needs gtsam, .venv-a)."""
import unittest

import numpy as np

try:
    import gtsam
    from uwbench.a1 import backend
except ImportError:
    gtsam = None

from uwbench.sonar import range_bearing

R_BODY_OPTICAL = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=float)
NOISE = {'accelerometer_noise_density': 1e-3, 'accelerometer_random_walk': 1e-4,
         'gyroscope_noise_density': 1e-4, 'gyroscope_random_walk': 1e-5}
K, BASE, DT, RATE = (300.0, 300.0, 320.0, 240.0), 0.2, 0.1, 200


def T(R, p):
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = R, p
    return M


class Scene:
    """Body (x forward, z up) moving at 0.3 m/s along world x with a sinusoidal 0.2 m heave, facing world +y."""
    def __init__(self, n=40):
        yaw = np.pi/2
        self.R = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        self.n = n
        self.T_bc = T(R_BODY_OPTICAL, [0.2, 0.1, 0.0])
        self.T_bs = T(np.diag([1.0, -1.0, -1.0]), [0.4, 0.0, -0.1])
        rng = np.random.default_rng(0)
        self.visual = np.column_stack([rng.uniform(-2, 14, 300), rng.uniform(3, 6, 300), rng.uniform(-2, 2, 300)])
        self.sonar = np.array([[x, 3.0, z] for x in (1.0, 3.0, 5.0, 7.0, 9.0) for z in (0.05, -0.1)])

    def p(self, t):
        return np.array([0.3*t, 0.0, 0.2*np.sin(0.8*t)])

    def v(self, t):
        return np.array([0.3, 0.0, 0.16*np.cos(0.8*t)])

    def a(self, t):
        return np.array([0.0, 0.0, -0.128*np.sin(0.8*t)])

    def imu(self, t0, t1):
        ts = np.arange(t0, t1 - 1e-9, 1/RATE)
        rows = []
        for t in ts:
            f = self.R.T @ (self.a(t + 0.5/RATE) - np.array([0, 0, -9.8]))   # midpoint specific force
            rows.append([1/RATE, *f, 0.0, 0.0, 0.0])
        return np.array(rows)

    def stereo(self, t):
        Tw = T(self.R, self.p(t)) @ self.T_bc
        pc = (self.visual - Tw[:3, 3]) @ Tw[:3, :3]
        rows = []
        for i, (x, y, z) in enumerate(pc):
            if z < 0.5:
                continue
            uL, v = K[0]*x/z + K[2], K[1]*y/z + K[3]
            uR = K[0]*(x - BASE)/z + K[2]
            if 0 <= uL < 640 and 0 <= v < 480 and 0 <= uR < 640:
                rows.append((i, uL, uR, v))
        return rows

    def sonar_obs(self, t, lids=None):
        Tw = T(self.R, self.p(t)) @ self.T_bs
        ps = (self.sonar - Tw[:3, 3]) @ Tw[:3, :3]
        rho, theta = range_bearing(ps)
        el = np.arctan2(ps[:, 2], np.hypot(ps[:, 0], ps[:, 1]))
        ok = (np.abs(theta) < np.radians(60)) & (np.abs(el) < np.radians(10)) & (rho < 10)
        return [(j, rho[j], theta[j], False) for j in np.nonzero(ok)[0] if lids is None or j in lids]


def run(scene, sonar_cfg, sonar_fn=lambda k, t, obs: obs):
    cfg = backend.merge(backend.DEFAULTS, {'sonar': sonar_cfg})
    sm = backend.Smoother(cfg, K, BASE, scene.T_bc, scene.T_bs, NOISE, np.radians(20))
    for k in range(scene.n):
        t = k*DT
        seg = scene.imu((k - 1)*DT, t) if k else np.zeros((0, 7))
        init = (T(scene.R, scene.p(0)), scene.v(0)) if k == 0 else None
        sm.add_keyframe(t, seg, scene.stereo(t), sonar_fn(k, t, scene.sonar_obs(t)), init)
    return sm


@unittest.skipIf(gtsam is None, 'gtsam not installed (use .venv-a)')
class BackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene = Scene()
        cls.off = run(cls.scene, {'enabled': False})

    def positions(self, sm):
        return np.array([P[:3, 3] for _, P in sm.poses_causal])

    def test_sonar_off_tracks_exact_data(self):
        truth = np.array([self.scene.p(k*DT) for k in range(self.scene.n)])
        self.assertLess(np.abs(self.positions(self.off) - truth).max(), 0.01)
        self.assertEqual(sum(c['sonar_factors'] for c in self.off.counts), 0)

    def test_zero_valid_sonar_factors_equals_sonar_off(self):
        sm = run(self.scene, {'enabled': True}, lambda k, t, obs: [])
        np.testing.assert_array_equal(self.positions(sm), self.positions(self.off))
        self.assertEqual(sum(c['sonar_factors'] for c in sm.counts), 0)

    def test_exact_sonar_enters_solution_with_near_zero_residual(self):
        sm = run(self.scene, {'enabled': True})
        self.assertGreater(len(sm.sonar_added), 3)
        self.assertGreater(len(sm.sonar_factors), 30)
        est = sm.isam.calculateEstimate()
        from gtsam.symbol_shorthand import S
        from uwbench.a1 import factors
        r = [factors.whitened_norm(f, est) for _, _, f, _ in sm.sonar_factors]
        self.assertLess(np.median(r), 0.1)
        err = [np.linalg.norm(est.atPoint3(S(l)) - self.scene.sonar[l]) for l in sm.sonar_added]
        self.assertLess(np.median(err), 0.05)

    def test_missing_sonar_frames_only_reduce_factor_count(self):
        full = run(self.scene, {'enabled': True})
        gaps = run(self.scene, {'enabled': True}, lambda k, t, obs: obs if k % 3 else [])
        self.assertLess(len(gaps.sonar_factors), len(full.sonar_factors))
        self.assertGreater(len(gaps.sonar_factors), 0)
        self.assertTrue(all(c['sonar_factors'] == 0 for c in gaps.counts if c['keyframe'] % 3 == 0 and c['keyframe'] > 0))

    def test_outlier_is_isolated_by_cauchy(self):
        seen = {}
        for k in range(self.scene.n):
            for j, *_ in self.scene.sonar_obs(k*DT):
                seen.setdefault(j, []).append(k)
        j_bad = max(seen, key=lambda j: len(seen[j]))
        k_bad = seen[j_bad][len(seen[j_bad])//2]          # mid-track: the landmark is initialised by then

        def corrupt(k, t, obs):
            return [(j, r + (1.5 if (j, k) == (j_bad, k_bad) else 0.0), b, (j, k) == (j_bad, k_bad)) for j, r, b, _ in obs]
        sm = run(self.scene, {'enabled': True, 'kernel': 'cauchy', 'kernel_k': 1.0}, corrupt)
        est = sm.isam.calculateEstimate()
        from uwbench.a1 import factors
        flagged = [factors.whitened_norm(f, est) for _, _, f, o in sm.sonar_factors if o]
        clean = [factors.whitened_norm(f, est) for _, _, f, o in sm.sonar_factors if not o]
        self.assertEqual(len(flagged), 1)
        self.assertGreater(flagged[0], 20)            # the 1.5 m (30 sigma) error stays in its residual
        self.assertLess(np.median(clean), 0.5)
        truth = np.array([self.scene.p(k*DT) for k in range(self.scene.n)])
        self.assertLess(np.abs(self.positions(sm) - truth).max(), 0.05)

    def test_landmark_without_baseline_stays_pending(self):
        sm = run(self.scene, {'enabled': True, 'init_min_baseline_m': 100.0})
        self.assertEqual(len(sm.sonar_added), 0)
        self.assertEqual(len(sm.sonar_factors), 0)
        self.assertGreater(len(sm.sonar_buffer), 0)
        np.testing.assert_array_equal(self.positions(sm), self.positions(self.off))

    def test_bad_kernel_rejected(self):
        with self.assertRaises(ValueError):
            run(Scene(n=2), {'enabled': True, 'kernel': 'tukey'})


if __name__ == '__main__':
    unittest.main()
