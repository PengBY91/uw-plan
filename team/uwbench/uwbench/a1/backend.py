"""A1 incremental smoother (GTSAM iSAM2) over keyframes: raw IMU preintegration, stereo reprojection and optional
sonar range-bearing factors. The same front-end tracks, keyframes and initialisation are used whether sonar is on
or off, so differences come from the sonar factors only.

Initialisation (fixed, identical for every configuration): keyframe 0 pose and velocity from ground truth with
tight priors, zero bias with a broad prior. This is a controlled-experiment choice and is written to the summary.
Output poses are causal: each keyframe's pose right after its own update. The final smoothed trajectory is also kept.
"""
import time

import gtsam
import numpy as np
from gtsam.symbol_shorthand import B, L, S, V, X

from uwbench.a1 import factors
from uwbench.a1.weights import ScalarWeight

DEFAULTS = {
    'keyframe_stride': 2,
    'vision': {'sigma_px': 1.0, 'huber_k': 1.345, 'min_observations': 2, 'max_depth_m': 20.0,
               'init_gate_px': 3.0, 'prune_gate_px': 4.0, 'prune_min_depth_m': 0.2, 'pruned_prior_sigma_m': 1.0},
    'sonar': {'enabled': False, 'sigma_range_m': 0.05, 'sigma_bearing_deg': 1.0, 'kernel': 'none', 'kernel_k': 1.345,
              'init_min_observations': 3, 'init_min_baseline_m': 0.3, 'init_max_cost': 9.0, 'init_max_buffer': 30,
              'init_max_attempts': 30, 'elevation_steps': 41, 'max_gap_s': 1.0, 'weight': {'kind': 'none'}},
    'prior': {'pose_sigma_rot_rad': 1e-3, 'pose_sigma_pos_m': 1e-3, 'velocity_sigma_mps': 1e-2,
              'bias_sigma_acc': 0.05, 'bias_sigma_gyro': 0.005},
    'gravity_mps2': 9.8,
    'isam_relinearize_threshold': 0.01,
}


def merge(base, override):
    out = dict(base)
    for k, v in (override or {}).items():
        out[k] = merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


def _pose(T):
    return gtsam.Pose3(gtsam.Rot3(T[:3, :3]), gtsam.Point3(*T[:3, 3]))


def imu_params(noise, g):
    p = gtsam.PreintegrationCombinedParams.MakeSharedU(g)
    p.setAccelerometerCovariance(np.eye(3)*noise['accelerometer_noise_density']**2)
    p.setGyroscopeCovariance(np.eye(3)*noise['gyroscope_noise_density']**2)
    p.setIntegrationCovariance(np.eye(3)*1e-8)
    p.setBiasAccCovariance(np.eye(3)*noise['accelerometer_random_walk']**2)
    p.setBiasOmegaCovariance(np.eye(3)*noise['gyroscope_random_walk']**2)
    return p


class TaggedGraph:
    """NonlinearFactorGraph plus a (kind, landmark id) tag per factor, to map iSAM2 factor indices back."""
    def __init__(self):
        self.graph, self.tags = gtsam.NonlinearFactorGraph(), []

    def add(self, factor, tag=(None, None)):
        self.graph.add(factor)
        self.tags.append(tag)


class Smoother:
    def __init__(self, cfg, K, baseline, T_body_cam, T_body_sonar, imu_noise, aperture_rad):
        self.cfg = cfg
        self.K = gtsam.Cal3_S2Stereo(K[0], K[1], 0.0, K[2], K[3], baseline)
        self.T_bc, self.T_bs = _pose(T_body_cam), _pose(T_body_sonar)
        self.imu_p = imu_params(imu_noise, cfg['gravity_mps2'])
        self.aperture = aperture_rad
        v = cfg['vision']
        self.stereo_model = gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber.Create(v['huber_k']),
                                                            gtsam.noiseModel.Isotropic.Sigma(3, v['sigma_px']))
        s = cfg['sonar']
        self.sonar_sigma = np.array([s['sigma_range_m'], np.radians(s['sigma_bearing_deg'])])
        self.sonar_model = factors.noise_model(*self.sonar_sigma, kernel=s['kernel'], k=s['kernel_k'])
        self.weighting = ScalarWeight(s['weight'])
        self._models = {}
        params = gtsam.ISAM2Params()
        params.setRelinearizeThreshold(cfg['isam_relinearize_threshold'])
        self.isam = gtsam.ISAM2(params)
        self.visual_pending, self.visual_added, self.visual_pruned = {}, set(), set()
        self.visual_factor_idx = {}   # landmark id -> iSAM2 factor indices of its stereo factors
        self.sonar_buffer, self.sonar_added, self.sonar_rejected, self.sonar_attempts = {}, set(), set(), {}
        self.sonar_factors = []   # (keyframe, track id, factor, is_injected_outlier, A2 weight)
        self.counts = []
        self.poses_causal, self.k = [], -1
        self.state = None

    def _project(self, pose, p):
        """(uL, uR, v, depth) of world point p seen from body pose."""
        pc = pose.compose(self.T_bc).transformTo(p)
        fx, fy, cx, cy, b = self.K.fx(), self.K.fy(), self.K.px(), self.K.py(), self.K.baseline()
        z = pc[2]
        if z <= 1e-6:
            return None, z
        return np.array([fx*pc[0]/z + cx, fx*(pc[0] - b)/z + cx, fy*pc[1]/z + cy]), z

    def _prune(self, k, tracks):
        """Remove landmarks whose current observation is behind/too close or off by more than prune_gate_px
        (KLT tracks sliding along stripes, mismatches). Their factors are dropped and a weak prior keeps the
        orphaned landmark determinate; it no longer touches any pose. Same rule for every configuration."""
        v = self.cfg['vision']
        pose = self.isam.calculateEstimatePose3(X(k))
        bad = []
        for tid, uL, uR, vv in tracks:
            tid = int(tid)
            if tid not in self.visual_added or tid in self.visual_pruned:
                continue
            uv, z = self._project(pose, self.isam.calculateEstimatePoint3(L(tid)))
            if uv is None or z < v['prune_min_depth_m'] or np.max(np.abs(uv - np.array([uL, uR, vv]))) > v['prune_gate_px']:
                bad.append(tid)
        if not bad:
            return 0
        graph, remove = gtsam.NonlinearFactorGraph(), []
        for tid in bad:
            remove += self.visual_factor_idx.pop(tid, [])
            graph.add(gtsam.PriorFactorPoint3(L(tid), self.isam.calculateEstimatePoint3(L(tid)),
                                              gtsam.noiseModel.Isotropic.Sigma(3, v['pruned_prior_sigma_m'])))
            self.visual_pruned.add(tid)
        self.isam.update(graph, gtsam.Values(), remove)
        return len(bad)

    def _triangulate(self, pose, uL, uR, v):
        fx, fy, cx, cy, b = self.K.fx(), self.K.fy(), self.K.px(), self.K.py(), self.K.baseline()
        d = uL - uR
        if d <= 0:
            return None
        z = fx*b/d
        if z > self.cfg['vision']['max_depth_m']:
            return None
        p_c = np.array([(uL - cx)*z/fx, (v - cy)*z/fy, z])
        return pose.compose(self.T_bc).transformFrom(p_c)

    def add_keyframe(self, t, imu_segment, tracks, sonar_obs, init=None):
        """imu_segment: rows (dt, acc(3), gyro(3)) since the previous keyframe; tracks: rows (id, uL, uR, v);
        sonar_obs: rows (landmark id, rho, theta, outlier flag); init: (T, v) for keyframe 0."""
        wall = time.perf_counter()
        self.k += 1
        k, graph, values = self.k, TaggedGraph(), gtsam.Values()
        c = {'keyframe': k, 't': t, 'imu_samples': 0, 'stereo_factors': 0, 'new_visual_landmarks': 0,
             'rejected_visual_inits': 0, 'pruned_visual_landmarks': 0,
             'sonar_factors': 0, 'new_sonar_landmarks': 0, 'sonar_buffered': 0, 'sonar_rejected': 0}
        if k == 0:
            T0, v0 = init
            pose, vel, bias = _pose(T0), np.asarray(v0, dtype=float), gtsam.imuBias.ConstantBias()
            p = self.cfg['prior']
            graph.add(gtsam.PriorFactorPose3(X(0), pose, gtsam.noiseModel.Diagonal.Sigmas(
                np.r_[[p['pose_sigma_rot_rad']]*3, [p['pose_sigma_pos_m']]*3])))
            graph.add(gtsam.PriorFactorVector(V(0), vel, gtsam.noiseModel.Isotropic.Sigma(3, p['velocity_sigma_mps'])))
            graph.add(gtsam.PriorFactorConstantBias(B(0), bias, gtsam.noiseModel.Diagonal.Sigmas(
                np.r_[[p['bias_sigma_acc']]*3, [p['bias_sigma_gyro']]*3])))
        else:
            pose_prev, vel_prev, bias = self.state
            pim = gtsam.PreintegratedCombinedMeasurements(self.imu_p, bias)
            for row in imu_segment:
                pim.integrateMeasurement(row[1:4], row[4:7], row[0])
            c['imu_samples'] = len(imu_segment)
            if not len(imu_segment):
                raise ValueError('No IMU samples between keyframes %d and %d' % (k - 1, k))
            nav = pim.predict(gtsam.NavState(pose_prev, vel_prev), bias)
            pose, vel = nav.pose(), nav.velocity()
            graph.add(gtsam.CombinedImuFactor(X(k - 1), V(k - 1), X(k), V(k), B(k - 1), B(k), pim))
        values.insert(X(k), pose)
        values.insert(V(k), vel)
        values.insert(B(k), bias)
        self._add_vision(k, pose, tracks, graph, values, c)
        if self.cfg['sonar']['enabled']:
            self._add_sonar(k, pose, sonar_obs, graph, values, c)
        result = self.isam.update(graph.graph, values)
        for (kind, lid), idx in zip(graph.tags, result.getNewFactorsIndices()):
            if kind == 'vis':
                self.visual_factor_idx.setdefault(lid, []).append(int(idx))
        c['pruned_visual_landmarks'] = self._prune(k, tracks)
        est = self.isam.calculateEstimate()
        self.state = (est.atPose3(X(k)), est.atVector(V(k)), est.atConstantBias(B(k)))
        self.poses_causal.append((t, self.state[0].matrix()))
        c['solve_s'] = time.perf_counter() - wall
        self.counts.append(c)
        return est

    def _add_vision(self, k, pose, tracks, graph, values, c):
        min_obs = self.cfg['vision']['min_observations']
        for tid, uL, uR, v in tracks:
            tid = int(tid)
            obs = (k, gtsam.StereoPoint2(uL, uR, v))
            if tid in self.visual_pruned:
                continue
            if tid in self.visual_added:
                graph.add(gtsam.GenericStereoFactor3D(obs[1], self.stereo_model, X(k), L(tid), self.K, self.T_bc), ('vis', tid))
                c['stereo_factors'] += 1
                continue
            pend = self.visual_pending.setdefault(tid, [])
            pend.append((k, uL, uR, v))
            if len(pend) < min_obs:
                continue
            p = self._triangulate(pose, uL, uR, v)
            if p is None:
                continue
            consistent = True
            for kk, a, b, vv in pend[:-1]:   # the triangulating observation reprojects exactly
                uv, z = self._project(self.isam.calculateEstimatePose3(X(kk)), p)
                if uv is None or np.max(np.abs(uv - np.array([a, b, vv]))) > self.cfg['vision']['init_gate_px']:
                    consistent = False
                    break
            if not consistent:
                pend.pop(0)
                c['rejected_visual_inits'] = c.get('rejected_visual_inits', 0) + 1
                continue
            values.insert(L(tid), p)
            for kk, a, b, vv in pend:
                graph.add(gtsam.GenericStereoFactor3D(gtsam.StereoPoint2(a, b, vv), self.stereo_model, X(kk), L(tid), self.K, self.T_bc), ('vis', tid))
                c['stereo_factors'] += 1
            self.visual_added.add(tid)
            del self.visual_pending[tid]
            c['new_visual_landmarks'] += 1
        # tracks not seen this keyframe will never continue (KLT ids are not reused)
        seen = {int(r[0]) for r in tracks}
        for tid in [t for t in self.visual_pending if t not in seen]:
            del self.visual_pending[tid]

    def _pose_estimate(self, kk, k, pose_k):
        return pose_k if kk == k else self.isam.calculateEstimatePose3(X(kk))

    def _weighted_factor(self, k, lid, z, pose, p_landmark, outlier):
        """Sonar factor with the A2 scalar weight (1 for A1); the innovation uses the current estimates only."""
        innovation = None
        if self.weighting.cfg['kind'] == 'innovation':
            e, _, _ = factors.residual(pose, self.T_bs, p_landmark, z)
            innovation = float(np.linalg.norm(e/self.sonar_sigma))
        w = self.weighting.weight(lid, innovation)
        key = round(w, 4)
        if key not in self._models:
            s = self.cfg['sonar']
            self._models[key] = factors.noise_model(*(self.sonar_sigma/np.sqrt(key)), kernel=s['kernel'], k=s['kernel_k'])
        f = factors.sonar_factor(X(k), S(lid), z, self.T_bs, self._models[key])
        self.sonar_factors.append((k, lid, f, outlier, w))
        return f

    def _add_sonar(self, k, pose, sonar_obs, graph, values, c):
        s = self.cfg['sonar']
        for lid, rho, theta, outlier in sonar_obs:
            lid = int(lid)
            z = np.array([rho, theta])
            if lid in self.sonar_rejected:
                continue
            if lid in self.sonar_added:
                p_l = self.isam.calculateEstimatePoint3(S(lid))
                graph.add(self._weighted_factor(k, lid, z, pose, p_l, bool(outlier)))
                c['sonar_factors'] += 1
                continue
            buf = self.sonar_buffer.setdefault(lid, [])
            buf.append((k, z, bool(outlier)))
            if len(buf) < s['init_min_observations']:
                c['sonar_buffered'] += 1
                continue
            poses = [self._pose_estimate(kk, k, pose) for kk, _, _ in buf]
            spread = max(np.linalg.norm(a.translation() - poses[0].translation()) for a in poses)
            if spread < s['init_min_baseline_m']:
                c['sonar_buffered'] += 1
                continue
            res = factors.elevation_search(poses[0], self.T_bs, buf[0][1], list(zip(poses[1:], [b[1] for b in buf[1:]])),
                                           self.aperture, self.sonar_sigma, s['elevation_steps'])
            if res is None or res[2] > s['init_max_cost']:
                self.sonar_attempts[lid] = self.sonar_attempts.get(lid, 0) + 1
                if self.sonar_attempts[lid] >= s['init_max_attempts']:
                    self.sonar_rejected.add(lid)
                    del self.sonar_buffer[lid]
                    c['sonar_rejected'] += 1
                else:
                    if len(buf) > s['init_max_buffer']:
                        buf.pop(0)
                    c['sonar_buffered'] += 1
                continue
            values.insert(S(lid), res[0])
            for (kk, zz, out), T in zip(buf, poses):
                graph.add(self._weighted_factor(kk, lid, zz, T, res[0], out))
                c['sonar_factors'] += 1
            self.sonar_added.add(lid)
            del self.sonar_buffer[lid]
            c['new_sonar_landmarks'] += 1
