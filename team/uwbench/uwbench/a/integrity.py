"""Online input-integrity monitor (method spec M3.4, product output; runs in front of the estimator).

Everything here is causal: each decision uses only messages that have already arrived, and nothing reads reference
truth or estimator state (the time-offset estimate is independent of the estimator's own online calibration, which is
exactly what fails silently in the W4 fault injection, reports/w4/fault-injection.md).

Checks (tolerances from INTEGRITY, overridable from the config; to be re-verified on platform data before freezing).
The offset thresholds are 20 / 30 ms (method spec v2.5): between the largest harmless (15 ms) and the smallest
divergent (40 ms) W4-05 offset with margin on both sides; spec v2.4 had 15 / 40 ms, the test faults themselves.
  imu/cam time regression  a sample stamp not after the previous one on the same stream       -> LOST
  imu outage               no IMU sample for >= imu_outage_s                                  -> LOST
  cam outage               no stereo pair for >= cam_outage_s                                 -> LOST
  imu rate                 IMU samples in the last rate_window_s below rate_min_fraction of nominal -> WARN
  time offset              |IMU-minus-camera offset| > warn_ms -> WARN, > lost_ms for lost_hold estimates -> LOST
An outage is seen online in two ways: the next sample's stamp gap on that stream, or the other stream's clock running
past the last sample by the tolerance (the watchdog, so a stream that stops for good is still caught). LOST latches
until reset() (estimator re-initialisation, new epoch); WARN clears with its cause.

Time offset (IMU stamp = true time + delta): the camera's rotation over a few frames comes from stereo PnP (triangulate at
a keyframe, track to the frame `stride` pairs later), so it is not contaminated by translation as a monocular estimate would be at AUV
speeds. For each candidate delta the gyro is integrated over [t_{k-1} + delta, t_k + delta] in IMU stamp time and
compared with the visual rotation in the IMU frame, with a constant gyro bias fitted per candidate (Huber IRLS). The
minimum is refined by a parabola (se_ms from the cost curvature is reported for information only: PnP residuals are
autocorrelated, and on the checkerboard sim control it claimed 1.8 ms for a basin 40 ms wide). The decision uses the
near-optimal set instead: every candidate whose cost is within offset_cost_margin of the minimum ([ci_lo, ci_hi]); an
offset alarm needs that whole set beyond the threshold. A window is only used with enough rotational excitation and a
gyro that explains at least offset_min_fit_r2 of the visual rotation variance at the best offset; otherwise it is
'unobservable' (no alarm), so low-rotation or visually degenerate segments never raise offset alarms but also cannot
clear the check.
"""
import numpy as np

INTEGRITY = {'imu_outage_s': 0.5, 'cam_outage_s': 3.0, 'rate_window_s': 1.0, 'rate_min_fraction': 0.8,
             'imu_rate_hz': None, 'time_offset_warn_ms': 20.0, 'time_offset_lost_ms': 30.0, 'lost_hold': 2,
             'offset_window_s': 20.0, 'offset_step_s': 5.0, 'offset_search_ms': 100.0, 'offset_grid_ms': 1.0,
             'offset_min_rot_rad_s': 0.03, 'offset_min_fit_r2': 0.8, 'offset_cost_margin': 0.1}
STATES = ('NORMAL', 'WARN', 'LOST')


def rotvec(R):
    """Rotation matrix -> rotation vector (axis * angle)."""
    c = np.clip((np.trace(R) - 1)/2, -1.0, 1.0)
    angle = np.arccos(c)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if angle < 1e-8:
        return w/2
    return w*angle/(2*np.sin(angle))


def gyro_integral(t, gyro):
    """Cumulative trapezoid of the gyro on its stamps: returns (t, C) with C(t_i) = integral up to t_i."""
    t = np.asarray(t, float)
    g = np.asarray(gyro, float)
    C = np.zeros_like(g)
    C[1:] = np.cumsum((g[1:] + g[:-1])/2*np.diff(t)[:, None], axis=0)
    return t, C


def estimate_offset(intervals, theta_imu, imu_t, gyro, cfg=None):
    """IMU-minus-camera offset from visual rotations. intervals (K, 2) camera stamps [t_{k-1}, t_k]; theta_imu (K, 3)
    visual rotation vectors over each interval, already in the IMU frame; imu_t, gyro: IMU stamps and rates.
    Returns dict(offset_ms, se_ms, excitation_rad_s, n, observable, reason, curve) for one window."""
    c = {**INTEGRITY, **(cfg or {})}
    iv = np.asarray(intervals, float)
    th = np.asarray(theta_imu, float)
    dt = iv[:, 1] - iv[:, 0]
    out = {'offset_ms': np.nan, 'se_ms': np.nan, 'n': int(len(iv)), 'observable': False}
    rate = th/dt[:, None] if len(iv) else np.zeros((0, 3))
    out['excitation_rad_s'] = float(np.sqrt(np.mean(np.sum((rate - rate.mean(0))**2, 1)))) if len(iv) > 2 else 0.0
    if len(iv) < 10:
        return {**out, 'reason': 'too_few_intervals'}
    if out['excitation_rad_s'] < c['offset_min_rot_rad_s']:
        return {**out, 'reason': 'low_rotation'}
    t, C = gyro_integral(imu_t, gyro)
    lim = c['offset_search_ms']/1000
    if iv[0, 0] - lim < t[0] or iv[-1, 1] + lim > t[-1]:
        return {**out, 'reason': 'imu_does_not_cover_window'}
    grid = np.arange(-lim, lim + 1e-12, c['offset_grid_ms']/1000)
    cost = np.empty(len(grid))
    for j, d in enumerate(grid):
        G = np.stack([np.interp(iv[:, 1] + d, t, C[:, a]) - np.interp(iv[:, 0] + d, t, C[:, a]) for a in range(3)], 1)
        w = np.ones(len(iv))
        for _ in range(3):  # Huber IRLS on the per-interval residual norm, bias refitted each time
            b = ((w*dt)[:, None]*(G - th)).sum(0)/np.sum(w*dt**2)
            r = np.linalg.norm(th - G + b*dt[:, None], axis=1)
            k = 1.345*max(np.median(r)/0.6745, 1e-9)
            w = np.where(r <= k, 1.0, k/np.maximum(r, 1e-12))
        cost[j] = np.sum(np.where(r <= k, r**2/2, k*r - k**2/2))
    out['curve'] = {'grid_ms': (grid*1000).tolist(), 'cost': cost.tolist()}
    j = int(np.argmin(cost))
    if j == 0 or j == len(grid) - 1:
        return {**out, 'reason': 'minimum_at_search_bound'}
    h = grid[1] - grid[0]
    y0, y1, y2 = cost[j - 1], cost[j], cost[j + 1]
    curv = (y0 - 2*y1 + y2)/h**2
    if curv <= 0:
        return {**out, 'reason': 'flat_cost'}
    est = grid[j] + h*(y0 - y2)/(2*(y0 - 2*y1 + y2))
    dof = max(3*len(iv) - 4, 1)
    se = np.sqrt(2*(2*y1/dof)/curv)   # Huber cost ~ r^2/2 near the optimum: noise variance 2*cost/dof
    G = np.stack([np.interp(iv[:, 1] + grid[j], t, C[:, a]) - np.interp(iv[:, 0] + grid[j], t, C[:, a]) for a in range(3)], 1)
    b = (dt[:, None]*(G - th)).sum(0)/np.sum(dt**2)
    resid = th - G + b*dt[:, None]
    centred = th - dt[:, None]*(th.sum(0)/dt.sum())
    fit = 1 - np.sum(resid**2)/max(np.sum(centred**2), 1e-18)
    near = grid[cost <= cost[j]*(1 + c['offset_cost_margin'])]
    out.update({'offset_ms': float(est*1000), 'se_ms': float(se*1000), 'fit_r2': float(fit),
                'ci_lo_ms': float(near.min()*1000), 'ci_hi_ms': float(near.max()*1000)})
    if fit < c['offset_min_fit_r2']:
        return {**out, 'reason': 'visual_rotation_disagrees_with_gyro'}
    return {**out, 'observable': True, 'reason': 'ok'}


class StereoRotation:
    """Camera rotation over `stride` stereo pairs from stereo PnP (causal). calib: dict with K0, D0, K1, D1 (3x3,
    4 radtan), T_imu_cam0, T_imu_cam1 (4x4). A stride of several frames raises the rotation signal against the
    per-solve PnP noise, which at AUV rotation rates dominates frame-to-frame rotations (stripe-beam sim control)."""

    def __init__(self, calib, stride=4, max_features=300, min_inliers=25):
        import cv2
        self.stride = stride
        self.cv2 = cv2
        self.c = calib
        self.max_features, self.min_inliers = max_features, min_inliers
        T01 = np.linalg.inv(calib['T_imu_cam1']) @ calib['T_imu_cam0']   # cam0 -> cam1
        self.P0 = np.hstack([np.eye(3), np.zeros((3, 1))])
        self.P1 = np.hstack([T01[:3, :3], T01[:3, 3:]])
        self.R_ic = calib['T_imu_cam0'][:3, :3]
        self.prev = None

    def _norm(self, pts, K, D):
        return self.cv2.undistortPoints(pts.reshape(-1, 1, 2), K, D).reshape(-1, 2)

    def _landmarks(self, left, right):
        cv2 = self.cv2
        p0 = cv2.goodFeaturesToTrack(left, self.max_features, 0.01, 8)
        if p0 is None or len(p0) < self.min_inliers:
            return None
        p0 = p0.reshape(-1, 2).astype(np.float32)
        p1, st, _ = cv2.calcOpticalFlowPyrLK(left, right, p0, None, winSize=(21, 21), maxLevel=3)
        back, st2, _ = cv2.calcOpticalFlowPyrLK(right, left, p1, None, winSize=(21, 21), maxLevel=3)
        ok = (st.ravel() == 1) & (st2.ravel() == 1) & (np.linalg.norm(back - p0, axis=1) < 1.0)
        if ok.sum() < self.min_inliers:
            return None
        n0 = self._norm(p0[ok], self.c['K0'], self.c['D0'])
        n1 = self._norm(p1[ok], self.c['K1'], self.c['D1'])
        X = cv2.triangulatePoints(self.P0, self.P1, n0.T.astype(np.float64), n1.T.astype(np.float64))
        X = (X[:3]/X[3]).T
        good = (X[:, 2] > 0.2) & (X[:, 2] < 40) & np.isfinite(X).all(1)
        if good.sum() < self.min_inliers:
            return None
        return p0[ok][good], X[good]

    def push(self, stamp, left, right):
        """Feed one stereo pair. Every `stride` pairs returns (keyframe stamp, IMU-frame rotation vector from the
        keyframe to this pair), else None. Landmarks are triangulated at the keyframe and tracked frame to frame."""
        cv2 = self.cv2
        out = None
        if self.prev is not None:
            t_key, img, pts, X, n = self.prev
            p, st, _ = cv2.calcOpticalFlowPyrLK(img, left, pts, None, winSize=(21, 21), maxLevel=3)
            ok = st.ravel() == 1
            pts, X, n = p[ok], X[ok], n + 1
            if len(pts) < self.min_inliers:
                self.prev = None
            elif n < self.stride:
                self.prev = (t_key, left, pts, X, n)
                return None
            else:
                self.prev = None
                nrm = self._norm(pts, self.c['K0'], self.c['D0'])
                found, rv, tv, inl = cv2.solvePnPRansac(X, nrm, np.eye(3), None, reprojectionError=0.004,
                                                        iterationsCount=100, flags=cv2.SOLVEPNP_EPNP)
                if found and inl is not None and len(inl) >= self.min_inliers:
                    idx = inl.ravel()
                    found, rv, tv = cv2.solvePnP(X[idx], nrm[idx], np.eye(3), None, rv, tv, True,
                                                 flags=cv2.SOLVEPNP_ITERATIVE)
                    R = cv2.Rodrigues(rv)[0]          # X_k = R X_key + t: camera rotated by R^T
                    out = (t_key, self.R_ic @ rotvec(R.T))
        if self.prev is None:
            lm = self._landmarks(left, right)
            self.prev = (stamp, left, *lm, 0) if lm is not None else None
        return out


class InputMonitor:
    """Streaming monitor. Feed imu(stamp_s, gyro) and stereo(stamp_s, rotation_or_None) in arrival order; each call
    returns the current state. transitions and offset_estimates collect the history."""

    def __init__(self, cfg=None):
        self.c = {**INTEGRITY, **(cfg or {})}
        self.reset(first=True)

    def reset(self, first=False):
        self.epoch = 0 if first else self.epoch + 1
        self.state, self.lost_reason, self.warn = 'NORMAL', None, set()
        self.last = {'imu': None, 'cam': None}
        self.imu_t, self.imu_g = [], []
        self.cam_iv, self.cam_th = [], []
        self.next_estimate, self.over_lost = None, 0
        if first:
            self.transitions, self.offset_estimates = [], []
        self.now = None

    def _move(self, now, new, reason):
        if new != self.state:
            self.transitions.append({'t_s': float(now), 'from': self.state, 'to': new, 'reason': reason,
                                     'epoch_id': self.epoch})
            self.state = new

    def _lost(self, now, reason):
        if self.state != 'LOST':
            self.lost_reason = reason
            self._move(now, 'LOST', reason)

    def _update_warn(self, now):
        if self.state == 'LOST':
            return
        if self.warn:
            self._move(now, 'WARN', ','.join(sorted(self.warn)))
        else:
            self._move(now, 'NORMAL', 'cleared')

    def _watchdog(self, now):
        """The other stream's clock running past a silent stream catches outages that never end."""
        for kind, tol in (('imu', self.c['imu_outage_s']), ('cam', self.c['cam_outage_s'])):
            if self.last[kind] is not None and now - self.last[kind] >= tol - 1e-9:
                self._lost(now, f'{kind}_outage')

    def _sample(self, kind, stamp):
        prev = self.last[kind]
        if prev is not None and stamp <= prev:
            self._lost(stamp, f'{kind}_time_regression')
            return False
        tol = self.c['imu_outage_s'] if kind == 'imu' else self.c['cam_outage_s']
        if prev is not None and stamp - prev >= tol - 1e-9:
            self._lost(stamp, f'{kind}_outage')
        self.last[kind] = stamp
        return True

    def imu(self, stamp, gyro):
        stamp = float(stamp)
        if self._sample('imu', stamp):
            self.imu_t.append(stamp)
            self.imu_g.append(np.asarray(gyro, float))
            self._rate(stamp)
            self._offset(stamp)
        self._watchdog(stamp)
        self._update_warn(stamp)
        return self.state

    def _rate(self, now):
        t = self.imu_t
        if self.c['imu_rate_hz'] is None:
            if now - t[0] >= 2.0 and len(t) > 10:
                self.c['imu_rate_hz'] = (len(t) - 1)/(t[-1] - t[0])
            return
        if now - t[0] < self.c['rate_window_s']:
            return
        lo = now - self.c['rate_window_s']
        k = len(t) - int(np.searchsorted(t, lo, side='right'))
        low = k < self.c['rate_min_fraction']*self.c['imu_rate_hz']*self.c['rate_window_s']
        (self.warn.add if low else self.warn.discard)('imu_rate_low')

    def stereo(self, stamp, rotation):
        """rotation: None, or (start stamp, IMU-frame rotation vector from start to this pair) from StereoRotation."""
        stamp = float(stamp)
        if self._sample('cam', stamp):
            if rotation is not None:
                self.cam_iv.append((float(rotation[0]), stamp))
                self.cam_th.append(np.asarray(rotation[1], float))
            self._offset(stamp)
        self._watchdog(stamp)
        self._update_warn(stamp)
        return self.state

    def _offset(self, now):
        """Estimate over [end - window, end] once both streams have passed end (+ the search range for the IMU)."""
        c = self.c
        if not self.cam_iv:
            return
        if self.next_estimate is None:
            self.next_estimate = max(self.cam_iv[0][0], self.imu_t[0] + c['offset_search_ms']/1000) + c['offset_window_s'] if self.imu_t else None
            if self.next_estimate is None:
                return
        end, lim = self.next_estimate, c['offset_search_ms']/1000
        if self.last['cam'] is None or self.last['cam'] < end or not self.imu_t or self.imu_t[-1] < end + lim:
            return
        self.next_estimate += c['offset_step_s']
        lo = end - c['offset_window_s']
        iv = np.array(self.cam_iv)
        sel = (iv[:, 0] >= lo) & (iv[:, 1] <= end)
        t = np.array(self.imu_t)
        keep = (t >= lo - lim - 0.05) & (t <= end + lim + 0.05)
        est = estimate_offset(iv[sel], np.array(self.cam_th)[sel], t[keep], np.array(self.imu_g)[keep], c)
        est.pop('curve', None)
        self.offset_estimates.append({'t_s': float(now), 'window_end_s': float(end), 'epoch_id': self.epoch, **est})
        # drop history the next window no longer needs
        nxt = lo + c['offset_step_s']
        k = int(np.searchsorted(t, nxt - lim - 0.1))
        del self.imu_t[:max(0, min(k, len(t) - 2))], self.imu_g[:max(0, min(k, len(t) - 2))]
        m = int(np.sum(iv[:, 0] < nxt))
        del self.cam_iv[:m], self.cam_th[:m]
        if not est['observable']:
            return   # unobservable windows neither raise nor clear the offset alarm
        def beyond(limit):   # the whole near-optimal set lies outside +-limit
            return est['ci_lo_ms'] > limit or est['ci_hi_ms'] < -limit
        (self.warn.add if beyond(c['time_offset_warn_ms']) else self.warn.discard)('time_offset')
        self.over_lost = self.over_lost + 1 if beyond(c['time_offset_lost_ms']) else 0
        if self.over_lost >= c['lost_hold']:
            self._lost(now, 'time_offset')


# ----------------------------------------------------------------------------------------------- offline replay CLI

TOPICS = {'imu': '/imu0', 'cam0': '/cam0/image_raw', 'cam1': '/cam1/image_raw'}


def load_calib(path):
    """OpenVINS kalibr_imucam_chain.yaml (with its %YAML:1.0 header) -> StereoRotation calib dict."""
    import yaml
    text = '\n'.join(l for l in open(path).read().splitlines() if not l.startswith('%YAML'))
    y = yaml.safe_load(text)
    out = {}
    for i in (0, 1):
        cam = y[f'cam{i}']
        fx, fy, cx, cy = cam['intrinsics']
        out[f'K{i}'] = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], float)
        out[f'D{i}'] = np.array(cam['distortion_coeffs'], float)
        out[f'T_imu_cam{i}'] = np.array(cam['T_imu_cam'], float)
    return out


def fault_filter(fault):
    """fault.json (scripts/data/inject_fault.py) -> (keep(topic, stamp_ns), shift_ns(topic)), reproducing the derived
    bag from its parent: dropout drops [start_ns, end_ns] inclusive, outage drops [start_ns, end_ns)."""
    topics = set(fault.get('stream_topics', []))
    iv = fault.get('intervals', [])
    shift = {k: int(v) for k, v in fault.get('header_shift_ns', {}).items()}
    closed = fault.get('fault') == 'dropout'

    def keep(topic, ns):
        if topic not in topics:
            return True
        return not any(i['start_ns'] <= ns <= i['end_ns'] if closed else i['start_ns'] <= ns < i['end_ns'] for i in iv)
    return keep, lambda topic: shift.get(topic, 0)


def replay(bag, calib, cfg=None, fault=None, progress=False):
    """Run the monitor over a bag in log-time (arrival) order. Returns (monitor, counts per topic)."""
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore
    keep, shift = fault_filter(fault) if fault else ((lambda topic, ns: True), (lambda topic: 0))
    mon, rot = InputMonitor(cfg), StereoRotation(calib)
    counts = {t: 0 for t in TOPICS.values()}
    pending = {}
    with AnyReader([bag], default_typestore=get_typestore(Stores.ROS2_JAZZY)) as reader:
        conns = [c for c in reader.connections if c.topic in counts]
        for conn, _, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            ns = int(msg.header.stamp.sec)*10**9 + int(msg.header.stamp.nanosec)
            if not keep(conn.topic, ns):
                continue
            counts[conn.topic] += 1
            ns += shift(conn.topic)
            if conn.topic == TOPICS['imu']:
                v = msg.angular_velocity
                mon.imu(ns*1e-9, (v.x, v.y, v.z))
                continue
            img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)[:, :msg.width]
            pending.setdefault(ns, {})[conn.topic] = img
            pair = pending.get(ns)
            if TOPICS['cam0'] in pair and TOPICS['cam1'] in pair:
                del pending[ns]
                for old in [k for k in pending if k < ns]:
                    del pending[old]   # an unpaired frame never gets its partner after a later pair
                mon.stereo(ns*1e-9, rot.push(ns*1e-9, pair[TOPICS['cam0']], pair[TOPICS['cam1']]))
                if progress and counts[TOPICS['cam0']] % 500 == 0:
                    print(f'{counts[TOPICS["cam0"]]} pairs, state {mon.state}', flush=True)
    return mon, counts


def main(argv=None):
    import argparse
    import csv
    import json
    from pathlib import Path
    import yaml
    p = argparse.ArgumentParser(prog='uwbench input-integrity', description=__doc__.split('\n')[0])
    p.add_argument('--bag', type=Path, required=True, help='ROS2 bag directory (the parent bag when --fault is given)')
    p.add_argument('--calib', type=Path, required=True, help='OpenVINS kalibr_imucam_chain.yaml')
    p.add_argument('--fault', type=Path, help='fault.json: replay the derived fault bag from its parent')
    p.add_argument('--config', type=Path, help='YAML with an a_input_integrity block overriding INTEGRITY')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--progress', action='store_true')
    a = p.parse_args(argv)
    cfg = (yaml.safe_load(a.config.read_text()) or {}).get('a_input_integrity', {}) if a.config else {}
    fault = json.loads(a.fault.read_text()) if a.fault else None
    mon, counts = replay(a.bag, load_calib(a.calib), cfg, fault, a.progress)
    if fault:   # the reconstruction must match what inject_fault.py wrote
        want = {t: c['written_count'] for t, c in fault['checks'].items()}
        if any(counts[t] != n for t, n in want.items()):
            raise SystemExit(f'fault reconstruction mismatch: {counts} vs {want}')
    a.out.mkdir(parents=True, exist_ok=True)
    with open(a.out/'transitions.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, ['t_s', 'from', 'to', 'reason', 'epoch_id'])
        w.writeheader()
        w.writerows(mon.transitions)
    keys = ['t_s', 'window_end_s', 'epoch_id', 'offset_ms', 'ci_lo_ms', 'ci_hi_ms', 'se_ms', 'fit_r2', 'excitation_rad_s', 'n', 'observable', 'reason']
    with open(a.out/'offset_estimates.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(mon.offset_estimates)
    t0 = min(e['window_end_s'] for e in mon.offset_estimates) - mon.c['offset_window_s'] if mon.offset_estimates else None
    obs = [e for e in mon.offset_estimates if e['observable']]
    lost = next((t for t in mon.transitions if t['to'] == 'LOST'), None)
    summary = {'bag': str(a.bag), 'fault': str(a.fault) if a.fault else None, 'counts': counts,
               'config': {k: v for k, v in mon.c.items()}, 'final_state': mon.state,
               'first_lost': lost, 'warn_reasons': sorted({t['reason'] for t in mon.transitions if t['to'] == 'WARN'}),
               'offset': {'windows': len(mon.offset_estimates), 'observable': len(obs),
                          'median_ms': float(np.median([e['offset_ms'] for e in obs])) if obs else None,
                          'unobservable_reasons': sorted({e['reason'] for e in mon.offset_estimates if not e['observable']})},
               'first_window_start_s': t0}
    (a.out/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ('final_state', 'first_lost', 'warn_reasons', 'offset')}, indent=1))
    return 0
