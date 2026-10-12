"""uwbench detect-a, W2 pilot (P18 contract): causal replay of detector statistics over finished estimator runs.

Pilot scope, stated so nobody reads it as the M3 protocol:
- detectors: DET-NIS, DET-COV, DET-FEAT, and two DET-DIR parts: information gap tau_k (equivalent to thresholding a
  monotone drift envelope) and cross-sensor conflict (gyro-only vs estimated heading, depth gauge vs estimated z).
  DET-SCALAR is not implemented (A2 score undefined). Sonar is not used by A0, so x/y have no conflict source.
- operating points: leave-one-run-out; thresholds from the other runs' normal samples, false alarms counted as the
  fraction of normal samples with any alarm (too little normal data for alarms per 10 min). With early_warning_s > 0,
  samples that far before an exceedance onset are neither normal nor exceedance (alarms there are lead time).
- conflicts come in two forms: over a trailing window (injection rate) and since the first estimate (accumulated);
  the second was added after the first pilot pass showed the windowed form peaks before onset (candidate 2 of 10).
- the accumulated heading conflict integrates the gyro with three bias sources (method spec v2.2 M3.1): the
  estimator's online bias (absorbs part of the conflict), the online bias frozen at the first estimate, and, in
  simulation only, the simulator's true bias (an oracle bound, never a detector). The frozen form is also scored
  against its drift budget e(T) = n_sigma * sqrt(N^2 T + K^2 T^3 / 3) from the run's IMU noise density N and random
  walk K; the budget leaves out the error of the frozen bias itself, so it is optimistic.
- sonar (when the run's dataset has sonar images): per frame, the first return above a threshold in each azimuth
  column (range and bearing only, as an imaging sonar gives) and a RANSAC line through them. The first line after t_0
  fixes a world plane through the estimated pose (sonar_reference_s > 0 averages the lines of that long instead; an
  error d in the reference normal adds d times the distance travelled along the structure, and averaging over the
  opening manoeuvre also averages in its heading error, which on the pilot runs was the larger effect); afterwards
  the measured distance to the line is compared with the distance the estimated position predicts. The difference is a cumulative conflict along the plane normal,
  projected on the estimated body x and y. Frames whose line normal is more than gate_deg from the reference plane
  (seen through the estimated attitude) are skipped, so another structure does not count as conflict.
  The multi-plane form (sonar_conflict_multi) keeps every surface it has seen: a line is matched to a known plane by
  normal (sonar_match_deg) and by a small change of that plane's conflict since it was last seen (sonar_jump_m);
  unmatched lines start a new plane through the current estimated pose that inherits the conflict accumulated so far.
  The scan-to-map form (sonar_conflict_map) keeps the first-return points themselves: the first scan after t_0 is
  the map, every later scan is registered to it by 2-D point-to-plane ICP over a horizontal translation and a
  rotation (points weighted by 1/map_sigma_m^2, plus a random-walk prior on the change of the correction between
  scans, so directions the geometry does not constrain stay put and single-scan noise does not bend the map),
  and the scan's new points are added where the registration puts them, not where the estimate does. The map
  therefore stays in the t_0 frame as it grows; the conflict is the registered minus the estimated sonar position,
  and the registered rotation is a sonar heading conflict.
- every statistic at time t uses only data stamped <= t; reference errors are read only to score.
Inputs per run: estimator.log with the b64db17 [DIAG] lines, estimate_bag (poseimu + odomimu), the input bag (imu0,
depth/pressure) recorded in run.json, and eval-traj/axis_errors.csv for labels.
"""
import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import yaml
from rosbags.highlevel import AnyReader
from scipy.optimize import brentq
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from uwbench.metrics.axes import AXES, body_axis_sigma
from uwbench.trajectory import DEFAULTS, read_estimate

ATMOSPHERE_PA, RHO, G = 101325.0, 1000.0, 9.81  # same constants as scripts/sim_to_bag.py
SIM_EPOCH_S = 1_700_000_000  # scripts/sim_to_bag.py maps simulator t = 0 to this stamp
DIAG = re.compile(r'\[DIAG\] (MSCKF|SLAM) (.*)$')
CONFIG = {'nis_window_s': 1.0, 'feat_window_s': 1.0, 'info_window_s': 1.0, 'conflict_window_s': 10.0,
          'lambda_min_quantile': 0.10, 'false_alarm_targets': [0.0, 0.01, 0.05],
          'early_warning_s': [0.0, 10.0], 'budget_n_sigma': 3.0,
          'sonar_threshold': 0.25, 'sonar_inlier_m': 0.05, 'sonar_min_inliers': 20, 'sonar_gate_deg': 15.0,
          'sonar_reference_s': 0.0, 'sonar_max_thickness': 0.2, 'sonar_hold_s': 1.0, 'sonar_match_deg': 3.0, 'sonar_jump_m': 0.15,
          'map_voxel_m': 0.05, 'map_assoc_m': 0.3, 'map_huber_m': 0.05, 'map_min_pairs': 20,
          'map_sigma_m': 0.03, 'map_prior_m': 0.02, 'map_prior_deg': 0.02, 'map_iterations': 6,
          'map_normal_k': 12, 'map_degenerate_ratio': 0.02, 'map_lever_m': 2.0,
          'map_grow': False, 'map_rotation': False, 'map_init_s': 0.0, 'sonar_footprint': False}
DETECTORS = {'DET-NIS': 'global', 'DET-COV': 'axis', 'DET-FEAT': 'global', 'DIR-gap': 'axis', 'DIR-conflict': 'axis',
             'DIR-gap+conflict': 'axis', 'DIR-cumconflict': 'axis', 'DIR-gap+cumconflict': 'axis',
             'DIR-cumconflict-frozen': 'axis', 'DIR-cumconflict-frozen-budget': 'axis', 'ORACLE-cumconflict-truebias': 'axis',
             'DIR-cumconflict+sonar': 'axis', 'DIR-cumconflict+sonar-multi': 'axis', 'DIR-cumconflict+sonar-map': 'axis',
             'DIR-sonar-map+yaw': 'axis'}
CUMULATIVE = {'DIR-cumconflict': 'conflict_cum', 'DIR-cumconflict-frozen': 'conflict_cum_frozen',
              'DIR-cumconflict-frozen-budget': 'conflict_cum_frozen_excess', 'ORACLE-cumconflict-truebias': 'conflict_cum_true',
              'DIR-cumconflict+sonar': 'conflict_cum_sonar', 'DIR-cumconflict+sonar-multi': 'conflict_cum_sonar_multi',
              'DIR-cumconflict+sonar-map': 'conflict_cum_sonar_map', 'DIR-sonar-map+yaw': 'conflict_cum_sonar_map_yaw'}
SOURCE_AXES = {'conflict_cum_sonar': (0, 1, 2, 3), 'conflict_cum_sonar_multi': (0, 1, 2, 3),
               'conflict_cum_sonar_map': (0, 1, 2, 3), 'conflict_cum_sonar_map_yaw': (0, 1, 2, 3)}  # every other cumulative form: depth for z, gyro for heading only


def parse_diag(log_path):
    """Per-update rows: time, kind (0 MSCKF, 1 SLAM), tested, rejected, chi2_acc, dof_acc, chi2_all, dof_all, 6x6 gain."""
    rows, gains = [], []
    for line in Path(log_path).read_text(errors='replace').splitlines():
        m = DIAG.search(line)
        if not m:
            continue
        v = m.group(2).split()
        if len(v) != 7 + 21:
            raise ValueError(f'Malformed DIAG line: {line[:120]}')
        rows.append([float(v[0]), 0 if m.group(1) == 'MSCKF' else 1] + [float(x) for x in v[1:7]])
        upper = np.array(v[7:], dtype=float)
        gain = np.zeros((6, 6))
        gain[np.triu_indices(6)] = upper
        gains.append(gain + np.triu(gain, 1).T)
    if not rows:
        raise ValueError(f'No [DIAG] lines in {log_path}; run with the diagnostic OpenVINS build')
    return np.array(rows), np.array(gains)


def read_series(bag, topic, field):
    out = []
    with AnyReader([Path(bag)]) as reader:
        for conn, _, raw in reader.messages(connections=[c for c in reader.connections if c.topic == topic]):
            msg = reader.deserialize(raw, conn.msgtype)
            t = int(msg.header.stamp.sec) + int(msg.header.stamp.nanosec)*1e-9
            if field == 'gyro':
                w = msg.angular_velocity
                out.append([t, w.x, w.y, w.z])
            elif field == 'twist_angular':
                w = msg.twist.twist.angular
                out.append([t, w.x, w.y, w.z])
            elif field == 'pressure':
                out.append([t, float(msg.fluid_pressure)])
    if not out:
        raise ValueError(f'No {topic} messages in {bag}')
    return np.array(out)


def windowed(update_t, values, t, window, reduce):
    """reduce(values in (t - window, t]) for each t; NaN where the window is empty."""
    out = np.full(len(t), np.nan)
    lo = np.searchsorted(update_t, t - window, side='right')
    hi = np.searchsorted(update_t, t, side='right')
    for i, (a, b) in enumerate(zip(lo, hi)):
        if b > a:
            out[i] = reduce(values[a:b])
    return out


def gyro_heading(imu, bias_t, bias):
    """Body-to-world rotation from integrating gyro minus the latest causal bias (bias_t, bias), starting at identity."""
    k = np.clip(np.searchsorted(bias_t, imu[:, 0], side='right') - 1, 0, None)
    w = imu[:, 1:] - bias[k]
    w[imu[:, 0] < bias_t[0]] = 0.0  # before the estimator exists there is no bias and nothing to compare
    dt = np.diff(imu[:, 0], prepend=imu[0, 0])
    steps = Rotation.from_rotvec(w*dt[:, None]).as_matrix()
    Q = np.empty_like(steps)
    current = np.eye(3)
    for i, step in enumerate(steps):
        current = current @ step
        Q[i] = current
    return Q


def drift_budget_deg(T, noise_density, random_walk, n_sigma):
    """n_sigma bound on heading error from integrating a gyro whose bias was frozen T seconds ago (bias error excluded)."""
    T = np.clip(T, 0, None)
    return np.degrees(n_sigma*np.sqrt(noise_density**2*T + random_walk**2*T**3/3))


def sonar_first_returns(image, geometry, threshold):
    """Sonar-frame (x forward, y left) points from the first return above threshold in each azimuth column.

    Image layout as HoloOcean ImagingSonar: [RangeBins, AzimuthBins], row 0 at RangeMin, column 0 at -Azimuth/2.
    """
    above = image > threshold
    has = above.any(axis=0)
    row = np.argmax(above, axis=0)[has]
    col = np.nonzero(has)[0]
    rng_m = geometry['RangeMin'] + (row + 0.5)*(geometry['RangeMax'] - geometry['RangeMin'])/image.shape[0]
    az = np.radians(-geometry['Azimuth']/2 + (col + 0.5)*geometry['Azimuth']/image.shape[1])
    return np.column_stack([rng_m*np.cos(az), rng_m*np.sin(az)])


def sonar_echo_thickness(image, geometry, threshold, gap_bins=3):
    """Median over azimuth columns of the echo's range extent from the first return, relative to that range.

    An imaging sonar does not measure elevation, so the first return comes from the nearest point within the elevation
    fan. On a vertical face the fan spans ranges D to D/cos(E/2) (relative extent 1.5% for E = 20 deg, plus one range
    bin); on a face tilted b > E/2 from vertical it spans D/cos(b - E/2) to D/cos(b + E/2), and the first return slides
    along the face when the range changes, which a 2-D model reads as wall motion. The run ends at the first gap of
    more than gap_bins rows. NaN without returns.
    """
    above = image > threshold
    cols = np.nonzero(above.any(axis=0))[0]
    if not len(cols):
        return np.nan
    bin_m = (geometry['RangeMax'] - geometry['RangeMin'])/image.shape[0]
    rel = []
    for c in cols:
        rows = np.nonzero(above[:, c])[0]
        gaps = np.nonzero(np.diff(rows) > gap_bins)[0]
        end = rows[gaps[0]] if len(gaps) else rows[-1]
        rel.append((end - rows[0] + 1)/(rows[0] + 0.5 + geometry['RangeMin']/bin_m))
    return float(np.median(rel))


def fit_line(points, inlier_m, min_inliers, rng, iterations=100):
    """RANSAC line m . p = d through 2-D points, m a unit normal pointing away from the origin; None if too few."""
    if len(points) < min_inliers:
        return None
    best = None
    for _ in range(iterations):
        a, b = points[rng.choice(len(points), 2, replace=False)]
        direction = b - a
        if np.linalg.norm(direction) < 1e-6:
            continue
        m = np.array([-direction[1], direction[0]])/np.linalg.norm(direction)
        inliers = np.abs(points @ m - a @ m) < inlier_m
        if best is None or inliers.sum() > best.sum():
            best = inliers
    if best is None or best.sum() < min_inliers:
        return None
    q = points[best]
    centre = q.mean(axis=0)
    m = np.linalg.eigh(np.cov((q - centre).T))[1][:, 0]  # least-squares refit on the inliers
    d = centre @ m
    return (m, d) if d > 0 else (-m, -d)


def sonar_lines(stamps, images, geometry, cfg=CONFIG):
    """Per sonar frame: (t, m_x, m_y, d) of the dominant line, NaN where none or where the echo is too thick for a vertical face."""
    rng = np.random.default_rng(0)
    out = np.full((len(stamps), 4), np.nan)
    out[:, 0] = stamps
    for i, image in enumerate(images):
        if not sonar_echo_thickness(image, geometry, cfg['sonar_threshold']) <= cfg['sonar_max_thickness']:
            continue  # no vertical structure in view (inclined face or nothing): elevation ambiguity, no measurement
        line = fit_line(sonar_first_returns(image, geometry, cfg['sonar_threshold']), cfg['sonar_inlier_m'],
                        cfg['sonar_min_inliers'], rng)
        if line is not None:
            out[i, 1:3], out[i, 3] = line
    return out


def hold(value_t, values, t, hold_s, width):
    """Latest value at or before each t, NaN (not scorable) before the first one or when it is older than hold_s."""
    out = np.full((len(t), width), np.nan)
    if values:
        value_t = np.asarray(value_t)
        k = np.searchsorted(value_t, t, side='right') - 1
        ok = k >= 0
        ok[ok] = t[ok] - value_t[k[ok]] <= hold_s
        out[ok] = np.asarray(values)[k[ok]]
    return out


def sonar_conflict(t, est, lines, T_imu_sonar, gate_deg, reference_s=0.0, hold_s=np.inf):
    """Cumulative conflict along a reference plane normal, projected on estimated body x and y, at stamps t.

    The reference plane averages the lines seen in the first reference_s after the first one. NaN (not scorable) until
    it is fixed and wherever the last measured frame is older than hold_s (gated or skipped frames are not measurements).
    """
    est_t, R, p = est[0], est[1], est[2]
    R_is, t_is = T_imu_sonar[:3, :3], T_imu_sonar[:3, 3]
    value_t, values = [], []
    ref, window = None, []
    for ts, mx, my, d in lines:
        if not np.isfinite(d) or ts < est_t[0]:
            continue
        j = np.searchsorted(est_t, ts, side='right') - 1  # latest estimate at or before the frame
        p_s = p[j] + R[j] @ t_is
        n_w = R[j] @ R_is @ np.array([mx, my, 0.0])
        if ref is None:
            if window and np.degrees(np.arccos(np.clip(n_w @ window[0][1], -1, 1))) > gate_deg:
                continue
            window.append((ts, n_w, p_s, d))
            if ts - window[0][0] >= reference_s:
                n_ref = np.mean([w[1] for w in window], axis=0)
                n_ref /= np.linalg.norm(n_ref)
                ref = (n_ref, np.mean([n_ref @ w[2] + w[3] for w in window]))
            if ref is None:
                continue  # the frame that fixes the reference is measured against it below
        n_ref, h = ref
        if np.degrees(np.arccos(np.clip(n_w @ n_ref, -1, 1))) > gate_deg:
            continue
        c = abs(d - (h - n_ref @ p_s))
        value_t.append(ts)
        values.append(c*np.abs(R[j].T @ n_ref)[:2])
    return hold(value_t, values, t, hold_s, 2)


def sonar_conflict_multi(t, est, lines, T_imu_sonar, match_deg, jump_m, hold_s=np.inf):
    """Cumulative horizontal conflict over a growing set of local planes, projected on estimated body x and y.

    Each frame's line is associated with a known plane if the normals (through the estimated attitude) are within
    match_deg and the conflict on that plane moved by less than jump_m since the plane was last seen: drift is slow,
    a different surface jumps. Otherwise a new plane is fixed through the current estimated pose and inherits the
    current conflict vector, so changing surfaces does not reset what has accumulated. The conflict vector is
    carry + c * n of the plane in view; only its component along that normal is measured. NaN where the last
    associated frame is older than hold_s.
    """
    est_t, R, p = est[0], est[1], est[2]
    R_is, t_is = T_imu_sonar[:3, :3], T_imu_sonar[:3, 3]
    cos_match = np.cos(np.radians(match_deg))
    planes = []  # dicts: n, h, carry (3-vector, est world, horizontal), c_last
    current = np.zeros(3)
    value_t, values, counts = [], [], []
    for ts, mx, my, d in lines:
        if not np.isfinite(d) or ts < est_t[0]:
            continue
        j = np.searchsorted(est_t, ts, side='right') - 1  # latest estimate at or before the frame
        p_s = p[j] + R[j] @ t_is
        n_w = R[j] @ R_is @ np.array([mx, my, 0.0])
        best, best_jump = None, jump_m
        for plane in planes:
            if n_w @ plane['n'] < cos_match:
                continue
            c = d - (plane['h'] - plane['n'] @ p_s)
            if abs(c - plane['c_last']) < best_jump:
                best, best_jump, best_c = plane, abs(c - plane['c_last']), c
        if best is None:
            planes.append({'n': n_w, 'h': n_w @ p_s + d, 'carry': current.copy(), 'c_last': 0.0})
            value_t.append(ts)  # measured: zero on the new plane, so the conflict is the carried vector
            values.append(np.abs(R[j].T @ current)[:2])
            counts.append(len(planes))
            continue
        best['c_last'] = best_c
        current = best['carry'] + best_c*best['n']
        value_t.append(ts)
        values.append(np.abs(R[j].T @ current)[:2])
        counts.append(len(planes))
    return hold(value_t, values, t, hold_s, 2), len(planes)


def scan_normals(points, k=12):
    """Unit normals of 2-D scan points from PCA of their k nearest neighbours in the same scan."""
    _, nb = cKDTree(points).query(points, k=min(k, len(points)))
    X = points[nb] - points[nb].mean(axis=1, keepdims=True)
    return np.linalg.eigh(np.einsum('nki,nkj->nij', X, X))[1][:, :, 0]


def footprint_correct(points, half_bin, k=12):
    """Move each first return to the bearing in its azimuth bin nearest to the surface.

    The first return of a bin comes from the bin edge closest to the foot of the local surface normal (or from the foot
    itself if it lies inside the bin), not from the bin centre where the image puts it. At oblique incidence this puts
    points about r sin(incidence) * half_bin too close (1.6 cm at 60 deg and 4.5 m with 0.94 deg bins), bending the
    fan edges; a growing map that inserts them bends with them. Range is unchanged; the bearing moves by at most half_bin
    toward the foot of the normal estimated from the k nearest scan points.
    """
    if len(points) < 3:
        return points
    n = scan_normals(points, k)
    n = n*np.where(np.einsum('ij,ij->i', n, points) > 0, -1.0, 1.0)[:, None]  # normal toward the sensor
    r, th = np.hypot(points[:, 0], points[:, 1]), np.arctan2(points[:, 1], points[:, 0])
    foot = np.arctan2(-n[:, 1], -n[:, 0])
    th = th + np.clip(np.angle(np.exp(1j*(foot - th))), -half_bin, half_bin)
    return np.column_stack([r*np.cos(th), r*np.sin(th)])


def rot2(angle):
    c, s_ = np.cos(angle), np.sin(angle)
    return np.array([[c, -s_], [s_, c]])


def sonar_conflict_map(t, est, scans, T_imu_sonar, cfg=CONFIG):
    """Scan-to-map sonar conflict at stamps t: columns body-x and body-y position conflict (m), heading conflict (deg).

    scans: list of (stamp, [n, 2] sonar-frame points); NaN where the last registered scan is older than sonar_hold_s. The correction (phi, tau) maps estimated-world horizontal points
    into the map frame: q_map = rot(phi) q + tau.
    """
    est_t, R, p = est[0], est[1], est[2]
    R_is, t_is = T_imu_sonar[:3, :3], T_imu_sonar[:3, 3]
    phi, tau = 0.0, np.zeros(2)
    map_pts, map_nrm, voxels, tree = None, None, set(), None
    # per-scan random-walk prior on the correction: how far it may move between scans (drift, not measurement noise)
    w_t, w_phi = 1/cfg['map_prior_m']**2, 1/np.radians(cfg['map_prior_deg'])**2
    w_point = 1/cfg['map_sigma_m']**2
    huber = cfg['map_huber_m']
    value_t, values = [], []

    def add(points, normals):
        nonlocal map_pts, map_nrm, tree
        keys = np.floor(points/cfg['map_voxel_m']).astype(np.int64)
        keep = []
        for i, key in enumerate(map(tuple, keys)):
            if key not in voxels:
                voxels.add(key)
                keep.append(i)
        if keep:
            map_pts = points[keep] if map_pts is None else np.vstack([map_pts, points[keep]])
            map_nrm = normals[keep] if map_nrm is None else np.vstack([map_nrm, normals[keep]])
            tree = cKDTree(map_pts)

    for ts, pts in scans:
        if ts < est_t[0] or len(pts) < cfg['map_min_pairs']:
            continue
        j = np.searchsorted(est_t, ts, side='right') - 1
        c3 = p[j] + R[j] @ t_is
        q = (R[j] @ R_is @ np.column_stack([pts, np.zeros(len(pts))]).T).T[:, :2] + c3[:2]  # estimated world, horizontal
        nrm = scan_normals(pts, cfg['map_normal_k']) @ (R[j] @ R_is)[:2, :2].T
        if map_pts is None or (not cfg['map_grow'] and ts - map_t0 <= cfg['map_init_s']):
            # the first scans (map_init_s after the first) define the map, in the estimated world at t_0
            if map_pts is None:
                map_t0 = ts
            add(q, nrm)
            value_t.append(ts)  # the scans that define the map match it with zero correction
            values.append([0.0, 0.0, 0.0])
            continue
        c = c3[:2]
        registered = True
        for _ in range(cfg['map_iterations']):
            y = q @ rot2(phi).T + tau
            cy = rot2(phi) @ c + tau
            dist, idx = tree.query(y)
            if cfg['map_grow']:
                ok = dist < cfg['map_assoc_m']
            else:  # fixed map: a point beyond its edge is compared with the tangent of the nearest map point
                ok = np.abs(np.einsum('ij,ij->i', map_nrm[idx], y - map_pts[idx])) < cfg['map_assoc_m']
            if ok.sum() < cfg['map_min_pairs']:
                registered = False  # no overlap with the map: keep the correction, add nothing
                break
            n, m, yy = map_nrm[idx[ok]], map_pts[idx[ok]], y[ok]
            r = np.einsum('ij,ij->i', n, yy - m)
            lever = yy - cy
            J = np.column_stack([n, n[:, 0]*(-lever[:, 1]) + n[:, 1]*lever[:, 0]])
            w = w_point*np.where(np.abs(r) <= huber, 1.0, huber/np.abs(r))
            if not cfg['map_rotation']:
                J = J[:, :2]
            H_data = J.T @ (w[:, None]*J)
            prior = np.diag([w_t, w_t, w_phi][:J.shape[1]])
            delta = -np.linalg.solve(H_data + prior, J.T @ (w*r))
            if not cfg['map_rotation']:
                H_data, delta = np.pad(H_data, (0, 1)), np.append(delta, 0.0)
                H_data[2, 2] = 0.0
            # do not move along directions the scan barely constrains (a flat wall along itself): with quantised ranges
            # and noisy normals they carry spurious information that slides the registration
            S = np.diag([1.0, 1.0, 1/cfg['map_lever_m']])  # rotation expressed as metres at the lever arm
            lam, V = np.linalg.eigh(S @ H_data @ S)
            weak = V[:, lam < cfg['map_degenerate_ratio']*lam.max()]
            scaled = np.linalg.solve(S, delta)
            delta = S @ (scaled - weak @ (weak.T @ scaled))
            phi += delta[2]
            tau = rot2(delta[2]) @ (tau - cy) + cy + delta[:2]
        if registered:
            if cfg['map_grow']:
                add(q @ rot2(phi).T + tau, nrm @ rot2(phi).T)
            e = (rot2(phi) @ c + tau) - c
            value_t.append(ts)
            values.append([*np.abs(R[j].T @ np.array([e[0], e[1], 0.0]))[:2], abs(np.degrees(phi))])
    return hold(value_t, values, t, cfg.get('sonar_hold_s', np.inf), 3), (0 if map_pts is None else len(map_pts))


def statistics(t, est, diag, imu, odom, pressure, cfg=CONFIG, gyro_noise=None, true_bias=None, sonar=None):
    """Detector statistics at estimate stamps t (all causal). Larger means more alarming for every column.

    gyro_noise: (noise density, random walk) for the frozen-bias drift budget; true_bias: simulator (t, bias xyz), oracle;
    sonar: (lines from sonar_lines, 4x4 sonar pose in the IMU frame).
    """
    est_t, R, p, cov = est
    k = np.searchsorted(est_t, t)
    R_t, p_t, cov_t = R[k], p[k], cov[k]
    rows, gains = diag
    ut = rows[:, 0]
    order = np.argsort(ut, kind='stable')
    rows, gains, ut = rows[order], gains[order], ut[order]
    stats = {}
    nis = windowed(ut, rows[:, [4, 5]], t, cfg['nis_window_s'], lambda v: v[:, 0].sum()/v[:, 1].sum() if v[:, 1].sum() else np.nan)
    stats['DET-NIS'] = np.nan_to_num(nis, nan=0.0)
    # accepted features per update stamp (MSCKF + SLAM chunks at the same stamp), minimum over the window
    stamps, inverse = np.unique(ut, return_inverse=True)
    accepted = np.bincount(inverse, weights=rows[:, 2] - rows[:, 3])
    feat = windowed(stamps, accepted, t, cfg['feat_window_s'], np.min)
    stats['DET-FEAT'] = -np.nan_to_num(feat, nan=0.0)
    tol = np.array([DEFAULTS['risk']['tolerance_position_m']]*3 + [DEFAULTS['risk']['tolerance_yaw_deg']])
    stats['DET-COV'] = body_axis_sigma(R_t, cov_t)/tol
    # information gain per body axis: position along estimated body axes, heading = world z in the local error frame
    ku = np.clip(np.searchsorted(est_t, ut, side='right') - 1, 0, None)
    Ru = R[ku]
    lam_p = np.einsum('nik,nij,njk->nk', Ru, gains[:, 3:, 3:], Ru)
    u = np.swapaxes(Ru, 1, 2)[:, :, 2]  # world z expressed in the body frame
    lam_yaw = np.einsum('ni,nij,nj->n', u, gains[:, :3, :3], u)
    lam = np.column_stack([lam_p, lam_yaw])
    stats['lambda'] = np.column_stack([windowed(ut, lam[:, i], t, cfg['info_window_s'], np.sum) for i in range(4)])
    # conflicts over a trailing window: gyro-only heading change vs estimate, depth gauge vs estimated world z
    W = cfg['conflict_window_s']
    bias_t, bias = odom[:, 0], None
    ki = np.clip(np.searchsorted(imu[:, 0], odom[:, 0]), 0, len(imu) - 1)
    bias = imu[ki, 1:] - odom[:, 1:]
    Q = gyro_heading(imu, bias_t, bias)
    t0 = np.maximum(t - W, est_t[0])
    k0 = np.searchsorted(est_t, t0)
    qi0 = np.clip(np.searchsorted(imu[:, 0], est_t[k0], side='right') - 1, 0, None)
    qi1 = np.clip(np.searchsorted(imu[:, 0], t, side='right') - 1, 0, None)
    dR_gyro = np.swapaxes(Q[qi0], 1, 2) @ Q[qi1]
    dR_est = np.swapaxes(R[k0], 1, 2) @ R_t
    e_body = Rotation.from_matrix(np.swapaxes(dR_gyro, 1, 2) @ dR_est).as_rotvec()
    yaw_conflict = np.degrees(np.abs(np.einsum('nij,nj->ni', R_t, e_body)[:, 2]))
    z_depth = -(pressure[:, 1] - ATMOSPHERE_PA)/(RHO*G)
    zi0 = np.clip(np.searchsorted(pressure[:, 0], est_t[k0], side='right') - 1, 0, None)
    zi1 = np.clip(np.searchsorted(pressure[:, 0], t, side='right') - 1, 0, None)
    z_conflict = np.abs((p_t[:, 2] - p[k0, 2]) - (z_depth[zi1] - z_depth[zi0]))
    stats['conflict'] = np.column_stack([np.zeros(len(t)), np.zeros(len(t)), z_conflict, yaw_conflict])
    # cumulative versions since the first estimate: heading is unobservable in VIO, so drift that vision injected
    # stays in the estimate; the windowed form measures the injection rate, this one the accumulated amount
    q_first = np.clip(np.searchsorted(imu[:, 0], est_t[0], side='right') - 1, 0, None)
    dR_gyro0 = np.swapaxes(Q[q_first], -1, -2) @ Q[qi1]
    dR_est0 = np.swapaxes(R[0], -1, -2) @ R_t
    e0 = Rotation.from_matrix(np.swapaxes(dR_gyro0, 1, 2) @ dR_est0).as_rotvec()
    yaw_cum = np.degrees(np.abs(np.einsum('nij,nj->ni', R_t, e0)[:, 2]))
    z_first = np.clip(np.searchsorted(pressure[:, 0], est_t[0], side='right') - 1, 0, None)
    z_cum = np.abs((p_t[:, 2] - p[0, 2]) - (z_depth[zi1] - z_depth[z_first]))
    stats['conflict_cum'] = np.column_stack([np.zeros(len(t)), np.zeros(len(t)), z_cum, yaw_cum])

    def cumulative_yaw(Q_alt):
        dR = np.swapaxes(Q_alt[q_first], -1, -2) @ Q_alt[qi1]
        e = Rotation.from_matrix(np.swapaxes(dR, 1, 2) @ dR_est0).as_rotvec()
        return np.degrees(np.abs(np.einsum('nij,nj->ni', R_t, e)[:, 2]))
    # the online bias follows the estimate, so wrong vision information partly moves into it; freeze it at t_0
    yaw_frozen = cumulative_yaw(gyro_heading(imu, bias_t[:1], bias[:1]))
    stats['conflict_cum_frozen'] = np.column_stack([np.zeros(len(t)), np.zeros(len(t)), z_cum, yaw_frozen])
    budget = np.zeros(len(t)) if gyro_noise is None else drift_budget_deg(t - est_t[0], *gyro_noise, cfg['budget_n_sigma'])
    stats['budget_yaw_deg'] = budget
    stats['conflict_cum_frozen_excess'] = np.column_stack([np.zeros(len(t)), np.zeros(len(t)), z_cum, yaw_frozen - budget])
    if true_bias is not None:
        stats['conflict_cum_true'] = np.column_stack([np.zeros(len(t)), np.zeros(len(t)), z_cum,
                                                      cumulative_yaw(gyro_heading(imu, true_bias[:, 0], true_bias[:, 1:]))])
    if sonar is not None:
        stats['conflict_cum_sonar'] = np.column_stack([sonar_conflict(t, est, sonar[0], sonar[1], cfg['sonar_gate_deg'],
                                                                      cfg['sonar_reference_s'], cfg['sonar_hold_s']),
                                                       z_cum, yaw_cum])
        multi, n_planes = sonar_conflict_multi(t, est, sonar[0], sonar[1], cfg['sonar_match_deg'], cfg['sonar_jump_m'],
                                                cfg['sonar_hold_s'])
        stats['conflict_cum_sonar_multi'] = np.column_stack([multi, z_cum, yaw_cum])
        stats['sonar_planes'] = n_planes
        if len(sonar) > 2:
            m, n_map = sonar_conflict_map(t, est, sonar[2], sonar[1], cfg)
            stats['conflict_cum_sonar_map'] = np.column_stack([m[:, :2], z_cum, yaw_cum])
            stats['conflict_cum_sonar_map_yaw'] = np.column_stack([m[:, :2], z_cum, m[:, 2]])
            stats['sonar_map_points'] = n_map
    return stats


def information_gap(t, lam, lam_min):
    """tau_k: time since the windowed information on axis k last reached lam_min (from the first stamp if never)."""
    tau = np.zeros_like(lam)
    for a in range(lam.shape[1]):
        last = t[0]
        for i, (ti, v) in enumerate(zip(t, lam[:, a])):
            if np.isfinite(v) and v >= lam_min[a]:
                last = ti
            tau[i, a] = ti - last
    return tau


def load_sim_sonar(dataset, cfg=CONFIG):
    """Sonar lines, extrinsic and first-return scans of a simulated dataset (raycast replay if present, else raw)."""
    raw = dataset/'raw'
    if not (raw/'meta.json').exists():
        return None
    meta = json.loads((raw/'meta.json').read_text())
    source = dataset/'sonar_raycast' if (dataset/'sonar_raycast'/'sonar_t.npy').exists() else raw
    stamps = np.load(source/'sonar_t.npy') + SIM_EPOCH_S
    images = np.concatenate([np.load(f) for f in sorted(source.glob('sonar_[0-9]*.npy'))])
    geometry = meta['route']['sensors']['sonar']
    lines = sonar_lines(stamps, images, geometry, cfg)
    scans = [(ts, sonar_first_returns(image, geometry, cfg['sonar_threshold'])
              if sonar_echo_thickness(image, geometry, cfg['sonar_threshold']) <= cfg['sonar_max_thickness'] else np.zeros((0, 2)))
             for ts, image in zip(stamps, images)]
    if cfg.get('sonar_footprint'):
        half_bin = np.radians(geometry['Azimuth']/geometry['AzimuthBins'])/2
        scans = [(ts, footprint_correct(pts, half_bin, cfg['map_normal_k'])) for ts, pts in scans]
    return lines, np.asarray(meta['socket_extrinsics_in_imu_frame']['sonar']), scans


def load_run(run_dir, cfg=CONFIG):
    run_dir = Path(run_dir)
    record = json.loads((run_dir/'run.json').read_text())
    bag = Path(record['bag'])
    labels = np.genfromtxt(run_dir/'eval-traj'/'axis_errors.csv', delimiter=',', names=True)
    est = read_estimate(run_dir/'estimate_bag', DEFAULTS['estimate_topic'])
    if est[3] is None:
        raise ValueError(f'{run_dir}: estimate has no covariance')
    chain = Path(record['config_dir'])/'kalibr_imu_chain.yaml'
    text = chain.read_text()
    imu0 = yaml.safe_load(text.split('\n', 1)[1] if text.startswith('%YAML') else text)['imu0']
    gyro_noise = (imu0['gyroscope_noise_density'], imu0['gyroscope_random_walk'])
    raw = bag.parent/'raw'/'imu.npy'  # simulator output: columns t, accel, gyro, accel bias, gyro bias (same frame as /imu0)
    true_bias = None
    if raw.exists():
        r = np.load(raw)
        true_bias = np.column_stack([r[:, 0] + SIM_EPOCH_S, r[:, 10:13]])
    data = {'name': run_dir.name, 't': labels['t_s'], 'scored': labels['in_align_window'] == 0,
            'errors': np.column_stack([labels[f'err_{a}'] for a in AXES])}
    data['stats'] = statistics(data['t'], est, parse_diag(run_dir/'estimator.log'),
                               read_series(bag, '/imu0', 'gyro'),
                               read_series(run_dir/'estimate_bag', '/ov_msckf/odomimu', 'twist_angular'),
                               read_series(bag, '/depth/pressure', 'pressure'), cfg=cfg, gyro_noise=gyro_noise,
                               true_bias=true_bias, sonar=load_sim_sonar(bag.parent, cfg))
    data['gyro_noise'] = gyro_noise
    return data


def exceed_labels(errors):
    tol = np.array([DEFAULTS['risk']['tolerance_position_m']]*3 + [DEFAULTS['risk']['tolerance_yaw_deg']])
    return np.abs(errors) > tol


def threshold(values, target):
    """Smallest threshold whose alarm fraction (values > thr) on these normal samples is <= target.

    NaN samples (not scorable) are left out; with none left the threshold is infinite (never alarms).
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.inf
    return float(values.max()) if target <= 0 else float(np.quantile(values, 1 - target))


def evaluate(runs, cfg=CONFIG):
    """Leave-one-run-out operating points; returns per-run and pooled miss/false-alarm tables."""
    results = []
    for grace, target in [(g, f) for g in cfg['early_warning_s'] for f in cfg['false_alarm_targets']]:
        for r in runs:
            r['exceed'] = exceed_labels(r['errors']) & r['scored'][:, None]
            any_exceed = r['exceed'].any(axis=1)
            onsets = r['t'][any_exceed & ~np.r_[False, any_exceed[:-1]]]
            warning = np.zeros(len(r['t']), bool)
            for onset in onsets:
                warning |= (r['t'] >= onset - grace) & (r['t'] < onset)
            r['normal'] = r['scored'] & ~any_exceed & ~warning
        for test in runs:
            train = [r for r in runs if r is not test]
            normal = lambda key, col=None: np.concatenate([(r['stats'][key] if col is None else r['stats'][key][:, col])[r['normal']] for r in train])
            lam_min = [float(np.quantile(np.nan_to_num(normal('lambda', a)), cfg['lambda_min_quantile'])) for a in range(4)]
            for r in runs:
                r['stats']['tau'] = information_gap(r['t'], np.nan_to_num(r['stats']['lambda']), lam_min)
            alarms, scorable = {}, {}
            for name in ('DET-NIS', 'DET-FEAT'):
                alarms[name] = np.repeat((test['stats'][name] > threshold(normal(name), target))[:, None], 4, axis=1)
            per_axis = target/4  # any-axis false alarms at most the target on the training normals
            alarms['DET-COV'] = np.column_stack([test['stats']['DET-COV'][:, a] > threshold(normal('DET-COV', a), per_axis) for a in range(4)])
            alarms['DIR-gap'] = np.column_stack([test['stats']['tau'][:, a] > threshold(normal('tau', a), per_axis) for a in range(4)])
            conflict = np.zeros((len(test['t']), 4), bool)
            for a in (2, 3):  # depth for z, gyro for heading; x and y have no independent sensor in A0
                conflict[:, a] = test['stats']['conflict'][:, a] > threshold(normal('conflict', a), per_axis/2)
            alarms['DIR-conflict'] = conflict
            gap_half = np.column_stack([test['stats']['tau'][:, a] > threshold(normal('tau', a), per_axis/2) for a in range(4)])
            alarms['DIR-gap+conflict'] = gap_half | conflict
            for name, key in CUMULATIVE.items():
                if not all(key in r['stats'] for r in runs):
                    continue  # the oracle needs the simulator's bias, the sonar form sonar data, in every run
                cum = np.zeros((len(test['t']), 4), bool)
                for a in SOURCE_AXES.get(key, (2, 3)):  # same per-axis budget everywhere: z and heading thresholds match
                    cum[:, a] = test['stats'][key][:, a] > threshold(normal(key, a), per_axis/2)
                alarms[name] = cum
                # NaN = the source has no current measurement (e.g. sonar gated on an inclined face): not scorable
                scorable[name] = np.isfinite(test['stats'][key])
            alarms['DIR-gap+cumconflict'] = gap_half | alarms['DIR-cumconflict']
            for name, alarm in alarms.items():
                ok = scorable.get(name, np.ones(alarm.shape, bool))
                row = {'early_warning_s': grace, 'target': target, 'detector': name, 'run': test['name'],
                       'normal_samples': int(test['normal'].sum()),
                       'false_alarm_fraction': float(alarm[test['normal']].any(axis=1).mean()) if test['normal'].any() else None,
                       'normal_unscorable': int((test['normal'] & ~ok.all(axis=1)).sum())}
                for a, axis in enumerate(AXES):
                    ex = test['exceed'][:, a]
                    row[f'exceed_{axis}'] = int(ex.sum())
                    row[f'unscorable_{axis}'] = int((ex & ~ok[:, a]).sum())  # reported apart, neither caught nor missed
                    row[f'missed_{axis}'] = int((ex & ok[:, a] & ~alarm[:, a]).sum())
                results.append(row)
    return results


def pooled(results):
    table = {}
    for row in results:
        key = (row['early_warning_s'], row['target'], row['detector'])
        t = table.setdefault(key, {'early_warning_s': row['early_warning_s'], 'target': row['target'],
                                   'detector': row['detector'], 'normal': 0, 'false': 0.0,
                                   'normal_unscorable': 0, **{f'exceed_{a}': 0 for a in AXES},
                                   **{f'unscorable_{a}': 0 for a in AXES}, **{f'missed_{a}': 0 for a in AXES}})
        if row['false_alarm_fraction'] is not None:
            t['normal'] += row['normal_samples']
            t['false'] += row['false_alarm_fraction']*row['normal_samples']
            t['normal_unscorable'] += row['normal_unscorable']
        for a in AXES:
            t[f'exceed_{a}'] += row[f'exceed_{a}']
            t[f'unscorable_{a}'] += row[f'unscorable_{a}']
            t[f'missed_{a}'] += row[f'missed_{a}']
    out = []
    for t in table.values():
        entry = {'early_warning_s': t['early_warning_s'], 'target': t['target'], 'detector': t['detector'],
                 'false_alarm_fraction': t['false']/t['normal'] if t['normal'] else None,
                 'normal_unscorable_fraction': t['normal_unscorable']/t['normal'] if t['normal'] else None}
        for a in AXES:  # miss rate over the scorable exceedances; the unscorable share is reported next to it
            scored = t[f'exceed_{a}'] - t[f'unscorable_{a}']
            entry[f'r_miss_{a}'] = t[f'missed_{a}']/scored if scored else None
            entry[f'r_unscorable_{a}'] = t[f'unscorable_{a}']/t[f'exceed_{a}'] if t[f'exceed_{a}'] else None
        out.append(entry)
    return out


def main(argv):
    parser = argparse.ArgumentParser(prog='uwbench detect-a')
    parser.add_argument('--runs', type=Path, nargs='+', required=True, help='run directories (diagnostic OpenVINS build)')
    parser.add_argument('--config', type=Path, help='YAML overriding the pilot defaults')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    cfg = {**CONFIG, **(yaml.safe_load(args.config.read_text()) if args.config else {})}
    runs = [load_run(r, cfg) for r in args.runs]
    results = evaluate(runs, cfg)
    summary = pooled(results)
    args.out.mkdir(parents=True, exist_ok=False)
    with (args.out/'per_run.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    for r in runs:
        s = r['stats']
        header = ['t_s', 'scored', 'nis', 'neg_min_features'] + [f'cov_ratio_{a}' for a in AXES] + \
                 [f'lambda_{a}' for a in AXES] + [f'tau_{a}' for a in AXES] + ['conflict_z_m', 'conflict_yaw_deg',
                                                                                'cum_conflict_z_m', 'cum_conflict_yaw_deg',
                                                                                'cum_conflict_yaw_frozen_deg', 'budget_yaw_deg',
                                                                                'cum_conflict_yaw_truebias_deg',
                                                                                'cum_conflict_sonar_x_m', 'cum_conflict_sonar_y_m',
                                                                                'cum_conflict_sonar_multi_x_m',
                                                                                'cum_conflict_sonar_multi_y_m',
                                                                                'cum_conflict_sonar_map_x_m',
                                                                                'cum_conflict_sonar_map_y_m',
                                                                                'sonar_map_yaw_deg']
        true_yaw = s['conflict_cum_true'][:, 3] if 'conflict_cum_true' in s else np.full(len(r['t']), np.nan)
        sonar_xy = s['conflict_cum_sonar'][:, :2] if 'conflict_cum_sonar' in s else np.full((len(r['t']), 2), np.nan)
        multi_xy = s['conflict_cum_sonar_multi'][:, :2] if 'conflict_cum_sonar_multi' in s else np.full((len(r['t']), 2), np.nan)
        map_xyh = (np.column_stack([s['conflict_cum_sonar_map'][:, :2], s['conflict_cum_sonar_map_yaw'][:, 3]])
                   if 'conflict_cum_sonar_map' in s else np.full((len(r['t']), 3), np.nan))
        cols = np.column_stack([r['t'], r['scored'], s['DET-NIS'], s['DET-FEAT'], s['DET-COV'], s['lambda'], s['tau'],
                                s['conflict'][:, 2:], s['conflict_cum'][:, 2:], s['conflict_cum_frozen'][:, 3],
                                s['budget_yaw_deg'], true_yaw, sonar_xy, multi_xy, map_xyh])
        np.savetxt(args.out/f'stats_{r["name"]}.csv', cols, delimiter=',', header=','.join(header), comments='',
                   fmt=['%.6f', '%d'] + ['%.6g']*(cols.shape[1] - 2))
    tol_yaw = DEFAULTS['risk']['tolerance_yaw_deg']
    horizon = {r['name']: float(brentq(lambda T: drift_budget_deg(T, *r['gyro_noise'], cfg['budget_n_sigma']) - tol_yaw, 1e-6, 1e7))
               for r in runs}
    record = {'pilot': True, 'config': cfg, 'runs': [str(r) for r in args.runs], 'pooled': summary,
              'frozen_budget_reaches_yaw_tolerance_s': horizon,
              'sonar_planes': {r['name']: r['stats'].get('sonar_planes') for r in runs},
              'sonar_map_points': {r['name']: r['stats'].get('sonar_map_points') for r in runs},
              'notes': 'tau in the last LORO fold is what stats_*.csv holds; false alarms are normal-sample fractions'}
    (args.out/'summary.json').write_text(json.dumps(record, indent=2) + '\n')
    for e in summary:
        print(json.dumps(e))
    return 0
