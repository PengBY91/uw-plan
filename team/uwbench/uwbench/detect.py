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
          'sonar_reference_s': 0.0}
DETECTORS = {'DET-NIS': 'global', 'DET-COV': 'axis', 'DET-FEAT': 'global', 'DIR-gap': 'axis', 'DIR-conflict': 'axis',
             'DIR-gap+conflict': 'axis', 'DIR-cumconflict': 'axis', 'DIR-gap+cumconflict': 'axis',
             'DIR-cumconflict-frozen': 'axis', 'DIR-cumconflict-frozen-budget': 'axis', 'ORACLE-cumconflict-truebias': 'axis',
             'DIR-cumconflict+sonar': 'axis'}
CUMULATIVE = {'DIR-cumconflict': 'conflict_cum', 'DIR-cumconflict-frozen': 'conflict_cum_frozen',
              'DIR-cumconflict-frozen-budget': 'conflict_cum_frozen_excess', 'ORACLE-cumconflict-truebias': 'conflict_cum_true',
              'DIR-cumconflict+sonar': 'conflict_cum_sonar'}
SOURCE_AXES = {'conflict_cum_sonar': (0, 1, 2, 3)}  # every other cumulative form: depth for z, gyro for heading only


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
    """Per sonar frame: (t, m_x, m_y, d) of the dominant line, NaN where none."""
    rng = np.random.default_rng(0)
    out = np.full((len(stamps), 4), np.nan)
    out[:, 0] = stamps
    for i, image in enumerate(images):
        line = fit_line(sonar_first_returns(image, geometry, cfg['sonar_threshold']), cfg['sonar_inlier_m'],
                        cfg['sonar_min_inliers'], rng)
        if line is not None:
            out[i, 1:3], out[i, 3] = line
    return out


def sonar_conflict(t, est, lines, T_imu_sonar, gate_deg, reference_s=0.0):
    """Cumulative conflict along a reference plane normal, projected on estimated body x and y, at stamps t.

    The reference plane averages the lines seen in the first reference_s after the first one; zero until it is fixed.
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
            continue
        n_ref, h = ref
        if np.degrees(np.arccos(np.clip(n_w @ n_ref, -1, 1))) > gate_deg:
            continue
        c = abs(d - (h - n_ref @ p_s))
        value_t.append(ts)
        values.append(c*np.abs(R[j].T @ n_ref)[:2])
    out = np.zeros((len(t), 2))
    if values:
        k = np.searchsorted(np.asarray(value_t), t, side='right') - 1
        out[k >= 0] = np.asarray(values)[k[k >= 0]]
    return out


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
                                                                      cfg['sonar_reference_s']),
                                                       z_cum, yaw_cum])
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
    """Sonar lines and extrinsic of a simulated dataset: the raycast replay (props) if present, else the raw images."""
    raw = dataset/'raw'
    if not (raw/'meta.json').exists():
        return None
    meta = json.loads((raw/'meta.json').read_text())
    source = dataset/'sonar_raycast' if (dataset/'sonar_raycast'/'sonar_t.npy').exists() else raw
    stamps = np.load(source/'sonar_t.npy') + SIM_EPOCH_S
    images = np.concatenate([np.load(f) for f in sorted(source.glob('sonar_[0-9]*.npy'))])
    lines = sonar_lines(stamps, images, meta['route']['sensors']['sonar'], cfg)
    return lines, np.asarray(meta['socket_extrinsics_in_imu_frame']['sonar'])


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
                               read_series(bag, '/depth/pressure', 'pressure'), gyro_noise=gyro_noise, true_bias=true_bias,
                               sonar=load_sim_sonar(bag.parent, cfg))
    data['gyro_noise'] = gyro_noise
    return data


def exceed_labels(errors):
    tol = np.array([DEFAULTS['risk']['tolerance_position_m']]*3 + [DEFAULTS['risk']['tolerance_yaw_deg']])
    return np.abs(errors) > tol


def threshold(values, target):
    """Smallest threshold whose alarm fraction (values > thr) on these normal samples is <= target."""
    values = np.asarray(values, dtype=float)
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
            alarms = {}
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
            alarms['DIR-gap+cumconflict'] = gap_half | alarms['DIR-cumconflict']
            for name, alarm in alarms.items():
                row = {'early_warning_s': grace, 'target': target, 'detector': name, 'run': test['name'],
                       'normal_samples': int(test['normal'].sum()),
                       'false_alarm_fraction': float(alarm[test['normal']].any(axis=1).mean()) if test['normal'].any() else None}
                for a, axis in enumerate(AXES):
                    ex = test['exceed'][:, a]
                    row[f'exceed_{axis}'] = int(ex.sum())
                    row[f'missed_{axis}'] = int((ex & ~alarm[:, a]).sum())
                results.append(row)
    return results


def pooled(results):
    table = {}
    for row in results:
        key = (row['early_warning_s'], row['target'], row['detector'])
        t = table.setdefault(key, {'early_warning_s': row['early_warning_s'], 'target': row['target'],
                                   'detector': row['detector'], 'normal': 0, 'false': 0.0,
                                   **{f'exceed_{a}': 0 for a in AXES}, **{f'missed_{a}': 0 for a in AXES}})
        if row['false_alarm_fraction'] is not None:
            t['normal'] += row['normal_samples']
            t['false'] += row['false_alarm_fraction']*row['normal_samples']
        for a in AXES:
            t[f'exceed_{a}'] += row[f'exceed_{a}']
            t[f'missed_{a}'] += row[f'missed_{a}']
    out = []
    for t in table.values():
        entry = {'early_warning_s': t['early_warning_s'], 'target': t['target'], 'detector': t['detector'],
                 'false_alarm_fraction': t['false']/t['normal'] if t['normal'] else None}
        for a in AXES:
            entry[f'r_miss_{a}'] = t[f'missed_{a}']/t[f'exceed_{a}'] if t[f'exceed_{a}'] else None
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
                                                                                'cum_conflict_sonar_x_m', 'cum_conflict_sonar_y_m']
        true_yaw = s['conflict_cum_true'][:, 3] if 'conflict_cum_true' in s else np.full(len(r['t']), np.nan)
        sonar_xy = s['conflict_cum_sonar'][:, :2] if 'conflict_cum_sonar' in s else np.full((len(r['t']), 2), np.nan)
        cols = np.column_stack([r['t'], r['scored'], s['DET-NIS'], s['DET-FEAT'], s['DET-COV'], s['lambda'], s['tau'],
                                s['conflict'][:, 2:], s['conflict_cum'][:, 2:], s['conflict_cum_frozen'][:, 3],
                                s['budget_yaw_deg'], true_yaw, sonar_xy])
        np.savetxt(args.out/f'stats_{r["name"]}.csv', cols, delimiter=',', header=','.join(header), comments='',
                   fmt=['%.6f', '%d'] + ['%.6g']*(cols.shape[1] - 2))
    tol_yaw = DEFAULTS['risk']['tolerance_yaw_deg']
    horizon = {r['name']: float(brentq(lambda T: drift_budget_deg(T, *r['gyro_noise'], cfg['budget_n_sigma']) - tol_yaw, 1e-6, 1e7))
               for r in runs}
    record = {'pilot': True, 'config': cfg, 'runs': [str(r) for r in args.runs], 'pooled': summary,
              'frozen_budget_reaches_yaw_tolerance_s': horizon,
              'notes': 'tau in the last LORO fold is what stats_*.csv holds; false alarms are normal-sample fractions'}
    (args.out/'summary.json').write_text(json.dumps(record, indent=2) + '\n')
    for e in summary:
        print(json.dumps(e))
    return 0
