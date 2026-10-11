"""uwbench eval-traj (P04): manifest reference + estimate -> APE/RPE, time coverage, lost intervals, per-axis exceedance.

Two alignment modes are reported side by side and never mixed:
- public: whole matched trajectory SE(3), no scale (comparable with published APE);
- risk (A plan v2): yaw + translation over a predeclared initial window, window excluded from scoring.
Unalarmed exceedance needs an estimator health field; without one it is not_evaluable, never "normal".
"""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml
from evo.core import metrics
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface
from rosbags.highlevel import AnyReader
from scipy.spatial.transform import Rotation

from uwbench.metrics.axes import AXES, align_yaw_translation, body_axis_errors, body_axis_sigma, exceedance
from uwbench.metrics.trajectory import associate, score_rpe, summarize, time_coverage

DEFAULTS = {'association_tolerance_s': 0.01, 'max_gap_s': 0.075, 'timestamp_offset_s': 0.0,
            'estimate_topic': '/ov_msckf/poseimu', 'reference_valid': None,
            'risk': {'align_window_s': 10.0, 'tolerance_position_m': 0.15, 'tolerance_yaw_deg': 3.0}}


def to_se3(R, p):
    T = np.tile(np.eye(4), (len(p), 1, 1))
    T[:, :3, :3], T[:, :3, 3] = R, p
    return T


def read_reference(path):
    first = path.read_text().lstrip().splitlines()[0]
    traj = (file_interface.read_euroc_csv_trajectory(path) if first.startswith('#') and ',' in first
            else file_interface.read_tum_trajectory_file(path))
    T = np.asarray(traj.poses_se3)
    return np.asarray(traj.timestamps, dtype=float), T[:, :3, :3], T[:, :3, 3]


def read_estimate(path, topic):
    if path.is_file():
        traj = file_interface.read_tum_trajectory_file(path)
        T = np.asarray(traj.poses_se3)
        return np.asarray(traj.timestamps, dtype=float), T[:, :3, :3], T[:, :3, 3], None
    stamps, poses, cov = [], [], []
    with AnyReader([path]) as reader:
        for conn, _, raw in reader.messages(connections=[c for c in reader.connections if c.topic == topic]):
            msg = reader.deserialize(raw, conn.msgtype)
            pose = msg.pose.pose if hasattr(msg.pose, 'pose') else msg.pose
            stamps.append(int(msg.header.stamp.sec) + int(msg.header.stamp.nanosec)*1e-9)
            poses.append([pose.position.x, pose.position.y, pose.position.z,
                          pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
            if hasattr(msg.pose, 'covariance'):
                cov.append(np.asarray(msg.pose.covariance, dtype=float).reshape(6, 6))
    poses = np.asarray(poses).reshape(-1, 7)
    R = Rotation.from_quat(poses[:, 3:]).as_matrix() if len(poses) else np.zeros((0, 3, 3))
    return np.asarray(stamps), R, poses[:, :3], np.asarray(cov) if len(cov) == len(stamps) and cov else None


def evaluate(ref_t, ref_R, ref_p, est_t, est_R, est_p, est_cov=None, config=None):
    cfg = copy.deepcopy(DEFAULTS)
    for key, value in (config or {}).items():
        cfg[key] = {**cfg[key], **value} if isinstance(cfg.get(key), dict) else value
    est_t = np.asarray(est_t, dtype=float) + cfg['timestamp_offset_s']
    if len(est_t) < 2 or not np.isfinite(est_p).all() or not np.all(np.diff(est_t) > 0):
        raise ValueError('Estimate must have at least two finite poses with increasing time; empty output is a failure')
    valid = cfg['reference_valid'] or [[float(ref_t[0]), float(ref_t[-1])]]
    in_valid = np.zeros(len(est_t), bool)
    for a, b in valid:
        in_valid |= (est_t >= a) & (est_t <= b)
    nearest, matched = associate(ref_t, est_t, cfg['association_tolerance_s'])
    matched &= in_valid
    counts = {'estimate': int(len(est_t)), 'matched': int(matched.sum()),
              'outside_reference_valid': int((~in_valid).sum()), 'unmatched_inside_valid': int((in_valid & ~matched).sum())}
    if matched.sum() < 2:
        raise ValueError('Fewer than two estimates matched to valid reference')
    t, R, p = est_t[matched], est_R[matched], est_p[matched]
    rR, rp = ref_R[nearest[matched]], ref_p[nearest[matched]]
    result = {'status': 'scored', 'config': cfg, 'counts': counts,
              'coverage': time_coverage(est_t, valid, cfg['max_gap_s']),
              'initialization_delay_s': float(est_t[in_valid][0] - valid[0][0]) if in_valid.any() else None}

    ref_traj = PoseTrajectory3D(poses_se3=list(to_se3(rR, rp)), timestamps=t)
    est_traj = PoseTrajectory3D(poses_se3=list(to_se3(R, p)), timestamps=t)
    aligned = copy.deepcopy(est_traj)
    aligned.align(ref_traj, correct_scale=False)
    ape = metrics.APE(metrics.PoseRelation.translation_part)
    ape.process_data((ref_traj, aligned))
    public = {'alignment': 'whole matched trajectory SE(3), no scale', 'ape_translation_m': summarize(ape.error)}
    for delta in (1.0, 5.0):
        try:
            rpe = score_rpe(ref_traj, est_traj, delta)
            rpe.pop('pairs')
            public[f'rpe_{int(delta)}s'] = rpe
        except ValueError as exc:
            public[f'rpe_{int(delta)}s'] = {'status': 'not_evaluable', 'reason': str(exc)}
    result['public'] = public

    risk_cfg = cfg['risk']
    window = t <= t[0] + risk_cfg['align_window_s']
    R_align, t_align, alignment = align_yaw_translation(rR, rp, R, p, window)
    R_a, p_a = R_align @ R, p @ R_align.T + t_align
    errors = body_axis_errors(rR, rp, R_a, p_a)
    # Covariance lives in the estimator's own world frame: project with the unaligned attitude (body axes are
    # invariant to the alignment; using R_a with an unrotated covariance swaps x/y sigma at 90 deg alignment yaw)
    sigma = body_axis_sigma(R, est_cov[matched]) if est_cov is not None else None
    scored = ~window
    tolerances = dict(zip(AXES, [risk_cfg['tolerance_position_m']]*3 + [risk_cfg['tolerance_yaw_deg']]))
    axes = {}
    if scored.sum() >= 2:
        for i, axis in enumerate(AXES):
            axes[axis] = {'abs_error_max': float(np.abs(errors[scored, i]).max()),
                          'exceedance': exceedance(t[scored], errors[scored, i], tolerances[axis], cfg['max_gap_s'])}
            if sigma is not None:
                z = np.abs(errors[scored, i])/np.maximum(sigma[scored, i], 1e-12)
                axes[axis]['abs_error_over_sigma'] = {'median': float(np.median(z)), 'p95': float(np.percentile(z, 95))}
    result['risk'] = {'alignment': 'yaw + translation over predeclared initial window; window not scored',
                      'alignment_fit': alignment, 'tolerances': tolerances,
                      'tolerance_status': 'provisional until configs/acceptance_w2.yaml is frozen',
                      'scored_samples': int(scored.sum()), 'axes': axes or 'not_evaluable: nothing after alignment window',
                      'covariance': 'from estimate' if sigma is not None else 'absent',
                      'unalarmed_exceedance': 'not_evaluable: estimate carries no health field'}
    rows = np.column_stack([t, window, errors, sigma if sigma is not None else np.full_like(errors, np.nan)])
    return result, rows


def main(argv):
    parser = argparse.ArgumentParser(prog='uwbench eval-traj')
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--estimate', type=Path, required=True, help='TUM file, or ROS2 bag directory with the estimate topic')
    parser.add_argument('--config', type=Path, help='YAML overriding defaults; acceptance thresholds come from acceptance_w2.yaml')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = yaml.safe_load(args.manifest.read_text())
    config = yaml.safe_load(args.config.read_text()) if args.config else {}
    reference = manifest.get('reference_uri')
    if not isinstance(reference, str):
        parser.error(f'reference unavailable in manifest: {reference}; scoring blocked')
    reference = Path(reference) if Path(reference).is_absolute() else Path(manifest['data_root'])/reference
    args.out.mkdir(parents=True, exist_ok=False)
    record = {'manifest': str(args.manifest.resolve()), 'manifest_sha256': hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
              'reference': str(reference), 'estimate': str(args.estimate.resolve()),
              'acceptance': 'not_evaluated: configs/acceptance_w2.yaml not frozen; diagnostic only'}
    try:
        est = read_estimate(args.estimate, (config or {}).get('estimate_topic', DEFAULTS['estimate_topic']))
        result, rows = evaluate(*read_reference(reference), *est, config=config)
    except ValueError as exc:
        record.update(status='failed', reason=str(exc))
        (args.out/'metrics.json').write_text(json.dumps(record, indent=2)+'\n')
        print(json.dumps(record, indent=2))
        return 1
    record.update(result)
    (args.out/'metrics.json').write_text(json.dumps(record, indent=2)+'\n')
    with (args.out/'axis_errors.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['t_s', 'in_align_window'] + [f'err_{a}' for a in AXES] + [f'sigma_{a}' for a in AXES])
        for row in rows:
            writer.writerow([f'{row[0]:.9f}', int(row[1])] + [f'{v:.6g}' for v in row[2:]])
    with (args.out/'lost_intervals.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['start', 'end', 'duration_s', 'kind'])
        writer.writeheader()
        writer.writerows(result['coverage']['lost_intervals'])
    summary = {'counts': result['counts'], 'coverage_fraction': result['coverage']['coverage_fraction'],
               'ape_rmse_m': result['public']['ape_translation_m']['rmse'],
               'axes_max': {a: v['abs_error_max'] for a, v in result['risk']['axes'].items()} if isinstance(result['risk']['axes'], dict) else None}
    print(json.dumps(summary, indent=2))
    return 0
