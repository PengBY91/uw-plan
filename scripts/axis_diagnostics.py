"""Export body-axis error, estimator sigma and exceedance time for an A0 run (A plan v2 scoring preview).

Tolerances default to the provisional 0.15 m / 3 deg from the month-2 runbook; they are frozen in W2,
not here. Alignment uses only a predeclared initial window so that later drift is not redistributed.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from evo.tools import file_interface
from rosbags.highlevel import AnyReader
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'team/uwbench'))
from uwbench.metrics.axes import AXES, align_yaw_translation, body_axis_errors, body_axis_sigma, exceedance


def read_poseimu(bag):
    stamps, poses, covariances = [], [], []
    with AnyReader([bag]) as reader:
        selected = [c for c in reader.connections if c.topic == '/ov_msckf/poseimu']
        for conn, _, raw in reader.messages(connections=selected):
            msg = reader.deserialize(raw, conn.msgtype)
            p, q = msg.pose.pose.position, msg.pose.pose.orientation
            stamps.append(int(msg.header.stamp.sec)*10**9 + int(msg.header.stamp.nanosec))
            poses.append([p.x, p.y, p.z, q.x, q.y, q.z, q.w])
            covariances.append(np.asarray(msg.pose.covariance, dtype=float).reshape(6, 6))
    if len(stamps) < 2:
        raise ValueError('Empty estimate is not a successful baseline')
    poses = np.asarray(poses)
    return np.asarray(stamps, dtype=np.int64)/1e9, Rotation.from_quat(poses[:, 3:]).as_matrix(), poses[:, :3], np.asarray(covariances)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--gt', type=Path, required=True, help='EuRoC-format reference CSV')
    parser.add_argument('--align-window-s', type=float, default=10.0)
    parser.add_argument('--tol-position-m', type=float, default=0.15)
    parser.add_argument('--tol-yaw-deg', type=float, default=3.0)
    parser.add_argument('--max-gap-s', type=float, default=0.075)
    parser.add_argument('--association-tolerance-s', type=float, default=0.01)
    args = parser.parse_args()
    if args.align_window_s <= 0:
        parser.error('Alignment window must be positive')
    out = args.run/'axis-diagnostics'
    out.mkdir(exist_ok=False)

    t, est_R, est_p, covariance = read_poseimu(args.run/'estimate_bag')
    ref = file_interface.read_euroc_csv_trajectory(args.gt)
    k = np.clip(np.searchsorted(ref.timestamps, t), 1, len(ref.timestamps)-1)
    nearest = np.where(np.abs(ref.timestamps[k-1]-t) <= np.abs(ref.timestamps[k]-t), k-1, k)
    matched = np.abs(ref.timestamps[nearest]-t) <= args.association_tolerance_s
    if matched.sum() < 2:
        raise ValueError('Too few estimates matched to the reference')
    ref_se3 = np.asarray(ref.poses_se3)[nearest[matched]]
    t, est_R, est_p, covariance = t[matched], est_R[matched], est_p[matched], covariance[matched]
    ref_R, ref_p = ref_se3[:, :3, :3], ref_se3[:, :3, 3]

    window = t <= t[0] + args.align_window_s
    R_align, t_align, alignment = align_yaw_translation(ref_R, ref_p, est_R, est_p, window)
    est_R, est_p = R_align @ est_R, est_p @ R_align.T + t_align
    errors = body_axis_errors(ref_R, ref_p, est_R, est_p)
    sigma = body_axis_sigma(est_R, covariance)

    tolerances = dict(zip(AXES, [args.tol_position_m]*3 + [args.tol_yaw_deg]))
    scored = ~window  # the window is fitted to the reference and is excluded from scoring
    if scored.sum() < 2:
        raise ValueError('Nothing left to score after the alignment window')
    axes = {}
    for i, axis in enumerate(AXES):
        z = np.abs(errors[scored, i])/np.maximum(sigma[scored, i], 1e-12)
        axes[axis] = {'unit': 'deg' if axis == 'yaw' else 'm',
                      'abs_error_max': float(np.abs(errors[scored, i]).max()),
                      'sigma_median': float(np.median(sigma[scored, i])),
                      'abs_error_over_sigma': {'median': float(np.median(z)), 'p95': float(np.percentile(z, 95)),
                                               'fraction_above_3': float(np.mean(z > 3))},
                      'exceedance': exceedance(t[scored], errors[scored, i], tolerances[axis], args.max_gap_s)}
    report = {'status': 'diagnostic', 'reference': str(args.gt),
              'protocol': {'error_frame': 'estimated body axes; yaw = world-z component of Log(R_est R_ref^T)',
                           'alignment': 'yaw + translation over predeclared initial window, gravity kept, no scale',
                           'align_window_s': args.align_window_s, 'association_tolerance_s': args.association_tolerance_s,
                           'max_gap_s': args.max_gap_s, 'tolerances': tolerances,
                           'tolerance_status': 'provisional runbook initial values; frozen in W2',
                           'covariance_source': '/ov_msckf/poseimu: world position, local IMU orientation error'},
              'counts': {'matched': int(len(t)), 'unmatched': int((~matched).sum()), 'alignment_window': int(window.sum()),
                         'scored': int(scored.sum())},
              'alignment': alignment, 'axes': axes,
              'interpretation': 'Public-data pipeline check of the v2 scoring rule; not underwater evidence and not a detector result.'}
    (out/'axis_diagnostics.json').write_text(json.dumps(report, indent=2)+'\n')
    with (out/'axis_errors.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['t_s', 'in_align_window'] + [f'err_{a}' for a in AXES] + [f'sigma_{a}' for a in AXES])
        for row in zip(t, window, errors, sigma):
            writer.writerow([f'{row[0]:.9f}', int(row[1])] + [f'{v:.6g}' for v in [*row[2], *row[3]]])

    fig, ax = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    for i, axis in enumerate(AXES):
        rel = t - t[0]
        ax[i].plot(rel, errors[:, i], label='error')
        ax[i].fill_between(rel, -3*sigma[:, i], 3*sigma[:, i], alpha=0.25, label='±3σ estimator')
        ax[i].axhline(tolerances[axis], color='r', ls='--', lw=0.8)
        ax[i].axhline(-tolerances[axis], color='r', ls='--', lw=0.8)
        ax[i].axvspan(0, args.align_window_s, color='0.85')
        ax[i].set_ylabel(f"{axis} [{axes[axis]['unit']}]")
    ax[0].legend(loc='upper left')
    ax[-1].set_xlabel('Time since first matched estimate [s] (grey: alignment window)')
    fig.tight_layout(); fig.savefig(out/'axis_errors.png', dpi=140); plt.close(fig)
    print(json.dumps({k: report[k] for k in ['counts', 'alignment']} | {'axes': {a: {
        'max': v['abs_error_max'], 'exceed_s': v['exceedance']['exceed_s'],
        'z_p95': v['abs_error_over_sigma']['p95']} for a, v in axes.items()}}, indent=2))


if __name__ == '__main__':
    main()
