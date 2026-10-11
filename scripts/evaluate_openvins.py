"""Evaluate poseimu with evo and explicit time-based RPE pair selection."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from evo.core import lie_algebra, metrics, sync
from evo.tools import file_interface
from rosbags.highlevel import AnyReader


def summarize(values):
    a = np.asarray(values, dtype=float)
    if a.size == 0 or not np.isfinite(a).all():
        raise ValueError('Cannot score empty or non-finite errors')
    return {'count': int(a.size), 'rmse': float(np.sqrt(np.mean(a*a))),
            'mean': float(a.mean()), 'median': float(np.median(a)),
            'p95': float(np.percentile(a, 95)), 'max': float(a.max())}


def time_pairs(timestamps, delta, tolerance=0.025):
    times = np.asarray(timestamps)
    if len(times) < 2 or not np.isfinite(times).all() or not np.all(np.diff(times) > 0):
        raise ValueError('Time pairing requires a nonempty increasing trajectory')
    if delta <= 0 or tolerance < 0:
        raise ValueError('Invalid pair interval')
    pairs = []
    for i, t in enumerate(times):
        target = t + delta
        k = int(np.searchsorted(times, target))
        choices = [j for j in (k-1, k) if i < j < len(times)]
        if choices:
            j = min(choices, key=lambda j: abs(times[j]-target))
            if abs(times[j]-target) <= tolerance:
                pairs.append((i, j))
    return pairs


def score_rpe(ref, est, delta):
    pairs = time_pairs(est.timestamps, delta)
    if not pairs:
        raise ValueError('No valid time-based RPE pairs')
    values, angles, dts, ref_dts = [], [], [], []
    for i, j in pairs:
        error = metrics.RPE.rpe_base(ref.poses_se3[i], ref.poses_se3[j],
                                     est.poses_se3[i], est.poses_se3[j])
        values.append(np.linalg.norm(error[:3, 3]))
        angles.append(lie_algebra.so3_log_angle(error[:3, :3], degrees=True))
        dts.append(est.timestamps[j] - est.timestamps[i])
        ref_dts.append(ref.timestamps[j] - ref.timestamps[i])
    possible = int(np.sum(est.timestamps + delta <= est.timestamps[-1] + 0.025))
    return {'translation_m': summarize(values), 'rotation_deg': summarize(angles),
            'target_interval_s': delta, 'pair_tolerance_s': 0.025,
            'actual_est_interval_s': [float(min(dts)), float(max(dts))],
            'actual_ref_interval_s': [float(min(ref_dts)), float(max(ref_dts))],
            'eligible_start_count': possible, 'selected_pair_count': len(pairs),
            'pairs': pairs}


def inspect_outputs(bag):
    outputs = {}
    with AnyReader([bag]) as reader:
        for c, _, data in reader.messages():
            if c.topic not in ['/ov_msckf/poseimu', '/ov_msckf/odomimu']:
                continue
            msg = reader.deserialize(data, c.msgtype)
            pose = msg.pose.pose
            values = [pose.position.x, pose.position.y, pose.position.z,
                      pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
            if not np.isfinite(values).all():
                raise ValueError('Non-finite output pose')
            norm = float(np.linalg.norm(values[3:]))
            if abs(norm-1) > 1e-5:
                raise ValueError('Invalid quaternion norm')
            stamp = int(msg.header.stamp.sec)*10**9 + int(msg.header.stamp.nanosec)
            outputs.setdefault(c.topic, {'timestamps_ns': [], 'quaternion_norms': []})
            outputs[c.topic]['timestamps_ns'].append(stamp)
            outputs[c.topic]['quaternion_norms'].append(norm)
    assert all(t in outputs for t in ['/ov_msckf/poseimu', '/ov_msckf/odomimu'])
    for item in outputs.values():
        ts = np.array(item.pop('timestamps_ns'), dtype=np.int64)
        norms = item.pop('quaternion_norms')
        assert len(ts) > 1 and np.all(np.diff(ts) > 0)
        gaps = np.diff(ts)/1e9
        item.update(count=int(len(ts)), first_timestamp_ns=int(ts[0]), last_timestamp_ns=int(ts[-1]),
                    median_interval_s=float(np.median(gaps)), max_interval_s=float(gaps.max()),
                    quaternion_norm_range=[min(norms), max(norms)], finite=True, strictly_monotonic=True)
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--gt-original', type=Path, required=True)
    parser.add_argument('--gt-corrected', type=Path, required=True)
    args = parser.parse_args()
    export = args.run/'evaluation'
    export.mkdir(exist_ok=False)
    outputs = inspect_outputs(args.run/'estimate_bag')
    cmd = [str(Path(sys.executable).parent/'evo_traj'), 'bag2', str(args.run/'estimate_bag'),
           '/ov_msckf/poseimu', '--save_as_tum']
    p = subprocess.run(cmd, cwd=export, capture_output=True, text=True)
    (export/'export.log').write_text(repr(cmd)+'\n'+p.stdout+p.stderr)
    p.check_returncode()
    files = list(export.glob('*.tum'))
    assert len(files) == 1
    est = file_interface.read_tum_trajectory_file(files[0])
    if est.num_poses < 2:
        raise ValueError('Empty estimate is not a successful baseline')
    camera_ns = np.array(json.loads(args.audit.read_text())['source']['camera_stamps_ns'], dtype=np.int64)
    camera = camera_ns / 1e9
    matched_camera = set()
    for t in est.timestamps:
        k = int(np.searchsorted(camera, t))
        candidates = [j for j in (k-1, k) if 0 <= j < len(camera)]
        j = min(candidates, key=lambda j: abs(camera[j]-t))
        if abs(camera[j]-t) <= 0.025:
            matched_camera.add(j)
    gaps = np.diff(est.timestamps)
    gap_indices = np.where(gaps > 0.075)[0]
    coverage = {'input_stereo_pair_count': int(len(camera)), 'estimate_count': est.num_poses,
                'camera_slots_with_estimate': len(matched_camera),
                'camera_slot_coverage': len(matched_camera)/len(camera),
                'camera_match_tolerance_s': 0.025,
                'first_estimate_delay_s': float(est.timestamps[0]-camera[0]),
                'last_estimate_tail_s': float(camera[-1]-est.timestamps[-1]),
                'post_start_gaps_above_75ms': [{'start': float(est.timestamps[i]),
                    'end': float(est.timestamps[i+1]), 'duration_s': float(gaps[i])} for i in gap_indices],
                'interpretation': 'output-message coverage; not an accuracy-gated localization availability metric'}
    report = {'status': 'scored', 'outputs': outputs, 'coverage': coverage,
              'protocol': {'ape_alignment': 'full matched trajectory SE3, no scale',
                           'association_tolerance_s': 0.01, 'timestamp_offset_s': 0.0,
                           'primary_reference': 'OpenVINS corrected V1_01_easy',
                           'secondary_reference': 'ETH original ASL state_groundtruth_estimate0',
                           'rpe_engine': 'evo.metrics.RPE.rpe_base, timestamp pairs, no fitted scale'},
              'references': {}}
    for label, path in [('corrected', args.gt_corrected), ('original', args.gt_original)]:
        ref = file_interface.read_euroc_csv_trajectory(path)
        synced_ref, synced_est = sync.associate_trajectories(ref, est, max_diff=0.01)
        aligned = copy.deepcopy(synced_est)
        aligned.align(synced_ref, correct_scale=False)
        metric = metrics.APE(metrics.PoseRelation.translation_part)
        metric.process_data((synced_ref, aligned))
        result = {'reference_poses': ref.num_poses, 'matched_estimate_count': synced_est.num_poses,
                  'estimate_match_fraction': synced_est.num_poses/est.num_poses,
                  'max_timestamp_difference_s': float(np.max(np.abs(synced_ref.timestamps-synced_est.timestamps))),
                  'ape_translation_m': summarize(metric.error),
                  'rpe_1s': score_rpe(synced_ref, synced_est, 1.0),
                  'rpe_5s': score_rpe(synced_ref, synced_est, 5.0)}
        command = [str(Path(sys.executable).parent/'evo_ape'), 'euroc', str(path), str(files[0]),
                   '--align', '--t_max_diff', '0.01', '--save_results', str(export/(label+'_ape_se3.zip'))]
        p = subprocess.run(command, capture_output=True, text=True)
        (export/(label+'_ape.log')).write_text(repr(command)+'\n'+p.stdout+p.stderr)
        p.check_returncode()
        report['references'][label] = result
        if label == 'corrected':
            fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
            ax[0].plot(synced_ref.positions_xyz[:, 0], synced_ref.positions_xyz[:, 1], label='Corrected reference')
            ax[0].plot(aligned.positions_xyz[:, 0], aligned.positions_xyz[:, 1], label='OpenVINS (SE3 aligned)', alpha=0.8)
            ax[0].set(xlabel='x [m]', ylabel='y [m]', title='V1_01_easy stereo-inertial baseline')
            ax[0].axis('equal'); ax[0].legend()
            ax[1].plot(synced_est.timestamps-synced_est.timestamps[0], metric.error)
            ax[1].set(xlabel='Time since first matched estimate [s]', ylabel='Translation APE [m]', title='No scale fitting')
            fig.tight_layout(); fig.savefig(export/'trajectory_ape.png', dpi=160); plt.close(fig)
    (export/'metrics.json').write_text(json.dumps(report, indent=2)+'\n')
    compact = copy.deepcopy(report)
    for value in compact['references'].values():
        for key in ['rpe_1s', 'rpe_5s']:
            value[key].pop('pairs')
    print(json.dumps(compact, indent=2))


if __name__ == '__main__':
    main()
