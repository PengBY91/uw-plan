"""uwbench run-a: manifest + A config -> A1 trajectory (TUM, readable by eval-traj), factor counts, residual statistics.

Simulated datasets only for now (raw/ HoloOcean layout); real D1/D2 data needs the real sonar mapping and
correspondences (W3-03) and is rejected rather than approximated.
"""
import argparse
import csv
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import yaml

KINDS = {'GenericStereoFactor3D': 'stereo', 'CombinedImuFactor': 'imu', 'CustomFactor': 'sonar'}


def _git(path):
    try:
        commit = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
        dirty = bool(subprocess.check_output(['git', '-C', str(path), 'status', '--porcelain'], text=True).strip())
        return {'commit': commit, 'dirty': dirty}
    except Exception:
        return {'commit': None, 'dirty': None}


def segment_tracks(rows, keyframe_times, max_gap_s):
    """Split oracle landmark ids into continuous tracks: a landmark unseen for more than max_gap_s starts a new id,
    so sonar gets no revisit (loop-closure) association that the KLT front end does not have. Returns rows with the
    track id in column 1 and the landmark id appended."""
    out, last, track_of, next_id = [], {}, {}, 0
    for k, lid, rho, theta, outlier in sorted(rows, key=lambda r: (r[0], r[1])):
        t = keyframe_times[int(k)]
        if lid not in track_of or (max_gap_s is not None and t - last[lid] > max_gap_s + 1e-9):
            track_of[lid], next_id = next_id, next_id + 1
        last[lid] = t
        out.append((int(k), track_of[lid], rho, theta, outlier, int(lid)))
    return out


def tracks_for(ds, keyframe_frames, path, params):
    from uwbench.a.backend import frontend
    p = dict(frontend.DEFAULTS, **(params or {}))
    if path.exists():
        data = np.load(path)
        meta = json.loads(str(data['meta']))
        if meta['params'] != p or meta['frames'] != [int(f) for f in keyframe_frames] or meta['dataset'] != str(ds.root):
            raise SystemExit('Existing front-end cache %s was built with different inputs; use a new path' % path)
        return data['rows'], meta
    rows, p = frontend.track(ds, keyframe_frames, p)
    meta = {'dataset': str(ds.root), 'frames': [int(f) for f in keyframe_frames], 'params': p,
            'params_hash': frontend.params_hash(p), 'code': _git(Path(__file__).parent),
            'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, rows=rows, meta=json.dumps(meta))
    return rows, meta


def whitened(f, values):
    model = f.noiseModel()
    import gtsam
    base = model.noise() if isinstance(model, gtsam.noiseModel.Robust) else model
    return float(np.linalg.norm(base.whiten(f.unwhitenedError(values))))


def stats(x):
    x = np.asarray(x, dtype=float)
    if not len(x):
        return {'count': 0}
    return {'count': int(len(x)), 'mean': float(x.mean()), 'p50': float(np.median(x)), 'p95': float(np.percentile(x, 95)),
            'max': float(x.max())}


def write_tum(path, poses):
    from scipy.spatial.transform import Rotation
    with open(path, 'w') as f:
        for t, T in poses:
            q = Rotation.from_matrix(T[:3, :3]).as_quat()
            f.write('%.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f\n' % (t, *T[:3, 3], *q))


def main(argv, after_keyframe=None):
    """after_keyframe(sm, k, t, tracks): read-only hook after each keyframe update (uwbench diagnose-a)."""
    parser = argparse.ArgumentParser(prog='uwbench run-a', description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True, help='A1 YAML (configs/a1/*.yaml)')
    parser.add_argument('--tracks', type=Path, required=True, help='Shared front-end cache (.npz); created if missing')
    parser.add_argument('--out', type=Path, required=True, help='New run directory')
    parser.add_argument('--max-keyframes', type=int, help='Stop early (tests and smoke runs only)')
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error('Refusing to overwrite ' + str(args.out))
    from uwbench.a.backend import simdata, smoother
    cfg_file = yaml.safe_load(args.config.read_text()) or {}
    cfg = smoother.merge(smoother.DEFAULTS, cfg_file.get('backend'))
    manifest = yaml.safe_load(args.manifest.read_text())
    if not (Path(manifest['data_root'])/'raw/meta.json').exists():
        parser.error('run-a supports simulated datasets only; real data needs the W3-03 sonar mapping and correspondences')
    ds = simdata.SimDataset(args.manifest)
    stride = cfg['keyframe_stride']
    frames = np.arange(0, len(ds.left_t), stride)
    times = ds.left_t[frames]
    wall0 = time.perf_counter()
    tracks, tracks_meta = tracks_for(ds, frames, args.tracks, cfg_file.get('frontend'))  # always the full sequence
    if args.max_keyframes:
        frames, times = frames[:args.max_keyframes], times[:args.max_keyframes]
    sonar_cfg = cfg['sonar']
    sonar_rows, landmarks, sonar_info = [], np.zeros((0, 3)), {'enabled': sonar_cfg['enabled']}
    if sonar_cfg['enabled']:
        obs_cfg = cfg_file.get('sonar_observations', {})
        height = simdata.nominal_sonar_height(ds)
        landmarks, skipped = simdata.edge_landmarks(ds.route['props'], height)
        raw_rows, in_sector = simdata.sonar_observations(ds, landmarks, times, obs_cfg.get('visibility_radius_m', 0.03),
                                                         obs_cfg.get('outlier_rate', 0.0), obs_cfg.get('seed', 0))
        sonar_rows = segment_tracks(raw_rows, times, sonar_cfg['max_gap_s'])
        sonar_info.update({'association': 'ORACLE (scene geometry); controlled experiment, not automatic sonar SLAM',
                           'landmarks': 'convex vertical prop edges at world z = %.3f m (nominal sonar height)' % height,
                           'landmark_count': int(len(landmarks)), 'skipped_props': skipped,
                           'observations': len(sonar_rows), 'injected_outliers': int(sum(r[4] for r in sonar_rows)),
                           'tracks': len({r[1] for r in sonar_rows}), 'in_sector_landmark_frames': int(sum(in_sector)),
                           'measurement': 'nearest raycast return within visibility radius, quantised to image bin centres',
                           'observation_config': obs_cfg, 'max_gap_s': sonar_cfg['max_gap_s'],
                           'vertical_aperture_deg': float(ds.sonar_geometry['Elevation']),
                           'aperture_note': 'simulation setting, true in simulation; not the platform value'})
    sm = smoother.Smoother(cfg, ds.K, ds.baseline, ds.T_imu_cam0, ds.T_imu_sonar, ds.imu_noise,
                          np.radians(ds.sonar_geometry['Elevation']))
    by_kf = {}
    for r in tracks:
        by_kf.setdefault(int(r[0]), []).append(r[1:])
    sonar_by_kf = {}
    for r in sonar_rows:
        sonar_by_kf.setdefault(r[0], []).append((r[1], r[2], r[3], r[4]))
    imu_t = ds.imu[:, 0]
    status, error = 'completed', None
    try:
        for k, t in enumerate(times):
            seg = np.zeros((0, 7))
            if k:
                sel = (imu_t >= times[k - 1] - 1e-9) & (imu_t < t - 1e-9)
                rows = ds.imu[sel]
                dt = np.diff(np.r_[rows[:, 0], t])
                seg = np.column_stack([dt, rows[:, 1:4], rows[:, 4:7]])
            init = (ds.gt_pose(t), ds.gt_velocity(t)) if k == 0 else None
            sm.add_keyframe(float(t), seg, by_kf.get(k, []), sonar_by_kf.get(k, []), init)
            if after_keyframe is not None:
                after_keyframe(sm, k, float(t), by_kf.get(k, []))
    except Exception as exc:  # keep the failure evidence; a failed run is a result
        status, error = 'failed', repr(exc)
    args.out.mkdir(parents=True)
    write_tum(args.out/'trajectory_causal.tum', [(simdata.EPOCH_S + t, T) for t, T in sm.poses_causal])
    summary = {'status': status, 'error': error, 'command': ['uwbench', 'run-a'] + list(argv),
               'start_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'wall_s': time.perf_counter() - wall0, 'code': _git(Path(__file__).parent),
               'config_file': str(args.config), 'config_sha256': hashlib.sha256(args.config.read_bytes()).hexdigest(),
               'config_resolved': cfg, 'manifest': str(args.manifest),
               'tracks': {'path': str(args.tracks), 'sha256': simdata.file_sha256(args.tracks), 'rows': int(len(tracks)),
                          'params_hash': tracks_meta['params_hash']},
               'initialisation': 'keyframe 0 pose and velocity from ground truth (tight prior), zero bias prior; identical for all configs',
               'keyframes': {'stride': stride, 'count': len(sm.poses_causal), 'planned': int(len(times))},
               'sonar': sonar_info,
               'output_poses': 'trajectory_causal.tum: each keyframe pose right after its own update; '
                               'trajectory_smoothed.tum: final iSAM2 estimate'}
    if sm.poses_causal:
        import gtsam
        from gtsam.symbol_shorthand import X, S
        est = sm.isam.calculateEstimate()
        write_tum(args.out/'trajectory_smoothed.tum',
                  [(simdata.EPOCH_S + t, est.atPose3(X(k)).matrix()) for k, (t, _) in enumerate(sm.poses_causal)])
        graph = sm.isam.getFactorsUnsafe()
        res = {'stereo': [], 'imu': [], 'sonar': [], 'prior': []}
        for i in range(graph.size()):
            f = graph.at(i)
            if f is None:
                continue
            kind = KINDS.get(type(f).__name__, 'prior')
            try:
                res[kind].append(whitened(f, est))
            except Exception:
                res[kind].append(float(np.sqrt(2*f.error(est))))
        summary['factor_totals'] = {k: len(v) for k, v in res.items()}
        summary['whitened_residual_final'] = {k: stats(v) for k, v in res.items()}
        if sm.sonar_factors:
            from uwbench.a.backend import factors
            from uwbench.sonar import elevation
            el, inl, out = [], [], []
            for kk, lid, f, outlier, _ in sm.sonar_factors:
                p_s, _, _, _ = factors.predict(est.atPose3(X(kk)), sm.T_bs, est.atPoint3(S(lid)))
                el.append(float(elevation(p_s)))
                (out if outlier else inl).append(whitened(f, est))
            half = np.radians(ds.sonar_geometry['Elevation'])/2
            truth = {r[1]: landmarks[r[5]] for r in sonar_rows}
            err = [float(np.linalg.norm(est.atPoint3(S(l)) - truth[l])) for l in sm.sonar_added]
            summary['sonar_diagnostics'] = {
                'landmarks_initialised': len(sm.sonar_added), 'landmarks_rejected': len(sm.sonar_rejected),
                'landmarks_pending': len(sm.sonar_buffer),
                'elevation_outside_aperture_fraction': float(np.mean(np.abs(el) > half + 1e-9)),
                'whitened_residual_inliers': stats(inl), 'whitened_residual_injected_outliers': stats(out),
                'landmark_position_error_m': stats(err),
                'weights': dict(stats([w for *_, w in sm.sonar_factors]), policy=sm.weighting.cfg,
                                at_minimum_fraction=float(np.mean([w <= sm.weighting.cfg['w_min'] + 1e-12 for *_, w in sm.sonar_factors])))}
    with open(args.out/'factors.csv', 'w', newline='') as f:
        if sm.counts:
            w = csv.DictWriter(f, fieldnames=list(sm.counts[0]))
            w.writeheader()
            w.writerows(sm.counts)
    totals = {k: int(sum(c[k] for c in sm.counts)) for k in ['stereo_factors', 'sonar_factors', 'new_visual_landmarks',
                                                             'new_sonar_landmarks', 'imu_samples']} if sm.counts else {}
    summary['factor_counts'] = totals
    summary['zero_sonar_factor_keyframes'] = int(sum(c['sonar_factors'] == 0 for c in sm.counts)) if sm.counts else 0
    summary['solve_s'] = stats([c['solve_s'] for c in sm.counts])
    summary['isam_rebuilds'] = sm.rebuilds   # recoveries from indeterminate iSAM2 updates (keyframe, stage)
    (args.out/'summary.json').write_text(json.dumps(summary, indent=2, default=float) + '\n')
    print(json.dumps({k: summary.get(k) for k in ['status', 'error', 'wall_s', 'keyframes', 'factor_counts', 'sonar_diagnostics', 'isam_rebuilds']},
                     indent=2, default=float))
    return 0 if status == 'completed' else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
