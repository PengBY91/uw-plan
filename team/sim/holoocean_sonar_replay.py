"""Replay a finished run's reference trajectory through the raycast sonar (W2-11). Runs under the Windows Python.

HoloOcean's ImagingSonar does not see spawned props (they are not in the world octree), so prop scenes recorded with
it have empty sonar. This script re-creates the scene (world + props), teleports the vehicle to the recorded IMU pose
on every simulator tick and records only the raycast sonar of holoocean_generate.py: same sweep per frame, same frame
stamps as the original run, same rendering. Cameras and IMU are not touched, so estimator runs on the original raw
data stay valid. Teleport accuracy is measured from a PoseSensor at the IMU socket and written to meta.json.

usage: python holoocean_sonar_replay.py <raw_dir> <output_dir>
Output: meta.json, sonar_XXXX.npy chunks, sonar_t.npy, sonar_points.npy (as holoocean_generate.py with 'raycast').
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import holoocean

from holoocean_generate import CHUNK, raycast_config, render_sonar


def euler_deg(R):
    """[roll, pitch, yaw] in degrees for R = Rz(yaw) Ry(pitch) Rx(roll)."""
    return np.degrees([np.arctan2(R[2, 1], R[2, 2]), -np.arcsin(np.clip(R[2, 0], -1, 1)), np.arctan2(R[1, 0], R[0, 0])])


def main():
    raw, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=False)
    meta = json.loads((raw/'meta.json').read_text())
    route = meta['route']
    s = {**route['sensors'], 'sonar_model': 'raycast'}
    tps = route['ticks_per_sec']
    gt = np.load(raw/'gt_imu.npy')
    frame_t = np.load(raw/'sonar_t.npy')
    T_imu_origin = np.asarray(meta['socket_extrinsics_in_imu_frame']['origin'])
    period = int(round(tps/s['sonar_hz']))
    T0 = gt[0, 1:].reshape(4, 4) @ T_imu_origin
    cfg = {'name': route['name'] + '_sonar_replay', 'world': route['world'], 'package_name': route.get('package', 'Ocean'),
           'main_agent': 'auv0', 'ticks_per_sec': tps, 'frames_per_sec': False,
           'agents': [{'agent_name': 'auv0', 'agent_type': 'HoveringAUV', 'control_scheme': 0,
                       'location': T0[:3, 3].tolist(), 'rotation': euler_deg(T0[:3, :3]).tolist(),
                       'sensors': [{'sensor_type': 'PoseSensor', 'sensor_name': 'Pose_imu', 'socket': 'IMUSocket'},
                                   {'sensor_type': 'RaycastLidar', 'sensor_name': 'SonarRays', 'socket': 'SonarSocket', 'Hz': tps,
                                    'configuration': raycast_config(s, tps)}]}]}
    rng = np.random.default_rng(s.get('raycast_noise_seed', 0))
    frames, stamps, points, sweep, chunk = [], [], [], [], 0
    pos_err, rot_err = [], []
    wall0 = time.time()
    with holoocean.make(scenario_cfg=cfg, show_viewport=False, verbose=False) as env:
        for prop in route.get('props', []):
            env.spawn_prop(prop['type'], location=prop['location'], rotation=prop.get('rotation', [0, 0, 0]),
                           scale=prop['scale'], material=prop.get('material', ''))
        for _ in range(route.get('pre_ticks', 20)):
            env.agents['auv0'].teleport(T0[:3, 3], euler_deg(T0[:3, :3]))
            env.tick()
        for tick, row in enumerate(gt):
            t = row[0]
            T = row[1:].reshape(4, 4) @ T_imu_origin
            env.agents['auv0'].teleport(T[:3, 3], euler_deg(T[:3, :3]))
            env.act('auv0', np.zeros(8))
            state = env.tick()
            P = np.asarray(state['Pose_imu'], float)
            G = row[1:].reshape(4, 4)
            pos_err.append(np.linalg.norm(P[:3, 3] - G[:3, 3]))
            rot_err.append(np.degrees(np.arccos(np.clip((np.trace(G[:3, :3].T @ P[:3, :3]) - 1)/2, -1, 1))))
            if 'SonarRays' in state:
                hits = np.asarray(state['SonarRays'], np.float32).reshape(-1, 5)[:, :3]
                sweep.append(hits[np.abs(hits).sum(axis=1) > 0])
            if tick % period == 0:
                if len(stamps) >= len(frame_t) or abs(frame_t[len(stamps)] - t) > 0.5/tps:
                    raise RuntimeError(f'frame {len(stamps)} at t={t} does not match the original sonar stamps')
                hits = np.concatenate(sweep) if sweep else np.zeros((0, 3), np.float32)
                points.append(np.column_stack([np.full(len(hits), len(stamps), np.float32), hits]))
                frames.append(render_sonar(hits, s['sonar'], s.get('raycast_rays_per_bin', 2), rng))
                stamps.append(t)
                sweep.clear()
                if len(frames) >= CHUNK:
                    np.save(out/f'sonar_{chunk:04d}.npy', np.stack(frames))
                    chunk += 1
                    frames.clear()
    if frames:
        np.save(out/f'sonar_{chunk:04d}.npy', np.stack(frames))
    np.save(out/'sonar_t.npy', np.asarray(stamps))
    np.save(out/'sonar_points.npy', np.concatenate(points))
    record = {'source_raw': str(raw), 'route_name': route['name'], 'sonar_model': 'raycast', 'sensors': s,
              'raycast_config': raycast_config(s, tps), 'holoocean_version': holoocean.__version__,
              'teleport_error_imu': {'position_m_p50': float(np.median(pos_err)), 'position_m_max': float(np.max(pos_err)),
                                     'rotation_deg_p50': float(np.median(rot_err)), 'rotation_deg_max': float(np.max(rot_err))},
              'conventions': {'sonar_points': 'frame index into sonar_t, then hit x y z in the sonar socket frame '
                                              '(x forward, y left, z up); hits from ticks in (t - 1/sonar_hz, t]',
                              'images': '[RangeBins, AzimuthBins], row 0 at RangeMin, column 0 at -Azimuth/2 (right)'},
              'counts': {'ticks': len(gt), 'sonar': len(stamps)}, 'wall_seconds': time.time() - wall0}
    (out/'meta.json').write_text(json.dumps(record, indent=2))
    print(json.dumps(record['counts']), json.dumps(record['teleport_error_imu']), 'wall_s', round(record['wall_seconds'], 1))


if __name__ == '__main__':
    main()
