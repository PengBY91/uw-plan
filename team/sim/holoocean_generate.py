"""Generate one deterministic-clock HoloOcean run (W2-11). Runs under the Windows Python that has holoocean.

Only holoocean and numpy are required. Time is the simulator clock (tick / ticks_per_sec), never wall time.
Extrinsics are measured from socket PoseSensors at run time; nothing about the vehicle geometry is assumed.
Output (raw, read-only afterwards): meta.json, imu.npy, gt_imu.npy, depth.npy, left/right/sonar chunks and stamps.

usage: python holoocean_generate.py <route.json> <output_dir>
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import holoocean

SOCKETS = {'imu': 'IMUSocket', 'left': 'CameraLeftSocket', 'right': 'CameraRightSocket', 'sonar': 'SonarSocket',
           'depth': 'DepthSocket', 'origin': 'Origin'}
CHUNK = 200  # frames per image chunk file


def scenario(route):
    s = route['sensors']
    camera = lambda name, socket: {'sensor_type': 'CameraSensor', 'sensor_name': name, 'socket': socket, 'Hz': s['camera_hz'],
                                   'configuration': {'CaptureWidth': s['width'], 'CaptureHeight': s['height'], 'FovAngle': s['fov_deg']}}
    sensors = [{'sensor_type': 'PoseSensor', 'sensor_name': f'Pose_{k}', 'socket': v} for k, v in SOCKETS.items()]
    sensors += [{'sensor_type': 'IMUSensor', 'sensor_name': 'IMU', 'socket': 'IMUSocket', 'Hz': s['imu_hz'],
                 'configuration': {**s['imu_noise'], 'ReturnBias': True}},
                {'sensor_type': 'DepthSensor', 'sensor_name': 'Depth', 'socket': 'DepthSocket', 'Hz': s['depth_hz'],
                 'configuration': {'Sigma': s['depth_sigma_m']}},
                camera('Left', 'CameraLeftSocket'), camera('Right', 'CameraRightSocket'),
                {'sensor_type': 'ImagingSonar', 'sensor_name': 'Sonar', 'socket': 'SonarSocket', 'Hz': s['sonar_hz'],
                 'configuration': s['sonar']}]
    start = route['waypoints'][0]
    return {'name': route['name'], 'world': route['world'], 'package_name': route.get('package', 'Ocean'), 'main_agent': 'auv0',
            'ticks_per_sec': route['ticks_per_sec'], 'frames_per_sec': False,
            'agents': [{'agent_name': 'auv0', 'agent_type': 'HoveringAUV', 'control_scheme': 1,
                        'location': start['pose'][:3], 'rotation': start['pose'][3:], 'sensors': sensors}]}


def setpoint(waypoints, t):
    """Piecewise-linear [x, y, z, roll, pitch, yaw] setpoint; yaw interpolated along the short arc."""
    times = [w['t'] for w in waypoints]
    if t <= times[0]:
        return np.array(waypoints[0]['pose'], float)
    if t >= times[-1]:
        return np.array(waypoints[-1]['pose'], float)
    k = int(np.searchsorted(times, t)) - 1
    a, b = np.array(waypoints[k]['pose'], float), np.array(waypoints[k+1]['pose'], float)
    u = (t - times[k])/(times[k+1] - times[k])
    out = a + u*(b - a)
    out[3:] = a[3:] + u*((b[3:] - a[3:] + 180) % 360 - 180)
    return out


def gray(bgra):
    return (bgra[..., :3].astype(np.float32) @ np.array([0.114, 0.587, 0.299], np.float32)).round().clip(0, 255).astype(np.uint8)


def main():
    route = json.loads(Path(sys.argv[1]).read_text())
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=False)
    cfg = scenario(route)
    tps, duration = route['ticks_per_sec'], route['waypoints'][-1]['t']
    imu, gt, depth = [], [], []
    frames = {'left': [], 'right': [], 'sonar': []}
    stamps = {'left': [], 'right': [], 'sonar': []}
    chunk_index = {'left': 0, 'right': 0, 'sonar': 0}
    extrinsics = None

    def flush(kind, final=False):
        if frames[kind] and (final or len(frames[kind]) >= CHUNK):
            np.save(out/f'{kind}_{chunk_index[kind]:04d}.npy', np.stack(frames[kind]))
            chunk_index[kind] += 1
            frames[kind].clear()

    wall0 = time.time()
    with holoocean.make(scenario_cfg=cfg, show_viewport=False, verbose=False) as env:
        for tick in range(int(round(duration*tps)) + 1):
            t = tick/tps
            env.act('auv0', setpoint(route['waypoints'], t))
            s = env.tick()
            T_imu = np.asarray(s['Pose_imu'], float)
            if not np.abs(T_imu).sum():
                raise RuntimeError('Agent did not spawn (all-zero pose); move the first waypoint away from geometry')
            if extrinsics is None:
                inv = np.linalg.inv(T_imu)
                extrinsics = {k: (inv @ np.asarray(s[f'Pose_{k}'], float)).tolist() for k in SOCKETS}
            gt.append(np.concatenate([[t], T_imu.ravel()]))
            if 'IMU' in s:
                m = np.asarray(s['IMU'], float)
                imu.append(np.concatenate([[t], m[0], m[1], m[2], m[3]]))
            if 'Depth' in s:
                depth.append([t, float(np.asarray(s['Depth']).ravel()[0])])
            for kind, key in (('left', 'Left'), ('right', 'Right')):
                if key in s:
                    frames[kind].append(gray(np.asarray(s[key])))
                    stamps[kind].append(t)
                    flush(kind)
            if 'Sonar' in s:
                frames['sonar'].append(np.asarray(s['Sonar'], np.float32))
                stamps['sonar'].append(t)
                flush('sonar')
    for kind in frames:
        flush(kind, final=True)
        np.save(out/f'{kind}_t.npy', np.asarray(stamps[kind]))
    np.save(out/'imu.npy', np.asarray(imu))
    np.save(out/'gt_imu.npy', np.asarray(gt))
    np.save(out/'depth.npy', np.asarray(depth))
    meta = {'route': route, 'scenario': cfg, 'holoocean_version': holoocean.__version__,
            'socket_extrinsics_in_imu_frame': extrinsics,
            'conventions': {'time': 'simulator clock tick/ticks_per_sec, seconds from run start',
                            'imu_frame': 'IMUSocket (NED-like, z down); accel is specific force incl. gravity',
                            'imu_columns': 't, accel(3), gyro(3), accel_bias(3), gyro_bias(3)',
                            'gt': 'world pose of IMUSocket every tick, t + row-major 4x4',
                            'camera_socket_frame': 'x forward, y left, z up; image row 0 is top (UE convention)',
                            'images': 'grayscale from BGRA with 0.114/0.587/0.299',
                            'depth': 'DepthSensor output = world z of DepthSocket in metres (negative below surface)'},
            'counts': {'ticks': len(gt), 'imu': len(imu), 'depth': len(depth), **{k: len(v) for k, v in stamps.items()}},
            'wall_seconds': time.time() - wall0}
    (out/'meta.json').write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta['counts']), 'wall_s', round(meta['wall_seconds'], 1))


if __name__ == '__main__':
    main()
