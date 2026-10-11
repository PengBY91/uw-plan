"""Convert one HoloOcean raw run (team/sim/holoocean_generate.py) into an auditable dataset (W2-11).

Writes, next to raw/: ros2/ (MCAP bag), reference/gt_imu.csv (EuRoC layout), openvins/ (three YAMLs derived from
measured socket extrinsics and the configured noise), manifest.yaml (runbook E6 schema) and checksums.sha256.
Raw arrays are only read. Estimator algorithm parameters are copied from configs/euroc_v101 so A0 stays one config.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import yaml
from rosbags.rosbag2 import StoragePlugin, Writer
from rosbags.typesys import Stores, get_typestore
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'team/uwbench'))
from uwbench.mapping import R_BODY_OPTICAL  # noqa: E402

TS = get_typestore(Stores.ROS2_JAZZY)
EPOCH_NS = 1_700_000_000 * 10**9  # sim t=0 maps here; only offsets matter
ATMOSPHERE_PA, RHO, G = 101325.0, 1000.0, 9.81
# Sim A0 baseline on top of configs/euroc_v101. Sim calibration is exact, so online camera calibration is off:
# in sim_pier_truss_v1_s01_a0_r02 it drifted (time offset +62 ms) and amplified a weak-parallax failure to km scale.
SIM_BASELINE = {'gravity_mag': 9.80, 'calib_cam_extrinsics': False, 'calib_cam_intrinsics': False, 'calib_cam_timeoffset': False}
SIM_BASELINE_REASON = 'Sim baseline: gravity 9.80; online camera calibration off because the simulated calibration is exact (pier r02 diagnosis).'


def stamp(t):
    ns = EPOCH_NS + int(round(t*1e9))
    return ns, TS.types['builtin_interfaces/msg/Time'](sec=ns // 10**9, nanosec=ns % 10**9)


def header(t, frame):
    return TS.types['std_msgs/msg/Header'](stamp=stamp(t)[1], frame_id=frame)


def mono(t, frame, image):
    h, w = image.shape
    return TS.types['sensor_msgs/msg/Image'](header=header(t, frame), height=h, width=w, encoding='mono8',
                                             is_bigendian=0, step=w, data=np.ascontiguousarray(image).ravel())


def chunks(raw, kind):
    for path in sorted(raw.glob(f'{kind}_[0-9]*.npy')):
        yield from np.load(path)


def write_bag(raw, out, meta):
    v3 = TS.types['geometry_msgs/msg/Vector3']
    identity = TS.types['geometry_msgs/msg/Quaternion'](x=0.0, y=0.0, z=0.0, w=1.0)
    unknown = np.zeros(9)
    unknown[0] = -1  # ROS convention: orientation not provided
    messages = []
    for row in np.load(raw/'imu.npy'):
        msg = TS.types['sensor_msgs/msg/Imu'](header=header(row[0], 'imu'), orientation=identity, orientation_covariance=unknown,
                                              angular_velocity=v3(x=row[4], y=row[5], z=row[6]), angular_velocity_covariance=np.zeros(9),
                                              linear_acceleration=v3(x=row[1], y=row[2], z=row[3]), linear_acceleration_covariance=np.zeros(9))
        messages.append(('/imu0', row[0], msg))
    for t, z in np.load(raw/'depth.npy'):
        pressure = ATMOSPHERE_PA + RHO*G*max(-z, 0.0)
        messages.append(('/depth/pressure', t, TS.types['sensor_msgs/msg/FluidPressure'](header=header(t, 'depth'), fluid_pressure=pressure, variance=0.0)))
    for kind, topic, frame in (('left', '/cam0/image_raw', 'cam0'), ('right', '/cam1/image_raw', 'cam1')):
        for t, image in zip(np.load(raw/f'{kind}_t.npy'), chunks(raw, kind)):
            messages.append((topic, t, mono(t, frame, image)))
    for t, image in zip(np.load(raw/'sonar_t.npy'), chunks(raw, 'sonar')):
        messages.append(('/sonar/image', t, mono(t, 'sonar', (np.clip(image, 0, 1)*255).round().astype(np.uint8))))
    messages.sort(key=lambda m: (m[1], m[0]))
    with Writer(out, version=9, storage_plugin=StoragePlugin.MCAP) as writer:
        connections = {}
        for topic, t, msg in messages:
            if topic not in connections:
                connections[topic] = writer.add_connection(topic, type(msg).__msgtype__, typestore=TS)
            writer.write(connections[topic], stamp(t)[0], TS.serialize_cdr(msg, type(msg).__msgtype__))
    return {topic: sum(m[0] == topic for m in messages) for topic in connections}


def write_reference(raw, path):
    gt = np.load(raw/'gt_imu.npy')
    T = gt[:, 1:].reshape(-1, 4, 4)
    q = Rotation.from_matrix(T[:, :3, :3]).as_quat()  # x y z w
    rows = np.column_stack([[stamp(t)[0] for t in gt[:, 0]], T[:, :3, 3], q[:, 3], q[:, :3], np.zeros((len(gt), 9))])
    head = '#timestamp, p_RS_R_x [m], p_RS_R_y [m], p_RS_R_z [m], q_RS_w [], q_RS_x [], q_RS_y [], q_RS_z [], ' \
           'v_RS_R_x [m s^-1], v_RS_R_y [m s^-1], v_RS_R_z [m s^-1], b_w_RS_S_x [rad s^-1], b_w_RS_S_y [rad s^-1], ' \
           'b_w_RS_S_z [rad s^-1], b_a_RS_S_x [m s^-2], b_a_RS_S_y [m s^-2], b_a_RS_S_z [m s^-2]'
    path.parent.mkdir(parents=True)
    with path.open('w') as f:
        f.write(head + '\n')
        for r in rows:
            f.write('%d,' % r[0] + ','.join('%.9f' % v for v in r[1:]) + '\n')


def write_openvins(meta, euroc_dir, out):
    s = meta['route']['sensors']
    width, height, fov = s['width'], s['height'], np.radians(s['fov_deg'])
    f = float(width/2/np.tan(fov/2))  # UE horizontal FOV, square pixels, centred principal point
    out.mkdir(parents=True)
    cams = {}
    for i, socket in enumerate(('left', 'right')):
        T = np.asarray(meta['socket_extrinsics_in_imu_frame'][socket])
        T_imu_cam = T.copy()
        T_imu_cam[:3, :3] = T[:3, :3] @ R_BODY_OPTICAL
        cams[f'cam{i}'] = {'T_imu_cam': T_imu_cam.round(9).tolist(), 'cam_overlaps': [1 - i], 'camera_model': 'pinhole',
                           'distortion_coeffs': [0.0, 0.0, 0.0, 0.0], 'distortion_model': 'radtan',
                           'intrinsics': [f, f, width/2, height/2], 'resolution': [width, height], 'rostopic': f'/cam{i}/image_raw'}
    rate = s['imu_hz']
    noise = s['imu_noise']
    imu = {'imu0': {'T_i_b': np.eye(4).tolist(),
                    'accelerometer_noise_density': float(noise['AccelSigma']/np.sqrt(rate)),
                    'accelerometer_random_walk': float(noise['AccelBiasSigma']*np.sqrt(rate)),
                    'gyroscope_noise_density': float(noise['AngVelSigma']/np.sqrt(rate)),
                    'gyroscope_random_walk': float(noise['AngVelBiasSigma']*np.sqrt(rate)),
                    'rostopic': '/imu0', 'time_offset': 0.0, 'update_rate': float(rate), 'model': 'kalibr',
                    'Tw': np.eye(3).tolist(), 'R_IMUtoGYRO': np.eye(3).tolist(), 'Ta': np.eye(3).tolist(),
                    'R_IMUtoACC': np.eye(3).tolist(), 'Tg': np.zeros((3, 3)).tolist()}}
    class Indented(yaml.SafeDumper):  # OpenCV FileStorage wants sequence items indented under their key
        def increase_indent(self, flow=False, indentless=False):
            return super().increase_indent(flow, False)
    dump = lambda d: '%YAML:1.0\n\n' + yaml.dump(d, Dumper=Indented, default_flow_style=None, sort_keys=False)
    (out/'kalibr_imucam_chain.yaml').write_text(dump(cams))
    (out/'kalibr_imu_chain.yaml').write_text(dump(imu))
    estimator = (euroc_dir/'estimator_config.yaml').read_text()
    overrides = {**SIM_BASELINE, **meta['route'].get('estimator_overrides', {}).get('values', {})}
    for key, value in overrides.items():
        value = str(value).lower() if isinstance(value, bool) else value
        estimator, n = re.subn(rf'^{key}:[^#\n]*', f'{key}: {value} ', estimator, flags=re.M)
        if n != 1:
            raise ValueError(f'estimator override {key} matched {n} lines')
    head, rest = estimator.split('\n', 1)  # %YAML directive must stay on the first line for OpenCV
    note = SIM_BASELINE_REASON + ' Route: ' + meta['route'].get('estimator_overrides', {}).get('reason', 'none')
    (out/'estimator_config.yaml').write_text(head + f'\n# Derived from configs/euroc_v101 by scripts/sim_to_bag.py. Overrides: {overrides}. Reason: {note}\n' + rest)
    return {'focal_px': f, 'baseline_m': float(np.linalg.norm(np.subtract(cams['cam0']['T_imu_cam'], cams['cam1']['T_imu_cam'])[:3, 3]))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True, help='data/sim/<run_id> containing raw/')
    parser.add_argument('--lab', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--overrides-from', type=Path, help='take estimator_overrides from this route JSON instead of the generated one')
    parser.add_argument('--restore-bag', action='store_true', help='rebuild ros2/ from raw/ and verify it against checksums.sha256')
    parser.add_argument('--openvins-only', type=Path, help='only write OpenVINS YAMLs to this new directory; bag, reference and audit are reused')
    args = parser.parse_args()
    raw = args.run/'raw'
    meta = json.loads((raw/'meta.json').read_text())
    if args.overrides_from:
        meta['route']['estimator_overrides'] = json.loads(args.overrides_from.read_text())['estimator_overrides']
        meta['route']['estimator_overrides']['source'] = str(args.overrides_from)
    if args.openvins_only:
        print(json.dumps(write_openvins(meta, args.lab/'configs/euroc_v101', args.openvins_only), indent=2))
        return
    if args.restore_bag:
        if (args.run/'ros2').exists():
            parser.error('ros2/ already exists; refusing to overwrite')
        write_bag(raw, args.run/'ros2', meta)
        recorded = dict(line.split()[::-1] for line in (args.run/'checksums.sha256').read_text().splitlines())
        for f in sorted((args.run/'ros2').iterdir()):
            if recorded.get(f'ros2/{f.name}') != hashlib.sha256(f.read_bytes()).hexdigest():
                raise SystemExit(f'restored {f.name} does not match checksums.sha256')
        print('restored ros2/ matches checksums.sha256')
        return
    counts = write_bag(raw, args.run/'ros2', meta)
    write_reference(raw, args.run/'reference'/'gt_imu.csv')
    calibration = write_openvins(meta, args.lab/'configs/euroc_v101', args.run/'openvins')
    route = meta['route']
    manifest = {'schema_version': 1, 'run_id': args.run.name, 'data_root': str(args.run.resolve()), 'split': 'sim_dev',
                'raw_uri': 'ros2', 'calibration_uri': 'openvins/kalibr_imucam_chain.yaml', 'reference_uri': 'reference/gt_imu.csv',
                'clock_model_uri': {'unavailable': 'simulator clock; all streams share tick time, no offset or jitter modelled'},
                'world_frame': f"HoloOcean {route['world']} world (z up)", 'body_frame': 'IMUSocket (z down)',
                'checksums_uri': 'checksums.sha256',
                'streams': {'left': {'topic': '/cam0/image_raw', 'acquisition_clock': 'sim'},
                            'right': {'topic': '/cam1/image_raw', 'acquisition_clock': 'sim', 'pairing_tolerance_s': 0.0},
                            'imu': {'topic': '/imu0', 'units': 'm/s^2, rad/s', 'min_rate_hz': 190},
                            'sonar': {'topic': '/sonar/image', 'geometry': {k: route['sensors']['sonar'][k] for k in ('RangeMin', 'RangeMax', 'Azimuth', 'Elevation', 'RangeBins', 'AzimuthBins')}},
                            'depth': {'topic': '/depth/pressure', 'units': 'Pa absolute; fresh water rho=1000'}},
                'conditions': {'layout': route['name'], 'visibility': 'simulated; see route purpose', 'assumptions': route.get('assumptions')},
                'generator': {'holoocean_version': meta['holoocean_version'], 'counts': meta['counts'], 'wall_seconds': meta['wall_seconds']},
                'derived': {'bag_message_counts': counts, **calibration}}
    files = sorted(p for p in (args.run/'ros2').iterdir()) + [args.run/'reference'/'gt_imu.csv'] + sorted((args.run/'openvins').iterdir())
    (args.run/'checksums.sha256').write_text(''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(args.run)}\n' for p in files))
    (args.run/'manifest.yaml').write_text(yaml.safe_dump(manifest, sort_keys=False))
    print(json.dumps({'counts': counts, **calibration}, indent=2))


if __name__ == '__main__':
    main()
