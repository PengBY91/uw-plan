"""What detect-a reads from a finished run: [DIAG] lines of the diagnostic OpenVINS build, bag series, labels and
tolerances from eval-traj, the IMU noise model and (in simulation) the true gyro bias."""
import json
import re
from pathlib import Path

import numpy as np
import yaml
from rosbags.highlevel import AnyReader

from uwbench.a.detect.config import SIM_EPOCH_S, default_tolerances
from uwbench.eval_traj import DEFAULTS, read_estimate
from uwbench.metrics.axes import AXES


DIAG = re.compile(r'\[DIAG\] (MSCKF|SLAM) (.*)$')


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


def read_tolerances(run_dir):
    """Per-axis tolerances the labels were scored with (eval-traj/metrics.json), else the shared defaults."""
    metrics = Path(run_dir)/'eval-traj'/'metrics.json'
    if metrics.exists():
        tol = json.loads(metrics.read_text()).get('risk', {}).get('tolerances')
        if tol:
            return np.array([float(tol[a]) for a in AXES])
    return default_tolerances()


def load_inputs(run_dir):
    """Everything detect-a reads from a finished run that does not depend on the detector configuration."""
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
    dataset = bag.parent
    manifest = dataset/'manifest.yaml'
    assumed = manifest.exists() and 'imu_noise: ASSUMPTION' in manifest.read_text()
    source = ('simulator noise configuration (ASSUMPTION, not a measured Allan variance)' if assumed else 'run config') + f': {chain}'
    raw = dataset/'raw'/'imu.npy'  # simulator output: columns t, accel, gyro, accel bias, gyro bias (same frame as /imu0)
    true_bias = None
    if raw.exists():
        r = np.load(raw)
        true_bias = np.column_stack([r[:, 0] + SIM_EPOCH_S, r[:, 10:13]])
    return {'name': run_dir.name, 'dataset': dataset, 't': labels['t_s'], 'scored': labels['in_align_window'] == 0,
            'window': labels['in_align_window'] == 1, 'errors': np.column_stack([labels[f'err_{a}'] for a in AXES]),
            'tol': read_tolerances(run_dir), 'est': est, 'diag': parse_diag(run_dir/'estimator.log'),
            'imu': read_series(bag, '/imu0', 'gyro'),
            'odom': read_series(run_dir/'estimate_bag', '/ov_msckf/odomimu', 'twist_angular'),
            'pressure': read_series(bag, '/depth/pressure', 'pressure'),
            'gyro_noise': (imu0['gyroscope_noise_density'], imu0['gyroscope_random_walk']), 'gyro_noise_source': source,
            'true_bias': true_bias}
