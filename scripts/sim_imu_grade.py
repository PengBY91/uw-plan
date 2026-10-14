"""Derive a simulated dataset whose gyro has a different noise grade, for the pre-hardware IMU sweep (topic A v2.2).

The heading conflict and its drift budget depend on the gyro's noise density N and bias random walk K, which the
simulator only gives as an assumption. This writes data/sim/<route>_<grade>/raw/ where imu.npy carries
  gyro = omega_true + b(t) + white,   b(0) = the simulator's true initial bias,
  b random walk with per-sample step K*sqrt(dt), white with per-sample sigma N/sqrt(dt),
and omega_true is the central difference of the ground-truth IMU attitude. The reconstruction is checked first: on
the source run, measured gyro - true bias - omega_true must be zero-mean white noise at the configured AngVelSigma,
otherwise nothing is written. Accelerometer columns are copied unchanged, so only the gyro grade differs.
Every other raw file and sonar_raycast/ are symlinked; then run scripts/sim_to_bag.py and uwbench audit as usual.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def true_body_rate(gt):
    """Body-frame angular velocity from t + row-major 4x4 world poses; central difference, ends copied."""
    rot = Rotation.from_matrix(gt[:, 1:].reshape(-1, 4, 4)[:, :3, :3])
    w = np.empty((len(gt), 3))
    w[1:-1] = (rot[:-2].inv()*rot[2:]).as_rotvec()/(gt[2:, 0] - gt[:-2, 0])[:, None]
    w[0], w[-1] = w[1], w[-2]
    return w


def check_reconstruction(imu, gt, sigma, rel_tol=0.05):
    """Residual of measured gyro minus true bias minus omega_true; raises unless it matches the configured white noise."""
    r = (imu[:, 4:7] - imu[:, 10:13] - true_body_rate(gt))[1:-1]
    mean, std = r.mean(0), r.std(0)
    stats = {'residual_mean_rad_s': mean.tolist(), 'residual_std_rad_s': std.tolist(), 'configured_sigma_rad_s': sigma}
    if np.any(np.abs(std/sigma - 1) > rel_tol) or np.any(np.abs(mean) > 5*sigma/np.sqrt(len(r))):
        raise SystemExit(f'gyro reconstruction does not match the simulator noise model: {stats}')
    return stats


def synthesize(imu, gt, noise_density, random_walk, seed):
    """New imu array: gyro and gyro-bias columns replaced, everything else copied."""
    rng = np.random.default_rng(seed)
    dt = np.diff(imu[:, 0])
    steps = rng.standard_normal((len(dt), 3))*(random_walk*np.sqrt(dt))[:, None]
    bias = imu[0, 10:13] + np.vstack([np.zeros(3), np.cumsum(steps, axis=0)])
    white = rng.standard_normal((len(imu), 3))*noise_density/np.sqrt(np.median(dt))
    out = imu.copy()
    out[:, 4:7] = true_body_rate(gt) + bias + white
    out[:, 10:13] = bias
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True, help='source data/sim/<route> containing raw/')
    parser.add_argument('--grade', required=True, help='suffix of the derived dataset, e.g. g2')
    parser.add_argument('--noise-density', type=float, required=True, help='gyro N, rad/s/sqrt(Hz)')
    parser.add_argument('--random-walk', type=float, required=True, help='gyro bias random walk K, rad/s^2/sqrt(Hz)')
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--label', default='', help='free text recorded in meta.json, e.g. the grade it stands for')
    args = parser.parse_args()
    src = args.run/'raw'
    out = args.run.parent/f'{args.run.name}_{args.grade}'
    if out.exists():
        parser.error(f'{out} exists; refusing to overwrite')
    meta = json.loads((src/'meta.json').read_text())
    noise = meta['route']['sensors']['imu_noise']
    rate = meta['route']['sensors']['imu_hz']
    imu, gt = np.load(src/'imu.npy'), np.load(src/'gt_imu.npy')
    if len(imu) != len(gt) or not np.allclose(imu[:, 0], gt[:, 0]):
        raise SystemExit('imu.npy and gt_imu.npy are not sampled on the same ticks')
    check = check_reconstruction(imu, gt, noise['AngVelSigma'])
    new = synthesize(imu, gt, args.noise_density, args.random_walk, args.seed)

    (out/'raw').mkdir(parents=True)
    for f in sorted(src.iterdir()):
        if f.name not in ('imu.npy', 'meta.json'):
            (out/'raw'/f.name).symlink_to(f.resolve())
    if (args.run/'sonar_raycast').exists():
        (out/'sonar_raycast').symlink_to((args.run/'sonar_raycast').resolve())
    np.save(out/'raw'/'imu.npy', new)
    # sim_to_bag derives the OpenVINS IMU noise from these per-sample values, so the estimator is told the true grade
    noise['AngVelSigma'] = args.noise_density*np.sqrt(rate)
    noise['AngVelBiasSigma'] = args.random_walk/np.sqrt(rate)
    meta['imu_resynthesis'] = {
        'tool': 'scripts/sim_imu_grade.py', 'source_raw': str(src.resolve()),
        'source_imu_sha256': hashlib.sha256((src/'imu.npy').read_bytes()).hexdigest(),
        'grade': args.grade, 'label': args.label, 'seed': args.seed,
        'gyro_noise_density': args.noise_density, 'gyro_random_walk': args.random_walk,
        'initial_bias': 'simulator true gyro bias at t=0', 'accelerometer': 'copied unchanged from source',
        'omega_true': 'central difference of ground-truth IMU attitude', 'reconstruction_check': check,
        'note': 'ASSUMPTION: synthetic gyro grade for a sweep, not a measured Allan variance'}
    (out/'raw'/'meta.json').write_text(json.dumps(meta, indent=1))
    print(json.dumps({'out': str(out), **meta['imu_resynthesis']}, indent=1))


if __name__ == '__main__':
    main()
