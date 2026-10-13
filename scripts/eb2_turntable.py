"""Synthetic turntable sequences for E-B2 (review step 3, 2026-10-12).

The HoveringAUV pose controller does not track commanded pitch or roll (eb2_pitch_v1 reached 1.7 deg, eb2_roll_v1
tumbled), so pure rotation sweeps are produced by teleporting: this script writes a dataset directory with
raw/meta.json (copied from a source route, name and purpose replaced), raw/gt_imu.npy (the source pose at the
reference time composed with a linear rotation sweep about one IMU axis) and raw/sonar_t.npy; the raycast sonar is then
rendered by team/sim/holoocean_sonar_replay.py, and uwbench motion-b reads the result with --raw.

usage: eb2_turntable.py --source data/sim/eb2_pitch_v1_s01/raw --name eb2_pitch_tt_v1 --axis y --amplitude-deg 12 --out data/sim/eb2_pitch_tt_v1_s01
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def sweep(amp, t, knots=(0, 6, 16, 26, 32, 34)):
    """0 during the initial hold, +amp at the first extreme, -amp at the second, back to 0."""
    return np.interp(t, knots, [0, 0, amp, -amp, 0, 0])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--source', type=Path, required=True, help='raw/ of a generated route (meta.json, gt_imu.npy)')
    ap.add_argument('--name', required=True)
    ap.add_argument('--axis', choices=['x', 'y', 'z'], required=True, help='IMU body axis: x roll, y pitch, z yaw')
    ap.add_argument('--amplitude-deg', type=float, required=True)
    ap.add_argument('--reference-t', type=float, default=6.0)
    ap.add_argument('--duration-s', type=float, default=34.0)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    meta = json.loads((a.source/'meta.json').read_text())
    gt = np.load(a.source/'gt_imu.npy')
    tps = meta['route']['ticks_per_sec']
    T0 = gt[int(round(a.reference_t*tps)), 1:].reshape(4, 4)
    t = np.arange(0, a.duration_s + 0.5/tps, 1/tps)
    ang = sweep(a.amplitude_deg, t)
    rows = []
    for ti, deg in zip(t, ang):
        T = T0.copy()
        T[:3, :3] = T0[:3, :3] @ Rotation.from_euler(a.axis, deg, degrees=True).as_matrix()
        rows.append(np.concatenate([[ti], T.ravel()]))
    dst = a.out/'raw'
    dst.mkdir(parents=True, exist_ok=False)
    np.save(dst/'gt_imu.npy', np.asarray(rows))
    np.save(dst/'sonar_t.npy', np.arange(0, a.duration_s + 1e-9, 1/meta['route']['sensors']['sonar_hz']))
    m = json.loads(json.dumps(meta))
    m['route']['name'] = a.name
    m['route']['purpose'] = (f'E-B2 turntable analogue: teleported pure rotation sweep of +-{a.amplitude_deg:g} deg about the IMU '
                             f'{a.axis} axis at the reference pose (t = {a.reference_t:g} s) of {meta["route"]["name"]}; only gt_imu.npy, '
                             'sonar_t.npy and the raycast sonar replay exist')
    m['route']['waypoints'] = [{'t': 0, 'pose': 'synthetic, see purpose'}]
    m['synthetic'] = {'source_raw': str(a.source), 'axis': a.axis, 'amplitude_deg': a.amplitude_deg,
                      'profile': 'linear 0 (6 s) -> +amp (16 s) -> -amp (26 s) -> 0 (32 s)', 'tool': 'scripts/eb2_turntable.py'}
    m['counts'] = {'ticks': len(rows), 'sonar': int(a.duration_s*meta['route']['sensors']['sonar_hz']) + 1}
    (dst/'meta.json').write_text(json.dumps(m, indent=2))
    print(a.name, len(rows), 'ticks; angle range', float(ang.min()), float(ang.max()))


if __name__ == '__main__':
    main()
