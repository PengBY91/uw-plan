"""Simulated dataset access for A1 and the controlled sonar observation set (known correspondence).

Sonar landmarks are convex vertical box edges of the route props, sampled at a fixed world height (the route's
nominal depth). Correspondence comes from the scene geometry, not from the sonar image: this is a controlled
experiment ("oracle association") and must be reported as such, never as automatic sonar SLAM.
A landmark is observed in a sonar frame when it is inside the simulated sector and the simulator's raycast sonar
returned a hit within visibility_radius_m of it (handles occlusion). The measurement is that nearest hit's
(range, bearing), quantised to the sonar image bin centres; elevation is not part of the measurement.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from uwbench.sonar import elevation, range_bearing

EPOCH_S = 1_700_000_000  # scripts/data/sim_to_bag.py EPOCH_NS: sim t = 0 maps here in bags and references


def _kalibr(path):
    text = Path(path).read_text()
    return yaml.safe_load('\n'.join(l for l in text.splitlines() if not l.startswith('%YAML')))


class SimDataset:
    def __init__(self, manifest_path):
        self.manifest_path = Path(manifest_path)
        self.manifest = yaml.safe_load(self.manifest_path.read_text())
        self.root = Path(self.manifest['data_root'])
        self.raw = self.root/'raw'
        self.meta = json.loads((self.raw/'meta.json').read_text())
        self.route = self.meta['route']
        self.imu = np.load(self.raw/'imu.npy')            # t, accel(3), gyro(3), true biases (unused)
        self.gt = np.load(self.raw/'gt_imu.npy')          # t, row-major T_world_imu
        self.left_t = np.load(self.raw/'left_t.npy')
        self.sonar_t = np.load(self.raw/'sonar_t.npy')
        calib = _kalibr(self.root/self.manifest['calibration_uri'])
        self.T_imu_cam0 = np.asarray(calib['cam0']['T_imu_cam'], dtype=float)
        self.T_imu_cam1 = np.asarray(calib['cam1']['T_imu_cam'], dtype=float)
        fx, fy, cx, cy = calib['cam0']['intrinsics']
        self.K = (fx, fy, cx, cy)
        rel = np.linalg.inv(self.T_imu_cam0) @ self.T_imu_cam1
        if not np.allclose(rel[:3, :3], np.eye(3), atol=1e-6) or abs(rel[1, 3]) > 1e-6 or abs(rel[2, 3]) > 1e-6:
            raise ValueError('A1 stereo factor needs a rectified pair with the right camera on +x')
        self.baseline = float(rel[0, 3])
        self.T_imu_sonar = np.asarray(self.meta['socket_extrinsics_in_imu_frame']['sonar'], dtype=float)
        self.sonar_geometry = self.manifest['streams']['sonar']['geometry']
        imu_yaml = _kalibr(self.root/'openvins/kalibr_imu_chain.yaml')['imu0']
        self.imu_noise = {k: float(imu_yaml[k]) for k in ['accelerometer_noise_density', 'accelerometer_random_walk',
                                                          'gyroscope_noise_density', 'gyroscope_random_walk']}
        self._chunks = {}

    def gt_pose(self, t):
        i = int(round(t*self.meta['scenario']['ticks_per_sec']))
        if abs(self.gt[i, 0] - t) > 1e-6:
            raise ValueError('No ground-truth tick at %.6f' % t)
        return self.gt[i, 1:].reshape(4, 4)

    def gt_velocity(self, t, h=0.005):
        return (self.gt_pose(t + h)[:3, 3] - self.gt_pose(t - h)[:3, 3])/(2*h) if t >= h else \
            (self.gt_pose(t + h)[:3, 3] - self.gt_pose(t)[:3, 3])/h

    def image(self, side, index):
        chunk = index // 200
        key = (side, chunk)
        if key not in self._chunks:
            if len(self._chunks) > 4:
                self._chunks.pop(next(iter(self._chunks)))
            self._chunks[key] = np.load(self.raw/f'{side}_{chunk:04d}.npy', mmap_mode='r')
        return np.asarray(self._chunks[key][index % 200])

    def sonar_points_path(self):
        for p in [self.root/'sonar_raycast/sonar_points.npy', self.raw/'sonar_points.npy']:
            if p.exists():
                return p
        return None


def _inside_any(points, boxes):
    inside = np.zeros(len(points), bool)
    for lo, hi in boxes:
        inside |= np.all((points > lo) & (points < hi), axis=1)
    return inside


def edge_landmarks(props, height, eps=0.02):
    """Convex vertical edges of axis-aligned boxes, sampled at world z = height.

    An edge counts when exactly one of the four horizontal diagonal probe points around it lies inside the union of
    boxes (a protruding corner); joints between coplanar boxes (two inside) and re-entrant corners (three) are not
    distinguishable features and are dropped. Rotated boxes are not supported and are reported back.
    """
    boxes, skipped = [], []
    for i, p in enumerate(props):
        if p.get('type') != 'box' or any(abs(r) > 1e-9 for r in p.get('rotation', [0, 0, 0])):
            skipped.append(i)
            continue
        c, s = np.asarray(p['location'], dtype=float), np.asarray(p['scale'], dtype=float)
        boxes.append((c - s/2, c + s/2))
    points = []
    for lo, hi in boxes:
        if not lo[2] <= height <= hi[2]:
            continue
        for x in (lo[0], hi[0]):
            for y in (lo[1], hi[1]):
                probes = np.array([[x + dx, y + dy, height] for dx in (-eps, eps) for dy in (-eps, eps)])
                if _inside_any(probes, boxes).sum() == 1:
                    points.append((x, y, height))
    points = np.unique(np.round(np.array(points, dtype=float).reshape(-1, 3), 6), axis=0)
    return points, skipped


def quantise(rho, theta, geometry):
    lo, hi, nr = geometry['RangeMin'], geometry['RangeMax'], geometry['RangeBins']
    az, na = np.radians(geometry['Azimuth']), geometry['AzimuthBins']
    dr, da = (hi - lo)/nr, az/na
    i = np.clip(np.floor((rho - lo)/dr), 0, nr - 1)
    j = np.clip(np.floor((theta + az/2)/da), 0, na - 1)
    return lo + (i + 0.5)*dr, -az/2 + (j + 0.5)*da


def sonar_observations(ds, landmarks, keyframe_times, visibility_radius_m=0.03, outlier_rate=0.0, seed=0):
    """Oracle-associated sonar measurements at keyframes that coincide with sonar frames.

    Returns rows (keyframe index, landmark id, rho, theta, is_injected_outlier) and per-frame counts. Outliers, when
    requested, replace a measurement by a uniformly random in-sector bin (seeded) and are flagged for scoring only.
    """
    path = ds.sonar_points_path()
    if path is None:
        raise FileNotFoundError('No raycast sonar points for ' + str(ds.root) + '; this route cannot give oracle sonar observations')
    pts = np.load(path, mmap_mode='r')
    frame = np.asarray(pts[:, 0]).astype(np.int64)
    order = np.argsort(frame, kind='stable')
    bounds = np.searchsorted(frame[order], np.arange(len(ds.sonar_t) + 1))
    g = ds.sonar_geometry
    half_az, half_el = np.radians(g['Azimuth'])/2, np.radians(g['Elevation'])/2
    rng = np.random.default_rng(seed)
    rows, in_sector = [], []
    for k, t in enumerate(keyframe_times):
        f = int(np.argmin(np.abs(ds.sonar_t - t)))
        if abs(ds.sonar_t[f] - t) > 1e-6:
            in_sector.append(0)
            continue
        T_ws = ds.gt_pose(t) @ ds.T_imu_sonar
        p_s = (landmarks - T_ws[:3, 3]) @ T_ws[:3, :3]
        rho, theta = range_bearing(p_s)
        sector = (rho >= g['RangeMin']) & (rho <= g['RangeMax']) & (np.abs(theta) <= half_az) & (np.abs(elevation(p_s)) <= half_el)
        in_sector.append(int(sector.sum()))
        if not sector.any():
            continue
        hits = np.asarray(pts[order[bounds[f]:bounds[f + 1]], 1:], dtype=float)
        if not len(hits):
            continue
        for j in np.nonzero(sector)[0]:
            d = np.linalg.norm(hits - p_s[j], axis=1)
            n = int(np.argmin(d))
            if d[n] > visibility_radius_m:
                continue
            r, b = range_bearing(hits[n])
            outlier = bool(outlier_rate > 0 and rng.random() < outlier_rate)
            if outlier:
                r = rng.uniform(g['RangeMin'], g['RangeMax'])
                b = rng.uniform(-half_az, half_az)
            rq, bq = quantise(r, b, g)
            rows.append((k, int(j), float(rq), float(bq), outlier))
    return rows, in_sector


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(8*1024*1024):
            h.update(chunk)
    return h.hexdigest()


def nominal_sonar_height(ds):
    """Median world z of the sonar over the run; edge landmarks are sampled at this height."""
    T = ds.gt[:, 1:].reshape(-1, 4, 4)
    return float(np.median(T[:, 2, 3] + T[:, 2, :3] @ ds.T_imu_sonar[:3, 3]))
