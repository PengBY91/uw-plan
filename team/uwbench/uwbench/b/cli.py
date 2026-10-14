"""uwbench map / eval-map (P05/P09) on simulated datasets.

map: B0 stereo occupancy plus optional sonar updates (b1, b2a, b2c) along one pose source:
  gt               ground-truth IMU poses;
  <file>.tum       A1 trajectory (already in the reference frame: A1 starts from ground truth);
  <run dir>        A0 OpenVINS run (estimate_bag), aligned to the reference with the eval-traj risk-mode
                   rule: yaw + translation over the first 10 s of matched poses; drift after that stays in the map.
ROI and resolution are derived only from the route (waypoints and props), so every method shares them.
eval-map: B plan section 6 ratios in that ROI, per height layer, structure thickness and error, plus (review step 3,
2026-10-12) the surface view from uwbench.b.metrics: where predicted occupancy lies relative to the box surfaces,
recall of the route-facing surface band, the same ratios restricted to the visible ROI (voxels not shadowed by a box
and not box interior), and the main curve as effective coverage and surface recall against the false-occupied rate
over occupancy thresholds. With ground-truth poses the false-free rate of every method is near zero, so coverage
against false occupied, not against false free, is the axis that separates the methods.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import yaml
from scipy.spatial.transform import Rotation, Slerp

from uwbench.b.voxel_map import VoxelGrid, truth_from_boxes
from uwbench.b.occupancy import map_ratios

DEFAULTS = {'resolution_m': 0.10, 'stereo_pixel_stride': 4, 'max_stereo_range_m': 6.0, 'sonar_stride': 5, 'veto': False,
            'sonar_threshold': 0.25, 'roi_margin_m': {'x': 3.0, 'y_behind_props': 1.0, 'y_vehicle': 0.5, 'z': 1.5},
            'b2c': {'hit': 0.85, 'miss': -0.4}}


def merge(base, override):
    out = dict(base)
    for k, v in (override or {}).items():
        out[k] = merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


class Poses:
    def __init__(self, ds, source):
        self.ds, self.source, self.alignment = ds, str(source), None
        if self.source == 'gt':
            self.t = ds.gt[:, 0]
            T = ds.gt[:, 1:].reshape(-1, 4, 4)
            self.R, self.p = T[:, :3, :3], T[:, :3, 3]
            return
        from uwbench.a.backend.simdata import EPOCH_S
        path = Path(source)
        if path.is_file():
            d = np.loadtxt(path).reshape(-1, 8)
            t, p, R = d[:, 0] - EPOCH_S, d[:, 1:4], Rotation.from_quat(d[:, 4:8]).as_matrix()
        else:
            from uwbench.eval_traj import read_estimate
            t, R, p, _ = read_estimate(path/'estimate_bag', '/ov_msckf/poseimu')
            t = t - EPOCH_S
            from uwbench.metrics.axes import align_yaw_translation
            i = np.clip(np.round(t*ds.meta['scenario']['ticks_per_sec']).astype(int), 0, len(ds.gt) - 1)
            G = ds.gt[i, 1:].reshape(-1, 4, 4)
            window = t <= t[0] + 10.0
            Ra, ta, info = align_yaw_translation(G[:, :3, :3], G[:, :3, 3], R, p, window)
            R, p = Ra @ R, p @ Ra.T + ta
            self.alignment = dict(info, rule='eval-traj risk mode: yaw + translation over the first 10 s')
        self.t, self.R, self.p = t, R, p
        self._slerp = Slerp(t, Rotation.from_matrix(R))

    def at(self, t):
        if self.source == 'gt':
            return self.ds.gt_pose(t)
        if t < self.t[0] or t > self.t[-1]:
            return None
        T = np.eye(4)
        T[:3, :3] = self._slerp([t]).as_matrix()[0]
        T[:3, 3] = [np.interp(t, self.t, self.p[:, k]) for k in range(3)]
        return T


def route_roi(ds, cfg, boxes):
    m = cfg['roi_margin_m']
    T = ds.gt[:, 1:].reshape(-1, 4, 4)
    traj = T[:, :3, 3]
    from uwbench.a.backend.simdata import nominal_sonar_height
    z0 = nominal_sonar_height(ds)
    lo = np.array([traj[:, 0].min() - m['x'], traj[:, 1].min() - m['y_vehicle'], z0 - m['z']])
    hi = np.array([traj[:, 0].max() + m['x'], max(b[1][1] for b in boxes) + m['y_behind_props'], z0 + m['z']])
    res = cfg['resolution_m']
    shape = np.ceil((hi - lo)/res).astype(int)
    return lo, shape, z0


def chunked_images(folder, prefix='sonar', chunk=200):
    """image(i) over <prefix>_XXXX.npy chunk files (chunk frames each), keeping one chunk memory-mapped at a time."""
    cache = {}

    def image(i):
        c = i//chunk
        if c not in cache:
            cache.clear()
            cache[c] = np.load(Path(folder)/f'{prefix}_{c:04d}.npy', mmap_mode='r')
        return np.asarray(cache[c][i % chunk])
    return image


def sonar_images(ds):
    """Sonar image reader of a dataset (raycast replay if present, else raw) and the folder used."""
    folder = ds.root/'sonar_raycast' if (ds.root/'sonar_raycast/sonar_0000.npy').exists() else ds.raw
    return chunked_images(folder), str(folder)


def map_main(argv):
    parser = argparse.ArgumentParser(prog='uwbench map', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--method', choices=['b0', 'b1', 'b2a', 'b2c'], required=True)
    parser.add_argument('--depth', type=Path, required=True, help='uwbench depth output directory')
    parser.add_argument('--poses', required=True, help="'gt', an A1 .tum file, or an A0 run directory")
    parser.add_argument('--config', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error('Refusing to overwrite ' + str(args.out))
    from uwbench.a.backend.simdata import SimDataset
    from uwbench.b.scene import boxes_from_props
    from uwbench.b.sonar_map import SonarUpdater
    cfg = merge(DEFAULTS, (yaml.safe_load(args.config.read_text()) or {}).get('map') if args.config else None)
    ds = SimDataset(args.manifest)
    boxes = boxes_from_props(ds.route['props'])
    origin, shape, z0 = route_roi(ds, cfg, boxes)
    grid = VoxelGrid(origin, shape, cfg['resolution_m'])
    if cfg['veto']:
        if args.method == 'b0':
            parser.error('veto needs a sonar method')
        grid.veto = np.zeros(grid.shape, bool)
    poses = Poses(ds, args.poses)
    with np.load(args.depth/'depth.npz') as f:   # decompress once; npz members re-decompress on every access
        dep = {k: f[k] for k in ['times', 'depth', 'valid']}
    fx, fy, cx, cy = ds.K
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    s = cfg['stereo_pixel_stride']
    events = [(float(t), 'depth', i) for i, t in enumerate(dep['times'])]
    sonar = None
    if args.method != 'b0':
        g = ds.sonar_geometry
        kw = cfg['b2c'] if args.method == 'b2c' else {}
        sonar = SonarUpdater(grid, g, np.radians(g['Elevation']), cfg['sonar_threshold'], args.method, **kw)
        image, sonar_source = sonar_images(ds)
        events += [(float(t), 'sonar', i) for i, t in enumerate(ds.sonar_t) if i % cfg['sonar_stride'] == 0]
    skipped = 0
    wall = time.perf_counter()
    for t, kind, i in sorted(events):
        T = poses.at(t)
        if T is None:
            skipped += 1   # no trusted pose: no unmarked update
            continue
        if kind == 'depth':
            depth = dep['depth'][i].astype(np.float32)[::s, ::s]
            valid = dep['valid'][i][::s, ::s]
            Ks = K.copy()
            Ks[0, 0] /= s; Ks[1, 1] /= s; Ks[0, 2] = (cx + 0.5)/s - 0.5; Ks[1, 2] = (cy + 0.5)/s - 0.5
            grid.integrate_depth(T, ds.T_imu_cam0, Ks, depth, valid, t, cfg['max_stereo_range_m'])
        else:
            sonar.update(image(i), T @ ds.T_imu_sonar, t)
    args.out.mkdir(parents=True)
    np.savez_compressed(args.out/'map.npz', origin=grid.origin, shape=np.array(grid.shape), resolution=grid.resolution,
                        log_odds=grid.log_odds, observed=grid.observed, last_update=grid.last_update,
                        sources=grid.sources, state=grid.state(),
                        occluded=sonar.occluded if sonar is not None and args.method == 'b2c' else np.zeros(grid.shape, np.int32),
                        veto=grid.veto if grid.veto is not None else np.zeros(grid.shape, bool))
    meta = {'method': args.method, 'manifest': str(args.manifest), 'poses': args.poses, 'pose_alignment': poses.alignment,
            'depth': str(args.depth), 'config': cfg, 'roi': {'origin': grid.origin.tolist(), 'shape': list(grid.shape),
            'resolution_m': grid.resolution, 'nominal_sonar_height_m': z0}, 'events': len(events),
            'skipped_without_pose': skipped, 'wall_s': time.perf_counter() - wall,
            'sonar': sonar.stats if sonar else None, 'sonar_images': sonar_source if sonar else None,
            'vertical_aperture_deg': ds.sonar_geometry['Elevation'] if sonar else None,
            'aperture_note': 'simulation setting, true in simulation; not the platform value' if sonar else None,
            'occupancy': {'hit': grid.hit, 'miss': grid.miss, 'clamp': list(grid.clamp)},
            'veto': {'enabled': bool(cfg['veto']), 'vetoed_free_updates': grid.vetoed_free_updates,
                     'veto_voxels': int(grid.veto.sum()) if grid.veto is not None else 0}}
    (args.out/'meta.json').write_text(json.dumps(meta, indent=2, default=float) + '\n')
    print(json.dumps({k: meta[k] for k in ['method', 'poses', 'events', 'skipped_without_pose', 'wall_s', 'sonar']}, indent=2, default=float))
    return 0


def _distance_to_boxes(points, boxes):
    d = np.full(len(points), np.inf)
    for lo, hi in boxes:
        gap = np.linalg.norm(np.maximum(np.maximum(lo - points, points - hi), 0), axis=1)
        d = np.minimum(d, gap)
    return d


def eval_main(argv):
    parser = argparse.ArgumentParser(prog='uwbench eval-map', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--map', type=Path, required=True, help='uwbench map output directory')
    parser.add_argument('--band-m', type=float, help='Boundary exclusion band (default: one voxel)')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error('Refusing to overwrite ' + str(args.out))
    from uwbench.a.backend.simdata import SimDataset
    from uwbench.b.scene import boxes_from_props
    ds = SimDataset(args.manifest)
    boxes = boxes_from_props(ds.route['props'])
    m = np.load(args.map/'map.npz')
    meta = json.loads((args.map/'meta.json').read_text())
    grid = VoxelGrid(m['origin'], tuple(m['shape']), float(m['resolution']))
    grid.log_odds, grid.observed = m['log_odds'], m['observed']
    band = args.band_m if args.band_m is not None else grid.resolution
    truth = truth_from_boxes(grid, boxes, band)
    names = np.array(['unknown', 'free', 'occ'], dtype=object)

    def labels(occ_above=0.0, free_below=0.0):
        grid.occupied_above, grid.free_below = occ_above, free_below
        return names[grid.state()]

    pred = labels()
    result = {'map': str(args.map), 'method': meta['method'], 'poses': meta['poses'], 'band_m': band,
              'roi': meta['roi'], 'overall': map_ratios(truth, pred)}
    from uwbench.b.metrics import occupied_placement, route_facing_surface, signed_depth, surface_recall, visible_roi
    centres_all = grid.centers().reshape(-1, 3)
    y_vehicle = float(np.median(ds.gt[:, 1:].reshape(-1, 4, 4)[:, 1, 3]))
    depth_signed = signed_depth(centres_all, boxes)
    surface = route_facing_surface(centres_all, boxes, band, y_vehicle)
    visible = visible_roi(centres_all, boxes, band, y_vehicle).reshape(grid.shape)
    occ_flat = (pred == 'occ').reshape(-1)
    result['surface'] = {**occupied_placement(depth_signed, occ_flat, band), 'visible_surface_voxels': int(surface.sum()),
                         'visible_surface_recall': surface_recall(occ_flat, surface),
                         'note': 'route-facing surface band of the box props; interior and far-side voxels are not what a sensor on the route sees'}
    result['visible'] = {'fraction_of_roi': float(visible.mean()), 'fraction_of_true_free': float(visible[truth == 'free'].mean()) if (truth == 'free').any() else None,
                         'ratios': map_ratios(truth[visible], pred[visible]),
                         'note': 'ROI without voxels shadowed by a box (behind its far face within its x-z footprint) or inside a box beyond the band'}
    z = grid.centers()[..., 2]
    z0 = meta['roi']['nominal_sonar_height_m']
    layers = {'below_z0_minus_0.5': z < z0 - 0.5, 'within_0.5_of_z0': np.abs(z - z0) <= 0.5, 'above_z0_plus_0.5': z > z0 + 0.5}
    result['layers'] = {k: map_ratios(truth[v], pred[v]) for k, v in layers.items()}
    occ = pred == 'occ'
    centers = grid.centers()[occ]
    d = _distance_to_boxes(centers, boxes)
    result['structure_error_m'] = {'predicted_occupied': int(occ.sum()),
        'median': float(np.median(d)) if len(d) else None, 'p95': float(np.percentile(d, 95)) if len(d) else None,
        'fraction_within_band': float(np.mean(d <= band)) if len(d) else None}
    # thickness along y (the wall normal of these routes) per (x, z) column that truly contains structure
    t_occ = (truth == 'occ').any(axis=1)
    p_cnt = occ.sum(axis=1)
    cols = t_occ & (p_cnt > 0)
    result['thickness_y_m'] = {'columns_with_structure': int(t_occ.sum()), 'columns_with_prediction': int(cols.sum()),
        'predicted_median': float(np.median(p_cnt[cols])*grid.resolution) if cols.any() else None,
        'true_median': float(np.median((truth == 'occ').sum(axis=1)[t_occ])*grid.resolution) if t_occ.any() else None,
        'note': 'occupied voxel count along y times resolution; includes far-side spill and is per column'}
    curve = []
    for occ_above in [0.0, 0.5, 1.0, 2.0, 3.0]:
        for free_below in [0.0, -0.5, -1.0]:
            r = map_ratios(truth, labels(occ_above, free_below))
            curve.append({'occupied_above': occ_above, 'free_below': free_below, 'effective_coverage': r['effective_coverage'],
                          'false_free_rate': r['false_free_rate'], 'false_occupied_rate': r['false_occupied_rate'],
                          'occupied_recall': r['occupied_recall']})
    result['curve'] = curve
    # main curve (review step 3): coverage and surface recall against false occupied, visible ROI, free threshold fixed
    main_curve = []
    for occ_above in [0.0, 0.5, 1.0, 2.0, 3.0]:
        lab = labels(occ_above, 0.0)
        r = map_ratios(truth[visible], lab[visible])
        main_curve.append({'occupied_above': occ_above, 'false_occupied_rate': r['false_occupied_rate'],
                           'effective_coverage': r['effective_coverage'], 'false_free_rate': r['false_free_rate'],
                           'visible_surface_recall': surface_recall((lab == 'occ').reshape(-1), surface)})
    result['main_curve'] = main_curve
    labels()
    if 'occluded' in m.files and m['occluded'].any():
        st = labels()
        result['b2c_tri_state'] = {'occupied': int((st == 'occ').sum()), 'verified_free': int((st == 'free').sum()),
                                   'occluded_unknown': int(((st == 'unknown') & (m['occluded'] > 0)).sum()),
                                   'never_observed': int(((st == 'unknown') & (m['occluded'] == 0)).sum())}
    args.out.mkdir(parents=True)
    (args.out/'metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ['method', 'poses', 'overall', 'visible', 'surface']}, indent=2))
    return 0
