"""B0 stereo depth (P05): SGBM disparity -> metric depth with left-right consistency, range and match masks.

Inputs are rectified pairs (the simulated pair is ideal pinhole, same orientation, right camera on +x); real data
must be rectified with its own underwater calibration first, which this module refuses to guess. Depth is optical z
in metres. Invalid pixels are NaN and never become free rays downstream.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from uwbench.b.scene import boxes_from_props

DEFAULTS = {'num_disparities': 128, 'block_size': 5, 'uniqueness': 10, 'speckle_window': 100, 'speckle_range': 2,
            'lr_max_px': 1.0, 'min_depth_m': 0.3, 'max_depth_m': 8.0, 'min_disparity_px': 1.0,
            'texture_window_px': 9, 'min_texture_std': 1.0}


def local_std(image, window):
    """Grey-level standard deviation in a window x window box around each pixel."""
    x = image.astype(np.float32)
    m = cv2.boxFilter(x, -1, (window, window))
    m2 = cv2.boxFilter(x*x, -1, (window, window))
    return np.sqrt(np.maximum(m2 - m*m, 0))


def _sgbm(p, flip=False):
    n = int(np.ceil(p['num_disparities']/16)*16)
    b = p['block_size']
    return cv2.StereoSGBM_create(minDisparity=0, numDisparities=n, blockSize=b, P1=8*b*b, P2=32*b*b,
                                 uniquenessRatio=p['uniqueness'], speckleWindowSize=p['speckle_window'],
                                 speckleRange=p['speckle_range'], mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)


def disparity_pair(left, right, p):
    """Left disparity and right disparity (computed on mirrored images), both in pixels, NaN where unmatched."""
    m = _sgbm(p)
    dl = m.compute(left, right).astype(np.float32)/16.0       # SGBM returns fixed point x16
    dr = m.compute(right[:, ::-1].copy(), left[:, ::-1].copy()).astype(np.float32)[:, ::-1]/16.0
    dl[dl <= 0] = np.nan
    dr[dr <= 0] = np.nan
    return dl, dr


def depth_from_pair(left, right, fx, baseline, params=None):
    p = dict(DEFAULTS, **(params or {}))
    dl, dr = disparity_pair(left, right, p)
    h, w = dl.shape
    u = np.arange(w)[None, :].repeat(h, 0)
    ur = np.round(u - np.nan_to_num(dl, nan=0)).astype(int)
    inside = (ur >= 0) & (ur < w) & np.isfinite(dl)
    back = np.full_like(dl, np.nan)
    rows = np.arange(h)[:, None].repeat(w, 1)
    back[inside] = dr[rows[inside], ur[inside]]
    masks = {'matched': np.isfinite(dl), 'lr_consistent': np.abs(dl - back) <= p['lr_max_px'],
             'min_disparity': np.nan_to_num(dl, nan=0) >= p['min_disparity_px'],
             # SGBM fills textureless fog with plausible, left-right consistent disparities; require texture
             'textured': local_std(left, p['texture_window_px']) >= p['min_texture_std']}
    depth = fx*baseline/dl
    masks['in_range'] = (depth >= p['min_depth_m']) & (depth <= p['max_depth_m'])
    valid = masks['matched'] & masks['lr_consistent'] & masks['min_disparity'] & masks['in_range'] & masks['textured']
    depth = np.where(valid, depth, np.nan).astype(np.float32)
    return depth, valid, masks, p


def render_box_depth(T_world_cam, K, boxes, shape):
    """Optical-z depth of the first axis-aligned box hit per pixel (reference for the simulated scenes); NaN = no hit."""
    fx, fy, cx, cy = K
    v, u = np.mgrid[0:shape[0], 0:shape[1]]
    rays = np.stack([(u + 0.5 - cx)/fx, (v + 0.5 - cy)/fy, np.ones(shape)], axis=-1)
    d = rays @ T_world_cam[:3, :3].T
    o = T_world_cam[:3, 3]
    depth = np.full(shape, np.inf)
    with np.errstate(divide='ignore', invalid='ignore'):
        for lo, hi in boxes:
            t1, t2 = (np.asarray(lo) - o)/d, (np.asarray(hi) - o)/d
            near, far = np.nanmax(np.minimum(t1, t2), axis=-1), np.nanmin(np.maximum(t1, t2), axis=-1)
            hit = (near <= far) & (near > 0)
            depth = np.where(hit & (near < depth), near, depth)
    depth[~np.isfinite(depth)] = np.nan
    return depth


def main(argv):
    parser = argparse.ArgumentParser(prog='uwbench depth', description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--config', type=Path, help='YAML with a depth: section overriding DEFAULTS')
    parser.add_argument('--stride', type=int, default=10, help='Every n-th stereo frame')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error('Refusing to overwrite ' + str(args.out))
    import yaml
    from uwbench.a1.simdata import SimDataset
    cfg = (yaml.safe_load(args.config.read_text()) or {}).get('depth') if args.config else None
    ds = SimDataset(args.manifest)
    try:
        boxes = boxes_from_props(ds.route.get('props', []))
    except ValueError:
        boxes = None
    frames = np.arange(0, len(ds.left_t), args.stride)
    depths, valids, errors, coverage = [], [], [], []
    for f in frames:
        depth, valid, masks, p = depth_from_pair(ds.image('left', f), ds.image('right', f), ds.K[0], ds.baseline, cfg)
        depths.append(depth.astype(np.float16))
        valids.append(valid)
        coverage.append({k: float(v.mean()) for k, v in masks.items()} | {'valid': float(valid.mean())})
        if boxes:
            ref = render_box_depth(ds.gt_pose(ds.left_t[f]) @ ds.T_imu_cam0, ds.K, boxes, depth.shape)
            both = valid & np.isfinite(ref)
            errors.append(np.column_stack([ref[both], depth[both] - ref[both]]))
            coverage[-1]['valid_on_reference'] = float(both.sum()/max(np.isfinite(ref).sum(), 1))
            coverage[-1]['valid_without_reference'] = float((valid & ~np.isfinite(ref)).sum()/max(valid.sum(), 1))
    args.out.mkdir(parents=True)
    np.savez_compressed(args.out/'depth.npz', frames=frames, times=ds.left_t[frames], depth=np.stack(depths),
                        valid=np.stack(valids))
    report = {'manifest': str(args.manifest), 'params': p, 'stride': args.stride, 'frames': int(len(frames)),
              'units': 'optical z, metres; NaN invalid',
              'coverage_mean': {k: float(np.mean([c[k] for c in coverage if k in c])) for k in coverage[0]},
              'reference': 'analytic ray cast of the route props with ground-truth camera pose' if boxes else None}
    if boxes and errors:
        e = np.concatenate(errors)
        bins = np.arange(0, p['max_depth_m'] + 1.0, 1.0)
        report['error_by_range'] = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            sel = (e[:, 0] >= lo) & (e[:, 0] < hi)
            if sel.sum():
                x = e[sel, 1]
                report['error_by_range'].append({'range_m': [float(lo), float(hi)], 'n': int(sel.sum()),
                    'median_m': float(np.median(x)), 'mad_m': float(np.median(np.abs(x - np.median(x)))),
                    'p95_abs_m': float(np.percentile(np.abs(x), 95)),
                    'gross_fraction_gt_0.1m': float(np.mean(np.abs(x) > 0.1))})
    (args.out/'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return 0
