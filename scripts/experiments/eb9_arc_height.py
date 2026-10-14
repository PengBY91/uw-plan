"""E-B9 (W5-06): actual false-occupied thickness of B2a under ground-truth poses, by slant range, against the arc-height
model h = 2 rho sin(alpha/2).

Replays the sonar frames of a ground-truth-pose B2a map (same stride, threshold and aperture as `uwbench map`) and
measures, along the surface normal (distance from the nearest prop box surface), how far the arc voxels reach outside
the structure:
- per echo: the largest outside distance among the voxels of that echo's arc (0 when the arc stays within the band);
- per map voxel: final B2a occupied voxels outside the band of the map, each attributed to the smallest slant range
  of any echo arc that wrote it.
Echoes are split into hit-backed (the raycast sonar put at least one hit in that range x azimuth cell) and noise-only
(the cell crossed the threshold through the render's additive Rayleigh noise alone; sonar_points.npy, raycast routes
only). Also splits by the axis of the nearest face (z = top/bottom faces, where the arc crosses the surface; x/y = side faces,
where it lies in the surface). With ground-truth poses and a geometry-only sonar the only sources of outside voxels are
the elevation arc and voxelisation, so a systematic excess over h of more than one voxel layer would mean something
other than elevation ambiguity dominates (runbook E-B9).

usage: eb9_arc_height.py --run runs/sim_<ds>_b2a_gt_r01 [--run ...] --out reports/w5/eb9/arc_height.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from uwbench.a.backend.simdata import SimDataset
from uwbench.b.cli import DEFAULTS, merge, sonar_images
from uwbench.b.metrics import signed_depth
from uwbench.b.scene import boxes_from_props
from uwbench.b.sonar_map import cell_geometry, echo_cells
from uwbench.b.voxel_map import VoxelGrid
from uwbench.sonar import arc_height_span, arc_points

BINS = np.arange(0.5, 10.01, 0.5)


def nearest_face_axis(points, boxes):
    """Axis (0 x, 1 y, 2 z) of the largest gap to the nearest box, for points outside every box."""
    best, axis = np.full(len(points), np.inf), np.zeros(len(points), int)
    for lo, hi in boxes:
        gap = np.maximum(np.maximum(np.asarray(lo) - points, points - np.asarray(hi)), 0)
        d = np.linalg.norm(gap, axis=1)
        closer = d < best
        best = np.where(closer, d, best)
        axis = np.where(closer, np.argmax(gap, axis=1), axis)
    return axis


def hit_cells(ds, geometry, shape):
    """frame index -> set of (range bin, azimuth bin) holding at least one raycast hit (render_sonar's binning)."""
    path = ds.sonar_points_path()
    if path is None:
        return None
    pts = np.load(path)
    pts = pts[np.argsort(pts[:, 0], kind='stable')]
    frames = pts[:, 0].astype(int)
    r = np.linalg.norm(pts[:, 1:4], axis=1)
    az = np.degrees(np.arctan2(pts[:, 2], pts[:, 1]))
    nr, na = shape
    ri = np.floor((r - geometry['RangeMin'])/(geometry['RangeMax'] - geometry['RangeMin'])*nr).astype(int)
    ai = np.floor((az + geometry['Azimuth']/2)/geometry['Azimuth']*na).astype(int)
    if ds.route['sensors']['sonar'].get('RaycastFlipAzimuth', False):
        ai = na - 1 - ai
    ok = (ri >= 0) & (ri < nr) & (ai >= 0) & (ai < na)
    starts = np.searchsorted(frames, np.arange(frames.max() + 2))
    key = ri*na + ai
    return lambda i: set(np.unique(key[starts[i]:starts[i + 1]][ok[starts[i]:starts[i + 1]]]).tolist()) if i + 1 < len(starts) else set()


def summarise(rho, thick, alpha, voxel, band):
    rows = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        sel = (rho >= lo) & (rho < hi)
        if sel.sum() < 20:
            continue
        x = thick[sel]
        h_hi = float(arc_height_span(hi, alpha))
        pos = x[x > 0]
        rows.append({'rho_lo': float(lo), 'rho_hi': float(hi), 'n': int(sel.sum()), 'fraction_outside': float(np.mean(x > 0)),
                     'median_outside_m': float(np.median(pos)) if len(pos) else 0.0,
                     'p95_m': float(np.percentile(x, 95)), 'max_m': float(x.max()),
                     'h_model_at_rho_hi_m': h_hi, 'h_model_at_mid_m': float(arc_height_span(0.5*(lo + hi), alpha)),
                     'max_minus_h_layers': float((x.max() - h_hi)/voxel),
                     'exceed_h_plus_voxel': float(np.mean(x > h_hi + voxel))})
    return rows


def analyse(run_dir):
    run_dir = Path(run_dir)
    meta = json.loads((run_dir/'map/meta.json').read_text())
    if meta['poses'] != 'gt' or meta['method'] != 'b2a':
        raise SystemExit(f'{run_dir}: E-B9 needs a ground-truth-pose B2a map')
    cfg = merge(DEFAULTS, meta['config'])
    ds = SimDataset(meta['manifest'])
    boxes = boxes_from_props(ds.route['props'])
    m = np.load(run_dir/'map/map.npz')
    grid = VoxelGrid(m['origin'], tuple(m['shape']), float(m['resolution']))
    voxel, band = grid.resolution, 0.10
    g = ds.sonar_geometry
    alpha = np.radians(g['Elevation'])
    image, folder = sonar_images(ds)
    step = voxel/2
    min_rho = np.full(grid.shape, np.inf)
    min_rho_backed = np.full(grid.shape, np.inf)
    cells_of = hit_cells(ds, g, (g['RangeBins'], g['AzimuthBins']))
    echo_rho, echo_thick, echo_z, echo_backed = [], [], [], []
    for i, t in enumerate(ds.sonar_t):
        if i % cfg['sonar_stride']:
            continue
        img = image(i)
        rows, cols = echo_cells(img, cfg['sonar_threshold'])
        if not len(rows):
            continue
        rho, theta, _, _ = cell_geometry(g, img.shape)
        T = ds.gt_pose(float(t)) @ ds.T_imu_sonar
        backed_cells = cells_of(i) if cells_of else None
        for row_i, col_i, r, c in zip(rows, cols, rho(rows), theta(cols)):
            backed = True if backed_cells is None else (int(row_i)*g['AzimuthBins'] + int(col_i)) in backed_cells
            n = max(int(np.ceil(r*alpha/step)) + 1, 2)
            pts = arc_points(r, c, np.linspace(-alpha/2, alpha/2, n)) @ T[:3, :3].T + T[:3, 3]
            idx, inside = grid.index(pts)
            if not inside.any():
                continue
            idx = np.unique(idx[inside], axis=0)
            centres = grid.origin + (idx + 0.5)*voxel
            d = signed_depth(centres, boxes)
            out = -d[d < -band]
            echo_rho.append(r)
            echo_backed.append(backed)
            echo_thick.append(float(out.max()) if len(out) else 0.0)
            echo_z.append(bool(len(out)) and bool(nearest_face_axis(centres[d < -band][[np.argmax(out)]], boxes)[0] == 2))
            v = tuple(idx.T)
            min_rho[v] = np.minimum(min_rho[v], r)
            if backed:
                min_rho_backed[v] = np.minimum(min_rho_backed[v], r)
    echo_rho, echo_thick, echo_z, echo_backed = map(np.asarray, (echo_rho, echo_thick, echo_z, echo_backed))
    state = m['state']
    occ = (state == 2) & np.isfinite(min_rho)          # occupied and written by a sonar arc
    centres = grid.centers()[occ]
    d = signed_depth(centres, boxes)
    outside = d < -band
    vox_rho, vox_thick = min_rho[occ][outside], -d[outside]
    vox_axis = nearest_face_axis(centres[outside], boxes)
    vox_backed_rho = min_rho_backed[occ][outside]
    backed_v = np.isfinite(vox_backed_rho)
    res = {'run': str(run_dir), 'dataset': ds.manifest['run_id'], 'sonar_images': folder, 'vertical_aperture_deg': g['Elevation'],
           'voxel_m': voxel, 'band_m': band, 'range_resolution_m': (g['RangeMax'] - g['RangeMin'])/g['RangeBins'],
           'echoes': int(len(echo_rho)), 'echo_fraction_outside': float(np.mean(echo_thick > 0)),
           'echo_outside_nearest_face_z_fraction': float(np.mean(echo_z[echo_thick > 0])) if (echo_thick > 0).any() else None,
           'hit_cells_available': cells_of is not None,
           'noise_only_echoes': int((~echo_backed).sum()),
           'noise_only_echo_fraction_outside': float(np.mean(echo_thick[~echo_backed] > 0)) if (~echo_backed).any() else None,
           'per_echo': summarise(echo_rho, echo_thick, alpha, voxel, band),
           'per_echo_hit_backed': summarise(echo_rho[echo_backed], echo_thick[echo_backed], alpha, voxel, band),
           'outside_voxels_noise_only': int((~backed_v).sum()),
           'per_voxel_hit_backed': summarise(vox_backed_rho[backed_v], vox_thick[backed_v], alpha, voxel, band),
           'sonar_occupied_voxels': int(occ.sum()), 'outside_voxels': int(outside.sum()),
           'outside_voxels_by_face_axis': {a: int(np.sum(vox_axis == k)) for k, a in enumerate('xyz')},
           'per_voxel': summarise(vox_rho, vox_thick, alpha, voxel, band),
           'per_voxel_z_faces': summarise(vox_rho[vox_axis == 2], vox_thick[vox_axis == 2], alpha, voxel, band),
           'per_voxel_side_faces': summarise(vox_rho[vox_axis != 2], vox_thick[vox_axis != 2], alpha, voxel, band)}
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', action='append', required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    out = [analyse(r) for r in a.run]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2) + '\n')
    for r in out:
        print(r['dataset'], r['echoes'], r['echo_fraction_outside'], r['noise_only_echoes'], r['outside_voxels'],
              r['outside_voxels_noise_only'], r['outside_voxels_by_face_axis'])


if __name__ == '__main__':
    main()
