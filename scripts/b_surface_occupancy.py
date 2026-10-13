"""Supplementary B map metric: occupancy measured on the structure surface instead of the box interior.

eval-map's occupied recall uses truth_from_boxes, which excludes a band around every box boundary, so its true
occupied voxels are box interior. A camera or sonar only sees the surface; interior hits come from pose or depth
error smearing the surface inward, so that recall rewards smear. This script reads uwbench map outputs and reports,
per map, where the predicted occupied voxels lie relative to the box surfaces (outside, surface band, interior) and
the fraction of surface-band voxels on the faces seen from the route that are predicted occupied.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def signed_depth(points, boxes):
    """>0: depth inside the nearest containing box; <0: minus distance to the nearest box when outside all."""
    outside = np.full(len(points), np.inf)
    inside = np.zeros(len(points))
    for lo, hi in boxes:
        gap = np.linalg.norm(np.maximum(np.maximum(lo - points, points - hi), 0), axis=1)
        outside = np.minimum(outside, gap)
        ins = np.all((points >= lo) & (points <= hi), axis=1)
        inside = np.where(ins, np.maximum(inside, np.minimum(points - lo, hi - points).min(axis=1)), inside)
    return np.where(inside > 0, inside, -outside)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--maps', type=Path, nargs='+', required=True, help='uwbench map output directories')
    ap.add_argument('--band-m', type=float, default=0.10)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    from uwbench.a1.simdata import SimDataset
    from uwbench.b.scene import boxes_from_props
    ds = SimDataset(a.manifest)
    boxes = boxes_from_props(ds.route['props'])
    y_vehicle = float(np.median(ds.gt[:, 1:].reshape(-1, 4, 4)[:, 1, 3]))
    rows = []
    for mp in a.maps:
        m = np.load(mp/'map.npz')
        shape, origin, res = tuple(m['shape']), m['origin'], float(m['resolution'])
        centres = origin + (np.indices(shape).reshape(3, -1).T + 0.5)*res
        d = signed_depth(centres, boxes)
        # faces seen from the route: the box side facing the vehicle (smaller |y - y_vehicle|) plus the x ends;
        # approximated by voxels in the surface band whose y lies between the vehicle and the box centre plane
        near = np.zeros(len(centres), bool)
        for lo, hi in boxes:
            ymid = 0.5*(lo[1] + hi[1])
            side = (centres[:, 1] - ymid)*np.sign(y_vehicle - ymid) >= 0
            inb = np.all((centres >= lo - a.band_m) & (centres <= hi + a.band_m), axis=1)
            near |= inb & side
        surface = (np.abs(d) <= a.band_m) & near
        occ = m['state'].reshape(-1) == 2
        rows.append({'map': str(mp), 'band_m': a.band_m, 'predicted_occupied': int(occ.sum()),
                     'occupied_outside_band': float(np.mean(d[occ] < -a.band_m)) if occ.any() else None,
                     'occupied_in_surface_band': float(np.mean(np.abs(d[occ]) <= a.band_m)) if occ.any() else None,
                     'occupied_interior_beyond_band': float(np.mean(d[occ] > a.band_m)) if occ.any() else None,
                     'visible_surface_voxels': int(surface.sum()),
                     'visible_surface_recall': float(np.mean(occ[surface])) if surface.any() else None})
    a.out.write_text(json.dumps(rows, indent=2) + '\n')
    for r in rows:
        print(Path(r['map']).parent.name, r['predicted_occupied'], *(f"{r[k]:.3f}" for k in
              ['occupied_outside_band', 'occupied_in_surface_band', 'occupied_interior_beyond_band', 'visible_surface_recall']))


if __name__ == '__main__':
    main()
