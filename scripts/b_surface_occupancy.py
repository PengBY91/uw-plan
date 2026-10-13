"""Supplementary B map metric (now also inside eval-map via uwbench.b.metrics): occupancy on the structure surface.

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


from uwbench.b.metrics import occupied_placement, route_facing_surface, signed_depth, surface_recall  # noqa: E402


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
        surface = route_facing_surface(centres, boxes, a.band_m, y_vehicle)
        occ = m['state'].reshape(-1) == 2
        place = occupied_placement(d, occ, a.band_m)
        rows.append({'map': str(mp), 'band_m': a.band_m, 'predicted_occupied': place['predicted_occupied'],
                     'occupied_outside_band': place['outside_band'], 'occupied_in_surface_band': place['surface_band'],
                     'occupied_interior_beyond_band': place['interior_beyond_band'],
                     'visible_surface_voxels': int(surface.sum()), 'visible_surface_recall': surface_recall(occ, surface)})
    a.out.write_text(json.dumps(rows, indent=2) + '\n')
    for r in rows:
        print(Path(r['map']).parent.name, r['predicted_occupied'], *(f"{r[k]:.3f}" for k in
              ['occupied_outside_band', 'occupied_in_surface_band', 'occupied_interior_beyond_band', 'visible_surface_recall']))


if __name__ == '__main__':
    main()
