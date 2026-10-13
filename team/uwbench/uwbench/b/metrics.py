"""Surface-based and visibility-aware map metrics for B (review step 3, 2026-10-12).

eval-map's section-6 ratios use truth_from_boxes, whose true-occupied set is the box interior beyond the exclusion
band, and an ROI that includes the slab behind the structure and the margins past the route ends. A camera or sonar
sees surfaces, not interiors, and cannot see behind a wall from the route, so:
- occupied recall and the structure error reward smearing the surface inward;
- effective coverage is dominated by volume no sensor on the route could observe.
This module gives the complementary view: where the predicted occupied voxels lie relative to the box surfaces
(outside, surface band, interior), the recall of the surface band on the faces turned toward the route, and a
visibility mask that removes voxels shadowed by a box (and box interiors beyond the band) from the denominators.
The route-facing side is approximated per box by the sign of (vehicle median y - box centre y), which holds for the
simulated routes (vehicle at y < 0, props at y about 0); the x ends of a box count as visible.
"""
import numpy as np


def signed_depth(points, boxes):
    """> 0: depth inside the nearest containing box; < 0: minus the distance to the nearest box when outside all."""
    points = np.asarray(points, dtype=float)
    outside = np.full(len(points), np.inf)
    inside = np.zeros(len(points))
    for lo, hi in boxes:
        gap = np.linalg.norm(np.maximum(np.maximum(lo - points, points - hi), 0), axis=1)
        outside = np.minimum(outside, gap)
        ins = np.all((points >= lo) & (points <= hi), axis=1)
        inside = np.where(ins, np.maximum(inside, np.minimum(points - lo, hi - points).min(axis=1)), inside)
    return np.where(inside > 0, inside, -outside)


def route_facing_surface(points, boxes, band, y_vehicle):
    """Voxels within band of a box surface, on the side of the box's centre plane that faces the vehicle (plus the x ends)."""
    points = np.asarray(points, dtype=float)
    d = signed_depth(points, boxes)
    near = np.zeros(len(points), bool)
    for lo, hi in boxes:
        ymid = 0.5*(lo[1] + hi[1])
        side = (points[:, 1] - ymid)*np.sign(y_vehicle - ymid) >= 0
        inb = np.all((points >= lo - band) & (points <= hi + band), axis=1)
        near |= inb & side
    return (np.abs(d) <= band) & near


def shadowed(points, boxes, y_vehicle):
    """Voxels a sensor on the route (at y = y_vehicle, looking along y) cannot see: behind a box's far face within the
    box's x-z footprint, or inside a box. Occlusion by other boxes along oblique rays is not modelled."""
    points = np.asarray(points, dtype=float)
    out = np.zeros(len(points), bool)
    for lo, hi in boxes:
        toward = np.sign(lo[1] + hi[1] - 2*y_vehicle)  # +1 when the box lies at larger y than the vehicle
        far = hi[1] if toward > 0 else lo[1]
        footprint = (points[:, 0] >= lo[0]) & (points[:, 0] <= hi[0]) & (points[:, 2] >= lo[2]) & (points[:, 2] <= hi[2])
        behind = (points[:, 1] - far)*toward > 0
        inside = np.all((points >= lo) & (points <= hi), axis=1)
        out |= footprint & (behind | inside)
    return out


def visible_roi(points, boxes, band, y_vehicle):
    """Voxels that stay in the denominators: not shadowed, and not box interior beyond the surface band."""
    d = signed_depth(points, boxes)
    return ~shadowed(points, boxes, y_vehicle) & ~(d > band)


def occupied_placement(d, occupied, band):
    """Where the predicted occupied voxels lie: fractions outside the band, in the surface band, interior beyond it."""
    n = int(np.count_nonzero(occupied))
    if not n:
        return {'predicted_occupied': 0, 'outside_band': None, 'surface_band': None, 'interior_beyond_band': None}
    dd = d[occupied]
    return {'predicted_occupied': n, 'outside_band': float(np.mean(dd < -band)),
            'surface_band': float(np.mean(np.abs(dd) <= band)), 'interior_beyond_band': float(np.mean(dd > band))}


def surface_recall(occupied, surface):
    """Fraction of route-facing surface-band voxels predicted occupied; None without surface voxels."""
    return float(np.mean(occupied[surface])) if surface.any() else None
