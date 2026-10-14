"""B0 tri-state voxel map (P05; log-odds and thresholds default to OctoMap's 0.7/0.4 and 0.5) on a dense grid, so B can later attach provenance and elevation candidates per voxel.

Only valid depth produces evidence. Invalid or out-of-range depth never becomes a free ray, space behind a hit is
never updated, and voxels without evidence stay unknown. Depth is metres; obviously millimetre input is rejected.
"""
import numpy as np

UNKNOWN, FREE, OCCUPIED = 0, 1, 2
SOURCE_STEREO, SOURCE_SONAR = 1, 2

# Camera optical frame (x right, y down, z forward) expressed in body axes (x forward, y left, z up).
R_BODY_OPTICAL = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=float)


class VoxelGrid:
    def __init__(self, origin, shape, resolution, hit=0.85, miss=-0.4, occupied_above=0.0, free_below=0.0,
                 clamp=(-2.0, 3.5)):
        if resolution <= 0:
            raise ValueError('Resolution must be positive')
        self.origin = np.asarray(origin, dtype=float)
        self.shape = tuple(int(s) for s in shape)
        self.resolution = float(resolution)
        self.hit, self.miss, self.clamp = hit, miss, clamp
        self.occupied_above, self.free_below = occupied_above, free_below
        self.log_odds = np.zeros(self.shape, np.float32)
        self.observed = np.zeros(self.shape, bool)
        self.last_update = np.full(self.shape, np.nan)
        self.sources = np.zeros(self.shape, np.uint8)
        self.veto = None          # sonar veto V (B plan 3.4): voxels holding a sonar candidate refuse stereo free updates
        self.vetoed_free_updates = 0

    def index(self, points):
        idx = np.floor((np.asarray(points) - self.origin)/self.resolution).astype(int)
        inside = np.all((idx >= 0) & (idx < self.shape), axis=-1)
        return idx, inside

    def centers(self):
        grid = np.stack(np.meshgrid(*[np.arange(s) for s in self.shape], indexing='ij'), axis=-1)
        return self.origin + (grid + 0.5)*self.resolution

    def state(self):
        s = np.full(self.shape, UNKNOWN, np.uint8)
        s[self.observed & (self.log_odds < self.free_below)] = FREE
        s[self.observed & (self.log_odds > self.occupied_above)] = OCCUPIED
        return s

    def _apply(self, voxels, delta, stamp, source):
        if not len(voxels):
            return
        v = tuple(np.unique(voxels, axis=0).T)
        self.log_odds[v] = np.clip(self.log_odds[v] + delta, *self.clamp)
        self.observed[v] = True
        self.last_update[v] = stamp
        self.sources[v] |= source

    def integrate_rays(self, origin, endpoints, stamp, source=SOURCE_STEREO, endpoint_band=None):
        """Free along each ray up to the endpoint band, occupied at the endpoint; nothing behind it."""
        origin, endpoints = np.asarray(origin, dtype=float), np.asarray(endpoints, dtype=float).reshape(-1, 3)
        if not len(endpoints):
            return
        band = self.resolution if endpoint_band is None else endpoint_band
        vectors = endpoints - origin
        lengths = np.linalg.norm(vectors, axis=1)
        step = self.resolution/3
        samples = np.arange(0, lengths.max(), step)
        points = origin + vectors[:, None, :]/lengths[:, None, None]*samples[None, :, None]
        keep = samples[None, :] < (lengths[:, None] - band)
        free_idx, inside = self.index(points[keep])
        hit_idx, hit_inside = self.index(endpoints)
        hit_set = {tuple(i) for i in hit_idx[hit_inside]}
        free = np.array([i for i in free_idx[inside] if tuple(i) not in hit_set]).reshape(-1, 3)
        if self.veto is not None and len(free):
            free = np.unique(free, axis=0)
            blocked = self.veto[tuple(free.T)]
            self.vetoed_free_updates += int(blocked.sum())
            free = free[~blocked]
        self._apply(free, self.miss, stamp, source)
        self._apply(hit_idx[hit_inside], self.hit, stamp, source)

    def integrate_depth(self, T_world_body, T_body_camera, K, depth_m, valid, stamp, max_range):
        depth_m = np.asarray(depth_m, dtype=float)
        valid = np.asarray(valid, bool) & np.isfinite(depth_m) & (depth_m > 0)
        if valid.any() and np.median(depth_m[valid]) > 100:  # before range gating, or mm input is silently dropped
            raise ValueError('Depth looks like millimetres; convert to metres before mapping')
        valid &= depth_m <= max_range
        v, u = np.nonzero(valid)
        z = depth_m[v, u]
        optical = np.column_stack([(u + 0.5 - K[0, 2])*z/K[0, 0], (v + 0.5 - K[1, 2])*z/K[1, 1], z])
        T = T_world_body @ T_body_camera
        endpoints = optical @ T[:3, :3].T + T[:3, 3]
        self.integrate_rays(T[:3, 3], endpoints, stamp, SOURCE_STEREO)


def truth_from_boxes(grid, boxes, band):
    """Reference labels: 'occ' inside boxes, 'free' elsewhere, 'excluded' within band of any box surface."""
    c = grid.centers()
    occ = np.zeros(grid.shape, bool)
    distance = np.full(grid.shape, np.inf)
    for lo, hi in boxes:
        lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
        inside = np.all((c >= lo) & (c <= hi), axis=-1)
        occ |= inside
        outside_gap = np.linalg.norm(np.maximum(np.maximum(lo - c, c - hi), 0), axis=-1)
        inside_gap = np.min(np.minimum(c - lo, hi - c), axis=-1)
        distance = np.minimum(distance, np.where(inside, inside_gap, outside_gap))
    labels = np.where(occ, 'occ', 'free').astype(object)
    labels[distance < band] = 'excluded'
    return labels


def predicted_labels(grid):
    return np.array(['unknown', 'free', 'occ'], dtype=object)[grid.state()]
