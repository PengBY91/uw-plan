"""Sonar map updates on the B0 VoxelGrid (P09). Image layout: [RangeBins, AzimuthBins], row 0 at RangeMin, column 0 at
-Azimuth/2; sonar frame x forward, y left, z up. An echo is an image cell above a fixed threshold.

B1  diagnostic: each echo is put at elevation 0 (the horizontal-projection assumption) and marked occupied.
B2a conservative: each echo marks its whole elevation arc occupied (thick but safe); missing echoes never clear.
B2c Two-Stage reproduction: each echo's full 3-D reflection region (range bin x azimuth bin x elevation aperture) gets
    an independent log-odds hit; inside each azimuth column the region in front of the first echo is verified free;
    behind the first echo is occluded (counted, never updated); columns without any echo are not updated.
Sonar veto V (map config veto: true, any sonar method): every voxel a sonar update has hit refuses later stereo
free-space updates (B plan 3.4: the stereo ray "saw" free space where the sonar arc says something may be). V is the
only source of a false-free reduction claim and is ablated on its own; with V off the host method is unchanged.
The vertical aperture is a parameter: the simulated value is a simulation setting, not a platform measurement.
"""
import numpy as np

from uwbench.mapping import SOURCE_SONAR
from uwbench.sonar import arc_points


def echo_cells(image, threshold):
    rows, cols = np.nonzero(image > threshold)
    return rows, cols


def cell_geometry(geometry, shape):
    nr, na = shape
    dr = (geometry['RangeMax'] - geometry['RangeMin'])/nr
    da = np.radians(geometry['Azimuth'])/na
    rho = lambda r: geometry['RangeMin'] + (np.asarray(r) + 0.5)*dr
    theta = lambda c: -np.radians(geometry['Azimuth'])/2 + (np.asarray(c) + 0.5)*da
    return rho, theta, dr, da


def _world(T_world_sonar, points):
    return points @ T_world_sonar[:3, :3].T + T_world_sonar[:3, 3]


def _voxels(grid, points):
    idx, inside = grid.index(points)
    return np.unique(idx[inside], axis=0) if inside.any() else np.zeros((0, 3), int)


def _region(rho_lo, rho_hi, th_lo, th_hi, ph_lo, ph_hi, step):
    """Sample points filling a sonar-frame spherical cell with spacing <= step."""
    nr = max(int(np.ceil((rho_hi - rho_lo)/step)) + 1, 2)
    r = np.linspace(rho_lo, rho_hi, nr)
    nt = max(int(np.ceil(rho_hi*(th_hi - th_lo)/step)) + 1, 2)
    na = max(int(np.ceil(rho_hi*(ph_hi - ph_lo)/step)) + 1, 2)
    R, Th, Ph = np.meshgrid(r, np.linspace(th_lo, th_hi, nt), np.linspace(ph_lo, ph_hi, na), indexing='ij')
    return np.stack([R*np.cos(Ph)*np.cos(Th), R*np.cos(Ph)*np.sin(Th), R*np.sin(Ph)], axis=-1).reshape(-1, 3)


class SonarUpdater:
    def __init__(self, grid, geometry, aperture_rad, threshold, method, hit=None, miss=None, free_band_m=None):
        if method not in ('b1', 'b2a', 'b2c'):
            raise ValueError('Unknown sonar method ' + str(method))
        if aperture_rad <= 0:
            raise ValueError('Vertical aperture must be positive')
        self.grid, self.g, self.aperture, self.threshold, self.method = grid, geometry, aperture_rad, threshold, method
        self.hit = grid.hit if hit is None else hit
        self.miss = grid.miss if miss is None else miss
        self.band = grid.resolution if free_band_m is None else free_band_m
        self.occluded = np.zeros(grid.shape, np.int32)
        self.stats = {'frames': 0, 'echo_cells': 0, 'hit_voxel_updates': 0, 'free_voxel_updates': 0, 'occluded_marks': 0}

    def update(self, image, T_world_sonar, stamp):
        rows, cols = echo_cells(image, self.threshold)
        self.stats['frames'] += 1
        self.stats['echo_cells'] += int(len(rows))
        if not len(rows):
            return
        rho, theta, dr, da = cell_geometry(self.g, image.shape)
        step = self.grid.resolution/2
        if self.method == 'b1':
            pts = arc_points(rho(rows), theta(cols), [0.0])
            hits = _voxels(self.grid, _world(T_world_sonar, np.asarray(pts).reshape(-1, 3)))
            free = np.zeros((0, 3), int)
        elif self.method == 'b2a':
            pts = []
            for r, c in zip(rho(rows), theta(cols)):
                n = max(int(np.ceil(r*self.aperture/step)) + 1, 2)
                pts.append(arc_points(r, c, np.linspace(-self.aperture/2, self.aperture/2, n)))
            hits = _voxels(self.grid, _world(T_world_sonar, np.concatenate(pts)))
            free = np.zeros((0, 3), int)
        else:
            pts, free_pts, occ_pts = [], [], []
            half = self.aperture/2
            first = {}
            for r, c in zip(rows, cols):
                first[c] = min(first.get(c, r), r)
                rc, tc = rho(r), theta(c)
                pts.append(_region(rc - dr/2, rc + dr/2, tc - da/2, tc + da/2, -half, half, step))
            for c, r1 in first.items():
                tc, r_first = theta(c), rho(r1) - dr/2
                if r_first - self.band > self.g['RangeMin']:
                    free_pts.append(_region(self.g['RangeMin'], r_first - self.band, tc - da/2, tc + da/2, -half, half, step))
                occ_pts.append(_region(rho(r1) + dr/2, self.g['RangeMax'], tc - da/2, tc + da/2, -half, half, step))
            hits = _voxels(self.grid, _world(T_world_sonar, np.concatenate(pts)))
            free = _voxels(self.grid, _world(T_world_sonar, np.concatenate(free_pts))) if free_pts else np.zeros((0, 3), int)
            if len(free) and len(hits):   # a voxel that is a candidate in this frame is not verified free
                hit_set = {tuple(v) for v in hits}
                free = np.array([v for v in free if tuple(v) not in hit_set]).reshape(-1, 3)
            occ = _voxels(self.grid, _world(T_world_sonar, np.concatenate(occ_pts))) if occ_pts else np.zeros((0, 3), int)
            if len(occ):
                self.occluded[tuple(occ.T)] += 1
                self.stats['occluded_marks'] += int(len(occ))
        self.grid._apply(hits, self.hit, stamp, SOURCE_SONAR)
        self.stats['hit_voxel_updates'] += int(len(hits))
        if self.grid.veto is not None and len(hits):   # V: from now on these candidates block stereo free rays
            self.grid.veto[tuple(hits.T)] = True
        if len(free):
            self.grid._apply(free, self.miss, stamp, SOURCE_SONAR)
            self.stats['free_voxel_updates'] += int(len(free))
