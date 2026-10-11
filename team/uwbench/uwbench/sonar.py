"""Forward-looking sonar geometry shared by A (range-bearing factor) and B (elevation candidates).

Sonar frame: x forward, y to bearing-positive side, z completing a right-handed frame.
Range is the slant range; elevation is not observed and must never be assumed zero.
"""
import numpy as np


def range_bearing(points):
    """Slant range rho = |p| and bearing theta = atan2(y, x) of sonar-frame points."""
    p = np.asarray(points, dtype=float)
    return np.linalg.norm(p, axis=-1), np.arctan2(p[..., 1], p[..., 0])


def arc_points(rho, theta, elevations):
    """q(rho, theta, phi) for each elevation phi; one echo maps to this whole arc."""
    phi = np.asarray(elevations, dtype=float)
    return np.stack([rho*np.cos(phi)*np.cos(theta), rho*np.cos(phi)*np.sin(theta), rho*np.sin(phi)], axis=-1)


def arc_height_span(rho, vertical_aperture_rad):
    """Vertical extent 2 rho sin(alpha/2) of one echo's elevation arc."""
    if np.any(np.asarray(vertical_aperture_rad) <= 0):
        raise ValueError('Vertical aperture must be measured and positive')
    return 2*np.asarray(rho, dtype=float)*np.sin(np.asarray(vertical_aperture_rad)/2)


def effect_size_row(rho, vertical_aperture_rad, voxel_m):
    """B plan 5.3 pre-estimate: arc span and the voxel layers B2a marks occupied per echo."""
    if voxel_m <= 0:
        raise ValueError('Voxel size must be positive')
    span = float(arc_height_span(rho, vertical_aperture_rad))
    return {'range_m': float(rho), 'vertical_aperture_deg': float(np.degrees(vertical_aperture_rad)),
            'arc_height_span_m': span, 'voxel_m': float(voxel_m),
            'b2a_vertical_voxels_per_echo': int(np.ceil(span/voxel_m - 1e-9)),
            'approximation': 'vertical extent only; ignores horizontal beam width and voxel grid phase'}
