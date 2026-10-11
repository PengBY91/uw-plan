"""Body-axis errors and exceedance time for the topic-A v2 detection track.

Errors are expressed on the estimated body axes x, y, z and world yaw, because an online
detector only knows the estimated attitude. Yaw error is the world-z component of
Log(R_est R_ref^T); near level flight it equals the yaw difference.
Time is integrated over sample holds; holds longer than max_gap are unavailable, never bridged.
"""
import numpy as np
from scipy.spatial.transform import Rotation

AXES = ('x', 'y', 'z', 'yaw')


def heading(R):
    return np.arctan2(R[..., 1, 0], R[..., 0, 0])


def align_yaw_translation(ref_R, ref_p, est_R, est_p, window):
    """Gravity-preserving 4-DoF alignment from a predeclared window; uses orientation, not just path."""
    window = np.asarray(window, dtype=bool)
    if window.sum() < 2:
        raise ValueError('Alignment window needs at least two matched poses')
    offsets = heading(ref_R[window] @ np.swapaxes(est_R[window], 1, 2))
    yaw = float(np.arctan2(np.sin(offsets).mean(), np.cos(offsets).mean()))
    R_align = Rotation.from_euler('z', yaw).as_matrix()
    t_align = (ref_p[window] - est_p[window] @ R_align.T).mean(axis=0)
    residual = Rotation.from_matrix(ref_R[window] @ np.swapaxes(R_align @ est_R[window], 1, 2)).as_rotvec()
    return R_align, t_align, {'yaw_deg': float(np.degrees(yaw)),
                              'window_residual_tilt_deg_max': float(np.degrees(np.abs(residual[:, :2]).max()))}


def body_axis_errors(ref_R, ref_p, est_R, est_p):
    """Rows of [x, y, z] in metres on estimated body axes, then yaw in degrees."""
    position = np.einsum('nji,nj->ni', est_R, est_p - ref_p)
    rotation = Rotation.from_matrix(est_R @ np.swapaxes(ref_R, 1, 2)).as_rotvec()
    return np.column_stack([position, np.degrees(rotation[:, 2])])


def body_axis_sigma(est_R, covariance):
    """OpenVINS pose covariance: world position, then local IMU-frame orientation error."""
    covariance = np.asarray(covariance, dtype=float)
    position = np.einsum('nji,njk,nkl->nil', est_R, covariance[:, :3, :3], est_R)
    world_rotation = est_R @ covariance[:, 3:, 3:] @ np.swapaxes(est_R, 1, 2)
    variances = np.column_stack([np.diagonal(position, axis1=1, axis2=2), world_rotation[:, 2, 2]])
    if (variances < -1e-12).any():
        raise ValueError('Covariance is not positive semidefinite on body axes')
    sigma = np.sqrt(np.clip(variances, 0, None))
    sigma[:, 3] = np.degrees(sigma[:, 3])
    return sigma


def exceedance(times, values, tolerance, max_gap):
    """Integrate time above |tolerance|; holds longer than max_gap count as unavailable."""
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    if len(times) < 2 or not np.all(np.diff(times) > 0) or not np.isfinite(values).all():
        raise ValueError('Exceedance requires a nonempty increasing finite series')
    if tolerance <= 0 or max_gap <= 0:
        raise ValueError('Invalid tolerance or gap')
    dt = np.diff(times)
    available = dt <= max_gap
    over = np.abs(values[:-1]) > tolerance
    intervals = []
    for i in np.flatnonzero(available & over):
        if intervals and intervals[-1][1] == times[i]:
            intervals[-1][1] = times[i+1]
        else:
            intervals.append([times[i], times[i+1]])
    return {'tolerance': float(tolerance), 'span_s': float(times[-1]-times[0]),
            'exceed_s': float(dt[available & over].sum()), 'within_s': float(dt[available & ~over].sum()),
            'unavailable_s': float(dt[~available].sum()), 'max_gap_s': float(max_gap),
            'exceed_intervals': [{'start': float(a), 'end': float(b), 'duration_s': float(b-a)} for a, b in intervals]}
