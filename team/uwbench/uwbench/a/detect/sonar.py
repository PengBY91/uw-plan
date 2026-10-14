"""Sonar conflict sources for detect-a: first returns and lines from imaging-sonar frames, and the single-plane,
multi-plane and scan-to-map conflicts against the estimated poses (see the package docstring)."""
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from uwbench.a.detect.config import CONFIG, SIM_EPOCH_S
from uwbench.a.detect.signals import hold, rereference



def sonar_first_returns(image, geometry, threshold):
    """Sonar-frame (x forward, y left) points from the first return above threshold in each azimuth column.

    Image layout as HoloOcean ImagingSonar: [RangeBins, AzimuthBins], row 0 at RangeMin, column 0 at -Azimuth/2.
    """
    above = image > threshold
    has = above.any(axis=0)
    row = np.argmax(above, axis=0)[has]
    col = np.nonzero(has)[0]
    rng_m = geometry['RangeMin'] + (row + 0.5)*(geometry['RangeMax'] - geometry['RangeMin'])/image.shape[0]
    az = np.radians(-geometry['Azimuth']/2 + (col + 0.5)*geometry['Azimuth']/image.shape[1])
    return np.column_stack([rng_m*np.cos(az), rng_m*np.sin(az)])


def sonar_echo_thickness(image, geometry, threshold, gap_bins=3):
    """Median over azimuth columns of the echo's range extent from the first return, relative to that range.

    An imaging sonar does not measure elevation, so the first return comes from the nearest point within the elevation
    fan. On a vertical face the fan spans ranges D to D/cos(E/2) (relative extent 1.5% for E = 20 deg, plus one range
    bin); on a face tilted b > E/2 from vertical it spans D/cos(b - E/2) to D/cos(b + E/2), and the first return slides
    along the face when the range changes, which a 2-D model reads as wall motion. The run ends at the first gap of
    more than gap_bins rows. NaN without returns.
    """
    above = image > threshold
    cols = np.nonzero(above.any(axis=0))[0]
    if not len(cols):
        return np.nan
    bin_m = (geometry['RangeMax'] - geometry['RangeMin'])/image.shape[0]
    rel = []
    for c in cols:
        rows = np.nonzero(above[:, c])[0]
        gaps = np.nonzero(np.diff(rows) > gap_bins)[0]
        end = rows[gaps[0]] if len(gaps) else rows[-1]
        rel.append((end - rows[0] + 1)/(rows[0] + 0.5 + geometry['RangeMin']/bin_m))
    return float(np.median(rel))


def fit_line(points, inlier_m, min_inliers, rng, iterations=100):
    """RANSAC line m . p = d through 2-D points, m a unit normal pointing away from the origin; None if too few."""
    if len(points) < min_inliers:
        return None
    best = None
    for _ in range(iterations):
        a, b = points[rng.choice(len(points), 2, replace=False)]
        direction = b - a
        if np.linalg.norm(direction) < 1e-6:
            continue
        m = np.array([-direction[1], direction[0]])/np.linalg.norm(direction)
        inliers = np.abs(points @ m - a @ m) < inlier_m
        if best is None or inliers.sum() > best.sum():
            best = inliers
    if best is None or best.sum() < min_inliers:
        return None
    q = points[best]
    centre = q.mean(axis=0)
    m = np.linalg.eigh(np.cov((q - centre).T))[1][:, 0]  # least-squares refit on the inliers
    d = centre @ m
    return (m, d) if d > 0 else (-m, -d)


def sonar_lines(stamps, images, geometry, cfg=CONFIG, footprint_half_bin=None):
    """Per sonar frame: (t, m_x, m_y, d) of the dominant line, NaN where none or where the echo is too thick for a vertical face.

    footprint_half_bin (rad): when given, first returns are moved within their azimuth bin toward the surface before the
    fit (footprint_correct), the same correction the map variants get with sonar_footprint; None leaves them at the
    bin centres (all pilot rounds).
    """
    rng = np.random.default_rng(0)
    out = np.full((len(stamps), 4), np.nan)
    out[:, 0] = stamps
    for i, image in enumerate(images):
        if not sonar_echo_thickness(image, geometry, cfg['sonar_threshold']) <= cfg['sonar_max_thickness']:
            continue  # no vertical structure in view (inclined face or nothing): elevation ambiguity, no measurement
        points = sonar_first_returns(image, geometry, cfg['sonar_threshold'])
        if footprint_half_bin is not None:
            points = footprint_correct(points, footprint_half_bin, cfg['map_normal_k'])
        line = fit_line(points, cfg['sonar_inlier_m'], cfg['sonar_min_inliers'], rng)
        if line is not None:
            out[i, 1:3], out[i, 3] = line
    return out


def sonar_conflict(t, est, lines, T_imu_sonar, gate_deg, reference_s=0.0, hold_s=np.inf, window=None):
    """Cumulative conflict along a reference plane normal, projected on estimated body x and y, at stamps t.

    The reference plane averages the lines seen in the first reference_s after the first one. NaN (not scorable) until
    it is fixed and wherever the last measured frame is older than hold_s (gated or skipped frames are not measurements).
    window re-references the signed conflict to the alignment window (see rereference).
    """
    c, proj = sonar_conflict_signed(t, est, lines, T_imu_sonar, gate_deg, reference_s, hold_s)
    return np.abs(rereference(c, window))[:, None]*proj


def sonar_conflict_signed(t, est, lines, T_imu_sonar, gate_deg, reference_s=0.0, hold_s=np.inf):
    """Signed conflict along the reference normal (measured minus predicted distance, m) and the |R^T n_ref| body x, y
    projection factors, held at stamps t; NaN where not scorable."""
    est_t, R, p = est[0], est[1], est[2]
    R_is, t_is = T_imu_sonar[:3, :3], T_imu_sonar[:3, 3]
    value_t, values = [], []
    ref, window = None, []
    for ts, mx, my, d in lines:
        if not np.isfinite(d) or ts < est_t[0]:
            continue
        j = np.searchsorted(est_t, ts, side='right') - 1  # latest estimate at or before the frame
        p_s = p[j] + R[j] @ t_is
        n_w = R[j] @ R_is @ np.array([mx, my, 0.0])
        if ref is None:
            if window and np.degrees(np.arccos(np.clip(n_w @ window[0][1], -1, 1))) > gate_deg:
                continue
            window.append((ts, n_w, p_s, d))
            if ts - window[0][0] >= reference_s:
                n_ref = np.mean([w[1] for w in window], axis=0)
                n_ref /= np.linalg.norm(n_ref)
                ref = (n_ref, np.mean([n_ref @ w[2] + w[3] for w in window]))
            if ref is None:
                continue  # the frame that fixes the reference is measured against it below
        n_ref, h = ref
        if np.degrees(np.arccos(np.clip(n_w @ n_ref, -1, 1))) > gate_deg:
            continue
        value_t.append(ts)
        values.append([d - (h - n_ref @ p_s), *np.abs(R[j].T @ n_ref)[:2]])
    held = hold(value_t, values, t, hold_s, 3)
    return held[:, 0], held[:, 1:]


def sonar_conflict_multi(t, est, lines, T_imu_sonar, match_deg, jump_m, hold_s=np.inf):
    """Cumulative horizontal conflict over a growing set of local planes, projected on estimated body x and y.

    Each frame's line is associated with a known plane if the normals (through the estimated attitude) are within
    match_deg and the conflict on that plane moved by less than jump_m since the plane was last seen: drift is slow,
    a different surface jumps. Otherwise a new plane is fixed through the current estimated pose and inherits the
    current conflict vector, so changing surfaces does not reset what has accumulated. The conflict vector is
    carry + c * n of the plane in view; only its component along that normal is measured. NaN where the last
    associated frame is older than hold_s.
    """
    est_t, R, p = est[0], est[1], est[2]
    R_is, t_is = T_imu_sonar[:3, :3], T_imu_sonar[:3, 3]
    cos_match = np.cos(np.radians(match_deg))
    planes = []  # dicts: n, h, carry (3-vector, est world, horizontal), c_last
    current = np.zeros(3)
    value_t, values, counts = [], [], []
    for ts, mx, my, d in lines:
        if not np.isfinite(d) or ts < est_t[0]:
            continue
        j = np.searchsorted(est_t, ts, side='right') - 1  # latest estimate at or before the frame
        p_s = p[j] + R[j] @ t_is
        n_w = R[j] @ R_is @ np.array([mx, my, 0.0])
        best, best_jump = None, jump_m
        for plane in planes:
            if n_w @ plane['n'] < cos_match:
                continue
            c = d - (plane['h'] - plane['n'] @ p_s)
            if abs(c - plane['c_last']) < best_jump:
                best, best_jump, best_c = plane, abs(c - plane['c_last']), c
        if best is None:
            planes.append({'n': n_w, 'h': n_w @ p_s + d, 'carry': current.copy(), 'c_last': 0.0})
            value_t.append(ts)  # measured: zero on the new plane, so the conflict is the carried vector
            values.append(np.abs(R[j].T @ current)[:2])
            counts.append(len(planes))
            continue
        best['c_last'] = best_c
        current = best['carry'] + best_c*best['n']
        value_t.append(ts)
        values.append(np.abs(R[j].T @ current)[:2])
        counts.append(len(planes))
    return hold(value_t, values, t, hold_s, 2), len(planes)


def scan_normals(points, k=12):
    """Unit normals of 2-D scan points from PCA of their k nearest neighbours in the same scan."""
    _, nb = cKDTree(points).query(points, k=min(k, len(points)))
    X = points[nb] - points[nb].mean(axis=1, keepdims=True)
    return np.linalg.eigh(np.einsum('nki,nkj->nij', X, X))[1][:, :, 0]


def footprint_correct(points, half_bin, k=12):
    """Move each first return to the bearing in its azimuth bin nearest to the surface.

    The first return of a bin comes from the bin edge closest to the foot of the local surface normal (or from the foot
    itself if it lies inside the bin), not from the bin centre where the image puts it. At oblique incidence this puts
    points about r sin(incidence) * half_bin too close (1.6 cm at 60 deg and 4.5 m with 0.94 deg bins), bending the
    fan edges; a growing map that inserts them bends with them. Range is unchanged; the bearing moves by at most half_bin
    toward the foot of the normal estimated from the k nearest scan points.
    """
    if len(points) < 3:
        return points
    n = scan_normals(points, k)
    n = n*np.where(np.einsum('ij,ij->i', n, points) > 0, -1.0, 1.0)[:, None]  # normal toward the sensor
    r, th = np.hypot(points[:, 0], points[:, 1]), np.arctan2(points[:, 1], points[:, 0])
    foot = np.arctan2(-n[:, 1], -n[:, 0])
    th = th + np.clip(np.angle(np.exp(1j*(foot - th))), -half_bin, half_bin)
    return np.column_stack([r*np.cos(th), r*np.sin(th)])


def rot2(angle):
    c, s_ = np.cos(angle), np.sin(angle)
    return np.array([[c, -s_], [s_, c]])


def sonar_conflict_map(t, est, scans, T_imu_sonar, cfg=CONFIG):
    """Scan-to-map sonar conflict at stamps t: columns body-x and body-y position conflict (m), heading conflict (deg).

    scans: list of (stamp, [n, 2] sonar-frame points); NaN where the last registered scan is older than sonar_hold_s.
    The correction (phi, tau) maps estimated-world horizontal points into the map frame: q_map = rot(phi) q + tau.
    """
    est_t, R, p = est[0], est[1], est[2]
    R_is, t_is = T_imu_sonar[:3, :3], T_imu_sonar[:3, 3]
    phi, tau = 0.0, np.zeros(2)
    map_pts, map_nrm, voxels, tree = None, None, set(), None
    # per-scan random-walk prior on the correction: how far it may move between scans (drift, not measurement noise)
    w_t, w_phi = 1/cfg['map_prior_m']**2, 1/np.radians(cfg['map_prior_deg'])**2
    w_point = 1/cfg['map_sigma_m']**2
    huber = cfg['map_huber_m']
    value_t, values = [], []

    def add(points, normals):
        nonlocal map_pts, map_nrm, tree
        keys = np.floor(points/cfg['map_voxel_m']).astype(np.int64)
        keep = []
        for i, key in enumerate(map(tuple, keys)):
            if key not in voxels:
                voxels.add(key)
                keep.append(i)
        if keep:
            map_pts = points[keep] if map_pts is None else np.vstack([map_pts, points[keep]])
            map_nrm = normals[keep] if map_nrm is None else np.vstack([map_nrm, normals[keep]])
            tree = cKDTree(map_pts)

    for ts, pts in scans:
        if ts < est_t[0] or len(pts) < cfg['map_min_pairs']:
            continue
        j = np.searchsorted(est_t, ts, side='right') - 1
        c3 = p[j] + R[j] @ t_is
        q = (R[j] @ R_is @ np.column_stack([pts, np.zeros(len(pts))]).T).T[:, :2] + c3[:2]  # estimated world, horizontal
        nrm = scan_normals(pts, cfg['map_normal_k']) @ (R[j] @ R_is)[:2, :2].T
        if map_pts is None or (not cfg['map_grow'] and ts - map_t0 <= cfg['map_init_s']):
            # the first scans (map_init_s after the first) define the map, in the estimated world at t_0
            if map_pts is None:
                map_t0 = ts
            add(q, nrm)
            value_t.append(ts)  # the scans that define the map match it with zero correction
            values.append([0.0, 0.0, 0.0])
            continue
        c = c3[:2]
        registered = True
        for _ in range(cfg['map_iterations']):
            y = q @ rot2(phi).T + tau
            cy = rot2(phi) @ c + tau
            dist, idx = tree.query(y)
            if cfg['map_grow']:
                ok = dist < cfg['map_assoc_m']
            else:  # fixed map: a point beyond its edge is compared with the tangent of the nearest map point
                ok = np.abs(np.einsum('ij,ij->i', map_nrm[idx], y - map_pts[idx])) < cfg['map_assoc_m']
            if ok.sum() < cfg['map_min_pairs']:
                registered = False  # no overlap with the map: keep the correction, add nothing
                break
            n, m, yy = map_nrm[idx[ok]], map_pts[idx[ok]], y[ok]
            r = np.einsum('ij,ij->i', n, yy - m)
            lever = yy - cy
            J = np.column_stack([n, n[:, 0]*(-lever[:, 1]) + n[:, 1]*lever[:, 0]])
            w = w_point*np.where(np.abs(r) <= huber, 1.0, huber/np.abs(r))
            if not cfg['map_rotation']:
                J = J[:, :2]
            H_data = J.T @ (w[:, None]*J)
            prior = np.diag([w_t, w_t, w_phi][:J.shape[1]])
            delta = -np.linalg.solve(H_data + prior, J.T @ (w*r))
            if not cfg['map_rotation']:
                H_data, delta = np.pad(H_data, (0, 1)), np.append(delta, 0.0)
                H_data[2, 2] = 0.0
            # do not move along directions the scan barely constrains (a flat wall along itself): with quantised ranges
            # and noisy normals they carry spurious information that slides the registration
            S = np.diag([1.0, 1.0, 1/cfg['map_lever_m']])  # rotation expressed as metres at the lever arm
            lam, V = np.linalg.eigh(S @ H_data @ S)
            weak = V[:, lam < cfg['map_degenerate_ratio']*lam.max()]
            scaled = np.linalg.solve(S, delta)
            delta = S @ (scaled - weak @ (weak.T @ scaled))
            phi += delta[2]
            tau = rot2(delta[2]) @ (tau - cy) + cy + delta[:2]
        if registered:
            if cfg['map_grow']:
                add(q @ rot2(phi).T + tau, nrm @ rot2(phi).T)
            e = (rot2(phi) @ c + tau) - c
            value_t.append(ts)
            values.append([*np.abs(R[j].T @ np.array([e[0], e[1], 0.0]))[:2], abs(np.degrees(phi))])
    return hold(value_t, values, t, cfg.get('sonar_hold_s', np.inf), 3), (0 if map_pts is None else len(map_pts))


def load_sim_sonar_images(dataset):
    """(stamps, images, geometry, 4x4 sonar pose in the IMU frame) of a simulated dataset (raycast replay if present, else raw); None without."""
    dataset = Path(dataset)
    raw = dataset/'raw'
    if not (raw/'meta.json').exists():
        return None
    meta = json.loads((raw/'meta.json').read_text())
    source = dataset/'sonar_raycast' if (dataset/'sonar_raycast'/'sonar_t.npy').exists() else raw
    stamps = np.load(source/'sonar_t.npy') + SIM_EPOCH_S
    images = np.concatenate([np.load(f) for f in sorted(source.glob('sonar_[0-9]*.npy'))])
    return stamps, images, meta['route']['sensors']['sonar'], np.asarray(meta['socket_extrinsics_in_imu_frame']['sonar'])


def sonar_products(stamps, images, geometry, T_imu_sonar, cfg=CONFIG):
    """Sonar lines, extrinsic and first-return scans for the detector from loaded sonar images."""
    half_bin = np.radians(geometry['Azimuth']/geometry['AzimuthBins'])/2 if cfg.get('sonar_footprint') else None
    lines = sonar_lines(stamps, images, geometry, cfg, footprint_half_bin=half_bin)  # the x source gets the same correction as the maps
    scans = [(ts, sonar_first_returns(image, geometry, cfg['sonar_threshold'])
              if sonar_echo_thickness(image, geometry, cfg['sonar_threshold']) <= cfg['sonar_max_thickness'] else np.zeros((0, 2)))
             for ts, image in zip(stamps, images)]
    if half_bin is not None:
        scans = [(ts, footprint_correct(pts, half_bin, cfg['map_normal_k'])) for ts, pts in scans]
    return lines, T_imu_sonar, scans


def load_sim_sonar(dataset, cfg=CONFIG):
    """Sonar lines, extrinsic and first-return scans of a simulated dataset; None without sonar images."""
    loaded = load_sim_sonar_images(dataset)
    return None if loaded is None else sonar_products(*loaded, cfg=cfg)
