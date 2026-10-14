"""uwbench motion-b (P19): E-B2 in simulation, the motion-information question of B's hypothesis H-B2.

A forward-looking sonar echo fixes range and bearing; its elevation is anywhere in the vertical aperture, so one echo
is an arc of candidate positions. H-B2 says how much that candidate set can shrink under later views is predicted by
the motion primitive (heave, pitch, roll against translation and yaw), the vision-sonar association and the aperture.
This tool measures the motion part with ground-truth poses standing in for the rail / turntable of the plan's stage 0:

- reference frame: every echo cell gets `candidates` elevation samples on its arc (sonar-frame points at the bin
  centre range and bearing), lifted to the world with the reference pose;
- each later frame: every candidate is projected into that frame and tested under two rules, reported side by side:
  * conservative (B's semantics, Two-Stage's "verified free in front of the first echo"): a candidate is eliminated
    only when it lies inside the fan and more than `tolerance_bins` nearer than the first echo of its azimuth column
    (within `tolerance_bins` columns). A candidate behind the first echo is occluded, a column without an echo gives
    no evidence: both survive. Missing echoes never clear;
  * aggressive (space-carving style): a candidate inside the fan whose own cell (within tolerance) holds no echo is
    eliminated, whatever the rest of the column shows. This is the rule the plan declines to use online (occlusion and
    weak reflectors clear real structure); it is kept as the upper bound on what motion could give.
  Survivors accumulate as an intersection over views;
- shrinkage = surviving candidates / all. Predicted shrinkage uses only the poses: for each candidate taken as the
  sole reflector, count the candidates the rule would keep; average over a uniform prior on the reflector. No image
  is read for the prediction. Translation separates candidates by range (heave, surge) or hardly at all (sway); a
  rotation about the sonar centre keeps every candidate in its cell and separates nothing under the conservative rule
  (only the lever arm between rotation centre and sonar translates it), and under the aggressive rule only through
  the fan edge;
- vision-sonar association (config stereo: true; datasets with a stereo pair): the B0 stereo depth of the nearest
  left/right pair projects every candidate into the left camera. A candidate more than a depth-dependent margin in
  front of the stereo surface lies in space the camera has verified free and is eliminated; one behind the surface
  (occluded) or without valid depth is kept. stereo_ref uses the reference frame only (same-time association, no
  motion), stereo adds every later view, conservative+stereo also applies the conservative sonar rule. Candidates
  within the margin of the surface are 'supported' (reported, never used to collapse the arc);
- truth, for scoring only: the simulator's raycast hits that fell into the reference cell give the true elevations
  (an arc may hold several reflectors, e.g. two beams at one range and bearing). `truth_retained` = no true elevation
  was eliminated; `precision` = share of survivors within `truth_margin_deg` of a true elevation; `clusters` = number
  of contiguous surviving elevation runs.

Input: a dataset manifest (SimDataset) or `--raw <dataset dir>` holding raw/meta.json, raw/gt_imu.npy and a raycast
sonar in raw/ or sonar_raycast/ (the teleported turntable sequences have only these). Outputs motion_info.csv (one row
per echo and view) and summary.json (aggregates and the motion of the sonar relative to the reference pose, in the
reference sonar frame: x along the beam, y lateral, z up, roll / pitch / yaw in degrees).
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from uwbench.sonar import arc_points, elevation, range_bearing

DEFAULTS = {'reference_t': 6.0, 'view_stride': 5, 'candidates': 81, 'tolerance_bins': 1, 'threshold': 0.25,
            'truth_margin_deg': 0.5, 'max_echoes': 400, 'seed': 0,
            # vision-sonar association (off by default; needs a manifest dataset with stereo images)
            'stereo': False, 'stereo_sigma_disparity_px': 0.5, 'stereo_k_sigma': 3.0, 'stereo_floor_m': 0.05,
            'stereo_window_px': 5, 'stereo_min_valid_fraction': 0.5, 'stereo_max_dt_s': 0.025}
RULES = ('conservative', 'aggressive')
STEREO_RULES = ('stereo_ref', 'stereo', 'conservative+stereo')


def stereo_test(world, T_world_cam, depth, K, baseline, cfg):
    """(eliminated, supported, testable) for world points against one stereo depth image.

    A candidate is testable when it projects into the image in front of the camera and at least
    stereo_min_valid_fraction of the window around its pixel has valid depth. Its stereo depth Z_s is the window
    minimum (a nearer edge in the window makes elimination harder, never easier). Eliminated: the candidate lies more
    than margin in front of Z_s, i.e. in space the stereo ray crossed before reaching a surface (verified free).
    Supported: within margin of Z_s. Behind the surface (occluded) or untestable: kept. margin =
    max(floor, k * Z_s^2 * sigma_d / (fx * B))."""
    fx, fy, cx, cy = K
    p = (world - T_world_cam[:3, 3]) @ T_world_cam[:3, :3]
    z = p[:, 2]
    h, w = depth.shape
    with np.errstate(divide='ignore', invalid='ignore'):
        u = np.floor(fx*p[:, 0]/z + cx).astype(int)
        v = np.floor(fy*p[:, 1]/z + cy).astype(int)
    half = cfg['stereo_window_px']//2
    elim, supp, test = np.zeros(len(p), bool), np.zeros(len(p), bool), np.zeros(len(p), bool)
    for i in np.nonzero((z > 0.1) & (u >= half) & (u < w - half) & (v >= half) & (v < h - half))[0]:
        win = depth[v[i] - half:v[i] + half + 1, u[i] - half:u[i] + half + 1]
        ok = np.isfinite(win)
        if ok.mean() < cfg['stereo_min_valid_fraction']:
            continue
        zs = float(win[ok].min())
        margin = max(cfg['stereo_floor_m'], cfg['stereo_k_sigma']*zs*zs*cfg['stereo_sigma_disparity_px']/(fx*baseline))
        test[i] = True
        elim[i] = z[i] < zs - margin
        supp[i] = abs(z[i] - zs) <= margin
    return elim, supp, test


class RawDataset:
    """Minimal dataset for stage-0 sequences: raw/meta.json + raw/gt_imu.npy + raycast sonar (sonar_raycast/ or raw/)."""
    def __init__(self, root):
        self.root = Path(root)
        self.raw = self.root/'raw'
        self.meta = json.loads((self.raw/'meta.json').read_text())
        self.route = self.meta['route']
        self.gt = np.load(self.raw/'gt_imu.npy')
        self.T_imu_sonar = np.asarray(self.meta['socket_extrinsics_in_imu_frame']['sonar'], dtype=float)
        self.sonar_geometry = self.route['sensors']['sonar']
        src = self.root/'sonar_raycast' if (self.root/'sonar_raycast'/'sonar_t.npy').exists() else self.raw
        self.sonar_t = np.load(src/'sonar_t.npy')
        self._src = src
        from uwbench.b.cli import chunked_images
        self.image = chunked_images(src)

    def gt_pose(self, t):
        i = int(round(t*self.route['ticks_per_sec']))
        if abs(self.gt[i, 0] - t) > 1e-6:
            raise ValueError('No ground-truth tick at %.6f' % t)
        return self.gt[i, 1:].reshape(4, 4)

    def sonar_points_path(self):
        for p in [self.root/'sonar_raycast/sonar_points.npy', self.raw/'sonar_points.npy']:
            if p.exists():
                return p
        return None


def cells(rho, theta, geometry, shape):
    """Image cell indices (row, col) of sonar-frame (rho, theta), floor-binned as render_sonar does; may be out of range."""
    nr, na = shape
    r = np.floor((rho - geometry['RangeMin'])/(geometry['RangeMax'] - geometry['RangeMin'])*nr).astype(int)
    a = np.floor((np.degrees(theta) + geometry['Azimuth']/2)/geometry['Azimuth']*na).astype(int)
    return r, a


def in_fan(rho, theta, phi, geometry):
    return ((rho >= geometry['RangeMin']) & (rho <= geometry['RangeMax']) & (np.abs(np.degrees(theta)) <= geometry['Azimuth']/2)
            & (np.abs(np.degrees(phi)) <= geometry['Elevation']/2))


def project(points_world, T_world_sonar):
    p = (points_world - T_world_sonar[:3, 3]) @ T_world_sonar[:3, :3]
    rho, theta = range_bearing(p)
    return rho, theta, elevation(p)


def first_echo_rows(image):
    """Row of the first echo per column (image boolean), -1 where the column has none."""
    has = image.any(axis=0)
    first = np.where(has, np.argmax(image, axis=0), -1)
    return first


def echo_present(image, r, a, tol):
    """True where any cell within tol bins of (r, a) is above threshold (image is boolean)."""
    nr, na = image.shape
    out = np.zeros(len(r), bool)
    for dr in range(-tol, tol + 1):
        for da in range(-tol, tol + 1):
            rr, aa = r + dr, a + da
            ok = (rr >= 0) & (rr < nr) & (aa >= 0) & (aa < na)
            out[ok] |= image[rr[ok], aa[ok]]
    return out


def in_front_of_first_echo(first, r, a, tol):
    """(eliminated, testable) for candidates at cells (r, a): testable when a column within tol has an echo; eliminated
    when the candidate is more than tol rows nearer than the nearest first echo among those columns."""
    na = len(first)
    nearest = np.full(len(r), np.inf)
    for da in range(-tol, tol + 1):
        aa = a + da
        ok = (aa >= 0) & (aa < na)
        f = np.where(ok, first[np.clip(aa, 0, na - 1)], -1)
        has = ok & (f >= 0)
        nearest[has] = np.minimum(nearest[has], f[has])
    testable = np.isfinite(nearest)
    return testable & (r < nearest - tol), testable


def relative_motion(T_ref, T_k):
    """Motion of the sonar from the reference pose, in the reference sonar frame: dx (along beam), dy (lateral), dz (up), roll, pitch, yaw (deg)."""
    rel = np.linalg.inv(T_ref) @ T_k
    rpy = Rotation.from_matrix(rel[:3, :3]).as_euler('xyz', degrees=True)
    return {'dx_m': float(rel[0, 3]), 'dy_m': float(rel[1, 3]), 'dz_m': float(rel[2, 3]),
            'roll_deg': float(rpy[0]), 'pitch_deg': float(rpy[1]), 'yaw_deg': float(rpy[2])}


def predicted_survivors(signatures, testable, rule, tol=1):
    """Survivors per hypothesised sole reflector j, from the poses alone. signatures [n, V, 2] (row, col); testable [n, V] = inside the fan.

    aggressive: i survives view v unless i is in the fan and (j is out of the fan or i's cell is more than tol from j's).
    conservative: i survives unless both are in the fan, i's column is within tol of j's and i is more than tol rows nearer.
    """
    both = testable[:, None, :] & testable[None, :, :]  # [i, j, v]
    d = signatures[:, None, :, :] - signatures[None, :, :, :]  # i - j
    if rule == 'aggressive':
        close = np.all(np.abs(d) <= tol, axis=-1)
        keep = ~testable[:, None, :] | (both & close)
    else:
        same_col = np.abs(d[..., 1]) <= tol
        nearer = d[..., 0] < -tol
        keep = ~(both & same_col & nearer)
    return np.all(keep, axis=-1).sum(axis=0)


def runs_count(mask):
    m = np.asarray(mask, bool)
    return int(np.count_nonzero(m[1:] & ~m[:-1]) + (1 if len(m) and m[0] else 0))


def analyse(ds, image, cfg):
    """Candidate tracking over one dataset; returns rows, per-view motion and run info. image(i) -> sonar image i."""
    g = ds.sonar_geometry
    aperture = np.radians(g['Elevation'])
    sonar_t = ds.sonar_t
    ref = int(np.argmin(np.abs(sonar_t - cfg['reference_t'])))
    views = list(range(ref + cfg['view_stride'], len(sonar_t), cfg['view_stride']))
    T_ref = ds.gt_pose(float(sonar_t[ref])) @ ds.T_imu_sonar
    img_ref = np.asarray(image(ref)) > cfg['threshold']
    shape = img_ref.shape
    rows_e, cols_e = np.nonzero(img_ref)
    rng = np.random.default_rng(cfg['seed'])
    if len(rows_e) > cfg['max_echoes']:
        pick = np.sort(rng.choice(len(rows_e), cfg['max_echoes'], replace=False))
        rows_e, cols_e = rows_e[pick], cols_e[pick]
    dr = (g['RangeMax'] - g['RangeMin'])/shape[0]
    da = np.radians(g['Azimuth'])/shape[1]
    rho_c = g['RangeMin'] + (rows_e + 0.5)*dr
    th_c = -np.radians(g['Azimuth'])/2 + (cols_e + 0.5)*da
    phis = np.linspace(-aperture/2, aperture/2, cfg['candidates'])
    phis_deg = np.degrees(phis)
    tol = cfg['tolerance_bins']
    truth = {}  # reference cell -> true elevations (deg) of the raycast hits in it
    path = ds.sonar_points_path()
    if path is not None:
        pts = np.load(path, mmap_mode='r')
        sel = np.asarray(pts[:, 0]).astype(int) == ref
        hits = np.asarray(pts[sel, 1:], dtype=float)
        if len(hits):
            hr, ha = cells(*range_bearing(hits), g, shape)
            he = np.degrees(elevation(hits))
            for r, a in zip(rows_e, cols_e):
                m = (hr == r) & (ha == a)
                if m.any():
                    truth[(int(r), int(a))] = he[m]
    motions, T_views = [], []
    for k in views:
        T_k = ds.gt_pose(float(sonar_t[k])) @ ds.T_imu_sonar
        T_views.append(T_k)
        motions.append({'view': k, 't_s': float(sonar_t[k]), **relative_motion(T_ref, T_k)})
    images_k = [np.asarray(image(k)) > cfg['threshold'] for k in views]
    firsts_k = [first_echo_rows(img) for img in images_k]
    stereo_views = stereo_depths(ds, [ref] + views, cfg) if cfg['stereo'] else None
    n = cfg['candidates']
    rows = []
    for e, (r0, a0, rho0, th0) in enumerate(zip(rows_e, cols_e, rho_c, th_c)):
        world = arc_points(rho0, th0, phis) @ T_ref[:3, :3].T + T_ref[:3, 3]
        alive = {rule: np.ones(n, bool) for rule in RULES + (STEREO_RULES if cfg['stereo'] else ())}
        if cfg['stereo']:   # same-time association at the reference frame
            T_c, depth = stereo_views[0]
            supported = np.zeros(n, bool)
            if depth is not None:
                elim, supported, _ = stereo_test(world, T_c, depth, ds.K, ds.baseline, cfg)
                for rule in STEREO_RULES:
                    alive[rule] &= ~elim
        sig = np.zeros((n, len(views), 2), int)
        fan_all = np.zeros((n, len(views)), bool)
        he = truth.get((int(r0), int(a0)))
        true_mask = None if he is None else (np.abs(phis_deg[:, None] - he[None, :]) <= cfg['truth_margin_deg']).any(axis=1)
        # tight truth: for every real reflector only the elevation sample nearest to it (0.25 deg spacing). The
        # elevation-margin label above also marks candidates just past a structure edge, in open water, as true, so a
        # rule that correctly clears them looks like it lost the truth; a metric distance cannot separate those either,
        # because the cell-centre arc already sits up to ~3 cm from the hits.
        true_near = None
        if he is not None:
            true_near = np.zeros(n, bool)
            true_near[np.argmin(np.abs(phis_deg[:, None] - he[None, :]), axis=0)] = True
        for v, (T_k, img, first) in enumerate(zip(T_views, images_k, firsts_k)):
            rho, th, ph = project(world, T_k)
            fan = in_fan(rho, th, ph, g)
            r, a = cells(rho, th, g, shape)
            sig[:, v] = np.column_stack([r, a])
            fan_all[:, v] = fan
            rc, ac = np.clip(r, 0, shape[0] - 1), np.clip(a, 0, shape[1] - 1)
            seen = echo_present(img, rc, ac, tol)
            alive['aggressive'] &= ~fan | seen
            elim, _ = in_front_of_first_echo(first, rc, ac, tol)
            alive['conservative'] &= ~(fan & elim)
            if cfg['stereo']:
                alive['conservative+stereo'] &= ~(fan & elim)
                T_c, depth = stereo_views[v + 1]
                if depth is not None:
                    s_elim, _, _ = stereo_test(world, T_c, depth, ds.K, ds.baseline, cfg)
                    alive['stereo'] &= ~s_elim
                    alive['conservative+stereo'] &= ~s_elim
            row = {'echo': e, 'ref_row': int(r0), 'ref_col': int(a0), 'rho_m': float(rho0), 'theta_deg': float(np.degrees(th0)),
                   'view': views[v], 't_s': float(sonar_t[views[v]]), 'n_views': v + 1, **motions[v],
                   'testable_fraction': float(fan.mean()),
                   'true_elev_min_deg': None if he is None else float(he.min()), 'true_elev_max_deg': None if he is None else float(he.max()),
                   'true_elev_median_deg': None if he is None else float(np.median(he)),
                   'true_reflectors': None if he is None else runs_count(true_mask)}
            if cfg['stereo']:
                row['stereo_ref_supported_fraction'] = float(supported.mean())
                row['stereo_ref_supported_has_truth'] = None if true_mask is None else bool((supported & true_mask).any())
            for rule in alive:
                al = alive[rule]
                pred = float(predicted_survivors(sig[:, :v + 1], fan_all[:, :v + 1], rule, tol).mean()) if rule in RULES else np.nan
                surv = phis_deg[al]
                row[f'{rule}_fraction_actual'] = float(al.mean())
                row[f'{rule}_fraction_predicted'] = pred/n
                row[f'{rule}_span_deg'] = float(surv.max() - surv.min()) if len(surv) else 0.0
                row[f'{rule}_clusters'] = runs_count(al)
                row[f'{rule}_truth_retained'] = None if true_mask is None else bool(al[true_mask].all())
                row[f'{rule}_truth_nearest_retained'] = None if true_near is None else bool(al[true_near].all())
                row[f'{rule}_precision'] = None if true_mask is None else (float(true_mask[al].mean()) if al.any() else None)
            rows.append(row)
    return rows, motions, {'reference_frame': ref, 'reference_t': float(sonar_t[ref]), 'echoes': int(len(rows_e)),
                           'echoes_with_truth': len(truth), 'views': len(views), 'candidates': n}


def stereo_depths(ds, sonar_indices, cfg):
    """(T_world_cam0, depth or None) per sonar frame: the nearest left/right pair within stereo_max_dt_s, B0 depth."""
    from uwbench.b.depth import depth_from_pair
    out = []
    for k in sonar_indices:
        t = float(ds.sonar_t[k])
        j = int(np.argmin(np.abs(ds.left_t - t)))
        if abs(ds.left_t[j] - t) > cfg['stereo_max_dt_s']:
            out.append((None, None))
            continue
        depth, _, _, _ = depth_from_pair(ds.image('left', j), ds.image('right', j), ds.K[0], ds.baseline)
        out.append((ds.gt_pose(float(ds.left_t[j])) @ ds.T_imu_cam0, depth))
    return out


def summarise(rows, motions):
    if not rows:
        return {}
    last_view = max(r['n_views'] for r in rows)
    final = [r for r in rows if r['n_views'] == last_view]
    m = np.array([[abs(x['dx_m']), abs(x['dy_m']), abs(x['dz_m']), abs(x['roll_deg']), abs(x['pitch_deg']), abs(x['yaw_deg'])] for x in motions])
    out = {'final_view': last_view, 'echoes': len(final),
           'motion_max_abs': dict(zip(['dx_m', 'dy_m', 'dz_m', 'roll_deg', 'pitch_deg', 'yaw_deg'], m.max(axis=0).tolist()))}
    for rule in [r for r in RULES + STEREO_RULES if f'{r}_fraction_actual' in rows[0]]:
        fa = np.array([r[f'{rule}_fraction_actual'] for r in rows])
        fp = np.array([r[f'{rule}_fraction_predicted'] for r in rows])
        wt = [r for r in final if r[f'{rule}_truth_retained'] is not None]
        prec = [r[f'{rule}_precision'] for r in wt if r[f'{rule}_precision'] is not None]
        out[rule] = {'fraction_actual_median': float(np.median([r[f'{rule}_fraction_actual'] for r in final])),
                     'fraction_actual_mean': float(np.mean([r[f'{rule}_fraction_actual'] for r in final])),
                     'span_deg_median': float(np.median([r[f'{rule}_span_deg'] for r in final])),
                     'clusters_median': float(np.median([r[f'{rule}_clusters'] for r in final])),
                     'truth_retained_fraction': float(np.mean([r[f'{rule}_truth_retained'] for r in wt])) if wt else None,
                     'truth_nearest_retained_fraction': (lambda v: float(np.mean(v)) if v else None)(
                         [r[f'{rule}_truth_nearest_retained'] for r in final if r[f'{rule}_truth_nearest_retained'] is not None]),
                     'precision_median': float(np.median(prec)) if prec else None}
        if rule in RULES:   # the pose-only prediction exists for the motion rules only
            out[rule].update({
                'fraction_predicted_median': float(np.median([r[f'{rule}_fraction_predicted'] for r in final])),
                'pred_actual_correlation': float(np.corrcoef(fp, fa)[0, 1]) if fa.std() > 0 and fp.std() > 0 else None,
                'pred_actual_median_abs_diff': float(np.median(np.abs(fp - fa)))})
    if 'stereo_ref_supported_fraction' in rows[0]:
        st = [r for r in final if r['stereo_ref_supported_has_truth'] is not None]
        out['stereo_ref_support'] = {
            'supported_fraction_median': float(np.median([r['stereo_ref_supported_fraction'] for r in final])),
            'echoes_with_any_support': float(np.mean([r['stereo_ref_supported_fraction'] > 0 for r in final])),
            'support_contains_truth_fraction': float(np.mean([r['stereo_ref_supported_has_truth'] for r in st])) if st else None}
    return out


def main(argv):
    parser = argparse.ArgumentParser(prog='uwbench motion-b', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument('--manifest', type=Path)
    src.add_argument('--raw', type=Path, help='dataset directory with raw/ (and sonar_raycast/) but no manifest')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error('Refusing to overwrite ' + str(args.out))
    import yaml
    cfg = {**DEFAULTS, **((yaml.safe_load(args.config.read_text()) or {}) if args.config else {})}
    if args.manifest:
        from uwbench.a.backend.simdata import SimDataset
        from uwbench.b.cli import sonar_images
        ds = SimDataset(args.manifest)
        image, source = sonar_images(ds)
    else:
        ds = RawDataset(args.raw)
        image, source = ds.image, str(ds._src)
    rows, motions, info = analyse(ds, image, cfg)
    args.out.mkdir(parents=True)
    with (args.out/'motion_info.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ['echo'])
        w.writeheader()
        w.writerows(rows)
    summary = {'source': str(args.manifest or args.raw), 'route': ds.route['name'], 'sonar_images': source, 'config': cfg, **info,
               'vertical_aperture_deg': ds.sonar_geometry['Elevation'],
               'aperture_note': 'simulation setting; not the platform value', 'poses': 'ground truth (stage-0 analogue)',
               'motions': motions, 'summary': summarise(rows, motions)}
    (args.out/'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({'route': ds.route['name'], **info, **summary['summary']}, indent=2))
    return 0
