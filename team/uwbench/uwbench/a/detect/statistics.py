"""Detector statistics of one run (all causal): NIS, features, covariance, information, and the windowed and
cumulative cross-sensor conflicts that DET-DIR and DET-SCALAR are built from."""
import numpy as np
from scipy.spatial.transform import Rotation

from uwbench.a.detect.config import ATMOSPHERE_PA, CONFIG, G, RHO, default_tolerances
from uwbench.a.detect.inputs import load_inputs
from uwbench.a.detect.signals import drift_budget_deg, gyro_heading, rereference, windowed
from uwbench.a.detect.sonar import (load_sim_sonar, sonar_conflict_map, sonar_conflict_multi,
                                    sonar_conflict_signed)
from uwbench.metrics.axes import body_axis_sigma



def statistics(t, est, diag, imu, odom, pressure, cfg=CONFIG, gyro_noise=None, true_bias=None, sonar=None,
               window=None, tol=None):
    """Detector statistics at estimate stamps t (all causal). Larger means more alarming for every column.

    gyro_noise: (noise density, random walk) for the frozen-bias drift budget; true_bias: simulator (t, bias xyz), oracle;
    sonar: (lines from sonar_lines, 4x4 sonar pose in the IMU frame[, first-return scans]); window: boolean mask of the
    label alignment window at t (cumulative conflicts are referenced to it when cfg['conflict_reference'] is 'window');
    tol: per-axis tolerances [x, y, z, yaw] used by DET-COV and DET-SCALAR (eval-traj's by default).
    """
    est_t, R, p, cov = est
    tol = default_tolerances() if tol is None else np.asarray(tol, dtype=float)
    ref = window if cfg.get('conflict_reference', 'window') == 'window' else None
    k = np.searchsorted(est_t, t)
    R_t, p_t, cov_t = R[k], p[k], cov[k]
    rows, gains = diag
    ut = rows[:, 0]
    order = np.argsort(ut, kind='stable')
    rows, gains, ut = rows[order], gains[order], ut[order]
    stats = {}
    nis = windowed(ut, rows[:, [4, 5]], t, cfg['nis_window_s'], lambda v: v[:, 0].sum()/v[:, 1].sum() if v[:, 1].sum() else np.nan)
    stats['DET-NIS'] = np.nan_to_num(nis, nan=0.0)
    # accepted features per update stamp (MSCKF + SLAM chunks at the same stamp), minimum over the window
    stamps, inverse = np.unique(ut, return_inverse=True)
    accepted = np.bincount(inverse, weights=rows[:, 2] - rows[:, 3])
    feat = windowed(stamps, accepted, t, cfg['feat_window_s'], np.min)
    stats['DET-FEAT'] = -np.nan_to_num(feat, nan=0.0)
    stats['DET-COV'] = body_axis_sigma(R_t, cov_t)/tol
    # information gain per body axis: position along estimated body axes, heading = world z in the local error frame
    ku = np.clip(np.searchsorted(est_t, ut, side='right') - 1, 0, None)
    Ru = R[ku]
    lam_p = np.einsum('nik,nij,njk->nk', Ru, gains[:, 3:, 3:], Ru)
    u = np.swapaxes(Ru, 1, 2)[:, :, 2]  # world z expressed in the body frame
    lam_yaw = np.einsum('ni,nij,nj->n', u, gains[:, :3, :3], u)
    lam = np.column_stack([lam_p, lam_yaw])
    stats['lambda'] = np.column_stack([windowed(ut, lam[:, i], t, cfg['info_window_s'], np.sum) for i in range(4)])
    # conflicts over a trailing window: gyro-only heading change vs estimate, depth gauge vs estimated world z
    W = cfg['conflict_window_s']
    bias_t, bias = odom[:, 0], None
    ki = np.clip(np.searchsorted(imu[:, 0], odom[:, 0]), 0, len(imu) - 1)
    bias = imu[ki, 1:] - odom[:, 1:]
    Q = gyro_heading(imu, bias_t, bias)
    t0 = np.maximum(t - W, est_t[0])
    k0 = np.searchsorted(est_t, t0)
    qi0 = np.clip(np.searchsorted(imu[:, 0], est_t[k0], side='right') - 1, 0, None)
    qi1 = np.clip(np.searchsorted(imu[:, 0], t, side='right') - 1, 0, None)
    dR_gyro = np.swapaxes(Q[qi0], 1, 2) @ Q[qi1]
    dR_est = np.swapaxes(R[k0], 1, 2) @ R_t
    e_body = Rotation.from_matrix(np.swapaxes(dR_gyro, 1, 2) @ dR_est).as_rotvec()
    yaw_conflict = np.degrees(np.abs(np.einsum('nij,nj->ni', R_t, e_body)[:, 2]))
    z_depth = -(pressure[:, 1] - ATMOSPHERE_PA)/(RHO*G)
    zi0 = np.clip(np.searchsorted(pressure[:, 0], est_t[k0], side='right') - 1, 0, None)
    zi1 = np.clip(np.searchsorted(pressure[:, 0], t, side='right') - 1, 0, None)
    z_conflict = np.abs((p_t[:, 2] - p[k0, 2]) - (z_depth[zi1] - z_depth[zi0]))
    stats['conflict'] = np.column_stack([np.zeros(len(t)), np.zeros(len(t)), z_conflict, yaw_conflict])
    # cumulative versions since the first estimate: heading is unobservable in VIO, so drift that vision injected
    # stays in the estimate; the windowed form measures the injection rate, this one the accumulated amount.
    # Signed first, then referenced to the label alignment window (rereference), then folded to a magnitude.
    q_first = np.clip(np.searchsorted(imu[:, 0], est_t[0], side='right') - 1, 0, None)
    dR_est0 = np.swapaxes(R[0], -1, -2) @ R_t

    def cumulative_yaw(Q_alt):  # signed, degrees, about world z
        dR = np.swapaxes(Q_alt[q_first], -1, -2) @ Q_alt[qi1]
        e = Rotation.from_matrix(np.swapaxes(dR, 1, 2) @ dR_est0).as_rotvec()
        return np.degrees(np.einsum('nij,nj->ni', R_t, e)[:, 2])
    yaw_cum = np.abs(rereference(cumulative_yaw(Q), ref))
    z_first = np.clip(np.searchsorted(pressure[:, 0], est_t[0], side='right') - 1, 0, None)
    z_cum = np.abs(rereference((p_t[:, 2] - p[0, 2]) - (z_depth[zi1] - z_depth[z_first]), ref))
    stats['conflict_cum'] = np.column_stack([np.zeros(len(t)), np.zeros(len(t)), z_cum, yaw_cum])
    # the online bias follows the estimate, so wrong vision information partly moves into it; freeze it at t_0
    yaw_frozen = np.abs(rereference(cumulative_yaw(gyro_heading(imu, bias_t[:1], bias[:1])), ref))
    stats['conflict_cum_frozen'] = np.column_stack([np.zeros(len(t)), np.zeros(len(t)), z_cum, yaw_frozen])
    budget = np.zeros(len(t)) if gyro_noise is None else drift_budget_deg(t - est_t[0], *gyro_noise, cfg['budget_n_sigma'])
    stats['budget_yaw_deg'] = budget
    stats['conflict_cum_frozen_excess'] = np.column_stack([np.zeros(len(t)), np.zeros(len(t)), z_cum, yaw_frozen - budget])
    if true_bias is not None:
        yaw_true = np.abs(rereference(cumulative_yaw(gyro_heading(imu, true_bias[:, 0], true_bias[:, 1:])), ref))
        stats['conflict_cum_true'] = np.column_stack([np.zeros(len(t)), np.zeros(len(t)), z_cum, yaw_true])
    sonar_x = np.full(len(t), np.nan)
    if sonar is not None:
        c_signed, proj = sonar_conflict_signed(t, est, sonar[0], sonar[1], cfg['sonar_gate_deg'], cfg['sonar_reference_s'],
                                               cfg['sonar_hold_s'])
        xy = np.abs(rereference(c_signed, ref))[:, None]*proj
        stats['conflict_cum_sonar'] = np.column_stack([xy, z_cum, yaw_cum])
        sonar_x = xy[:, 0]
        # evaluated variants (negative results in the pilot): kept on the t_0 reference as they were scored
        multi, n_planes = sonar_conflict_multi(t, est, sonar[0], sonar[1], cfg['sonar_match_deg'], cfg['sonar_jump_m'],
                                                cfg['sonar_hold_s'])
        stats['conflict_cum_sonar_multi'] = np.column_stack([multi, z_cum, yaw_cum])
        stats['sonar_planes'] = n_planes
        if len(sonar) > 2:
            m, n_map = sonar_conflict_map(t, est, sonar[2], sonar[1], cfg)
            stats['conflict_cum_sonar_map'] = np.column_stack([m[:, :2], z_cum, yaw_cum])
            stats['conflict_cum_sonar_map_yaw'] = np.column_stack([m[:, :2], z_cum, m[:, 2]])
            stats['sonar_map_points'] = n_map
    # DET-DIR, the pilot's main detector (pilot rounds 1-15): one source per axis, the best each axis has so far.
    # x: single-plane sonar conflict (vertical-structure gate, NaN without a current measurement, NaN throughout when
    # the run has no sonar); y: no source yet (NaN, scored as missed in r_miss and left out of r_miss_scorable); z: depth
    # gauge; heading: gyro with the bias frozen at t_0, minus its drift budget (method spec v2.2 M3.1; depends on the
    # platform gyro, see the Allan-variance check). dir_yaw_source 'online' uses the estimator's online bias instead
    # (absorbs part of the conflict, but does not depend on the gyro bias staying put).
    yaw_dir = {'frozen_budget': yaw_frozen - budget, 'online': yaw_cum}[cfg.get('dir_yaw_source', 'frozen_budget')]
    stats['conflict_dir'] = np.column_stack([sonar_x, np.full(len(t), np.nan), z_cum, yaw_dir])
    # DET-SCALAR: the same sources folded into one number with one threshold, so DET-DIR against DET-SCALAR isolates
    # what splitting by axis contributes; tolerance-normalised so metres and degrees can be combined
    norm = stats['conflict_dir']/tol
    finite = np.isfinite(norm)
    none = ~finite.any(axis=1)
    stats['DET-SCALAR'] = np.where(none, np.nan, np.max(np.where(finite, norm, -np.inf), axis=1))
    # l2 over the positive parts only: a conflict below its drift budget (negative excess) is no evidence
    stats['DET-SCALAR-l2'] = np.where(none, np.nan, np.sqrt(np.sum(np.where(finite, np.maximum(norm, 0.0), 0.0)**2, axis=1)))
    return stats


def compute_run(inputs, cfg=CONFIG, sonar=None):
    """Detector statistics for one run under cfg; sonar from sonar_products (or None)."""
    gyro_noise, source = inputs['gyro_noise'], inputs['gyro_noise_source']
    if cfg.get('gyro_noise'):
        g = cfg['gyro_noise']
        gyro_noise, source = (float(g['noise_density']), float(g['random_walk'])), g.get('source', 'config override')
    data = {k: inputs[k] for k in ('name', 't', 'scored', 'window', 'errors', 'tol')}
    data['stats'] = statistics(inputs['t'], inputs['est'], inputs['diag'], inputs['imu'], inputs['odom'], inputs['pressure'],
                               cfg=cfg, gyro_noise=gyro_noise, true_bias=inputs['true_bias'], sonar=sonar,
                               window=inputs['window'], tol=inputs['tol'])
    data['gyro_noise'], data['gyro_noise_source'] = gyro_noise, source
    rows = inputs['diag'][0]
    data['obs_t'] = np.unique(rows[rows[:, 2] - rows[:, 3] > 0, 0])  # updates that accepted features: C0 observation age
    est_t, R, p, _ = inputs['est']
    k = np.clip(np.searchsorted(est_t, inputs['t']), 0, len(est_t) - 1)
    data['pose_finite'] = np.isfinite(R[k]).all(axis=(1, 2)) & np.isfinite(p[k]).all(axis=1)
    return data


def load_run(run_dir, cfg=CONFIG):
    inputs = load_inputs(run_dir)
    return compute_run(inputs, cfg, load_sim_sonar(inputs['dataset'], cfg))
