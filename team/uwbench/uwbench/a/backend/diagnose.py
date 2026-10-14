"""uwbench diagnose-a (method spec M1, prompt P14): direction information of the A1 smoother, read-only.

For every keyframe pair (k-1, k) the new stereo and sonar observations of landmarks seen at both keyframes are
linearised at the smoother's estimate right after keyframe k's update (causal). Rows are already whitened (robust
kernels reweighted as the solver does). X(k-1) is held fixed, so the result is information on the relative increment
xi of X(k) in its own body frame and no gauge freedom can pose as information. Each landmark is eliminated jointly
over all of its rows (F = Q2^T J_x, H = F^T F), never marginalised factor by factor. Order follows the spec:
translation first (m), rotation second (rad); GTSAM's [rot, trans] tangent is permuted explicitly.

Normalisation: x = S z, S = diag(L, L, L, 1, 1, 1), Hbar = S H S, with L a fixed characteristic length (frozen on the
dev set; 2 m default = sonar standoff of the sim routes). Per body axis k in {x, y, z, yaw},
lambda_k = a_k^T Hbar a_k, yaw being world z expressed in the body frame. The raw (L = 1) spectrum is reported too.

The IMU preintegration factor, the priors and any depth gauge are not part of these blocks (spec M1); A1 has no depth
factor, so z information here is camera/sonar only. With ground truth, E-A1 asks whether the weak directions of the
windowed information are where the relative-pose error actually grew; truth is used only in that scoring step.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

AXES = ('x', 'y', 'z', 'yaw')
MODALITIES = ('stereo', 'sonar')
PERM = [3, 4, 5, 0, 1, 2]   # GTSAM Pose3 tangent [rot, trans] -> spec order [trans, rot]


def eliminate(Ax, An, rcond=1e-9):
    """Information on x after eliminating landmark n from whitened rows [Ax An]: Ax^T (I - An An^+) Ax.

    A rank-deficient An (e.g. a sonar landmark whose elevation two range-bearing rows cannot fix) is projected out
    only over its determined subspace, so no information is invented along the undetermined one."""
    if An.size:
        U, s, _ = np.linalg.svd(An, full_matrices=False)
        r = int(np.sum(s > rcond*s.max())) if s.size and s.max() > 0 else 0
        U = U[:, :r]
        F = Ax - U @ (U.T @ Ax)
    else:
        F = Ax
    return F.T @ F


def scaling(length_m):
    return np.diag([length_m]*3 + [1.0]*3)


def axis_information(Hbar, R_world_body):
    """lambda_k on body x, y, z (translation) and yaw (rotation about world z) from normalised 6x6 information."""
    a = R_world_body.T @ np.array([0.0, 0.0, 1.0])
    return np.array([Hbar[0, 0], Hbar[1, 1], Hbar[2, 2], a @ Hbar[3:, 3:] @ a])


def spectrum(H):
    """Ascending eigenvalues and eigenvectors of a symmetric 6x6 (symmetrised first)."""
    w, V = np.linalg.eigh((H + H.T)/2)
    return w, V


def rotate_information(H, C):
    """Information on a perturbation expressed in frame j, re-expressed for a perturbation in frame k, C = R_k^T R_j
    (rotation part only; the lever-arm coupling of the adjoint is ignored over short windows)."""
    B = np.zeros((6, 6))
    B[:3, :3] = B[3:, 3:] = C.T
    return B.T @ H @ B


class Diagnoser:
    """after_keyframe hook for uwbench.a.backend.cli.main; reads the smoother, never updates it."""
    def __init__(self, length_m=2.0):
        self.L = float(length_m)
        self.S = scaling(self.L)
        self.rows, self.H, self.times, self.R = [], [], [], []
        self._sonar_seen, self._sonar_by_kf = 0, {}

    def __call__(self, sm, k, t, tracks):
        from gtsam.symbol_shorthand import X
        for kk, lid, f, *_ in sm.sonar_factors[self._sonar_seen:]:
            self._sonar_by_kf.setdefault(kk, []).append((lid, f))
        self._sonar_seen = len(sm.sonar_factors)
        if k == 0:
            return
        values = sm._estimate
        R_k = values.atPose3(X(k)).rotation().matrix()
        graph = sm.isam.getFactorsUnsafe()
        groups = {m: {} for m in MODALITIES}
        for tid in {int(r[0]) for r in tracks}:
            for i in sm.visual_factor_idx.get(tid, []):
                f = graph.at(i)
                if f is not None and any(key in (X(k), X(k - 1)) for key in f.keys()):
                    groups['stereo'].setdefault(tid, []).append(f)
        for kk in (k - 1, k):
            for lid, f in self._sonar_by_kf.get(kk, []):
                groups['sonar'].setdefault(lid, []).append(f)
        total = np.zeros((6, 6))
        for modality in MODALITIES:
            H, n_lm, n_f = np.zeros((6, 6)), 0, 0
            for fs in groups[modality].values():
                keys = {key for f in fs for key in f.keys()}
                if X(k) not in keys or X(k - 1) not in keys:
                    continue   # a landmark seen at only one of the two keyframes says nothing about the increment
                Ax, An = [], []
                for f in fs:
                    A, _ = f.linearize(values).jacobian()
                    cols, ax, an, c = list(f.keys()), np.zeros((A.shape[0], 6)), np.zeros((A.shape[0], 3)), 0
                    for key in cols:
                        width = 6 if key in (X(k), X(k - 1)) else 3
                        if key == X(k):
                            ax = A[:, c:c + width]
                        elif key != X(k - 1):
                            an = A[:, c:c + width]
                        c += width
                    Ax.append(ax)
                    An.append(an)
                H += eliminate(np.vstack(Ax), np.vstack(An))
                n_lm, n_f = n_lm + 1, n_f + len(fs)
            H = H[np.ix_(PERM, PERM)]
            total += H
            self._row(k, t, modality, H, R_k, n_lm, n_f)
        self._row(k, t, 'total', total, R_k, None, None)
        self.H.append(total)
        self.times.append((k, t))
        self.R.append(R_k)

    def _row(self, k, t, modality, H, R_k, n_lm, n_f):
        Hbar = self.S @ H @ self.S
        w, V = spectrum(Hbar)
        w_raw, _ = spectrum(H)
        top = max(w.max(), 0.0)
        self.rows.append({'keyframe': k, 't_s': t, 'modality': modality, 'landmarks': n_lm, 'factors': n_f,
                          **{f'lambda_{a}': v for a, v in zip(AXES, axis_information(Hbar, R_k))},
                          **{f'eig{i}': v for i, v in enumerate(w)}, **{f'eig_raw{i}': v for i, v in enumerate(w_raw)},
                          **{f'weak{i}': v for i, v in enumerate(V[:, 0])},
                          'rank': int(np.sum(w > 1e-9*top)) if top > 0 else 0})


def relative_error(T_est_a, T_est_b, T_ref_a, T_ref_b):
    """Log((T_ref_a^-1 T_ref_b)^-1 (T_est_a^-1 T_est_b)) in body frame b, spec order [trans, rot]."""
    import gtsam
    d_ref = np.linalg.inv(T_ref_a) @ T_ref_b
    d_est = np.linalg.inv(T_est_a) @ T_est_b
    E = np.linalg.inv(d_ref) @ d_est
    xi = gtsam.Pose3.Logmap(gtsam.Pose3(gtsam.Rot3(E[:3, :3]), gtsam.Point3(*E[:3, 3])))
    return np.asarray(xi)[PERM]


def correspondence(H_list, R_list, poses_est, poses_ref, length_m, window):
    """E-A1: per non-overlapping window of `window` pairs, does the relative-pose error lie in the weak directions?

    H_list[i], R_list[i]: pair (k-1, k) information (spec order) and R_world_body of keyframe k = i + 1;
    poses_est/poses_ref: 4x4 per keyframe (index k). Returns per-window rows for the normalised and raw spectra:
    frac_weak2 = share of the unit error in the two weakest eigen-directions (chance 2/6), cos_weak / cos_strong =
    |cosine| between the error and the weakest / strongest eigenvector.

    Two controls, because a weak direction that is always translational matches any mostly-translational error:
    trans_cos_weak compares the translation error with the weakest direction of the translation block alone (uniform
    chance 0.5 in 3-D), and every *_shift column re-scores the window against the information of the window half a
    run away (same statistics, wrong time), so a real correspondence must beat its own shifted control."""
    S = scaling(length_m)
    windows = []
    for end in range(window, len(H_list) + 1, window):
        R_k = R_list[end - 1]   # keyframe `end` is the window's last keyframe
        H = sum(rotate_information(H_list[i], R_k.T @ R_list[i]) for i in range(end - window, end))
        e = relative_error(poses_est[end - window], poses_est[end], poses_ref[end - window], poses_ref[end])
        windows.append((end, H, e))

    def scores(H, e, suffix):
        r = {}
        for name, M, z in (('norm', S @ H @ S, np.r_[e[:3]/length_m, e[3:]]), ('raw', H, e)):
            n = np.linalg.norm(z)
            if n == 0 or not np.all(np.isfinite(M)):
                continue
            u = z/n
            w, V = spectrum(M)
            r.update({f'{name}_frac_weak2{suffix}': float(np.sum((V[:, :2].T @ u)**2)),
                      f'{name}_cos_weak{suffix}': float(abs(V[:, 0] @ u)), f'{name}_cos_strong{suffix}': float(abs(V[:, -1] @ u))})
            if not suffix:
                r[f'{name}_weak_is_rotation'] = bool(np.sum(V[3:, 0]**2) > 0.5)
        nt = np.linalg.norm(e[:3])
        if nt > 0 and np.all(np.isfinite(H)):
            _, Vt = spectrum(H[:3, :3])
            r[f'trans_cos_weak{suffix}'] = float(abs(Vt[:, 0] @ e[:3]/nt))
        return r

    out = []
    for j, (end, H, e) in enumerate(windows):
        row = {'keyframe': end, 'error_trans_m': float(np.linalg.norm(e[:3])), 'error_rot_deg': float(np.degrees(np.linalg.norm(e[3:])))}
        row.update(scores(H, e, ''))
        if len(windows) > 1:
            row.update(scores(windows[(j + len(windows)//2) % len(windows)][1], e, '_shift'))
        out.append(row)
    return out


def summarise(rows):
    s = {'windows': len(rows), 'chance_frac_weak2': 2/6, 'chance_trans_cos_weak': 0.5}

    def mean(key):
        v = [r[key] for r in rows if key in r]
        return float(np.mean(v)) if v else None
    for name in ('norm', 'raw'):
        if mean(f'{name}_frac_weak2') is not None:
            s[name] = {m: mean(f'{name}_{m}') for m in ('frac_weak2', 'frac_weak2_shift', 'cos_weak', 'cos_weak_shift',
                                                        'cos_strong', 'weak_is_rotation')}
    s['trans'] = {'cos_weak': mean('trans_cos_weak'), 'cos_weak_shift': mean('trans_cos_weak_shift')}
    both = [r for r in rows if 'norm_weak_is_rotation' in r and 'raw_weak_is_rotation' in r]
    if both:
        s['normalisation_changes_weak_type_fraction'] = float(np.mean([r['norm_weak_is_rotation'] != r['raw_weak_is_rotation'] for r in both]))
    return s


def write_csv(path, rows):
    import csv
    with open(path, 'w', newline='') as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def main(argv):
    parser = argparse.ArgumentParser(prog='uwbench diagnose-a', description=__doc__)
    parser.add_argument('--length-m', type=float, default=2.0, help='characteristic length L (frozen; default 2 m)')
    parser.add_argument('--window', type=int, default=10, help='keyframe pairs per E-A1 window (default 10 = 2 s at stride 2)')
    args, rest = parser.parse_known_args(argv)
    from uwbench.a.backend import cli, simdata
    diag = Diagnoser(args.length_m)
    rc = cli.main(rest, after_keyframe=diag)
    out = Path(rest[rest.index('--out') + 1])
    manifest = Path(rest[rest.index('--manifest') + 1])
    information = diag.rows
    write_csv(out/'information.csv', information)
    write_csv(out/'axis_info.csv', [{'t_s': r['t_s'], 'keyframe': r['keyframe'], 'modality': r['modality'],
                                     **{f'lambda_{a}': r[f'lambda_{a}'] for a in AXES}, 'landmarks': r['landmarks'],
                                     'factors': r['factors'], 'depth_factor': 0} for r in information])
    gauge = {'anchor': 'X(k-1) held fixed per pair: information is on the relative increment of X(k) only',
             'excluded_freedoms': 'global position and yaw (VIO gauge) cannot appear; priors, IMU and depth are not in the blocks',
             'tangent_order': 'translation (m, body frame k) then rotation (rad); GTSAM [rot, trans] permuted',
             'length_m': args.length_m, 'length_note': 'frozen characteristic length; 2 m ~ sim sonar standoff (ASSUMPTION until the dev set)',
             'linearisation': 'smoother estimate right after keyframe k (causal); robust kernels reweighted as in the solver'}
    (out/'gauge.json').write_text(json.dumps(gauge, indent=2) + '\n')
    summary = {'pairs': len(diag.H), 'length_m': args.length_m, 'window_pairs': args.window,
               'run_status': json.loads((out/'summary.json').read_text())['status']}
    if diag.H:   # E-A1 scoring: the only place ground truth is read
        from scipy.spatial.transform import Rotation
        ds = simdata.SimDataset(manifest)
        traj = np.loadtxt(out/'trajectory_causal.tum', ndmin=2)   # row k = keyframe k right after its update
        poses_est, poses_ref = [], []
        for row in traj:
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = Rotation.from_quat(row[4:8]).as_matrix(), row[1:4]
            poses_est.append(T)
            poses_ref.append(ds.gt_pose(row[0] - simdata.EPOCH_S))
        n = min(len(diag.H), len(traj) - 1)
        rows = correspondence(diag.H[:n], diag.R[:n], poses_est, poses_ref, args.length_m, args.window)
        write_csv(out/'directions.csv', rows)
        summary['e_a1'] = summarise(rows)
        summary['e_a1']['note'] = 'ground truth used only here; windows rotate pair information into the last keyframe frame (no lever arm)'
    (out/'diagnose.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    return rc


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
