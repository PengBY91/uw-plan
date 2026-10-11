"""Check visual tracks of a sim run against ground truth (diagnostic, not part of any estimator).

Per left frame k: corners -> left/right KLT -> stereo triangulation with the derived calibration -> GT relative
pose to frame k+1 -> predicted pixels, compared with left k -> k+1 KLT. Large residuals mean stereo or temporal
mismatches (e.g. repeated structure), or a calibration/timing error if they are large everywhere.
Run with a Python that has OpenCV (ROS Jazzy system python3 has cv2 4.6).
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml


def frames(raw, kind):
    for path in sorted(raw.glob(f'{kind}_[0-9]*.npy')):
        yield from np.load(path)


def load_calibration(path):
    text = path.read_text().split('\n', 1)[1]  # drop the %YAML directive
    cams = yaml.safe_load(text)
    T = [np.array(cams[c]['T_imu_cam'], float) for c in ('cam0', 'cam1')]
    fx, fy, cx, cy = cams['cam0']['intrinsics']
    return T, np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True, help='data/sim/<run_id>')
    parser.add_argument('--bad-px', type=float, default=3.0)
    parser.add_argument('--step', type=int, default=2, help='evaluate every n-th frame')
    args = parser.parse_args()
    raw = args.run/'raw'
    (T_ic0, T_ic1), K = load_calibration(args.run/'openvins'/'kalibr_imucam_chain.yaml')
    gt = np.load(raw/'gt_imu.npy')
    stamps = np.load(raw/'left_t.npy')
    T_wi = {round(t, 6): row[1:].reshape(4, 4) for t, row in zip(gt[:, 0], gt)}
    T_c0c1 = np.linalg.inv(T_ic0) @ T_ic1
    P0 = K @ np.eye(3, 4)
    P1 = K @ np.linalg.inv(T_c0c1)[:3]
    lk = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    rows = []
    left, right = frames(raw, 'left'), frames(raw, 'right')
    prev = None
    for k, (t, L, R) in enumerate(zip(stamps, left, right)):
        if prev is not None and (k - 1) % args.step == 0:
            t0, L0, R0 = prev
            corners = cv2.goodFeaturesToTrack(L0, 400, 0.01, 10)
            row = {'t': float(t0), 'corners': 0 if corners is None else len(corners)}
            if corners is not None:
                pts = corners.reshape(-1, 2)
                pr, sr, _ = cv2.calcOpticalFlowPyrLK(L0, R0, pts, None, **lk)
                pt, st, _ = cv2.calcOpticalFlowPyrLK(L0, L, pts, None, **lk)
                ok = (sr.ravel() == 1) & (st.ravel() == 1) & (np.abs(pr[:, 1] - pts[:, 1]) < 2.0) & (pts[:, 0] - pr[:, 0] > 0.5)
                row['stereo_and_temporal_tracks'] = int(ok.sum())
                if ok.sum() >= 5:
                    X = cv2.triangulatePoints(P0, P1, pts[ok].T, pr[ok].T)
                    X = (X[:3]/X[3]).T
                    T_c0_k1 = np.linalg.inv(T_wi[round(float(t), 6)] @ T_ic0) @ (T_wi[round(float(t0), 6)] @ T_ic0)
                    Xk1 = X @ T_c0_k1[:3, :3].T + T_c0_k1[:3, 3]
                    front = Xk1[:, 2] > 0.1
                    proj = (Xk1 @ K.T)[:, :2]/np.maximum(Xk1[:, 2:3], 1e-9)
                    err = np.where(front, np.linalg.norm(proj - pt[ok], axis=1), np.inf)
                    row.update(median_err_px=float(np.median(err)), bad_fraction=float(np.mean(err > args.bad_px)),
                               median_depth_m=float(np.median(X[:, 2])), median_flow_px=float(np.median(np.linalg.norm(pt[ok]-pts[ok], axis=1))))
            rows.append(row)
        prev = (t, L, R)
    out = args.run/'track_check.json'
    out.write_text(json.dumps({'bad_px': args.bad_px, 'step': args.step, 'rows': rows}, indent=1))
    print(out, len(rows))


if __name__ == '__main__':
    main()
