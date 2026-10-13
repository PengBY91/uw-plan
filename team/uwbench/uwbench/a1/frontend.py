"""Fixed visual front end for A1: KLT tracks on the left image plus left-to-right stereo matching.

The output (tracks.npz) is computed once per dataset and parameter set and shared by every A1 configuration, so
sonar on/off comparisons see identical visual measurements.
"""
import hashlib
import json

import cv2
import numpy as np

DEFAULTS = {'max_features': 250, 'min_features': 150, 'quality': 0.01, 'min_distance_px': 12,
            'lk_window_px': 21, 'lk_levels': 3, 'fb_max_px': 0.7, 'stereo_max_dy_px': 1.0,
            'min_disparity_px': 0.5, 'border_px': 8}


def params_hash(params):
    return hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:16]


def _lk(a, b, pts, guess, p):
    kw = dict(winSize=(p['lk_window_px'],)*2, maxLevel=p['lk_levels'],
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    fwd, st1, _ = cv2.calcOpticalFlowPyrLK(a, b, pts, guess.copy(), flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **kw)
    back, st2, _ = cv2.calcOpticalFlowPyrLK(b, a, fwd, pts.copy(), flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **kw)
    ok = (st1.ravel() == 1) & (st2.ravel() == 1) & (np.linalg.norm(back - pts, axis=2).ravel() < p['fb_max_px'])
    return fwd, ok


def track(ds, frame_indices, params=None):
    """Rows (keyframe index, track id, uL, uR, v) for the given left/right frame indices."""
    p = dict(DEFAULTS, **(params or {}))
    rows, prev, prev_pts, prev_ids, next_id = [], None, np.zeros((0, 1, 2), np.float32), np.zeros(0, np.int64), 0
    for k, f in enumerate(frame_indices):
        left, right = ds.image('left', f), ds.image('right', f)
        h, w = left.shape
        pts, ids = np.zeros((0, 1, 2), np.float32), np.zeros(0, np.int64)
        if prev is not None and len(prev_pts):
            nxt, ok = _lk(prev, left, prev_pts, prev_pts, p)
            inside = ((nxt[:, 0, 0] >= p['border_px']) & (nxt[:, 0, 0] < w - p['border_px']) &
                      (nxt[:, 0, 1] >= p['border_px']) & (nxt[:, 0, 1] < h - p['border_px']))
            keep = ok & inside
            pts, ids = nxt[keep], prev_ids[keep]
        if len(pts) < p['min_features']:
            mask = np.full((h, w), 255, np.uint8)
            for x, y in pts[:, 0]:
                cv2.circle(mask, (int(x), int(y)), p['min_distance_px'], 0, -1)
            mask[:p['border_px']], mask[-p['border_px']:], mask[:, :p['border_px']], mask[:, -p['border_px']:] = 0, 0, 0, 0
            new = cv2.goodFeaturesToTrack(left, p['max_features'] - len(pts), p['quality'], p['min_distance_px'], mask=mask)
            if new is not None:
                pts = np.concatenate([pts, new.astype(np.float32)])
                ids = np.concatenate([ids, np.arange(next_id, next_id + len(new))])
                next_id += len(new)
        if len(pts):
            right_pts, ok = _lk(left, right, pts, pts, p)
            disparity = pts[:, 0, 0] - right_pts[:, 0, 0]
            ok &= (np.abs(right_pts[:, 0, 1] - pts[:, 0, 1]) <= p['stereo_max_dy_px']) & (disparity >= p['min_disparity_px'])
            for i in np.nonzero(ok)[0]:
                rows.append((k, int(ids[i]), float(pts[i, 0, 0]), float(right_pts[i, 0, 0]), float(pts[i, 0, 1])))
        prev, prev_pts, prev_ids = left, pts, ids
    return np.array(rows, dtype=float).reshape(-1, 5), p
