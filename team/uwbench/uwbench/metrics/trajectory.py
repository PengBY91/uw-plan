"""Trajectory scoring primitives shared by the W1 OpenVINS evaluator and uwbench eval-traj."""
import numpy as np
from evo.core import lie_algebra, metrics


def summarize(values):
    a = np.asarray(values, dtype=float)
    if a.size == 0 or not np.isfinite(a).all():
        raise ValueError('Cannot score empty or non-finite errors')
    return {'count': int(a.size), 'rmse': float(np.sqrt(np.mean(a*a))),
            'mean': float(a.mean()), 'median': float(np.median(a)),
            'p95': float(np.percentile(a, 95)), 'max': float(a.max())}


def time_pairs(timestamps, delta, tolerance=0.025):
    times = np.asarray(timestamps)
    if len(times) < 2 or not np.isfinite(times).all() or not np.all(np.diff(times) > 0):
        raise ValueError('Time pairing requires a nonempty increasing trajectory')
    if delta <= 0 or tolerance < 0:
        raise ValueError('Invalid pair interval')
    pairs = []
    for i, t in enumerate(times):
        target = t + delta
        k = int(np.searchsorted(times, target))
        choices = [j for j in (k-1, k) if i < j < len(times)]
        if choices:
            j = min(choices, key=lambda j: abs(times[j]-target))
            if abs(times[j]-target) <= tolerance:
                pairs.append((i, j))
    return pairs


def score_rpe(ref, est, delta):
    pairs = time_pairs(est.timestamps, delta)
    if not pairs:
        raise ValueError('No valid time-based RPE pairs')
    values, angles, dts, ref_dts = [], [], [], []
    for i, j in pairs:
        error = metrics.RPE.rpe_base(ref.poses_se3[i], ref.poses_se3[j],
                                     est.poses_se3[i], est.poses_se3[j])
        values.append(np.linalg.norm(error[:3, 3]))
        angles.append(lie_algebra.so3_log_angle(error[:3, :3], degrees=True))
        dts.append(est.timestamps[j] - est.timestamps[i])
        ref_dts.append(ref.timestamps[j] - ref.timestamps[i])
    possible = int(np.sum(est.timestamps + delta <= est.timestamps[-1] + 0.025))
    return {'translation_m': summarize(values), 'rotation_deg': summarize(angles),
            'target_interval_s': delta, 'pair_tolerance_s': 0.025,
            'actual_est_interval_s': [float(min(dts)), float(max(dts))],
            'actual_ref_interval_s': [float(min(ref_dts)), float(max(ref_dts))],
            'eligible_start_count': possible, 'selected_pair_count': len(pairs),
            'pairs': pairs}


def associate(ref_times, est_times, tolerance):
    """Nearest reference index per estimate and a mask of matches within tolerance."""
    ref_times, est_times = np.asarray(ref_times, dtype=float), np.asarray(est_times, dtype=float)
    k = np.clip(np.searchsorted(ref_times, est_times), 1, len(ref_times)-1)
    nearest = np.where(np.abs(ref_times[k-1]-est_times) <= np.abs(ref_times[k]-est_times), k-1, k)
    return nearest, np.abs(ref_times[nearest]-est_times) <= tolerance


def time_coverage(est_times, valid_intervals, max_gap):
    """Seconds of reference-valid time covered by estimates; lost spans are never bridged."""
    est = np.asarray(est_times, dtype=float)
    covered, lost, valid_total = 0.0, [], 0.0
    for start, end in valid_intervals:
        valid_total += end - start
        inside = est[(est >= start) & (est <= end)]
        if inside.size == 0:
            lost.append({'start': start, 'end': end, 'duration_s': end-start, 'kind': 'no_estimate'})
            continue
        if inside[0] - start > max_gap:
            lost.append({'start': start, 'end': float(inside[0]), 'duration_s': float(inside[0]-start), 'kind': 'before_first_estimate'})
        for a, b in zip(inside[:-1], inside[1:]):
            if b - a <= max_gap:
                covered += b - a
            else:
                lost.append({'start': float(a), 'end': float(b), 'duration_s': float(b-a), 'kind': 'gap'})
        if end - inside[-1] > max_gap:
            lost.append({'start': float(inside[-1]), 'end': end, 'duration_s': float(end-inside[-1]), 'kind': 'after_last_estimate'})
    return {'valid_reference_s': valid_total, 'covered_s': covered,
            'coverage_fraction': covered/valid_total if valid_total > 0 else None,
            'lost_intervals': lost, 'failure_count': sum(i['kind'] == 'gap' for i in lost), 'max_gap_s': max_gap}
