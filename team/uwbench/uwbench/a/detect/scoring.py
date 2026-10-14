"""Leave-one-run-out operating points and miss / false-alarm / misattribution scoring of every detector."""
from collections import namedtuple

import numpy as np

from uwbench.a.detect.config import CONFIG, CUMULATIVE, SOURCE_AXES, default_tolerances
from uwbench.a.detect.signals import information_gap
from uwbench.metrics.axes import AXES



def exceed_labels(errors, tol=None):
    return np.abs(errors) > (default_tolerances() if tol is None else np.asarray(tol, dtype=float))


def threshold(values, target):
    """Smallest threshold whose alarm fraction (values > thr) on these normal samples is <= target.

    NaN samples (not scorable) are left out; with none left the threshold is infinite (never alarms).
    """
    values = np.asarray(values, dtype=float)
    values = np.sort(values[np.isfinite(values)])[::-1]
    if not len(values):
        return np.inf
    # the (k+1)-th largest value, k = floor(target n): at most k samples lie strictly above it. (The pilot used the
    # interpolated (1 - target) quantile, which with ties or discrete statistics could let a few more through.)
    return float(values[min(int(np.floor(target*len(values) + 1e-12)), len(values) - 1)])


def union_false_alarm(columns, thresholds):
    """Fraction of samples where any channel exceeds its threshold (NaN never alarms)."""
    hit = np.zeros(len(columns[0]), bool)
    for c, h in zip(columns, thresholds):
        hit |= c > h
    return float(hit.mean()) if len(hit) else 0.0


def union_thresholds(columns, target, allocation='union', fractions=None, iterations=30):
    """Per-channel thresholds for a detector that alarms when any channel exceeds its own threshold.

    columns: the training normal samples of each channel (equal length, NaN = not scorable). 'union': the largest common
    per-channel quantile level whose any-channel alarm fraction on these samples is <= target (bisection), so every
    detector spends the same nominal false-alarm budget whatever its number of channels. 'fixed': channel i at
    fractions[i] (the pilot's allocation), whose union budget depends on the detector.
    """
    cols = [np.asarray(c, dtype=float) for c in columns]
    if allocation == 'fixed':
        return [threshold(c, f) for c, f in zip(cols, fractions)]
    if allocation != 'union':
        raise ValueError(f'unknown threshold_allocation {allocation!r}')
    if len(cols) == 1:
        return [threshold(cols[0], target)]
    thr = [threshold(c, target) for c in cols]
    if union_false_alarm(cols, thr) <= target:
        return thr
    lo, hi, best = 0.0, float(target), [threshold(c, 0.0) for c in cols]
    for _ in range(iterations):
        mid = (lo + hi)/2
        cand = [threshold(c, mid) for c in cols]
        if union_false_alarm(cols, cand) <= target:
            lo, best = mid, cand
        else:
            hi = mid
    return best


def comparable(false_alarm, target, tolerance):
    """Ablation-runbook rule: an operating point counts only when its realised false-alarm fraction is within
    tolerance (relative) of the target; at target 0 only zero false alarms qualify."""
    if false_alarm is None:
        return None
    return bool(false_alarm == 0.0) if target <= 0 else bool(abs(false_alarm - target) <= tolerance*target)


Channel = namedtuple('Channel', 'train test axis fraction')  # training normals, test values, axis (None = global), pilot fraction


def score_detector(test, name, chans, target, allocation, grace, tol_rel):
    """One per-run row for a detector given as channels; the alarm on an axis is the union of its channels there."""
    thr = union_thresholds([c.train for c in chans], target, allocation, [c.fraction for c in chans])
    train_fa = union_false_alarm([c.train for c in chans], thr)
    n = len(test['t'])
    alarm, finite = np.zeros((n, 4), bool), np.zeros((n, 4), bool)
    for c, h in zip(chans, thr):
        for a in (range(4) if c.axis is None else (c.axis,)):
            alarm[:, a] |= c.test > h
            finite[:, a] |= np.isfinite(c.test)
    sourced = sorted({a for c in chans for a in (range(4) if c.axis is None else (c.axis,))})
    fa = float(alarm[test['normal']].any(axis=1).mean()) if test['normal'].any() else None
    row = {'early_warning_s': grace, 'target': target, 'detector': name, 'run': test['name'],
           'normal_samples': int(test['normal'].sum()), 'false_alarm_fraction': fa,
           'train_false_alarm_fraction': train_fa, 'comparable': comparable(fa, target, tol_rel),
           # a normal sample is unscorable when an axis that has a source lacks a current measurement
           'normal_unscorable': int((test['normal'] & ~finite[:, sourced].all(axis=1)).sum())}
    failing = test['scored'] & ~test['normal']  # some axis exceeds (or is about to): attribution is at stake
    for a, axis in enumerate(AXES):
        ex = test['exceed'][:, a]
        row[f'exceed_{axis}'] = int(ex.sum())
        row[f'unscorable_{axis}'] = int((ex & ~finite[:, a]).sum())
        row[f'missed_{axis}'] = int((ex & ~alarm[:, a]).sum())  # includes the unscorable ones
        # attribution: alarms on an axis that is within tolerance while another axis is failing. A global detector
        # alarms every axis at once, so it "catches" co-occurring exceedances on all axes; this column is what a
        # per-axis detector buys in exchange for its per-axis misses
        row[f'offaxis_base_{axis}'] = int((failing & ~ex).sum())
        row[f'offaxis_alarm_{axis}'] = int((failing & ~ex & alarm[:, a]).sum())
    return row


def label_runs(runs, grace, tol_default=None):
    """Exceedance, early-warning and normal masks per run for one lead-time setting."""
    for r in runs:
        r['exceed'] = exceed_labels(r['errors'], r.get('tol', tol_default)) & r['scored'][:, None]
        any_exceed = r['exceed'].any(axis=1)
        onsets = r['t'][any_exceed & ~np.r_[False, any_exceed[:-1]]]
        warning = np.zeros(len(r['t']), bool)
        for onset in onsets:
            warning |= (r['t'] >= onset - grace) & (r['t'] < onset)
        r['normal'] = r['scored'] & ~any_exceed & ~warning


def evaluate(runs, cfg=CONFIG):
    """Leave-one-run-out operating points; returns per-run miss/false-alarm rows (pooled() aggregates them).

    Every detector is expressed as Channels; its alarm on an axis is the union of its channels on that axis. Scoring is
    identical for all detectors: a sample is missed when it exceeds and the detector did not alarm, whatever the reason
    (including an unscorable, NaN statistic); unscorable counts are reported apart so the pilot's scorable-only rate can
    still be formed.
    """
    allocation = cfg.get('threshold_allocation', 'union')
    tol_rel = cfg.get('comparable_tolerance', 0.5)
    results = []
    for grace, target in [(g, f) for g in cfg['early_warning_s'] for f in cfg['false_alarm_targets']]:
        label_runs(runs, grace)
        for test in runs:
            train = [r for r in runs if r is not test]

            def normal(key, col=None):
                return np.concatenate([(r['stats'][key] if col is None else r['stats'][key][:, col])[r['normal']] for r in train])
            lam_min = [float(np.quantile(np.nan_to_num(normal('lambda', a)), cfg['lambda_min_quantile'])) for a in range(4)]
            for r in runs:
                r['stats']['tau'] = information_gap(r['t'], np.nan_to_num(r['stats']['lambda']), lam_min)
            per_axis = target/4  # the pilot's fixed allocation: any-axis false alarms at most the target on the training normals

            def axis_channels(key, axes, fraction):
                return [Channel(normal(key, a), test['stats'][key][:, a], a, fraction) for a in axes]
            channels = {}
            for name in ('DET-NIS', 'DET-FEAT', 'DET-SCALAR', 'DET-SCALAR-l2'):
                if all(name in r['stats'] for r in runs):
                    channels[name] = [Channel(normal(name), test['stats'][name], None, target)]
            channels['DET-COV'] = axis_channels('DET-COV', range(4), per_axis)
            channels['DIR-gap'] = axis_channels('tau', range(4), per_axis)
            channels['DIR-conflict'] = axis_channels('conflict', (2, 3), per_axis/2)  # depth for z, gyro for heading
            channels['DIR-gap+conflict'] = axis_channels('tau', range(4), per_axis/2) + channels['DIR-conflict']
            for name, key in CUMULATIVE.items():
                if not all(key in r['stats'] for r in runs):
                    continue  # the oracle needs the simulator's bias, the sonar forms sonar data, in every run
                channels[name] = axis_channels(key, SOURCE_AXES.get(key, (2, 3)), per_axis/2)
            if 'DIR-cumconflict' in channels:
                channels['DIR-gap+cumconflict'] = axis_channels('tau', range(4), per_axis/2) + channels['DIR-cumconflict']
            for name, chans in channels.items():
                results.append(score_detector(test, name, chans, target, allocation, grace, tol_rel))
    return results


def pooled(results, comparable_tolerance=0.5):
    table = {}
    for row in results:
        key = (row['early_warning_s'], row['target'], row['detector'])
        t = table.setdefault(key, {'early_warning_s': row['early_warning_s'], 'target': row['target'],
                                   'detector': row['detector'], 'normal': 0, 'false': 0.0, 'runs': 0, 'runs_comparable': 0,
                                   'normal_unscorable': 0, **{f'exceed_{a}': 0 for a in AXES},
                                   **{f'unscorable_{a}': 0 for a in AXES}, **{f'missed_{a}': 0 for a in AXES},
                                   **{f'offaxis_base_{a}': 0 for a in AXES}, **{f'offaxis_alarm_{a}': 0 for a in AXES}})
        if row['false_alarm_fraction'] is not None:
            t['normal'] += row['normal_samples']
            t['false'] += row['false_alarm_fraction']*row['normal_samples']
            t['normal_unscorable'] += row['normal_unscorable']
            t['runs'] += 1
            t['runs_comparable'] += bool(row.get('comparable'))
        for a in AXES:
            t[f'exceed_{a}'] += row[f'exceed_{a}']
            t[f'unscorable_{a}'] += row[f'unscorable_{a}']
            t[f'missed_{a}'] += row[f'missed_{a}']
            t[f'offaxis_base_{a}'] += row.get(f'offaxis_base_{a}', 0)
            t[f'offaxis_alarm_{a}'] += row.get(f'offaxis_alarm_{a}', 0)
    out = []
    for t in table.values():
        fa = t['false']/t['normal'] if t['normal'] else None
        entry = {'early_warning_s': t['early_warning_s'], 'target': t['target'], 'detector': t['detector'],
                 'false_alarm_fraction': fa, 'comparable': comparable(fa, t['target'], comparable_tolerance),
                 'runs_comparable': f"{t['runs_comparable']}/{t['runs']}",
                 'normal_unscorable_fraction': t['normal_unscorable']/t['normal'] if t['normal'] else None}
        for a in AXES:
            ex, un, miss = t[f'exceed_{a}'], t[f'unscorable_{a}'], t[f'missed_{a}']
            entry[f'r_miss_{a}'] = miss/ex if ex else None  # unscorable counted as missed: same denominator for every detector
            entry[f'r_miss_scorable_{a}'] = (miss - un)/(ex - un) if ex - un else None  # the pilot's definition
            entry[f'r_unscorable_{a}'] = un/ex if ex else None
            base = t[f'offaxis_base_{a}']
            entry[f'r_offaxis_{a}'] = t[f'offaxis_alarm_{a}']/base if base else None  # misattribution while another axis fails
        out.append(entry)
    return out
