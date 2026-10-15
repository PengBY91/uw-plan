"""H-A2 on single-axis exceedance intervals (method spec v2.5 M3.2, runbook R4) from a detect-a output directory.

Per axis that DET-DIR sources: total single-axis duration (the axis exceeds, the other sourced axes are within
tolerance), DET-DIR and DET-SCALAR miss on those samples and misattribution (alarm on an axis within tolerance while
another sourced axis exceeds alone), the gap in percentage points, and the decision rule: miss not higher and
misattribution at least 50 points lower, decidable only when the axis has at least --min-single-s and the operating
points are comparable (realised false alarms within 50 % of the target, ablation runbook). Multi-axis exceedance time is
reported only.
usage: summarize_ha2_unit.py runs/detect-a/<out> [--min-single-s 60] [--json out.json]
"""
import argparse
import json
from pathlib import Path

AXES = ('x', 'z', 'yaw')   # DET-DIR sources (detect-a config SOURCE_AXES['conflict_dir'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('detect_out', type=Path)
    p.add_argument('--min-single-s', type=float, default=60.0)
    p.add_argument('--json', type=Path)
    a = p.parse_args()
    pooled = json.loads((a.detect_out/'summary.json').read_text())['pooled']
    by = {(e['early_warning_s'], e['target'], e['detector']): e for e in pooled}
    rows = []
    for (grace, target, det), d in sorted(by.items()):
        if det != 'DET-DIR' or (grace, target, 'DET-SCALAR') not in by:
            continue
        s = by[(grace, target, 'DET-SCALAR')]
        comparable = bool(d['comparable']) and bool(s['comparable'])
        for ax in AXES:
            dur = d[f'single_s_{ax}']
            dm, sm = d[f'r_single_miss_{ax}'], s[f'r_single_miss_{ax}']
            do, so = d[f'r_single_offaxis_{ax}'], s[f'r_single_offaxis_{ax}']
            enough = dur >= a.min_single_s
            meets = None if None in (dm, sm, do, so) else bool(dm <= sm and so - do >= 0.5)
            verdict = ('undecidable: single-axis time below minimum' if not enough else
                       'not comparable: false alarms off target' if not comparable else
                       ('supports H-A2' if meets else 'H-A2 fails'))
            rows.append({'early_warning_s': grace, 'target': target, 'axis': ax, 'single_s': round(dur, 1),
                         'multi_exceed_s': round(d['multi_exceed_s'], 1),
                         'false_alarm_dir': d['false_alarm_fraction'], 'false_alarm_scalar': s['false_alarm_fraction'],
                         'runs_comparable_dir': d['runs_comparable'], 'runs_comparable_scalar': s['runs_comparable'],
                         'miss_dir': dm, 'miss_scalar': sm, 'offaxis_dir': do, 'offaxis_scalar': so,
                         'offaxis_gap_pp': None if None in (do, so) else round(100*(so - do), 1),
                         'meets_rule_if_comparable': meets, 'verdict': verdict})
    f = lambda v: '   -' if v is None else f'{v:.3f}'
    print(f"{'grace':>5} {'target':>6} {'axis':4} {'single_s':>8} {'FA dir/scal':>13} {'miss dir/scal':>13} {'offaxis dir/scal':>16} {'gap_pp':>6} verdict")
    for r in rows:
        print(f"{r['early_warning_s']:5.0f} {r['target']:6.2f} {r['axis']:4} {r['single_s']:8.1f} "
              f"{f(r['false_alarm_dir'])}/{f(r['false_alarm_scalar'])} {f(r['miss_dir'])}/{f(r['miss_scalar'])} "
              f"{f(r['offaxis_dir'])}/{f(r['offaxis_scalar'])}    {str(r['offaxis_gap_pp']):>6} {r['verdict']}")
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps({'detect_out': str(a.detect_out), 'min_single_s': a.min_single_s, 'rows': rows}, indent=2) + '\n')


if __name__ == '__main__':
    main()
