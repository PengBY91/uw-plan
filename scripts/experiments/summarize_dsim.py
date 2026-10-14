"""D-sim scenario validity (checklist W5-11): per run, which body axes exceed tolerance and whether they co-occur.

Reads runs/<run>/{eval-traj/metrics.json, eval-traj/axis_errors.csv, input-audit.json}. A run counts as
axis-independent ("decorrelated") when exactly one axis exceeds, or when the exceeding axes overlap in less than
--overlap of the smaller axis' exceedance time; otherwise it is 'co-occurring' and is excluded from the H-A2 set.
Prints a markdown table and, with --json, writes the classification for detect-a selection.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

AXES = ['x', 'y', 'z', 'yaw']


def classify(run, overlap_max):
    m = json.loads((run/'eval-traj/metrics.json').read_text())
    if 'risk' not in m:
        return {'run': run.name, 'kind': 'failed', 'reason': m.get('reason')}
    tol = m['risk']['tolerances']
    rows = list(csv.DictReader((run/'eval-traj/axis_errors.csv').open()))
    scored = np.array([r['in_align_window'] == '0' for r in rows])
    err = np.array([[float(r[f'err_{a}']) for a in AXES] for r in rows])[scored]
    ex = np.abs(err) > np.array([tol[a] for a in AXES])
    ex &= np.isfinite(err)
    axes = [a for i, a in enumerate(AXES) if ex[:, i].any()]
    pair = {}
    for i, a in enumerate(AXES):
        for j, b in enumerate(AXES):
            if j > i and ex[:, i].any() and ex[:, j].any():
                pair[f'{a}+{b}'] = float((ex[:, i] & ex[:, j]).sum()/min(ex[:, i].sum(), ex[:, j].sum()))
    only = {a: float((ex[:, i] & (ex.sum(axis=1) == 1)).sum()) for i, a in enumerate(AXES)}
    dt = float(np.median(np.diff(np.array([float(r['t_s']) for r in rows])))) if len(rows) > 1 else 0.0
    audit = json.loads((run/'input-audit.json').read_text()) if (run/'input-audit.json').exists() else {}
    missing = sum(v.get('missing_count', 0) for v in audit.values() if isinstance(v, dict))
    kind = 'none' if not axes else 'single' if len(axes) == 1 else \
        'decorrelated' if max(pair.values()) < overlap_max else 'co-occurring'
    ax = m['risk']['axes']
    return {'run': run.name, 'axes': axes, 'kind': kind, 'overlap': pair, 'input_missing': missing if audit else None,
            'exceed_s': {a: ax[a]['exceedance']['exceed_s'] for a in AXES}, 'max_err': {a: ax[a]['abs_error_max'] for a in AXES},
            'alone_s': {a: only[a]*dt for a in AXES}, 'ape_rmse_m': m['public']['ape_translation_m']['rmse'],
            'duration_s': float(rows[-1]['t_s']) - float(rows[0]['t_s']) if rows else 0.0}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('runs', nargs='+', type=Path)
    p.add_argument('--overlap', type=float, default=0.1, help='max pairwise overlap fraction for "decorrelated"')
    p.add_argument('--json', type=Path)
    args = p.parse_args()
    out = [classify(r, args.overlap) for r in args.runs if (r/'eval-traj/metrics.json').exists()]
    print('| 运行 | 时长（s） | 输入缺样 | APE（m） | 越限时长 x / y / z / 航向（s） | 最大误差 x / y / z（m）/ 航向（°） | 越限轴 | 单独越限时长 x / y / z / 航向（s） | 轴间重叠 | 类别 |')
    print('| --- | ---: | ---: | ---: | --- | --- | --- | --- | --- | --- |')
    for o in out:
        if o['kind'] == 'failed':
            print(f"| {o['run']} | — | — | — | 估计器无输出：{o['reason']} | — | — | — | — | failed |")
            continue
        e, mx = o['exceed_s'], o['max_err']
        ov = ', '.join(f'{k} {100*v:.0f}%' for k, v in o['overlap'].items()) or '—'
        print(f"| {o['run']} | {o['duration_s']:.0f} | {o['input_missing']} | {o['ape_rmse_m']:.3f} | "
              f"{e['x']:.1f} / {e['y']:.1f} / {e['z']:.1f} / {e['yaw']:.1f} | "
              f"{mx['x']:.2f} / {mx['y']:.2f} / {mx['z']:.2f} / {mx['yaw']:.1f} | {'+'.join(o['axes']) or '无'} | {' / '.join(f"{o['alone_s'][a]:.1f}" for a in AXES)} | {ov} | {o['kind']} |")
    if args.json:
        args.json.write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    main()
