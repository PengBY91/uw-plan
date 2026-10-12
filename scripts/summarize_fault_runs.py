"""Summarise W4-05 fault-injection A0 runs against their unfaulted control (all numbers read from run evidence)."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load(run):
    out = {'run': run.name, 'status': json.loads((run/'run.json').read_text())['status']}
    audit = run/'input-audit.json'
    if audit.exists():
        a = json.loads(audit.read_text())
        out['input'] = {k: {'expected': a[k]['expected_count'], 'received': a[k]['received_count'],
                            'missing': a[k]['missing_count']} for k in ['imu', 'cam0', 'cam1']}
        out['complete_input'] = a['complete_input']
    m = run/'eval-traj/metrics.json'
    if not m.exists():
        return out
    e = json.loads(m.read_text())
    out.update(estimates=e['counts']['estimate'], coverage=e['coverage']['coverage_fraction'],
               lost=[(i['start'], i['duration_s'], i['kind']) for i in e['coverage']['lost_intervals']],
               matched=e['counts']['matched'], ape_rmse=e['public']['ape_translation_m']['rmse'],
               rpe1_rmse=e['public']['rpe_1s'].get('translation_m', {}).get('rmse'),   # None when not evaluable
               risk_max={k: v['abs_error_max'] for k, v in e['risk']['axes'].items()} if isinstance(e['risk'].get('axes'), dict) else None,
               exceed_s={k: v['exceedance']['exceed_s'] for k, v in e['risk']['axes'].items()} if isinstance(e['risk'].get('axes'), dict) else None)
    if not isinstance(e['risk'].get('axes'), dict):
        return out
    with open(run/'eval-traj/axis_errors.csv') as f:
        rows = [r for r in csv.DictReader(f) if r['in_align_window'] == '0']
    out['_axis'] = np.array([[float(r['t_s']), float(r['err_x']), float(r['err_y']), float(r['err_z']), float(r['err_yaw'])] for r in rows])
    return out


def window_max(axis, t0, t1):
    sel = (axis[:, 0] >= t0) & (axis[:, 0] < t1)
    return np.abs(axis[sel, 1:]).max(axis=0) if sel.any() else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--derived', type=Path, required=True, help='Directory of derived fault bags (fault.json)')
    parser.add_argument('--parent', required=True, help='Parent tag, e.g. v101 or sbeam01')
    parser.add_argument('--control', required=True, help='Control run id')
    parser.add_argument('--after-s', type=float, default=15.0, help='Window after fault start for the local max error')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    control = load(args.runs/args.control)
    rows = [dict(control, fault='none')]
    for run in sorted(args.runs.glob(f'fault_{args.parent}_*_a0_r*')):
        if run.name == args.control or 'control' in run.name:
            continue
        name = run.name[len(f'fault_{args.parent}_'):].rsplit('_a0_', 1)[0]
        fj = args.derived/f'{args.parent}_{name}'/'fault.json'
        r = load(run)
        if fj.exists():
            f = json.loads(fj.read_text())
            r.update(fault=f['fault'], stream=f['stream'], parameters=f['parameters'], intervals=f['intervals'],
                     modified_time=f['modified_time'], dropped={k: v['dropped'] for k, v in f['checks'].items()})
        rows.append(r)
    epoch = None
    for r in rows:
        if 'intervals' in r and r['intervals'] and '_axis' in r and '_axis' in control:
            t0 = r['intervals'][0]['start_ns']/1e9 if r['fault'] != 'offset' else None
            if t0 is None:
                continue
            r['local_max_after_fault'] = window_max(r['_axis'], t0, t0 + args.after_s).tolist() if window_max(r['_axis'], t0, t0 + args.after_s) is not None else None
            c = window_max(control['_axis'], t0, t0 + args.after_s)
            r['control_local_max_same_window'] = c.tolist() if c is not None else None
            r['gaps_after_fault'] = [(round(s - t0, 3), round(d, 3), k) for s, d, k in r.get('lost', []) if s >= t0 - 1]
    for r in rows:
        r.pop('_axis', None)
    args.out.write_text(json.dumps(rows, indent=2, default=float) + '\n')
    for r in rows:
        print(r['run'], r.get('fault'), r.get('parameters'), r.get('status'), 'cov %.3f' % r['coverage'] if 'coverage' in r else '',
              'APE %.3f' % r['ape_rmse'] if 'ape_rmse' in r else '', r.get('exceed_s'), r.get('local_max_after_fault'))


if __name__ == '__main__':
    main()
