"""uwbench detect-a: load runs, score every detector, write per_run.csv, stats_<run>.csv, summary.json and, with
health_target, the M3.3 state machine outputs."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml

from uwbench.a.detect.config import CONFIG, SOURCE_AXES
from uwbench.a.detect.scoring import evaluate, label_runs, pooled, union_thresholds
from uwbench.a.detect.signals import budget_horizon_s
from uwbench.a.detect.statistics import load_run
from uwbench.eval_traj import DEFAULTS
from uwbench.metrics.axes import AXES


STATS_COLUMNS = (['t_s', 'scored', 'nis', 'neg_min_features'] + [f'cov_ratio_{a}' for a in AXES] + [f'lambda_{a}' for a in AXES]
                 + [f'tau_{a}' for a in AXES]
                 + ['conflict_z_m', 'conflict_yaw_deg', 'cum_conflict_z_m', 'cum_conflict_yaw_deg', 'cum_conflict_yaw_frozen_deg',
                    'budget_yaw_deg', 'cum_conflict_yaw_truebias_deg', 'cum_conflict_sonar_x_m', 'cum_conflict_sonar_y_m',
                    'cum_conflict_sonar_multi_x_m', 'cum_conflict_sonar_multi_y_m', 'cum_conflict_sonar_map_x_m',
                    'cum_conflict_sonar_map_y_m', 'sonar_map_yaw_deg', 'dir_x_m', 'dir_z_m', 'dir_yaw_deg', 'scalar_max', 'scalar_l2'])


def write_stats(path, run):
    """Per-sample statistics of one run as CSV (STATS_COLUMNS); optional sources are NaN where the run lacks them."""
    s, n = run['stats'], len(run['t'])

    def opt(key, cols, width):
        return s[key][:, cols] if key in s else np.full((n, width), np.nan)
    map_xyh = (np.column_stack([s['conflict_cum_sonar_map'][:, :2], s['conflict_cum_sonar_map_yaw'][:, 3]])
               if 'conflict_cum_sonar_map' in s else np.full((n, 3), np.nan))
    cols = np.column_stack([run['t'], run['scored'], s['DET-NIS'], s['DET-FEAT'], s['DET-COV'], s['lambda'], s['tau'],
                            s['conflict'][:, 2:], s['conflict_cum'][:, 2:], s['conflict_cum_frozen'][:, 3], s['budget_yaw_deg'],
                            opt('conflict_cum_true', [3], 1), opt('conflict_cum_sonar', [0, 1], 2),
                            opt('conflict_cum_sonar_multi', [0, 1], 2), map_xyh,
                            s['conflict_dir'][:, [0, 2, 3]], s['DET-SCALAR'], s['DET-SCALAR-l2']])
    assert cols.shape[1] == len(STATS_COLUMNS)
    np.savetxt(path, cols, delimiter=',', header=','.join(STATS_COLUMNS), comments='', fmt=['%.6f', '%d'] + ['%.6g']*(cols.shape[1] - 2))


def health_outputs(runs, cfg, out):
    """M3.3 state machine on DET-DIR alarms, one directory per run under out/health/. Thresholds are the DET-DIR
    leave-one-run-out thresholds at cfg['health_target'] (strict scoring, no lead time), i.e. the same operating point
    the detector is scored at; the state machine itself never sees labels or errors."""
    from uwbench.a import health
    hcfg = {**health.HEALTH, **(cfg.get('health') or {})}
    label_runs(runs, 0.0)
    rows = []
    for test in runs:
        train = [r for r in runs if r is not test]
        axes = SOURCE_AXES['conflict_dir']
        cols = [np.concatenate([r['stats']['conflict_dir'][r['normal'], a] for r in train]) for a in axes]
        thr = np.full(4, np.inf)
        thr[list(axes)] = union_thresholds(cols, cfg['health_target'], cfg.get('threshold_allocation', 'union'), [cfg['health_target']/8]*len(axes))
        alarm, status = health.latch(test['stats']['conflict_dir'], thr, test['tol'], hcfg['release_tol_fraction'], axes)
        res = health.run(test['t'], alarm, status, test['obs_t'], test['pose_finite'], hcfg)
        d = out/'health'/test['name']
        d.mkdir(parents=True)
        with (d/'transitions.csv').open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['t_s', 'from', 'to', 'reason', 'epoch_id'])
            w.writeheader()
            w.writerows(res['transitions'])
        with (d/'status.csv').open('w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['t_s', 'state', 'reason', 'epoch_id', 'last_observation_s'] + [f'axis_ok_{a}' for a in AXES])
            for i in range(len(test['t'])):
                w.writerow([f"{test['t'][i]:.6f}", res['state'][i], res['reason'][i], res['epoch_id'][i],
                            f"{res['last_observation_s'][i]:.6f}"] + list(status[i]))
        dur = health.durations(test['t'], res['state'])
        # descriptive only (truth used after the fact): how much exceedance time was spent outside NORMAL
        ex = test['exceed'].any(axis=1)
        flagged = np.array([s != 'NORMAL' for s in res['state']])
        rows.append({'run': test['name'], 'thresholds': [None if not np.isfinite(x) else float(x) for x in thr],
                     'durations_s': dur, 'transitions': len(res['transitions']),
                     'exceedance_samples': int(ex.sum()), 'exceedance_not_normal_fraction': float(flagged[ex].mean()) if ex.any() else None,
                     'normal_samples': int(test['normal'].sum()),
                     'normal_not_normal_fraction': float(flagged[test['normal']].mean()) if test['normal'].any() else None})
    (out/'health'/'summary.json').write_text(json.dumps({'config': hcfg, 'target': cfg['health_target'], 'runs': rows,
        'note': 'product output (M3.3), not research evidence; truth only in the descriptive overlap columns'}, indent=2) + '\n')
    return rows


def main(argv):
    parser = argparse.ArgumentParser(prog='uwbench detect-a')
    parser.add_argument('--runs', type=Path, nargs='+', required=True, help='run directories (diagnostic OpenVINS build)')
    parser.add_argument('--config', type=Path, help='YAML overriding the pilot defaults')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    cfg = {**CONFIG, **(yaml.safe_load(args.config.read_text()) if args.config else {})}
    args.out.mkdir(parents=True, exist_ok=False)  # before the slow loading, so an existing directory fails early
    runs = [load_run(r, cfg) for r in args.runs]
    results = evaluate(runs, cfg)
    summary = pooled(results, cfg.get('comparable_tolerance', 0.5))
    with (args.out/'per_run.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    for r in runs:
        write_stats(args.out/f'stats_{r["name"]}.csv', r)
    tol_yaw = DEFAULTS['risk']['tolerance_yaw_deg']
    horizon = {r['name']: budget_horizon_s(r['gyro_noise'], cfg['budget_n_sigma'], tol_yaw) for r in runs}
    record = {'pilot': True, 'config': cfg, 'runs': [str(r) for r in args.runs], 'pooled': summary,
              'tolerances': {r['name']: [float(x) for x in r['tol']] for r in runs},
              'gyro_noise': {r['name']: {'noise_density': r['gyro_noise'][0], 'random_walk': r['gyro_noise'][1],
                                         'source': r['gyro_noise_source']} for r in runs},
              'frozen_budget_reaches_yaw_tolerance_s': horizon,
              'sonar_planes': {r['name']: r['stats'].get('sonar_planes') for r in runs},
              'sonar_map_points': {r['name']: r['stats'].get('sonar_map_points') for r in runs},
              'scoring': 'r_miss counts unscorable (NaN) exceedance samples as missed, same denominator for every detector; '
                         'r_miss_scorable leaves them out (pilot definition); comparable = realised false-alarm fraction '
                         f"within {cfg.get('comparable_tolerance', 0.5):.0%} of the target (ablation runbook)",
              'notes': 'tau in the last LORO fold is what stats_*.csv holds; false alarms are normal-sample fractions'}
    (args.out/'summary.json').write_text(json.dumps(record, indent=2) + '\n')
    if cfg.get('health_target') is not None:
        health_outputs(runs, cfg, args.out)
    for e in summary:
        print(json.dumps(e))
    return 0
