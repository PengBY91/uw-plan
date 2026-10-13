"""Summarise A1/A2 run-a runs (summary.json + eval-traj metrics) and apply the W4-01 selection rule.

Every number is read from run evidence; failed runs are listed, never dropped.
"""
import argparse
import json
import statistics
from pathlib import Path

import yaml


def load(run, eval_dir):
    s = json.loads((run/'summary.json').read_text()) if (run/'summary.json').exists() else {'status': 'missing'}
    out = {'run': run.name, 'status': s.get('status'), 'error': s.get('error'), 'wall_s': s.get('wall_s'),
           'commit': (s.get('code') or {}).get('commit', '')[:7], 'config': s.get('config_file')}
    if s.get('solve_s'):
        out['solve_max_s'] = s['solve_s']['max']
    if s.get('sonar', {}).get('enabled'):
        out['sonar_factors'] = s['factor_totals']['sonar']
        w = s.get('sonar_diagnostics', {}).get('weights')   # absent before the A2 commit
        if w:
            out['sonar_weight_mean'], out['sonar_weight_at_min'] = w['mean'], w['at_minimum_fraction']
    m = run/eval_dir/'metrics.json'
    if not m.exists():
        return out
    e = json.loads(m.read_text())
    out.update(coverage=e['coverage']['coverage_fraction'], ape=e['public']['ape_translation_m']['rmse'],
               ape_max=e['public']['ape_translation_m'].get('max'),
               rpe1=e['public']['rpe_1s'].get('translation_m', {}).get('rmse'),
               rpe5=e['public']['rpe_5s'].get('translation_m', {}).get('rmse'))
    axes = e['risk'].get('axes')
    if isinstance(axes, dict):
        out['risk_max'] = {k: v['abs_error_max'] for k, v in axes.items()}
        out['exceed_s'] = {k: v['exceedance']['exceed_s'] for k, v in axes.items()}
        out['risk_pos_max'] = max(axes[k]['abs_error_max'] for k in ('x', 'y', 'z'))
    return out


def config_name(run_name, dataset, tag):
    return run_name[len(f'sim_{dataset}_{tag}_'):].rsplit('_r', 1)[0]


def family(cfg):
    for f in ('fixed_weight', 'robust_kernel', 'scalar_weight'):
        if cfg.startswith(f):
            return f
    return 'reference'


def _leaves(d, prefix=''):
    for k, v in (d or {}).items():
        if isinstance(v, dict):
            yield from _leaves(v, f'{prefix}{k}.')
        else:
            yield f'{prefix}{k}', v


def n_params(cfg, config_dir, reference=Path('configs/a1/fixed.yaml')):
    """Non-default parameters of a candidate = leaves that differ from (or are absent in) the A1 fixed reference."""
    p = config_dir/f'{cfg}.yaml'
    if not p.exists():
        return 0
    ref = dict(_leaves(yaml.safe_load(reference.read_text())))
    return sum(1 for k, v in _leaves(yaml.safe_load(p.read_text())) if ref.get(k) != v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', type=Path, required=True)
    ap.add_argument('--datasets', nargs='+', required=True)
    ap.add_argument('--tag', default='a2', help='Run-id tag between dataset and config, e.g. a1 or a2')
    ap.add_argument('--run-suffix', default='r01')
    ap.add_argument('--eval-dir', default='eval-traj-causal')
    ap.add_argument('--select', action='store_true', help='Apply configs/w4_matrix.yaml selection per family')
    ap.add_argument('--config-dir', type=Path, default=Path('configs/a2'))
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    table = {}
    for ds in a.datasets:
        for run in sorted(a.runs.glob(f'sim_{ds}_{a.tag}_*_{a.run_suffix}')):
            cfg = config_name(run.name, ds, a.tag)
            if cfg == 'frontend':
                continue
            table.setdefault(cfg, {})[ds] = load(run, a.eval_dir)
    result = {'datasets': a.datasets, 'runs': table}
    if a.select:
        cands = {}
        for cfg, per in table.items():
            ok = all(ds in per and per[ds].get('status') == 'completed' and per[ds].get('ape') is not None
                     for ds in a.datasets)
            c = {'family': family(cfg), 'eligible': ok}
            if ok:
                c['median_ape'] = statistics.median(per[ds]['ape'] for ds in a.datasets)
                rp = [per[ds].get('risk_pos_max') for ds in a.datasets]
                c['median_risk_pos_max'] = statistics.median(rp) if None not in rp else float('inf')
                c['n_params'] = n_params(cfg, a.config_dir)
            cands[cfg] = c
        winners = {}
        for fam in ('fixed_weight', 'robust_kernel', 'scalar_weight'):
            el = [(k, v) for k, v in cands.items() if v['family'] == fam and v['eligible']]
            if el:
                winners[fam] = min(el, key=lambda kv: (kv[1]['median_ape'], kv[1]['median_risk_pos_max'],
                                                       kv[1]['n_params']))[0]
        result.update(candidates=cands, winners=winners)
    a.out.write_text(json.dumps(result, indent=2, default=float) + '\n')
    for cfg, per in sorted(table.items(), key=lambda kv: (family(kv[0]), kv[0])):
        cells = []
        for ds in a.datasets:
            r = per.get(ds, {})
            cells.append(f"{r.get('ape', float('nan')):.3f}" if r.get('ape') is not None else str(r.get('status')))
        extra = ''
        if a.select:
            c = result['candidates'][cfg]
            extra = f" med {c['median_ape']:.3f} risk {c['median_risk_pos_max']:.3f}" if c['eligible'] else ' INELIGIBLE'
        print(f'{cfg:40s}', ' '.join(cells), extra)
    if a.select:
        print('winners', result['winners'])


if __name__ == '__main__':
    main()
