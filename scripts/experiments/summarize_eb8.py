"""E-B8 table: paired change of each deferred-commit variant against the ungated B2a map on the same A0 run, plus the
runbook R4 gate (false-free relative drop >= 30% and surface-recall relative loss <= half of that drop).

usage: summarize_eb8.py --out reports/w5/eb8/table.json
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCENES = {'pilaster_wall_v1_s01': ('runs/sim_pilaster_wall_v1_s01_b2a_a0_r01', 'runs/sim_pilaster_wall_v1_s01_b2a_a0_r01_{v}_r01'),
          'slope_wall_v1_s01': ('runs/sim_slope_wall_v1_s01_b2a_a0_r01', 'runs/sim_slope_wall_v1_s01_b2a_a0_r01_{v}_r01'),
          'stripe_beam_v1_s01': ('runs/sim_stripe_beam_v1_s01_b2a_a0_r02_base_r01', 'runs/sim_stripe_beam_v1_s01_b2a_a0_r02_{v}_r01')}
GT = 'runs/sim_{ds}_b2a_gt_r01'
VARIANTS = ['dh', 'dhgt', 'dor', 'dpre']


def row(run):
    run = ROOT/run
    m = json.loads((run/'eval-map-v2-band0.10/metrics.json').read_text())
    meta = json.loads((run/'map/meta.json').read_text())
    d = meta.get('deferred') or {}
    events = meta['events']
    return {'run': str(run.relative_to(ROOT)), 'false_free': m['overall']['false_free_rate'],
            'free_hazard': m['overall']['free_hazard_fraction'], 'coverage': m['overall']['effective_coverage'],
            'visible_coverage': m['visible']['ratios']['effective_coverage'],
            'visible_false_occupied': m['visible']['ratios']['false_occupied_rate'],
            'surface_recall': m['surface']['visible_surface_recall'], 'occupied_outside': m['surface']['outside_band'],
            'committed_direct_fraction': (d['committed_direct']/events) if d else 1.0,
            'recommitted_fraction': (d['recommitted']/events) if d else 0.0,
            'buffered_reasons': d.get('buffered_reasons')}


def gate(base, var):
    ff_drop = (base['false_free'] - var['false_free'])/base['false_free']
    recall_loss = (base['surface_recall'] - var['surface_recall'])/base['surface_recall']
    return {'false_free_rel_drop': ff_drop, 'surface_recall_rel_loss': recall_loss,
            'coverage_rel_change': (var['coverage'] - base['coverage'])/base['coverage'],
            'passes_r4': bool(ff_drop >= 0.30 and recall_loss <= ff_drop/2)}


def agreement(pattern):
    """How D's commit decisions on A's alarms match the oracle gate's, per observation (descriptive; uses truth)."""
    import csv
    read = lambda v: {(r['kind'], r['index']): r['committed'] == '1' for r in csv.DictReader((ROOT/pattern.format(v=v)/'map/deferred.csv').open())}
    try:
        dh, orc = read('dh'), read('dor')
    except FileNotFoundError:
        return None
    both = sum(dh[k] and orc[k] for k in dh)
    return {'d_commits_within_tolerance': both/max(sum(dh.values()), 1),
            'within_tolerance_committed_by_d': both/max(sum(orc.values()), 1),
            'd_committed': sum(dh.values()), 'oracle_committed': sum(orc.values()), 'observations': len(dh)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    out = {}
    for ds, (base_run, pattern) in SCENES.items():
        base = row(base_run)
        rows = {'base': base, 'gt': row(GT.format(ds=ds))}
        for v in VARIANTS:
            if (ROOT/pattern.format(v=v)/'eval-map-v2-band0.10/metrics.json').exists():
                rows[v] = dict(row(pattern.format(v=v)), **gate(base, row(pattern.format(v=v))))
        rows['dh_vs_oracle'] = agreement(pattern)
        out[ds] = rows
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2) + '\n')
    for ds, rows in out.items():
        print(ds, 'dh vs oracle', rows['dh_vs_oracle'])
        for k, r in rows.items():
            if k == 'dh_vs_oracle':
                continue
            extra = f"  dFF {r['false_free_rel_drop']:+.2f} dRec {r['surface_recall_rel_loss']:+.2f} pass {r['passes_r4']}" if 'passes_r4' in r else ''
            print(f"{ds:22} {k:5} FF {r['false_free']:.3f} cov {r['coverage']:.3f} vcov {r['visible_coverage']:.3f} "
                  f"vFO {r['visible_false_occupied']:.4f} rec {r['surface_recall']:.3f} out {r['occupied_outside']:.3f} "
                  f"commit {r['committed_direct_fraction']:.2f}+{r['recommitted_fraction']:.2f}{extra}")


if __name__ == '__main__':
    main()
