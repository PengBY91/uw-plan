"""Score uwbench input-integrity runs against the M3.4 tolerances (month-2 runbook R4, input-integrity acceptance).

Expected action per W4-05 fault (reports/w4/fault-injection.md): IMU outage >= 0.5 s, camera outage >= 3 s and an
IMU offset >= 40 ms must reach LOST with the matching reason; short camera dropouts, the 1 s camera outage, offsets
<= 15 ms and the clean controls must raise nothing. A time-offset fault that the monitor cannot observe (no window
with enough rotation and agreeing visual rotation) is reported as 'unobservable', not as a pass.
usage: python summarize_input_integrity.py <run_root> [--json out.json]
"""
import argparse
import json
import re
from pathlib import Path

FAULTS = Path(__file__).resolve().parents[2]/'data/derived/fault'


def expected(name):
    """(fault kind, expected action, expected LOST reason) from the run name."""
    m = re.search(r'_(camdrop\d+|camout\d+s|imuout\d+ms|imuoff\d+|control)$', name)
    kind = m.group(1) if m else 'control'
    if kind.startswith('imuout'):
        return kind, 'LOST', 'imu_outage'
    if kind.startswith('camout') and int(kind[6:-1]) >= 3:
        return kind, 'LOST', 'cam_outage'
    if kind.startswith('imuoff') and int(kind[6:]) >= 40:
        return kind, 'LOST', 'time_offset'
    return kind, 'none', None


def onset(name):
    """Fault onset (sample clock, s) and bag start, from this fault's or a sibling's fault.json."""
    f = FAULTS/name/'fault.json'
    prefix = name.split('_')[0]
    starts = []
    for g in sorted(FAULTS.glob(prefix + '_*/fault.json')):
        d = json.loads(g.read_text())
        for i in d['intervals']:
            starts.append(i['start_ns']/1e9 - i['start_s_from_bag_start'])
    t0 = min(starts) if starts else None
    if f.exists():
        d = json.loads(f.read_text())
        if d['intervals']:
            return d['intervals'][0]['start_ns']/1e9, t0
    return t0, t0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    p.add_argument('--json', type=Path)
    a = p.parse_args()
    rows = []
    for s in sorted(a.root.glob('*/summary.json')):
        name = s.parent.name
        d = json.loads(s.read_text())
        kind, want, want_reason = expected(name)
        t_on, t0 = onset(name)
        lost = d['first_lost']
        got = 'LOST' if lost else ('WARN' if d['warn_reasons'] else 'none')
        o = d['offset']
        if want == 'LOST':
            ok = bool(lost) and lost['reason'] == want_reason
        else:
            ok = got == 'none'
        verdict = 'pass' if ok else 'fail'
        if kind.startswith('imuoff') and o['observable'] == 0:
            verdict = 'unobservable' if want == 'LOST' or got == 'none' else verdict
        rows.append({'run': name, 'fault': kind, 'expected': want, 'got': got,
                     'reason': lost['reason'] if lost else ','.join(d['warn_reasons']) or None,
                     'delay_s': round(lost['t_s'] - t_on, 3) if lost and t_on is not None else None,
                     'lost_from_start_s': round(lost['t_s'] - t0, 2) if lost and t0 is not None else None,
                     'offset_windows': o['windows'], 'offset_observable': o['observable'],
                     'offset_median_ms': None if o['median_ms'] is None else round(o['median_ms'], 2),
                     'verdict': verdict})
    w = max(len(r['run']) for r in rows) if rows else 10
    print(f"{'run':{w}} {'fault':12} {'expect':6} {'got':5} {'reason':18} {'delay_s':>8} {'obs/win':>8} {'off_ms':>7} verdict")
    for r in rows:
        print(f"{r['run']:{w}} {r['fault']:12} {r['expected']:6} {r['got']:5} {str(r['reason']):18} "
              f"{str(r['delay_s']):>8} {r['offset_observable']:>3}/{r['offset_windows']:<4} {str(r['offset_median_ms']):>7} {r['verdict']}")
    if a.json:
        a.json.write_text(json.dumps(rows, indent=2) + '\n')


if __name__ == '__main__':
    main()
