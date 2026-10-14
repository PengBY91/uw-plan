"""E-A1 statistics over uwbench diagnose-a runs: per window, the weak-direction score minus its shifted-window control,
with a block bootstrap (blocks of 10 windows = 20 s at stride 2). Usage: python scripts/experiments/ea1_correspondence_stats.py [glob]."""
import csv, glob, sys
import numpy as np
rng = np.random.default_rng(0)
def boot(d, block=10, n=4000):
    d = np.asarray(d); m = len(d); nb = int(np.ceil(m/block)); out = []
    for _ in range(n):
        starts = rng.integers(0, max(1, m - block + 1), nb)
        out.append(np.concatenate([d[s:s + block] for s in starts])[:m].mean())
    return np.percentile(out, [2.5, 97.5])
pool = {'norm_frac_weak2': [], 'trans_cos_weak': []}
pattern = sys.argv[1] if len(sys.argv) > 1 else 'runs/sim_*_a1_*_diag_r01/directions.csv'
for f in sorted(glob.glob(pattern)):
    rows = list(csv.DictReader(open(f)))
    name = f.split('/')[1][4:-8]
    out = [name.ljust(40)]
    for k in pool:
        a = np.array([float(r[k]) for r in rows if r.get(k) not in ('', None)]); b = np.array([float(r[k + '_shift']) for r in rows if r.get(k) not in ('', None)])
        d = a - b; pool[k].append(d); lo, hi = boot(d)
        out.append(f'{k}: {a.mean():.2f} - {b.mean():.2f} = {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]')
    print(' | '.join(out))
for k, ds in pool.items():
    d = np.concatenate(ds); lo, hi = boot(d)
    print(f'pooled {k}: {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] over {len(d)} windows')
