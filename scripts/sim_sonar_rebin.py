"""Derive a raycast-sonar dataset with a different range resolution from the hits already recorded (B aperture/range check).

The E-B2 negative result rests on the bound d * tan^2(alpha/2) / 2 against the range bin: 3.8 cm in the sim routes
(256 bins over 9.7 m). Real forward-looking sonars often resolve 1 cm or less, and the bound then clears two bins
beyond about 1.3 m at 20 deg. The raycast sonar records its hits (sonar_points.npy: frame, x, y, z in the sonar frame),
so the image can be re-rendered at any RangeBins with the generator's own render_sonar (same gain, saturation and
noise model, seeded here) without HoloOcean. Only the sonar changes; raw/, openvins/ and reference/ are symlinked.

Writes data/sim/<route>_rb<N>/ with sonar_raycast/ (sonar_XXXX.npy chunks, sonar_t.npy, sonar_points.npy link), a
manifest whose data_root and sonar geometry point at it, and raw/meta.json recording the change.
"""
import argparse
import json
from pathlib import Path
import sys
import types

import numpy as np
import yaml

sys.modules.setdefault('holoocean', types.ModuleType('holoocean'))   # render_sonar is pure numpy; the engine is not needed
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'team/sim'))
from holoocean_generate import CHUNK, render_sonar  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True, help='source data/sim/<route> with raycast sonar_points.npy')
    parser.add_argument('--range-bins', type=int, required=True)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    src = args.run
    sonar_src = src/'sonar_raycast' if (src/'sonar_raycast/sonar_points.npy').exists() else src/'raw'
    out = src.parent/f'{src.name}_rb{args.range_bins}'
    if out.exists():
        parser.error(f'{out} exists; refusing to overwrite')
    meta = json.loads((src/'raw/meta.json').read_text())
    son = dict(meta['route']['sensors']['sonar'])
    rays_per_bin = meta['route']['sensors'].get('raycast_rays_per_bin', 2)
    old_bins, son['RangeBins'] = son['RangeBins'], args.range_bins
    t = np.load(sonar_src/'sonar_t.npy')
    pts = np.load(sonar_src/'sonar_points.npy', mmap_mode='r')
    frame = np.asarray(pts[:, 0]).astype(int)
    order = np.searchsorted(frame, np.arange(len(t) + 1))   # points are stored frame by frame
    if np.any(np.diff(frame) < 0):
        raise SystemExit('sonar_points.npy is not sorted by frame')
    rng = np.random.default_rng(args.seed)
    (out/'raw').mkdir(parents=True)
    (out/'sonar_raycast').mkdir()
    for f in sorted((src/'raw').iterdir()):
        if f.name != 'meta.json':
            (out/'raw'/f.name).symlink_to(f.resolve())
    for d in ('openvins', 'reference', 'audit'):
        if (src/d).exists():
            (out/d).symlink_to((src/d).resolve())
    (out/'sonar_raycast/sonar_points.npy').symlink_to((sonar_src/'sonar_points.npy').resolve())
    np.save(out/'sonar_raycast/sonar_t.npy', t)
    for c in range(0, len(t), CHUNK):
        frames = [render_sonar(np.asarray(pts[order[i]:order[i + 1], 1:4], dtype=float), son, rays_per_bin, rng)
                  for i in range(c, min(c + CHUNK, len(t)))]
        np.save(out/'sonar_raycast'/f'sonar_{c//CHUNK:04d}.npy', np.stack(frames))
    meta['route']['sensors']['sonar'] = son
    meta['sonar_rebin'] = {'tool': 'scripts/sim_sonar_rebin.py', 'source': str(src.resolve()), 'range_bins_source': old_bins,
                           'range_bins': args.range_bins, 'range_bin_m': (son['RangeMax'] - son['RangeMin'])/args.range_bins,
                           'seed': args.seed, 'note': 'same raycast hits re-rendered; ASSUMPTION: echo strength = hit count model'}
    (out/'raw/meta.json').write_text(json.dumps(meta, indent=1))
    if (src/'manifest.yaml').exists():
        man = yaml.safe_load((src/'manifest.yaml').read_text())
        man['data_root'] = str(out.resolve())
        man['run_id'] = out.name
        man['streams']['sonar']['geometry']['RangeBins'] = args.range_bins
        man['derived'] = {**man.get('derived', {}), 'sonar_rebin': meta['sonar_rebin']}
        (out/'manifest.yaml').write_text(yaml.safe_dump(man, sort_keys=False))
    print(json.dumps(meta['sonar_rebin'], indent=1))


if __name__ == '__main__':
    main()
