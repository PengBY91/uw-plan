"""Write manifests/dsim_w5.yaml: index of the D-sim datasets (checklist W5-11) with role, route config, generation
record and the hashes that tie each derived g1 dataset to its raw run. Per-dataset manifests stay in data/sim/<ds>/.
usage: python scripts/data/dsim_manifest.py
"""
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
ROLES = {'probe_vstripe_heave_v1': 'z-only candidate, design 1 (failed: no degradation)',
         'probe_far_approach_v1': 'x-only candidate, design 1 probe, fog 1 (failed: never initialised)',
         'probe_far_approach_fog0_v1': 'x-only candidate, design 1 probe, fog 0.3 (failed: diverged from initialisation)',
         'probe_retreat_approach_v1': 'x-only candidate, design 2 (co-occurring x+y+z)',
         'vstripe_shuttle_v1': 'z-only candidate, design 2', 'vstripe_shuttle_b_v1': 'z-only candidate, design 2',
         'vstripe_shuttle_fog4_v1': 'z-only candidate, design 2',
         'normal_dam_v1': 'normal family dam', 'normal_dam_b_v1': 'normal family dam',
         'normal_checker_v1': 'normal family checker', 'normal_checker_fog1_v1': 'normal family checker',
         'normal_pilaster_v1': 'normal family pilaster', 'normal_pilaster_b_v1': 'normal family pilaster',
         'normal_checker_c_v1': 'normal family checker (2026-10-14 follow-up)',
         'normal_checker_t3_v1': 'normal family checker, 3 m tiles (2026-10-14 follow-up)',
         'normal_checker_t15_v1': 'normal family checker, 1.5 m tiles (2026-10-14 follow-up)'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def main():
    entries = []
    for route, role in ROLES.items():
        src = ROOT/'data/sim'/f'{route}_s01'
        if not (src/'raw/meta.json').exists():
            continue
        meta = json.loads((src/'raw/meta.json').read_text())
        g1 = ROOT/'data/sim'/f'{route}_s01_g1'
        g1meta = json.loads((g1/'raw/meta.json').read_text()) if (g1/'raw/meta.json').exists() else {}
        run = ROOT/'runs'/f'sim_{route}_s01_g1_a0_r01'
        entries.append({
            'route': route, 'role': role, 'route_config': f'configs/sim/{route}.json',
            'route_config_sha256': sha(ROOT/'configs/sim'/f'{route}.json'),
            'raw': str(src.relative_to(ROOT)/'raw'), 'raw_meta_sha256': sha(src/'raw/meta.json'),
            'raw_imu_sha256': sha(src/'raw/imu.npy'), 'duration_s': meta['route']['waypoints'][-1]['t'],
            'generation_wall_s': round(meta['wall_seconds'], 1), 'holoocean_version': meta['holoocean_version'],
            'g1_dataset': str(g1.relative_to(ROOT)) if g1.exists() else None,
            'g1_seed': g1meta.get('imu_resynthesis', {}).get('seed'),
            'g1_manifest_sha256': sha(g1/'manifest.yaml'), 'g1_checksums_sha256': sha(g1/'checksums.sha256'),
            'a0_run': str(run.relative_to(ROOT)) if run.exists() else None})
    doc = {'title': 'D-sim supplementary set (W5-11), pre-hardware', 'created': '2026-10-14',
           'imu': 'gyro grade g1 (EuRoC ADIS16448 Kalibr: N 1.6968e-4, K 1.9393e-5) via scripts/data/sim_imu_grade.py; '
                  'ASSUMPTION, pre-Allan placeholder for the platform IMU; accelerometer as simulated',
           'report': 'reports/w5/d-sim.md', 'datasets': entries}
    (ROOT/'manifests/dsim_w5.yaml').write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True))
    print(len(entries), 'datasets')


if __name__ == '__main__':
    main()
