"""Rerun existing detect-a outputs with the method-spec M3.1 bias freeze (bias_freeze: still, default since
2026-10-14) and otherwise their recorded configuration and runs.

For each runs/detect-a/<name>: read summary.json (config, runs), write runs/detect-a/<name>_still/config.yaml with
bias_freeze still, restore each run's dataset bag from raw (sim_to_bag.py --restore-bag, checked against
checksums.sha256) when ros2/ is missing, run uwbench detect-a, and delete the bags it restored afterwards.
usage: rerun_detect_still.py <name> [<name> ...]
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
PY = str(ROOT/'.venv-eval/bin/python')


def dataset_of(run):
    return Path(json.loads((ROOT/run/'run.json').read_text())['bag']).parent


def main(names):
    for name in names:
        src = ROOT/'runs/detect-a'/name
        out = ROOT/'runs/detect-a'/(name + '_still')
        if (out/'summary.json').exists():
            print(name, 'done already')
            continue
        rec = json.loads((src/'summary.json').read_text())
        cfg = dict(rec['config'], bias_freeze='still')
        cfg.setdefault('still_window_s', 3.0)
        restored = []
        for run in rec['runs']:
            ds = dataset_of(run)
            if not (ds/'ros2').exists():
                subprocess.run([PY, 'scripts/data/sim_to_bag.py', '--run', str(ds), '--restore-bag'], cwd=ROOT, check=True,
                               stdout=subprocess.DEVNULL)
                restored.append(ds)
        cfg_path = ROOT/'runs/detect-a'/(name + '_still.yaml')
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        log = (ROOT/'runs/detect-a'/(name + '_still.log')).open('w')
        rc = subprocess.run([PY, '-m', 'uwbench', 'detect-a', '--runs', *rec['runs'], '--config', str(cfg_path),
                             '--out', str(out)], cwd=ROOT, env={'PYTHONPATH': 'team/uwbench', 'PATH': '/usr/bin:/bin'},
                            stdout=log, stderr=subprocess.STDOUT).returncode
        for ds in restored:
            shutil.rmtree(ds/'ros2')
        print(name, 'rc', rc, 'restored', len(restored), flush=True)
        if rc:
            sys.exit(rc)


if __name__ == '__main__':
    main(sys.argv[1:])
