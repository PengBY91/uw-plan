"""Apply the predeclared W1 engineering regression bounds to immutable run evidence."""
import argparse
import json
from pathlib import Path


def check(run, metrics, inputs):
    outputs = metrics['outputs']
    ref = metrics['references']['corrected']
    return {
        'run_completed': run['status'] == 'playback_completed_requires_output_validation',
        'one_x': run['rate'] == 1.0,
        'complete_input': inputs['complete_input'] and inputs['imu']['expected_count'] == 29120 and inputs['cam0']['expected_count'] == 2912 and inputs['cam1']['expected_count'] == 2912,
        'pose_coverage': metrics['coverage']['camera_slot_coverage'] >= 0.95,
        'valid_outputs': all(o['finite'] and o['strictly_monotonic'] and o['quaternion_norm_range'][0] >= 0.999 and o['quaternion_norm_range'][1] <= 1.001 for o in outputs.values()),
        'no_gaps': not metrics['coverage']['post_start_gaps_above_75ms'],
        'ape': ref['ape_translation_m']['rmse'] <= 0.20,
        'rpe_1s': ref['rpe_1s']['translation_m']['rmse'] <= 0.08,
        'clean_shutdown': all(c == 0 for c in run['exit_codes'].values()) and not run['estimator_child_exit_codes'] and not run['shutdown_warnings'],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, type=Path)
    args = parser.parse_args()
    read = lambda f: json.loads((args.run/f).read_text())
    checks = check(read('run.json'), read('evaluation/metrics.json'), read('input-audit.json'))
    result = {'protocol': 'reports/w1/replay-fix-protocol.md', 'checks': checks, 'passed': all(checks.values())}
    with (args.run/'regression-acceptance.json').open('x') as f:
        f.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
