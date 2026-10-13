"""Apply the predeclared W1 engineering regression bounds to immutable run evidence."""
import argparse
import json
from pathlib import Path


V101_COUNTS = {'imu': 29120, 'cam0': 2912, 'cam1': 2912}


def check(run, metrics, inputs, counts=V101_COUNTS):
    outputs = metrics['outputs']
    ref = metrics['references']['corrected']
    return {
        'run_completed': run['status'] == 'playback_completed_requires_output_validation',
        'one_x': run['rate'] == 1.0,
        'complete_input': inputs['complete_input'] and all(inputs[k]['expected_count'] == n for k, n in counts.items()),
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
    parser.add_argument('--dataset-audit', type=Path,
                        help='audit_euroc.py audit.json giving the expected message counts (default: V1_01 counts)')
    args = parser.parse_args()
    read = lambda f: json.loads((args.run/f).read_text())
    counts = V101_COUNTS
    if args.dataset_audit:
        topics = json.loads(args.dataset_audit.read_text())['source']['topics']
        counts = {'imu': topics['/imu0']['count'], 'cam0': topics['/cam0/image_raw']['count'],
                  'cam1': topics['/cam1/image_raw']['count']}
    checks = check(read('run.json'), read('evaluation/metrics.json'), read('input-audit.json'), counts)
    result = {'protocol': 'reports/w1/replay-fix-protocol.md', 'expected_counts': counts,
              'checks': checks, 'passed': all(checks.values())}
    with (args.run/'regression-acceptance.json').open('x') as f:
        f.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
