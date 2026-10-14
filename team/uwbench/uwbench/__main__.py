"""uwbench CLI contract from the month-1 runbook E7 and month-2 runbook R5.

Every contracted subcommand is listed; unimplemented ones exit non-zero instead of writing placeholders.
"""
import argparse
import sys

# name: (track, owning prompt, contract)
CONTRACT = {
    'audit': ('shared', 'P03', 'manifest -> completeness, timing and unit report; missing required items exit non-zero'),
    'eval-traj': ('A', 'P04', 'manifest, estimate -> APE/RPE, matched coverage, failure and per-axis exceedance intervals'),
    'depth': ('B', 'P05', 'manifest, stereo config -> depth, valid mask, range-binned error'),
    'map': ('B', 'P05/P09', 'manifest, poses, map config -> tri-state voxels, provenance, update time'),
    'eval-map': ('B', 'P05/P09', 'manifest, voxels, reference ROI -> coverage, false free, structure error'),
    'run-a': ('A', 'P08/P10', 'manifest, A config -> fixed-backend run, trajectory and factor diagnostics'),
    'diagnose-a': ('A', 'P14', 'manifest, method config -> spectrum, directions, gauge, axis_info.csv'),
    'detect-a': ('A', 'P18', 'estimate runs, detect config -> causal detector replay, per-axis alarms, false-alarm operating points'),
    'input-integrity': ('A', 'P18', 'bag (+ fault.json), camera-IMU calib -> online M3.4 input checks, transitions, offset estimates'),
    'motion-b': ('B', 'P19', 'segment manifest, candidates -> motion_info.csv with predicted vs actual shrinkage'),
    'check-method': ('shared', 'P14-P20', 'suite, check config -> algebra, geometry, state and evidence checks'),
    'benchmark': ('shared', 'P11/P17/P21', 'matrix config -> independent runs, failures and resources'),
    'compare': ('shared', 'P21/P23', 'run root, protocol -> ablations, denominators, paired statistics'),
    'freeze': ('shared', 'P22', 'release config -> commit, patch, dependency, parameter and evaluator hashes'),
    'verify-freeze': ('shared', 'P22', 'release dir -> missing or changed items; non-zero on any difference'),
    'report': ('shared', 'P12/P22/P24', 'completed runs -> gate report with evidence links'),
}


def _audit(argv):
    from uwbench.audit import main
    return main(argv)


def _eval_traj(argv):
    from uwbench.eval_traj import main
    return main(argv)


def _run_a(argv):
    from uwbench.a.backend.cli import main  # needs gtsam/cv2 (in .venv-eval since 2026-10-12)
    return main(argv)


def _depth(argv):
    from uwbench.b.depth import main  # needs cv2 (in .venv-eval since 2026-10-12)
    return main(argv)


def _map(argv):
    from uwbench.b.cli import map_main
    return map_main(argv)


def _eval_map(argv):
    from uwbench.b.cli import eval_main
    return eval_main(argv)


def _diagnose_a(argv):
    from uwbench.a.backend.diagnose import main  # needs gtsam
    return main(argv)


def _detect_a(argv):
    from uwbench.a.detect.cli import main
    return main(argv)


def _input_integrity(argv):
    from uwbench.a.integrity import main  # needs cv2
    return main(argv)


def _motion_b(argv):
    from uwbench.b.motion import main
    return main(argv)


IMPLEMENTED = {'audit': _audit, 'eval-traj': _eval_traj, 'depth': _depth, 'map': _map, 'eval-map': _eval_map,
               'run-a': _run_a, 'diagnose-a': _diagnose_a, 'detect-a': _detect_a, 'input-integrity': _input_integrity, 'motion-b': _motion_b}


def main(argv=None):
    parser = argparse.ArgumentParser(prog='uwbench')
    parser.add_argument('command', choices=sorted(CONTRACT) + ['list'])
    args, rest = parser.parse_known_args(argv)
    if args.command == 'list':
        for name, (track, owner, text) in CONTRACT.items():
            state = 'implemented' if name in IMPLEMENTED else 'NOT IMPLEMENTED'
            print(f'{name:14} {track:6} {owner:12} {state:16} {text}')
        return 0
    if args.command not in IMPLEMENTED:
        track, owner, text = CONTRACT[args.command]
        print(f'uwbench {args.command} is not implemented yet ({owner}, track {track}): {text}', file=sys.stderr)
        return 2
    return IMPLEMENTED[args.command](rest)


if __name__ == '__main__':
    raise SystemExit(main())
