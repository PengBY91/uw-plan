"""Run an isolated, bounded ROS2 baseline and keep failure evidence."""
import argparse
import datetime
import hashlib
import json
import os
import re
from pathlib import Path
import signal
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lab', type=Path, required=True)
    parser.add_argument('--bag', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--rate', type=float, default=1.0)
    parser.add_argument('--imu-history-depth', type=int, default=1000)
    parser.add_argument('--imu-reliability', choices=['reliable', 'best_effort'], default='reliable')
    parser.add_argument('--config-dir', type=Path, help='OpenVINS YAML directory (default: configs/euroc_v101)')
    parser.add_argument('--dataset-audit', type=Path, help='uwbench audit.json that must have passed (default: W1 EuRoC audit)')
    parser.add_argument('--topics', nargs='+', default=['/cam0/image_raw', '/cam1/image_raw', '/imu0'])
    args = parser.parse_args()
    if args.rate <= 0 or args.imu_history_depth <= 0 or Path(args.run_id).name != args.run_id:
        parser.error('Invalid playback rate or run ID')
    dirty = subprocess.check_output(['git', '-C', str(args.lab), 'status', '--porcelain'], text=True)
    if dirty.strip():
        parser.error('Commit team code/config changes before starting a versioned baseline run')
    source_dirty = subprocess.check_output(['git', '-C', str(args.lab/'ros_ws/src/open_vins'), 'status', '--porcelain'], text=True)
    if source_dirty.strip():
        parser.error('Commit OpenVINS source changes before a versioned baseline run')
    if args.dataset_audit:
        if not json.loads(args.dataset_audit.read_text()).get('passed'):
            parser.error('uwbench dataset audit has not passed')
    elif json.loads((args.lab/'reports/w1/v101-data-audit/audit.json').read_text())['status'] != 'passed':
        parser.error('Dataset audit has not passed')
    config_directory = (args.config_dir or args.lab/'configs/euroc_v101').resolve()
    run = args.lab/'runs'/args.run_id
    run.mkdir(parents=True, exist_ok=False)
    processes, handles = {}, []
    record = {'run_id': args.run_id, 'start_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'rate': args.rate, 'bag': str(args.bag), 'config_dir': str(config_directory), 'topics': args.topics,
              'dataset_audit': str(args.dataset_audit) if args.dataset_audit else 'reports/w1/v101-data-audit/audit.json',
              'imu_reliability': args.imu_reliability, 'imu_history_depth': args.imu_history_depth, 'ros_domain_id': os.environ.get('ROS_DOMAIN_ID'),
              'rmw_implementation': os.environ.get('RMW_IMPLEMENTATION', 'default'),
              'cyclonedds_uri': os.environ.get('CYCLONEDDS_URI'),
              'clock_mode': 'header timestamps; no simulated clock',
              'team_commit': subprocess.check_output(['git', '-C', str(args.lab), 'rev-parse', 'HEAD'], text=True).strip(),
              'commands': {},
              'openvins_libraries_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.lab/'ros_ws/install/ov_msckf/lib').glob('*.so')},
              'openvins_commit': subprocess.check_output(['git', '-C', str(args.lab/'ros_ws/src/open_vins'), 'rev-parse', 'HEAD'], text=True).strip(),
              'openvins_binary_sha256': hashlib.sha256((args.lab/'ros_ws/install/ov_msckf/lib/ov_msckf/run_subscribe_msckf').read_bytes()).hexdigest(),
              'udp_rmem_max': Path('/proc/sys/net/core/rmem_max').read_text().strip(),
              'communication_config_sha256': hashlib.sha256((args.lab/'configs/cyclonedds-local.xml').read_bytes()).hexdigest() if os.environ.get('RMW_IMPLEMENTATION') == 'rmw_cyclonedds_cpp' else None}

    def start(name, command):
        handle = (run/(name+'.log')).open('w')
        handles.append(handle)
        record['commands'][name] = command
        env = dict(os.environ)
        if name == 'estimator':
            env['OPENVINS_INPUT_TRACE'] = str(run/'input-trace.csv')
        processes[name] = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                                           start_new_session=True, env=env)
        return processes[name]

    def stop(name):
        p = processes.get(name)
        if p is None or p.poll() is not None:
            return
        p.send_signal(signal.SIGINT)  # ROS launch forwards once to its child; avoid duplicate SIGINT.
        try:
            p.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGTERM)
            p.wait(timeout=10)

    try:
        start('estimator', ['ros2', 'launch', 'ov_msckf', 'subscribe.launch.py',
              'config_path:='+str(config_directory/'estimator_config.yaml'),
              'use_stereo:=true', 'max_cameras:=2', 'rviz_enable:=false',
              'imu_history_depth:='+str(args.imu_history_depth),
              'imu_reliable:='+str(args.imu_reliability == 'reliable').lower()])
        deadline = time.monotonic() + 35
        while time.monotonic() < deadline:
            if processes['estimator'].poll() is not None:
                raise RuntimeError('Estimator exited during startup')
            topics = subprocess.run(['ros2', 'topic', 'list', '--no-daemon'],
                                    capture_output=True, text=True, timeout=10)
            if '/ov_msckf/poseimu' in topics.stdout.splitlines():
                break
            time.sleep(1)
        else:
            raise RuntimeError('Estimator pose publisher was not discovered')
        start('recorder', ['ros2', 'bag', 'record', '-s', 'mcap', '-o', str(run/'estimate_bag'),
                           '--topics', '/ov_msckf/poseimu', '/ov_msckf/odomimu'])
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if processes['recorder'].poll() is not None:
                raise RuntimeError('Recorder exited before playback')
            text = (run/'recorder.log').read_text(errors='replace')
            if "Subscribed to topic '/ov_msckf/poseimu'" in text and "Subscribed to topic '/ov_msckf/odomimu'" in text:
                break
            time.sleep(0.5)
        else:
            raise RuntimeError('Recorder subscriptions were not confirmed')
        record['playback_start_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        start_time = time.monotonic()
        player = start('player', ['ros2', 'bag', 'play', str(args.bag), '--rate', str(args.rate),
                       '--read-ahead-queue-size', '1000', '--disable-keyboard-controls',
                       '--topics', *args.topics])
        deadline = start_time + 600 / args.rate
        while player.poll() is None:
            if processes['estimator'].poll() is not None or processes['recorder'].poll() is not None:
                raise RuntimeError('Estimator or recorder exited during playback')
            if time.monotonic() > deadline:
                raise RuntimeError('Playback exceeded bounded timeout')
            time.sleep(1)
        record['playback_wall_seconds'] = time.monotonic() - start_time
        if player.returncode:
            raise RuntimeError('Playback failed')
        time.sleep(5)
        record['status'] = 'playback_completed_requires_output_validation'
    except Exception as exc:
        record['status'] = 'failed'
        record['error'] = str(exc)
    finally:
        for name in ['player', 'estimator', 'recorder']:
            stop(name)
        for handle in handles:
            handle.close()
        record['exit_codes'] = {name: p.returncode for name, p in processes.items()}
        record['end_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        estimator_log = (run/'estimator.log').read_text(errors='replace') if (run/'estimator.log').exists() else ''
        record['estimator_child_exit_codes'] = [int(code) for code in re.findall(r'process has died \[pid \d+, exit code (-?\d+)', estimator_log)]
        record['shutdown_warnings'] = 'class_loader.ClassLoader' in estimator_log
        if any(code not in (0, -2) for code in record['estimator_child_exit_codes']):
            record['status'] = 'failed'
            record['error'] = 'Unexpected estimator child exit; inspect estimator.log'

        record['config_hashes'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in config_directory.glob('*.yaml')}
        (run/'run.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2))
    if record['status'] == 'failed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
