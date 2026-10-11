"""Run an isolated, bounded ROS2 baseline and keep failure evidence."""
import argparse
import datetime
import hashlib
import json
import os
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
    args = parser.parse_args()
    if args.rate <= 0 or Path(args.run_id).name != args.run_id:
        parser.error('Invalid playback rate or run ID')
    dirty = subprocess.check_output(['git', '-C', str(args.lab), 'status', '--porcelain'], text=True)
    if dirty.strip():
        parser.error('Commit team code/config changes before starting a versioned baseline run')
    audit = json.loads((args.lab/'reports/w1/v101-data-audit/audit.json').read_text())
    if audit['status'] != 'passed':
        parser.error('Dataset audit has not passed')
    config_directory = args.lab/'configs/euroc_v101'
    run = args.lab/'runs'/args.run_id
    run.mkdir(parents=True, exist_ok=False)
    processes, handles = {}, []
    record = {'run_id': args.run_id, 'start_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'rate': args.rate, 'ros_domain_id': os.environ.get('ROS_DOMAIN_ID'),
              'rmw_implementation': os.environ.get('RMW_IMPLEMENTATION', 'default'),
              'cyclonedds_uri': os.environ.get('CYCLONEDDS_URI'),
              'clock_mode': 'header timestamps; no simulated clock',
              'team_commit': subprocess.check_output(['git', '-C', str(args.lab), 'rev-parse', 'HEAD'], text=True).strip(),
              'commands': {},
              'udp_rmem_max': Path('/proc/sys/net/core/rmem_max').read_text().strip(),
              'communication_config_sha256': hashlib.sha256((args.lab/'configs/cyclonedds-local.xml').read_bytes()).hexdigest() if os.environ.get('RMW_IMPLEMENTATION') == 'rmw_cyclonedds_cpp' else None}

    def start(name, command):
        handle = (run/(name+'.log')).open('w')
        handles.append(handle)
        record['commands'][name] = command
        processes[name] = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                                           start_new_session=True)
        return processes[name]

    def stop(name):
        p = processes.get(name)
        if p is None or p.poll() is not None:
            return
        os.killpg(p.pid, signal.SIGINT)
        try:
            p.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGTERM)
            p.wait(timeout=10)

    try:
        start('estimator', ['ros2', 'launch', 'ov_msckf', 'subscribe.launch.py',
              'config_path:='+str(config_directory/'estimator_config.yaml'),
              'use_stereo:=true', 'max_cameras:=2', 'rviz_enable:=false'])
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
                       '--topics', '/cam0/image_raw', '/cam1/image_raw', '/imu0'])
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
        record['config_hashes'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in config_directory.glob('*.yaml')}
        (run/'run.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2))
    if record['status'] == 'failed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
