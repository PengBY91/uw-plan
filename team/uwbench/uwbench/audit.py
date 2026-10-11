"""uwbench audit (P03): manifest -> completeness, timing and unit report; any failed check exits non-zero.

Relative URIs resolve against manifest.data_root. A manifest entry may be a mapping {unavailable: <reason>},
which is reported, never silently passed. Raw data is only read.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml
from rosbags.highlevel import AnyReader

REQUIRED_KEYS = ('schema_version', 'run_id', 'data_root', 'split', 'raw_uri', 'calibration_uri', 'reference_uri',
                 'clock_model_uri', 'world_frame', 'body_frame', 'streams', 'checksums_uri')
STREAM_KINDS = ('left', 'right', 'imu', 'sonar', 'depth')
MAX_DROP_FRACTION = 0.01  # eight-week plan G1: effective record rate at least 99%
IMAGE_TYPES = ('sensor_msgs/msg/Image', 'sensor_msgs/msg/CompressedImage')


class Report:
    def __init__(self):
        self.items = []

    def add(self, check, ok, detail=None):
        status = {True: 'pass', False: 'fail', None: 'unavailable'}[ok]
        self.items.append({'check': check, 'status': status, 'detail': detail})

    @property
    def failed(self):
        return [i for i in self.items if i['status'] == 'fail']


def is_placeholder(value):
    return isinstance(value, str) and '<' in value and '>' in value


def unavailable(value):
    return isinstance(value, dict) and set(value) == {'unavailable'} and bool(value['unavailable'])


def check_manifest(manifest, report):
    for key in REQUIRED_KEYS:
        value = manifest.get(key)
        if unavailable(value):
            report.add(f'manifest.{key}', None, value['unavailable'])
        else:
            report.add(f'manifest.{key}', value not in (None, '') and not is_placeholder(value),
                       None if value not in (None, '') else 'missing')
    for kind in STREAM_KINDS:
        stream = (manifest.get('streams') or {}).get(kind)
        if unavailable(stream):
            report.add(f'streams.{kind}', None, stream['unavailable'])
        elif not isinstance(stream, dict) or not stream.get('topic') or is_placeholder(stream.get('topic')):
            report.add(f'streams.{kind}', False, 'topic missing; declare {unavailable: <reason>} if the device has none')
        else:
            report.add(f'streams.{kind}', True)


def resolve(manifest, uri):
    path = Path(uri)
    return path if path.is_absolute() else Path(manifest['data_root'])/path


def check_files(manifest, report, verify_hashes):
    raw = resolve(manifest, manifest['raw_uri'])
    report.add('files.raw', raw.exists(), str(raw))
    for key in ('calibration_uri', 'reference_uri', 'clock_model_uri'):
        if isinstance(manifest.get(key), str) and not is_placeholder(manifest[key]):
            path = resolve(manifest, manifest[key])
            report.add(f'files.{key}', path.exists(), str(path))
    sums = resolve(manifest, manifest['checksums_uri']) if isinstance(manifest.get('checksums_uri'), str) else None
    if sums is None or not sums.exists():
        report.add('files.checksums', False, f'missing {sums}')
        return
    entries = [line.split(None, 1) for line in sums.read_text().splitlines() if line.strip()]
    report.add('files.checksums_nonempty', bool(entries), str(sums))
    for digest, name in entries:
        path = sums.parent/name.strip().lstrip('*')
        if not path.exists():
            report.add(f'files.hash:{name.strip()}', False, 'listed file missing')
        elif verify_hashes:
            h = hashlib.sha256()
            with path.open('rb') as f:
                for block in iter(lambda: f.read(1 << 20), b''):
                    h.update(block)
            report.add(f'files.hash:{name.strip()}', h.hexdigest() == digest.lower())


def read_streams(raw, topics):
    data = {t: {'stamps': [], 'log': [], 'msgtype': None, 'shapes': set(), 'empty': 0, 'accel': [], 'gyro': [], 'values': []}
            for t in topics}
    with AnyReader([raw]) as reader:
        present = {c.topic for c in reader.connections}
        for conn, log_time, raw_msg in reader.messages(connections=[c for c in reader.connections if c.topic in data]):
            msg = reader.deserialize(raw_msg, conn.msgtype)
            d = data[conn.topic]
            d['msgtype'] = conn.msgtype
            d['stamps'].append(int(msg.header.stamp.sec)*10**9 + int(msg.header.stamp.nanosec))
            d['log'].append(int(log_time))
            if conn.msgtype == 'sensor_msgs/msg/Image':
                d['shapes'].add((int(msg.height), int(msg.width), msg.encoding))
                d['empty'] += int(len(msg.data) == 0 or not np.any(msg.data))
            elif conn.msgtype == 'sensor_msgs/msg/CompressedImage':
                d['shapes'].add(('compressed', msg.format))
                d['empty'] += int(len(msg.data) == 0)
            elif conn.msgtype == 'sensor_msgs/msg/Imu':
                a, w = msg.linear_acceleration, msg.angular_velocity
                d['accel'].append((a.x, a.y, a.z))
                d['gyro'].append((w.x, w.y, w.z))
            elif conn.msgtype == 'sensor_msgs/msg/FluidPressure':
                d['values'].append(float(msg.fluid_pressure))
    return data, present


def timing(stamps_ns, log_ns):
    t = np.asarray(stamps_ns, dtype=np.int64)
    dt = np.diff(t)/1e9
    positive = dt[dt > 0]
    median = float(np.median(positive)) if positive.size else None
    big = dt > 1.5*median if median else np.zeros(len(dt), bool)
    missing = int(np.sum(np.rint(dt[big]/median) - 1)) if median else 0
    latency = (np.asarray(log_ns, dtype=np.int64) - t)/1e9
    n = len(t)
    return {'count': n, 'rate_hz': 1/median if median else None, 'median_interval_s': median,
            'max_interval_s': float(dt.max()) if dt.size else None,
            'duplicates': int(np.sum(dt == 0)), 'out_of_order': int(np.sum((dt < 0) & (dt >= -1))),
            'clock_back_jumps': int(np.sum(dt < -1)),
            'estimated_missing': missing, 'estimated_expected': n + missing,
            'estimated_drop_fraction': missing/(n + missing) if n else None,
            'receive_minus_sample_s': {'min': float(latency.min()), 'median': float(np.median(latency)),
                                       'p95': float(np.percentile(latency, 95))} if n else None,
            'first_middle_last_index': [0, n//2, n-1] if n else [],
            'first_middle_last_stamp_ns': [int(t[0]), int(t[n//2]), int(t[-1])] if n else []}


def check_stream(kind, spec, d, report):
    stats = timing(d['stamps'], d['log'])
    stats['msgtype'] = d['msgtype']
    report.add(f'{kind}.nonempty', stats['count'] > 1, f"{stats['count']} messages")
    if stats['count'] < 2:
        return stats
    report.add(f'{kind}.monotonic', stats['duplicates'] == 0 and stats['out_of_order'] == 0 and stats['clock_back_jumps'] == 0,
               {k: stats[k] for k in ('duplicates', 'out_of_order', 'clock_back_jumps')})
    report.add(f'{kind}.drop_fraction', stats['estimated_drop_fraction'] <= MAX_DROP_FRACTION,
               f"estimated {stats['estimated_missing']}/{stats['estimated_expected']} from header gaps; no sequence numbers")
    if spec.get('min_rate_hz'):
        report.add(f'{kind}.rate', stats['rate_hz'] >= spec['min_rate_hz'], f"{stats['rate_hz']:.2f} Hz")
    if kind in ('left', 'right', 'sonar'):
        report.add(f'{kind}.msgtype', d['msgtype'] in IMAGE_TYPES, d['msgtype'])
        report.add(f'{kind}.constant_shape', len(d['shapes']) == 1, sorted(map(str, d['shapes'])))
        stats['empty_frames'] = d['empty']
        report.add(f'{kind}.no_empty_frames', d['empty'] == 0, f"{d['empty']} empty")
    elif kind == 'imu':
        report.add('imu.msgtype', d['msgtype'] == 'sensor_msgs/msg/Imu', d['msgtype'])
        if d['accel']:
            accel, gyro = np.asarray(d['accel']), np.asarray(d['gyro'])
            norm = float(np.median(np.linalg.norm(accel, axis=1)))
            gyro_p99 = float(np.percentile(np.linalg.norm(gyro, axis=1), 99))
            first = accel[np.asarray(d['stamps']) <= d['stamps'][0] + 10**9].mean(axis=0)
            stats.update(accel_norm_median=norm, gyro_norm_p99=gyro_p99,
                         mean_accel_first_second=first.tolist(),
                         gravity_body_axis=f"{'+-'[int(first[np.argmax(np.abs(first))] < 0)]}{'xyz'[np.argmax(np.abs(first))]}")
            report.add('imu.accel_units_m_s2', 8.8 <= norm <= 10.8, f'median |a| {norm:.3f}; ~1 means g, not m/s^2')
            report.add('imu.gyro_units_rad_s', gyro_p99 < 20, f'p99 |w| {gyro_p99:.3f}; large values suggest deg/s')
    elif kind == 'depth':
        if d['msgtype'] == 'sensor_msgs/msg/FluidPressure':
            values = np.asarray(d['values'])
            stats['pressure_pa_range'] = [float(values.min()), float(values.max())]
            report.add('depth.pressure_units_pa', 8e4 <= values.min() and values.max() <= 2e6,
                       'absolute pressure in Pa expected for 0-100 m water')
        else:
            report.add('depth.msgtype', False, f'{d["msgtype"]}: no format-layer check implemented')
    return stats


def check_pairing(left, right, tolerance_s, report):
    a, b = np.asarray(left, dtype=np.int64), np.asarray(right, dtype=np.int64)
    if not a.size or not b.size:
        return None
    k = np.clip(np.searchsorted(b, a), 1, max(len(b)-1, 1))
    nearest = np.minimum(np.abs(b[k-1]-a), np.abs(b[np.minimum(k, len(b)-1)]-a))/1e9
    paired = int(np.sum(nearest <= tolerance_s))
    result = {'left': len(a), 'right': len(b), 'paired': paired, 'tolerance_s': tolerance_s,
              'max_offset_s_of_paired': float(nearest[nearest <= tolerance_s].max()) if paired else None}
    report.add('stereo.pairing', paired >= (1-MAX_DROP_FRACTION)*max(len(a), len(b)), result)
    return result


def audit(manifest, verify_hashes=True):
    report = Report()
    check_manifest(manifest, report)
    if any(i['status'] == 'fail' and i['check'] in ('manifest.data_root', 'manifest.raw_uri') for i in report.items):
        return report, {}
    check_files(manifest, report, verify_hashes)
    raw = resolve(manifest, manifest['raw_uri'])
    streams = {k: v for k, v in (manifest.get('streams') or {}).items()
               if k in STREAM_KINDS and isinstance(v, dict) and v.get('topic') and not unavailable(v)}
    if not raw.exists() or not streams:
        return report, {}
    data, present = read_streams(raw, {s['topic'] for s in streams.values()})
    stats = {}
    for kind, spec in streams.items():
        if spec['topic'] not in present:
            report.add(f'{kind}.topic_present', False, spec['topic'])
            continue
        stats[kind] = check_stream(kind, spec, data[spec['topic']], report)
    if 'left' in stats and 'right' in stats:
        stats['stereo_pairing'] = check_pairing(data[streams['left']['topic']]['stamps'], data[streams['right']['topic']]['stamps'],
                                                float(streams['right'].get('pairing_tolerance_s', 0.0)), report)
    return report, stats


def main(argv):
    parser = argparse.ArgumentParser(prog='uwbench audit')
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--skip-hashes', action='store_true', help='existence only; the report records that hashes were skipped')
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = yaml.safe_load(args.manifest.read_text()) or {}
    report, stats = audit(manifest, verify_hashes=not args.skip_hashes)
    result = {'manifest': str(args.manifest.resolve()),
              'manifest_sha256': hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
              'hashes_verified': not args.skip_hashes, 'passed': not report.failed,
              'failed': [i['check'] for i in report.failed], 'checks': report.items, 'streams': stats,
              'interpretation': 'Drop estimates come from header gaps without sequence numbers; time-model offsets are W2-02.'}
    (args.out/'audit.json').write_text(json.dumps(result, indent=2, default=str)+'\n')
    with (args.out/'checks.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['check', 'status', 'detail'])
        for item in report.items:
            writer.writerow([item['check'], item['status'], json.dumps(item['detail'], default=str)])
    print(json.dumps({'passed': result['passed'], 'failed': result['failed']}, indent=2))
    return 0 if result['passed'] else 1
