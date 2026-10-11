"""Verify bag conversion and cross-check EuRoC ASL sensor data."""
import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
from PIL import Image
from rosbags.highlevel import AnyReader


def normalize(value):
    if dataclasses.is_dataclass(value):
        return {f.name: normalize(getattr(value, f.name))
                for f in dataclasses.fields(value) if f.name not in ('seq', '__msgtype__')}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [normalize(v) for v in value]
    return value


def stamp_ns(msg):
    return int(msg.header.stamp.sec) * 10**9 + int(msg.header.stamp.nanosec)


def scan(path, asl=None, image_output=None):
    stats, stamps, imu_rows, hashes, samples = {}, {}, [], {}, {}
    with AnyReader([path]) as reader:
        counts = {}
        for conn in reader.connections:
            counts[conn.topic] = counts.get(conn.topic, 0) + conn.msgcount
        for conn, receive_ns, raw in reader.messages():
            msg = reader.deserialize(raw, conn.msgtype)
            topic = conn.topic
            sample_ns = stamp_ns(msg) if hasattr(msg, 'header') else None
            if topic not in stats:
                stats[topic] = dict(msgtype=conn.msgtype, count=0, first_receive_ns=int(receive_ns),
                                    last_receive_ns=0, first_sample_ns=sample_ns, last_sample_ns=None,
                                    sample_non_increasing=0)
                hashes[topic] = hashlib.sha256()
                stamps[topic] = []
            stat = stats[topic]
            index = stat['count']
            if sample_ns is not None:
                if stat['last_sample_ns'] is not None and sample_ns <= stat['last_sample_ns']:
                    stat['sample_non_increasing'] += 1
                stamps[topic].append(sample_ns)
            stat['last_sample_ns'] = sample_ns
            stat['last_receive_ns'] = int(receive_ns)
            stat['count'] += 1
            hashes[topic].update(struct.pack('<q', receive_ns))
            if conn.msgtype == 'sensor_msgs/msg/Image':
                metadata = [sample_ns, msg.header.frame_id, msg.height, msg.width,
                            msg.encoding, msg.is_bigendian, msg.step]
                hashes[topic].update(json.dumps(metadata).encode())
                hashes[topic].update(msg.data.tobytes())
                assert msg.encoding == 'mono8' and (msg.width, msg.height) == (752, 480)
                assert len(msg.data) == msg.step * msg.height
                if index in {0, counts[topic] // 2, counts[topic] - 1}:
                    pixels = np.asarray(msg.data).reshape(msg.height, msg.step)[:, :msg.width]
                    camera = topic.split('/')[1]
                    sample = dict(topic=topic, index=index, stamp_ns=sample_ns,
                                  dimensions=[int(msg.width), int(msg.height)],
                                  pixel_sha256=hashlib.sha256(pixels.tobytes()).hexdigest())
                    if asl:
                        csv_rows = [line.split(',') for line in (asl/camera/'data.csv').read_text().splitlines() if line and not line.startswith('#')]
                        csv_stamp, filename = csv_rows[index]
                        sample['asl_stamp_delta_ns'] = int(csv_stamp) - sample_ns
                        assert abs(sample['asl_stamp_delta_ns']) <= 128
                        image_path = asl / camera / 'data' / filename
                        with Image.open(image_path) as im:
                            sample['asl_pixels_equal'] = bool(np.array_equal(pixels, np.asarray(im)))
                        assert sample['asl_pixels_equal']
                    if image_output:
                        Image.fromarray(pixels).save(image_output / f'{camera}_{index:04d}.png')
                    samples.setdefault(topic, []).append(sample)
            else:
                hashes[topic].update(json.dumps(normalize(msg), sort_keys=True, allow_nan=False).encode())
                if conn.msgtype == 'sensor_msgs/msg/Imu':
                    w, a = msg.angular_velocity, msg.linear_acceleration
                    row = [w.x, w.y, w.z, a.x, a.y, a.z]
                    assert np.isfinite(row).all()
                    if topic == '/imu0':
                        imu_rows.append(row)
        for topic, stat in stats.items():
            assert stat['count'] == counts[topic] > 0
            stat['content_sha256'] = hashes[topic].hexdigest()
            ts = np.array(stamps[topic], dtype=np.int64)
            if len(ts) > 1:
                intervals = np.diff(ts) / 1e9
                stat['sample_interval_s'] = {'min': float(intervals.min()),
                    'median': float(np.median(intervals)), 'max': float(intervals.max())}
    required = ['/cam0/image_raw', '/cam1/image_raw', '/imu0']
    assert all(t in stats for t in required)
    assert all(stats[t]['sample_non_increasing'] == 0 for t in required)
    assert stamps['/cam0/image_raw'] == stamps['/cam1/image_raw']
    asl_checks = {}
    if asl:
        for camera in ['cam0', 'cam1']:
            lines = [line.split(',')[0] for line in (asl/camera/'data.csv').read_text().splitlines()
                     if line and not line.startswith('#')]
            csv_stamps = list(map(int, lines))
            assert len(stamps[f'/{camera}/image_raw']) == len(csv_stamps)
            delta = np.array(csv_stamps, dtype=np.int64) - np.array(stamps[f'/{camera}/image_raw'], dtype=np.int64)
            max_delta = int(np.max(np.abs(delta)))
            assert max_delta <= 128
            png_count = len(list((asl/camera/'data').glob('*.png')))
            assert png_count == len(csv_stamps)
            asl_checks[camera] = {'timestamp_max_abs_delta_ns': max_delta, 'timestamp_tolerance_ns': 128, 'png_count': png_count}
        imu_lines = [line.split(',') for line in (asl/'imu0/data.csv').read_text().splitlines()
                     if line and not line.startswith('#')]
        imu_csv_stamps = np.array([int(row[0]) for row in imu_lines], dtype=np.int64)
        assert len(stamps['/imu0']) == len(imu_csv_stamps)
        imu_delta = int(np.max(np.abs(imu_csv_stamps - np.array(stamps['/imu0'], dtype=np.int64))))
        assert imu_delta <= 128
        values = np.array([[float(v) for v in row[1:7]] for row in imu_lines])
        error = float(np.max(np.abs(values - np.array(imu_rows))))
        assert error <= 1e-6
        asl_checks['imu0'] = {'timestamp_max_abs_delta_ns': imu_delta, 'timestamp_tolerance_ns': 128, 'max_abs_value_difference': error,
                              'value_tolerance': 1e-6, 'finite_values': True}
    return {'topics': stats, 'samples': samples, 'asl_checks': asl_checks,
            'camera_stamps_ns': stamps['/cam0/image_raw']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bag', type=Path, required=True)
    parser.add_argument('--ros2', type=Path, required=True)
    parser.add_argument('--asl', type=Path, required=True, help='mav0 directory')
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    args.report.mkdir(parents=True, exist_ok=False)
    source = scan(args.bag, args.asl, args.report)
    converted = scan(args.ros2)
    assert set(source['topics']) == set(converted['topics'])
    for topic in source['topics']:
        assert source['topics'][topic] == converted['topics'][topic], topic
    result = {'status': 'passed', 'comparison': 'all messages; Header.seq excluded across ROS1/ROS2',
              'source': source, 'converted': converted}
    (args.report/'audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': 'passed', 'topics': source['topics'],
                      'asl_checks': source['asl_checks']}, indent=2))


if __name__ == '__main__':
    main()
