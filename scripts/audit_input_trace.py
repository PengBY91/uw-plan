"""Compare actual OpenVINS callback/processing input against the verified ROS2 bag."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from rosbags.highlevel import AnyReader


def stream_summary(expected, received):
    expected = np.asarray(expected, dtype=np.int64)
    received = np.asarray(received, dtype=np.int64)
    missing = np.setdiff1d(expected, received)
    unexpected = np.setdiff1d(received, expected)
    return dict(expected_count=len(expected), received_count=len(received),
                missing_count=len(missing), unexpected_count=len(unexpected),
                duplicate_count=len(received)-len(np.unique(received)),
                non_increasing_count=int(np.sum(np.diff(received) <= 0)),
                max_sample_gap_s=float(np.diff(received).max()/1e9) if len(received)>1 else None,
                received_fraction=float(len(np.intersect1d(expected, received))/len(expected)),
                ordered_complete=bool(np.array_equal(expected, received)),
                missing_timestamps_ns=missing.tolist(), unexpected_timestamps_ns=unexpected.tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bag', required=True, type=Path)
    parser.add_argument('--run', required=True, type=Path)
    args = parser.parse_args()
    expected = {'/imu0': [], '/cam0/image_raw': [], '/cam1/image_raw': []}
    with AnyReader([args.bag]) as reader:
        selected = [c for c in reader.connections if c.topic in expected]
        for conn, _, raw in reader.messages(connections=selected):
            msg = reader.deserialize(raw, conn.msgtype)
            expected[conn.topic].append(int(msg.header.stamp.sec)*10**9 + int(msg.header.stamp.nanosec))
    with (args.run/'input-trace.csv').open() as f:
        rows = list(csv.DictReader(f))
    streams = {kind: [r for r in rows if r['event']==kind] for kind in ['imu', 'stereo', 'update']}
    assert all(streams.values()), 'Missing instrumentation events'
    imu = stream_summary(expected['/imu0'], [int(r['stamp_ns']) for r in streams['imu']])
    cam0 = stream_summary(expected['/cam0/image_raw'], [int(r['stamp_ns']) for r in streams['stereo']])
    cam1 = stream_summary(expected['/cam1/image_raw'], [int(r['secondary_ns']) for r in streams['stereo']])
    received = np.array([int(r['stamp_ns']) for r in streams['stereo']], dtype=np.int64)
    wall = np.array([int(r['steady_ns']) for r in streams['stereo']], dtype=np.int64)
    updates = np.array([int(r['stamp_ns']) for r in streams['update']], dtype=np.int64)
    # Core stores seconds as double; round trip near EuRoC epoch can differ by up to one 256 ns ULP.
    matched, waits, bad = [], [], []
    for row in streams['update']:
        t = int(row['stamp_ns']); k = int(np.searchsorted(received, t))
        choices = [j for j in (k-1,k) if 0 <= j < len(received)]
        j = min(choices, key=lambda j: abs(int(received[j])-t))
        if abs(int(received[j])-t) > 256:
            bad.append(t); continue
        matched.append(j); waits.append((int(row['steady_ns'])-int(wall[j]))/1e9)
    assert waits and min(waits) >= 0
    result = dict(imu=imu, cam0=cam0, cam1=cam1,
        stereo_mismatched_pairs=sum(int(r['stamp_ns']) != int(r['secondary_ns']) for r in streams['stereo']),
        updates=dict(count=len(updates), unique_input_pairs_processed=len(set(matched)),
                     non_increasing_count=int(np.sum(np.diff(updates)<=0)), unmatched_stamps=bad,
                     callback_to_update_wait_s=dict(median=float(np.median(waits)),p95=float(np.percentile(waits,95)),max=float(max(waits)))),
        interpretation='Estimator callback receipt and camera update entry, not a separate observer; queue wait excludes update compute time.')
    result['complete_input'] = all(x['ordered_complete'] for x in [imu,cam0,cam1]) and not bad and len(set(matched))==len(received) and not result['stereo_mismatched_pairs'] and len(updates)==len(received) and result['updates']['non_increasing_count']==0
    path = args.run/'input-audit.json'
    if path.exists(): raise FileExistsError(path)
    path.write_text(json.dumps(result,indent=2)+'\n')
    for x in [imu,cam0,cam1]:
        x.pop('missing_timestamps_ns');x.pop('unexpected_timestamps_ns')
    print(json.dumps(result,indent=2))

if __name__ == '__main__':
    main()
