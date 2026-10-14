"""Derive a fault-injected ROS2 bag from a verified parent bag (W4-05, P11). The parent is only read.

Faults are injected one kind at a time:
  dropout  short drops: --bursts bursts of --burst-frames consecutive frames at seeded random positions
  offset   one stream's header.stamp shifted by a constant --offset-ms relative to the others
  outage   one stream loses every message in [--start-s, --start-s + --duration-s)

The camera stream means cam0 and cam1 together (stereo pairs are dropped or shifted as a pair).
Only header.stamp (the sample time the estimator uses) is ever changed; the bag log time (receive /
playback time) of kept messages is copied unchanged. fault.json records the parent hash, seed,
intervals and parameters, then the output is re-read and checked against them.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.rosbag2 import StoragePlugin, Writer
from rosbags.typesys import Stores, get_typestore

STREAMS = {'cam': ['/cam0/image_raw', '/cam1/image_raw'], 'imu': ['/imu0']}
DEFAULT_TOPICS = ['/cam0/image_raw', '/cam1/image_raw', '/imu0']
TS = get_typestore(Stores.ROS2_JAZZY)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        while chunk := f.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def stamp_ns(msg):
    return int(msg.header.stamp.sec) * 10**9 + int(msg.header.stamp.nanosec)


def read_stamps(bag, topics):
    stamps = {t: [] for t in topics}
    with AnyReader([bag]) as reader:
        selected = [c for c in reader.connections if c.topic in topics]
        for conn, _, raw in reader.messages(connections=selected):
            stamps[conn.topic].append(stamp_ns(reader.deserialize(raw, conn.msgtype)))
    return {t: np.asarray(v, dtype=np.int64) for t, v in stamps.items()}


def plan(args, stamps):
    """Return (sample stamps to drop per topic, header shift ns per topic, record of the chosen intervals)."""
    topics = STREAMS[args.stream]
    key = np.asarray(stamps[topics[0]])
    for t in topics[1:]:
        if not np.array_equal(stamps[t], key):
            raise SystemExit('Stream topics are not sample-aligned; pairwise faults are undefined: ' + t)
    t0 = int(min(v[0] for v in stamps.values() if len(v)))
    drop, shift, intervals = set(), {}, []
    if args.fault == 'dropout':
        rng = np.random.default_rng(args.seed)
        # Keep bursts away from the first/last 10 % (initialisation and tail) and from each other.
        lo, hi = int(0.1 * len(key)), int(0.9 * len(key)) - args.burst_frames
        slots = (hi - lo) // (2 * args.burst_frames)
        if args.bursts > slots:
            raise SystemExit('Too many bursts for the sequence length')
        starts = np.sort(lo + 2 * args.burst_frames * rng.choice(slots, args.bursts, replace=False))
        for s in starts:
            chunk = key[s:s + args.burst_frames]
            drop.update(int(x) for x in chunk)
            intervals.append({'first_index': int(s), 'frames': int(args.burst_frames),
                              'start_ns': int(chunk[0]), 'end_ns': int(chunk[-1]),
                              'start_s_from_bag_start': (int(chunk[0]) - t0) / 1e9})
    elif args.fault == 'outage':
        start = t0 + int(round(args.start_s * 1e9))
        end = start + int(round(args.duration_s * 1e9))
        chunk = key[(key >= start) & (key < end)]
        if not len(chunk):
            raise SystemExit('Outage window contains no messages')
        drop.update(int(x) for x in chunk)
        intervals.append({'start_ns': start, 'end_ns': end, 'start_s_from_bag_start': args.start_s,
                          'duration_s': args.duration_s, 'messages_per_topic': int(len(chunk))})
    elif args.fault == 'offset':
        if args.offset_ms == 0:
            raise SystemExit('Zero offset is not a fault')
        shift = {t: int(round(args.offset_ms * 1e6)) for t in topics}
    return {t: drop for t in topics}, shift, intervals


def write(args, drop, shift):
    counts = {}
    with AnyReader([args.src]) as reader, Writer(args.out, version=9, storage_plugin=StoragePlugin.MCAP) as writer:
        selected = [c for c in reader.connections if c.topic in args.topics]
        if sorted({c.topic for c in selected}) != sorted(args.topics):
            raise SystemExit('Parent bag lacks a requested topic')
        out = {c.topic: writer.add_connection(c.topic, c.msgtype, typestore=TS) for c in selected}
        for conn, log_ns, raw in reader.messages(connections=selected):
            topic = conn.topic
            if topic in drop or topic in shift:
                msg = reader.deserialize(raw, conn.msgtype)
                if stamp_ns(msg) in drop.get(topic, ()):
                    continue
                if topic in shift:
                    t = stamp_ns(msg) + shift[topic]
                    if t < 0:
                        raise SystemExit('Shift produced a negative timestamp')
                    msg.header.stamp = TS.types['builtin_interfaces/msg/Time'](sec=t // 10**9, nanosec=t % 10**9)
                    raw = TS.serialize_cdr(msg, conn.msgtype)
            writer.write(out[topic], log_ns, raw)
            counts[topic] = counts.get(topic, 0) + 1
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--src', type=Path, required=True, help='Parent ROS2 bag directory (read only)')
    parser.add_argument('--parent-audit', type=Path, required=True, help='Passed audit of the parent data')
    parser.add_argument('--out', type=Path, required=True, help='New bag directory; must not exist')
    parser.add_argument('--fault', choices=['dropout', 'offset', 'outage'], required=True)
    parser.add_argument('--stream', choices=sorted(STREAMS), required=True)
    parser.add_argument('--topics', nargs='+', default=DEFAULT_TOPICS, help='Topics copied to the derived bag')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--burst-frames', type=int)
    parser.add_argument('--bursts', type=int)
    parser.add_argument('--offset-ms', type=float)
    parser.add_argument('--start-s', type=float)
    parser.add_argument('--duration-s', type=float)
    args = parser.parse_args()
    need = {'dropout': ['burst_frames', 'bursts'], 'offset': ['offset_ms'], 'outage': ['start_s', 'duration_s']}[args.fault]
    if any(getattr(args, n) is None for n in need):
        parser.error('--fault %s needs %s' % (args.fault, ', '.join('--' + n.replace('_', '-') for n in need)))
    audit = json.loads(args.parent_audit.read_text())
    if not (audit.get('passed') is True or audit.get('status') == 'passed'):
        parser.error('Parent audit has not passed')
    if not set(STREAMS[args.stream]) <= set(args.topics):
        parser.error('Faulted stream topics must be copied')
    if args.out.exists():
        parser.error('Refusing to overwrite ' + str(args.out))
    parent_files = sorted(p for p in args.src.iterdir() if p.is_file())
    parent_hash = {p.name: sha256(p) for p in parent_files}
    before = read_stamps(args.src, args.topics)
    drop, shift, intervals = plan(args, before)
    written = write(args, drop, shift)
    after = read_stamps(args.out, args.topics)
    checks = {}
    for t in args.topics:
        expected = np.array([s for s in before[t] if int(s) not in drop.get(t, ())], dtype=np.int64) + shift.get(t, 0)
        checks[t] = {'parent_count': int(len(before[t])), 'written_count': int(written.get(t, 0)),
                     'dropped': int(len(before[t]) - len(after[t])),
                     'header_stamps_as_specified': bool(np.array_equal(after[t], expected)),
                     'strictly_increasing': bool(np.all(np.diff(after[t]) > 0))}
    passed = all(c['header_stamps_as_specified'] and c['strictly_increasing'] for c in checks.values())
    record = {
        'schema_version': 1, 'passed': passed, 'kind': 'derived software fault injection (not a water-quality condition)',
        'parent': {'bag': str(args.src.resolve()), 'files_sha256': parent_hash, 'audit': str(args.parent_audit.resolve())},
        'fault': args.fault, 'stream': args.stream, 'stream_topics': STREAMS[args.stream], 'seed': args.seed,
        'parameters': {n: getattr(args, n) for n in need}, 'intervals': intervals,
        'header_shift_ns': shift,
        'modified_time': ('header.stamp (sample time) of the faulted stream; bag log/receive time unchanged'
                          if args.fault == 'offset' else
                          'messages removed; header.stamp and bag log/receive time of kept messages unchanged'),
        'copied_topics': args.topics, 'checks': checks,
        'output_files_sha256': {p.name: sha256(p) for p in sorted(args.out.iterdir()) if p.is_file()},
        'command': ['scripts/data/inject_fault.py'] + sys.argv[1:],
    }
    (args.out / 'fault.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({k: record[k] for k in ['passed', 'fault', 'stream', 'parameters', 'intervals', 'checks']}, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == '__main__':
    main()
