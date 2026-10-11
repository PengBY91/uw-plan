import copy
import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
from rosbags.rosbag2 import Writer
from rosbags.typesys import Stores, get_typestore

from uwbench.audit import audit

TS = get_typestore(Stores.ROS2_JAZZY)
T0 = 1_700_000_000 * 10**9


def header(ns, frame):
    return TS.types['std_msgs/msg/Header'](stamp=TS.types['builtin_interfaces/msg/Time'](sec=ns // 10**9, nanosec=ns % 10**9), frame_id=frame)


def image(ns, frame, fill=7, width=4):
    return TS.types['sensor_msgs/msg/Image'](header=header(ns, frame), height=2, width=width, encoding='mono8', is_bigendian=0,
                                             step=width, data=np.full(2*width, fill, dtype=np.uint8))


def imu(ns, g=9.81):
    v3 = TS.types['geometry_msgs/msg/Vector3']
    q = TS.types['geometry_msgs/msg/Quaternion'](x=0.0, y=0.0, z=0.0, w=1.0)
    return TS.types['sensor_msgs/msg/Imu'](header=header(ns, 'imu'), orientation=q, orientation_covariance=np.zeros(9),
                                           angular_velocity=v3(x=0.01, y=0.0, z=0.0), angular_velocity_covariance=np.zeros(9),
                                           linear_acceleration=v3(x=0.0, y=0.0, z=g), linear_acceleration_covariance=np.zeros(9))


def pressure(ns):
    return TS.types['sensor_msgs/msg/FluidPressure'](header=header(ns, 'depth'), fluid_pressure=111325.0, variance=0.0)


def clean_streams(seconds=2):
    cam = [T0 + i*50_000_000 for i in range(20*seconds)]
    return {'/left': [(t, image(t, 'left')) for t in cam],
            '/right': [(t, image(t, 'right')) for t in cam],
            '/imu': [(t, imu(t)) for t in (T0 + i*5_000_000 for i in range(200*seconds))],
            '/sonar': [(t, image(t, 'sonar', width=8)) for t in (T0 + i*100_000_000 for i in range(10*seconds))],
            '/depth': [(t, pressure(t)) for t in (T0 + i*20_000_000 for i in range(50*seconds))]}


def write_case(root, streams):
    (root/'raw').mkdir(parents=True)
    with Writer(root/'raw'/'bag', version=9) as writer:
        for topic, messages in streams.items():
            conn = writer.add_connection(topic, type(messages[0][1]).__msgtype__, typestore=TS)
            for log, msg in messages:
                writer.write(conn, log, TS.serialize_cdr(msg, conn.msgtype))
    for name in ('calibration.yaml', 'reference.yaml', 'clocks.yaml'):
        (root/name).write_text('placeholder: real file in a real run\n')
    files = sorted(p for p in (root/'raw'/'bag').iterdir())
    (root/'checksums.sha256').write_text(''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  raw/bag/{p.name}\n' for p in files))
    return {'schema_version': 1, 'run_id': 'synthetic', 'data_root': str(root), 'split': 'dev', 'raw_uri': 'raw/bag',
            'calibration_uri': 'calibration.yaml', 'reference_uri': 'reference.yaml', 'clock_model_uri': 'clocks.yaml',
            'world_frame': 'world_enu', 'body_frame': 'base_link', 'checksums_uri': 'checksums.sha256',
            'streams': {'left': {'topic': '/left'}, 'right': {'topic': '/right'}, 'imu': {'topic': '/imu', 'min_rate_hz': 190},
                        'sonar': {'topic': '/sonar'}, 'depth': {'topic': '/depth'}}}


class AuditTest(unittest.TestCase):
    def run_case(self, mutate=None, edit_manifest=None):
        streams = clean_streams()
        if mutate:
            mutate(streams)
        with tempfile.TemporaryDirectory() as d:
            manifest = write_case(Path(d), streams)
            if edit_manifest:
                edit_manifest(manifest, Path(d))
            report, stats = audit(manifest)
        return {i['check'] for i in report.failed}, stats

    def test_clean_record_passes(self):
        failed, stats = self.run_case()
        self.assertEqual(failed, set())
        self.assertEqual(stats['imu']['gravity_body_axis'], '+z')
        self.assertEqual(stats['stereo_pairing']['paired'], 40)

    def test_missing_raw_file_fails(self):
        failed, _ = self.run_case(edit_manifest=lambda m, d: m.update(raw_uri='raw/absent'))
        self.assertIn('files.raw', failed)

    def test_hash_mismatch_fails(self):
        def corrupt(m, d):
            sums = d/'checksums.sha256'
            sums.write_text('0'*64 + sums.read_text()[64:])
        failed, _ = self.run_case(edit_manifest=corrupt)
        self.assertTrue(any(f.startswith('files.hash:') for f in failed))

    def test_placeholder_and_missing_stream_fail(self):
        def edit(m, d):
            m['world_frame'] = '<confirmed world frame>'
            del m['streams']['depth']
        failed, _ = self.run_case(edit_manifest=edit)
        self.assertTrue({'manifest.world_frame', 'streams.depth'} <= failed)

    def test_declared_unavailable_stream_is_reported_not_failed(self):
        def edit(m, d):
            m['streams']['depth'] = {'unavailable': 'no pressure sensor on public data'}
        failed, stats = self.run_case(edit_manifest=edit)
        self.assertEqual(failed, set())
        self.assertNotIn('depth', stats)

    def test_out_of_order_fails(self):
        def swap(s):  # receive order kept, sample stamps reversed
            (a, ma), (b, mb) = s['/imu'][10], s['/imu'][11]
            s['/imu'][10], s['/imu'][11] = (a, mb), (b, ma)
        self.assertIn('imu.monotonic', self.run_case(swap)[0])

    def test_clock_back_jump_fails(self):
        def jump(s):
            s['/sonar'][15] = (s['/sonar'][15][0], image(T0 - 5*10**9, 'sonar', width=8))
        failed, stats = self.run_case(jump)
        self.assertIn('sonar.monotonic', failed)
        self.assertGreaterEqual(stats['sonar']['clock_back_jumps'], 1)

    def test_dropped_frames_are_estimated_with_denominator(self):
        def drop(s):
            del s['/left'][5:10]
        failed, stats = self.run_case(drop)
        self.assertEqual(stats['left']['estimated_missing'], 5)
        self.assertEqual(stats['left']['estimated_expected'], 40)
        self.assertTrue({'left.drop_fraction', 'stereo.pairing'} <= failed)

    def test_stereo_mismatch_fails(self):
        def shift(s):
            s['/right'] = [(t + 3_000_000, image(t + 3_000_000, 'right')) for t, _ in s['/right']]
        self.assertIn('stereo.pairing', self.run_case(shift)[0])

    def test_accel_in_g_fails(self):
        def in_g(s):
            s['/imu'] = [(t, imu(t, g=1.0)) for t, _ in s['/imu']]
        self.assertIn('imu.accel_units_m_s2', self.run_case(in_g)[0])

    def test_empty_sonar_fails(self):
        def blank(s):
            s['/sonar'] = [(t, image(t, 'sonar', fill=0, width=8)) for t, _ in s['/sonar']]
        self.assertIn('sonar.no_empty_frames', self.run_case(blank)[0])

    def test_low_imu_rate_fails(self):
        def slow(s):
            s['/imu'] = s['/imu'][::2]
        self.assertIn('imu.rate', self.run_case(slow)[0])


if __name__ == '__main__':
    unittest.main()
