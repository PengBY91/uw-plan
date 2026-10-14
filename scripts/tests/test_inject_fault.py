import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from rosbags.rosbag2 import StoragePlugin, Writer
from inject_fault import TS, read_stamps

SCRIPT = Path(__file__).resolve().parents[1]/'data'/'inject_fault.py'


def make_bag(path, seconds=10):
    Image, Imu = TS.types['sensor_msgs/msg/Image'], TS.types['sensor_msgs/msg/Imu']
    Header, Time = TS.types['std_msgs/msg/Header'], TS.types['builtin_interfaces/msg/Time']
    import numpy as np
    with Writer(path, version=9, storage_plugin=StoragePlugin.MCAP) as w:
        conns = {t: w.add_connection(t, m, typestore=TS) for t, m in
                 [('/cam0/image_raw', 'sensor_msgs/msg/Image'), ('/cam1/image_raw', 'sensor_msgs/msg/Image'),
                  ('/imu0', 'sensor_msgs/msg/Imu')]}
        V3, Q = TS.types['geometry_msgs/msg/Vector3'], TS.types['geometry_msgs/msg/Quaternion']
        for k in range(seconds * 200):
            t = 10**9 + k * 5_000_000
            h = Header(stamp=Time(sec=t // 10**9, nanosec=t % 10**9), frame_id='imu')
            imu = Imu(header=h, orientation=Q(0, 0, 0, 1), orientation_covariance=np.zeros(9),
                      angular_velocity=V3(0, 0, 0), angular_velocity_covariance=np.zeros(9),
                      linear_acceleration=V3(0, 0, 9.8), linear_acceleration_covariance=np.zeros(9))
            w.write(conns['/imu0'], t + 1000, TS.serialize_cdr(imu, 'sensor_msgs/msg/Imu'))
            if k % 10 == 0:
                img = Image(header=h, height=1, width=2, encoding='mono8', is_bigendian=0, step=2,
                            data=np.zeros(2, np.uint8))
                for cam in ['/cam0/image_raw', '/cam1/image_raw']:
                    w.write(conns[cam], t + 2000, TS.serialize_cdr(img, 'sensor_msgs/msg/Image'))


class InjectFaultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        make_bag(cls.root / 'parent')
        (cls.root / 'audit.json').write_text(json.dumps({'passed': True}))
        cls.parent = read_stamps(cls.root / 'parent', ['/cam0/image_raw', '/cam1/image_raw', '/imu0'])

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_fault(self, name, *extra):
        out = self.root / name
        p = subprocess.run([sys.executable, str(SCRIPT), '--src', str(self.root / 'parent'),
                            '--parent-audit', str(self.root / 'audit.json'), '--out', str(out), *extra],
                           capture_output=True, text=True)
        return p, out

    def test_camera_dropout_drops_stereo_pairs(self):
        p, out = self.run_fault('drop', '--fault', 'dropout', '--stream', 'cam', '--burst-frames', '3', '--bursts', '2', '--seed', '7')
        self.assertEqual(p.returncode, 0, p.stderr)
        after = read_stamps(out, ['/cam0/image_raw', '/cam1/image_raw', '/imu0'])
        self.assertEqual(len(self.parent['/cam0/image_raw']) - len(after['/cam0/image_raw']), 6)
        self.assertTrue((after['/cam0/image_raw'] == after['/cam1/image_raw']).all())
        self.assertTrue((after['/imu0'] == self.parent['/imu0']).all())
        record = json.loads((out / 'fault.json').read_text())
        self.assertTrue(record['passed'])
        self.assertEqual(record['seed'], 7)
        self.assertEqual(len(record['intervals']), 2)

    def test_seed_reproducible(self):
        a = json.loads((self.run_fault('s1', '--fault', 'dropout', '--stream', 'cam', '--burst-frames', '2', '--bursts', '2', '--seed', '3')[1] / 'fault.json').read_text())
        b = json.loads((self.run_fault('s2', '--fault', 'dropout', '--stream', 'cam', '--burst-frames', '2', '--bursts', '2', '--seed', '3')[1] / 'fault.json').read_text())
        self.assertEqual(a['intervals'], b['intervals'])

    def test_imu_offset_shifts_only_header(self):
        p, out = self.run_fault('off', '--fault', 'offset', '--stream', 'imu', '--offset-ms', '-10')
        self.assertEqual(p.returncode, 0, p.stderr)
        after = read_stamps(out, ['/cam0/image_raw', '/imu0'])
        self.assertTrue((after['/imu0'] == self.parent['/imu0'] - 10_000_000).all())
        self.assertTrue((after['/cam0/image_raw'] == self.parent['/cam0/image_raw']).all())
        from rosbags.highlevel import AnyReader
        with AnyReader([out]) as r, AnyReader([self.root / 'parent']) as q:
            log_out = [t for c, t, _ in r.messages() if c.topic == '/imu0']
            log_in = [t for c, t, _ in q.messages() if c.topic == '/imu0']
        self.assertEqual(log_out, log_in)

    def test_outage_window(self):
        p, out = self.run_fault('out', '--fault', 'outage', '--stream', 'cam', '--start-s', '4', '--duration-s', '1.5')
        self.assertEqual(p.returncode, 0, p.stderr)
        after = read_stamps(out, ['/cam0/image_raw'])['/cam0/image_raw']
        gap = (after[1:] - after[:-1]).max() / 1e9
        self.assertAlmostEqual(gap, 1.55, places=6)  # 15 frames at 20 Hz gone -> 16 intervals

    def test_refuses_overwrite_and_unpassed_parent(self):
        self.run_fault('dup', '--fault', 'offset', '--stream', 'cam', '--offset-ms', '5')
        p, _ = self.run_fault('dup', '--fault', 'offset', '--stream', 'cam', '--offset-ms', '5')
        self.assertNotEqual(p.returncode, 0)
        (self.root / 'bad.json').write_text(json.dumps({'passed': False}))
        p = subprocess.run([sys.executable, str(SCRIPT), '--src', str(self.root / 'parent'), '--parent-audit',
                            str(self.root / 'bad.json'), '--out', str(self.root / 'x'), '--fault', 'offset',
                            '--stream', 'cam', '--offset-ms', '5'], capture_output=True, text=True)
        self.assertNotEqual(p.returncode, 0)
        self.assertFalse((self.root / 'x').exists())


if __name__ == '__main__':
    unittest.main()
