"""M3.4 input-integrity monitor (uwbench.a.integrity): tolerances, offset estimate, causality (check-method
a_input_integrity, month-2 runbook R5)."""
import unittest

import numpy as np

from uwbench.a.integrity import InputMonitor, estimate_offset, fault_filter, gyro_integral

IMU_HZ, CAM_HZ = 200.0, 20.0


def motion(duration=60.0, amp=0.2):
    t = np.arange(0, duration, 1/IMU_HZ)
    w = np.stack([amp*np.sin(2*np.pi*f*t + p) for f, p in ((0.3, 0.0), (0.17, 1.0), (0.41, 2.0))], 1)
    return t, w


def streams(duration=60.0, offset_s=0.0, amp=0.2, imu_gap=None, cam_gap=None, seed=0):
    """Time-ordered events ('imu'|'cam', stamp, payload). IMU stamps = true + offset_s; the camera payload is the true
    rotation since the previous kept frame. *_gap = (start, length) removes that stream's samples."""
    rng = np.random.default_rng(seed)
    t, w = motion(duration, amp)
    tc, C = gyro_integral(t, w)
    ev = []
    for ti, wi in zip(t, w + 0.01 + rng.normal(0, 0.002, w.shape)):
        if imu_gap and imu_gap[0] <= ti < imu_gap[0] + imu_gap[1]:
            continue
        ev.append((ti, 'imu', ti + offset_s, wi))
    prev = None
    for tk in np.arange(0.5, duration - 0.5, 1/CAM_HZ):
        if cam_gap and cam_gap[0] <= tk < cam_gap[0] + cam_gap[1]:
            continue
        th = None if prev is None else np.array([np.interp(tk, tc, C[:, a]) - np.interp(prev, tc, C[:, a])
                                                 for a in range(3)]) + rng.normal(0, 2e-4, 3)
        ev.append((tk, 'cam', tk, None if th is None else (prev, th)))
        prev = tk
    ev.sort(key=lambda e: (e[0], e[1] == 'cam'))   # arrival order: by true time
    return ev


def feed(events, cfg=None, until=None):
    mon = InputMonitor(cfg)
    for arrival, kind, stamp, payload in events:
        if until is not None and arrival > until:
            break
        (mon.imu if kind == 'imu' else mon.stereo)(stamp, payload)
    return mon


def first(mon, to):
    return next((t for t in mon.transitions if t['to'] == to), None)


class OffsetEstimate(unittest.TestCase):
    def test_recovers_injected_offset(self):
        t, w = motion(30)
        tc, C = gyro_integral(t, w)
        cams = np.arange(5, 25, 1/CAM_HZ)
        iv = np.stack([cams[:-1], cams[1:]], 1)
        th = np.stack([np.interp(iv[:, 1], tc, C[:, a]) - np.interp(iv[:, 0], tc, C[:, a]) for a in range(3)], 1)
        for d in (0.0, 0.015, 0.040, -0.020):
            e = estimate_offset(iv, th, t + d, w + 0.01)
            self.assertTrue(e['observable'], e['reason'])
            self.assertAlmostEqual(e['offset_ms'], d*1000, delta=1.5)

    def test_low_rotation_is_unobservable_not_zero(self):
        t, w = motion(30, amp=0.002)
        tc, C = gyro_integral(t, w)
        cams = np.arange(5, 25, 1/CAM_HZ)
        iv = np.stack([cams[:-1], cams[1:]], 1)
        th = np.stack([np.interp(iv[:, 1], tc, C[:, a]) - np.interp(iv[:, 0], tc, C[:, a]) for a in range(3)], 1)
        e = estimate_offset(iv, th, t + 0.04, w)
        self.assertFalse(e['observable'])
        self.assertEqual(e['reason'], 'low_rotation')


    def test_visual_rotation_that_disagrees_is_unobservable(self):
        """PnP noise comparable to the rotation (stripe-beam sim) must not yield a confident offset."""
        t, w = motion(30)
        cams = np.arange(5, 25, 0.2)
        iv = np.stack([cams[:-1], cams[1:]], 1)
        th = np.random.default_rng(1).normal(0, 0.04, (len(iv), 3))
        e = estimate_offset(iv, th, t, w)
        self.assertFalse(e['observable'])
        self.assertEqual(e['reason'], 'visual_rotation_disagrees_with_gyro')


class Monitor(unittest.TestCase):
    def test_clean_streams_stay_normal(self):
        mon = feed(streams())
        self.assertEqual(mon.transitions, [])
        self.assertTrue(all(e['observable'] for e in mon.offset_estimates))

    def test_imu_outage_tolerance(self):
        mon = feed(streams(imu_gap=(30.0, 0.6)))
        lost = first(mon, 'LOST')
        self.assertEqual(lost['reason'], 'imu_outage')
        self.assertAlmostEqual(lost['t_s'], 30.0 + 0.5 - 1/IMU_HZ, delta=1/CAM_HZ + 1e-6)  # watchdog on the camera clock
        self.assertIsNone(first(feed(streams(imu_gap=(30.0, 0.4))), 'LOST'))

    def test_cam_outage_tolerance(self):
        lost = first(feed(streams(cam_gap=(30.0, 3.2))), 'LOST')
        self.assertEqual(lost['reason'], 'cam_outage')
        self.assertIsNone(first(feed(streams(cam_gap=(30.0, 2.5))), 'LOST'))

    def test_outage_that_never_ends_is_caught(self):
        ev = [e for e in streams() if not (e[1] == 'imu' and e[0] >= 30.0)]
        self.assertEqual(first(feed(ev), 'LOST')['reason'], 'imu_outage')

    def test_time_regression(self):
        ev = streams(duration=20)
        i = next(k for k, e in enumerate(ev) if e[1] == 'imu' and e[0] > 10)
        a, kind, stamp, g = ev[i]
        ev.insert(i + 1, (a, kind, stamp - 0.5, g))
        self.assertEqual(first(feed(ev), 'LOST')['reason'], 'imu_time_regression')

    def test_offset_thresholds(self):
        for off, warn, lost in ((0.010, False, False), (0.025, True, False), (0.045, True, True)):
            mon = feed(streams(offset_s=off))
            self.assertEqual(first(mon, 'WARN') is not None or (lost and first(mon, 'LOST') is not None), warn, off)
            self.assertEqual(first(mon, 'LOST') is not None, lost, off)
            if lost:
                self.assertEqual(first(mon, 'LOST')['reason'], 'time_offset')

    def test_unobservable_windows_raise_nothing(self):
        mon = feed(streams(offset_s=0.045, amp=0.002))
        self.assertEqual(mon.transitions, [])
        self.assertTrue(mon.offset_estimates)
        self.assertFalse(any(e['observable'] for e in mon.offset_estimates))

    def test_causal(self):
        ev = streams(offset_s=0.045, imu_gap=(50.0, 0.6))
        full = feed(ev)
        for cut in (26.0, 36.0, 51.0):
            part = feed(ev, until=cut)
            self.assertEqual(part.transitions, [t for t in full.transitions if t['t_s'] <= cut])

    def test_lost_latches_until_reset(self):
        mon = feed(streams(imu_gap=(30.0, 0.6)))
        self.assertEqual(mon.state, 'LOST')
        mon.reset()
        self.assertEqual((mon.state, mon.epoch), ('NORMAL', 1))


class FaultFilter(unittest.TestCase):
    def test_semantics_match_inject_fault(self):
        drop = {'fault': 'dropout', 'stream_topics': ['/cam0/image_raw', '/cam1/image_raw'],
                'intervals': [{'start_ns': 100, 'end_ns': 200}], 'header_shift_ns': {}}
        keep, shift = fault_filter(drop)
        self.assertFalse(keep('/cam0/image_raw', 200))   # dropout end is the last dropped frame
        self.assertTrue(keep('/imu0', 150))
        out = {'fault': 'outage', 'stream_topics': ['/imu0'], 'intervals': [{'start_ns': 100, 'end_ns': 200}],
               'header_shift_ns': {}}
        keep, _ = fault_filter(out)
        self.assertTrue(keep('/imu0', 200))              # outage is half-open
        self.assertFalse(keep('/imu0', 100))
        _, shift = fault_filter({'fault': 'offset', 'stream_topics': ['/imu0'], 'intervals': [],
                                 'header_shift_ns': {'/imu0': 40000000}})
        self.assertEqual((shift('/imu0'), shift('/cam0/image_raw')), (40000000, 0))


if __name__ == '__main__':
    unittest.main()
