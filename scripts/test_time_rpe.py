import unittest
import numpy as np
from evo.core import metrics
from evaluate_openvins import summarize, time_pairs


class TimeRpeTest(unittest.TestCase):
    def test_empty_estimate_is_error(self):
        with self.assertRaises(ValueError):
            time_pairs([], 1)
        with self.assertRaises(ValueError):
            summarize([])

    def test_seconds_are_not_frame_counts(self):
        self.assertEqual(time_pairs([0, 0.05, 0.1, 1.0, 1.05, 1.1], 1), [(0, 3), (1, 4), (2, 5)])
        self.assertEqual(time_pairs([0, 0.05, 0.1, 1.5, 1.55], 1), [])

    def test_duplicate_time_is_error(self):
        with self.assertRaises(ValueError):
            time_pairs([0, 0, 1], 1)

    def test_known_translation_error(self):
        identity = np.eye(4)
        ref, est = np.eye(4), np.eye(4)
        ref[0, 3], est[0, 3] = 1.0, 1.1
        error = metrics.RPE.rpe_base(identity, ref, identity, est)
        self.assertAlmostEqual(np.linalg.norm(error[:3, 3]), 0.1)

    def test_global_translation_does_not_change_relative_error(self):
        start, end = np.eye(4), np.eye(4)
        end[0, 3] = 2.0
        global_frame = np.eye(4); global_frame[:3, 3] = [5, 6, 7]
        error = metrics.RPE.rpe_base(start, end, global_frame@start, global_frame@end)
        np.testing.assert_allclose(error, np.eye(4), atol=1e-12)


if __name__ == '__main__':
    unittest.main()
