"""A2 scalar weight policy: bounds, hysteresis, support ramp, validation (pure Python)."""
import unittest

from uwbench.a.backend.weights import ScalarWeight


class ScalarWeightTests(unittest.TestCase):
    def test_none_is_unit_weight(self):
        w = ScalarWeight({'kind': 'none'})
        self.assertEqual({w.weight(1, 100.0) for _ in range(5)}, {1.0})

    def test_innovation_hysteresis_and_bounds(self):
        w = ScalarWeight({'kind': 'innovation', 'w_min': 0.2, 'ema_alpha': 1.0, 'high': 3.0, 'low': 1.5})
        seq = [0.5, 2.9, 3.1, 2.0, 1.6, 1.4, 2.5]
        got = [w.weight(7, e) for e in seq]
        self.assertEqual(got, [1.0, 1.0, 0.2, 0.2, 0.2, 1.0, 1.0])   # stays low between low and high
        self.assertTrue(all(0.2 <= g <= 1.0 for g in got))

    def test_innovation_smoothing_is_per_track(self):
        w = ScalarWeight({'kind': 'innovation', 'w_min': 0.1, 'ema_alpha': 0.3, 'high': 3.0, 'low': 1.5})
        w.weight(1, 1.0)
        self.assertEqual(w.weight(1, 30.0), 0.1)        # 0.3*30 + 0.7*1 = 9.7 > 3
        self.assertEqual(w.weight(2, 1.0), 1.0)         # another track is unaffected
        self.assertAlmostEqual(w.ema[1], 9.7)

    def test_support_ramp(self):
        w = ScalarWeight({'kind': 'support', 'w_min': 0.1, 'n_full': 4})
        got = [w.weight(3) for _ in range(6)]
        for a, b in zip(got, [0.1, 0.4, 0.7, 1.0, 1.0, 1.0]):
            self.assertAlmostEqual(a, b)

    def test_validation(self):
        for bad in [{'kind': 'tukey'}, {'kind': 'support', 'w_min': 0.0}, {'kind': 'innovation', 'low': 3.0, 'high': 2.0},
                    {'kind': 'support', 'n_full': 1}]:
            with self.assertRaises(ValueError):
                ScalarWeight(bad)
        with self.assertRaises(ValueError):
            ScalarWeight({'kind': 'innovation'}).weight(1)


if __name__ == '__main__':
    unittest.main()
