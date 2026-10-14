import unittest
import numpy as np
from uwbench.__main__ import main
from uwbench.metrics.occupancy import map_ratios
from uwbench.sonar import arc_points, effect_size_row, range_bearing


class SonarGeometryTest(unittest.TestCase):
    def test_whole_arc_has_one_range_and_bearing(self):
        points = arc_points(2.0, np.radians(20), np.radians([-10, 0, 10]))
        rho, theta = range_bearing(points)
        np.testing.assert_allclose(rho, 2.0)
        np.testing.assert_allclose(theta, np.radians(20))
        self.assertGreater(np.ptp(points[:, 2]), 0.6)

    def test_slant_range_is_not_horizontal_range(self):
        rho, _ = range_bearing([[1.0, 0, 1.0]])
        self.assertAlmostEqual(rho[0], np.sqrt(2))

    def test_plan_effect_size_examples(self):
        for aperture, rng, span in [(20, 1, 0.35), (20, 3, 1.05), (12, 1, 0.21), (12, 3, 0.63)]:
            row = effect_size_row(rng, np.radians(aperture), 0.10)
            self.assertAlmostEqual(row['arc_height_span_m'], span, delta=0.01)
        self.assertEqual(effect_size_row(1, np.radians(20), 0.10)['b2a_vertical_voxels_per_echo'], 4)
        with self.assertRaises(ValueError):
            effect_size_row(1, 0, 0.10)


class MapRatioTest(unittest.TestCase):
    def test_definitions(self):
        truth = ['occ', 'occ', 'free', 'free', 'excluded']
        predicted = ['free', 'occ', 'occ', 'unknown', 'free']
        r = map_ratios(truth, predicted)
        self.assertEqual(r['false_free_rate'], 0.5)
        self.assertEqual(r['free_hazard_fraction'], 1.0)
        self.assertEqual(r['effective_coverage'], 0.75)
        self.assertEqual(r['false_occupied_rate'], 0.5)

    def test_empty_denominator_is_undefined_not_zero(self):
        r = map_ratios(['occ', 'free'], ['occ', 'unknown'])
        self.assertIsNone(r['free_hazard_fraction'])

    def test_all_unknown_has_zero_coverage(self):
        self.assertEqual(map_ratios(['occ', 'free'], ['unknown', 'unknown'])['effective_coverage'], 0)


class ContractTest(unittest.TestCase):
    def test_unimplemented_command_fails(self):
        self.assertNotEqual(main(['check-method']), 0)


if __name__ == '__main__':
    unittest.main()
