"""Contracts for the fixed training-only shared-y scale estimator."""
import unittest

import numpy as np

from full_position.shared_y_trial import training_tau


class SharedYTrialContracts(unittest.TestCase):
    def test_training_tau_is_equal_group_rms_after_three_point_y_mean(self):
        # Group A has two frames with common-y means 1 and 3; group B has one
        # frame whose three y residuals average to 3. Equal-group RMS is sqrt(7).
        residuals = np.zeros((3, 3, 2), dtype=float)
        residuals[0, :, 1] = [1., 1., 1.]
        residuals[1, :, 1] = [3., 3., 3.]
        residuals[2, :, 1] = [9., 0., 0.]
        groups = np.array([10, 10, 20])
        tau = training_tau(residuals, groups)
        self.assertAlmostEqual(tau, np.sqrt(7.))
        frame_pooled = np.sqrt(np.mean([1.**2, 3.**2, 3.**2]))
        self.assertNotAlmostEqual(tau, frame_pooled)
        # Taking an RMS over the three point residuals before their common-y
        # mean would produce a different group-balanced estimate.
        pointwise_rms_first = np.sqrt((5. + 27.)/2.)
        self.assertNotAlmostEqual(tau, pointwise_rms_first)

    def test_training_tau_rejects_nonfinite_misaligned_and_empty_data(self):
        valid = np.zeros((2, 3, 2), dtype=float)
        groups = np.array([1, 2])
        for bad_value in (np.nan, np.inf, -np.inf):
            residuals = valid.copy()
            residuals[0, 0, 1] = bad_value
            with self.subTest(value=bad_value), self.assertRaisesRegex(ValueError, "finite"):
                training_tau(residuals, groups)
        with self.assertRaisesRegex(ValueError, "aligned nonempty"):
            training_tau(valid, groups[:1])
        with self.assertRaisesRegex(ValueError, "aligned nonempty"):
            training_tau(np.empty((0, 3, 2)), np.array([], dtype=int))
        with self.assertRaisesRegex(ValueError, "aligned nonempty"):
            training_tau(np.zeros((2, 2, 2)), groups)

    def test_training_tau_requires_vector_finite_numeric_group_labels(self):
        residuals = np.zeros((2, 3, 2), dtype=float)
        self.assertEqual(training_tau(residuals, np.array([1., 2.])), 0.)
        self.assertEqual(training_tau(residuals, np.array([1+0j, 2+0j])), 0.)
        with self.assertRaisesRegex(ValueError, "aligned nonempty"):
            training_tau(residuals, np.array([[1], [2]]))
        for bad_groups in (np.array([1., np.nan]), np.array([1., np.inf]),
                           np.array([1+0j, complex(2, np.inf)])):
            with self.subTest(groups=bad_groups), self.assertRaisesRegex(ValueError, "group labels must be finite"):
                training_tau(residuals, bad_groups)


if __name__ == "__main__":
    unittest.main()
