"""Contracts for saved-result paired diagnostics and training-only regression."""
import copy
import unittest

import numpy as np

from full_position.latest_audit import paired_summary, ridge_predict
from full_position.population import FIELDS, manifest
from test_phase83_information_contracts import _frame


def _audit_frame(capture, row, errors):
    frame = _frame(row, errors)
    frame.update(fold="audit", capture=capture, fixation=1, row=row,
                 source_frame_index=row, nominal_theta=0.)
    for slot in frame["slots"]:
        slot.update(fold="audit", capture=capture, fixation=1, row=row,
                    source_frame_index=row)
    return frame


class LatestAuditContracts(unittest.TestCase):
    def test_paired_summary_requires_manifest_and_uses_exact_joint_support_intersection(self):
        keys = [("audit", "capA", 1, 10, 10), ("audit", "capA", 1, 11, 11),
                ("audit", "capB", 1, 12, 12)]
        population = manifest([dict(zip(FIELDS, key)) for key in keys])
        reference = [_audit_frame(key[1], key[3], [[1., 0.], [2., 0.], [3., 0.]]) for key in keys]
        candidate = [_audit_frame(key[1], key[3], [[3., 0.], [4., 0.], [5., 0.]]) for key in keys]
        # The test cohort requires both saved arms to carry empirical state
        # support. Only two of three held points on capA share that support;
        # capB has none, so its scheduled exposure remains absent in aggregates.
        for arm in (reference, candidate):
            for slot in arm[1]["slots"]:
                if slot["held_point"] == 1:
                    slot["support"]["theta_empirical"] = False
            for slot in arm[2]["slots"]:
                slot["support"]["theta_empirical"] = False
        reference[0]["nominal_theta"] = -30.
        candidate[0]["nominal_theta"] = 70.
        report, membership = paired_summary(reference, candidate, population, "joint_empirical_state")
        self.assertEqual(report["shared_scored_points"], 5)
        self.assertEqual([point[-1] for point in membership["point_ids"]], [0, 1, 2, 0, 2])
        self.assertEqual(membership["frame_ids"], [keys[0]])
        self.assertEqual(report["point_changes"]["x"]["squared_change"]["absent_exposure_ids"],
                         [("audit", "capB", 1)])
        self.assertEqual(report["per_point_axis_changes"]["1"]["x"]["absent_exposure_ids"],
                         [("audit", "capB", 1)])
        self.assertEqual(report["point_changes"]["x"]["squared_change"]["mean"], 12.)
        with self.assertRaisesRegex(ValueError, "population mismatch"):
            paired_summary(reference, candidate, manifest([dict(zip(FIELDS, keys[0]))]),
                           "joint_empirical_state")

    def test_ridge_predictions_ignore_changed_targets_outside_training_rows(self):
        train_x = np.array([[0., 1.], [1., 0.], [2., 1.], [5., -2.]])
        targets = np.array([[1., -1.], [2., 1.], [2.5, .5], [8., 12.]])
        training = np.array([True, True, True, False])
        test = ~training
        first = ridge_predict(train_x[training], targets[training], train_x[test])
        changed = copy.deepcopy(targets)
        changed[test] = 1e9
        second = ridge_predict(train_x[training], changed[training], train_x[test])
        np.testing.assert_array_equal(first, second)


if __name__ == "__main__":
    unittest.main()
