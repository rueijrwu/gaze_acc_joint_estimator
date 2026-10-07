"""Population and ranking contracts for the fresh full-calibration wrapper."""
import unittest
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from full_position.accommodation_full import agreement_schedule, common_comparison, run, verify_conditions
from full_position.population import FIELDS, PopulationManifest
from full_position.data import core_rows, noise_blocks, training_data, window_rows


def scheduled_frame(key, *, metrics=None, complete=True):
    frame = dict(zip(FIELDS, key))
    frame.update(complete_triple=complete, complete_interior=True,
                 shared_accommodation_clipping=False)
    frame.update(metrics or {})
    frame["slots"] = [dict(held_point=j, **{field: frame[field] for field in FIELDS})
                      for j in range(3)]
    return frame


class AccommodationFullContracts(unittest.TestCase):
    def test_verify_conditions_requires_exact_twenty_condition_grid(self):
        groups = [dict(capture=f"capture_{capture}_detections.pkl",
                       target_theta_deg=angle)
                  for capture in range(1, 5)
                  for angle in (-10., -5., 0., 5., 10.)]
        self.assertEqual(verify_conditions(groups), list(range(20)))

        with self.assertRaisesRegex(ValueError, "all 20 reviewed conditions"):
            verify_conditions(groups[:-1])
        incomplete = [g for g in groups if not (
            g["capture"] == "capture_2_detections.pkl" and g["target_theta_deg"] == 5.)]
        incomplete.append(dict(capture="capture_2_detections.pkl", target_theta_deg=6.))
        with self.assertRaisesRegex(ValueError, "Incomplete calibration grid"):
            verify_conditions(incomplete)

    def test_fixation_period_agreement_schedule_keeps_every_original_row(self):
        name = "capture_1_detections.pkl"
        group = dict(capture=name, start_row=0, end_row_exclusive=20)
        capture = SimpleNamespace(
            name=name, frame=np.arange(20), timestamp=np.arange(20, dtype=float),
            ctx=SimpleNamespace(valid=np.ones(20, dtype=bool)),
            baseline_valid=np.ones(20, dtype=bool),
            point_valid=np.ones((20, 3), dtype=bool))
        capture.ctx.valid[1] = False
        capture.baseline_valid[1] = False
        capture.point_valid[1, 0] = False
        rows, population = agreement_schedule({name: capture}, [group], [0],
                                               count=0, window="fixation_period")
        self.assertEqual([row["row"] for row in rows], list(range(20)))
        self.assertEqual([row["selected_for_evaluation"] for row in rows], [True] * 20)
        self.assertEqual(len(population.frame_ids), 20)
        self.assertFalse(rows[1]["baseline_valid"])
        self.assertFalse(rows[1]["p1_valid_geometry"])
        self.assertEqual([row["row"] for row in agreement_schedule(
            {name: capture}, [group], [0], count=0, window="core")[0]], list(range(2, 18)))
        with self.assertRaisesRegex(ValueError, "requires count=0"):
            agreement_schedule({name: capture}, [group], [0],
                               count=1, window="fixation_period")

    def test_fixation_period_data_includes_endpoints_filters_invalid_training_and_splits_noise_gaps(self):
        name = "capture_1_detections.pkl"
        group = dict(capture=name, start_row=2, end_row_exclusive=12,
                     target_theta_deg=0., demand_diopters_label=2.)
        rows = np.arange(16)
        p = np.broadcast_to(np.array([[0., 0.], [1., 0.], [0., 1.]]), (16, 3, 2)).copy()
        q = p.copy()
        capture = SimpleNamespace(
            name=name, p=p, q=q, frame=rows.copy(), timestamp=rows.astype(float),
            ctx=SimpleNamespace(r=np.ones((16, 3, 2))),
            v=np.ones((16, 3, 2)), baseline_valid=np.ones(16, dtype=bool))
        capture.baseline_valid[[2, 8]] = False
        # A source-frame gap splits noise blocks even when the reviewed rows
        # remain adjacent; invalid detections also create a row gap.
        capture.frame[6:] += 1
        captures = {name: capture}
        np.testing.assert_array_equal(window_rows(group, "fixation_period"), np.arange(2, 12))
        np.testing.assert_array_equal(window_rows(group), core_rows(group))
        training, _ = training_data(captures, [group], [0], count=0,
                                    window="fixation_period")
        np.testing.assert_array_equal(training["rows"], [3, 4, 5, 6, 7, 9, 10, 11])
        blocks = noise_blocks(captures, [group], [0], window="fixation_period")
        # Runs [3,4,5], [6,7], and [9,10,11] produce two eligible
        # blocks; the two-row run is excluded by the minimum run length.
        self.assertEqual([len(block) for block in blocks], [3, 3])

    def test_default_core_sampling_and_noise_windows_are_unchanged(self):
        group = dict(start_row=0, end_row_exclusive=20)
        np.testing.assert_array_equal(window_rows(group), np.arange(2, 18))
        np.testing.assert_array_equal(window_rows(group, "core"), np.arange(2, 18))

    def setUp(self):
        # Two exposures, with unequal scheduled row counts, prove that ranking
        # gives each exposure equal weight rather than pooling frames.
        keys = (("full_calibration", "capture_1_detections.pkl", 0, 0, 10),
                ("full_calibration", "capture_1_detections.pkl", 0, 1, 11),
                ("full_calibration", "capture_1_detections.pkl", 1, 0, 20))
        self.population = PopulationManifest(keys)
        self.rows = {
            "ar27_log": [scheduled_frame(key) for key in keys],
            "ar27_sqrt": [scheduled_frame(key) for key in keys],
        }

    def _build_from_rows(self, calls):
        def fake_build(frames, *, expected_exposures, purpose):
            calls.append((frames, expected_exposures, purpose))
            outcomes = {}
            for output_metric, field in (("E_cross_px", "E_px"),
                                         ("G_theta_cross_deg", "G_theta_deg"),
                                         ("G_A_cross_D", "G_A_D"),
                                         ("worst_point_px", "worst_point_px")):
                by_exposure = {}
                for frame in frames:
                    by_exposure.setdefault((frame["fold"], frame["capture"], frame["fixation"]), []).append(
                        frame[field] ** 2)
                # Matches scorecard aggregation: average squared values within
                # each exposure, then give each exposure equal weight.
                mean = float(np.mean([np.mean(values) for values in by_exposure.values()]))
                outcomes[output_metric] = {"squared_error_aggregation": {"mean": mean}}
            return {"outcomes": outcomes}
        return fake_build

    def test_ranks_on_identical_complete_cohort_with_separate_metric_orders(self):
        # Log has lower E while sqrt has lower gaze disagreement. The first
        # exposure has two rows, so a pooled-row score would change the result.
        for name, values in {
            "ar27_log": [(1., 4., 3., 1.), (1., 4., 3., 1.), (0., 1., 1., 0.)],
            "ar27_sqrt": [(2., 1., 2., 2.), (2., 1., 2., 2.), (2., 1., 2., 2.)],
        }.items():
            for frame, (e, theta, accom, worst) in zip(self.rows[name], values):
                frame.update(E_px=e, G_theta_deg=theta, G_A_D=accom, worst_point_px=worst)
        calls = []
        with patch("full_position.accommodation_full.build", side_effect=self._build_from_rows(calls)):
            result = common_comparison(self.rows, {"ar27_log": True, "ar27_sqrt": True,
                "ar27_linear": False, "ar27_quadratic": False}, self.population)

        self.assertEqual(result["status"], "internal_cross_agreement_comparison")
        self.assertEqual(result["shared_frame_ids"], list(self.population.frame_ids))
        self.assertEqual(result["missing_exposures"], [])
        self.assertEqual(result["metric_order"]["E_cross_px"][0], "ar27_log")
        self.assertEqual(result["metric_order"]["G_theta_cross_deg"][0], "ar27_sqrt")
        self.assertEqual(result["lowest_cross_prediction_model"], "ar27_log")
        self.assertIsNone(result["selected_model"])
        self.assertFalse(result["promoted_for_deployment"])
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(call[1] == self.population.exposures for call in calls))
        self.assertTrue(all(call[2] == "internal_full_calibration" for call in calls))
        self.assertTrue(all(len(call[0]) == 3 for call in calls))

    def test_missing_common_exposure_prevents_winner(self):
        self.rows["ar27_log"][-1]["complete_triple"] = False
        self.rows["ar27_sqrt"][-1]["complete_triple"] = False
        result = common_comparison(self.rows, {"ar27_log": True, "ar27_sqrt": True,
            "ar27_linear": False, "ar27_quadratic": False}, self.population)
        self.assertEqual(result["status"], "incomplete_comparison")
        self.assertEqual(result["missing_exposures"], [self.population.exposures[1]])
        self.assertNotIn("lowest_cross_prediction_model", result)
        self.assertIsNone(result["selected_model"])

    def test_uncertified_log_or_single_certified_law_is_incomplete(self):
        for certified in (
            {"ar27_log": False, "ar27_sqrt": True, "ar27_linear": False, "ar27_quadratic": False},
            {"ar27_log": True, "ar27_sqrt": False, "ar27_linear": False, "ar27_quadratic": False},
        ):
            with self.subTest(certified=certified):
                result = common_comparison(self.rows, certified, self.population)
                self.assertEqual(result["status"], "incomplete_comparison")
                self.assertNotIn("lowest_cross_prediction_model", result)
                self.assertIsNone(result["selected_model"])

    def test_population_validation_rejects_missing_and_duplicate_scheduled_rows(self):
        missing = self.rows["ar27_log"][:-1]
        with self.assertRaisesRegex(ValueError, "Scheduled population mismatch"):
            common_comparison({"ar27_log": missing}, {"ar27_log": False,
                "ar27_sqrt": False, "ar27_linear": False, "ar27_quadratic": False}, self.population)

        duplicate = self.rows["ar27_log"] + [self.rows["ar27_log"][0]]
        with self.assertRaisesRegex(ValueError, "Duplicate scheduled frame identity"):
            common_comparison({"ar27_log": duplicate}, {"ar27_log": False,
                "ar27_sqrt": False, "ar27_linear": False, "ar27_quadratic": False}, self.population)

    def test_run_prepares_all_rows_and_dispatches_four_fresh_law_jobs(self):
        groups = [dict(capture=f"capture_{capture}_detections.pkl",
                       target_theta_deg=angle)
                  for capture in range(1, 5)
                  for angle in (-10., -5., 0., 5., 10.)]
        ids = list(range(20))
        frame = dict(zip(FIELDS, ("full_calibration", "capture_1_detections.pkl", 0, 0, 10)))
        population = [frame]
        pop = PopulationManifest((tuple(frame[field] for field in FIELDS),))
        data = dict(rows=np.ones((20, 4)), original_group=np.arange(20),
                    p=np.ones((20, 3, 2)), r=np.ones((20, 3, 2)))
        anchors = np.column_stack((np.arange(20), np.ones(20)))
        pilot = SimpleNamespace(beta=np.arange(27, dtype=float))
        jobs_seen = []

        def fake_fit_task(job):
            jobs_seen.append(job)
            root, output, fold, name, options = job
            # The worker receives only the candidate name and common config;
            # no historical model artifact is supplied as an initializer.
            self.assertEqual(fold, "full_calibration")
            self.assertNotIn("model_path", options)
            with np.load(Path(output) / "splits" / fold / "training_inputs.npz") as saved:
                np.testing.assert_array_equal(saved["rows"], data["rows"])
                np.testing.assert_array_equal(saved["covariance"], np.eye(6))
            return dict(candidate=name, converged=name != "ar27_quadratic")

        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ,
                {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}), \
             patch("full_position.accommodation_full.subprocess.check_output", return_value="a" * 40), \
             patch("full_position.accommodation_full.reviewed", return_value=({}, groups, "interval-hash")), \
             patch("full_position.accommodation_full.verify_conditions", return_value=ids) as verify, \
             patch("full_position.accommodation_full.training_data", return_value=(data, anchors)) as train, \
             patch("full_position.accommodation_full.agreement_schedule",
                   return_value=(population, pop)) as agreement_schedule_call, \
             patch("full_position.accommodation_full.source_hashes", return_value={}), \
             patch("full_position.accommodation_full.pilot_fit", return_value=pilot) as pilot_call, \
             patch("full_position.accommodation_full.noise_blocks", return_value="blocks") as noise, \
             patch("full_position.accommodation_full.coordinate_covariance",
                   return_value=(np.eye(12), {"kind": "mock"})), \
             patch("full_position.accommodation_full.reference_covariance",
                   return_value=np.eye(6)), \
             patch("full_position.accommodation_full._fit_task", side_effect=fake_fit_task), \
             patch("full_position.accommodation_full.summarize", return_value={
                 "mock": True,
                 "run_counts": {},
                 "common_model_comparison": {"status": "incomplete_comparison"}}):
            root = Path(tmp)
            output = root / "run"
            result = run(root, output, workers=1, max_nfev=12,
                         agreement_per_fixation=0, source_commit="a" * 40,
                         agreement_window="fixation_period")
            saved_training = __import__("json").loads(
                (output / "splits" / "full_calibration" / "training.json").read_text())
            config = __import__("json").loads((output / "config.json").read_text())
            completion = __import__("json").loads((output / "completion.json").read_text())

        self.assertEqual(result["mock"], True)
        self.assertFalse(result["experiment_success"])
        verify.assert_called_once()
        train.assert_called_once()
        self.assertEqual(train.call_args.kwargs["count"], 0)
        self.assertEqual(train.call_args.kwargs["window"], "core")
        self.assertEqual(noise.call_args.kwargs["window"], "core")
        agreement_schedule_call.assert_called_once_with({}, groups, ids, 0, "fixation_period")
        self.assertEqual(set(map(int, data["original_group"])), set(ids))
        self.assertEqual(pilot_call.call_count, 1)
        self.assertEqual([job[3] for job in jobs_seen],
                         ["ar27_log", "ar27_sqrt", "ar27_linear", "ar27_quadratic"])
        self.assertTrue(all(job[4]["coverage_policy"]["absolute_accuracy_thresholds"] is None
                            for job in jobs_seen))
        np.testing.assert_array_equal(saved_training["pilot_coefficients"], pilot.beta)
        np.testing.assert_array_equal(saved_training["coordinate_covariance"], np.eye(12))
        self.assertEqual(config["training_group_ids"], ids)
        self.assertTrue(config["fresh_calibration_per_law"])
        self.assertFalse(config["reused_previous_fitted_models"])
        self.assertEqual(config["agreement_window"], "fixation_period")
        self.assertEqual(config["training_window"], "core")
        self.assertEqual(config["agreement_sampling"], "all_fixation_period_frames")
        self.assertEqual(config["calibration_sampling"], "all_valid_central80_rows")
        self.assertEqual(config["calibration_rows"], 20)
        self.assertTrue(completion["complete"])
        self.assertFalse(completion["experiment_success"])
        self.assertEqual(completion["completed_fit_tasks"], 4)


if __name__ == "__main__":
    unittest.main()
