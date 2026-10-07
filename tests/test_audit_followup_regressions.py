"""Regression tests for the Phase 8.2 audit follow-up and retained-channel inverse."""
import copy
import unittest

import numpy as np

from full_position.audit_followup import training_gains, transitions
from full_position.geometry import context
from full_position.invert import invert, predict_holdout
from full_position.model import PositionModel, rescale_response, rescale_states
from full_position.noise import marginal
from test_full_position import synthetic_model


class AuditFollowupRegressions(unittest.TestCase):
    def test_exact_state_gauge_preserves_both_capacity_predictions_and_derivatives(self):
        rng = np.random.default_rng(20261007)
        p = np.array([[100., 60.], [180., 100.], [250., 50.]])
        r = context(p).r
        x = np.column_stack((rng.uniform(-8., 8., 20), rng.uniform(.2, 4.5, 20)))
        gains = (1.17, .91)
        transformed_x = rescale_states(x, *gains)
        for capacity in (27, 37):
            source = synthetic_model(capacity)
            transformed = rescale_response(source, *gains)
            expected, expected_j = source.predict(x, r, True)
            actual, actual_j = transformed.predict(transformed_x, r, True)
            np.testing.assert_allclose(actual, expected, atol=2e-13, rtol=2e-13)
            # Derivatives transform by the inverse physical coordinate gains.
            expected_j[..., 0] /= gains[0]
            expected_j[..., 1] /= gains[1]
            np.testing.assert_allclose(actual_j, expected_j, atol=2e-13, rtol=2e-13)

    def test_training_gains_recover_known_scale_and_reject_mismatched_rows(self):
        x = np.array([[-6., .5], [-2., 1.], [3., 2.], [8., 3.]])
        g, h = 1.25, .8
        y = rescale_states(x, g, h)
        reference = dict(rows=[10, 11, 20, 21], groups=[0, 0, 1, 1],
                         nominal_anchors=[[-5., 1.], [5., 2.]], states=x.tolist())
        candidate = dict(reference, states=y.tolist())
        got = training_gains(reference, candidate)
        self.assertAlmostEqual(got["gaze_gain"], g, places=12)
        self.assertAlmostEqual(got["one_plus_accommodation_gain"], h, places=12)
        np.testing.assert_allclose(got["residual_rms_physical"], [0., 0.], atol=1e-14)
        for key, bad_value in (("rows", [11, 10, 20, 21]),
                               ("groups", [0, 1, 0, 1]),
                               ("nominal_anchors", [[-5., 1.], [4., 2.]])):
            bad = copy.deepcopy(candidate)
            bad[key] = bad_value
            with self.assertRaisesRegex(ValueError, "identical ordered training rows/groups"):
                training_gains(reference, bad)

    @staticmethod
    def _point(j, scored, interior, err, capture="cap", fixation=1, theta=-5., row=None):
        return dict(split_family="gaze", fold="gaze_-5", model="conditional27", row=row,
                    held_point=j, scored=scored, interior=interior, error_px=err,
                    capture=capture, fixation=fixation, nominal_theta=theta)

    @classmethod
    def _frame(cls, row, slots, complete=True, interior=True, E=1., Gt=.2, Ga=.3, worst=1.2):
        return dict(split_family="gaze", fold="gaze_-5", model="conditional27", capture="cap",
                    fixation=1, row=row, slots=slots, nominal_theta=-5., demand=2., complete_triple=complete,
                    complete_interior=interior, E_px=E, G_theta_deg=Gt, G_A_D=Ga,
                    worst_point_px=worst)

    def test_transitions_report_paired_squared_changes_by_boundary_stratum(self):
        old = [self._frame(10, [self._point(0, True, True, [3., 4.], row=10),
                                self._point(1, True, False, [1., 0.], row=10),
                                self._point(2, True, True, [0., 2.], row=10)],
                           E=2., Gt=.3, Ga=.4, worst=5.),
               self._frame(11, [self._point(0, True, True, [1., 1.], row=11),
                                self._point(1, True, True, [2., 0.], row=11),
                                self._point(2, True, True, [0., 1.], row=11)],
                           E=1., Gt=.2, Ga=.3, worst=2.)]
        new = [self._frame(10, [self._point(0, True, False, [1., 1.], row=10),
                                self._point(1, True, True, [2., 0.], row=10),
                                self._point(2, False, False, None, row=10)],
                           complete=False, interior=False),
               self._frame(11, [self._point(0, True, True, [2., 0.], row=11),
                                self._point(1, True, False, [1., 1.], row=11),
                                self._point(2, True, True, [0., 1.], row=11)],
                           E=1.5, Gt=.4, Ga=.5, worst=2.5)]
        got = transitions(old, new)
        slots = {(p["identity"][5], p["identity"][-1]): p for p in got["points"]}
        # Same row/point identities pair, including a point unavailable on one side.
        self.assertAlmostEqual(slots[(10, 0)]["delta_squared_vector_error_px2"], -23.)
        self.assertEqual(slots[(10, 0)]["transition"], "interior_to_bound")
        self.assertFalse(slots[(10, 2)]["shared_scored"])
        self.assertEqual(slots[(10, 2)]["transition"], "unavailable_pair")
        self.assertAlmostEqual(slots[(11, 0)]["delta_squared_vector_error_px2"], 2.)
        self.assertAlmostEqual(slots[(11, 0)]["delta_squared_axis_error_px2"][0], 3.)
        by_stratum = got["strata"]["transition"]["interior_to_bound"]
        self.assertEqual(by_stratum["shared_scored_points"], 2)
        self.assertAlmostEqual(by_stratum["point_squared_error_change"]["pooled"]["mean"], -12.5)
        self.assertEqual(len(got["frames"]), 1)
        self.assertAlmostEqual(got["frames"][0]["delta_squared_E_px"], 1.25)
        self.assertAlmostEqual(got["frames"][0]["delta_squared_G_theta_deg"], .12)

    def test_retained_x_holdout_never_reads_held_point_and_uses_exact_marginal(self):
        p = np.array([[100., 60.], [180., 100.], [250., 50.]])
        ctx = context(p)
        model = synthetic_model(27)
        state = np.array([3., 2.])
        v = model.predict(state, ctx.r).reshape(3, 2)
        full_cov = np.eye(6)*1e-5 + np.ones((6, 6))*2e-6
        j = 1
        held = np.array([2*j, 2*j+1])
        kept = np.array([0, 4])
        first = predict_holdout(model, ctx, j, v[[0, 2], 0], full_cov, channels="x")
        changed = v.copy()
        changed[j] += [1e6, -8e5]
        second = predict_holdout(model, ctx, j, changed[[0, 2], 0], full_cov, channels="x")
        self.assertTrue(first["available"])
        self.assertEqual(first["start_count"], 49)
        self.assertEqual(first["held_point"], j)
        np.testing.assert_array_equal(first["state"], second["state"])
        self.assertEqual(first["branches"], second["branches"])
        self.assertEqual(first["predictions"], second["predictions"])
        # Compare the public helper result to the scalar inverse with the exact x marginal.
        direct = invert(model, ctx.r, v[[0, 2], 0], marginal(full_cov, kept), kept)
        self.assertEqual(first["state"], direct["state"])
        self.assertFalse(np.intersect1d(held, kept).size)
        self.assertNotIn("observed_q", first)

    def test_objective_scaling_preserves_statistical_cost_and_covariance_changes_it(self):
        p = np.array([[100., 60.], [180., 100.], [250., 50.]])
        r = context(p).r
        model = PositionModel(27, np.zeros(27))
        model.beta[4] = 1.
        model.beta[8] = .01
        model.beta[25] = .1
        y = model.predict([5., 2.], r)
        cov = np.eye(6)*1e-4
        starts = np.array([[-5., 2.], [5., 2.]])
        reference = invert(model, r, y, cov, starts=starts)
        numerical = invert(model, r, y, cov, starts=starts, objective_scale=4.)
        changed_cov = invert(model, r, y, cov*4, starts=starts)
        self.assertTrue(reference["available"] and numerical["available"] and changed_cov["available"])
        self.assertEqual(reference["rank"], 2)
        self.assertEqual(len(reference["branches"]), 2)
        self.assertEqual(len(reference["plausible_branches"]), 1)
        self.assertFalse(reference["ambiguous"])
        np.testing.assert_allclose(numerical["state"], reference["state"], atol=1e-8)
        np.testing.assert_allclose([b["cost"] for b in numerical["branches"]],
                                   [b["cost"] for b in reference["branches"]], rtol=1e-8, atol=1e-10)
        self.assertEqual(len(numerical["plausible_branches"]), 1)
        self.assertFalse(numerical["ambiguous"])
        np.testing.assert_allclose(changed_cov["state"], reference["state"], atol=1e-8)
        np.testing.assert_allclose([b["cost"] for b in changed_cov["branches"]],
                                   np.asarray([b["cost"] for b in reference["branches"]])/4,
                                   rtol=1e-7, atol=1e-9)
        self.assertEqual(len(changed_cov["plausible_branches"]), 2)
        self.assertTrue(changed_cov["ambiguous"])


class CalibrationFollowupRegressions(unittest.TestCase):
    def test_selective_curvature_penalty_adds_only_ten_extra_coefficient_rows(self):
        from scipy.optimize._numdiff import approx_derivative
        import pytest
        from full_position.calibrate import ProfiledProblem
        from full_position.model import STATE_SCALE
        from test_sensitivity import calibration_data

        _, groups, anchors, states, r, y, cov = calibration_data()
        model37 = synthetic_model(37)
        strength = .7
        problem = ProfiledProblem(model37, y, r, cov, groups, anchors,
                                  prior_strength=0., curvature_strength=strength)
        indices = np.array([7+6*j+k for j in range(5) for k in (4, 5)])
        self.assertEqual(problem.curvature_penalty.shape, (10, 37))
        nonzero_rows = np.flatnonzero(np.any(problem.curvature_penalty != 0., axis=1))
        self.assertEqual(len(nonzero_rows), 10)
        self.assertEqual(len(set(indices.tolist())), 10)
        np.testing.assert_array_equal(np.flatnonzero(np.any(problem.curvature_penalty != 0., axis=0)), indices)
        np.testing.assert_allclose(problem.curvature_penalty[np.arange(10), indices],
                                   np.sqrt(strength)*problem.column_scale[indices])
        z = (states/STATE_SCALE).ravel()
        rng = np.random.default_rng(314)
        direction = rng.normal(size=z.size)
        jac = problem.jac(z)
        jv = jac@direction
        numeric = approx_derivative(problem.fun, z, method="3-point")@direction
        np.testing.assert_allclose(jv, numeric, atol=3e-5, rtol=4e-4)
        problem.update(z)
        w = rng.normal(size=len(problem.residual))
        self.assertEqual(len(problem.residual), len(problem.b)+2*len(anchors))
        self.assertAlmostEqual(float(jv@w), float(direction@problem.vjp(w)), delta=2e-7)
        with pytest.raises(ValueError, match="requires conditional37"):
            ProfiledProblem(synthetic_model(27), y, r, cov, groups, anchors,
                            curvature_strength=strength)
        with pytest.raises(ValueError, match="finite and nonnegative"):
            ProfiledProblem(model37, y, r, cov, groups, anchors, curvature_strength=-1.)

    def test_continuation_never_accepts_solver_success_without_strict_stationarity(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from full_position.calibrate import fit
        from test_sensitivity import calibration_data

        model, groups, anchors, states, r, y, cov = calibration_data()
        calls = []

        def controlled_solver(fun, x0, **kwargs):
            calls.append((np.asarray(x0).copy(), kwargs))
            # Keep each continuation stage at a deliberately uncertified point.
            point = np.asarray(x0).copy()
            fun(point)
            return SimpleNamespace(x=point, cost=float(fun(point)@fun(point)/2), nfev=1,
                                   status=1, message="synthetic success", success=True)

        checkpoints = []
        with patch("full_position.calibrate.least_squares", side_effect=controlled_solver):
            fitted, selected, diag = fit(model, y, r, cov, groups, anchors,
                starts=1, max_nfev=1, additional_initial_states=[states],
                continuation_stages=2,
                checkpoint=lambda record, beta, x: checkpoints.append(record.copy()))
        self.assertEqual(len(calls), 2*(1+2))
        self.assertFalse(diag["converged"])
        self.assertTrue(all(not alternative["converged"] for alternative in diag["alternatives"]))
        self.assertTrue(all(alternative["acceptance_reason"] == "not_certified"
                            for alternative in diag["alternatives"]))
        self.assertTrue(all(len(alternative["continuation_history"]) == 2
                            for alternative in diag["alternatives"]))
        self.assertTrue(all(not record.get("converged", True) for record in checkpoints
                            if "continuation_stage" in record))


if __name__ == "__main__":
    unittest.main()
