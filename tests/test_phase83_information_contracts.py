"""Focused contracts for Phase 8.3 transformed masks and shared scorecards."""
from types import SimpleNamespace
import inspect
import unittest

import numpy as np
import pytest

from full_position.audit_followup import transitions
from full_position.diagnostics import enrich_subset_records, subset_support
from full_position.geometry import context
from full_position.information import (
    MASKS, TransformedResponse, predict_raw_holdout, predict_transformed, retained_transform,
)
from full_position.invert import objective_derivatives, STARTS
from full_position.model import PositionModel, STATE_SCALE
from full_position.noise import marginal, whitening
from full_position.profile import coordinate_polynomials, profile_inverse
from full_position.schema import protocol_metadata
from full_position.scorecard import build, equal_exposure, unique
from test_full_position import synthetic_model


P = np.array([[100., 60.], [180., 100.], [250., 50.]])


def _state_set(branches):
    return sorted(tuple(np.round(b["state"], 8)) for b in branches)


def _slot(j, *, row, scored=True, interior=True, error=(1., 0.), state=(0., 1.)):
    return dict(split_family="gaze", fold="gaze_+0", model="conditional27",
        capture="cap", fixation=1, row=row, nominal_theta=0., demand=2., held_point=j, scored=scored,
        input_valid=True, certified=scored, identifiable=scored, unambiguous=scored,
        error_px=list(error) if scored else None,
        error_normalized=list(error) if scored else None,
        state=list(state) if scored else None, bound=not interior if scored else None,
        interior=interior if scored else None, failure_reason=None if scored else "missing",
        support={"theta_anchor": True, "A_anchor": True, "theta_empirical": True,
                 "A_empirical": True, "P1_context_valid": True,
                 "context_outside_training_extrema": False, "p1_parity_seen_in_training": True})


def _frame(row, errors, *, exposure="cap", complete=True, complete_interior=True):
    slots = [_slot(j, row=row, error=errors[j], interior=complete_interior) for j in range(3)]
    return dict(split_family="gaze", fold="gaze_+0", model="conditional27",
        capture=exposure, fixation=1, row=row, source_frame_index=row,
        slots=slots, nominal_theta=0., demand=2., complete_triple=complete, complete_interior=complete and complete_interior,
        E_px=float(np.sqrt(np.mean([np.dot(e, e) for e in errors]))) if complete else None,
        E_normalized=float(np.sqrt(np.mean([np.dot(e, e) for e in errors]))) if complete else None,
        G_theta_deg=1. if complete else None, G_A_D=2. if complete else None,
        worst_point_px=max(np.linalg.norm(e) for e in errors) if complete else None,
        shared_accommodation_clipping=False)


class TestInformationMaskContracts(unittest.TestCase):
    def test_y_mask_batch_derivatives_polynomial_profiles(self):
        states = np.array([[-12., .6], [-2., 1.7], [11., 4.2]])
        for capacity in (27, 37):
            model = synthetic_model(capacity)
            for held in (0, 1, 2):
                kept, _ = retained_transform(held, "xy")
                for mask in MASKS[1:]:
                    _, H = retained_transform(held, mask)
                    transformed = TransformedResponse(model, kept, H)
                    values, jac = transformed.predict(states, context(P).r, True)
                    hess = transformed.state_hessian(states, context(P).r)
                    scalar = [transformed.predict(x, context(P).r, True) for x in states]
                    np.testing.assert_allclose(values, np.stack([item[0] for item in scalar]), atol=1e-12, rtol=1e-12)
                    np.testing.assert_allclose(jac, np.stack([item[1] for item in scalar]), atol=1e-12, rtol=1e-12)
                    np.testing.assert_allclose(hess, np.stack([
                        transformed.state_hessian(x, context(P).r) for x in states]), atol=1e-12, rtol=1e-12)

                    for accommodation in (.3, 1.9, 4.7):
                        coefficients = coordinate_polynomials(transformed, context(P).r, accommodation)
                        points = np.column_stack((np.linspace(-19., 19., 9),
                                                  np.full(9, accommodation)))
                        t = points[:, 0]/10.
                        expected = transformed.predict(points, context(P).r)
                        reconstructed = np.column_stack([
                            np.polynomial.polynomial.polyval(t, coefficients[:, c])
                            for c in range(transformed.channels)])
                        np.testing.assert_allclose(reconstructed, expected, atol=2e-11, rtol=2e-11)

        # Smoke the independent profile with a synthetic transformed response;
        # this checks batched polynomial evaluation, not global completeness.
        model = synthetic_model(27)
        kept, H = retained_transform(0, "x_y_common_difference")
        transformed = TransformedResponse(model, kept, H)
        observed = transformed.predict(np.array([3., 2.]), context(P).r)
        profile = profile_inverse(transformed, context(P).r, observed,
                                  np.eye(len(observed))*.05, grid=3)
        self.assertEqual(profile["profile_audit"]["accommodation_grid"], 3)
        self.assertGreater(profile["profile_audit"]["seed_count"], 0)

    def test_raw_x_mask_ignores_excluded_xy_and_unused_retained_y(self):
        for capacity in (27, 37):
            for held in (0, 1, 2):
                self._check_raw_x_mask(capacity, held)

    def _check_raw_x_mask(self, capacity, held):
        model = synthetic_model(capacity)
        ctx = context(P)
        full_cov = np.eye(12)*.04
        sigma = np.eye(12)*.002
        state = np.array([3., 2.])
        q = ctx.c+ctx.ell*model.predict(state, ctx.r).reshape(3, 2)
        valid = np.ones(3, bool)
        starts = np.array([[-10., 1.], [0., 2.], [10., 4.]])
        before = predict_raw_holdout(model, P, q, valid, held, model, [0., 2.], sigma,
                                     mask="x", starts=starts)
        # Full covariance input is constructed by the helper from sigma. The
        # held coordinates and unused retained y channels must be unread.
        changed = q.copy()
        changed[held] += np.array([2e5, -3e5])
        retained = [j for j in range(3) if j != held]
        changed[retained, 1] = np.nan
        after = predict_raw_holdout(model, P, changed, valid, held, model, [0., 2.], sigma,
                                    mask="x", starts=starts)
        self.assertEqual(before.get("available"), after.get("available"))
        self.assertEqual(before.get("reason"), after.get("reason"))
        if before.get("available"):
            np.testing.assert_allclose(before["state"], after["state"], atol=0., rtol=0.)
            self.assertEqual(before["branches"], after["branches"])
            self.assertEqual(before["predictions"], after["predictions"])
        args = inspect.signature(predict_raw_holdout).parameters
        self.assertFalse({"nominal_theta", "demand", "all_three_state"} & set(args))

    def test_random_spd_common_difference_transform_matches_xy_objective_gradient_hessian(self):
        rng = np.random.default_rng(20261008)
        model = synthetic_model(37)
        ctx = context(P)
        held = 1
        kept, Hxy = retained_transform(held, "xy")
        _, H = retained_transform(held, "x_y_common_difference")
        self.assertEqual(H.shape, (4, 4))
        self.assertEqual(np.linalg.matrix_rank(H), 4)
        root = rng.normal(size=(6, 6))
        full_cov = root@root.T+np.eye(6)*.5
        R = marginal(full_cov, kept)
        transformed = TransformedResponse(model, kept, H)
        test_state = np.array([4., 2.3])
        observed = model.predict(test_state, ctx.r)[kept]+rng.normal(size=4)*.01
        z = test_state/STATE_SCALE
        direct = objective_derivatives(model, ctx.r, observed, whitening(R), kept, z)
        mapped = objective_derivatives(transformed, ctx.r, H@observed,
                                       whitening(H@R@H.T), np.arange(4), z)
        np.testing.assert_allclose(mapped[0], direct[0], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(mapped[1], direct[1], rtol=2e-11, atol=2e-11)
        np.testing.assert_allclose(mapped[2], direct[2], rtol=2e-10, atol=2e-10)
        value, jac = transformed.predict(test_state, ctx.r, True)
        original, original_jac = model.predict(test_state, ctx.r, True)
        np.testing.assert_allclose(value, H@original[kept])
        np.testing.assert_allclose(jac, H@original_jac[kept])
        np.testing.assert_allclose(transformed.state_hessian(test_state, ctx.r),
                                   np.einsum("ij,jab->iab", H, model.state_hessian(test_state, ctx.r)[kept]))

    def test_full_xy_and_invertible_common_difference_have_same_inverse(self):
        model = synthetic_model(27)
        ctx = context(P)
        state = np.array([3., 2.])
        v = model.predict(state, ctx.r)
        cov = np.eye(6)*.02+np.ones((6, 6))*.001
        held = 1
        kept, _ = retained_transform(held, "xy")
        # This one full 49-start comparison verifies the transformed path all
        # the way through inverse, branch clustering, rank and held prediction.
        xy = predict_transformed(model, ctx, held, v[kept], cov, "xy", STARTS)
        both = predict_transformed(model, ctx, held, v[kept], cov,
                                   "x_y_common_difference", STARTS)
        self.assertTrue(xy["available"] and both["available"])
        self.assertEqual(xy["rank"], both["rank"])
        self.assertEqual(xy["ambiguous"], both["ambiguous"])
        self.assertEqual(xy["numerical_ties"], both["numerical_ties"])
        self.assertEqual(_state_set(xy["branches"]), _state_set(both["branches"]))
        np.testing.assert_allclose(xy["state"], both["state"], atol=2e-7, rtol=2e-7)
        np.testing.assert_allclose([p["pixel"] for p in xy["predictions"]],
                                   [p["pixel"] for p in both["predictions"]], atol=2e-6, rtol=2e-7)

    def test_protocol_metadata_does_not_zero_image_y_or_add_vertical_state(self):
        for capacity in (27, 37):
            self._check_protocol_metadata(capacity)

    def _check_protocol_metadata(self, capacity):
        protocol = protocol_metadata()
        self.assertEqual(protocol["state_variables"], ["theta_x_deg", "A_D"])
        self.assertEqual(protocol["calibration_theta_y_nominal_deg"], 0.)
        self.assertEqual(protocol["calibration_theta_y_role"],
                         "protocol_constraint_not_framewise_ground_truth")
        self.assertIn("image-y is not vertical gaze", protocol["image_axis_convention"])
        model = synthetic_model(capacity)
        value, jac = model.predict(np.array([0., 2.]), context(P).r, True)
        self.assertEqual(value.shape, (6,))
        self.assertEqual(jac.shape, (6, 2))
        # y coordinates are the odd slots; both vary with horizontal gaze and A.
        self.assertGreater(np.linalg.norm(jac[1::2, 0]), 0.)
        self.assertGreater(np.linalg.norm(jac[1::2, 1]), 0.)


class TestScorecardAndSupportContracts(unittest.TestCase):
    def test_support_preserves_unknowns_and_uses_each_subset_state(self):
        ctx = context(P)
        meta = {"provenance": {"training_group_ids": [0, 1],
            "anchor_range": [[-5., 0.], [5., 4.]]}}
        unknown = subset_support(ctx, meta, [1., 2.], empirical_range=None)
        self.assertIsNone(unknown["theta_empirical"])
        self.assertEqual(unknown["unavailable_reasons"]["theta_empirical"],
                         "training_empirical_range_unavailable")
        self.assertEqual(unknown["unavailable_reasons"]["context_outside_training_extrema"],
                         "training_context_metadata_unavailable")
        captures = {"cap": SimpleNamespace(p=P[None])}
        frame = {"capture": "cap", "row": 0, "all_three_state": [0., 2.],
                 "slots": [{"state": [8., 2.]}, {"state": [1., 2.]}]}
        enriched = enrich_subset_records([frame], captures, meta, [[-2., 0.], [2., 4.]])
        self.assertFalse(enriched[0]["slots"][0]["support"]["theta_anchor"])
        self.assertFalse(enriched[0]["slots"][0]["support"]["theta_empirical"])
        self.assertTrue(enriched[0]["slots"][1]["support"]["theta_anchor"])
        self.assertTrue(enriched[0]["slots"][1]["support"]["theta_empirical"])
        self.assertNotEqual(enriched[0]["slots"][0]["support"]["theta_anchor"],
                            enriched[0]["slots"][1]["support"]["theta_anchor"])

    def test_frame_only_bound_transition_is_retained_and_duplicates_rejected(self):
        ref = _frame(10, [(1., 0.), (1., 0.), (1., 0.)], complete=True,
                     complete_interior=False)
        cand = _frame(10, [(2., 0.), (2., 0.), (2., 0.)], complete=True,
                      complete_interior=False)
        # No individual point has bound->bound: transitions are i->b or b->i.
        for j, (a, b) in enumerate(((True, False), (False, True), (True, False))):
            ref["slots"][j]["interior"] = a
            ref["slots"][j]["bound"] = not a
            cand["slots"][j]["interior"] = b
            cand["slots"][j]["bound"] = not b
        pair = transitions([ref], [cand])
        self.assertIn("bound_to_bound", pair["strata"]["transition"])
        self.assertEqual(pair["strata"]["transition"]["bound_to_bound"]["paired_complete_frames"], 1)
        self.assertEqual(pair["strata"]["transition"]["bound_to_bound"]["scheduled_points"], 0)
        with pytest.raises(ValueError, match="Duplicate cross-check identity"):
            unique([ref, ref])
        dup_slot_frame = _frame(11, [(1., 0.), (1., 0.), (1., 0.)])
        dup_slot_frame["slots"][1]["held_point"] = dup_slot_frame["slots"][0]["held_point"]
        with pytest.raises(ValueError, match="Duplicate cross-check identity"):
            unique(dup_slot_frame["slots"], point=True)

    def test_scorecard_equal_exposure_keeps_partial_absent_and_pooled_distinct(self):
        rows = [dict(fold="f", capture="a", fixation=1),
                dict(fold="f", capture="a", fixation=1),
                dict(fold="f", capture="b", fixation=2)]
        got = equal_exposure(rows, [1., 1., 81.], [("f", "a", 1), ("f", "b", 2), ("f", "c", 3)])
        self.assertAlmostEqual(got["mean"], 41.)
        self.assertEqual(got["contributing_exposure_count"], 2)
        self.assertEqual(got["scheduled_exposure_count"], 3)
        self.assertEqual(got["status"], "partial_exposure_support")
        self.assertEqual(got["absent_exposure_ids"], [("f", "c", 3)])
        pooled_mean = np.mean([1., 1., 81.])
        self.assertNotAlmostEqual(got["mean"], pooled_mean)

        a0 = _frame(1, [(1., 0.), (0., 1.), (1., 1.)], exposure="a")
        a1 = _frame(2, [(1., 0.), (0., 1.), (1., 1.)], exposure="a")
        b0 = _frame(3, [(3., 0.), (0., 3.), (3., 3.)], exposure="b")
        partial = _frame(4, [(3., 0.), (0., 3.), (3., 3.)], exposure="d", complete=False)
        card = build([a0, a1, b0, partial], expected_exposures=[("gaze_+0", "c", 5)], purpose="inner_validation")
        self.assertEqual(card["purpose"], "inner_validation")
        self.assertEqual(card["coverage"]["scheduled_frames"], 4)
        self.assertEqual(card["coverage"]["complete_triples"], 3)
        agg = card["outcomes"]["E_cross_px"]["squared_error_aggregation"]
        self.assertEqual(agg["status"], "partial_exposure_support")
        self.assertIn("gaze_+0", str(card["membership"]["scheduled"]))


if __name__ == "__main__":
    import unittest
    unittest.main()
