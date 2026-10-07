"""Nested grouped selector contracts: train-only fitting and sealed outer groups."""
import copy
import unittest

import numpy as np

from full_position.selection import REQUIRED_GUARDS, choose, nested_grouped
from test_phase83_information_contracts import _frame


def _guards(**overrides):
    values = dict(minimum_scored_fraction=0.5, minimum_complete_fraction=0.5,
        maximum_G_theta_deg=100., maximum_G_A_D=100., maximum_worst_point_px=100.,
        maximum_x_axis_rms_px=100., maximum_y_axis_rms_px=100.,
        minimum_known_support_fraction=0., maximum_p95_E_px=100.,
        maximum_p95_worst_point_px=100., maximum_unsupported_fraction=1.,
        maximum_bound_slot_fraction=1., provenance="predeclared test guards")
    values.update(overrides)
    return values


def _record(group, row, err, *, candidate="conditional27", complete=True, known=True):
    # Group is also the fixation identifier required by the nested protocol.
    frame = _frame(row, [err, err, err], complete=complete)
    frame.update(fixation=group, fold="inner", model=candidate, capture="cap")
    if not known:
        for slot in frame["slots"]:
            for key in ("theta_anchor", "A_anchor", "theta_empirical", "A_empirical",
                        "P1_context_valid", "context_outside_training_extrema", "p1_parity_seen_in_training"):
                slot["support"][key] = None
    return frame


class NestedSelectionContracts(unittest.TestCase):
    def test_outer_groups_never_reach_fit_or_inner_evaluate_and_choice_precedes_outer_score(self):
        groups = {i: {"label": float(i), "pilot": f"pilot-{i}", "prior": f"prior-{i}"}
                  for i in range(5)}
        fit_calls, eval_calls, events = [], [], []

        def fit(train, candidate):
            fit_calls.append((tuple(train), copy.deepcopy(train), candidate))
            self.assertTrue(train)
            return {"candidate": candidate}

        def evaluate(model, validation):
            ids = tuple(validation)
            eval_calls.append((ids, model["candidate"]))
            # Deterministic inner behavior gives "better" lower errors.
            errors = {"reference": 2., "better": 1.}
            out = [_record(i, 100+i, [errors[model["candidate"]]]*2)
                   for i in ids]
            events.append(("outer" if set(ids) == {4} else "inner", ids,
                           model["candidate"], out[0]["E_px"]))
            return out

        # Observe the decision boundary: it must finish before outer evaluation.
        original_choose = __import__("full_position.selection", fromlist=["choose"]).choose
        def observed_choose(*args, **kwargs):
            result = original_choose(*args, **kwargs)
            events.append(("choice", (), result["selected"], None))
            return result
        from unittest.mock import patch
        with patch("full_position.selection.choose", side_effect=observed_choose):
            result = nested_grouped(groups, [("outer4", [4])], ["reference", "better"],
                                    fit, evaluate, "reference", _guards())

        for ids, payload, _ in fit_calls:
            self.assertNotIn(4, ids)
            self.assertEqual(set(payload), set(ids))
            self.assertTrue(all(set(v) == {"label", "pilot", "prior"} for v in payload.values()))
        self.assertTrue(all(4 not in ids for ids, _ in eval_calls[:-1]))
        self.assertEqual(eval_calls[-1], ((4,), "better"))
        choice_index = next(i for i, event in enumerate(events) if event[0] == "choice")
        outer_index = next(i for i, event in enumerate(events) if event[0] == "outer")
        self.assertLess(choice_index, outer_index)
        self.assertEqual(result["outer4"]["decision"]["selected"], "better")
        self.assertEqual(result["outer4"]["sealed_outer_group_ids"], [4])

    def test_outer_labels_and_observations_cannot_change_inner_choice_or_fit_inputs(self):
        base = {i: {"label": float(i), "observed": np.array([i, -i]),
                    "pilot": (i, "train-only"), "prior": (i, "train-only")} for i in range(5)}

        def execute(data):
            fit_inputs = []
            def fit(train, candidate):
                fit_inputs.append((tuple(train), copy.deepcopy(train), candidate))
                return candidate
            def evaluate(model, validation):
                return [_record(i, 200+i, [1., 0.] if model == "b" else [2., 0.])
                        for i in validation]
            result = nested_grouped(data, [("f", [4])], ["a", "b"], fit,
                                    evaluate, "a", _guards())
            return result["f"]["decision"]["selected"], fit_inputs

        first_choice, first_inputs = execute(base)
        changed = copy.deepcopy(base)
        changed[4]["label"] = 1e100
        changed[4]["observed"][:] = -1000000
        second_choice, second_inputs = execute(changed)
        self.assertEqual(first_choice, second_choice)
        def normalize(calls):
            return [(ids, candidate, {k: {name: (value.tolist() if isinstance(value, np.ndarray) else value)
                                          for name, value in group.items()}
                                      for k, group in payload.items()})
                    for ids, payload, candidate in calls]
        self.assertEqual(normalize(first_inputs), normalize(second_inputs))

    def test_empty_and_wrong_group_validation_results_are_rejected(self):
        groups = {i: {"value": i} for i in range(4)}
        fit = lambda train, candidate: candidate
        with self.assertRaisesRegex(ValueError, "outside its validation split"):
            nested_grouped(groups, [("f", [3])], ["a"], fit,
                lambda model, val: [_record(99, 9, [1., 0.])], "a", _guards())
        # A validation callback returning no records cannot provide a selection cohort.
        with self.assertRaisesRegex(ValueError, "outside its validation split"):
            nested_grouped(groups, [("f", [3])], ["a"], fit,
                lambda model, val: [], "a", _guards())

    def test_missing_guards_never_promote_and_fall_back_to_reference(self):
        groups = {i: {"value": i} for i in range(4)}
        calls = []
        def fit(train, candidate):
            calls.append(candidate)
            return candidate
        result = nested_grouped(groups, [("f", [3])], ["a", "b"], fit,
            lambda model, val: [_record(i, 300+i, [1., 0.]) for i in val], "a", None)
        fold = result["f"]
        self.assertEqual(fold["decision"]["status"], "no_promotion_decision")
        self.assertIsNone(fold["decision"]["selected"])
        self.assertTrue(fold["fallback_reference"])
        self.assertEqual(fold["evaluated_candidate"], "a")


class ChooseContracts(unittest.TestCase):
    def test_guards_reject_state_tail_coverage_and_missing_exposure_failures(self):
        rows = {
            "good": [_record(1, 1, [1., 1.]), _record(1, 2, [1., 1.])],
            "bad_state": [_record(1, 3, [1., 1.], known=False), _record(1, 4, [1., 1.], known=False)],
            "bad_tail": [_record(1, 5, [200., 0.]), _record(1, 6, [1., 1.])],
            "bad_coverage": [_record(1, 7, [1., 1.]), _record(1, 8, [1., 1.], complete=False)],
        }
        rows["bad_exposure"] = [_record(1, 9, [1., 1.]),
                                _record(1, 10, [1., 1.], complete=False)]
        rows["bad_exposure"][1]["capture"] = "scheduled_but_missing"
        rows["bad_training_support"] = [_record(1, 11, [1., 1.], known=False)]
        rows["bad_bound"] = [_record(1, 12, [1., 1.])]
        rows["bad_bound"][0]["slots"][0].update(bound=True, interior=False)
        # Separate folds make exposure identities unique across candidates; guard
        # maxima make only named pathologies fail.
        for name, values in rows.items():
            for f in values:
                f["fold"] = name
        decision = choose(rows, _guards(minimum_scored_fraction=1.,
            minimum_complete_fraction=1., maximum_worst_point_px=10.,
            maximum_p95_E_px=10., maximum_p95_worst_point_px=10.,
            minimum_known_support_fraction=1., maximum_unsupported_fraction=0.,
            maximum_bound_slot_fraction=0.), reference="good")
        self.assertEqual(decision["status"], "inner_selection")
        self.assertEqual(decision["selected"], "good")
        self.assertIn("support_known", decision["rejected"]["bad_state"])
        self.assertIn("worst_point_px", decision["rejected"]["bad_tail"])
        self.assertIn("E_cross_px_tail", decision["rejected"]["bad_tail"])
        self.assertTrue({"scored_fraction", "complete_fraction"} & set(decision["rejected"]["bad_coverage"]))
        self.assertIn("all_scheduled_exposures_contribute", decision["rejected"]["bad_exposure"])
        self.assertIn("training_support", decision["rejected"]["bad_training_support"])
        self.assertIn("bounds", decision["rejected"]["bad_bound"])

    def test_ranking_uses_same_complete_paired_cohort_and_rejects_model_tag_duplicates(self):
        a = [_record(1, 10, [0.1, 0.]), _record(1, 11, [10., 0.])]
        b = [_record(1, 10, [1., 0.]), _record(1, 11, [1., 0.], complete=False)]
        # Candidate A would lose if candidates were ranked on their own
        # surviving complete rows (50.005 vs 1). On the exact shared complete
        # cohort, A wins because only row 10 is eligible for both.
        decision = choose({"a": a, "b": b}, _guards(minimum_complete_fraction=0.5), "a")
        self.assertEqual(decision["selected"], "a")
        self.assertEqual(len(decision["shared_frame_ids"]), 1)

        duplicate = [_record(1, 30, [1., 0.]), _record(1, 30, [1., 0.], candidate="conditional37")]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            choose({"dup": duplicate, "other": [_record(1, 30, [2., 0.])]}, _guards(), "other")


if __name__ == "__main__":
    unittest.main()
