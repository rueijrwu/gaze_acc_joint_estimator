"""Nested grouped selector contracts: immutable scheduled populations."""
import copy
import unittest
from unittest.mock import patch

import numpy as np

from full_position.population import FIELDS, PopulationManifest, manifest
from full_position.selection import choose, nested_grouped
from test_phase83_information_contracts import _frame


def _guards(**overrides):
    values = dict(minimum_scored_fraction=0.5, minimum_complete_fraction=0.5,
        maximum_G_theta_deg=100., maximum_G_A_D=100., maximum_worst_point_px=100.,
        maximum_x_axis_rms_px=100., maximum_y_axis_rms_px=100.,
        minimum_known_support_fraction=0., maximum_p95_E_px=100.,
        maximum_p95_worst_point_px=100., maximum_unsupported_fraction=1.,
        maximum_bound_slot_fraction=1., minimum_shared_frame_fraction=0.,
        minimum_shared_frame_fraction_per_exposure=0., provenance="predeclared test guards")
    values.update(overrides)
    return values


def _record(group, row, err, *, fold="inner", capture="cap", source=None,
            candidate="conditional27", complete=True, known=True):
    frame = _frame(row, [err, err, err], complete=complete)
    frame.update(fixation=group, fold=fold, model=candidate, capture=capture,
                 source_frame_index=row if source is None else source)
    for slot in frame["slots"]:
        slot.update(fixation=group, fold=fold, model=candidate, capture=capture,
                    source_frame_index=frame["source_frame_index"])
    if not known:
        for slot in frame["slots"]:
            for key in ("theta_anchor", "A_anchor", "theta_empirical", "A_empirical",
                        "P1_context_valid", "context_outside_training_extrema", "p1_parity_seen_in_training"):
                slot["support"][key] = None
    return frame


def _scheduled_frame(key, err=1., complete=True):
    fold, capture, fixation, row, source = key
    return _record(fixation, row, [err, 0.], fold=fold, capture=capture,
                   source=source, complete=complete)


def _raw_groups(n=5):
    return {i: {"label": float(i), "pilot": f"pilot-{i}", "prior": f"prior-{i}",
                "raw_rows": [{"capture": f"cap{i//2}", "row": 100+i,
                              "source_frame_index": 100+i}]}
            for i in range(n)}


def _schedule(validation, split_id):
    # The schedule reads only raw frame identities and split identity, before
    # any model is fitted or evaluated.
    return manifest([dict(fold=str(split_id), capture=row["capture"], fixation=group,
                          row=row["row"], source_frame_index=row["source_frame_index"])
                     for group, payload in validation.items() for row in payload["raw_rows"]])


class NestedSelectionContracts(unittest.TestCase):
    def test_outer_groups_are_sealed_and_choice_precedes_one_outer_evaluation(self):
        groups = _raw_groups()
        fit_calls, eval_calls, events = [], [], []

        def fit(train, candidate):
            fit_calls.append((tuple(train), copy.deepcopy(train), candidate))
            self.assertTrue(train)
            return {"candidate": candidate}

        def evaluate(model, validation, population):
            ids = tuple(validation)
            eval_calls.append((ids, model["candidate"], population.frame_ids))
            err = {"reference": 2., "better": 1.}[model["candidate"]]
            out = [_scheduled_frame(key, err) for key in population.frame_ids]
            events.append(("outer" if set(ids) == {4} else "inner", ids,
                           model["candidate"], out[0]["E_px"]))
            return out

        original_choose = choose
        def observed_choose(*args, **kwargs):
            result = original_choose(*args, **kwargs)
            events.append(("choice", (), result["selected"], None))
            return result
        with patch("full_position.selection.choose", side_effect=observed_choose):
            result = nested_grouped(groups, [("outer4", [4])], ["reference", "better"],
                fit, evaluate, "reference", _guards(), schedule_validation=_schedule)

        for ids, payload, _ in fit_calls:
            self.assertNotIn(4, ids)
            self.assertEqual(set(payload), set(ids))
            self.assertTrue(all(set(v) == {"label", "pilot", "prior", "raw_rows"}
                                for v in payload.values()))
        self.assertTrue(all(4 not in ids for ids, _, _ in eval_calls[:-1]))
        self.assertEqual(eval_calls[-1][0:2], ((4,), "better"))
        self.assertEqual(len(eval_calls), 2 * 4 + 1)
        choice_index = next(i for i, event in enumerate(events) if event[0] == "choice")
        outer_index = next(i for i, event in enumerate(events) if event[0] == "outer")
        self.assertLess(choice_index, outer_index)
        self.assertEqual(result["outer4"]["decision"]["selected"], "better")
        self.assertEqual(result["outer4"]["sealed_outer_group_ids"], [4])

    def test_outer_labels_and_observations_cannot_change_inner_choice_or_fit_inputs(self):
        base = _raw_groups()
        for group, value in base.items():
            value["observed"] = np.array([group, -group])
        def execute(data):
            fit_inputs = []
            def fit(train, candidate):
                fit_inputs.append((tuple(train), copy.deepcopy(train), candidate))
                return candidate
            def evaluate(model, validation, population):
                err = 1. if model == "b" else 2.
                return [_scheduled_frame(key, err) for key in population.frame_ids]
            result = nested_grouped(data, [("f", [4])], ["a", "b"], fit, evaluate,
                                    "a", _guards(), schedule_validation=_schedule)
            return result["f"]["decision"]["selected"], fit_inputs
        first_choice, first_inputs = execute(base)
        changed = copy.deepcopy(base)
        changed[4]["label"] = 1e100
        changed[4]["observed"][:] = -1000000
        second_choice, second_inputs = execute(changed)
        self.assertEqual(first_choice, second_choice)
        for (ids_a, data_a, cand_a), (ids_b, data_b, cand_b) in zip(first_inputs, second_inputs):
            self.assertEqual((ids_a, cand_a), (ids_b, cand_b))
            self.assertEqual(set(data_a), set(data_b))
            for group in data_a:
                for field in data_a[group]:
                    if isinstance(data_a[group][field], np.ndarray):
                        np.testing.assert_array_equal(data_a[group][field], data_b[group][field])
                    else:
                        self.assertEqual(data_a[group][field], data_b[group][field])

    def test_explicit_inner_splits_partition_development_groups_without_outer_leakage(self):
        groups = _raw_groups()
        fit_ids = []
        def fit(train, candidate):
            fit_ids.append(set(train))
            return candidate
        def evaluate(model, validation, population):
            return [_scheduled_frame(key, 1.) for key in population.frame_ids]
        result = nested_grouped(groups, [("outer", [4])], ["a"], fit, evaluate, "a",
            _guards(), schedule_validation=_schedule,
            inner_splits=lambda ids: [("partition_01", [0, 1]), ("partition_23", [2, 3])])
        fold = result["outer"]
        self.assertEqual(fold["inner_grouping"], "explicit_predeclared_partition")
        self.assertEqual([r["group_ids"] for r in fold["inner_split_group_ids"]], [[0, 1], [2, 3]])
        self.assertTrue(all(4 not in ids for ids in fit_ids))
        self.assertTrue(all(ids != {0, 1, 2, 3} for ids in fit_ids[:-1]))
        self.assertEqual(fit_ids[-1], {0, 1, 2, 3})

    def test_missing_schedule_is_rejected_and_missing_guards_do_not_promote(self):
        groups = _raw_groups(4)
        fit = lambda train, candidate: candidate
        evaluate = lambda model, val, population: [_scheduled_frame(k) for k in population.frame_ids]
        with self.assertRaisesRegex(ValueError, "scheduled-population callback"):
            nested_grouped(groups, [("f", [3])], ["a"], fit, evaluate, "a", _guards())
        result = nested_grouped(groups, [("f", [3])], ["a", "b"], fit, evaluate,
            "a", None, schedule_validation=_schedule)
        fold = result["f"]
        self.assertEqual(fold["decision"]["status"], "no_promotion_decision")
        self.assertIsNone(fold["decision"]["selected"])
        self.assertTrue(fold["fallback_reference"])
        self.assertEqual(fold["evaluated_candidate"], "a")


class ChooseContracts(unittest.TestCase):
    def test_guards_reject_state_tail_coverage_and_missing_exposure_failures(self):
        keys = [("fold", "cap", 1, i, i) for i in range(1, 3)]
        pop = PopulationManifest(tuple(keys))
        rows = {
            "good": [_scheduled_frame(keys[0]), _scheduled_frame(keys[1])],
            "bad_state": [_scheduled_frame(keys[0]), _scheduled_frame(keys[1])],
            "bad_tail": [_scheduled_frame(keys[0], 200.), _scheduled_frame(keys[1])],
            "bad_coverage": [_scheduled_frame(keys[0]), _scheduled_frame(keys[1], complete=False)],
            "bad_training_support": [_scheduled_frame(keys[0]), _scheduled_frame(keys[1])],
            "bad_bound": [_scheduled_frame(keys[0]), _scheduled_frame(keys[1])],
        }
        for frame in rows["bad_state"]:
            for slot in frame["slots"]:
                for key in ("theta_anchor", "A_anchor", "theta_empirical", "A_empirical",
                            "P1_context_valid", "context_outside_training_extrema", "p1_parity_seen_in_training"):
                    slot["support"][key] = None
        for frame in rows["bad_training_support"]:
            for slot in frame["slots"]:
                slot["support"]["theta_anchor"] = None
        rows["bad_bound"][0]["slots"][0].update(bound=True, interior=False)
        # Build a complete manifest for each candidate's own fold identity.
        candidates = {}
        for name, values in rows.items():
            candidates[name] = []
            for i, frame in enumerate(values):
                new_key = (name, "cap", 1, i + 20, i + 20)
                frame["fold"], frame["row"], frame["source_frame_index"] = new_key[0], new_key[3], new_key[4]
                for slot in frame["slots"]:
                    slot.update(fold=new_key[0], row=new_key[3], source_frame_index=new_key[4])
                candidates[name].append(frame)
        # A single immutable population is shared by all candidates.
        common = [("test", "cap", 1, i + 40, i + 40) for i in range(2)]
        for values in candidates.values():
            for i, frame in enumerate(values):
                frame["fold"], frame["row"], frame["source_frame_index"] = common[i][0], common[i][3], common[i][4]
                for slot in frame["slots"]:
                    slot.update(fold=common[i][0], row=common[i][3], source_frame_index=common[i][4])
        decision = choose(candidates, _guards(minimum_scored_fraction=1., minimum_complete_fraction=0.75,
            maximum_worst_point_px=10., maximum_p95_E_px=10., maximum_p95_worst_point_px=10.,
            minimum_known_support_fraction=1., maximum_unsupported_fraction=0.,
            maximum_bound_slot_fraction=0.), reference="good", population=PopulationManifest(tuple(common)))
        self.assertEqual(decision["status"], "inner_selection")
        self.assertEqual(decision["selected"], "good")
        self.assertIn("support_known", decision["rejected"]["bad_state"])
        self.assertIn("worst_point_px", decision["rejected"]["bad_tail"])
        self.assertIn("E_cross_px_tail", decision["rejected"]["bad_tail"])
        self.assertIn("complete_fraction", decision["rejected"]["bad_coverage"])
        self.assertIn("training_support", decision["rejected"]["bad_training_support"])
        self.assertIn("bounds", decision["rejected"]["bad_bound"])

    def test_shared_cohort_coverage_is_checked_overall_and_per_exposure(self):
        keys = [("f", exposure, 1, i, i) for exposure, start in (("capA", 0), ("capB", 4))
                for i in range(start, start+4)]
        pop = manifest([dict(zip(FIELDS, key)) for key in keys])
        # Four shared complete frames: 3/4 from capA and 1/4 from capB.
        # Overall shared coverage is exactly 1/2, but capB is below its 1/2 floor.
        shared = {keys[0], keys[1], keys[2], keys[4]}
        rows = {candidate: [_scheduled_frame(key, 1., complete=key in shared) for key in keys]
                for candidate in ("a", "b")}
        decision = choose(rows, _guards(minimum_complete_fraction=0.,
            minimum_shared_frame_fraction=0.5, minimum_shared_frame_fraction_per_exposure=0.5),
            reference="a", population=pop)
        self.assertEqual(decision["status"], "no_promotion_decision")
        self.assertIsNone(decision["selected"])
        self.assertEqual(decision["shared_coverage"]["overall_fraction"], 0.5)
        self.assertEqual([x["fraction"] for x in decision["shared_coverage"]["per_exposure"]], [0.75, 0.25])

    def test_every_candidate_must_contribute_each_scheduled_exposure(self):
        keys = [("f", "capA", 1, 1, 1), ("f", "capB", 1, 2, 2)]
        pop = manifest([dict(zip(FIELDS, key)) for key in keys])
        rows = {candidate: [_scheduled_frame(keys[0]),
                            _scheduled_frame(keys[1], complete=False)]
                for candidate in ("a", "b")}
        decision = choose(rows, _guards(minimum_complete_fraction=0.), "a", pop)
        self.assertEqual(decision["status"], "no_promotion_decision")
        self.assertIn("all_scheduled_exposures_contribute", decision["rejected"]["a"])

    def test_support_known_false_guard_still_rejects_when_required(self):
        key = ("f", "cap", 1, 1, 1)
        pop = manifest([dict(zip(FIELDS, key))])
        unknown = _scheduled_frame(key)
        for slot in unknown["slots"]:
            slot["support"]["theta_anchor"] = None
        known = _scheduled_frame(key)
        decision = choose({"unknown": [unknown], "known": [known]},
            _guards(minimum_known_support_fraction=1.), "known", pop)
        self.assertEqual(decision["selected"], "known")
        self.assertIn("support_known", decision["rejected"]["unknown"])

    def test_exact_one_of_eight_shared_frames_is_rejected(self):
        keys = [("f", "cap", 1, i, i) for i in range(8)]
        pop = manifest([dict(zip(FIELDS, key)) for key in keys])
        rows = {candidate: [_scheduled_frame(key, 1., complete=(i == 0)) for i, key in enumerate(keys)]
                for candidate in ("a", "b")}
        decision = choose(rows, _guards(minimum_complete_fraction=0., minimum_shared_frame_fraction=0.2),
                          reference="a", population=pop)
        self.assertEqual(decision["status"], "no_promotion_decision")
        self.assertEqual(decision["shared_coverage"]["shared_frames"], 1)
        self.assertEqual(decision["shared_coverage"]["overall_fraction"], 1/8)

    def test_manifest_requires_exact_frames_and_consistent_three_slot_identities(self):
        key = ("f", "cap", 1, 1, 1)
        pop = manifest([dict(zip(FIELDS, key))])
        frame = _scheduled_frame(key)
        with self.assertRaisesRegex(ValueError, "exactly three"):
            pop.validate([dict(frame, slots=frame["slots"][:2])])
        mismatch = _scheduled_frame(key)
        mismatch["slots"][0]["capture"] = "other"
        with self.assertRaisesRegex(ValueError, "identity disagrees"):
            pop.validate([mismatch])
        mismatch = _scheduled_frame(key)
        mismatch["slots"][0]["source_frame_index"] = 99
        with self.assertRaisesRegex(ValueError, "source-frame identity"):
            pop.validate([mismatch])
        with self.assertRaisesRegex(ValueError, "unique"):
            manifest([dict(zip(FIELDS, key)), dict(zip(FIELDS, key))])

    def test_selection_never_infers_population_from_candidate_survivors(self):
        frame = _scheduled_frame(("f", "cap", 1, 1, 1))
        with self.assertRaisesRegex(ValueError, "immutable scheduled-population manifest"):
            choose({"a": [frame], "b": [copy.deepcopy(frame)]}, _guards(), "a")

    def test_ranking_uses_same_complete_paired_cohort_and_rejects_duplicate_frame_ids(self):
        keys = [("f", "cap", 1, i, i) for i in (10, 11)]
        pop = manifest([dict(zip(FIELDS, key)) for key in keys])
        a = [_scheduled_frame(keys[0], .1), _scheduled_frame(keys[1], 10.)]
        b = [_scheduled_frame(keys[0], 1.), _scheduled_frame(keys[1], 1., complete=False)]
        decision = choose({"a": a, "b": b}, _guards(minimum_complete_fraction=0.5), "a", pop)
        self.assertEqual(decision["selected"], "a")
        self.assertEqual(len(decision["shared_frame_ids"]), 1)
        duplicate = [_scheduled_frame(keys[0]), _scheduled_frame(keys[0])]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            choose({"dup": duplicate, "other": a}, _guards(), "other", pop)


if __name__ == "__main__":
    unittest.main()
