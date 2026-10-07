"""Regression tests for audit cross-check identity, masks, and null semantics."""
import math

import pytest

from full_position.crosscheck import (
    _metrics_by_group,
    _paired,
    _summarize,
    _triple_metrics,
    join_records,
)


def _population(capture="cap-a", *, fixation=1, row=100, p1=True, points=(True, True, True)):
    return {
        "capture": capture,
        "fixation": str(fixation),
        "row": str(row),
        "frame": str(row),
        "selected_for_evaluation": "True",
        "p1_valid_geometry": str(p1),
        **{f"p4_{j + 1}_valid": str(points[j]) for j in range(3)},
    }


def _frame(capture="cap-a", *, fixation=1, row=100, estimated=True):
    return {
        "capture": capture,
        "fixation": fixation,
        "row": row,
        "frame": row,
        "baseline_valid": True,
        "point_valid": [True, True, True],
        "estimated": estimated,
        "estimate_kind": "all_three" if estimated else "unavailable",
        "theta": 0.25 if estimated else None,
        "A": 1.5 if estimated else None,
        "nominal_theta": 0.0,
        "demand": 2.0,
    }


def _branch(state, certified=True):
    return {
        "state": list(state),
        "certificate": {
            "certified": certified,
            "reason": "certified_stationary_minimum" if certified else "uncertified",
        },
    }


def _holdout(j, state, *, capture="cap-a", fixation=1, row=100,
             error=None, normalized=None, branch_list=None, rank=2,
             ambiguous=False, bound=False):
    if error is None:
        error = [float(j + 1), float(j + 2)]
    if normalized is None:
        normalized = [error[0] / 2.0, error[1] / 2.0]
    return {
        "capture": capture,
        "fixation": fixation,
        "row": row,
        "held_point": j,
        "available": True,
        "state": list(state),
        "reason": "ok",
        "rank": rank,
        "singular_values_physical": [4.0, 2.0] if rank == 2 else [4.0, 0.0],
        "ambiguous": ambiguous,
        "testable": not ambiguous,
        "score_available": True,
        "error_px": list(error),
        "error_normalized": normalized,
        "at_bound": bound,
        "predictions": [],
        "branches": branch_list if branch_list is not None else [_branch(state)],
        "plausible_branches": [],
        "state_difference_from_all": [0.0, 0.0],
        "nominal_theta": 0.0,
        "demand": 2.0,
        "anchor_range": [[-10.0, 0.0], [10.0, 6.0]],
        "training_range": [[-8.0, 0.0], [8.0, 5.0]],
    }


def _join(populations, frames, holdouts):
    return join_records("gaze", "gaze_fold", "conditional27", populations, frames, holdouts)


def _all_three(capture="cap-a", *, fixation=1, row=100, states=None, error_vectors=None,
               normalized_vectors=None, interior=(True, True, True)):
    states = states or [[0.0, 1.0], [2.0, 2.0], [1.0, 4.0]]
    error_vectors = error_vectors or [[1.0, 0.0], [0.0, 2.0], [2.0, 2.0]]
    normalized_vectors = normalized_vectors or [[0.5, 0.0], [0.0, 1.0], [1.0, 1.0]]
    slots = []
    for j in range(3):
        slots.append({
            "held_point": j,
            "scored": True,
            "error_px": error_vectors[j],
            "error_normalized": normalized_vectors[j],
            "state": states[j],
            "all_three_state": [0.25, 1.5],
            "bound": not interior[j],
            "interior": interior[j],
            "capture": capture,
            "fixation": fixation,
            "row": row,
            "fold": "gaze_fold",
            "model": "conditional27",
            "split_family": "gaze",
        })
    return {
        "slots": slots,
        "fold": "gaze_fold",
        "model": "conditional27",
        "split_family": "gaze",
        "capture": capture,
        "fixation": fixation,
        "row": row,
        "complete_triple": True,
        "complete_interior": all(interior),
        "E_px": 1.0,
        "G_theta_deg": 2.0,
        "G_A_D": 1.0,
    }


def test_identity_preserves_distinct_captures_with_same_fixation_and_row():
    populations = [_population("cap-a"), _population("cap-b")]
    frames = [_frame("cap-a"), _frame("cap-b")]
    holds = [
        _holdout(j, [float(j), 1.0], capture=capture)
        for capture in ("cap-a", "cap-b") for j in range(3)
    ]

    joined = _join(populations, frames, holds)

    assert len(joined) == 2
    assert {frame["capture"] for frame in joined} == {"cap-a", "cap-b"}
    assert all(frame["scored_count"] == 3 for frame in joined)


@pytest.mark.parametrize("held_point", [3, 0.9])
def test_invalid_held_point_indices_are_rejected(held_point):
    hold = _holdout(held_point, [0.0, 1.0])

    with pytest.raises(ValueError, match="Invalid held_point"):
        _join([_population()], [_frame()], [hold])


def test_same_row_from_wrong_capture_is_an_orphan_holdout():
    hold = _holdout(0, [0.0, 1.0], capture="other-capture")

    with pytest.raises(ValueError, match="Orphan holdout"):
        _join([_population("cap-a")], [_frame("cap-a")], [hold])


def test_missing_frame_does_not_remove_population_scheduled_input_slots():
    joined = _join([_population()], [], [])

    assert len(joined) == 1
    summary = _summarize(joined, expected_frames=1)
    counts = summary["counts"]
    assert counts["scheduled_frames"] == 1
    assert counts["scheduled_slots"] == 3
    assert counts["input_valid_slots"] == 3
    assert counts["scored_slots"] == 0
    assert joined[0]["reason"] == "missing_frame_record"
    assert all(not slot["scored"] for slot in joined[0]["slots"])


def test_invalid_and_missing_slots_keep_the_same_support_schema():
    invalid_population = _population("cap-invalid", p1=False)
    invalid_frame = _frame("cap-invalid")
    missing_frame_population = _population("cap-missing")
    joined = _join([invalid_population, missing_frame_population], [invalid_frame], [])

    support_keys = [set(slot["support"]) for frame in joined for slot in frame["slots"]]
    assert support_keys
    assert all(keys == support_keys[0] for keys in support_keys)
    assert all(slot["support"] is not None for frame in joined for slot in frame["slots"])


def test_certification_uses_branch_matching_the_selected_state():
    selected = [2.5, 1.25]
    branches = [_branch([-7.0, 4.0], certified=False), _branch(selected, certified=True)]
    joined = _join([_population()], [_frame()], [_holdout(j, selected, branch_list=branches)
                                                  for j in range(3)])

    assert all(slot["certified"] for slot in joined[0]["slots"])
    assert all(slot["scored"] for slot in joined[0]["slots"])
    assert all(slot["certification"]["certified"] for slot in joined[0]["slots"])


def test_missing_normalized_error_does_not_zero_or_discard_pixel_complete_score():
    holds = [
        _holdout(0, [0.0, 1.0], error=[3.0, 4.0], normalized=[1.5, 2.0]),
        _holdout(1, [1.0, 2.0], error=[0.0, 2.0], normalized=None),
        _holdout(2, [2.0, 3.0], error=[0.0, 0.0], normalized=[0.0, 0.0]),
    ]
    # Ensure one genuinely missing normalized observation (the helper normally fills it).
    holds[1]["error_normalized"] = None
    joined = _join([_population()], [_frame()], holds)

    frame = joined[0]
    assert frame["complete_triple"]
    assert frame["E_px"] == pytest.approx(math.sqrt((25.0 + 4.0) / 3.0))
    assert frame["E_normalized"] is None
    assert frame["slots"][1]["error_normalized"] is None


def test_subset_state_agreement_survives_unavailable_all_three_estimate():
    frame = _frame(estimated=False)
    frame.update(estimate_kind="all_three", theta=2.0, A=4.0)  # stale finite fields must not count as available
    states = [[0.0, 1.0], [3.0, 2.0], [1.0, 5.0]]
    joined = _join([_population()], [frame], [_holdout(j, states[j]) for j in range(3)])

    result = joined[0]
    assert result["complete_triple"]
    assert not result["all_three_available"]
    assert result["all_three_state"] is None
    assert all(p["all_three_state"] is None for p in result["slots"])
    assert result["G_theta_deg"] == pytest.approx(math.sqrt(14.0 / 3.0))
    assert result["G_A_D"] == pytest.approx(math.sqrt(26.0 / 3.0))
    assert result["subset_minus_all_state"] is None
    assert all(difference is not None for difference in result["pairwise_state_differences"].values())


def test_paired_interior_uses_exact_intersection_for_point_and_frame_masks():
    model27 = [_all_three(interior=(True, True, False))]
    model37 = [_all_three(interior=(True, False, True))]
    model37[0]["model"] = "conditional37"
    for slot in model37[0]["slots"]:
        slot["model"] = "conditional37"

    paired = _paired({"conditional27": model27, "conditional37": model37})

    assert paired["shared_scored_point_count"] == 3
    assert paired["shared_scored_interior_point_count"] == 1
    assert paired["shared_scored_interior_point_ids"] == [
        ["gaze", "gaze_fold", "cap-a", 1, 100, 0]
    ]
    assert paired["shared_complete_frame_count"] == 1
    assert paired["shared_complete_interior_frame_count"] == 0


def test_equal_fixation_state_disagreement_rms_weights_fixations_equally():
    frames = []
    # Ten frame triples in one fixation have G_theta=1; one in another has 10.
    for row in range(10):
        frame = _all_three(fixation=1, row=row)
        frame["G_theta_deg"] = 1.0
        frames.append(frame)
    frame = _all_three(fixation=2, row=20)
    frame["G_theta_deg"] = 10.0
    frames.append(frame)

    got = _metrics_by_group(frames)["complete_frame_G_theta_deg"]

    assert got["pooled"]["rms"] == pytest.approx(math.sqrt(110.0 / 11.0))
    assert got["equal_fixation"]["rms"] == pytest.approx(math.sqrt((1.0 + 100.0) / 2.0))
    assert got["equal_fixation"]["exposure_count"] == 2


def test_perturbing_one_p4_changes_only_checks_that_retain_it():
    import json
    from pathlib import Path

    import numpy as np

    from test_full_position import synthetic_model
    from full_position.geometry import context
    from full_position.invert import predict_holdout, score_holdout

    case = json.loads((Path(__file__).parent / "fixtures/audit_inverse_cases.json").read_text())["cases"][0]
    ctx = context(np.asarray(case["p"], dtype=float))
    model = synthetic_model(37)
    true_state = np.array([3.0, 2.0])
    normalized = model.predict(true_state, ctx.r).reshape(3, 2)
    q = ctx.c + ctx.ell * normalized
    covariance = np.eye(6) * 1e-5

    def predict_and_score(measured_q):
        measured_v = ((measured_q - ctx.c) / ctx.ell).reshape(-1)
        results = []
        for held in range(3):
            retained = np.array([i for i in range(6) if i // 2 != held])
            prediction = predict_holdout(model, ctx, held, measured_v[retained], covariance)
            assert prediction["available"] and prediction["testable"]
            results.append(score_holdout(prediction, measured_q[held], ctx.ell))
        return results

    before = predict_and_score(q)
    changed_q = q.copy()
    changed_q[1] += ctx.ell * np.array([0.02, -0.01])
    after = predict_and_score(changed_q)

    # The P4_2 measurement is excluded from check 1 (held point index 1).
    for key in ("state", "branches", "predictions"):
        assert before[1][key] == after[1][key]
    np.testing.assert_allclose(
        np.asarray(after[1]["error_px"]) - np.asarray(before[1]["error_px"]),
        changed_q[1] - q[1], atol=1e-9,
    )

    # Checks 0 and 2 retain P4_2, so their state estimates can respond to it.
    assert not np.allclose(before[0]["state"], after[0]["state"], atol=1e-5, rtol=0)
    assert not np.allclose(before[2]["state"], after[2]["state"], atol=1e-5, rtol=0)

    def agreement(records):
        slots = [
            {"held_point": j, "scored": record["score_available"],
             "error_px": record["error_px"], "error_normalized": record["error_normalized"],
             "state": record["state"], "all_three_state": true_state.tolist()}
            for j, record in enumerate(records)
        ]
        return _triple_metrics(slots)

    before_g = agreement(before)["G_theta_deg"]
    after_g = agreement(after)["G_theta_deg"]
    assert before_g is not None and after_g is not None
    assert after_g != pytest.approx(before_g, abs=1e-5)
