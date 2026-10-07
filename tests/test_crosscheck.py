import math

import numpy as np
import pytest

from full_position.crosscheck import (
    _paired, _summarize, _triple_metrics, _slot, join_records, vector_metrics,
)


def _slots(errors=None, states=None, all_state=(1.0, 2.0), interior=True):
    errors = errors or [[3., 4.], [0., 2.], [0., 0.]]
    states = states or [[0., 1.], [3., 3.], [0., 4.]]
    out = []
    for j in range(3):
        out.append(dict(held_point=j, scored=True, error_px=errors[j],
            error_normalized=[x/2 for x in errors[j]], state=states[j],
            all_three_state=all_state, bound=not interior, interior=interior))
    return out


def test_hand_computed_complete_triple_metrics_and_unit_conversion():
    got = _triple_metrics(_slots())
    assert got["complete_triple"]
    assert got["E_px"] == pytest.approx(math.sqrt(29/3))
    assert got["worst_point_px"] == pytest.approx(5.)
    assert got["E_normalized"] == pytest.approx(math.sqrt(29/12))
    assert got["G_theta_deg"] == pytest.approx(math.sqrt(6))
    assert got["G_theta_arcmin"] == pytest.approx(60*math.sqrt(6))
    assert got["G_A_D"] == pytest.approx(math.sqrt(14/3))
    assert got["theta_range_deg"] == 3.
    assert got["A_range_D"] == 3.
    assert got["pairwise_state_differences"]["0-1"] == {"theta_deg": -3., "A_D": -2.}
    assert got["subset_minus_all_state"][0] == [-1., -1.]


def test_shuffled_point_order_does_not_change_metrics():
    source = _triple_metrics(_slots())
    shuffled = _triple_metrics([_slots()[2], _slots()[0], _slots()[1]])
    assert source == shuffled


def test_missing_slot_stays_partial_and_never_becomes_zero_error():
    slots = _slots()
    slots[1]["scored"] = False
    got = _triple_metrics(slots)
    assert not got["complete_triple"]
    assert got["E_px"] is None
    assert got["partial_divisor"] == 2
    assert got["partial_membership"] == [0, 2]
    assert got["partial_vector_rms_px"] == pytest.approx(math.sqrt(25/2))


def test_point_vector_rms_is_not_scalar_coordinate_rms():
    points = [dict(scored=True, error_px=[3., 4.], error_normalized=None, held_point=0)]
    got = vector_metrics(points)
    assert got["vector_norm_px"]["rms"] == 5.
    assert math.sqrt((3**2+4**2)/2) != got["vector_norm_px"]["rms"]
    assert got["vector_norm_normalized"]["n"] == 0


def test_equal_fixation_weighting_differs_from_pooled_weighting():
    frames = []
    # Exposure one contributes ten errors of magnitude 1; exposure two one of 10.
    for j in range(10):
        frames.append(dict(fold="f1", capture="c1", fixation=1, nominal_theta=0,
            split_family="gaze", model="conditional27", row=j, input_valid=True,
            all_three_available=False, all_three_input_valid_frames=False,
            slots=[dict(scored=True, held_point=0, error_px=[1., 0.], error_normalized=[1., 0.],
                input_valid=True, available=True, certified=True, identifiable=True, unambiguous=True,
                interior=True, nominal_theta=0., capture="c1", fixation=1, row=j, fold="f1", split_family="gaze", model="conditional27")],
            complete_triple=False))
    frames.append(dict(fold="f2", capture="c2", fixation=2, nominal_theta=5,
        split_family="gaze", model="conditional27", row=0, input_valid=True,
        all_three_available=False, all_three_input_valid_frames=False,
        slots=[dict(scored=True, held_point=0, error_px=[10., 0.], error_normalized=[10., 0.],
            input_valid=True, available=True, certified=True, identifiable=True, unambiguous=True,
            interior=True, nominal_theta=5., capture="c2", fixation=2, row=0, fold="f2", split_family="gaze", model="conditional27")],
        complete_triple=False))
    summary = _summarize(frames, expected_frames=11)["all_testable"]
    assert summary["pooled_point_vector_rms_px"] == pytest.approx(math.sqrt(110/11))
    assert summary["equal_fixation_point_vector_rms_px"]["rms"] == pytest.approx(math.sqrt((1+100)/2))
    assert len(summary["equal_fixation_point_vector_rms_px"]["membership"]) == 2


def _holdout(j, state, *, ambiguous=False, bound=False, score=True):
    return {"available": True, "reason": "ok", "state": state, "held_point": j,
        "fixation": 1, "row": 100, "capture": "cap",
        "branches": [{"state": state, "certificate": {"certified": True, "reason": "certified_stationary_minimum"}}],
        "plausible_branches": [{"state": state}], "rank": 2,
        "singular_values_physical": [2., 1.], "ambiguous": ambiguous,
        "testable": not ambiguous, "score_available": score,
        "error_px": [float(j), 1.], "error_normalized": None,
        "at_bound": bound, "predictions": [], "score_reason": None}


def _frame(fix=1, row=100, capture="cap"):
    return {"fixation": fix, "row": row, "frame": row, "capture": capture,
        "baseline_valid": True, "point_valid": [True]*3, "estimated": True,
        "estimate_kind": "all_three", "theta": 1., "A": 2.,
        "nominal_theta": 0., "demand": 2.}


def _population(fix=1, row=100, capture="cap"):
    return {"fixation": str(fix), "row": str(row), "frame": str(row), "capture": capture,
        "selected_for_evaluation": "True", "p1_valid_geometry": "True",
        "p4_1_valid": "True", "p4_2_valid": "True", "p4_3_valid": "True",
        "baseline_valid": "True"}


def test_join_preserves_ambiguous_and_boundary_slots_without_scoring_ambiguous():
    holdouts = [_holdout(0, [0., 1.]), _holdout(1, [1., 2.], ambiguous=True),
                _holdout(2, [2., 3.], bound=True)]
    frames = join_records("gaze", "gaze_0", "conditional27", [_population()], [_frame()], holdouts)
    assert len(frames) == 1 and len(frames[0]["slots"]) == 3
    assert [s["scored"] for s in frames[0]["slots"]] == [True, False, True]
    assert frames[0]["slots"][2]["bound"]
    assert not frames[0]["slots"][2]["interior"]
    assert frames[0]["E_px"] is None


def test_join_rejects_duplicate_frame_point_and_orphan_records():
    pop = [_population()]
    frame = [_frame()]
    with pytest.raises(ValueError, match="Duplicate source frame"):
        join_records("gaze", "gaze_0", "conditional27", pop, frame*2, [])
    holds = [_holdout(0, [0., 1.])]
    with pytest.raises(ValueError, match="Duplicate holdout"):
        join_records("gaze", "gaze_0", "conditional27", pop, frame, holds*2)
    with pytest.raises(ValueError, match="Orphan holdout"):
        join_records("gaze", "gaze_0", "conditional27", [], [], holds)


def test_join_creates_three_invalid_slots_and_keeps_missing_reasons():
    pop = _population()
    pop["p1_valid_geometry"] = "False"
    rows = join_records("gaze", "gaze_0", "conditional27", [pop], [_frame()], [])
    assert len(rows[0]["slots"]) == 3
    assert all(s["failure_reason"] == "invalid_input" for s in rows[0]["slots"])
    assert all(not s["scored"] for s in rows[0]["slots"])


def test_paired_comparison_uses_identical_ids_and_shared_interior_intersection():
    def make(model, bound=False):
        f = {"split_family": "capture", "fold": "capture_1", "model": model,
             "capture": "c1", "fixation": 1, "row": 1, "complete_triple": True,
             "complete_interior": not bound, "E_px": 2.}
        ss = _slots(interior=not bound)
        for p in ss:
            p.update({"split_family": "capture", "fold": "capture_1", "model": model,
                      "capture": "c1", "fixation": 1, "row": 1, "scored": True})
        f["slots"] = ss
        return [f]
    result = _paired({"conditional27": make("conditional27"),
                      "conditional37": make("conditional37", bound=True)})
    assert result["shared_scored_point_count"] == 3
    assert result["shared_scored_interior_point_count"] == 0
    assert result["shared_complete_frame_count"] == 1
    assert result["shared_complete_interior_frame_count"] == 0
