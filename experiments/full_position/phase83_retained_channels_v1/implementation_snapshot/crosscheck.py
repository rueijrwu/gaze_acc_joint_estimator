"""Post-process frozen polished holdouts into explicit three-point cross-checks.

This module never evaluates a model. It joins saved frame and holdout records by
their recorded identities and computes descriptive metrics only.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
from collections import defaultdict

import numpy as np

from .schema import write_json, clean_json

VERSION = "exp5_crosscheck_v1"
COORDINATE_MODELS = ("conditional27", "conditional37")
POINTS = range(3)
IDENTITY = ("split_family", "fold", "model", "capture", "fixation", "row")


def _finite_vector(value, n=2):
    try:
        a = np.asarray(value, dtype=float)
        return a.shape == (n,) and bool(np.isfinite(a).all())
    except (TypeError, ValueError):
        return False


def _rms(values):
    a = np.asarray(list(values), float)
    a = a[np.isfinite(a)]
    return None if not len(a) else float(np.sqrt(np.mean(a*a)))


def stats(values):
    """Scalar descriptive statistics; empty input is explicitly n=0/null."""
    a = np.asarray(list(values), float)
    a = a[np.isfinite(a)]
    if not len(a):
        return {"n": 0, "mean": None, "median": None, "rms": None,
                "p90": None, "p95": None}
    return {"n": int(len(a)), "mean": float(a.mean()), "median": float(np.median(a)),
            "rms": float(np.sqrt(np.mean(a*a))), "p90": float(np.percentile(a, 90)),
            "p95": float(np.percentile(a, 95))}


def vector_metrics(points):
    """Metrics for point records with finite ``error_px`` vectors."""
    valid = [p for p in points if p.get("scored") and _finite_vector(p.get("error_px"))]
    out = {"count": len(valid), "vector_norm_px": stats(np.linalg.norm(p["error_px"]) for p in valid)}
    out["axis_px"] = {axis: stats(p["error_px"][i] for p in valid)
                       for i, axis in enumerate(("x", "y"))}
    out["bias_px"] = [float(np.mean([p["error_px"][i] for p in valid])) for i in range(2)] if valid else None
    norm = [p for p in valid if _finite_vector(p.get("error_normalized"))]
    out["vector_norm_normalized"] = stats(np.linalg.norm(p["error_normalized"]) for p in norm)
    out["axis_normalized"] = {axis: stats(p["error_normalized"][i] for p in norm)
                               for i, axis in enumerate(("x", "y"))}
    out["bias_normalized"] = [float(np.mean([p["error_normalized"][i] for p in norm]))
                               for i in range(2)] if norm else None
    out["per_point"] = {str(j): stats(np.linalg.norm(p["error_px"]) for p in valid
                                      if p["held_point"] == j) for j in POINTS}
    out["per_point_axis_px"] = {str(j): {axis: stats(p["error_px"][i] for p in valid
        if p["held_point"] == j) for i, axis in enumerate(("x", "y"))} for j in POINTS}
    out["per_point_axis_normalized"] = {str(j): {axis: stats(p["error_normalized"][i] for p in norm
        if p["held_point"] == j) for i, axis in enumerate(("x", "y"))} for j in POINTS}
    return out


def _equal_fixation_vector_rms(points, field="error_px"):
    groups = defaultdict(list)
    for p in points:
        if p.get("scored") and _finite_vector(p.get(field)):
            key = (p["fold"], p["capture"], p["fixation"])
            groups[key].append(float(np.dot(p[field], p[field])))
    return {"rms": float(np.sqrt(np.mean([np.mean(v) for v in groups.values()]))) if groups else None,
            "exposure_count": len(groups),
            "membership": {json.dumps(k, separators=(",", ":")): {
                "n_points": len(v), "mean_squared_vector_error": float(np.mean(v))}
                for k, v in sorted(groups.items())}}


def _frame_scalar_bundle(frames, field):
    members = [f for f in frames if f.get("complete_triple") and f.get(field) is not None
               and np.isfinite(f[field])]
    groups = defaultdict(list)
    for f in members:
        groups[(f["fold"], f["capture"], f["fixation"])].append(float(f[field]))
    return {"pooled": stats(f[field] for f in members),
            "equal_fixation": {"rms": float(np.sqrt(np.mean([np.mean(np.square(v)) for v in groups.values()]))) if groups else None,
                "exposure_count": len(groups),
                "membership": {json.dumps(k, separators=(",", ":")): len(v) for k, v in sorted(groups.items())}}}


def _point_metric_bundle(points):
    vectors = vector_metrics(points)
    eligible = [p for p in points if p.get("scored") and _finite_vector(p.get("error_px"))]
    normalized = [p for p in points if p.get("scored") and _finite_vector(p.get("error_normalized"))]
    return {"metrics": vectors,
            "pooled_point_vector_rms_px": _rms(np.linalg.norm(p["error_px"]) for p in eligible),
            "equal_fixation_point_vector_rms_px": _equal_fixation_vector_rms(eligible, "error_px"),
            "pooled_point_vector_rms_normalized": _rms(np.linalg.norm(p["error_normalized"]) for p in normalized),
            "equal_fixation_point_vector_rms_normalized": _equal_fixation_vector_rms(normalized, "error_normalized")}


def _triple_metrics(slot_records):
    """Compute frame metrics on precisely three fully eligible slots."""
    scored = [p for p in slot_records if p.get("scored")]
    scored_points = sorted(p["held_point"] for p in scored)
    result = {"scored_count": len(scored), "scored_points": scored_points,
              "complete_triple": len(scored) == 3 and {p["held_point"] for p in scored} == set(POINTS)}
    if scored:
        sq = [float(np.dot(p["error_px"], p["error_px"])) for p in scored]
        result["partial_vector_rms_px"] = float(np.sqrt(np.mean(sq)))
        result["partial_divisor"] = len(scored)
        result["partial_membership"] = scored_points
    else:
        result.update(partial_vector_rms_px=None, partial_divisor=0, partial_membership=[])
    if not result["complete_triple"]:
        result.update(E_px=None, worst_point_px=None, E_normalized=None,
                      G_theta_deg=None, G_theta_arcmin=None, G_A_D=None,
                      theta_range_deg=None, A_range_D=None, pairwise_state_differences=None,
                      subset_minus_all_state=None, complete_interior=False,
                      shared_accommodation_clipping=False)
        return result
    byj = {p["held_point"]: p for p in scored}
    errs = np.asarray([byj[j]["error_px"] for j in POINTS], float)
    en = [byj[j].get("error_normalized") for j in POINTS]
    result["E_px"] = float(np.sqrt(np.mean(np.sum(errs*errs, axis=1))))
    result["worst_point_px"] = float(np.max(np.linalg.norm(errs, axis=1)))
    result["E_normalized"] = (float(np.sqrt(np.mean([np.dot(v, v) for v in en])))
                              if all(_finite_vector(v) for v in en) else None)
    states = [byj[j].get("state") for j in POINTS]
    all_states = [byj[j].get("all_three_state") for j in POINTS]
    pair = {}
    for j in POINTS:
        for k in range(j+1, 3):
            a, b = states[j], states[k]
            pair[f"{j}-{k}"] = ({"theta_deg": float(a[0]-b[0]), "A_D": float(a[1]-b[1])}
                                  if _finite_vector(a) and _finite_vector(b) else None)
    theta = [s[0] for s in states if _finite_vector(s)]
    accom = [s[1] for s in states if _finite_vector(s)]
    theta_diffs = [v["theta_deg"] for v in pair.values() if v is not None]
    accom_diffs = [v["A_D"] for v in pair.values() if v is not None]
    result["pairwise_state_differences"] = pair
    if len(theta_diffs) == len(accom_diffs) == 3:
        result["G_theta_deg"] = float(np.sqrt(np.mean(np.square(theta_diffs))))
        result["G_theta_arcmin"] = result["G_theta_deg"]*60.
        result["G_A_D"] = float(np.sqrt(np.mean(np.square(accom_diffs))))
    else:
        result.update(G_theta_deg=None, G_theta_arcmin=None, G_A_D=None)
    result["theta_range_deg"] = float(max(theta)-min(theta)) if len(theta)==3 else None
    result["A_range_D"] = float(max(accom)-min(accom)) if len(accom)==3 else None
    result["subset_minus_all_state"] = ([list(np.asarray(states[j])-np.asarray(all_states[j])) for j in POINTS]
        if all(_finite_vector(states[j]) and _finite_vector(all_states[j]) for j in POINTS) else None)
    result["complete_interior"] = all(p.get("interior") is True for p in scored)
    result["shared_accommodation_clipping"] = bool(all(p.get("bound") for p in scored)
        and all(abs(float(s[1])-float(states[0][1])) <= 1e-5 for s in states)
        and any(abs(float(states[0][1])-edge) <= 1e-5 for edge in (0., 6.)))
    return result


def _load_json(path):
    return json.loads(Path(path).read_text())


def _hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def _unique_map(records, keyfn, what):
    out = {}
    for r in records:
        key = keyfn(r)
        if key in out:
            raise ValueError(f"Duplicate {what} identity: {key}")
        out[key] = r
    return out


def _held_index(record):
    value = record.get("held_point")
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid held_point value: {value!r}")
    if isinstance(value, bool) or not np.isfinite(numeric) or numeric != int(numeric) or int(numeric) not in POINTS:
        raise ValueError(f"Invalid held_point value: {value!r}; expected stored index 0, 1, or 2")
    return int(numeric)


def join_records(split_family, fold, model, population, frames, holdouts):
    """Join source tables and make exactly three explicit slots per scheduled frame.

    Population rows provide validity for invalid frames, whose holdout file may
    contain no records. Duplicate identities, orphan holdouts, and index/capture
    disagreements are rejected.
    """
    selected = [dict(r) for r in population if _truth(r.get("selected_for_evaluation"))]
    pmap = _unique_map(selected, lambda r: (r["capture"], int(r["fixation"]), int(r["row"])), "population frame")
    fmap = _unique_map(frames, lambda r: (r["capture"], int(r["fixation"]), int(r["row"])), "source frame")
    hmap = _unique_map(holdouts, lambda r: (r["capture"], int(r["fixation"]), int(r["row"]), _held_index(r)), "holdout point")
    for key in hmap:
        if key[:3] not in pmap:
            raise ValueError(f"Orphan holdout record: {key}")
    for key, frame in fmap.items():
        if key not in pmap:
            raise ValueError(f"Source frame missing from population: {key}")
        if frame["capture"] != pmap[key]["capture"]:
            raise ValueError(f"Capture mismatch for source frame {key}")
    results = []
    for key, pop in sorted(pmap.items()):
        fr = fmap.get(key)
        ident = dict(split_family=split_family, fold=fold, model=model,
                     capture=pop["capture"], fixation=key[1], row=key[2],
                     source_frame_index=int(pop["frame"]))
        if fr and (int(fr["frame"]) != int(pop["frame"]) or int(fr["row"]) != key[2]):
            raise ValueError(f"Source frame index mismatch for {key}")
        point_valid = [bool(_truth(pop.get(f"p4_{j+1}_valid"))) for j in POINTS]
        p1_valid = bool(_truth(pop.get("p1_valid_geometry")))
        slots = []
        for j in POINTS:
            h = hmap.get((key[0], key[1], key[2], j))
            if h and (h["capture"] != pop["capture"] or int(h["held_point"]) != j):
                raise ValueError(f"Holdout identity mismatch for {(key, j)}")
            retained_valid = p1_valid and all(point_valid[k] for k in POINTS if k != j)
            slots.append(_slot(ident, j, retained_valid, p1_valid, point_valid[j], fr, h))
        fm = _triple_metrics(slots)
        results.append({**ident, "input_valid": bool(p1_valid and all(point_valid)),
                        "point_validity": point_valid, "slots": slots, **fm,
                        "all_three_available": bool(fr and _truth(fr.get("estimated")) and fr.get("estimate_kind")=="all_three"),
                        "all_three_state": ([fr.get("theta"), fr.get("A")] if fr and _truth(fr.get("estimated")) and fr.get("estimate_kind")=="all_three" and _finite_vector([fr.get("theta"), fr.get("A")]) else None),
                        "nominal_theta": fr.get("nominal_theta") if fr else None,
                        "demand": fr.get("demand") if fr else None,
                        "reason": (fr.get("reason") if fr else "missing_frame_record")})
    return results


def _truth(x):
    if isinstance(x, str):
        return x.strip().lower() in ("true", "1", "yes")
    return bool(x)


def _slot(ident, j, input_valid, p1_valid, held_point_valid, frame, holdout):
    slot = {**ident, "held_point": j, "input_valid": bool(input_valid), "holdout_record_present": holdout is not None,
            "held_point_measurement_valid": bool(held_point_valid),
            "available": False, "certified": False, "identifiable": False, "unambiguous": False,
            "score_available": False, "scored": False, "failure_reason": None,
            "state": None, "all_three_state": None, "error_px": None, "error_normalized": None,
            "bound": None, "interior": None,
            "support": {"theta_anchor": None, "A_anchor": None, "theta_empirical": None,
                "A_empirical": None, "P1_context_valid": bool(p1_valid),
                "context_outside_training_extrema": None, "p1_parity_seen_in_training": None},
            "retained_subset_cost": None, "source_reason": None, "source_termination": None,
            "acceptance_contract": None, "numerical_ties": None, "failed_starts": None,
            "start_count": None, "recovered_starts": None, "plausible_delta": None,
            "nominal_theta": frame.get("nominal_theta") if frame else None,
            "demand": frame.get("demand") if frame else None}
    if (frame and _truth(frame.get("estimated")) and frame.get("estimate_kind")=="all_three"
            and _finite_vector([frame.get("theta"), frame.get("A")])):
        slot["all_three_state"] = [frame["theta"], frame["A"]]
    if not input_valid:
        slot["failure_reason"] = "invalid_input"
    elif holdout is None:
        slot["failure_reason"] = "invalid_held_measurement" if not held_point_valid else "missing_record"
    else:
        branches = holdout.get("branches") or []
        selected_state = holdout.get("state")
        selected_branch = next((b for b in branches if _finite_vector(b.get("state"))
                                and _finite_vector(selected_state)
                                and np.allclose(b["state"], selected_state, atol=1e-10, rtol=0)), None)
        cert = bool(holdout.get("available")) and selected_branch is not None and bool(
            (selected_branch.get("certificate") or {}).get("certified"))
        rank = holdout.get("rank")
        identifiable = bool(holdout.get("available")) and rank == 2
        ambiguous = bool(holdout.get("ambiguous"))
        testable = bool(holdout.get("testable"))
        source_score_available = bool(holdout.get("score_available"))
        eligible = (input_valid and held_point_valid and cert and identifiable and not ambiguous and testable
                    and source_score_available and _finite_vector(holdout.get("error_px"))
                    and _finite_vector(holdout.get("state")))
        # Preserve source evidence even if not eligible for primary scoring.
        slot.update(available=bool(holdout.get("available")), certified=cert,
                    identifiable=identifiable, unambiguous=not ambiguous,
                    score_available=source_score_available, scored=eligible,
                    state=holdout.get("state"), bound=bool(holdout.get("at_bound")),
                    interior=bool(holdout.get("available")) and not bool(holdout.get("at_bound")),
                    plausible_branches=holdout.get("plausible_branches", []),
                    branches=branches, certification=({k: selected_branch.get("certificate", {}).get(k)
                        for k in ("certified", "reason", "stationarity_encoded", "physical_correction", "stable_cost", "local_minimum")}
                        if selected_branch else None),
                    rank=rank, singular_values_physical=holdout.get("singular_values_physical"),
                    condition_number=(float(holdout["singular_values_physical"][0]/holdout["singular_values_physical"][-1])
                        if rank == 2 and holdout.get("singular_values_physical") and holdout["singular_values_physical"][-1] else None),
                    condition_number_status=("finite" if rank == 2 and holdout.get("singular_values_physical")
                        and holdout["singular_values_physical"][-1] else "unavailable_rank_weak_or_zero_singular_value"),
                    testable=testable, score_reason=holdout.get("score_reason"),
                    retained_subset_cost=holdout.get("cost"),
                    source_reason=holdout.get("reason"),
                    source_termination=holdout.get("termination", holdout.get("termination_reason")),
                    acceptance_contract=holdout.get("acceptance_contract"),
                    numerical_ties=holdout.get("numerical_ties"),
                    failed_starts=holdout.get("failed_starts"), start_count=holdout.get("start_count"),
                    recovered_starts=holdout.get("recovered_starts"), plausible_delta=holdout.get("plausible_delta"),
                    error_px=holdout.get("error_px") if _finite_vector(holdout.get("error_px")) else None,
                    error_normalized=holdout.get("error_normalized") if _finite_vector(holdout.get("error_normalized")) else None,
                    selected_prediction=(holdout.get("predictions") or [{}])[0],
                    prediction_branches=holdout.get("predictions", []),
                    state_minus_all=holdout.get("state_difference_from_all"),
                    support={"theta_anchor": _axis_support(holdout.get("state"), 0, holdout.get("anchor_range")),
                             "A_anchor": _axis_support(holdout.get("state"), 1, holdout.get("anchor_range")),
                             "theta_empirical": _axis_support(holdout.get("state"), 0, holdout.get("training_range")),
                             "A_empirical": _axis_support(holdout.get("state"), 1, holdout.get("training_range")),
                             "P1_context_valid": bool(p1_valid)},
                    nominal_theta=frame.get("nominal_theta") if frame else None,
                    demand=frame.get("demand") if frame else None)
        if not cert:
            slot["failure_reason"] = holdout.get("reason", "uncertified")
        elif not identifiable:
            slot["failure_reason"] = "rank_below_2"
        elif ambiguous or not testable:
            slot["failure_reason"] = holdout.get("score_reason", "ambiguous_or_untestable")
        elif not held_point_valid:
            slot["failure_reason"] = "invalid_held_measurement"
        elif not source_score_available:
            slot["failure_reason"] = holdout.get("score_reason", "score_unavailable")
        elif not eligible:
            slot["failure_reason"] = "inconsistent_score_record_missing_finite_state_or_pixel_error"
    return slot


def _axis_support(state, axis, bounds):
    if not _finite_vector(state) or not bounds or len(bounds) != 2:
        return None
    return bool(bounds[0][axis] <= state[axis] <= bounds[1][axis])


def _read_population(path):
    with gzip.open(path, "rt", newline="") as f:
        return list(csv.DictReader(f))


def _read_holdouts(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _metrics_by_group(frames):
    slots = [p for f in frames for p in f["slots"]]
    by = {}
    for name, members in (("point", lambda p: str(p["held_point"])),
                          ("evaluation_exposure", lambda p: json.dumps([p["fold"], p["capture"], p["fixation"]], separators=(",", ":"))),
                          ("nominal_gaze", lambda p: str(p.get("nominal_theta"))),
                          ("capture", lambda p: str(p["capture"]))):
        groups = defaultdict(list)
        for p in slots:
            if p["scored"]:
                groups[members(p)].append(p)
        by[name] = {k: vector_metrics(v) for k, v in sorted(groups.items())}
    point_bundle = _point_metric_bundle(slots)
    return {"points": vector_metrics(slots), "by": by,
            **{k: v for k, v in point_bundle.items() if k != "metrics"},
            "complete_frame_E_px": _frame_scalar_bundle(frames, "E_px"),
            "complete_frame_E_normalized": _frame_scalar_bundle(frames, "E_normalized"),
            "complete_frame_worst_point_px": _frame_scalar_bundle(frames, "worst_point_px"),
            "complete_frame_G_theta_deg": _frame_scalar_bundle(frames, "G_theta_deg"),
            "complete_frame_G_A_D": _frame_scalar_bundle(frames, "G_A_D"),
            "complete_frame_theta_range_deg": _frame_scalar_bundle(frames, "theta_range_deg"),
            "complete_frame_A_range_D": _frame_scalar_bundle(frames, "A_range_D")}


def _summarize(frames, expected_frames=160):
    slots = [p for f in frames for p in f["slots"]]
    counts = {"scheduled_frames": len(frames), "expected_scheduled_frames": expected_frames,
              "scheduled_slots": len(slots), "expected_scheduled_slots": expected_frames*3,
              "input_valid_slots": sum(p["input_valid"] for p in slots),
              "available_slots": sum(p["available"] for p in slots),
              "certified_slots": sum(p["certified"] for p in slots),
              "identifiable_slots": sum(p["identifiable"] for p in slots),
              "unambiguous_slots": sum(p["unambiguous"] and p["available"] and p["certified"]
                                       and p["identifiable"] and p.get("testable", False) for p in slots),
              "scored_slots": sum(p["scored"] for p in slots),
              "boundary_slots": sum(p.get("bound") is True for p in slots),
              "shared_accommodation_clipping_frames": sum(f.get("shared_accommodation_clipping", False) for f in frames),
              "complete_triple_frames": sum(f["complete_triple"] for f in frames),
              "all_three_available_frames": sum(f["all_three_available"] for f in frames),
              "all_three_input_valid_frames": sum(f["input_valid"] for f in frames)}
    failures = defaultdict(int)
    for p in slots:
        if p.get("failure_reason"):
            failures[p["failure_reason"]] += 1
    # Membership masks make all-testable and interior comparisons auditable.
    masks = {"all_testable_slot_ids": [_slot_id(p) for p in slots if p["scored"]],
             "interior_slot_ids": [_slot_id(p) for p in slots if p["scored"] and p["interior"]],
             "complete_frame_ids": [_frame_id(f) for f in frames if f["complete_triple"]],
             "complete_interior_frame_ids": [_frame_id(f) for f in frames if f["complete_triple"] and f["complete_interior"]]}
    stats_all = _metrics_by_group(frames)
    interior = [p for p in slots if p["scored"] and p["interior"]]
    interior_frames = [f for f in frames if f["complete_triple"] and f["complete_interior"]]
    boundary = [p for p in slots if p["scored"] and p.get("bound") is True]
    boundary_frames = [f for f in frames if f["complete_triple"]
                       and any(p.get("bound") is True for p in f["slots"])]
    interior_point_bundle = _point_metric_bundle(interior)
    boundary_point_bundle = _point_metric_bundle(boundary)
    return {"counts": counts, "failure_reasons": dict(sorted(failures.items())),
            "membership": masks, "all_testable": stats_all,
            "interior_only": {"points": vector_metrics(interior),
                **{k: v for k, v in interior_point_bundle.items() if k != "metrics"},
                "complete_frame_E_px": _frame_scalar_bundle(interior_frames, "E_px"),
                "complete_frame_E_normalized": _frame_scalar_bundle(interior_frames, "E_normalized"),
                "complete_frame_worst_point_px": _frame_scalar_bundle(interior_frames, "worst_point_px"),
                "complete_frame_G_theta_deg": _frame_scalar_bundle(interior_frames, "G_theta_deg"),
                "complete_frame_G_A_D": _frame_scalar_bundle(interior_frames, "G_A_D"),
                "complete_frame_theta_range_deg": _frame_scalar_bundle(interior_frames, "theta_range_deg"),
                "complete_frame_A_range_D": _frame_scalar_bundle(interior_frames, "A_range_D"),
                "complete_frame_ids": [_frame_id(f) for f in interior_frames]},
            "boundary_only": {"point_count": len(boundary), "point_ids": [_slot_id(p) for p in boundary],
                "points": vector_metrics(boundary),
                **{k: v for k, v in boundary_point_bundle.items() if k != "metrics"},
                "complete_triple_frame_count": len(boundary_frames),
                "complete_triple_frame_ids": [_frame_id(f) for f in boundary_frames],
                "complete_frame_E_px": _frame_scalar_bundle(boundary_frames, "E_px"),
                "complete_frame_E_normalized": _frame_scalar_bundle(boundary_frames, "E_normalized"),
                "complete_frame_worst_point_px": _frame_scalar_bundle(boundary_frames, "worst_point_px"),
                "complete_frame_G_theta_deg": _frame_scalar_bundle(boundary_frames, "G_theta_deg"),
                "complete_frame_G_A_D": _frame_scalar_bundle(boundary_frames, "G_A_D"),
                "complete_frame_theta_range_deg": _frame_scalar_bundle(boundary_frames, "theta_range_deg"),
                "complete_frame_A_range_D": _frame_scalar_bundle(boundary_frames, "A_range_D")},
            "two_channel13_cross_prediction": "unavailable_no_individual_P4_prediction"}


def _slot_id(p):
    return [p[k] for k in IDENTITY] + [p["held_point"]]


def _frame_id(f):
    return [f[k] for k in IDENTITY]


def _paired(model_frames):
    """Exact 27/37 shared point and complete-frame intersections."""
    d = {m: model_frames[m] for m in COORDINATE_MODELS}
    keys = {m: {_tuple([p[k] for k in IDENTITY if k != "model"]+[p["held_point"]]): p
                for f in d[m] for p in f["slots"]} for m in COORDINATE_MODELS}
    common_points = sorted(keys[COORDINATE_MODELS[0]].keys() & keys[COORDINATE_MODELS[1]].keys())
    common_points = [k for k in common_points if keys[COORDINATE_MODELS[0]][k]["scored"] and keys[COORDINATE_MODELS[1]][k]["scored"]]
    out = {"full_population_coverage": {m: {"scheduled_frames": len(d[m]),
                "scheduled_slots": 3*len(d[m])} for m in COORDINATE_MODELS},
           "shared_scored_point_ids": [list(k) for k in common_points], "shared_scored_point_count": len(common_points)}
    for model in COORDINATE_MODELS:
        out[model] = _point_metric_bundle([keys[model][k] for k in common_points])
    common_interior_points = [k for k in common_points if all(keys[m][k]["interior"] for m in COORDINATE_MODELS)]
    out["shared_scored_interior_point_ids"] = [list(k) for k in common_interior_points]
    out["shared_scored_interior_point_count"] = len(common_interior_points)
    for model in COORDINATE_MODELS:
        out[model+"_interior_points"] = _point_metric_bundle([keys[model][k] for k in common_interior_points])
    fmaps = {m: {_tuple([f[k] for k in IDENTITY if k != "model"]): f for f in d[m]} for m in COORDINATE_MODELS}
    common = sorted(fmaps[COORDINATE_MODELS[0]].keys() & fmaps[COORDINATE_MODELS[1]].keys())
    common = [k for k in common if fmaps[COORDINATE_MODELS[0]][k]["complete_triple"] and fmaps[COORDINATE_MODELS[1]][k]["complete_triple"]]
    out["shared_complete_frame_ids"] = [list(k) for k in common]
    out["shared_complete_frame_count"] = len(common)
    metric_fields = ("E_px", "E_normalized", "worst_point_px", "G_theta_deg", "G_A_D",
                     "theta_range_deg", "A_range_D")
    for model in COORDINATE_MODELS:
        common_frames = [fmaps[model][k] for k in common]
        for field in metric_fields:
            out[model+"_shared_complete_"+field] = _frame_scalar_bundle(common_frames, field)
    interior = [k for k in common if all(fmaps[m][k]["complete_interior"] for m in COORDINATE_MODELS)]
    out["shared_complete_interior_frame_ids"] = [list(k) for k in interior]
    out["shared_complete_interior_frame_count"] = len(interior)
    for model in COORDINATE_MODELS:
        interior_frames = [fmaps[model][k] for k in interior]
        for field in metric_fields:
            out[model+"_shared_complete_interior_"+field] = _frame_scalar_bundle(interior_frames, field)
    return out


def _tuple(value):
    return tuple(tuple(x) if isinstance(x, list) else x for x in value)


def _frame_csv_row(frame):
    fields = {k: frame.get(k) for k in IDENTITY}
    fields.update(source_frame_index=frame["source_frame_index"], input_valid=frame["input_valid"],
        point_validity=json.dumps(frame["point_validity"]), scheduled_points=3,
        scored_points=frame["scored_count"], scored_membership=json.dumps(frame["scored_points"]),
        partial_vector_rms_px=frame["partial_vector_rms_px"], partial_divisor=frame["partial_divisor"],
        complete_triple=frame["complete_triple"], E_px=frame["E_px"], worst_point_px=frame["worst_point_px"],
        E_normalized=frame["E_normalized"], G_theta_deg=frame["G_theta_deg"],
        G_theta_arcmin=frame["G_theta_arcmin"], G_A_D=frame["G_A_D"],
        theta_range_deg=frame["theta_range_deg"], A_range_D=frame["A_range_D"],
        pairwise_state_differences=json.dumps(frame["pairwise_state_differences"]),
        subset_minus_all_state=json.dumps(frame["subset_minus_all_state"]),
        shared_accommodation_clipping=frame["shared_accommodation_clipping"],
        all_three_available=frame["all_three_available"], all_three_state=json.dumps(frame["all_three_state"]),
        complete_interior=frame["complete_interior"], nominal_theta=frame["nominal_theta"],
        demand=frame["demand"], reason=frame["reason"])
    return fields


def _source_split_dirs(source):
    return sorted(p for p in Path(source).iterdir() if p.is_dir() and (p/"population.csv.gz").exists())


def run(source, output):
    """Create a fresh ``crosscheck_v1`` artifact directory from frozen records."""
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(f"Output directory must be fresh: {output}")
    if not source.is_dir():
        raise FileNotFoundError(source)
    output.mkdir(parents=True, exist_ok=False)
    all_frames = {}
    provenance = {}
    read_hashes = {}
    for split in _source_split_dirs(source):
        family = split.name.split("_")[0]
        population_path = split/"population.csv.gz"
        read_hashes[str(population_path)] = _hash(population_path)
        population = _read_population(population_path)
        provenance[split.name] = {"population_sha256": read_hashes[str(population_path)]}
        family_frames = {}
        for model in COORDINATE_MODELS:
            fold_dir = split/model
            if not fold_dir.exists():
                raise FileNotFoundError(f"Missing coordinate model output: {fold_dir}")
            frame_path, holdout_path, model_path = fold_dir/"frames.json", fold_dir/"holdouts.jsonl", fold_dir/"model.json"
            for path in (frame_path, holdout_path, model_path, fold_dir/"training_states.json"):
                read_hashes[str(path)] = _hash(path)
            frames = _load_json(frame_path)
            holdouts = _read_holdouts(holdout_path)
            model_meta = _load_json(model_path)
            uncertainty_path = fold_dir/"frame_uncertainty.json"
            if uncertainty_path.exists():
                read_hashes[str(uncertainty_path)] = _hash(uncertainty_path)
            uncertainty = _load_json(uncertainty_path) if uncertainty_path.exists() else {}
            uncertainty_map = _unique_map(uncertainty.get("records", []),
                lambda r: (int(r["fixation"]), int(r["row"]), r["capture"]), "frame uncertainty")
            anchor = model_meta.get("provenance", {}).get("anchor_range")
            training = _load_json(fold_dir/"training_states.json")
            train_states = np.asarray(training.get("states", []), float)
            training_range = ([np.min(train_states, axis=0).tolist(), np.max(train_states, axis=0).tolist()]
                              if train_states.ndim == 2 and train_states.shape[1] == 2 and len(train_states) else None)
            # Sources intentionally remain byte-for-byte untouched. Anchor range
            # support is attached as post-process metadata to states below.
            joined = join_records(family, split.name, model, population, frames, holdouts)
            for fr in joined:
                ui = uncertainty_map.get((fr["fixation"], fr["row"], fr["capture"]), {})
                for slot in fr["slots"]:
                    slot["support"]["context_outside_training_extrema"] = ui.get("context_outside_training_extrema")
                    slot["support"]["p1_parity_seen_in_training"] = ui.get("p1_parity_seen_in_training")
                    if slot.get("state"):
                        slot["support"]["theta_anchor"] = _axis_support(slot["state"], 0, anchor)
                        slot["support"]["A_anchor"] = _axis_support(slot["state"], 1, anchor)
                        tr = training_range
                        slot["support"]["theta_empirical"] = _axis_support(slot["state"], 0, tr)
                        slot["support"]["A_empirical"] = _axis_support(slot["state"], 1, tr)
                family_frames[model] = family_frames.get(model, []) + [fr]
            provenance[split.name][model] = {"frames_sha256": read_hashes[str(frame_path)],
                "holdouts_sha256": read_hashes[str(holdout_path)], "model_sha256": read_hashes[str(model_path)],
                "training_states_sha256": read_hashes[str(fold_dir/"training_states.json")],
                "frame_uncertainty_sha256": read_hashes.get(str(uncertainty_path))}
        all_frames[split.name] = family_frames
    flat = [f for fold in all_frames.values() for model in COORDINATE_MODELS for f in fold[model]]
    csv_rows = [_frame_csv_row(f) for f in flat]
    if csv_rows:
        with (output/"crosscheck_frames.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0]), lineterminator="\n")
            writer.writeheader(); writer.writerows(csv_rows)
    with gzip.open(output/"crosscheck_points.jsonl.gz", "wt") as f:
        for frame in flat:
            for slot in frame["slots"]:
                f.write(json.dumps(clean_json(slot), allow_nan=False, separators=(",", ":"))+"\n")
    summaries = {}
    families = defaultdict(lambda: {m: [] for m in COORDINATE_MODELS})
    for split, models in all_frames.items():
        summaries[split] = {"models": {m: _summarize(models[m], len(models[m])) for m in COORDINATE_MODELS}}
        summaries[split]["paired_27_37"] = _paired(models)
        family = split.split("_")[0]
        for model in COORDINATE_MODELS:
            families[family][model].extend(models[model])
    family_summaries = {}
    for family, models in families.items():
        family_summaries[family] = {"models": {m: _summarize(models[m], 160) for m in COORDINATE_MODELS},
                                    "paired_27_37": _paired(models)}
    _plot_agreement(output, [f for models in all_frames.values() for m in COORDINATE_MODELS for f in models[m]])
    write_json(output/"crosscheck_summary.json", {"schema": VERSION, "source": str(source),
        "source_provenance": provenance,
        "scheduled_expectation_per_split_family": {"frames": 160, "slots": 480},
        "fold_scheduled_expectation": "actual selected population row count per fold, with three slots per row",
        "metric_conventions": {"point_rms": "RMS of 2D vector norms; not scalar-coordinate RMS",
            "equal_fixation": "sqrt(mean over fixations of within-fixation mean squared point-vector error)",
            "state_disagreement": "raw physical units; no independent covariance sum or tolerance",
            "condition_number": "ratio of physical singular values for reference increments of 1 degree and 1 diopter; diagnostic only",
            "shared_clipping": "descriptive computational-bound flag using saved 1e-5 physical bound tolerance",
            "pairwise_state_difference": "theta_deg and A_D; value is state at lower held_point index minus higher index",
            "comparison": "exact shared point/frame IDs; membership IDs retained"},
        "two_channel13_cross_prediction": "unavailable_no_individual_P4_prediction",
        "splits": summaries, "split_families": family_summaries})
    changed = [path for path, digest in read_hashes.items() if _hash(path) != digest]
    if changed:
        raise RuntimeError(f"Frozen source artifacts changed during cross-check processing: {changed}")
    return output


def _plot_agreement(output, frames):
    """Plot complete-frame coordinate and raw state disagreement diagnostics."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True, sharex="col")
    colors = {"conditional27": "#2674a8", "conditional37": "#db7a29"}
    markers = {True: "o", False: "x"}
    for ri, family in enumerate(("gaze", "capture")):
        for model in COORDINATE_MODELS:
            points = [f for f in frames if f["split_family"] == family and f["model"] == model and f["complete_triple"]]
            for interior in (True, False):
                selected = [f for f in points if f["complete_interior"] == interior]
                if not selected:
                    continue
                axes[ri, 0].scatter([f["E_px"] for f in selected], [f["G_theta_deg"] for f in selected],
                    c=colors[model], marker=markers[interior], alpha=.65, label=f"{model}, {'interior' if interior else 'boundary'}")
                axes[ri, 1].scatter([f["E_px"] for f in selected], [f["G_A_D"] for f in selected],
                    c=colors[model], marker=markers[interior], alpha=.65, label=f"{model}, {'interior' if interior else 'boundary'}")
        axes[ri, 0].set_ylabel(f"{family} folds: Gθ (degrees)")
        axes[ri, 1].set_ylabel(f"{family} folds: GA (diopters)")
        for ax in axes[ri]:
            ax.grid(alpha=.2)
            ax.legend(fontsize=7)
    axes[1, 0].set_xlabel("Complete-frame E (pixels)")
    axes[1, 1].set_xlabel("Complete-frame E (pixels)")
    fig.savefig(Path(output)/"crosscheck_agreement.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.source, args.output)


if __name__ == "__main__":
    main()
