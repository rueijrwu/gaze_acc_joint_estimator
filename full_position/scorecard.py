"""Authoritative excluded-P4 cross-check scorecard, never a label-fit ranking."""
from __future__ import annotations
from collections import Counter, defaultdict
import json
import numpy as np
from .schema import protocol_metadata

SCHEMA = "three_pair_crosscheck_scorecard_v1"
EXPOSURE = ("fold", "capture", "fixation")
SUPPORT = ("theta_anchor", "A_anchor", "theta_empirical", "A_empirical",
           "P1_context_valid", "context_outside_training_extrema", "p1_parity_seen_in_training")


def exposure(record):
    return tuple(record[k] for k in EXPOSURE)


def unique(records, point=False):
    from .crosscheck import IDENTITY
    out = {}
    for r in records:
        key = tuple(r[k] for k in IDENTITY)+(tuple([r["held_point"]]) if point else ())
        if key in out:
            raise ValueError(f"Duplicate cross-check identity: {key}")
        out[key] = r
    return out


def equal_exposure(records, values, expected):
    groups = defaultdict(list)
    for row, value in zip(records, values):
        if np.isfinite(value):
            groups[exposure(row)].append(float(value))
    absent = sorted(set(expected)-groups.keys())
    mean = float(np.mean([np.mean(v) for v in groups.values()])) if groups else None
    return dict(mean=mean, contributing_exposure_count=len(groups), scheduled_exposure_count=len(expected),
        absent_exposure_ids=absent, status="complete_exposure_support" if not absent else "partial_exposure_support",
        per_exposure={json.dumps(k): dict(count=len(v), mean=float(np.mean(v))) for k, v in sorted(groups.items())})


def build(frames, expected_exposures=None, purpose="development", individual_decoder=True):
    from .crosscheck import stats, vector_metrics
    unique(frames)
    slots = [s for f in frames for s in f["slots"]]
    unique(slots, True)
    expected = sorted(set(exposure(f) for f in frames) | set(tuple(k) for k in (expected_exposures or [])))
    complete = [f for f in frames if f["complete_triple"]]
    scored = [s for s in slots if s["scored"]]
    metrics = {}
    for name, field in (("E_cross_px", "E_px"), ("G_theta_cross_deg", "G_theta_deg"),
                        ("G_A_cross_D", "G_A_D"), ("worst_point_px", "worst_point_px"),
                        ("E_cross_normalized", "E_normalized")):
        values = [f[field] for f in complete if f.get(field) is not None]
        rows = [f for f in complete if f.get(field) is not None]
        eq = equal_exposure(rows, np.square(values), expected)
        metrics[name] = dict(equal_exposure_rms=np.sqrt(eq["mean"]) if eq["mean"] is not None else None,
            squared_error_aggregation=eq, distribution=stats(values))
    axes = {}
    for axis, name in enumerate(("x", "y")):
        values = [s["error_px"][axis] for s in scored]
        squared = equal_exposure(scored, np.square(values), expected)
        bias = equal_exposure(scored, values, expected)
        axes[name] = dict(equal_exposure_rms=np.sqrt(squared["mean"]) if squared["mean"] is not None else None,
            equal_exposure_bias=bias["mean"], signed_distribution=stats(values),
            squared_error_aggregation=squared, bias_aggregation=bias)
    counts = {key: sum(bool(s.get(field)) for s in slots) for key, field in (
        ("input_eligible", "input_valid"), ("certified", "certified"), ("identifiable", "identifiable"),
        ("unambiguous", "unambiguous"), ("scored", "scored"))}
    counts.update(scheduled_frames=len(frames), scheduled_slots=len(slots), complete_triples=len(complete),
                  bound_slots=sum(s.get("bound") is True for s in slots),
                  shared_clipping_frames=sum(f.get("shared_accommodation_clipping", False) for f in frames))
    counts["unambiguous"] = sum(bool(s.get("unambiguous") and s.get("available") and s.get("certified")
        and s.get("identifiable") and s.get("testable")) for s in slots)
    support = {key: dict(Counter("unknown" if s.get("support", {}).get(key) is None else
        "true" if s["support"][key] else "false" for s in slots)) for key in SUPPORT}
    cohort_ids = {"scheduled": [list(k) for k in unique(frames)],
        "complete": [list(k) for k in unique(complete)],
        "complete_interior": [list(k) for k in unique([f for f in complete if f["complete_interior"]])],
        "scored_points": [list(k) for k in unique(scored, True)]}
    return dict(schema=SCHEMA, purpose=purpose, protocol=protocol_metadata(),
        status="available" if individual_decoder else "not_applicable_no_individual_P4_decoder",
        scientific_criterion="excluded-P4 prediction with cross-subset state agreement, coverage, support and tails",
        weighting="equal exposure: mean within-exposure squared values, equal mean across contributing exposures, square root",
        outcomes=metrics, axis_errors=axes, per_point_axis=vector_metrics(scored), coverage=counts,
        support_counts=support, membership=cohort_ids, expected_exposure_ids=expected,
        absent_scheduled_exposure_ids=sorted(set(expected)-set(exposure(f) for f in frames)),
        failure_reasons=dict(Counter(s["failure_reason"] for s in slots if s.get("failure_reason"))),
        promotion=dict(status="no_promotion_decision", reason="development scorecard; no declared selection guards"))


def comparison(reference, candidate, purpose="development"):
    from .audit_followup import transitions
    pair = transitions(reference, candidate)
    return dict(schema=SCHEMA, purpose=purpose, direction="candidate_minus_reference",
        cohort_rule="identical scheduled IDs; exact shared scored points and complete frames",
        paired=pair, reference=build(reference, purpose=purpose), candidate=build(candidate, purpose=purpose))
