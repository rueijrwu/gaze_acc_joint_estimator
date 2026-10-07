"""Nested grouped cross-check selection with explicit, externally declared guards.

The coordinator passes restricted group dictionaries to fitting/evaluation.
It cannot retroactively turn development folds into independent final data.
"""
from __future__ import annotations
import numpy as np
from .scorecard import build

REQUIRED_GUARDS = ("minimum_scored_fraction", "minimum_complete_fraction", "maximum_G_theta_deg",
                   "maximum_G_A_D", "maximum_worst_point_px", "maximum_x_axis_rms_px",
                   "maximum_y_axis_rms_px", "minimum_known_support_fraction")
REQUIRED_GUARDS += ("maximum_p95_E_px", "maximum_p95_worst_point_px",
                    "maximum_unsupported_fraction", "maximum_bound_slot_fraction")


def choose(inner_records, guards=None, reference=None):
    """No nominal labels, training costs, outer predictions or outer scores enter."""
    if not guards or any(k not in guards for k in REQUIRED_GUARDS) or not guards.get("provenance"):
        return dict(status="no_promotion_decision", selected=None, reference=reference,
                    reason="complete predeclared cross-check guards and their provenance are required")
    if any(not np.isfinite(guards[k]) or guards[k] < 0 for k in REQUIRED_GUARDS):
        raise ValueError("Selection guards must be finite nonnegative values")
    if any(guards[k] > 1 for k in REQUIRED_GUARDS if "fraction" in k):
        raise ValueError("Fraction guards must be in [0,1]")
    cards = {name: build(rows, purpose="inner_validation") for name, rows in inner_records.items()}
    eligible, rejected = [], {}
    for name, card in cards.items():
        n = card["coverage"]["scheduled_slots"]
        nf = card["coverage"]["scheduled_frames"]
        checks = {"scored_fraction": n and card["coverage"]["scored"]/n >= guards["minimum_scored_fraction"],
                  "complete_fraction": nf and card["coverage"]["complete_triples"]/nf >= guards["minimum_complete_fraction"]}
        for outcome, guard in (("G_theta_cross_deg", "maximum_G_theta_deg"), ("G_A_cross_D", "maximum_G_A_D"),
                               ("worst_point_px", "maximum_worst_point_px")):
            value = card["outcomes"][outcome]["equal_exposure_rms"]
            checks[outcome] = value is not None and value <= guards[guard]
        for axis in ("x", "y"):
            value = card["axis_errors"][axis]["equal_exposure_rms"]
            checks[axis+"_axis"] = value is not None and value <= guards[f"maximum_{axis}_axis_rms_px"]
        for name_out, guard in (("E_cross_px", "maximum_p95_E_px"), ("worst_point_px", "maximum_p95_worst_point_px")):
            value = card["outcomes"][name_out]["distribution"]["p95"]
            checks[name_out+"_tail"] = value is not None and value <= guards[guard]
        known = all(n and (n-v.get("unknown", 0))/n >= guards["minimum_known_support_fraction"]
                    for v in card["support_counts"].values())
        checks["support_known"] = known
        slots = [s for f in inner_records[name] for s in f["slots"]]
        def unsupported(slot):
            s = slot.get("support", {})
            return (any(s.get(k) is not True for k in ("theta_anchor", "A_anchor", "theta_empirical",
                "A_empirical", "P1_context_valid", "p1_parity_seen_in_training")) or
                s.get("context_outside_training_extrema") is not False)
        checks["training_support"] = bool(n and sum(unsupported(s) for s in slots)/n <= guards["maximum_unsupported_fraction"])
        checks["bounds"] = bool(n and card["coverage"]["bound_slots"]/n <= guards["maximum_bound_slot_fraction"])
        checks["all_scheduled_exposures_contribute"] = not card["outcomes"]["E_cross_px"]["squared_error_aggregation"]["absent_exposure_ids"]
        if all(checks.values()):
            eligible.append(name)
        else:
            rejected[name] = [key for key, value in checks.items() if not value]
    if not eligible:
        return dict(status="no_promotion_decision", selected=None, reference=reference,
                    reason="no candidate satisfies declared guards", rejected=rejected)
    # Selection compares exactly shared complete frames. Candidate labels do not
    # change the frame identity, and surviving-only single-candidate means cannot rank.
    def key(f):
        return tuple(f[k] for k in ("fold", "capture", "fixation", "row"))
    tables = {n: {key(f): f for f in inner_records[n] if f["complete_triple"]} for n in eligible}
    if any(len(tables[n]) != sum(f["complete_triple"] for f in inner_records[n]) for n in eligible):
        raise ValueError("Duplicate scientific frame IDs inside an inner candidate")
    shared = set.intersection(*(set(t) for t in tables.values()))
    if not shared:
        return dict(status="no_promotion_decision", selected=None, reference=reference, reason="no shared complete cohort")
    losses = {}
    for name, table in tables.items():
        card = build([table[k] for k in sorted(shared)], purpose="inner_validation")
        losses[name] = card["outcomes"]["E_cross_px"]["squared_error_aggregation"]["mean"]
    selected = min(eligible, key=lambda n: (losses[n], n != reference, n))
    return dict(status="inner_selection", selected=selected, reference=reference, shared_frame_ids=sorted(shared),
                paired_inner_L_cross=losses, rejected=rejected, guards=dict(guards))


def nested_grouped(group_data, outer_splits, candidates, fit_candidate, evaluate_candidate,
                   reference, guards=None):
    """fit_candidate(train_groups,candidate) never receives sealed outer groups.

    evaluate_candidate(model,validation_groups) returns joined cross-check frames.
    Fit callbacks own train-only pilot/noise/prior construction. One whole group
    is inner-held at a time, without slicing an outer group into the training pool.
    The outer evaluator is called only after the inner decision is frozen.
    """
    if reference not in candidates or len(set(candidates)) != len(candidates):
        raise ValueError("Declare a unique candidate list containing its reference")
    all_ids = set(group_data)
    results = {}
    for fold, held in outer_splits:
        held = set(held)
        if not held or not held <= all_ids:
            raise ValueError("Outer held groups must be nonempty known groups")
        develop = all_ids-held
        if len(develop) < 2:
            raise ValueError("Nested selection needs at least two development groups")
        cards = {name: [] for name in candidates}
        for gi in sorted(develop):
            train = {k: group_data[k] for k in sorted(develop-{gi})}
            validation = {gi: group_data[gi]}
            for name in candidates:
                model = fit_candidate(train, name)
                frames = evaluate_candidate(model, validation)
                if set(f["fixation"] for f in frames) != {gi}:
                    raise ValueError("Inner evaluator returned a group outside its validation split")
                cards[name].extend(frames)
        decision = choose(cards, guards, reference)
        selected = decision["selected"] or reference
        model = fit_candidate({k: group_data[k] for k in sorted(develop)}, selected)
        outer = evaluate_candidate(model, {k: group_data[k] for k in sorted(held)})
        if set(f["fixation"] for f in outer) != held:
            raise ValueError("Outer evaluator returned a group outside the sealed test split")
        results[fold] = dict(decision=decision, evaluated_candidate=selected,
            fallback_reference=decision["selected"] is None, outer_scorecard=build(outer, purpose="outer_evaluation"),
            inner_scorecards={n: build(rows, purpose="inner_validation") for n, rows in cards.items()},
            training_group_ids=sorted(develop), sealed_outer_group_ids=sorted(held),
            interpretation="nested execution; independence additionally requires data not previously used for development")
    return results
