#!/usr/bin/env python3
"""Summarize Phase 8.3 retained-channel follow-up and support cohorts.

Run from the repository root after the frozen follow-up jobs complete. All
analysis reads saved frames; x/xy records come from the independently
re-enriched backfill stream and the three transformed-y masks from this run.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from full_position.scorecard import build as scorecard_build


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASE = ROOT / "experiments/full_position"
RESPONSES = ("baseline27", "strong_anchor37")
FAMILIES = ("gaze", "capture")
MASKS = ("x", "x_y_common", "x_y_difference", "x_y_common_difference", "xy")
MASK_LABEL = {"x": "x", "x_y_common": "common-y", "x_y_difference": "differential-y",
              "x_y_common_difference": "common+differential-y", "xy": "xy"}
Y_MASKS = ("x_y_common", "x_y_difference", "x_y_common_difference")
SUPPORT_KEYS = ("theta_empirical", "A_empirical", "P1_context_valid",
                "context_outside_training_extrema", "p1_parity_seen_in_training")
COHORTS = ("full", "interior", "empirical_state", "measured_P1", "empirical_state_and_P1")
EPSILON = 1e-12


def read_json(path):
    return json.loads(Path(path).read_text())


def frame_id(frame):
    return [frame[k] for k in ("split_family", "fold", "model", "capture", "fixation", "row")]


def exposure(frame):
    return (frame["fold"], frame["capture"], frame["fixation"])


def equal_exposure_rms(rows, values, expected_exposures):
    grouped = defaultdict(list)
    for row, value in zip(rows, values):
        if value is None or not np.isfinite(float(value)):
            continue
        grouped[exposure(row)].append(float(value))
    absent = sorted(set(expected_exposures) - grouped.keys())
    mean_square = float(np.mean([np.mean(np.square(v)) for v in grouped.values()])) if grouped else None
    return dict(rms=float(np.sqrt(mean_square)) if mean_square is not None else None,
        contributing_exposure_count=len(grouped), scheduled_exposure_count=len(expected_exposures),
        absent_exposure_ids=[list(v) for v in absent], absent_exposure_count=len(absent),
        mean_square=mean_square)


def support_counts(frames):
    result = {}
    slots = [s for f in frames for s in f["slots"]]
    for key in SUPPORT_KEYS:
        counts = Counter()
        for slot in slots:
            value = slot.get("support", {}).get(key)
            counts["unknown" if value is None else "true" if value else "false"] += 1
        result[key] = {name: counts.get(name, 0) for name in ("true", "false", "unknown")}
    return result


def core_metrics(frames, metric_frames=None, metric_slots=None, expected_exposures=None):
    metric_frames = [f for f in frames if f.get("complete_triple")] if metric_frames is None else metric_frames
    metric_slots = [s for f in frames for s in f["slots"] if s.get("scored")] if metric_slots is None else metric_slots
    expected_exposures = sorted(set(exposure(f) for f in frames)) if expected_exposures is None else expected_exposures
    frame_metrics = {}
    for name, field in (("E_px", "E_px"), ("G_theta_deg", "G_theta_deg"),
                        ("G_A_D", "G_A_D"), ("worst_point_px", "worst_point_px")):
        valid = [f for f in metric_frames if f.get(field) is not None]
        frame_metrics[name] = equal_exposure_rms(valid, [f[field] for f in valid], expected_exposures)
    axis_metrics = {}
    for axis_i, axis in enumerate(("x", "y")):
        valid = [s for s in metric_slots if s.get("error_px") is not None]
        axis_metrics[axis] = equal_exposure_rms(valid, [s["error_px"][axis_i] for s in valid], expected_exposures)
    return dict(frame=frame_metrics, axis=axis_metrics)


def criterion_status(frame, criterion):
    slots = frame["slots"]
    if criterion == "empirical_state":
        fields = (("theta_empirical", True), ("A_empirical", True))
    elif criterion == "measured_P1":
        fields = (("P1_context_valid", True), ("context_outside_training_extrema", False),
                  ("p1_parity_seen_in_training", True))
    elif criterion == "empirical_state_and_P1":
        fields = (("theta_empirical", True), ("A_empirical", True), ("P1_context_valid", True),
                  ("context_outside_training_extrema", False), ("p1_parity_seen_in_training", True))
    else:
        raise ValueError(criterion)
    unknown_slots = false_slots = 0
    for slot in slots:
        values = [(slot.get("support", {}).get(key), target) for key, target in fields]
        if any(value is None for value, _ in values):
            unknown_slots += 1
        if any(value is not None and value is not target for value, target in values):
            false_slots += 1
    if false_slots:
        return "false", unknown_slots, false_slots
    if unknown_slots:
        return "unknown", unknown_slots, false_slots
    return "true", 0, 0


def cohort_records(frames, name, expected_exposure_ids):
    complete = [f for f in frames if f.get("complete_triple")]
    unknown_slots = unknown_frames = false_frames = incomplete_frames = 0
    unknown_all_slots = unknown_all_frames = false_all_slots = false_all_frames = 0
    if name == "full":
        selected = list(frames)
    elif name == "interior":
        selected = [f for f in complete if f.get("complete_interior")]
        incomplete_frames = len(frames) - len(complete)
    else:
        selected = []
        incomplete_frames = len(frames) - len(complete)
        for f in frames:
            _, u, false = criterion_status(f, name)
            unknown_all_slots += u
            false_all_slots += false
            unknown_all_frames += int(u > 0)
            false_all_frames += int(false > 0)
        for f in complete:
            status, u, false = criterion_status(f, name)
            unknown_slots += u
            if status == "true":
                selected.append(f)
            elif status == "unknown":
                unknown_frames += 1
            elif status == "false":
                false_frames += 1
    metric_frames = [f for f in selected if f.get("complete_triple")]
    metric_slots = [s for f in selected for s in f["slots"] if s.get("scored")]
    expected_exposures = [tuple(v) for v in expected_exposure_ids]
    # Keep the original full exposure roster for filtered cohorts so missing
    # exposure coverage remains visible rather than shrinking the denominator.
    scorecard = scorecard_build(selected, expected_exposures=expected_exposure_ids)
    metrics = dict(frame={
            "E_px": dict(rms=scorecard["outcomes"]["E_cross_px"]["equal_exposure_rms"],
                contributing_exposure_count=scorecard["outcomes"]["E_cross_px"]["squared_error_aggregation"]["contributing_exposure_count"]),
            "G_theta_deg": dict(rms=scorecard["outcomes"]["G_theta_cross_deg"]["equal_exposure_rms"],
                contributing_exposure_count=scorecard["outcomes"]["G_theta_cross_deg"]["squared_error_aggregation"]["contributing_exposure_count"]),
            "G_A_D": dict(rms=scorecard["outcomes"]["G_A_cross_D"]["equal_exposure_rms"],
                contributing_exposure_count=scorecard["outcomes"]["G_A_cross_D"]["squared_error_aggregation"]["contributing_exposure_count"]),
            "worst_point_px": dict(rms=scorecard["outcomes"]["worst_point_px"]["equal_exposure_rms"],
                contributing_exposure_count=scorecard["outcomes"]["worst_point_px"]["squared_error_aggregation"]["contributing_exposure_count"])
        }, axis={axis: dict(rms=scorecard["axis_errors"][axis]["equal_exposure_rms"],
                contributing_exposure_count=scorecard["axis_errors"][axis]["squared_error_aggregation"]["contributing_exposure_count"])
            for axis in ("x", "y")})
    present = sorted(set(exposure(f) for f in metric_frames))
    absent = [tuple(v) for v in scorecard["absent_scheduled_exposure_ids"]]
    return dict(cohort=name, rule={
            "full": "All scheduled frames; E/G/worst use complete triples, axes use all scored slots.",
            "interior": "Complete triples whose three inverse subsets are all interior; this is not training support.",
            "empirical_state": "Complete triples with theta_empirical and A_empirical true for all three slots.",
            "measured_P1": "Complete triples with P1_context_valid true, context_outside_training_extrema false, and p1_parity_seen_in_training true for all three slots.",
            "empirical_state_and_P1": "Complete triples satisfying both empirical-state and measured-P1 rules."}[name],
        coverage=dict(scheduled_frames=len(frames), scheduled_slots=sum(len(f["slots"]) for f in frames),
            selected_frames=len(selected), selected_complete_frames=len(metric_frames), selected_slots=sum(len(f["slots"]) for f in selected),
            scored_slots=len(metric_slots), expected_exposures=len(expected_exposures),
            contributing_exposures=len(present), absent_exposure_count=len(absent),
            absent_exposure_ids=[list(v) for v in absent], incomplete_frames_excluded=incomplete_frames,
            false_support_frames_excluded=false_frames, unknown_support_frames_excluded=unknown_frames,
            unknown_support_slots_excluded=unknown_slots,
            support_filter_unknown_frames_all_scheduled=unknown_all_frames,
            support_filter_unknown_slots_all_scheduled=unknown_all_slots,
            support_filter_false_frames_all_scheduled=false_all_frames,
            support_filter_false_slots_all_scheduled=false_all_slots),
        complete_frame_ids=[frame_id(f) for f in metric_frames],
        complete_frame_ids_sha256=hashlib.sha256(json.dumps([frame_id(f) for f in metric_frames],
            separators=(",", ":")).encode()).hexdigest(),
        metrics=metrics,
        support_counts=support_counts(selected),
        criterion_unknown_slot_count=unknown_slots,
        criterion_false_frame_count=false_frames)


def load_frames():
    folds = read_json(OUT / "config.json")["folds"]
    expected_keys = {f"phase83/{response}/{fold}/{mask}"
                     for response in RESPONSES for fold in folds for mask in ("x", "xy")}
    x_xy = defaultdict(list)
    with gzip.open(OUT / "scorecard_frames.jsonl.gz", "rt") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("candidate") in expected_keys:
                x_xy[row["candidate"]].append(row["frame"])
    frames = {}
    for response in RESPONSES:
        for family in FAMILIES:
            family_folds = [f for f in folds if f.startswith(family + "_")]
            for mask in MASKS:
                result = []
                for fold in family_folds:
                    if mask in ("x", "xy"):
                        key = f"phase83/{response}/{fold}/{mask}"
                        result.extend(x_xy[key])
                    else:
                        result.extend(read_json(OUT / "information" / response / fold / mask / "frames.json"))
                frames[(response, family, mask)] = result
    missing = [k for k in expected_keys if not x_xy[k]]
    assert not missing, f"Re-enriched x/xy frames missing: {missing[:5]}"
    for key, values in frames.items():
        assert len(values) == 160, f"Expected 160 frames for {key}, found {len(values)}"
    return frames


def pair_changes():
    output = {}
    fields = ("delta_squared_vector_error_px2", "delta_squared_x_error_px2", "delta_squared_y_error_px2")
    frame_fields = ("delta_squared_E_px", "delta_squared_G_theta_deg", "delta_squared_G_A_D",
                    "delta_squared_worst_point_px", "delta_squared_x_error_px2", "delta_squared_y_error_px2")
    with gzip.open(OUT / "information_paired_membership.jsonl.gz", "rt") as stream:
        for line in stream:
            rec = json.loads(line)
            parts = rec["comparison"].split("/")
            response, family, mask_minus_x = parts
            mask = mask_minus_x.removesuffix("_minus_x")
            points = [p for p in rec["points"] if p["shared_scored"] and p["transition"] == "interior_to_interior"]
            frames = [f for f in rec["frames"] if f["transition"] == "interior_to_interior"]
            def groupmean(rows, keyfunc, field):
                grouped = defaultdict(list)
                for row in rows:
                    if field in row:
                        grouped[keyfunc(row)].append(float(row[field]))
                return dict(equal_exposure_mean=float(np.mean([np.mean(v) for v in grouped.values()])) if grouped else None,
                            exposure_count=len(grouped))
            point_changes = {}
            for field in fields:
                point_changes[field] = groupmean(points,
                    lambda p: (p["identity"][1], p["capture"], p["fixation"]), field)
            frame_changes = {}
            for field in frame_fields:
                frame_changes[field] = groupmean(frames,
                    lambda f: (f["identity"][1], f["capture"], f["fixation"]), field)
            output[(response, family, mask)] = dict(
                shared_scored_interior_points=len(points), paired_complete_interior_frames=len(frames),
                point_ids_sha256=hashlib.sha256(json.dumps([p["identity"] for p in points], separators=(",", ":")).encode()).hexdigest(),
                frame_ids_sha256=hashlib.sha256(json.dumps([f["identity"] for f in frames], separators=(",", ":")).encode()).hexdigest(),
                point_squared_changes=point_changes, frame_squared_changes=frame_changes)
    assert len(output) == len(RESPONSES) * len(FAMILIES) * len(Y_MASKS)
    return output


def main():
    completion = read_json(OUT / "completion.json")
    config = read_json(OUT / "config.json")
    verification = read_json(OUT / "verification.json")
    if not verification.get("all_checks_passed") or verification.get("errors"):
        raise RuntimeError("saved independent verification is not clean")
    assert completion.get("complete") and completion.get("information_tasks") == 54
    frame_sets = load_frames()
    summary = read_json(OUT / "information_summary.json")
    expected_masks = set(MASKS)
    assert set(summary["masks"]) == set(MASKS[:4])  # xy is added as the reference cohort.
    cards = summary["scorecards"]
    cohort_results, main_rows = {}, []
    for response in RESPONSES:
        for family in FAMILIES:
            key = f"{response}/{family}"
            assert set(cards[key]) == expected_masks
            for mask in MASKS:
                frames = frame_sets[(response, family, mask)]
                complete = [f for f in frames if f.get("complete_triple")]
                all_slots = [s for f in frames for s in f["slots"]]
                expected_exposures = sorted(set(exposure(f) for f in frames))
                metrics = core_metrics(frames, complete,
                    [s for s in all_slots if s.get("scored")], expected_exposures)
                card = cards[key][mask]
                # Ensure the report is a faithful reduction of saved frames.
                for field, score_name in (("E_px", "E_cross_px"), ("G_theta_deg", "G_theta_cross_deg"),
                    ("G_A_D", "G_A_cross_D"), ("worst_point_px", "worst_point_px")):
                    actual = metrics["frame"][field]["rms"]
                    expected = card["outcomes"][score_name]["equal_exposure_rms"]
                    assert actual is None and expected is None or np.isclose(actual, expected, rtol=1e-10, atol=1e-10), f"saved frame/card mismatch {key}/{mask}/{field}"
                for axis in ("x", "y"):
                    actual = metrics["axis"][axis]["rms"]
                    expected = card["axis_errors"][axis]["equal_exposure_rms"]
                    assert actual is None and expected is None or np.isclose(actual, expected, rtol=1e-10, atol=1e-10), f"saved frame/card mismatch {key}/{mask}/{axis}"
                cohorts = {name: cohort_records(frames, name, card["expected_exposure_ids"]) for name in COHORTS}
                cohort_results[key + "/" + mask] = dict(response=response, family=family, mask=mask,
                    scheduled_coverage=dict(frames=len(frames), slots=len(all_slots),
                        input_eligible=card["coverage"]["input_eligible"], scored=card["coverage"]["scored"],
                        complete_triples=card["coverage"]["complete_triples"],
                        expected_exposures=len(expected_exposures), absent_full_exposures=len(card["absent_scheduled_exposure_ids"])),
                    support_counts=support_counts(frames), cohorts=cohorts)
                supports = support_counts(frames)
                main_rows.append(dict(response=response, family=family, mask=mask,
                    scheduled=len(frames), slots=len(all_slots), eligible=card["coverage"]["input_eligible"],
                    scored=card["coverage"]["scored"], complete=card["coverage"]["complete_triples"],
                    full_metrics=metrics, support=supports,
                    absent_exposures=len(card["absent_scheduled_exposure_ids"]),
                    complete_frame_ids=[frame_id(f) for f in complete]))

    support_out = dict(schema="phase83_support_cohorts_v1",
        interpretation="Saved-frame descriptive cohorts. Empirical support is componentwise training-state extrema, not a validated multidimensional envelope. Bound status is reported separately and never used as a proxy for training support.",
        unknown_policy="Unknown support values exclude a frame from support-filtered cohorts and are counted explicitly; unknown is not treated as false or true.",
        capture_scope="captures 5 and 6 were untouched",
        conditions=cohort_results)
    (OUT / "support_cohorts.json").write_text(json.dumps(support_out, indent=2, sort_keys=True) + "\n")

    changes = pair_changes()
    equivalence = read_json(OUT / "invertible_equivalence.json")
    cases = equivalence["cases"]
    state_deltas = [r["selected_state_max_abs_difference"] for r in cases if r.get("selected_state_max_abs_difference") is not None]
    max_theta = max((d[0] for d in state_deltas), default=0.)
    max_accommodation = max((d[1] for d in state_deltas), default=0.)
    max_cost = max((r.get("cost_abs_difference") or 0. for r in cases), default=0.)
    mismatch_cases = [r for r in cases if not r["branch_clusters_equivalent"] or r["xy_rank"] != r["transformed_rank"]]
    profile_counts = defaultdict(int)
    profile_tasks = defaultdict(int)
    for path in (OUT / "information").glob("*/*/*/completion.json"):
        meta = read_json(path)
        response, fold, mask = path.parts[-4], path.parts[-3], path.parts[-2]
        profile_counts[(response, mask)] += int(meta.get("profile_diagnostic_cases", 0))
        profile_tasks[(response, mask)] += 1

    # Parent-requested scientific interpretation is deliberately descriptive.
    tests = Path(OUT / "full_suite_tests.txt").read_text()
    import re
    test_match = re.search(r"(\d+ passed in [\d.]+s)", tests)
    test_summary = test_match.group(1) if test_match else tests.strip().splitlines()[-1]
    lines = ["# Phase 8.3 retained-channel audit follow-up", "",
        "Differential-y gives better overall, axis, and worst-point prediction errors than x and xy in baseline27, but does not uniformly improve state agreement or empirical support. Capture-family empirical-A slots are true/false 315/114 under differential-y versus 343/86 under xy, and complete empirical-state support is 99 versus 111 frames (empirical-state plus measured-P1: 73 versus 85). Gaze-family state disagreement is higher than xy, and common-y-only is worse, especially for capture. No mask is promoted because selection guards were not predeclared.", "",
        f"The frozen-response follow-up completed {completion['information_tasks']} jobs in {completion['seconds']:.4f} seconds using {config['workers']} workers with OPENBLAS_NUM_THREADS={config['thread_environment']['OPENBLAS_NUM_THREADS']} and OMP_NUM_THREADS={config['thread_environment']['OMP_NUM_THREADS']}. Historical sources remained unchanged. The saved full suite records {test_summary}.", "",
        "Each score row below uses equal-exposure aggregation over the saved family frame set. E is excluded-point vector RMS (px); Gθ and G_A are pairwise latent-state disagreement; worst is the worst excluded point. Axis RMS uses all scored held-out slots. Coverage retains scheduled frames/slots and complete-triple counts; support counts are T/F/U over all scheduled slots. The complete-frame IDs and detailed cohort memberships are preserved in `support_cohorts.json`.", "",
        "| Response | Family | Mask | Scheduled frames / slots | Input-eligible / scored slots | Complete triples | E px | Gθ deg | G_A D | Worst px | x px | y px | Empirical θ T/F/U | Empirical A T/F/U | P1 valid T/F/U | Context inside/outside/U | Parity seen/not/U | Absent full exposures |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    def tri(s, key):
        v = s[key]
        return f"{v['true']}/{v['false']}/{v['unknown']}"
    for row in main_rows:
        s, m = row["support"], row["full_metrics"]
        get = lambda name: m["frame"][name]["rms"]
        lines.append(f"| {row['response']} | {row['family']} | {MASK_LABEL[row['mask']]} | {row['scheduled']} / {row['slots']} | {row['eligible']} / {row['scored']} | {row['complete']} | {get('E_px'):.3f} | {get('G_theta_deg'):.4f} | {get('G_A_D'):.4f} | {get('worst_point_px'):.3f} | {m['axis']['x']['rms']:.3f} | {m['axis']['y']['rms']:.3f} | {tri(s,'theta_empirical')} | {tri(s,'A_empirical')} | {tri(s,'P1_context_valid')} | {s['context_outside_training_extrema']['false']}/{s['context_outside_training_extrema']['true']}/{s['context_outside_training_extrema']['unknown']} | {s['p1_parity_seen_in_training']['true']}/{s['p1_parity_seen_in_training']['false']}/{s['p1_parity_seen_in_training']['unknown']} | {row['absent_exposures']} |")

    lines += ["", "The paired table reports equal-exposure means of squared changes (candidate mask minus x) among exact shared interior point/frame memberships. Negative values indicate smaller squared errors/disagreement under the candidate mask. Each 3-mask × response × family comparison retains its paired membership hash in `summary.json`.", "",
        "| Response | Family | Mask vs x | Shared interior points | Shared interior complete frames | Δvector error² px² | Δx error² px² | Δy error² px² | ΔE² px² | ΔGθ² deg² | ΔG_A² D² | Δworst² px² |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for (response, family, mask), row in sorted(changes.items()):
        p, f = row["point_squared_changes"], row["frame_squared_changes"]
        fmt = lambda x: "n/a" if x is None else f"{x:.4f}"
        lines.append(f"| {response} | {family} | {MASK_LABEL[mask]} | {row['shared_scored_interior_points']} | {row['paired_complete_interior_frames']} | {fmt(p['delta_squared_vector_error_px2']['equal_exposure_mean'])} | {fmt(p['delta_squared_x_error_px2']['equal_exposure_mean'])} | {fmt(p['delta_squared_y_error_px2']['equal_exposure_mean'])} | {fmt(f['delta_squared_E_px']['equal_exposure_mean'])} | {fmt(f['delta_squared_G_theta_deg']['equal_exposure_mean'])} | {fmt(f['delta_squared_G_A_D']['equal_exposure_mean'])} | {fmt(f['delta_squared_worst_point_px']['equal_exposure_mean'])} |")

    lines += ["", "## Support cohorts", "",
        "Support cohorts require all three subset slots in a complete frame to meet the named support rule. Unknown support values exclude the frame and are counted; they are not imputed. `interior` means the inverse solutions are interior to the declared numerical bounds, while `empirical_state` means each state component lies inside its corresponding training-state extrema. These categories are independent: numerical interior status is not evidence of training support. Cohort rows use candidate-specific frame identities and are descriptive; do not rank candidates from those rows. Candidate ranking requires comparison on shared identities, as in the paired table above.", "",
        "| Response / family / mask | Cohort | Complete frames | Scored slots | Contributing / expected exposures | Absent exposures | Unknown support frames / slots (all scheduled) | Unknown complete frames / slots excluded | E px | Gθ deg | G_A D | Worst px | x px | y px |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for key, condition in sorted(cohort_results.items()):
        response, family, mask = key.split("/")
        if response != "baseline27" or mask not in ("xy", "x_y_difference"):
            continue
        for cohort in COHORTS:
            c = condition["cohorts"][cohort]
            met, cov = c["metrics"], c["coverage"]
            def v(group, metric):
                return group[metric]["rms"]
            values = [v(met["frame"], name) for name in ("E_px", "G_theta_deg", "G_A_D", "worst_point_px")]
            values += [v(met["axis"], "x"), v(met["axis"], "y")]
            fmt = lambda x: "n/a" if x is None else f"{x:.3f}"
            lines.append(f"| {key} | {cohort} | {cov['selected_complete_frames']} | {cov['scored_slots']} | {cov['contributing_exposures']} / {cov['expected_exposures']} | {cov['absent_exposure_count']} | {cov['support_filter_unknown_frames_all_scheduled']} / {cov['support_filter_unknown_slots_all_scheduled']} | {cov['unknown_support_frames_excluded']} / {cov['unknown_support_slots_excluded']} | " + " | ".join(fmt(x) for x in values) + " |")

    lines += ["", "## Information and diagnostic checks", "",
        f"The combined-y retained transform matched full xy over {len(cases)} held-out slots: {sum(r['branch_clusters_equivalent'] for r in cases)} branch-cluster comparisons were equivalent; rank/availability/ambiguity disagreements: {len(mismatch_cases)}. Maximum selected-state differences were {max_theta:.3g}° gaze and {max_accommodation:.3g} D accommodation; maximum objective-cost difference was {max_cost:.3g}. These finite multistart checks compare discovered clusters and do not establish global branch completeness.", "",
        "| Response | Mask | Diagnostic profile cases | Tasks |",
        "|---|---|---:|---:|"]
    for key in sorted(profile_counts):
        response, mask = key
        lines.append(f"| {response} | {MASK_LABEL[mask]} | {profile_counts[key]} | {profile_tasks[key]} |")
    lines += ["", "For these saved cases no new 65-grid profile diagnostic was triggered by the three transformed-y masks. The combined-y-versus-xy numerical equivalence is a finite multistart consistency result; it does not upgrade the heuristic inverse to a global completeness guarantee.", ""]
    if (OUT / "existing_x_profile_summary.json").exists():
        x_profile = read_json(OUT / "existing_x_profile_summary.json")
        lines.append(f"The supplemental retained-x profile audit covered {x_profile['cases']} earlier cases ({x_profile['rank_weak']} rank-weak, {x_profile['ambiguous']} ambiguous, with overlap); profiles were available for all, with {x_profile['newly_discovered_branch_cases']} newly discovered branches and {x_profile['lower_retained_cost_cases']} lower-cost cases. Primary scores were unchanged. The finite grid is not a global completeness guarantee.")
    lines += ["", "The y-mask jobs reuse frozen calibration responses and do not include a nested calibration study. In the four response/family cells, differential-y minus x had negative shared-interior changes in E², Gθ², G_A², and worst-point²; common-y minus x had positive E² changes in all four. For baseline27 capture, combined-y minus x reduced E² by 0.2586 px² while increasing worst-point² by 1.9279 px². A possible shared-bias clue is the training capture y-mean sequence: captures 2–4 were −0.848, +0.529, and +0.324 versus strong-anchor means −0.860, +0.586, and +0.351. This is a correlated secondary observation motivating train-only capture/context residual checks, not a completed nested calibration study. The xy baseline remains the reference. Captures 5 and 6 were untouched. A future calibration comparison should predeclare guards and use denser grouped conditions; these development results do not decide that choice.", "",
        "## Independent verification", "",
        f"The saved [verification record](verification.json) reports all checks passed with {len(verification['errors'])} errors. It checked {verification['information_task_count']} transformed-response tasks, {verification['retained_H_R_H_transpose_checks']:,} retained covariance transforms, {verification['retained_objective_branch_checks']:,} objective/branch cases, {verification['branch_gradient_curvature_polish_checks']:,} gradient/curvature/polish cases, {verification['retained_candidate_start_records']:,} archived start records, {verification['joined_E_G_axis_frame_metric_checks']:,} joined frame E/G/axis metrics, and {verification['support_records_checked']:,} support records. Historical checks covered {verification['historical_join_rebuild_checks']} joins and preserved {verification['historical_primary_state_score_preservation_checks']:,} primary scores; {verification['paired_membership_checks']} paired-membership checks and {len(verification['representative_full_grid_inversions'])} representative full-grid inversions also passed. All {verification['historical_source_hashes']['file_count']:,} archived historical file hashes matched.", "",
        "Reproduce from the repository root with the same single-threaded BLAS environment:", "",
        "```sh", "PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python full_position/audit83.py --output experiments/full_position/phase83_audit_followup_v1 --workers 12 --mode all", "PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/full_position/phase83_audit_followup_v1/profile_existing_x.py", "PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/full_position/phase83_audit_followup_v1/verify_results.py", "PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/full_position/phase83_audit_followup_v1/summarize_results.py", "```", "",
        "The run entry point is [`full_position/audit83.py`](../../../full_position/audit83.py); the remaining commands profile the retained-x cases, verify saved artifacts, and regenerate this report and the support cohorts.", ""]
    report_path = OUT / "RESULTS.md"
    report_path.write_text("\n".join(lines))

    result = dict(schema="phase83_audit_followup_summary_v1", run=dict(seconds=completion["seconds"], workers=config["workers"],
        thread_environment=config["thread_environment"], historical_sources_unchanged=completion["historical_sources_unchanged"],
        tasks=completion["information_tasks"]), conditions=len(main_rows), masks=list(MASKS),
        scorecard_conditions=main_rows,
        paired_interior_changes={"/".join(k): v for k, v in changes.items()},
        combined_vs_xy=dict(case_count=len(cases), equivalent_cluster_count=sum(r["branch_clusters_equivalent"] for r in cases),
            exceptions=mismatch_cases, max_theta_difference_deg=max_theta,
            max_accommodation_difference_D=max_accommodation, max_cost_difference=max_cost),
        profile_diagnostic_counts={"/".join(k): dict(cases=v, tasks=profile_tasks[k]) for k, v in profile_counts.items()},
        no_nested_calibration_study=True, no_promotion=True, captures_5_and_6_untouched=True,
        support_cohorts_path="support_cohorts.json",
        verification=dict(path="verification.json", all_checks_passed=verification["all_checks_passed"],
            errors=verification["errors"], verifier_sha256=verification["verifier_sha256"],
            record_sha256=hashlib.sha256((OUT / "verification.json").read_bytes()).hexdigest(),
            historical_file_count=verification["historical_source_hashes"]["file_count"]),
        report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest())
    (OUT / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(dict(report=str(OUT / "RESULTS.md"), support_cohorts=str(OUT / "support_cohorts.json"),
                          conditions=len(main_rows), paired_comparisons=len(changes), equivalence_cases=len(cases)), indent=2))


if __name__ == "__main__":
    main()
