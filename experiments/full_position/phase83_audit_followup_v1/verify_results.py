#!/usr/bin/env python3
"""Independent checks for Phase 8.3 retained-y audit follow-up artifacts.

This verifier intentionally reads raw detections for captures 1–4 only. It does
not refit models or repeat the full 49-start grids; it recomputes each archived
state's retained-only objective and certificate, plus a small representative
set of inversions.
"""
from __future__ import annotations

import collections
import gzip
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "experiments/full_position"
CONFIG_PATH = OUT / "config.json"
if not CONFIG_PATH.is_file():
    raise SystemExit("audit83 config.json is not available yet")
CONFIG = json.loads(CONFIG_PATH.read_text())

from full_position.audit_followup import transitions
from full_position.crosscheck import _read_population, join_records
from full_position.data import load_capture
from full_position.geometry import context
from full_position.information import MASKS, TransformedResponse, predict_raw_holdout, retained_transform
from full_position.invert import STARTS, objective_derivatives, polish, projected_gradient
from full_position.model import LOWER, STATE_SCALE, UPPER, PositionModel
from full_position.noise import marginal, reference_covariance, whitening
from full_position.phase83 import RESPONSES
from full_position.schema import clean_json, load_model
from full_position.scorecard import build, unique
from full_position.sensitivity import digest, read_holdouts, read_json

MASKS_Y = tuple(MASKS[1:])
PHASES = ("audit_polished_v1", "joint_sensitivity_v1", "axis_anchor_sensitivity_v1",
          "phase83_retained_channels_v1")
SOURCE_DIRS = {"phase81": ("audit_polished_v1", None),
               "phase82": ("joint_sensitivity_v1", None),
               "axis_anchors": ("axis_anchor_sensitivity_v1", None)}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value) -> str:
    return json.dumps(clean_json(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def equal(a, b, atol=2e-7, rtol=2e-8) -> bool:
    if isinstance(a, np.ndarray):
        a = a.tolist()
    if isinstance(b, np.ndarray):
        b = b.tolist()
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k], atol, rtol) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(equal(x, y, atol, rtol) for x, y in zip(a, b))
    if a is None or b is None:
        return a is b
    if isinstance(a, (bool, str)) or isinstance(b, (bool, str)):
        return a == b
    if isinstance(a, (int, float, np.number)) and isinstance(b, (int, float, np.number)):
        return bool(np.isclose(a, b, atol=atol, rtol=rtol, equal_nan=True))
    return a == b


def require(condition, errors, label):
    if not condition:
        errors.append(label)


def truth(value):
    return str(value).strip().lower() in ("true", "1", "yes") if isinstance(value, str) else bool(value)


def load_local_reviewed():
    """Load labels plus only raw captures 1–4; never open captures 5 or 6."""
    interval_path = ROOT / "data/fixations/fixation_intervals.json"
    report = json.loads(interval_path.read_text())
    interval_sha = sha(interval_path)
    sources = {row["capture"]: row["sha256"] for row in report["sources"]}
    relevant = sorted({g["capture"] for g in report["fixations"]
                       if Path(g["capture"]).name.startswith(("capture_1", "capture_2", "capture_3", "capture_4"))})
    captures = {}
    for name in relevant:
        captures[name] = load_capture(ROOT / "data/detections" / name, sources[name])
    groups = report["fixations"]
    return captures, groups, interval_sha, {name: captures[name].sha256 for name in captures}


def frozen_response(response, fold):
    variant, name = RESPONSES[response]
    return BASE / "joint_sensitivity_v1/variants" / variant / fold / name


def source_for_candidate(key):
    parts = key.split("/")
    if parts[0] == "phase81":
        _, _, fold, name = parts
        return BASE / "audit_polished_v1", fold, name
    if parts[0] == "phase82":
        _, variant, fold, name = parts
        return BASE / "joint_sensitivity_v1/variants" / variant, fold, name
    if parts[0] == "axis_anchors":
        _, variant, fold, name = parts
        return BASE / "axis_anchor_sensitivity_v1/variants" / variant, fold, name
    if parts[0] == "phase83":
        _, response, fold, mask = parts
        variant, name = RESPONSES[response]
        return BASE / "joint_sensitivity_v1/variants" / variant, fold, name, response, mask
    raise ValueError(f"Unknown scorecard candidate: {key}")


def primary_projection(frame):
    """Strip only diagnostics added by support enrichment."""
    extras = {"protocol", "E_cross", "G_theta_cross", "G_A_cross", "nominal_theta", "demand"}
    result = {k: v for k, v in frame.items() if k not in extras and k != "slots"}
    slots = []
    slot_extras = {"support", "retained_image_channels", "normalization", "frozen_model_sha256",
                   "frozen_noise_policy_sha256", "protocol", "nominal_theta", "demand"}
    for slot in frame.get("slots", []):
        slots.append({k: v for k, v in slot.items() if k not in slot_extras})
    result["slots"] = slots
    return result


def expected_support(ctx, meta, state, empirical_range):
    provenance = meta.get("provenance", {})
    result = {"theta_anchor": None, "A_anchor": None, "theta_empirical": None,
              "A_empirical": None, "P1_context_valid": bool(ctx.valid),
              "context_outside_training_extrema": None, "p1_parity_seen_in_training": None}
    reasons = {}
    state_array = np.asarray(state, float) if state is not None else None
    for label, bounds in (("anchor", provenance.get("anchor_range")), ("empirical", empirical_range)):
        for axis, axis_name in enumerate(("theta", "A")):
            key = axis_name + "_" + label
            if state_array is None or state_array.shape != (2,) or not np.isfinite(state_array).all():
                reasons[key] = "subset_state_unavailable"
            elif bounds is None:
                reasons[key] = "training_" + label + "_range_unavailable"
            else:
                result[key] = bool(bounds[0][axis] <= state_array[axis] <= bounds[1][axis])
    support = provenance.get("conditional_context_support", {})
    if not ctx.valid:
        reasons["context_outside_training_extrema"] = "invalid_P1_geometry"
        reasons["p1_parity_seen_in_training"] = "invalid_P1_geometry"
    elif "r_min" in support and "r_max" in support:
        result["context_outside_training_extrema"] = bool(np.any((ctx.r < support["r_min"]) |
                                                                  (ctx.r > support["r_max"])))
    else:
        reasons["context_outside_training_extrema"] = "training_context_metadata_unavailable"
    if not ctx.valid:
        pass
    elif "signed_P1_area_branches" in support:
        result["p1_parity_seen_in_training"] = bool(np.sign(ctx.signed_area) in support["signed_P1_area_branches"])
    else:
        reasons["p1_parity_seen_in_training"] = "training_context_metadata_unavailable"
    return result, reasons


def compare_frame_metrics(frames, errors, label):
    checks = 0
    for frame in frames:
        slots = frame.get("slots", [])
        require(len(slots) == 3, errors, f"{label}: scheduled frame lacks three slots")
        scored = [s for s in slots if s.get("scored")]
        if len(scored) != 3 or {int(s["held_point"]) for s in scored} != {0, 1, 2}:
            require(frame.get("complete_triple") is False, errors,
                    f"{label}: partial slot set marked a complete triple")
            continue
        order = sorted(scored, key=lambda s: int(s["held_point"]))
        e = np.asarray([s["error_px"] for s in order], float)
        states = [np.asarray(s["state"], float) for s in order]
        expected_E = float(np.sqrt(np.mean(np.sum(e * e, axis=1))))
        expected_worst = float(np.max(np.linalg.norm(e, axis=1)))
        dtheta = [(states[i][0] - states[j][0]) for i, j in ((0, 1), (0, 2), (1, 2))]
        dA = [(states[i][1] - states[j][1]) for i, j in ((0, 1), (0, 2), (1, 2))]
        expected_gt = float(np.sqrt(np.mean(np.square(dtheta))))
        expected_ga = float(np.sqrt(np.mean(np.square(dA))))
        for field, value in (("E_px", expected_E), ("worst_point_px", expected_worst),
                             ("G_theta_deg", expected_gt), ("G_A_D", expected_ga)):
            require(equal(frame.get(field), value), errors,
                    f"{label}: frame {frame.get('capture')}/{frame.get('row')} {field} differs")
        checks += 1
    return checks


def main():
    started = time.time()
    errors = []
    config = CONFIG
    require(config.get("schema") == "phase83_audit_followup_v1", errors, "unexpected audit schema")
    require(config.get("mode") == "all", errors, "run was not a full all-mode audit")
    run_completion = read_json(OUT / "completion.json") if (OUT / "completion.json").is_file() else {}
    require(run_completion.get("complete") is True and run_completion.get("information_tasks") == 54,
            errors, "audit completion is missing all 54 information tasks")
    require(run_completion.get("historical_sources_unchanged") is True, errors,
            "runner did not attest historical sources unchanged")

    source_checks = {}
    configured = config.get("frozen_source_hashes", {})
    actual_set = set()
    for source_name in PHASES:
        root_dir = BASE / source_name
        actual_set |= {str(p.relative_to(ROOT)) for p in root_dir.rglob("*") if p.is_file()}
    require(actual_set == set(configured), errors, "historical source file set differs from run config")
    for rel, expected in configured.items():
        path = ROOT / rel
        actual = sha(path) if path.is_file() else None
        source_checks[rel] = {"expected_sha256": expected, "actual_sha256": actual,
                              "matches": actual == expected}
    require(all(v["matches"] for v in source_checks.values()), errors, "historical source hash mismatch")

    implementation_checks = {}
    for rel, expected in config.get("implementation_hashes", {}).items():
        folder = "implementation_snapshot" if rel.startswith("full_position/") else "design_snapshot"
        snapshot = OUT / folder / Path(rel).name
        actual = sha(snapshot) if snapshot.is_file() else None
        current = sha(ROOT / rel) if (ROOT / rel).is_file() else None
        implementation_checks[rel] = {"configured_sha256": expected, "snapshot_sha256": actual,
                                      "matches_config": actual == expected,
                                      "current_sha256": current,
                                      "current_matches_snapshot": current == actual}
    require(all(v["matches_config"] for v in implementation_checks.values()), errors,
            "implementation/design snapshot does not match run config")
    require(all(v["current_matches_snapshot"] for v in implementation_checks.values()), errors,
            "final implementation differs from audited implementation snapshot")

    # Load just captures 1–4; raw captures 5 and 6 are never opened here.
    captures, groups, interval_sha, raw_capture_shas = load_local_reviewed()
    require(set(captures) == {name for name in captures if Path(name).name.startswith(
        ("capture_1", "capture_2", "capture_3", "capture_4"))}, errors,
        "verifier unexpectedly loaded a raw capture outside captures 1–4")

    info_jobs = 0
    selected_counts = {}
    objective_checks = 0
    certificate_checks = 0
    transformed_covariance_checks = 0
    candidate_archive_lines = 0
    candidate_archived_starts = 0
    metric_checks = 0
    support_checks = 0
    support_unknown_counts = collections.Counter()
    support_unknown_reasons = collections.Counter()
    support_value_counts = collections.defaultdict(collections.Counter)
    task_frame_counts = {}
    task_slot_counts = {}
    info_frames_by_cell = {}
    response_models = {}

    # audit83 imports the fixed response mapping from Phase 8.3.
    response_pairs = RESPONSES
    interval_report = json.loads((ROOT / "data/fixations/fixation_intervals.json").read_text())
    raw_hashes_report = {row["capture"]: row["sha256"] for row in interval_report["sources"]}
    for response, (variant, model_name) in response_pairs.items():
        for fold in config["folds"]:
            frozen = frozen_response(response, fold)
            model, meta = load_model(frozen / "model.json")
            training_states = read_json(frozen / "training_states.json")["states"]
            empirical = [np.min(training_states, axis=0).tolist(), np.max(training_states, axis=0).tolist()]
            pilot = PositionModel(27, meta["pilot_coefficients"])
            reference = np.asarray(meta["reference_state"], float)
            sigma = np.asarray(meta["coordinate_covariance"], float)
            response_models[(response, fold)] = (model, meta, pilot, reference, sigma)
            source_provenance = meta.get("provenance", {})
            require(source_provenance.get("interval_sha256") == interval_sha, errors,
                    f"{response}/{fold}: interval hash differs from frozen response")
            expected_capture_map = {n: raw_hashes_report[n] for n in raw_capture_shas}
            require(source_provenance.get("capture_sha256") == expected_capture_map, errors,
                    f"{response}/{fold}: capture hashes differ from frozen response")
            for mask in MASKS_Y:
                info_jobs += 1
                task_dir = OUT / "information" / response / fold / mask
                completion = read_json(task_dir / "completion.json") if (task_dir / "completion.json").is_file() else {}
                require(completion.get("complete") is True, errors,
                        f"{response}/{fold}/{mask}: task completion missing")
                require(completion.get("frozen_model_sha256") == digest(frozen / "model.json"), errors,
                        f"{response}/{fold}/{mask}: frozen model digest differs")
                population = _read_population(frozen.parent / "population.csv.gz")
                chosen = [row for row in population if truth(row.get("selected_for_evaluation"))]
                selected_counts[fold] = len(chosen)
                source_frames = read_json(frozen / "frames.json")
                saved_frames = read_json(task_dir / "frames.json")
                holdouts = read_holdouts(task_dir / "holdouts.jsonl")
                candidates_by_key = {}
                with gzip.open(task_dir / "inverse_candidates.jsonl.gz", "rt") as archive:
                    for line in archive:
                        if not line.strip():
                            continue
                        item = json.loads(line)
                        key = (int(item["fixation"]), int(item["row"]), int(item["held_point"]))
                        candidates_by_key[key] = item["candidates"]
                        candidate_archive_lines += 1
                hold_map = {(h["capture"], int(h["fixation"]), int(h["row"]), int(h["held_point"])): h
                            for h in holdouts}
                require(len(hold_map) == len(holdouts), errors,
                        f"{response}/{fold}/{mask}: duplicate holdout identity")
                cap_frame_map = {(f["capture"], int(f["fixation"]), int(f["row"])): f for f in saved_frames}
                expected_joined = join_records(fold.split("_")[0], fold, response, population,
                                               source_frames, holdouts)
                actual_by_key = {(f["capture"], int(f["fixation"]), int(f["row"])): f for f in saved_frames}
                joined_by_key = {(f["capture"], int(f["fixation"]), int(f["row"])): f for f in expected_joined}
                require(actual_by_key.keys() == joined_by_key.keys(), errors,
                        f"{response}/{fold}/{mask}: joined scheduled identity set differs")
                for key in actual_by_key.keys() & joined_by_key.keys():
                    require(equal(primary_projection(actual_by_key[key]), primary_projection(joined_by_key[key])),
                            errors, f"{response}/{fold}/{mask}: stored joined score/state changed at {key}")
                task_frame_counts[f"{response}/{fold}/{mask}"] = len(saved_frames)
                task_slot_counts[f"{response}/{fold}/{mask}"] = sum(len(f.get("slots", [])) for f in saved_frames)
                info_frames_by_cell[(response, fold, mask)] = saved_frames
                metric_checks += compare_frame_metrics(saved_frames, errors, f"{response}/{fold}/{mask}")

                for frame in saved_frames:
                    capture_name = frame["capture"]
                    require(capture_name in captures, errors,
                            f"{response}/{fold}/{mask}: frame references raw capture outside 1–4")
                    cap = captures[capture_name]
                    row_ix = int(frame["row"])
                    ctx = context(cap.p[row_ix])
                    for slot in frame["slots"]:
                        supp = slot.get("support", {})
                        want, reasons = expected_support(ctx, meta, slot.get("state"), empirical)
                        for field, value in want.items():
                            require(equal(supp.get(field), value), errors,
                                    f"{response}/{fold}/{mask}: support {field} mismatch at {frame['row']}/{slot['held_point']}")
                            if value is None:
                                support_unknown_counts[field] += 1
                                reason = supp.get("unavailable_reasons", {}).get(field)
                                expected_reason = reasons.get(field)
                                require(reason == expected_reason, errors,
                                        f"{response}/{fold}/{mask}: unknown support {field} reason differs")
                                support_unknown_reasons[f"{field}:{reason}"] += 1
                        require(supp.get("training_group_ids") == meta.get("provenance", {}).get("training_group_ids"),
                                errors, f"{response}/{fold}/{mask}: subset support training groups differ")
                        require(supp.get("unavailable_reasons", {}) == reasons, errors,
                                f"{response}/{fold}/{mask}: support unknown reasons are incomplete or stale")
                        for field, value in want.items():
                            support_value_counts[field]["unknown" if value is None else str(bool(value)).lower()] += 1
                        support_checks += 1

                        held = int(slot["held_point"])
                        obs = hold_map.get((capture_name, int(frame["fixation"]), row_ix, held))
                        valid_retained = bool(cap.point_valid[row_ix][np.arange(3) != held].all())
                        raw_ctx_valid = bool(ctx.valid)
                        candidates = candidates_by_key.get((int(frame["fixation"]), row_ix, held), [])
                        if not raw_ctx_valid or not valid_retained:
                            require(len(candidates) == 0, errors,
                                    f"{response}/{fold}/{mask}: invalid P1/retained subset archived candidates")
                            continue
                        require(len(candidates) == len(STARTS) == 49, errors,
                                f"{response}/{fold}/{mask}: valid subset does not archive 49 starts")
                        require(sorted(int(c.get("start", -1)) for c in candidates) == list(range(49)), errors,
                                f"{response}/{fold}/{mask}: candidate archive start IDs are incomplete/duplicated")
                        candidate_archived_starts += len(candidates)

                        kept, H = retained_transform(held, mask)
                        full_cov = reference_covariance(cap.p[row_ix:row_ix+1], pilot, reference, sigma)[0]
                        R = marginal(full_cov, kept)
                        Rt = H @ R @ H.T
                        raw_values = cap.q[row_ix].reshape(-1)[kept]
                        y = (raw_values - ctx.c[kept % 2]) / ctx.ell
                        yt = H @ y
                        if obs is not None:
                            require(obs.get("retained_indices") == kept.tolist(), errors,
                                    f"{response}/{fold}/{mask}: retained channel indices differ")
                            require(equal(obs.get("measurement_transform"), H.tolist()), errors,
                                    f"{response}/{fold}/{mask}: saved H transform differs")
                            require(equal(obs.get("statistical_covariance"), Rt), errors,
                                    f"{response}/{fold}/{mask}: retained covariance differs from H R_ret H.T")
                            transformed_covariance_checks += 1
                        transformed = TransformedResponse(model, kept, H)
                        W = whitening(Rt)
                        branches = slot.get("branches") or []
                        for branch in branches:
                            state = np.asarray(branch["state"], float)
                            z = state / STATE_SCALE
                            cost, grad, hess = objective_derivatives(transformed, ctx.r, yt,
                                                                      W, np.arange(len(H)), z)
                            require(equal(branch.get("cost"), 2 * cost, atol=1e-6, rtol=2e-7), errors,
                                    f"{response}/{fold}/{mask}: branch objective differs at {frame['row']}/{held}")
                            objective_checks += 1
                            cert = branch.get("certificate", {})
                            lower, upper = LOWER / STATE_SCALE, UPPER / STATE_SCALE
                            active = (((z - lower < 1e-9) & (grad > 1e-7)) |
                                      ((upper - z < 1e-9) & (grad < -1e-7)))
                            free = np.flatnonzero(~active)
                            eig = np.linalg.eigvalsh(hess[np.ix_(free, free)]) if len(free) else np.array([])
                            pg = projected_gradient(z, grad, lower, upper)
                            require(pg <= 1e-4 and (not len(eig) or eig[0] >= -1e-10 * max(1., float(np.max(np.abs(eig))))),
                                    errors, f"{response}/{fold}/{mask}: branch gradient/curvature check fails")
                            polish_state, polish_cert = polish(transformed, ctx.r, yt, W,
                                                               np.arange(len(H)), z)
                            require(np.allclose(polish_state, z, atol=1e-9, rtol=0) and
                                    polish_cert.get("certified") is True and polish_cert.get("steps") == 0 and
                                    polish_cert.get("stationarity_encoded", math.inf) <= 1e-4 and
                                    polish_cert.get("physical_correction", math.inf) <= 1e-5 and
                                    polish_cert.get("stable_cost") is True and
                                    polish_cert.get("local_minimum") is True and cert.get("certified") is True,
                                    errors, f"{response}/{fold}/{mask}: branch certificate/polish mismatch")
                            certificate_checks += 1

                        if obs and obs.get("available") and obs.get("state") is not None:
                            z = np.asarray(obs["state"], float) / STATE_SCALE
                            state_cost = 2 * objective_derivatives(transformed, ctx.r, yt, W,
                                                                  np.arange(len(H)), z)[0]
                            require(equal(slot.get("retained_subset_cost"), state_cost, atol=1e-6, rtol=2e-7), errors,
                                    f"{response}/{fold}/{mask}: selected state cost differs from H R H.T objective")
                            require(obs.get("rank") == slot.get("rank") and obs.get("ambiguous") == (not slot.get("unambiguous")),
                                    errors, f"{response}/{fold}/{mask}: joined rank/ambiguity differs from holdout")
                            omitted = np.array([2 * held, 2 * held + 1])
                            pred_records = obs.get("predictions", [])
                            branches_plausible = obs.get("plausible_branches", [])
                            require(len(pred_records) == len(branches_plausible), errors,
                                    f"{response}/{fold}/{mask}: prediction/branch count differs")
                            for br, pred_record in zip(branches_plausible, pred_records):
                                v = model.predict(np.asarray(br["state"], float), ctx.r)[omitted]
                                pixel = ctx.c + ctx.ell * v
                                require(equal(pred_record.get("normalized"), v) and
                                        equal(pred_record.get("pixel"), pixel), errors,
                                        f"{response}/{fold}/{mask}: excluded pixel prediction does not recompute")
                            if slot.get("scored"):
                                error_px = cap.q[row_ix, held] - (ctx.c + ctx.ell * model.predict(np.asarray(obs["state"]), ctx.r)[omitted])
                                require(equal(slot.get("error_px"), error_px, atol=5e-7, rtol=2e-8), errors,
                                        f"{response}/{fold}/{mask}: raw heldout pixel error differs")

    require(info_jobs == 54, errors, f"expected 54 information task cells; saw {info_jobs}")
    expected_information_frames = len(MASKS_Y) * len(response_pairs) * sum(selected_counts.values())
    require(sum(task_frame_counts.values()) == expected_information_frames, errors,
            "information task scheduled frame denominator differs from selected populations")

    # Rebuild scorecards from saved joined frames and guard every identity map.
    backfill_cards = read_json(OUT / "backfill_scorecards.json")
    information_summary = read_json(OUT / "information_summary.json")
    scorecard_rows = {}
    with gzip.open(OUT / "scorecard_frames.jsonl.gz", "rt") as stream:
        for line in stream:
            if not line.strip():
                continue
            item = json.loads(line)
            scorecard_rows.setdefault(item["candidate"], []).append(item["frame"])
    for candidate, frames in scorecard_rows.items():
        unique(frames)
        unique([slot for frame in frames for slot in frame["slots"]], point=True)
        require(candidate in backfill_cards["folds"], errors,
                f"backfilled candidate scorecard missing for {candidate}")
        if candidate in backfill_cards["folds"]:
            require(equal(build(frames), backfill_cards["folds"][candidate]), errors,
                    f"backfilled fold scorecard differs from saved joined frames: {candidate}")
    unbackfilled_controls = {key for key, card in backfill_cards["folds"].items()
                             if card.get("status") == "not_applicable_no_individual_P4_decoder"}
    require(set(backfill_cards["folds"]) - set(scorecard_rows) == unbackfilled_controls and
            len(unbackfilled_controls) == 9,
            errors,
            "backfill candidate/frame keys differ beyond the nine decoder-free controls")

    prior_weak_slots = collections.Counter()
    exposure_support = {}
    for candidate, frames in scorecard_rows.items():
        card = backfill_cards["folds"].get(candidate, {})
        if candidate.startswith("phase82/prior_weak/"):
            for reason, count in card.get("failure_reasons", {}).items():
                prior_weak_slots[reason] += count
            prior_weak_slots["scheduled_slots"] += card.get("coverage", {}).get("scheduled_slots", 0)
            prior_weak_slots["scored_slots"] += card.get("coverage", {}).get("scored", 0)
        exposure_support[candidate] = {
            "expected_exposure_count": len(card.get("expected_exposure_ids", [])),
            "absent_scheduled_exposure_count": len(card.get("absent_scheduled_exposure_ids", [])),
            "outcome_absent_exposure_counts": {
                outcome: len(metric.get("squared_error_aggregation", {}).get("absent_exposure_ids", []))
                for outcome, metric in card.get("outcomes", {}).items()},
        }

    family_groups = collections.defaultdict(list)
    for candidate, frames in scorecard_rows.items():
        parts = candidate.split("/")
        if parts[0] == "phase81":
            family_key = "/".join((parts[0], parts[1], parts[2].split("_")[0], parts[3]))
        elif parts[0] in ("phase82", "axis_anchors"):
            family_key = "/".join((parts[0], parts[1], parts[2].split("_")[0], parts[3]))
        else:
            family_key = "/".join((parts[0], parts[1], parts[2].split("_")[0], parts[3]))
        family_groups[family_key].extend(frames)
    require(set(family_groups) == set(backfill_cards["families"]), errors,
            "backfill family scorecard key set differs from serialized frames")
    family_scorecard_checks = 0
    for key, frames in family_groups.items():
        require(equal(build(frames), backfill_cards["families"].get(key)), errors,
                f"backfill family scorecard differs from serialized frames: {key}")
        family_scorecard_checks += 1

    information_scorecards_checked = 0
    for response in response_pairs:
        for family in ("gaze", "capture"):
            for mask in (*MASKS, "xy"):
                gathered = []
                for fold in config["folds"]:
                    if fold.startswith(family + "_"):
                        if mask in MASKS_Y:
                            rows = info_frames_by_cell[(response, fold, mask)]
                        else:
                            rows = scorecard_rows.get(f"phase83/{response}/{fold}/{mask}", [])
                        gathered.extend(rows)
                saved = information_summary["scorecards"][f"{response}/{family}"][mask]
                require(equal(build(gathered), saved), errors,
                        f"family scorecard differs from saved joined frames: {response}/{family}/{mask}")
                information_scorecards_checked += 1
    for response in response_pairs:
        for family in ("gaze", "capture"):
            x_rows, xy_rows = [], []
            for fold in config["folds"]:
                if fold.startswith(family + "_"):
                    x_rows += scorecard_rows.get(f"phase83/{response}/{fold}/x", [])
                    xy_rows += scorecard_rows.get(f"phase83/{response}/{fold}/xy", [])
            rebuilt = transitions(x_rows, xy_rows)
            saved = backfill_cards["paired_phase83"][f"{response}/{family}"]
            require(equal(saved, rebuilt), errors, f"Phase 8.3 baseline paired scorecard differs: {response}/{family}")

    # Exact paired-membership rows for each added y mask against retained-x.
    membership_entries = []
    with gzip.open(OUT / "information_paired_membership.jsonl.gz", "rt") as stream:
        membership_entries = [json.loads(line) for line in stream if line.strip()]
    membership_checks = 0
    expected_membership_count = len(response_pairs) * 2 * len(MASKS_Y)
    for response in response_pairs:
        for family in ("gaze", "capture"):
            refs, bymask = [], {mask: [] for mask in MASKS_Y}
            for fold in config["folds"]:
                if fold.startswith(family + "_"):
                    refs.extend(scorecard_rows.get(f"phase83/{response}/{fold}/x", []))
                    for mask in MASKS_Y:
                        bymask[mask].extend(info_frames_by_cell[(response, fold, mask)])
            for mask in MASKS_Y:
                pair = transitions(refs, bymask[mask])
                saved = next((row for row in membership_entries
                              if row.get("comparison") == f"{response}/{family}/{mask}_minus_x"), None)
                require(saved is not None, errors, f"paired membership missing {response}/{family}/{mask}")
                if saved:
                    require(canonical(saved.get("points")) == canonical(pair["points"]) and
                            canonical(saved.get("frames")) == canonical(pair["frames"]), errors,
                            f"paired membership identities/counts differ {response}/{family}/{mask}")
                    membership_checks += 1
    require(len(membership_entries) == expected_membership_count, errors,
            f"paired membership file has {len(membership_entries)} rows, expected {expected_membership_count}")

    # Duplicate guards are explicitly exercised with copies; source records stay untouched.
    duplicate_guard_checks = 0
    if scorecard_rows:
        sample_frames = next(iter(scorecard_rows.values()))
        if sample_frames:
            try:
                unique(sample_frames + [sample_frames[0]])
            except ValueError:
                duplicate_guard_checks += 1
            else:
                errors.append("duplicate frame identity was not rejected")
            if sample_frames[0].get("slots"):
                slots = [s for f in sample_frames for s in f["slots"]]
                try:
                    unique(slots + [slots[0]], point=True)
                except ValueError:
                    duplicate_guard_checks += 1
                else:
                    errors.append("duplicate point identity was not rejected")

    # Verify historical scores/states are still represented exactly in backfill.
    historic_rebuilt_checks = 0
    historic_source_preservation_checks = 0
    historical_sources = read_json(OUT / "backfill_scorecards.json")
    source_keys = list(scorecard_rows)
    for candidate in source_keys:
        resolved = source_for_candidate(candidate)
        base, fold, name = resolved[:3]
        if len(resolved) == 5:
            _, _, _, response, mask = resolved
            frozen = frozen_response(response, fold)
            path = BASE / "phase83_retained_channels_v1" / response / fold
            holds_path = path / ("holdouts_x.jsonl" if mask == "x" else "../"+"holdouts.jsonl")
            if mask == "xy":
                variant, model_name = response_pairs[response]
                holds_path = frozen / "holdouts.jsonl"
            else:
                holds_path = path / "holdouts_x.jsonl"
            model_name_for_join = response
        else:
            path = base / fold / name
            holds_path = path / "holdouts.jsonl"
            model_name_for_join = name
        population_path = base / fold / "population.csv.gz"
        if len(resolved) == 5:
            population_path = frozen.parent / "population.csv.gz"
            source_frames_path = frozen / "frames.json"
        else:
            source_frames_path = path / "frames.json"
        population = _read_population(population_path)
        old_frames = read_json(source_frames_path) if source_frames_path.exists() else []
        old_holds = read_holdouts(holds_path)
        expected = join_records(fold.split("_")[0], fold, model_name_for_join,
                                population, old_frames, old_holds)
        observed = scorecard_rows[candidate]
        expected_map = {(f["capture"], f["fixation"], f["row"]): f for f in expected}
        observed_map = {(f["capture"], f["fixation"], f["row"]): f for f in observed}
        require(expected_map.keys() == observed_map.keys(), errors,
                f"historical scheduled exposure membership changed for {candidate}")
        for key in expected_map.keys() & observed_map.keys():
            require(equal(primary_projection(expected_map[key]), primary_projection(observed_map[key])), errors,
                    f"historical score/state/invalid-slot preservation failed for {candidate} at {key}")
            historic_source_preservation_checks += 1
        historic_rebuilt_checks += 1

    # Verify each saved subset support value against its own state and fold's training data.
    for candidate, frames in scorecard_rows.items():
        resolved = source_for_candidate(candidate)
        base, fold, name = resolved[:3]
        model_dir = base / fold / name
        if len(resolved) == 5:
            response = resolved[3]
            model_dir = frozen_response(response, fold)
        model_path = model_dir / "model.json"
        if model_path.exists():
            _, meta = load_model(model_path)
            states = np.asarray(read_json(model_dir / "training_states.json")["states"], float)
            empirical = [states.min(0).tolist(), states.max(0).tolist()]
        else:
            checkpoint = model_dir / "failed_checkpoint.json"
            meta = read_json(checkpoint) if checkpoint.exists() else {}
            empirical = None
        for frame in frames:
            p1cap = captures[frame["capture"]]
            ctx = context(p1cap.p[int(frame["row"])])
            for slot in frame["slots"]:
                want, reasons = expected_support(ctx, meta, slot.get("state"), empirical)
                supp = slot.get("support", {})
                for field, value in want.items():
                    require(equal(supp.get(field), value), errors,
                            f"historical subset support differs {candidate}/{frame['row']}/{slot['held_point']}/{field}")
                    if value is None:
                        reason = supp.get("unavailable_reasons", {}).get(field)
                        require(reason == reasons.get(field), errors,
                                f"historical support unknown reason differs {candidate}/{field}")
                        support_unknown_counts[field] += 1
                        support_unknown_reasons[f"{field}:{reason}"] += 1
                    support_value_counts[field]["unknown" if value is None else str(bool(value)).lower()] += 1
                require(supp.get("unavailable_reasons", {}) == reasons, errors,
                        f"historical support unknown reasons are incomplete or stale {candidate}/{frame['row']}/{slot['held_point']}")
                support_checks += 1

    # Recompute output scorecards from their serialized joined records.
    task_scorecard_checks = 0
    for response in response_pairs:
        for fold in config["folds"]:
            for mask in MASKS_Y:
                task_dir = OUT / "information" / response / fold / mask
                frames = read_json(task_dir / "frames.json")
                stored = read_json(task_dir / "scorecard.json")
                require(equal(build(frames), stored), errors,
                        f"task scorecard differs from frame recomputation: {response}/{fold}/{mask}")
                task_scorecard_checks += 1

    # The previous baseline27/capture y-axis and worst-point regressions must remain present.
    capture_cards = information_summary["scorecards"]["baseline27/capture"]
    base_x, base_xy = capture_cards["x"], capture_cards["xy"]
    y_x = base_x["axis_errors"]["y"]["equal_exposure_rms"]
    y_xy = base_xy["axis_errors"]["y"]["equal_exposure_rms"]
    worst_x = base_x["outcomes"]["worst_point_px"]["equal_exposure_rms"]
    worst_xy = base_xy["outcomes"]["worst_point_px"]["equal_exposure_rms"]
    require(y_xy > y_x and worst_xy > worst_x, errors,
            "frozen baseline27/capture xy y-axis or worst-point regression is not visible")

    # Finite multistart equivalence is a descriptive comparison, not a required exact identity.
    equivalence_file = read_json(OUT / "invertible_equivalence.json")
    eq_rows = equivalence_file.get("cases", [])
    eq_recomputed = {}
    eq_status_mismatches = collections.Counter()
    for response in response_pairs:
        for family in ("gaze", "capture"):
            transformed_frames, xy_frames = [], []
            for fold in config["folds"]:
                if not fold.startswith(family + "_"):
                    continue
                transformed_frames += info_frames_by_cell[(response, fold, "x_y_common_difference")]
                xy_frames += scorecard_rows.get(f"phase83/{response}/{fold}/xy", [])
            xy_slots = {(f["capture"], int(f["fixation"]), int(f["row"]), int(s["held_point"])): s
                        for f in xy_frames for s in f["slots"]}
            for frame in transformed_frames:
                for slot in frame["slots"]:
                    key = (frame["capture"], int(frame["fixation"]), int(frame["row"]), int(slot["held_point"]))
                    other = xy_slots[key]
                    branch_a, branch_b = slot.get("branches", []), other.get("branches", [])
                    matches_a = [any(np.all(np.abs(np.asarray(a["state"])-np.asarray(b["state"])) < [.01, .01])
                                     for b in branch_b) for a in branch_a]
                    matches_b = [any(np.all(np.abs(np.asarray(b["state"])-np.asarray(a["state"])) < [.01, .01])
                                     for a in branch_a) for b in branch_b]
                    clusters_equal = (slot.get("available") == other.get("available") and
                                      slot.get("rank") == other.get("rank") and
                                      slot.get("unambiguous") == other.get("unambiguous") and
                                      all(matches_a) and all(matches_b))
                    state_diff = (np.abs(np.asarray(slot["state"]) - np.asarray(other["state"])).tolist()
                                  if slot.get("state") is not None and other.get("state") is not None else None)
                    cost_diff = (abs(slot["retained_subset_cost"] - other["retained_subset_cost"])
                                 if slot.get("retained_subset_cost") is not None and
                                 other.get("retained_subset_cost") is not None else None)
                    eq_recomputed[(response, family, key)] = {
                        "branch_clusters_equivalent": bool(clusters_equal),
                        "transformed_branch_count": len(branch_a), "xy_branch_count": len(branch_b),
                        "transformed_rank": slot.get("rank"), "xy_rank": other.get("rank"),
                        "state_diff": state_diff, "cost_diff": cost_diff,
                        "available_mismatch": slot.get("available") != other.get("available"),
                        "rank_mismatch": slot.get("rank") != other.get("rank"),
                        "unambiguous_mismatch": slot.get("unambiguous") != other.get("unambiguous"),
                    }
    saved_eq = {(r["response"], r["family"], tuple(r["identity"])): r for r in eq_rows}
    require(set(saved_eq) == set(eq_recomputed), errors,
            "invertible equivalence case identities differ from the joined branch records")
    for key, actual in eq_recomputed.items():
        row = saved_eq.get(key)
        if row is None:
            continue
        require(row.get("branch_clusters_equivalent") == actual["branch_clusters_equivalent"] and
                row.get("xy_branch_count") == actual["xy_branch_count"] and
                row.get("transformed_branch_count") == actual["transformed_branch_count"] and
                row.get("xy_rank") == actual["xy_rank"] and row.get("transformed_rank") == actual["transformed_rank"],
                errors, f"saved invertible equivalence branch/status fields differ at {key}")
        require(equal(row.get("selected_state_max_abs_difference"), actual["state_diff"]) and
                equal(row.get("cost_abs_difference"), actual["cost_diff"]), errors,
                f"saved invertible equivalence state/cost difference differs at {key}")
        for field in ("available_mismatch", "rank_mismatch", "unambiguous_mismatch"):
            eq_status_mismatches[field] += actual[field]
    eq_summary = {"cases": len(eq_rows), "branch_clusters_equivalent": sum(bool(r["branch_clusters_equivalent"]) for r in eq_rows),
                  "branch_cluster_difference": sum(not bool(r["branch_clusters_equivalent"]) for r in eq_rows),
                  "rank_status_differences": sum(r.get("xy_rank") != r.get("transformed_rank") for r in eq_rows),
                  "branch_count_pairs": collections.Counter(
                      f"{r.get('xy_branch_count')}->{r.get('transformed_branch_count')}" for r in eq_rows)}
    selected_state_diffs = [v for r in eq_rows for v in (r.get("selected_state_max_abs_difference") or [])]
    cost_diffs = [r["cost_abs_difference"] for r in eq_rows if r.get("cost_abs_difference") is not None]
    eq_summary["selected_state_abs_difference_max_by_axis"] = ([float(max(selected_state_diffs[::2])),
                                                                  float(max(selected_state_diffs[1::2]))]
                                                                 if selected_state_diffs else None)
    eq_summary["selected_cost_abs_difference_max"] = float(max(cost_diffs)) if cost_diffs else None
    eq_summary["available_rank_ambiguity_status_mismatches"] = dict(eq_status_mismatches)
    require(bool(eq_rows), errors, "invertible equivalence artifact has no branch comparisons")

    # A few representative complete 49-start solves corroborate the transformed path.
    representative_runs = []
    reps_needed = set(MASKS_Y)
    for response in response_pairs:
        if not reps_needed:
            break
        for fold in config["folds"]:
            if not fold.startswith("capture_1"):
                continue
            frozen = frozen_response(response, fold)
            model, meta, pilot, reference, sigma = response_models[(response, fold)]
            frames_by_mask = {mask: info_frames_by_cell[(response, fold, mask)] for mask in MASKS_Y}
            for mask in list(reps_needed):
                chosen = None
                for frame in frames_by_mask[mask]:
                    cap = captures[frame["capture"]]
                    for slot in frame["slots"]:
                        if slot.get("scored") and slot.get("interior") and slot.get("rank") == 2:
                            chosen = (frame, slot, cap)
                            break
                    if chosen:
                        break
                if not chosen:
                    continue
                frame, slot, cap = chosen
                prediction = predict_raw_holdout(model, cap.p[int(frame["row"])], cap.q[int(frame["row"])],
                    cap.point_valid[int(frame["row"])], int(slot["held_point"]), pilot, reference, sigma, mask)
                saved = next(h for h in read_holdouts(OUT/"information"/response/fold/mask/"holdouts.jsonl")
                             if h["capture"] == frame["capture"] and h["row"] == frame["row"] and
                             h["held_point"] == slot["held_point"])
                require(prediction.get("available") == saved.get("available") and
                        prediction.get("rank") == saved.get("rank") and
                        prediction.get("ambiguous") == saved.get("ambiguous"), errors,
                        f"representative inversion status differs for {response}/{fold}/{mask}")
                if prediction.get("available") and saved.get("available"):
                    require(equal(prediction.get("cost"), saved.get("cost"), atol=1e-5, rtol=2e-6) and
                            equal(prediction.get("state"), saved.get("state"), atol=1e-2, rtol=0), errors,
                            f"representative inversion state/cost differs for {response}/{fold}/{mask}")
                representative_runs.append({"response": response, "fold": fold, "mask": mask,
                    "saved_rank": saved.get("rank"), "replayed_rank": prediction.get("rank"),
                    "saved_cost": saved.get("cost"), "replayed_cost": prediction.get("cost"),
                    "state_max_abs_difference": (float(np.max(np.abs(np.asarray(saved["state"])-prediction["state"])))
                        if saved.get("state") is not None and prediction.get("state") is not None else None),
                    "saved_branch_count": len(saved.get("branches") or []),
                    "replayed_branch_count": len(prediction.get("branches") or [])})
                reps_needed.remove(mask)
            if not reps_needed:
                break
    require(not reps_needed, errors, f"representative inversion missing for masks: {sorted(reps_needed)}")

    # Independently verify the supplemental 65-grid profiles of the 17 older
    # ambiguous/weak-rank retained-x slots. Profiles are diagnostic only.
    profile_summary_path = OUT / "existing_x_profile_summary.json"
    profile_archive_path = OUT / "existing_x_profile_audits.jsonl.gz"
    profile_summary = read_json(profile_summary_path) if profile_summary_path.exists() else {}
    profile_script = OUT / "profile_existing_x.py"
    require(profile_summary.get("script_sha256") == sha(profile_script), errors,
            "supplemental profile script hash differs from its summary")
    profile_rows = []
    if profile_archive_path.exists():
        with gzip.open(profile_archive_path, "rt") as stream:
            profile_rows = [json.loads(line) for line in stream if line.strip()]
    profile_x_frames = {}
    for row in profile_rows:
        response, fold = row["response"], row["fold"]
        frame_key = (response, fold, row["capture"], int(row["fixation"]), int(row["row"]))
        if (response, fold) not in profile_x_frames:
            profile_x_frames[(response, fold)] = read_json(BASE / "phase83_retained_channels_v1" /
                                                           response / fold / "x_frames.json")
        frame = next(f for f in profile_x_frames[(response, fold)]
                     if f["capture"] == row["capture"] and int(f["fixation"]) == int(row["fixation"]) and
                     int(f["row"]) == int(row["row"]))
        slot = next(s for s in frame["slots"] if int(s["held_point"]) == int(row["held_point"]))
        require(slot.get("available") and (not slot.get("identifiable") or not slot.get("unambiguous")), errors,
                f"profile input is not a legacy ambiguous/weak-rank x slot: {frame_key}")
        require(row.get("score_replaced") is False and equal(row.get("primary_state"), slot.get("state")) and
                equal(row.get("primary_cost"), slot.get("retained_subset_cost")), errors,
                f"supplemental profile altered or mismatched primary x score: {frame_key}")
        require(row.get("frozen_model_sha256") == digest(frozen_response(response, fold) / "model.json"), errors,
                f"supplemental profile used a different frozen response: {frame_key}")
        cap = captures[row["capture"]]
        row_ix, held = int(row["row"]), int(row["held_point"])
        ctx = context(cap.p[row_ix])
        model, meta, pilot, reference, sigma = response_models[(response, fold)]
        kept, H = retained_transform(held, "x")
        full_cov = reference_covariance(cap.p[row_ix:row_ix+1], pilot, reference, sigma)[0]
        cov_x = H @ marginal(full_cov, kept) @ H.T
        x_ix = kept[::2]
        yx = (cap.q[row_ix].reshape(-1)[x_ix] - ctx.c[x_ix % 2]) / ctx.ell
        transformed = TransformedResponse(model, kept, H)
        W = whitening(cov_x)
        for state in (row["primary_state"], row.get("profile", {}).get("state")):
            if state is None:
                continue
            z = np.asarray(state, float) / STATE_SCALE
            calc_cost = 2 * objective_derivatives(transformed, ctx.r, yx, W,
                                                   np.arange(len(H)), z)[0]
            target_cost = row["primary_cost"] if state == row["primary_state"] else row["profile"].get("cost")
            require(equal(calc_cost, target_cost, atol=1e-6, rtol=2e-7), errors,
                    f"supplemental retained-x profile objective recomputation differs: {frame_key}")
            objective_checks += 1
        selected_profile_state = row.get("profile", {}).get("state")
        if selected_profile_state is not None:
            z = np.asarray(selected_profile_state, float) / STATE_SCALE
            _, grad, hess = objective_derivatives(transformed, ctx.r, yx, W, np.arange(len(H)), z)
            lower, upper = LOWER / STATE_SCALE, UPPER / STATE_SCALE
            active = (((z-lower < 1e-9) & (grad > 1e-7)) | ((upper-z < 1e-9) & (grad < -1e-7)))
            free = np.flatnonzero(~active)
            eig = np.linalg.eigvalsh(hess[np.ix_(free, free)]) if len(free) else np.array([])
            profile_cert = polish(transformed, ctx.r, yx, W, np.arange(len(H)), z)[1]
            require(projected_gradient(z, grad, lower, upper) <= 1e-4 and
                    (not len(eig) or eig[0] >= -1e-10 * max(1., float(np.max(np.abs(eig))))) and
                    profile_cert.get("certified") is True and profile_cert.get("steps") == 0,
                    errors, f"supplemental profile selected branch fails retained-x certificate: {frame_key}")
            certificate_checks += 1
    profile_counts = {
        "cases": profile_summary.get("cases"),
        "rank_weak": profile_summary.get("rank_weak"),
        "ambiguous": profile_summary.get("ambiguous"),
        "profiles_available": profile_summary.get("profiles_available"),
        "newly_discovered_branch_cases": profile_summary.get("newly_discovered_branch_cases"),
        "lower_retained_cost_cases": profile_summary.get("lower_retained_cost_cases"),
        "primary_scores_changed": profile_summary.get("primary_scores_changed"),
        "script_sha256": profile_summary.get("script_sha256"),
        "summary_sha256": sha(profile_summary_path) if profile_summary_path.exists() else None,
        "archive_sha256": sha(profile_archive_path) if profile_archive_path.exists() else None,
    }
    require(profile_summary.get("cases") == len(profile_rows) and
            profile_summary.get("primary_scores_changed") is False and len(profile_rows) == 17,
            errors, "supplemental profile case inventory or unchanged-score claim differs")
    require(all(row.get("score_replaced") is False for row in profile_rows), errors,
            "a supplemental profile marks a primary score as replaced")
    profile_weak = sum((row.get("primary_rank") or 0) < 2 for row in profile_rows)
    profile_ambiguous = sum(not bool(row.get("primary_unambiguous")) for row in profile_rows)
    profile_available = sum(bool(row.get("profile", {}).get("available")) for row in profile_rows)
    profile_new_branch_cases = sum(bool(row.get("newly_discovered_branches")) for row in profile_rows)
    profile_lower_cost_cases = sum(row.get("diagnostic_cost_minus_primary") is not None and
        row["diagnostic_cost_minus_primary"] < -1e-8 * (1 + abs(row["primary_cost"])) for row in profile_rows)
    require(profile_weak == profile_summary.get("rank_weak") and
            profile_ambiguous == profile_summary.get("ambiguous") and
            profile_available == profile_summary.get("profiles_available") and
            profile_new_branch_cases == profile_summary.get("newly_discovered_branch_cases") and
            profile_lower_cost_cases == profile_summary.get("lower_retained_cost_cases"), errors,
            "supplemental profile summary counts differ from the retained-only case records")

    test_record = (OUT / "full_suite_tests.txt").read_text().strip() if (OUT / "full_suite_tests.txt").exists() else ""
    test_sources = {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / "tests").rglob("*.py"))}
    test_output_path = OUT / "full_suite_tests.txt"
    training_residuals = read_json(OUT / "training_residual_summary.json")["summaries"]
    capture_y_training_bias = {
        response: {cap: result["signed_axis_px"]["y"]["mean"]
                   for cap, result in training_residuals[f"{response}/capture"].items()}
        for response in response_pairs}
    details = {
        "schema": "phase83_audit_followup_verification_v1",
        "verified_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "all_checks_passed": not errors,
        "errors": errors,
        "completion": run_completion,
        "historical_source_hashes": {"file_count": len(source_checks), "file_set_matches": actual_set == set(configured),
                                      "checks": source_checks},
        "implementation_design_snapshot_hashes": implementation_checks,
        "raw_inputs_read": {"captures": sorted(captures), "interval_sha256": interval_sha,
                            "capture_sha256": raw_capture_shas,
                            "explicitly_not_read": ["capture_5", "capture_6"]},
        "frozen_response_provenance_checks": {
            f"{response}/{fold}": {"interval_matches": read_json(frozen_response(response, fold)/"model.json")["provenance"].get("interval_sha256") == interval_sha,
                "capture_matches": read_json(frozen_response(response, fold)/"model.json")["provenance"].get("capture_sha256") ==
                    {n: raw_hashes_report[n] for n in raw_capture_shas}}
            for response in response_pairs for fold in config["folds"]},
        "selected_rows_per_fold": selected_counts,
        "information_task_frame_counts": task_frame_counts,
        "information_task_slot_counts": task_slot_counts,
        "information_task_count": info_jobs,
        "retained_H_R_H_transpose_checks": transformed_covariance_checks,
        "retained_objective_branch_checks": objective_checks,
        "branch_gradient_curvature_polish_checks": certificate_checks,
        "retained_candidate_archive_lines": candidate_archive_lines,
        "retained_candidate_start_records": candidate_archived_starts,
        "joined_E_G_axis_frame_metric_checks": metric_checks,
        "support_records_checked": support_checks,
        "support_unknown_counts": dict(support_unknown_counts),
        "support_unknown_reasons": dict(support_unknown_reasons),
        "backfill_candidate_count": len(scorecard_rows),
        "backfill_primary_scorecard_rebuild_checks": len(scorecard_rows),
        "backfill_family_scorecard_rebuild_checks": family_scorecard_checks,
        "information_family_scorecard_rebuild_checks": information_scorecards_checked,
        "information_task_scorecard_rebuild_checks": task_scorecard_checks,
        "historical_join_rebuild_checks": historic_rebuilt_checks,
        "historical_primary_state_score_preservation_checks": historic_source_preservation_checks,
        "historical_weak_prior_slot_disposition_counts": dict(prior_weak_slots),
        "historical_exposure_support_by_candidate": exposure_support,
        "paired_membership_checks": membership_checks,
        "duplicate_identity_guard_checks": duplicate_guard_checks,
        "baseline27_capture_regression_visibility": {"x_y_axis_equal_exposure_rms": y_x,
            "xy_y_axis_equal_exposure_rms": y_xy, "x_worst_point_equal_exposure_rms": worst_x,
            "xy_worst_point_equal_exposure_rms": worst_xy,
            "both_regressions_visible": y_xy > y_x and worst_xy > worst_x},
        "in_sample_capture_y_residual_bias_px": capture_y_training_bias,
        "finite_multistart_invertible_equivalence_summary": eq_summary,
        "representative_full_grid_inversions": representative_runs,
        "supplemental_existing_x_profiles": profile_counts,
        "verifier_sha256": sha(Path(__file__).resolve()),
        "regression_test_evidence": {"saved_output": test_record,
            "saved_output_sha256": sha(test_output_path) if test_output_path.exists() else None,
            "test_source_sha256": test_sources,
            "test_source_count": len(test_sources),
            "verifier_sha256": sha(Path(__file__).resolve())},
        "elapsed_seconds": time.time() - started,
        "interpretation": "development-only follow-up; branch-set equality describes finite multistart results and is not a proof of global branch completeness",
    }
    (OUT / "verification.json").write_text(json.dumps(details, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"all_checks_passed": not errors, "errors": errors,
                      "source_files": len(source_checks), "information_tasks": info_jobs,
                      "objective_checks": objective_checks, "certificate_checks": certificate_checks,
                      "scorecard_rows": len(scorecard_rows), "paired_membership_checks": membership_checks,
                      "representative_runs": len(representative_runs),
                      "elapsed_seconds": details["elapsed_seconds"]}, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
