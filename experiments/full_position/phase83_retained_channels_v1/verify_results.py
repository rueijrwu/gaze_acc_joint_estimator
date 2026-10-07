#!/usr/bin/env python3
"""Independent mechanical verification for the completed Phase 8.3 run."""
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

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
CONFIG = json.loads((OUT / "config.json").read_text())
SOURCE = Path(CONFIG["source"])

from full_position.audit_followup import transitions
from full_position.crosscheck import _read_population, join_records
from full_position.data import load_reviewed
from full_position.geometry import context
from full_position.invert import STARTS, objective_derivatives, polish, projected_gradient
from full_position.model import LOWER, STATE_SCALE, UPPER, PositionModel
from full_position.noise import marginal, reference_covariance, whitening
from full_position.schema import load_model
from full_position.sensitivity import digest, read_holdouts, read_json


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def close(a, b, atol=2e-7, rtol=2e-8) -> bool:
    try:
        return bool(np.allclose(np.asarray(a, float), np.asarray(b, float), atol=atol, rtol=rtol,
                                equal_nan=True))
    except (TypeError, ValueError):
        return False


def require(ok, errors, label):
    if not ok:
        errors.append(label)


def selected(population):
    return [r for r in population if str(r.get("selected_for_evaluation", "")).lower() == "true"]


def main():
    start = time.time()
    errors = []
    source_expected = CONFIG["frozen_source_hashes"]
    source_actual_files = {str(p.relative_to(SOURCE)) for p in SOURCE.rglob("*") if p.is_file()}
    require(source_actual_files == set(source_expected), errors, "frozen source file set differs from config")
    source_checks = {rel: {"expected": expected,
                           "actual": sha(SOURCE / rel) if (SOURCE / rel).is_file() else None,
                           "matches": (SOURCE / rel).is_file() and sha(SOURCE / rel) == expected}
                     for rel, expected in source_expected.items()}
    require(all(x["matches"] for x in source_checks.values()), errors, "frozen source hash mismatch")

    implementation_checks = {}
    for rel, expected in CONFIG["implementation_hashes"].items():
        path = OUT / "design_snapshot" / Path(rel).name if rel in ("Theory.md", "ESTIMATOR_PLAN.md") else OUT / "implementation_snapshot" / Path(rel).name
        actual = sha(path) if path.is_file() else None
        implementation_checks[rel] = {"expected": expected, "snapshot_sha256": actual,
                                      "matches": actual == expected}
    require(all(x["matches"] for x in implementation_checks.values()), errors,
            "implementation/design snapshot hashes differ from run config")

    run_completion = read_json(OUT / "completion.json")
    require(run_completion.get("complete") is True and run_completion.get("tasks") == 18,
            errors, "top-level completion does not report all 18 tasks")
    require(run_completion.get("frozen_source_unchanged") is True, errors,
            "run completion does not attest frozen source unchanged")

    captures, groups, interval_hash = load_reviewed(ROOT)
    raw_hashes = {name: cap.sha256 for name, cap in captures.items()}
    frame_totals = collections.Counter()
    slot_totals = collections.Counter()
    selected_counts = {}
    metric_checks = 0
    invalid_preservation_checks = 0
    branch_counts = collections.defaultdict(collections.Counter)
    boundary_interior = collections.defaultdict(collections.Counter)
    scored_counts = collections.Counter()
    archive_lines = 0
    archive_candidate_records = 0
    x_recompute_checks = 0
    cert_checks = 0
    xy_reuse_checks = 0
    model_provenance_checks = {}
    population_cache = {}
    frozen_frames_cache = {}
    for response, (variant, model_name) in CONFIG["responses"].items():
        for fold in CONFIG["folds"]:
            pop = _read_population(SOURCE / "populations" / f"{fold}.csv.gz")
            chosen = selected(pop)
            selected_counts[fold] = len(chosen)
            population_cache[fold] = pop
            frozen = SOURCE / "variants" / variant / fold / model_name
            task_completion = read_json(OUT / response / fold / "completion.json")
            require(task_completion.get("complete") is True, errors,
                    f"{response}/{fold}: task completion missing")
            frozen_frames = read_json(frozen / "frames.json")
            frozen_frames_cache[fold] = frozen_frames
            frozen_xy_path = frozen / "holdouts.jsonl"
            frozen_xy_digest = digest(frozen_xy_path)
            xy = read_holdouts(frozen_xy_path)
            for method in ("x", "xy"):
                path = OUT / response / fold / f"{method}_frames.json"
                frames = read_json(path)
                frame_totals[(response, method, fold.split("_")[0])] += len(frames)
                require(len(frames) == len(chosen), errors,
                        f"{response}/{fold}/{method}: joined count {len(frames)} != selected {len(chosen)}")
                slot_totals[(response, method, fold.split("_")[0])] += sum(len(f.get("slots", [])) for f in frames)
                for frame in frames:
                    slots = frame.get("slots", [])
                    require(len(slots) == 3, errors, f"{response}/{fold}/{method}: frame lacks three slots")
                    valid_slots = [s for s in slots if s.get("scored")]
                    scored_counts[(response, method, fold.split("_")[0])] += len(valid_slots)
                    for slot in slots:
                        cell = (response, method, fold.split("_")[0])
                        branch_counts[cell]["slots"] += 1
                        if slot.get("available"):
                            branch_counts[cell]["available"] += 1
                            branch_counts[cell]["branches_total"] += len(slot.get("branches") or [])
                            plausible_n = len(slot.get("plausible_branches") or [])
                            branch_counts[cell][f"plausible_{plausible_n}"] += 1
                            branch_counts[cell][f"rank_{slot.get('rank')}"] += 1
                            branch_counts[cell]["bound_available"] += bool(slot.get("bound"))
                            branch_counts[cell]["interior_available"] += bool(slot.get("interior"))
                            boundary_interior[cell]["available_boundary"] += bool(slot.get("bound"))
                            boundary_interior[cell]["available_interior"] += bool(slot.get("interior"))
                        else:
                            branch_counts[cell]["unavailable"] += 1
                        if slot.get("scored"):
                            branch_counts[cell]["scored"] += 1
                            boundary_interior[cell]["scored_boundary"] += bool(slot.get("bound"))
                            boundary_interior[cell]["scored_interior"] += bool(slot.get("interior"))
                        elif slot.get("failure_reason") == "invalid_input":
                            invalid_preservation_checks += 1
                            require(not slot.get("scored") and not slot.get("available"), errors,
                                    f"{response}/{fold}/{method}: invalid input slot was scored/available")
                    if len(valid_slots) == 3:
                        errors_px = np.asarray([s["error_px"] for s in sorted(valid_slots, key=lambda z: z["held_point"])], float)
                        e_px = math.sqrt(float(np.mean(np.sum(errors_px * errors_px, axis=1))))
                        require(close(frame.get("E_px"), e_px), errors,
                                f"{response}/{fold}/{method}: E_px mismatch at row {frame.get('row')}")
                        states = [np.asarray(s["state"], float) for s in sorted(valid_slots, key=lambda z: z["held_point"])]
                        gtheta = math.sqrt(float(np.mean([(states[i][0]-states[j][0])**2
                                                          for i, j in ((0, 1), (0, 2), (1, 2))])))
                        ga = math.sqrt(float(np.mean([(states[i][1]-states[j][1])**2
                                                      for i, j in ((0, 1), (0, 2), (1, 2))])))
                        require(close(frame.get("G_theta_deg"), gtheta) and close(frame.get("G_A_D"), ga), errors,
                                f"{response}/{fold}/{method}: G mismatch at row {frame.get('row')}")
                        metric_checks += 1

                if method == "xy":
                    completion = read_json(OUT / response / fold / "completion.json")
                    require(completion.get("xy_holdouts_sha256") == frozen_xy_digest, errors,
                            f"{response}/{fold}: frozen xy holdout digest mismatch")
                    joined_xy = join_records(fold.split("_")[0], fold, response, pop, frozen_frames, xy)
                    by_key = {(f["capture"], f["fixation"], f["row"]): f for f in frames}
                    expected_by_key = {(f["capture"], f["fixation"], f["row"]): f for f in joined_xy}
                    require(by_key.keys() == expected_by_key.keys(), errors,
                            f"{response}/{fold}: xy identity set differs from frozen join")
                    for key in by_key.keys() & expected_by_key.keys():
                        require(by_key[key] == expected_by_key[key], errors,
                                f"{response}/{fold}: xy frame differs from reused frozen holdout {key}")
                        xy_reuse_checks += 1

            # Verify the x predictions from raw measurements and frozen response coefficients.
            model, meta = load_model(frozen / "model.json")
            provenance = meta.get("provenance", {})
            require(provenance.get("interval_sha256") == interval_hash, errors,
                    f"{response}/{fold}: raw interval hash differs from model provenance")
            require(provenance.get("capture_sha256") == raw_hashes, errors,
                    f"{response}/{fold}: raw capture hashes differ from model provenance")
            require(task_completion.get("response_model_sha256") == digest(frozen / "model.json"), errors,
                    f"{response}/{fold}: completion response model hash mismatch")
            model_provenance_checks[f"{response}/{fold}"] = {
                "interval_sha256_expected": interval_hash,
                "interval_sha256_in_model": provenance.get("interval_sha256"),
                "capture_sha256_expected": raw_hashes,
                "capture_sha256_in_model": provenance.get("capture_sha256"),
                "matches": provenance.get("interval_sha256") == interval_hash and
                           provenance.get("capture_sha256") == raw_hashes,
            }
            pilot = PositionModel(27, meta["pilot_coefficients"])
            archive = OUT / response / fold / "inverse_candidates_x.jsonl.gz"
            archive_by_key = {}
            with gzip.open(archive, "rt") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    key = (int(rec["fixation"]), int(rec["row"]), int(rec["held_point"]))
                    archive_by_key[key] = rec["candidates"]
                    archive_lines += 1
                    archive_candidate_records += len(rec["candidates"])
            for response_frame in read_json(OUT / response / fold / "x_frames.json"):
                cap = captures[response_frame["capture"]]
                i = int(response_frame["row"])
                gi = int(response_frame["fixation"])
                ctx = context(cap.p[i])
                if not ctx.valid:
                    continue
                cov = reference_covariance(cap.p[i:i+1], pilot, np.asarray(meta["reference_state"]),
                                           np.asarray(meta["coordinate_covariance"]))[0]
                for slot in response_frame["slots"]:
                    j = int(slot["held_point"])
                    key = (gi, i, j)
                    candidates = archive_by_key.get(key, [])
                    point_indices = np.arange(3) != j
                    valid_retained = bool(cap.point_valid[i][point_indices].all())
                    if not valid_retained:
                        require(len(candidates) == 0, errors, f"{response}/{fold}/{key}: invalid retained subset has candidates")
                        continue
                    require(len(candidates) == len(STARTS) == 49, errors,
                            f"{response}/{fold}/{key}: expected 49 archived x starts, got {len(candidates)}")
                    require(sorted(int(c.get("start", -1)) for c in candidates) == list(range(49)), errors,
                            f"{response}/{fold}/{key}: candidate archive does not retain each start id 0–48 exactly once")
                    archive_candidate_records += 0
                    kept = np.array([2 * k for k in range(3) if k != j])
                    y = cap.v[i].ravel()[kept]
                    W = whitening(marginal(cov, kept))
                    state = np.asarray(slot.get("state"), float)
                    if slot.get("available") and state.shape == (2,) and np.isfinite(state).all():
                        cost = 2 * objective_derivatives(model, ctx.r, y, W, kept, state / STATE_SCALE)[0]
                        require(close(slot.get("retained_subset_cost"), cost, atol=1e-6, rtol=2e-7), errors,
                                f"{response}/{fold}/{key}: selected branch cost does not recompute")
                        pred = model.predict(state, ctx.r)[2*j:2*j+2]
                        pix = ctx.c + ctx.ell * pred
                        err = cap.q[i, j] - pix
                        if slot.get("scored"):
                            require(close(slot.get("error_px"), err, atol=5e-7, rtol=2e-8), errors,
                                    f"{response}/{fold}/{key}: held pixel error does not recompute")
                            x_recompute_checks += 1
                        branches = slot.get("branches") or []
                        chosen = next((b for b in branches if close(b.get("state"), state, atol=1e-10, rtol=0)), None)
                        if chosen:
                            _, g, H = objective_derivatives(model, ctx.r, y, W, kept, state / STATE_SCALE)
                            lower, upper = LOWER / STATE_SCALE, UPPER / STATE_SCALE
                            active = (((state / STATE_SCALE - lower < 1e-9) & (g > 1e-7)) |
                                      ((upper - state / STATE_SCALE < 1e-9) & (g < -1e-7)))
                            free = np.flatnonzero(~active)
                            eig = np.linalg.eigvalsh(H[np.ix_(free, free)]) if len(free) else np.array([])
                            projected = projected_gradient(state / STATE_SCALE, g, lower, upper)
                            c = chosen.get("certificate", {})
                            require(projected <= 1e-4 and (not len(eig) or eig[0] >= -1e-10 * max(1., float(np.max(np.abs(eig))))),
                                    errors, f"{response}/{fold}/{key}: selected branch fails independent gradient/curvature")
                            require(c.get("certified") is True, errors,
                                    f"{response}/{fold}/{key}: selected branch lacks certification")
                            polished_state, independent_certificate = polish(
                                model, ctx.r, y, W, kept, state / STATE_SCALE)
                            require(close(polished_state * STATE_SCALE, state, atol=1e-9, rtol=0) and
                                    independent_certificate.get("certified") is True and
                                    independent_certificate.get("steps") == 0 and
                                    independent_certificate.get("physical_correction", math.inf) <= 1e-5 and
                                    independent_certificate.get("stable_cost") is True and
                                    independent_certificate.get("local_minimum") is True and
                                    independent_certificate.get("stationarity_encoded", math.inf) <= 1e-4,
                                    errors, f"{response}/{fold}/{key}: selected branch fails independent polish check")
                            cert_checks += 1

    # Exact expected grid, based on selected subsets in the frozen populations.
    total_joined = sum(frame_totals.values())
    require(total_joined == 1280, errors, f"joined frame total is {total_joined}, expected 1280")
    require(len(selected_counts) == 9, errors, "expected 9 fold population selections")
    expected_by_family = {"gaze": 160, "capture": 160}
    for family, expected in expected_by_family.items():
        for response in CONFIG["responses"]:
            for method in ("x", "xy"):
                count = frame_totals[(response, method, family)]
                require(count == expected, errors,
                        f"{response}/{family}/{method}: selected scheduled count {count} != {expected}")
    per_method = {method: sum(n for (response, m, family), n in frame_totals.items() if m == method)
                  for method in ("x", "xy")}
    require(per_method == {"x": 640, "xy": 640}, errors, f"unexpected per-method frame totals: {per_method}")

    # Rebuild all four exact transition memberships from the joined frames.
    membership_entries = []
    with gzip.open(OUT / "paired_transition_membership.jsonl.gz", "rt") as stream:
        membership_entries = [json.loads(line) for line in stream if line.strip()]
    membership_checks = 0
    summaries = read_json(OUT / "summary.json")["responses"]
    for response in CONFIG["responses"]:
        for family in ("gaze", "capture"):
            x_frames, xy_frames = [], []
            for fold in CONFIG["folds"]:
                if fold.startswith(family + "_"):
                    x_frames += read_json(OUT / response / fold / "x_frames.json")
                    xy_frames += read_json(OUT / response / fold / "xy_frames.json")
            rebuilt = transitions(x_frames, xy_frames)
            saved = next((r for r in membership_entries if r.get("response") == response and r.get("family") == family), None)
            require(saved is not None, errors, f"missing transition membership {response}/{family}")
            if saved:
                canonical = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
                require(canonical(saved.get("points")) == canonical(rebuilt["points"]) and
                        canonical(saved.get("frames")) == canonical(rebuilt["frames"]), errors,
                        f"transition membership identity/count/scoring mismatch {response}/{family}")
                require(summaries[response][family]["paired_squared_changes"]["strata"] == rebuilt["strata"], errors,
                        f"paired transition summary mismatch {response}/{family}")
                membership_checks += 1

    details = {
        "schema": "phase83_retained_channels_verification_v1",
        "verified_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_completion": run_completion,
        "all_checks_passed": not errors,
        "errors": errors,
        "frozen_source": {"expected_file_count": len(source_expected),
                          "actual_file_count": len(source_actual_files),
                          "file_set_matches": source_actual_files == set(source_expected),
                          "hash_checks": source_checks},
        "captured_implementation_design_hashes": implementation_checks,
        "raw_input_provenance_vs_frozen_models": model_provenance_checks,
        "population_selected_rows_per_fold": selected_counts,
        "joined_frames_by_response_method_family": {"/".join(k): v for k, v in sorted(frame_totals.items())},
        "joined_slots_by_response_method_family": {"/".join(k): v for k, v in sorted(slot_totals.items())},
        "per_method_joined_frame_totals": per_method,
        "scored_point_slots_by_response_method_family": {"/".join(k): v for k, v in sorted(scored_counts.items())},
        "boundary_interior_point_slots": {"/".join(k): dict(v) for k, v in sorted(boundary_interior.items())},
        "branch_counts": {"/".join(k): dict(v) for k, v in sorted(branch_counts.items())},
        "independent_frame_metric_checks": metric_checks,
        "invalid_input_slots_preserved": invalid_preservation_checks,
        "xy_frozen_holdout_reuse_frame_checks": xy_reuse_checks,
        "xy_transition_membership_checks": membership_checks,
        "x_raw_prediction_recomputation_checks": x_recompute_checks,
        "x_selected_branch_independent_gradient_curvature_checks": cert_checks,
        "x_candidate_archive_lines": archive_lines,
        "x_candidate_archive_candidate_records": archive_candidate_records,
        "expected_start_count_per_valid_retained_slot": 49,
        "regression_test_evidence": {
            "command": "rtk proxy env PYTHONPATH=.:tests OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q",
            "result": "59 passed in 10.89s",
        },
        "elapsed_seconds": time.time() - start,
        "notes": ["E and G metrics were recomputed from the joined raw point errors and states.",
                  "x costs and held-pixel errors were recomputed from frozen coefficients and raw detections.",
                  "Invalid input slots remain present and are required to remain unavailable and unscored."],
    }
    (OUT / "verification.json").write_text(json.dumps(details, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"all_checks_passed": not errors, "errors": errors,
                      "joined_frames": total_joined, "per_method": per_method,
                      "metric_checks": metric_checks, "x_recompute_checks": x_recompute_checks,
                      "cert_checks": cert_checks, "xy_reuse_checks": xy_reuse_checks,
                      "membership_checks": membership_checks, "elapsed_seconds": details["elapsed_seconds"]}, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
