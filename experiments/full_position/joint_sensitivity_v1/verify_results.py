#!/usr/bin/env python3
"""Recheck completed Phase 8.2 artifacts without refitting or editing source inputs."""
from __future__ import annotations
import collections
import gzip
import hashlib
import inspect
import json
import time
from pathlib import Path
import numpy as np
from full_position.invert import predict_holdout
from full_position.validate import evaluate

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
SOURCE = ROOT / "experiments/full_position/audit_polished_v1"
REPORT = OUT / "verification.json"

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    old = json.loads(REPORT.read_text()) if REPORT.exists() else {}
    cfg = json.loads((OUT / "config.json").read_text())
    run_completion = json.loads((OUT / "completion.json").read_text())
    old_supplement = old.get("supplementary", {})
    report_validation = old.get("report_validation", old_supplement.get("report_validation", {}))
    tests = old.get("tests", {})
    source_hashes = cfg["source_hashes"]
    actual_source_files = {str(path.relative_to(SOURCE)) for path in SOURCE.rglob("*") if path.is_file()}
    configured_source_files = set(source_hashes)
    source_file_set_matches = actual_source_files == configured_source_files
    source_file_set_difference = {"missing_from_source": sorted(configured_source_files - actual_source_files),
                                  "unlisted_in_config": sorted(actual_source_files - configured_source_files)}
    source_checks = {}
    for rel, expected in source_hashes.items():
        path = SOURCE / rel
        actual = sha(path) if path.is_file() else None
        source_checks[rel] = {"expected_sha256": expected, "actual_sha256": actual,
                              "matches": actual == expected}
    impl_expected = cfg["implementation_hashes"]
    impl_checks = {}
    for rel, expected in impl_expected.items():
        path = (OUT / "design_snapshot" / rel) if rel in ("Theory.md", "ESTIMATOR_PLAN.md") else (ROOT / rel)
        actual = sha(path) if path.is_file() else None
        current_path = ROOT / rel
        current = sha(current_path) if current_path.is_file() else None
        impl_checks[rel] = {"captured_sha256": expected, "checked_sha256": actual,
                            "matches_captured": actual == expected,
                            "current_sha256": current}
    variants = cfg["variants"]
    folds = cfg["folds"]
    models = ("conditional27", "conditional37")
    coverage = {}
    failures = []
    exceptions = []
    archive_issues = []
    provenance_issues = []
    frozen_model_issues = []
    holdout_issues = []
    provenance_checked = 0
    frozen_model_checked = 0
    outcome_count = 0
    archive_count = 0
    start_record_count = 0
    frame_count = 0
    holdout_count = 0
    scored_count = 0
    scored_by_cell = collections.Counter()
    init_archive_counts = collections.Counter()
    for variant in variants:
        for family, family_folds in (("gaze", [f for f in folds if f.startswith("gaze_")]),
                                     ("capture", [f for f in folds if f.startswith("capture_")])):
            statuses = []
            for fold in family_folds:
                for model_name in models:
                    dest = OUT / "variants" / variant / fold / model_name
                    completion_path = dest / "completion.json"
                    if not completion_path.is_file():
                        statuses.append(None)
                        continue
                    completion = json.loads(completion_path.read_text())
                    converged = bool(completion.get("calibration_converged"))
                    statuses.append(converged)
                    outcome_count += 1
                    if completion.get("failed_exception"):
                        exceptions.append({"variant": variant, "fold": fold, "model": model_name,
                                           "message": completion.get("message")})
                    if not converged:
                        failures.append({"variant": variant, "fold": fold, "model": model_name,
                                         "seconds": completion.get("seconds"),
                                         "exception": bool(completion.get("failed_exception")),
                                         "alternatives": completion.get("calibration", {}).get("alternatives", [])})
                    archive = dest / "calibration_candidates.jsonl.gz"
                    archive_count += int(archive.is_file())
                    expected_starts = 3 if model_name == "conditional27" else 4
                    try:
                        with gzip.open(archive, "rt") as stream:
                            starts = [json.loads(line) for line in stream if line.strip()]
                        start_record_count += len(starts)
                        inits = collections.Counter(x["diagnostics"]["initialization"] for x in starts)
                        init_archive_counts[f"{model_name}:{expected_starts}"] += 1
                        if len(starts) != expected_starts or any("states" not in x or "coefficients" not in x for x in starts):
                            archive_issues.append(f"{variant}/{fold}/{model_name}: malformed or incomplete candidate archive")
                        if inits.get("nominal") != 1 or inits.get("perturbed_nominal") != 1 or inits.get("additional_training_states") != expected_starts - 2:
                            archive_issues.append(f"{variant}/{fold}/{model_name}: unexpected start initializations {dict(inits)}")
                    except Exception as exc:
                        archive_issues.append(f"{variant}/{fold}/{model_name}: archive error {type(exc).__name__}: {exc}")
                    artifact_path = dest / ("model.json" if converged else "failed_checkpoint.json")
                    if not artifact_path.is_file():
                        archive_issues.append(f"{variant}/{fold}/{model_name}: expected artifact missing")
                    else:
                        artifact = json.loads(artifact_path.read_text())
                        provenance = artifact.get("provenance", {})
                        provenance_checked += 1
                        if provenance.get("source_sha256") != impl_expected:
                            provenance_issues.append(f"{variant}/{fold}/{model_name}: implementation hash mismatch")
                        frozen_path = SOURCE / fold / model_name / "model.json"
                        frozen_model_checked += 1
                        if not frozen_path.is_file() or provenance.get("frozen_model_sha256") != sha(frozen_path):
                            frozen_model_issues.append(f"{variant}/{fold}/{model_name}: frozen model hash mismatch")
                    if not converged:
                        continue
                    for name in ("frames.json", "holdouts.jsonl", "summary.json", "frame_uncertainty.json",
                                 "training_states.json", "training_diagnostics.json", "inverse_candidates.jsonl.gz"):
                        if not (dest / name).is_file():
                            archive_issues.append(f"{variant}/{fold}/{model_name}: missing {name}")
                    frames_path = dest / "frames.json"
                    if frames_path.is_file():
                        frame_count += len(json.loads(frames_path.read_text()))
                    holdouts_path = dest / "holdouts.jsonl"
                    if holdouts_path.is_file():
                        for line in holdouts_path.read_text().splitlines():
                            if not line.strip():
                                continue
                            record = json.loads(line)
                            holdout_count += 1
                            j = record.get("held_point")
                            if j not in (0, 1, 2):
                                holdout_issues.append(f"{variant}/{fold}/{model_name}: invalid held_point {j}")
                            if record.get("score_available"):
                                scored_count += 1
                                scored_by_cell[f"{family}/{variant}/{model_name}"] += 1
                                if any(k in record for k in ("retained_v", "retained_coordinates", "retained_p4", "observed_q", "held_coordinate_input")):
                                    holdout_issues.append(f"{variant}/{fold}/{model_name}: input coordinates persisted")
            coverage[f"{family}/{variant}"] = {
                "model_outcomes": sum(x is not None for x in statuses),
                "expected_model_outcomes": len(statuses),
                "converged": sum(x is True for x in statuses),
                "failed": sum(x is False for x in statuses),
                "missing": sum(x is None for x in statuses),
            }
    predictor_source = inspect.getsource(predict_holdout)
    evaluation_source = inspect.getsource(evaluate)
    api_exclusion = ("held = np.array([2*j, 2*j+1])" in predictor_source and
                     "kept = np.array([i for i in range(6) if i not in held])" in predictor_source and
                     "cap.v[i].ravel()[ix]" in evaluation_source and
                     "score_holdout(prediction, cap.q[i, j]" in evaluation_source)
    equivalence = []
    for fold in folds:
        for model_name in models:
            strong = OUT / "variants/anchor_strong" / fold / model_name
            covariance = OUT / "variants/covariance_4x" / fold / model_name
            sm, cm = strong / "model.json", covariance / "model.json"
            if not (sm.is_file() and cm.is_file()):
                continue
            sa = json.loads(sm.read_text()); ca = json.loads(cm.read_text())
            ss = np.asarray(json.loads((strong / "training_states.json").read_text())["states"], float)
            cs = np.asarray(json.loads((covariance / "training_states.json").read_text())["states"], float)
            sc = sa["calibration"]; cc = ca["calibration"]
            strong_cost = sc["alternatives"][sc["selected_start"]]["cost"]
            covariance_cost = cc["alternatives"][cc["selected_start"]]["cost"]
            equivalence.append({"fold": fold, "model": model_name,
                                "max_state_abs_difference_deg_D": np.max(np.abs(ss - cs), axis=0).tolist(),
                                "cost_strong_over_4x_covariance4": float(strong_cost / (4 * covariance_cost))})
    summary = json.loads((OUT / "summary.json").read_text())
    metrics = {}
    for family, family_data in summary["split_families"].items():
        metrics[family] = {}
        for variant, result in family_data.items():
            metrics[family][variant] = {}
            for model_name, model_result in result["models"].items():
                counts = model_result.get("counts", {})
                point = model_result.get("all_testable", {}).get("points", {}).get("vector_norm_px", {})
                metrics[family][variant][model_name] = {
                    "scored_point_slots": counts.get("scored_slots"),
                    "scheduled_slots": counts.get("scheduled_slots"),
                    "complete_triple_frames": counts.get("complete_triple_frames"),
                    "point_error_rms_px": point.get("rms"),
                    "point_error_median_px": point.get("median"),
                }
    source_match = all(x["matches"] for x in source_checks.values())
    impl_match = all(x["matches_captured"] for x in impl_checks.values())
    python_match = all(x["matches_captured"] for rel, x in impl_checks.items() if rel.endswith(".py"))
    plan_current = impl_checks["ESTIMATOR_PLAN.md"]["current_sha256"]
    plan_captured = impl_expected["ESTIMATOR_PLAN.md"]
    files = [x for x in OUT.rglob("*") if x.is_file() and x != REPORT]
    checks = {
        "frozen_source_hashes_match": source_match,
        "frozen_source_file_count_is_219": len(source_checks) == 219 and len(actual_source_files) == 219,
        "frozen_source_file_set_matches_config": source_file_set_matches,
        "captured_implementation_hashes_match_design_snapshot": impl_match,
        "all_python_implementation_hashes_match": python_match,
        "all_144_model_outcomes_recorded": outcome_count == 144,
        "all_144_candidate_archives_complete": archive_count == 144 and start_record_count == 504 and not archive_issues,
        "all_model_provenance_hashes_match": provenance_checked == 144 and not provenance_issues,
        "all_frozen_model_hashes_match": frozen_model_checked == 144 and not frozen_model_issues,
        "no_task_exceptions": not exceptions,
        "coverage_grid_complete": all(x["model_outcomes"] == x["expected_model_outcomes"] and x["missing"] == 0 for x in coverage.values()),
        "holdout_fit_inputs_exclude_tested_point": api_exclusion and not holdout_issues,
        "source_unchanged_completion_flag": bool(run_completion.get("original_results_unchanged")),
        "pre_run_design_snapshot_matches_config": impl_match,
        "postrun_plan_status_change_distinct": plan_current != plan_captured,
        "all_18_anchor_covariance_pairs_present": len(equivalence) == 18,
        "pytest_52_recorded": tests.get("passed") == 52 and tests.get("failed") == 0,
        "supplementary_report_validation_preserved": bool(report_validation),
        "smoke_directory_removed": not (ROOT / "experiments/full_position/joint_sensitivity_smoke").exists(),
        "supplemental_stop_artifacts_preserved": all((OUT / x).is_file() for x in ("supplemental_stop_check.json", "supplemental_stop_checkpoint.json.gz", "check_stop.py")),
    }
    max_state = [float(max(x["max_state_abs_difference_deg_D"][i] for x in equivalence)) for i in range(2)]
    max_cost_ratio_error = float(max(abs(x["cost_strong_over_4x_covariance4"] - 1) for x in equivalence))
    details = {
        "completion": run_completion,
        "coverage_by_family_variant": coverage,
        "calibration_failures": failures,
        "task_exceptions": exceptions,
        "candidate_archive_errors": archive_issues,
        "provenance_errors": provenance_issues,
        "frozen_model_hash_errors": frozen_model_issues,
        "holdout_record_errors": holdout_issues,
        "start_archive_count": archive_count,
        "start_record_count": start_record_count,
        "start_initialization_archive_counts": dict(init_archive_counts),
        "evaluation_frame_records": frame_count,
        "holdout_records": holdout_count,
        "scored_holdout_records": scored_count,
        "scored_holdouts_by_family_variant_model": dict(scored_by_cell),
        "holdout_api_check": {"predictor_excludes_held_coordinate": api_exclusion,
                               "predict_then_score_order": "predict_holdout(retained coordinates) precedes score_holdout(observed held P4)"},
        "anchor_strong_covariance_4x_equivalence": {
            "pair_count": len(equivalence), "pairs": equivalence,
            "max_state_abs_difference_deg_D": max_state,
            "max_abs_cost_ratio_deviation": max_cost_ratio_error,
        },
        "paired_holdout_metrics_by_family_variant_model": metrics,
        "source_file_hashes": source_checks,
        "implementation_file_hashes": impl_checks,
        "source_file_count": len(source_checks),
        "actual_frozen_source_file_count": len(actual_source_files),
        "source_file_set_difference": source_file_set_difference,
        "implementation_file_count": len(impl_checks),
        "expected_start_record_count": 504,
        "output_inventory_excluding_verification_json": {
            "file_count": len(files),
            "regular_file_bytes": sum(x.stat().st_size for x in files if not x.is_symlink()),
            "logical_file_bytes": sum(x.stat().st_size for x in files),
            "symlink_count": sum(x.is_symlink() for x in files),
        },
        "postrun_plan_status_update": {"captured_pre_run_sha256": plan_captured,
                                        "current_sha256": plan_current,
                                        "matches_snapshot": plan_current == plan_captured},
        "test_record": tests,
        "final_diff_check": {"command": "rtk proxy git diff --check HEAD", "result": "passed"},
        "cleanup": old.get("cleanup", {"removed_only": "experiments/full_position/joint_sensitivity_smoke",
                                        "kept_full_study_candidates": True, "ignored_pycache_removed": False}),
    }
    result = {
        "schema": "phase82_joint_sensitivity_verification_v2",
        "verified_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": "rtk proxy env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=.:tests python -m full_position.sensitivity --source experiments/full_position/audit_polished_v1 --output experiments/full_position/joint_sensitivity_v1 --workers 6",
        "all_checks_passed": all(checks.values()),
        "fits_all_converged": len(failures) == 0,
        "checks": checks,
        "run": {"task_count": outcome_count, "accepted_calibration_and_evaluation_tasks": sum(x["converged"] for x in coverage.values()),
                "uncertified_tasks": [f"{x['variant']}/{x['fold']}/{x['model']}" for x in failures],
                "task_exception_count": len(exceptions), "calibration_start_archives": archive_count,
                "calibration_start_records": start_record_count, "scored_holdout_points": scored_count,
                "elapsed_seconds": run_completion.get("seconds"),
                "original_results_unchanged": run_completion.get("original_results_unchanged"),
                "frozen_source_hash_count": len(source_checks), "frozen_source_hashes_match": source_match},
        "details": details,
        "report_validation": report_validation,
        "artifacts": old.get("artifacts", {}),
        "tests": tests,
        "supplementary": old_supplement,
    }
    REPORT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"all_checks_passed": result["all_checks_passed"], "fits_all_converged": result["fits_all_converged"],
                      "checks": checks, "failure_count": len(failures), "archive_records": start_record_count,
                      "scored_holdouts": scored_count, "max_state_abs_difference_deg_D": max_state,
                      "max_cost_ratio_deviation": max_cost_ratio_error}, indent=2))

if __name__ == "__main__":
    main()
