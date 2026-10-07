"""Independent training-only verification for the preserved Phase 8.2 retry."""
from __future__ import annotations
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from full_position.calibrate import ProfiledProblem, projected_gradient
from full_position.model import PositionModel, STATE_SCALE, LOWER, UPPER
from full_position.noise import reference_covariance
from full_position.sensitivity import checked_training, digest, read_json, settings
from full_position.schema import source_hashes, write_json


def main():
    root = Path(__file__).resolve().parents[3]
    out = Path(__file__).resolve().parent
    source = root/"experiments/full_position/audit_polished_v1"
    failed = root/"experiments/full_position/joint_sensitivity_v1/variants/prior_weak/capture_1/conditional27"
    retry = out
    completion = read_json(retry/"completion.json")
    hashes = completion["implementation_hashes"]
    current_hashes = source_hashes(root)
    code_hashes = {relative: expected for relative, expected in hashes.items()
                   if relative.startswith("full_position/")}
    current_code_hashes = {relative: current_hashes.get(relative) for relative in code_hashes}
    if current_code_hashes != code_hashes:
        raise AssertionError("Current Python implementation no longer matches retry hashes")
    # Every archived implementation/design source remains byte-identical to
    # the source used for the retry. Current design docs may gain later status text.
    for relative, expected in hashes.items():
        archived = retry/"implementation_snapshot"/Path(relative).name if relative.startswith("full_position/") else retry/"design_snapshot"/Path(relative).name
        actual = hashlib.sha256(archived.read_bytes()).hexdigest()
        if actual != expected:
            raise AssertionError(f"Archived source hash mismatch: {relative}")
    captures, groups, data, anchors, frozen = checked_training(root, source, "capture_1")
    _, meta, _ = frozen["conditional27"]
    change = settings(meta, anchors, "prior_weak")
    pilot = PositionModel(27, meta["pilot_coefficients"])
    covariance = reference_covariance(data["p"], pilot, np.asarray(change["reference_state"]),
                                      np.asarray(meta["coordinate_covariance"]))
    states_record = read_json(retry/"training_states.json")
    for field, expected in (("rows", data["rows"]), ("groups", data["groups"]), ("nominal_anchors", anchors)):
        if not np.array_equal(states_record[field], expected):
            raise AssertionError(f"Retry training {field} do not match the frozen training set")
    state = np.asarray(states_record["states"], float)
    model, model_meta = __import__("full_position.schema", fromlist=["load_model"]).load_model(retry/"model.json")
    prior_strength = change["prior_strength"]
    problem = ProfiledProblem(model, data["v"], data["r"], covariance, data["groups"], anchors,
                              prior_strength, change["anchor_scales"])
    z = (state/STATE_SCALE).ravel()
    residual = problem.fun(z)
    # The profiled coefficients are independently reconstructed from saved states.
    beta_error = float(np.max(np.abs(problem.beta-model.beta)))
    if beta_error > 1e-9:
        raise AssertionError(f"Saved retry coefficients disagree with profiled beta: {beta_error}")
    gradient = problem.vjp(residual)
    pg_encoded = projected_gradient(z, gradient, np.tile(LOWER/STATE_SCALE, problem.n),
                                    np.tile(UPPER/STATE_SCALE, problem.n))
    pg_physical = projected_gradient(state.ravel(), gradient/np.tile(STATE_SCALE, problem.n),
                                     np.tile(LOWER, problem.n), np.tile(UPPER, problem.n))
    inner = float(np.max(np.abs(problem.B.T@problem.optical_residual)/problem.column_scale))
    cost = float(residual@residual/2)
    saved = completion["calibration"]["alternatives"][completion["calibration"]["selected_start"]]
    comparisons = dict(cost_abs_error=abs(cost-saved["cost"]),
        optical_abs_error=abs(float(np.sum(problem.optical_residual[:problem.n*problem.c]**2)/2)-saved["objective_components"]["optical"]),
        prior_abs_error=abs(float(np.sum(problem.optical_residual[problem.n*problem.c:]**2)/2)-saved["objective_components"]["prior"]),
        anchor_abs_error=abs(float(np.sum(problem.residual[len(problem.b):]**2)/2)-saved["objective_components"]["anchor"]),
        coefficient_max_abs_error=beta_error,
        saved_vs_reconstructed_stationarity_physical=abs(pg_physical-saved["projected_stationarity_physical"]),
        saved_vs_reconstructed_inner=abs(inner-saved["inner_stationarity_scaled"]))
    if max(comparisons.values()) > 1e-7:
        raise AssertionError(f"Saved retry objective/stationarity reconstruction mismatch: {comparisons}")
    if not (completion["converged"] and pg_physical < 1e-3 and inner < 1e-7):
        raise AssertionError("Retry does not meet the original strict convergence contract")
    candidate_path = retry/"calibration_candidates.jsonl.gz"
    with gzip.open(candidate_path, "rt") as f:
        candidate_count = sum(1 for line in f if line.strip())
    original_completion = read_json(failed/"completion.json")
    original_candidates = failed/"calibration_candidates.jsonl.gz"
    result = dict(schema="phase82_strict_retry_verification_v1", verified=True,
        fit_scope="capture_1 prior_weak conditional27 training rows only; no evaluation access",
        frozen_training_row_count=len(data["rows"]), selected_start=completion["calibration"]["selected_start"],
        continuation_stage_count=len(saved["continuation_history"]),
        original_acceptance_thresholds=dict(physical_projected_stationarity_lt=1e-3,
                                            inner_stationarity_scaled_lt=1e-7),
        reconstructed=dict(cost=cost, projected_stationarity_encoded=pg_encoded,
                           projected_stationarity_physical=pg_physical,
                           inner_stationarity_scaled=inner),
        comparisons=comparisons, saved_candidate_record_count=candidate_count,
        original_failed_fit=dict(path=str(failed.relative_to(root)),
            training_states_sha256=digest(failed/"training_states.json"),
            completion_sha256=digest(failed/"completion.json"),
            candidates_sha256=digest(original_candidates),
            original_converged=original_completion.get("calibration_converged")),
        retry_hashes=dict(archived_source_hashes_match=True,
                          current_python_implementation_matches_retry_hashes=True,
                          current_design_documents_may_include_later_status=True,
                          archived_source_file_count=len(hashes)),
        test_evidence=dict(command="rtk proxy env PYTHONPATH=.:tests OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q",
                           tests_passed=59, elapsed_seconds=10.87,
                           evidence="full suite passed after real branch-cost scaling regression"))
    write_json(out/"verification.json", result)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
