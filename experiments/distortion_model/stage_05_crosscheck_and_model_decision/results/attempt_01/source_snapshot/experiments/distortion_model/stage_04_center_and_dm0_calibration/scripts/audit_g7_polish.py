#!/usr/bin/env python3
"""Independent public-optics/objective reconstruction of saved G7 attempt04."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "docs/Theory.md").is_file())
STAGE = ROOT / "experiments/distortion_model/stage_04_center_and_dm0_calibration"
sys.path.insert(0, str(ROOT))

from distortion_model.geometry import relative_coordinates
from distortion_model.joint import JointSpec
from distortion_model.optics import predict_relative


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_hashes(folder: Path) -> dict[str, str]:
    return {str(path.relative_to(folder)): digest(path)
            for path in folder.rglob("*") if path.is_file()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", type=Path, default=STAGE / "results/g7_attempt_04")
    args = parser.parse_args()
    attempt = args.attempt.resolve()
    runner_path = STAGE / "scripts/run_joint.py"
    loader = importlib.util.spec_from_file_location("audit_g7_polish_runner", runner_path)
    runner = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(runner)

    parent03 = STAGE / "results/g7_attempt_03"
    parent_before = file_hashes(parent03)
    inputs, _, g6 = runner.check_parent(STAGE / "results/g6_attempt_02")
    spec = JointSpec(np.array(g6["b1_reference_px"]), np.array(g6["b4_reference_px"]),
                     g6["omega1_visual_deg"], g6["omega4_visual_deg"], g6["Aref_D"])
    fit = json.loads((attempt / "polished/fit.json").read_text())
    with np.load(attempt / "polished/solution.npz", allow_pickle=False) as archive:
        states = archive["states"].copy()
        globals_ = archive["scaled_globals"].copy()
    with np.load(attempt / "fitted.npz", allow_pickle=False) as archive:
        stored = {key: archive[key].copy() for key in
                  ("input_valid", "g", "prediction", "theta_visual_deg", "accommodation_D")}

    population = inputs["population"]
    valid = inputs["valid"]
    exposure = inputs["exposure"]
    observed = relative_coordinates(population["p1"][valid], population["p4"][valid])
    assert np.array_equal(valid, stored["input_valid"])
    assert states.shape == (int(valid.sum()), 2) and globals_.shape == (19,)
    state_theta_match = bool(np.array_equal(states[:, 0], stored["theta_visual_deg"][valid]))
    state_accommodation_match = bool(np.array_equal(states[:, 1], stored["accommodation_D"][valid]))
    unavailable_nan = {key: bool(np.all(np.isnan(stored[key][~valid])))
                       for key in ("g", "prediction", "theta_visual_deg", "accommodation_D")}
    assert state_theta_match and state_accommodation_match and all(unavailable_nan.values())
    p1 = population["p1"][valid]
    p1_edges = np.concatenate((p1[:, 1] - p1[:, 0], p1[:, 2] - p1[:, 0]), axis=1)
    parameters = spec.parameters(globals_)
    prediction, scale, optics_valid = predict_relative(
        states[:, 0], states[:, 1], p1_edges, inputs["covariance"]["R11"], parameters)

    residual = observed - prediction
    precision = np.linalg.solve(inputs["covariance"]["relative"], np.eye(10))
    counts = np.bincount(exposure, minlength=20)
    weights = 1.0 / (20.0 * counts[exposure])
    point = .5 * np.sum(weights * np.einsum("ni,ij,nj->n", residual, precision, residual))
    targets = np.array([[population["target_theta_deg"][np.flatnonzero(population["exposure"] == k)[0]],
                         population["demand_diopters"][np.flatnonzero(population["exposure"] == k)[0]]]
                        for k in range(20)])
    means = np.stack([states[exposure == k].mean(axis=0) for k in range(20)])
    deviations = means - targets
    theta_anchor = .5 * np.sum(deviations[:, 0] ** 2 / .10 ** 2) / 20
    accommodation_anchor = .5 * np.sum(deviations[:, 1] ** 2 / .25 ** 2) / 20
    regularization = (.5 * np.sum((globals_[3:6] / spec.template_sigma) ** 2)
                      + .5 * np.sum(globals_[17:19] ** 2))
    independent = {"point": float(point), "theta_anchor": float(theta_anchor),
                   "A_anchor": float(accommodation_anchor), "regularization": float(regularization),
                   "temporal": 0.0}
    independent["cost"] = sum(independent.values())
    saved = fit["components"]
    component_differences = {key: independent[key] - saved[key] for key in independent}
    prediction_error = float(np.max(np.abs(prediction - stored["prediction"][valid])))
    scale_error = float(np.max(np.abs(scale - stored["g"][valid])))
    means_error = float(np.max(np.abs(means - np.asarray(fit["means"]))))
    all_valid = bool(np.all(optics_valid))
    assert all_valid and max(map(abs, component_differences.values())) < 1e-8
    assert prediction_error < 1e-8 and scale_error < 1e-8 and means_error < 1e-8
    parent_after = file_hashes(parent03)
    assert parent_before == parent_after

    result = {
        "kind": "independent public-optics and full-objective reconstruction; no inference, optimization, or tests",
        "audit_script": str(Path(__file__).resolve().relative_to(ROOT)),
        "audit_script_sha256": digest(Path(__file__).resolve()),
        "attempt": str(attempt.relative_to(ROOT)),
        "population_rows": int(valid.sum()), "all_public_optics_valid": all_valid,
        "all_saved_frame_arrays_match_input_mask": True,
        "saved_state_theta_exactly_matches_solution": state_theta_match,
        "saved_state_accommodation_exactly_matches_solution": state_accommodation_match,
        "unavailable_frame_arrays_are_nan": unavailable_nan,
        "reconstructed_components": independent, "saved_components": saved,
        "component_differences_independent_minus_saved": component_differences,
        "prediction_max_abs_difference_px": prediction_error,
        "g_max_abs_difference": scale_error,
        "exposure_mean_max_abs_difference": means_error,
        "point_loss_definition": "equal-exposure weighted full ten-coordinate covariance quadratic on all complete valid rows",
        "gaze_anchor_definition": "mean of 20 exposure mean theta deviations; .10 degree sigma",
        "accommodation_anchor_definition": "mean of 20 exposure mean A deviations; .25 D sigma",
        "prior_definition": "half squared template coordinates p[3:6]/template_sigma plus half squared D coefficients p[17:19]",
        "parent_attempt03_hashes_unchanged": parent_before == parent_after,
        "parent_attempt03_file_count": len(parent_before),
        "passed_tolerances": {"components_abs_lt_1e-8": True, "prediction_abs_lt_1e-8_px": True,
                              "g_abs_lt_1e-8": True, "means_abs_lt_1e-8": True},
    }
    (attempt / "independent_public_optics_audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in
                      ("population_rows", "all_public_optics_valid",
                       "component_differences_independent_minus_saved",
                       "prediction_max_abs_difference_px", "g_max_abs_difference",
                       "exposure_mean_max_abs_difference", "parent_attempt03_hashes_unchanged")}, indent=2))


if __name__ == "__main__":
    main()
