#!/usr/bin/env python3
"""Finalize first-pass P4 outputs using the lowest-objective stationary start."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
sys.path.insert(0, str(ROOT))

from distortion_model.p4_scale_comparison import ScaleComparisonProblem
from experiments.reverse_transform.p4_gaze_accommodation_decoupling.stage_00_baseline_sensitivity.p4_p1_scale_effect import run as runner


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(runner.safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    started = time.monotonic()
    summary_path = OUT / "summary.json"
    arrays_path = OUT / "arrays.npz"
    summary = json.loads(summary_path.read_text())
    with np.load(arrays_path, allow_pickle=False) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    first_summary_path = OUT / "summary_first_pass.json"
    if not first_summary_path.exists():
        first_summary_path.write_text(summary_path.read_text())
    for branch in ("baseline_g1", "p1m"):
        first_checkpoint = OUT / f"{branch}_fit_checkpoint_first_pass.json"
        current_checkpoint = OUT / f"{branch}_fit_checkpoint.json"
        if current_checkpoint.exists() and not first_checkpoint.exists():
            first_checkpoint.write_text(current_checkpoint.read_text())

    capture = np.asarray(arrays["capture"], dtype=np.int64)
    source = np.asarray(arrays["source_row"], dtype=np.int64)
    fixation = np.asarray(arrays["fixation_id"], dtype=np.int64)
    theta = np.asarray(arrays["saved_gaze_theta_deg"], dtype=np.float64)
    X = np.asarray(arrays["P4_observed"], dtype=np.float64)
    m = np.asarray(arrays["p1_m"], dtype=np.float64)
    reference = np.asarray(arrays["reference_B4_initial_px"], dtype=np.float64)
    R0 = float(arrays["frozen_R04_px"])
    demand = np.asarray(arrays["nominal_demand_diopters"], dtype=np.float64)
    target = np.asarray(arrays["nominal_theta_deg"], dtype=np.float64)
    problems = {
        "baseline_g1": ScaleComparisonProblem(X, theta, capture, fixation, reference,
            g=np.ones(len(capture)), R0=R0),
        "p1m": ScaleComparisonProblem(X, theta, capture, fixation, reference,
            g=m, R0=R0),
    }
    all_parameter_rows, multistart_rows, fixation_rows = [], [], []
    for branch, problem in problems.items():
        fit = summary["fit"][branch]
        starts = fit["starts"]
        stationary = [record for record in starts if record["stationary"]]
        if not stationary:
            raise ValueError(f"{branch}: no stationary start available for checkpoint selection")
        first_min = min(starts, key=lambda record: float(record["objective"]))
        selected = min(stationary, key=lambda record: float(record["objective"]))
        params = np.asarray(selected["params"], dtype=np.float64)
        evaluation = problem.evaluate(params)
        diagnostics = problem.diagnostics(params)
        fit["sensitivity_scaled_gradient"] = diagnostics["sensitivity_scaled_gradient"]
        fit["projected_gradient_norm"] = diagnostics["projected_gradient_norm"]
        fit["gradient_norm"] = diagnostics["gradient_norm"]
        fit["scaled_gradient"] = diagnostics["scaled_gradient"]
        if not evaluation["domain_valid"] or not np.all(evaluation["valid"]):
            raise ValueError(f"{branch}: selected stationary checkpoint is outside the all-row domain")
        selected_loss_delta = float(selected["objective"]-first_min["objective"])
        if not np.isclose(evaluation["objective"], selected["objective"], rtol=1e-10, atol=1e-12):
            raise ValueError(f"{branch}: independent objective recomputation differs from saved start")

        fit.update({key: selected[key] for key in (
            "params", "objective", "success", "reason", "stationary",
            "projected_gradient_norm", "gradient_norm", "scaled_gradient", "iterations", "seconds")})
        fit["loss"] = float(selected["objective"])
        fit["best_start"] = int(selected["start"])
        fit["first_pass_best_start"] = int(first_min["start"])
        fit["first_pass_best_objective"] = float(first_min["objective"])
        fit["first_pass_best_stationary"] = bool(first_min["stationary"])
        fit["stationarity_selection_rule"] = "minimum-objective stationary multistart; identical rule applied to both branches"
        fit["stationary_objective_excess_over_first_pass_minimum"] = selected_loss_delta
        fit["independent_objective_recomputed"] = float(evaluation["objective"])
        fit["selected_start_record"] = selected

        # The target roster is identical for all captures; preserve it by the
        # saved per-row labels and assert each fixation has one nominal value.
        target_by_capture = {}
        for cid in range(1, 5):
            target_by_capture[cid] = np.asarray([
                float(target[(capture == cid) & (fixation == fid)][0]) for fid in range(5)])
        row_rms, valid, fixation_part, capture_metrics = runner.capture_and_fixation_metrics(
            branch, evaluation["residual"], evaluation["valid"], capture, fixation, theta, target_by_capture)
        fixation_rows.extend(fixation_part)
        for cid in range(1, 5):
            metric = summary["capture_metrics"][str(cid)]
            metric[branch] = capture_metrics[str(cid)]
        arrays[f"{branch}_prediction"] = evaluation["prediction"]
        arrays[f"{branch}_residual"] = evaluation["residual"]
        arrays[f"{branch}_valid"] = valid.astype(np.uint8)
        arrays[f"{branch}_row_vertex_rms_px"] = row_rms
        arrays[f"{branch}_denominator"] = evaluation["denominator"]
        arrays[f"{branch}_radial_factor"] = evaluation["radial_factor"]
        arrays[f"{branch}_radial_derivative"] = evaluation["radial_derivative"]
        arrays[f"{branch}_stretch_exponent"] = evaluation["exponent"]
        arrays[f"{branch}_domain_margin"] = np.minimum.reduce((
            np.min(evaluation["denominator"], axis=1)-1e-8,
            np.min(evaluation["radial_factor"], axis=1)-1e-8,
            np.min(evaluation["radial_derivative"], axis=1)-1e-8,
            20.-np.abs(evaluation["exponent"])))
        arrays[f"{branch}_params"] = params
        arrays[f"{branch}_B_fit_px"] = diagnostics["B_px"]
        arrays[f"{branch}_diagnostic_singular_values"] = diagnostics["singular_values"]
        arrays[f"{branch}_diagnostic_right_vectors_scaled"] = diagnostics["right_singular_vectors_scaled"]

        summary["domain"][branch] = {**evaluation["domain"], "domain_valid": evaluation["domain_valid"],
            "box_valid": evaluation["box_valid"], "invalid_rows": int(np.sum(~evaluation["valid"])),
            "minimum_row_domain_margin": float(np.min(arrays[f"{branch}_domain_margin"])),
            "per_capture_core": diagnostics["per_capture"]}
        summary["stationarity"][branch] = {key: diagnostics[key] for key in (
            "stationary", "gradient_norm", "scaled_gradient", "projected_gradient_norm",
            "sensitivity_scaled_gradient", "active_bounds", "bound_contacts")}
        summary["identifiability"][branch] = {key: diagnostics[key] for key in (
            "rank", "parameters", "singular_values", "right_singular_vectors_scaled", "rank_rtol",
            "rank_tolerance", "condition_number", "spectrum_definition", "gauge", "scale_identifiability",
            "parameter_names", "parameter_scales", "weight_groups")}
        nominal_demand = {cid: float(demand[capture == cid][0]) for cid in range(1, 5)}
        all_parameter_rows.extend(runner.parameter_rows(branch, fit, diagnostics, nominal_demand))
        for start_record in starts:
            multistart_rows.append(dict(branch=branch, start=int(start_record["start"]),
                initial_params=json.dumps(runner.safe(start_record["initial_params"]), separators=(",", ":")),
                params=json.dumps(runner.safe(start_record["params"]), separators=(",", ":")),
                objective=float(start_record["objective"]), selected=bool(start_record["start"] == selected["start"]),
                success=bool(start_record["success"]), stationary=bool(start_record["stationary"]),
                reason=start_record["reason"], projected_gradient_norm=float(start_record["projected_gradient_norm"]),
                iterations=int(start_record["iterations"]), runtime_seconds=float(start_record["seconds"])))
        checkpoint = {
            "branch": branch, "selection_rule": fit["stationarity_selection_rule"],
            "selected_start": selected, "first_pass_best_start": first_min,
            "starts": starts, "selected_diagnostics": diagnostics,
        }
        write_json(OUT / f"{branch}_fit_checkpoint.json", checkpoint)
        np.savez_compressed(OUT / f"{branch}_selected_fit_checkpoint.npz",
            params=params, prediction=evaluation["prediction"], residual=evaluation["residual"],
            valid=evaluation["valid"].astype(np.uint8), row_vertex_rms_px=row_rms,
            B_fit_px=diagnostics["B_px"], singular_values=diagnostics["singular_values"],
            source_capture=capture, source_row=source, fixation_id=fixation)

    baseline_values = {(row["capture"], row["parameter"]): row["value"]
                       for row in all_parameter_rows if row["branch"] == "baseline_g1"}
    for row in all_parameter_rows:
        baseline = baseline_values.get((row["capture"], row["parameter"]))
        row["delta_vs_g1"] = None if baseline is None else float(row["value"]-baseline)
    write_csv(OUT / "parameters.csv", all_parameter_rows)
    write_csv(OUT / "multistarts.csv", multistart_rows)
    write_csv(OUT / "fixation_summary.csv", fixation_rows)
    np.savez_compressed(arrays_path, **arrays)
    summary["selection_repair"] = {
        "rule": "choose minimum-objective stationary start per branch; all branches now have a certified stationary selection",
        "continuation_run": False,
        "independent_all_row_objective_recomputed": True,
        "repair_runtime_seconds": time.monotonic()-started,
    }
    summary["compute"]["repair_runtime_seconds"] = summary["selection_repair"]["repair_runtime_seconds"]
    summary["selection_rule"] = "select minimum-objective stationary start in each branch; selected row predictions and metrics independently recomputed from the saved start parameters"
    summary["input_hashes"][str((HERE / "run.py").relative_to(ROOT))] = sha256(HERE / "run.py")
    summary["input_hashes"][str((HERE / "repair_stationarity.py").relative_to(ROOT))] = sha256(HERE / "repair_stationarity.py")
    summary["input_hashes"][str((OUT / "source_snapshot/run_executed.py").relative_to(ROOT))] = sha256(OUT / "source_snapshot/run_executed.py")
    summary["outputs"] = ["arrays.npz", "parameters.csv", "fixation_summary.csv", "multistarts.csv",
        "summary.json", "summary_first_pass.json", "baseline_g1_fit_checkpoint.json",
        "baseline_g1_fit_checkpoint_first_pass.json", "baseline_g1_selected_fit_checkpoint.npz",
        "p1m_fit_checkpoint.json", "p1m_fit_checkpoint_first_pass.json", "p1m_selected_fit_checkpoint.npz",
        "source_snapshot/run_executed.py"]
    write_json(summary_path, summary)


if __name__ == "__main__":
    main()
