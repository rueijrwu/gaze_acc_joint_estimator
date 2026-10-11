#!/usr/bin/env python3
"""Matched full-roster P4 fits with unit and frozen P1-derived input scale."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
sys.path.insert(0, str(ROOT))

from distortion_model.p4_scale_comparison import ScaleComparisonProblem
from experiments.reverse_transform.joint_keystone_barrel_learned_reference.data import load_inputs
from experiments.reverse_transform.p4_gaze_accommodation_decoupling.stage_00_baseline_sensitivity.p1_joint_bounded_magnification.data import load_dataset


P1_ARRAYS = ROOT / "experiments/reverse_transform/p4_gaze_accommodation_decoupling/stage_00_baseline_sensitivity/p1_joint_bounded_magnification/results/arrays.npz"
P1_SUMMARY = P1_ARRAYS.with_name("summary.json")
P1_M_KEY = "frozen_learned_primary_m"
N_STARTS = 8
SEED = 86040


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def safe(value):
    if isinstance(value, dict):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def assemble_inputs():
    """Join P4 observations and frozen P1 m strictly by capture/source row."""
    p4 = load_inputs(ROOT)
    p1 = load_dataset(ROOT)
    with np.load(P1_ARRAYS, allow_pickle=False) as saved:
        p1_m = np.asarray(saved[P1_M_KEY], dtype=np.float64)
        p1_capture = np.asarray(saved["capture"], dtype=np.int16)
        p1_source = np.asarray(saved["source_row"], dtype=np.int64)
        p1_fixation = np.asarray(saved["fixation_id"], dtype=np.int8)
        p1_gaze = np.asarray(saved["actual_theta_x_deg"], dtype=np.float64)
    if len(p1_capture) != 88978 or len(p4["captures"]) != 4:
        raise ValueError("Unexpected cohort size or capture roster")
    if len(np.unique(np.stack((p1_capture, p1_source), axis=1), axis=0)) != len(p1_capture):
        raise ValueError("P1 source identifiers are not unique")

    p4_capture_parts, p4_source_parts, p4_fix_parts, p4_gaze_parts = [], [], [], []
    p4_x_parts, demand_parts, target_parts = [], [], []
    for cap in p4["captures"]:
        cid = int(cap["capture"])
        source = np.asarray(cap["source_rows"], dtype=np.int64)
        fixation = np.asarray(cap["fixation"], dtype=np.int64)
        gaze = np.asarray(cap["theta"], dtype=np.float64)
        x = np.asarray(cap["X"], dtype=np.float64)
        if x.shape != (len(source), 3, 2) or len(np.unique(source)) != len(source):
            raise ValueError(f"Capture {cid}: invalid P4 rows or source identifiers")
        p1_index = np.flatnonzero(p1_capture == cid)
        mapping = {int(row): int(ix) for ix, row in zip(p1_index, p1_source[p1_index])}
        if len(mapping) != len(source) or set(mapping) != set(map(int, source)):
            raise ValueError(f"Capture {cid}: P1/P4 source row roster mismatch")
        order = np.asarray([mapping[int(row)] for row in source], dtype=np.int64)
        if not np.array_equal(p1_fixation[order], fixation):
            raise ValueError(f"Capture {cid}: fixation ID mismatch after source alignment")
        if not np.allclose(p1_gaze[order], gaze, rtol=0., atol=1e-12):
            raise ValueError(f"Capture {cid}: saved P1/P4 gaze differs after source alignment")
        if not np.allclose(p1["actual_theta_x_deg"][order], gaze, rtol=0., atol=1e-12):
            raise ValueError(f"Capture {cid}: loader gaze differs from P4 saved gaze")
        if not np.all(np.isfinite(p1_m[order])) or np.any(p1_m[order] <= 0):
            raise ValueError(f"Capture {cid}: frozen P1 scale is nonfinite or nonpositive")
        target = np.asarray([float(record["target_theta_deg"]) for record in cap["fixation_masks"]], dtype=np.float64)
        p4_capture_parts.append(np.full(len(source), cid, dtype=np.int16))
        p4_source_parts.append(source)
        p4_fix_parts.append(fixation.astype(np.int8))
        p4_gaze_parts.append(gaze)
        p4_x_parts.append(x)
        demand_parts.append(np.full(len(source), float(cap["demand"]), dtype=np.float64))
        target_parts.append(target[fixation])

    capture = np.concatenate(p4_capture_parts)
    source_row = np.concatenate(p4_source_parts)
    fixation = np.concatenate(p4_fix_parts)
    gaze = np.concatenate(p4_gaze_parts)
    X = np.concatenate(p4_x_parts)
    demand = np.concatenate(demand_parts)
    target = np.concatenate(target_parts)
    # Re-align m in P4 concatenation order, rather than relying on either
    # loader's concatenation order.
    m_lookup = {(int(c), int(s)): float(m) for c, s, m in zip(p1_capture, p1_source, p1_m)}
    m = np.asarray([m_lookup[(int(c), int(s))] for c, s in zip(capture, source_row)], dtype=np.float64)
    if len(capture) != 88978 or len(np.unique(np.stack((capture, source_row), axis=1), axis=0)) != 88978:
        raise ValueError("P4 cohort does not contain the exact 88,978 unique source rows")
    if set(zip(map(int, capture), map(int, source_row))) != set(zip(map(int, p1_capture), map(int, p1_source))):
        raise ValueError("P4/P1 keyed identity audit failed")

    # Empirical common initial template: mean centered observed P4 at C1's
    # nominal-zero target. This and its radius are frozen for both branches.
    zero = (capture == 1) & (fixation == 2)
    if int(zero.sum()) == 0 or not np.array_equal(target[np.flatnonzero(zero)], np.zeros(int(zero.sum()))):
        raise ValueError("C1 nominal-zero P4 reference cohort is empty or mislabeled")
    reference = X[zero].mean(axis=0)
    R0 = float(np.sqrt(np.mean(np.sum(reference * reference, axis=1))))
    if not np.isfinite(R0) or R0 <= 0:
        raise ValueError("Invalid frozen P4 reference radius")
    return dict(capture=capture, source_row=source_row, fixation=fixation, gaze=gaze,
                X=X, demand=demand, target=target, m=m, reference=reference,
                R0=R0, p4=p4, p1=p1)


def frame_metrics(residual: np.ndarray, valid: np.ndarray):
    residual = np.asarray(residual, dtype=np.float64)
    squared = np.mean(np.sum(residual * residual, axis=-1), axis=-1)
    ok = np.asarray(valid, dtype=bool) & np.isfinite(squared)
    return np.sqrt(squared), ok


def metrics_by_group(residual, valid, capture, fixation):
    per_row_rms, valid = frame_metrics(residual, valid)
    grouped = {}
    for cid in range(1, 5):
        for fid in range(5):
            ix = (capture == cid) & (fixation == fid)
            used = ix & valid
            vals = per_row_rms[used]
            grouped[f"{cid}:{fid}"] = dict(n=int(ix.sum()), valid_n=int(used.sum()),
                invalid_n=int((ix & ~valid).sum()),
                vertex_rms_px=float(np.sqrt(np.mean(vals * vals))) if len(vals) else None,
                row_rms_mean_px=float(np.mean(vals)) if len(vals) else None)
    return per_row_rms, valid, grouped


def parameter_rows(branch, fit, diagnostics, nominal_demand):
    params = np.asarray(fit["params"], dtype=np.float64)
    names = list(fit["param_names"])
    B = np.asarray(diagnostics["B_px"], dtype=np.float64)
    alpha = np.asarray(diagnostics["alpha"], dtype=np.float64)
    kappas = np.asarray(diagnostics["kappa_px_inv2"], dtype=np.float64)
    beta = np.asarray(diagnostics["beta"], dtype=np.float64)
    rows = []

    def add(capture, demand, name, value, unit):
        rows.append(dict(branch=branch, capture=capture, nominal_demand_diopters=demand,
                         parameter=name, value=float(value), unit=unit))

    for name, value, unit in zip(("b", "cx", "cy"), alpha, ("dimensionless", "dimensionless", "dimensionless")):
        add("all", "", name, value, unit)
    add("all", "", "theta04_deg", diagnostics["theta04_deg"], "degrees")
    for vertex in range(3):
        for axis, axis_name in enumerate(("x", "y")):
            add("all", "", f"B_v{vertex+1}_{axis_name}_px", B[vertex, axis], "pixels")
    for cid in range(1, 5):
        add(cid, nominal_demand[cid], f"kappa_capture{cid}_px_inv2", kappas[cid-1], "px^-2")
        add(cid, nominal_demand[cid], f"beta_capture{cid}", beta[cid-1], "dimensionless")
    for name, value in zip(names, params):
        if name.startswith("beta_capture"):
            cid = int(name.removeprefix("beta_capture"))
            add(cid, nominal_demand[cid], name + "_fit_parameter", value, "dimensionless")
    return rows


def branch_checkpoint(branch, event):
    result = event["result"]
    best = event["best"]
    write_json(OUT / f"{branch}_fit_checkpoint.json", {
        "branch": branch, "completed_start": event["completed_start"],
        "current_start": {k: result[k] for k in
            ("start", "params", "objective", "success", "reason", "stationary",
             "projected_gradient_norm", "gradient_norm", "scaled_gradient", "iterations", "seconds", "history")},
        "best_so_far": {k: best[k] for k in
            ("start", "params", "objective", "success", "reason", "stationary",
             "projected_gradient_norm", "gradient_norm", "scaled_gradient", "iterations", "seconds", "history")},
    })


def capture_and_fixation_metrics(branch, residual, valid, capture, fixation, theta, targets):
    row_rms, valid, _ = metrics_by_group(residual, valid, capture, fixation)
    fix_rows, capture_summary = [], {}
    for cid in range(1, 5):
        capture_mask = capture == cid
        fix_mse = []
        pooled = row_rms[capture_mask & valid]
        for fid in range(5):
            ix = capture_mask & (fixation == fid)
            use = ix & valid
            mse = float(np.mean(row_rms[use] ** 2)) if np.any(use) else None
            rms = float(np.sqrt(mse)) if mse is not None else None
            if mse is not None:
                fix_mse.append(mse)
            fix_rows.append(dict(branch=branch, capture=cid,
                nominal_theta_deg=float(targets[cid][fid]),
                actual_gaze_theta_deg=float(np.mean(theta[ix])),
                actual_gaze_min_deg=float(np.min(theta[ix]),),
                actual_gaze_max_deg=float(np.max(theta[ix]),),
                n=int(ix.sum()), valid_n=int(use.sum()), invalid_n=int((ix & ~valid).sum()),
                vertex_rms_px=rms))
        capture_summary[str(cid)] = dict(
            n=int(capture_mask.sum()), valid_n=int((capture_mask & valid).sum()),
            invalid_n=int((capture_mask & ~valid).sum()),
            vertex_rms_px=float(np.sqrt(np.mean(fix_mse))) if fix_mse else None,
            pooled_vertex_rms_px=float(np.sqrt(np.mean(pooled**2))) if len(pooled) else None,
            equal_fixation_mse_px2=float(np.mean(fix_mse)) if fix_mse else None)
    return row_rms, valid, fix_rows, capture_summary


def input_hashes(data):
    p4_prov = data["p4"]["provenance"]
    p1_prov = data["p1"]["provenance"]
    files = {
        str(P1_ARRAYS.relative_to(ROOT)): sha256(P1_ARRAYS),
        str(P1_SUMMARY.relative_to(ROOT)): sha256(P1_SUMMARY),
        "experiments/reverse_transform/fixed_keystone_barrel_zero_gaze/results/summary.json": p4_prov["barrel_summary_sha256"],
        "experiments/reverse_transform/fixed_keystone_barrel_zero_gaze/results/parameters.csv": p4_prov["barrel_parameters_sha256"],
    }
    for item in p4_prov["per_capture_sources"]:
        files[item["summary_path"]] = item["summary_sha256"]
        files[item["arrays_path"]] = item["arrays_sha256"]
    for item in p1_prov["capture_sources"]:
        files[item["raw_detection_path"]] = item["raw_detection_sha256"]
    return files


def main():
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    data = assemble_inputs()
    capture, source, fix, theta = data["capture"], data["source_row"], data["fixation"], data["gaze"]
    X, m = data["X"], data["m"]
    # Both branches use precisely the same full roster, objective, initializer,
    # and deterministic multistart seeds. The nominal demand labels are retained
    # as metadata and are not fitted as a framewise accommodation variable.
    g1 = np.ones(len(m), dtype=np.float64)
    problem_g1 = ScaleComparisonProblem(X, theta, capture, fix, data["reference"], g=g1, R0=data["R0"])
    problem_m = ScaleComparisonProblem(X, theta, capture, fix, data["reference"], g=m, R0=data["R0"])
    # Same zero start, deterministic start-generation seed, count, and fit
    # objective in both branches. Checkpoints are compact and updated after
    # every start so an interrupted multi-hour run remains auditable.
    fit_g1 = problem_g1.fit(max_iter=250, n_starts=N_STARTS, seed=SEED,
                            checkpoint=lambda event: branch_checkpoint("baseline_g1", event))
    fit_m = problem_m.fit(max_iter=250, n_starts=N_STARTS, seed=SEED,
                          checkpoint=lambda event: branch_checkpoint("p1m", event))
    fits = {"baseline_g1": (problem_g1, fit_g1), "p1m": (problem_m, fit_m)}

    arrays = {
        "capture": capture, "source_row": source, "fixation_id": fix,
        "nominal_theta_deg": data["target"], "saved_gaze_theta_deg": theta,
        "nominal_demand_diopters": data["demand"], "P4_observed": X,
        "p1_m": m, "p1_m_source_row": source.copy(),
        "reference_B4_initial_px": data["reference"], "frozen_R04_px": np.asarray(data["R0"]),
        "evaluated": np.ones(len(capture), dtype=np.uint8),
    }
    summary = {
        "status": "completed", "experiment": "matched P4 shared-geometry refits with g=1 and frozen bounded P1 m",
        "analysis_scope": "conditional P4 model comparison only; no accommodation state or framewise A is fitted",
        "cohort": {"rows": int(len(capture)), "captures": [1, 2, 3, 4], "fixations": [0, 1, 2, 3, 4],
                   "inclusion_rule": data["p4"]["provenance"]["data_rule"],
                   "row_identity": "P4 observations and P1 m joined by exact (capture, source_row); fixation and saved gaze audited"},
        "reference": {"rule": "mean centered observed P4 rows in C1 nominal-zero fixation, using the exact frozen cohort",
                      "rows_n": int(np.sum((capture == 1) & (fix == 2))),
                      "B4_initial_px": data["reference"], "R04_frozen_px": data["R0"]},
        "nominal_A_labels_only": {str(cid): {"nominal_demand_diopters": float(data["demand"][capture == cid][0]),
                                               "target_theta_deg_by_fixation": [float(x) for x in data["p4"]["nominal_targets"]]}
                                  for cid in range(1, 5)},
        "objective": {"description": "equal capture, equal fixation within capture, equal row within fixation, native 3-vertex squared Euclidean error",
                      "branch_comparison": "identical observations, identities, reference, R0, weights, bounds, initialization, and matched multistart seed; only fixed g differs",
                      "fit_scope": "all 88,978 eligible rows in each branch", "g1": "g_i=1 for every row", "p1m": "g_i=frozen P1 learned m in [0.98,1.02] for the exact keyed source row"},
        "optimizer_protocol": {"starts_n": N_STARTS, "seed": SEED, "max_iterations_per_start": 250,
                               "same_start_generation": True, "all_rows_final_fit": True},
        "fit": {}, "domain": {}, "stationarity": {}, "identifiability": {},
        "capture_metrics": {}, "p1_scale_source": {}, "input_hashes": {},
    }
    fixation_rows, parameter_rows_all, multistart_rows = [], [], []
    full_targets = {cid: np.asarray(data["p4"]["nominal_targets"], dtype=np.float64) for cid in range(1, 5)}
    all_diagnostics = {}
    for branch, (problem, fit_result) in fits.items():
        evaluation = problem.evaluate(fit_result["params"])
        diagnostics = problem.diagnostics(fit_result["params"])
        all_diagnostics[branch] = diagnostics
        if not evaluation["domain_valid"] or not np.all(evaluation["valid"]):
            raise ValueError(f"{branch}: final selected fit is not valid for every eligible row")
        row_rms, row_valid, fix_rows, cap_summary = capture_and_fixation_metrics(
            branch, evaluation["residual"], evaluation["valid"], capture, fix, theta, full_targets)
        fixation_rows.extend(fix_rows)
        for cid in range(1, 5):
            entry = summary["capture_metrics"].setdefault(str(cid), {
                "nominal_demand_diopters": float(data["demand"][capture == cid][0]),
                "saved_gaze_min_deg": float(np.min(theta[capture == cid])),
                "saved_gaze_max_deg": float(np.max(theta[capture == cid])),
                "rows_n": int(np.sum(capture == cid)),
            })
            entry[branch] = cap_summary[str(cid)]
        arrays[f"{branch}_prediction"] = evaluation["prediction"]
        arrays[f"{branch}_residual"] = evaluation["residual"]
        arrays[f"{branch}_valid"] = row_valid.astype(np.uint8)
        arrays[f"{branch}_evaluated"] = np.ones(len(capture), dtype=np.uint8)
        arrays[f"{branch}_row_vertex_rms_px"] = row_rms
        arrays[f"{branch}_denominator"] = evaluation["denominator"]
        arrays[f"{branch}_radial_factor"] = evaluation["radial_factor"]
        arrays[f"{branch}_radial_derivative"] = evaluation["radial_derivative"]
        arrays[f"{branch}_stretch_exponent"] = evaluation["exponent"]
        domain_margin = np.minimum.reduce((
            np.min(evaluation["denominator"], axis=1)-1e-8,
            np.min(evaluation["radial_factor"], axis=1)-1e-8,
            np.min(evaluation["radial_derivative"], axis=1)-1e-8,
            20.-np.abs(evaluation["exponent"]),
        ))
        arrays[f"{branch}_domain_margin"] = domain_margin
        arrays[f"{branch}_params"] = fit_result["params"]
        arrays[f"{branch}_B_fit_px"] = diagnostics["B_px"]
        arrays[f"{branch}_diagnostic_singular_values"] = diagnostics["singular_values"]
        arrays[f"{branch}_diagnostic_right_vectors_scaled"] = diagnostics["right_singular_vectors_scaled"]
        fit_summary = {key: fit_result[key] for key in (
            "objective", "loss", "success", "reason", "stationary", "projected_gradient_norm",
            "gradient_norm", "scaled_gradient", "sensitivity_scaled_gradient", "iterations", "seconds",
            "best_start", "start_stability", "param_names", "bounds", "param_scales", "gauge",
            "weights_definition", "rejected_domain_evaluations") if key in fit_result}
        fit_summary["starts"] = [{key: start[key] for key in (
            "start", "initial_params", "params", "objective", "success", "reason", "stationary",
            "projected_gradient_norm", "gradient_norm", "scaled_gradient", "iterations", "seconds") if key in start}
            for start in fit_result["multistart"]]
        summary["fit"][branch] = fit_summary
        summary["domain"][branch] = {**evaluation["domain"], "domain_valid": evaluation["domain_valid"],
            "box_valid": evaluation["box_valid"], "invalid_rows": int(np.sum(~evaluation["valid"])),
            "minimum_row_domain_margin": float(np.min(domain_margin))}
        summary["stationarity"][branch] = {key: diagnostics[key] for key in (
            "stationary", "gradient_norm", "scaled_gradient", "projected_gradient_norm",
            "sensitivity_scaled_gradient", "active_bounds", "bound_contacts")}
        summary["identifiability"][branch] = {key: diagnostics[key] for key in (
            "rank", "parameters", "singular_values", "right_singular_vectors_scaled", "rank_rtol",
            "rank_tolerance", "condition_number", "spectrum_definition", "gauge", "scale_identifiability",
            "parameter_names", "parameter_scales", "weight_groups")}
        # Preserve full per-capture domain and weighted fit evidence from the
        # core alongside the independently grouped unweighted RMS metrics.
        summary["domain"][branch]["per_capture_core"] = diagnostics["per_capture"]
        nominal_demand = {cid: float(data["demand"][capture == cid][0]) for cid in range(1, 5)}
        parameter_rows_all.extend(parameter_rows(branch, fit_result, diagnostics, nominal_demand))
        for start in fit_result["multistart"]:
            multistart_rows.append(dict(branch=branch, start=int(start["start"]),
                initial_params=json.dumps(safe(start["initial_params"]), separators=(",", ":")),
                params=json.dumps(safe(start["params"]), separators=(",", ":")),
                objective=float(start["objective"]), selected=bool(start["start"] == fit_result["best_start"]),
                success=bool(start["success"]), stationary=bool(start["stationary"]),
                reason=start["reason"], projected_gradient_norm=float(start["projected_gradient_norm"]),
                iterations=int(start["iterations"]), runtime_seconds=float(start["seconds"])))
        # Selected-fit NPZ checkpoint includes the complete row result for this
        # branch, allowing export/report regeneration without refitting.
        np.savez_compressed(OUT / f"{branch}_selected_fit_checkpoint.npz",
            params=fit_result["params"], prediction=evaluation["prediction"], residual=evaluation["residual"],
            valid=evaluation["valid"].astype(np.uint8), row_vertex_rms_px=row_rms,
            B_fit_px=diagnostics["B_px"], singular_values=diagnostics["singular_values"],
            source_capture=capture, source_row=source, fixation_id=fix)
        print(json.dumps({"branch": branch, "objective": fit_result["objective"],
            "stationary": fit_result["stationary"], "rank": diagnostics["rank"],
            "condition_number": diagnostics["condition_number"], "min_denominator": evaluation["domain"]["minimum_denominator"],
            "runtime_seconds": fit_result["seconds"]}, separators=(",", ":")), flush=True)

    # Add matched parameter deltas, with the unscaled branch as the reference.
    baseline_values = {(r["capture"], r["parameter"]): r["value"] for r in parameter_rows_all if r["branch"] == "baseline_g1"}
    for row in parameter_rows_all:
        reference = baseline_values.get((row["capture"], row["parameter"]))
        row["delta_vs_g1"] = None if reference is None else float(row["value"]-reference)
    write_csv(OUT / "parameters.csv", parameter_rows_all)
    write_csv(OUT / "multistarts.csv", multistart_rows)
    write_csv(OUT / "fixation_summary.csv", fixation_rows)
    arrays["P1_m_lower_bound"] = np.asarray(.98)
    arrays["P1_m_upper_bound"] = np.asarray(1.02)
    np.savez_compressed(OUT / "arrays.npz", **arrays)

    p1_summary = json.loads(P1_SUMMARY.read_text())
    p1_model = p1_summary["models"]["m_pm_2pct_learned_all"]
    summary["p1_scale_source"] = {
        "artifact": str(P1_ARRAYS.relative_to(ROOT)), "artifact_sha256": sha256(P1_ARRAYS),
        "array_key": P1_M_KEY, "array_sha256": array_sha256(m),
        "fit_summary_artifact": str(P1_SUMMARY.relative_to(ROOT)), "fit_summary_sha256": sha256(P1_SUMMARY),
        "model": "m_pm_2pct_learned_all", "m_bounds": [0.98, 1.02],
        "m_min": float(np.min(m)), "m_max": float(np.max(m)), "m_mean": float(np.mean(m)),
        "source_row_alignment": "exact (capture, source_row); fixation exact; saved P1 gaze equals P4 saved gaze within 1e-12 deg",
        "P1_fit_params": p1_model["fit"]["params"], "P1_fit_param_names": p1_model["fit"]["param_names"],
        "P1_fit_scope": p1_model["fit_scope"], "P1_fit_gauge": p1_model["fit"]["gauge"],
        "P1_fit_stationary": p1_model["fit"]["stationary"],
    }
    source_files = input_hashes(data)
    for path in (HERE / "run.py", ROOT / "distortion_model/p4_scale_comparison.py",
                 ROOT / "distortion_model/shared_keystone.py",
                 ROOT / "experiments/reverse_transform/joint_keystone_barrel_learned_reference/data.py",
                 ROOT / "experiments/reverse_transform/p4_gaze_accommodation_decoupling/stage_00_baseline_sensitivity/p1_scale_analysis/data.py",
                 ROOT / "experiments/reverse_transform/p4_gaze_accommodation_decoupling/stage_00_baseline_sensitivity/p1_joint_bounded_magnification/run.py"):
        source_files[str(path.relative_to(ROOT))] = sha256(path)
    summary["input_hashes"] = source_files
    summary["array_hashes"] = {key: array_sha256(np.asarray(value)) for key, value in arrays.items()
                                if key in ("capture", "source_row", "fixation_id", "saved_gaze_theta_deg",
                                           "P4_observed", "p1_m", "reference_B4_initial_px")}
    summary["compute"] = {"backend": "CPU NumPy float64", "python": platform.python_version(),
        "numpy": np.__version__, "platform": platform.platform(),
        "thread_environment": {key: os.environ.get(key) for key in
            ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
        "runtime_seconds": time.monotonic()-started}
    summary["outputs"] = ["arrays.npz", "parameters.csv", "fixation_summary.csv", "multistarts.csv",
                          "baseline_g1_selected_fit_checkpoint.npz", "p1m_selected_fit_checkpoint.npz"]
    write_json(OUT / "summary.json", summary)
    print(json.dumps({"status": "completed", "rows": int(len(capture)),
                      "runtime_seconds": summary["compute"]["runtime_seconds"],
                      "output": str(OUT.relative_to(ROOT))}, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
