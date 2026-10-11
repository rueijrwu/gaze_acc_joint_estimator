#!/usr/bin/env python3
"""Matched full-roster P4 fits with unit and frozen P1-derived input scale."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
sys.path.insert(0, str(ROOT))

from distortion_model.p4_scale_comparison import ScaleComparisonProblem

INPUTS_NPZ = HERE / "baseline_inputs.npz"
INPUTS_JSON = HERE / "baseline_inputs.json"
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
    """Load the frozen cohort bundle without importing historical experiments."""
    if not INPUTS_NPZ.exists() or not INPUTS_JSON.exists():
        raise FileNotFoundError("Run prepare_inputs.py to create the frozen Stage 00 input bundle")
    manifest = json.loads(INPUTS_JSON.read_text())
    if manifest.get("schema") != "p4-scale-effect-inputs-v1":
        raise ValueError("Unrecognized Stage 00 input bundle schema")
    if sha256(INPUTS_NPZ) != manifest.get("bundle_sha256"):
        raise ValueError("Frozen Stage 00 input bundle checksum does not match its manifest")
    with np.load(INPUTS_NPZ, allow_pickle=False) as saved:
        data = {key: saved[key].copy() for key in saved.files}

    capture = np.asarray(data["capture"], dtype=np.int64)
    source = np.asarray(data["source_row"], dtype=np.int64)
    fixation = np.asarray(data["fixation_id"], dtype=np.int64)
    n = len(capture)
    if n != 88978 or len(np.unique(np.stack((capture, source), axis=1), axis=0)) != n:
        raise ValueError("Frozen P4 cohort must contain 88,978 unique (capture, source_row) identities")
    if set(np.unique(capture)) != {1, 2, 3, 4} or set(np.unique(fixation)) != {0, 1, 2, 3, 4}:
        raise ValueError("Frozen bundle has unexpected capture/fixation roster")

    X = np.asarray(data["P4_observed"], dtype=np.float64)
    gaze = np.asarray(data["saved_gaze_theta_deg"], dtype=np.float64)
    m = np.asarray(data["p1_m"], dtype=np.float64)
    target = np.asarray(data["nominal_theta_deg"], dtype=np.float64)
    demand = np.asarray(data["nominal_demand_diopters"], dtype=np.float64)
    reference = np.asarray(data["reference_B4_initial_px"], dtype=np.float64)
    R0 = float(data["frozen_R04_px"])
    if X.shape != (n, 3, 2) or not np.isfinite(X).all() or not np.isfinite(gaze).all():
        raise ValueError("Frozen P4 observations or saved gaze have invalid shape/values")
    if not np.isfinite(m).all() or np.any(m <= 0):
        raise ValueError("Frozen P1 scales must be positive and finite")

    targets = np.asarray(manifest["fixations"], dtype=np.float64)
    demand_by_capture = {int(k): float(v) for k, v in manifest["nominal_demand_diopters_by_capture"].items()}
    for cid in range(1, 5):
        if not np.all(demand[capture == cid] == demand_by_capture[cid]):
            raise ValueError(f"Capture {cid}: demand labels disagree with input manifest")
        for fid in range(5):
            rows = (capture == cid) & (fixation == fid)
            if not rows.any() or not np.all(target[rows] == targets[fid]):
                raise ValueError(f"Capture {cid}, fixation {fid}: target labels disagree with input manifest")
    if reference.shape != (3, 2) or not np.allclose(reference.mean(axis=0), 0., rtol=0., atol=1e-10):
        raise ValueError("Frozen P4 reference must be a centered (3,2) triangle")
    expected_R0 = float(np.sqrt(np.mean(np.sum(reference * reference, axis=1))))
    if not np.isclose(R0, expected_R0, rtol=0., atol=1e-12):
        raise ValueError("Frozen P4 reference radius does not match R0")

    return dict(capture=capture, source_row=source, fixation=fixation, gaze=gaze,
                X=X, demand=demand, target=target, m=m, reference=reference,
                R0=R0, manifest=manifest, nominal_targets=targets,
                demand_by_capture=demand_by_capture)

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


def select_stationary_minimum(fit):
    """Use the best stationary start when the lowest raw objective is not stationary."""
    starts = fit["multistart"]
    if not starts:
        raise ValueError("Core returned no multistart records")
    first_minimum = min(starts, key=lambda item: float(item["objective"]))
    stationary = [item for item in starts if bool(item.get("stationary", False))]
    if not stationary:
        raise ValueError("No stationary multistart solution; continue from the best checkpoint before exporting")
    selected = min(stationary, key=lambda item: float(item["objective"]))
    for key in ("params", "objective", "loss", "success", "reason", "stationary",
                "projected_gradient_norm", "gradient_norm", "scaled_gradient", "sensitivity_scaled_gradient",
                "iterations", "seconds", "history"):
        if key in selected:
            fit[key] = selected[key]
    fit["best_start"] = int(selected["start"])
    fit["first_pass_best_start"] = int(first_minimum["start"])
    fit["first_pass_best_objective"] = float(first_minimum["objective"])
    fit["first_pass_best_stationary"] = bool(first_minimum.get("stationary", False))
    fit["stationary_objective_excess_over_first_pass_minimum"] = float(selected["objective"]-first_minimum["objective"])
    fit["stationarity_selection_rule"] = "minimum-objective stationary multistart; export is blocked when no stationary solution exists"
    fit["selected_start_record"] = selected
    return fit


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
    """Hash currently required files; historical source hashes live in manifest."""
    return {
        str(INPUTS_NPZ.relative_to(ROOT)): sha256(INPUTS_NPZ),
        str(INPUTS_JSON.relative_to(ROOT)): sha256(INPUTS_JSON),
        str((HERE / "prepare_inputs.py").relative_to(ROOT)): sha256(HERE / "prepare_inputs.py"),
    }


def main():
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    snapshot_dir = OUT / "source_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(HERE / "run.py", snapshot_dir / "run_executed.py")
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
    fit_g1 = select_stationary_minimum(fit_g1)
    fit_m = select_stationary_minimum(fit_m)
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
                   "inclusion_rule": data["manifest"]["input_rule"],
                   "row_identity": "frozen P4 observations and P1 m exactly aligned by (capture, source_row); fixation and saved gaze checked at bundle creation"},
        "reference": {"rule": "mean centered observed P4 rows in C1 nominal-zero fixation, using the exact frozen cohort",
                      "rows_n": int(np.sum((capture == 1) & (fix == 2))),
                      "B4_initial_px": data["reference"], "R04_frozen_px": data["R0"]},
        "nominal_A_labels_only": {str(cid): {"nominal_demand_diopters": float(data["demand"][capture == cid][0]),
                                               "target_theta_deg_by_fixation": [float(x) for x in data["nominal_targets"]]}
                                  for cid in range(1, 5)},
        "objective": {"description": "equal capture, equal fixation within capture, equal row within fixation, native 3-vertex squared Euclidean error",
                      "branch_comparison": "identical observations, identities, reference, R0, weights, bounds, initialization, and matched multistart seed; only fixed g differs",
                      "fit_scope": "all 88,978 eligible rows in each branch", "g1": "g_i=1 for every row", "p1m": "g_i=frozen P1 learned m in [0.98,1.02] for the exact keyed source row"},
        "optimizer_protocol": {"starts_n": N_STARTS, "seed": SEED, "max_iterations_per_start": 250,
                               "same_start_generation": True, "all_rows_final_fit": True},
        "fit": {}, "domain": {}, "stationarity": {}, "identifiability": {},
        "selection_rule": "select minimum-objective stationary start in each branch; export is blocked when a branch has no stationary start",
        "capture_metrics": {}, "p1_scale_source": {}, "input_hashes": {},
        "frozen_source_provenance": data["manifest"]["source_input_hashes"],
    }
    fixation_rows, parameter_rows_all, multistart_rows = [], [], []
    full_targets = {cid: data["nominal_targets"] for cid in range(1, 5)}
    all_diagnostics = {}
    for branch, (problem, fit_result) in fits.items():
        evaluation = problem.evaluate(fit_result["params"])
        diagnostics = problem.diagnostics(fit_result["params"])
        fit_result["sensitivity_scaled_gradient"] = diagnostics["sensitivity_scaled_gradient"]
        fit_result["projected_gradient_norm"] = diagnostics["projected_gradient_norm"]
        fit_result["gradient_norm"] = diagnostics["gradient_norm"]
        fit_result["scaled_gradient"] = diagnostics["scaled_gradient"]
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
        scan_checkpoint = OUT / f"{branch}_fit_checkpoint.json"
        if scan_checkpoint.exists():
            shutil.copy2(scan_checkpoint, OUT / f"{branch}_fit_multistart_checkpoint.json")
        write_json(scan_checkpoint, {"branch": branch,
            "selection_rule": fit_result["stationarity_selection_rule"],
            "selected_start": fit_result["selected_start_record"],
            "starts": fit_summary["starts"], "diagnostics": diagnostics})
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

    p1_source = data["manifest"]["p1_scale_source"]
    summary["p1_scale_source"] = {
        **p1_source,
        "array_sha256": array_sha256(m),
        "m_bounds": [0.98, 1.02],
        "m_min": float(np.min(m)), "m_max": float(np.max(m)), "m_mean": float(np.mean(m)),
        "source_row_alignment": "exact (capture, source_row); fixation exact; saved P1 gaze equals P4 saved gaze within 1e-12 deg",
    }
    source_files = input_hashes(data)
    for path in (HERE / "run.py", ROOT / "distortion_model/p4_scale_comparison.py",
                 ROOT / "distortion_model/shared_keystone.py"):
        source_files[str(path.relative_to(ROOT))] = sha256(path)
    summary["input_hashes"] = source_files
    summary["input_hashes"][str((snapshot_dir / "run_executed.py").relative_to(ROOT))] = sha256(snapshot_dir / "run_executed.py")
    summary["array_hashes"] = {key: array_sha256(np.asarray(value)) for key, value in arrays.items()
                                if key in ("capture", "source_row", "fixation_id", "saved_gaze_theta_deg",
                                           "P4_observed", "p1_m", "reference_B4_initial_px")}
    summary["compute"] = {"backend": "CPU NumPy float64", "python": platform.python_version(),
        "numpy": np.__version__, "platform": platform.platform(),
        "thread_environment": {key: os.environ.get(key) for key in
            ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
        "runtime_seconds": time.monotonic()-started}
    summary["outputs"] = ["arrays.npz", "parameters.csv", "fixation_summary.csv", "multistarts.csv",
                          "baseline_g1_fit_checkpoint.json", "baseline_g1_fit_multistart_checkpoint.json",
                          "p1m_fit_checkpoint.json", "p1m_fit_multistart_checkpoint.json",
                          "baseline_g1_selected_fit_checkpoint.npz", "p1m_selected_fit_checkpoint.npz",
                          "source_snapshot/run_executed.py"]
    write_json(OUT / "summary.json", summary)
    print(json.dumps({"status": "completed", "rows": int(len(capture)),
                      "runtime_seconds": summary["compute"]["runtime_seconds"],
                      "output": str(OUT.relative_to(ROOT))}, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
