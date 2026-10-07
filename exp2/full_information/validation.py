"""Fail-closed evaluation against supplied independent withheld references."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


KEY_COLUMNS = ("capture", "fixation_index", "frame_index")
PREDICTION_COLUMNS = KEY_COLUMNS + (
    "prediction_success", "theta_deg", "A_D", "ambiguity_flag",
    "baseline_success", "baseline_theta_deg", "baseline_A_D", "baseline_ambiguity_flag")
INTERVAL_COLUMNS = ("theta_lower_deg", "theta_upper_deg", "A_lower_D", "A_upper_D")
REFERENCE_COLUMNS = KEY_COLUMNS + (
    "reference_theta_deg", "reference_A_D", "reference_theta_uncertainty_deg",
    "reference_A_uncertainty_D", "repeat_condition", "head_condition")
THRESHOLDS = (
    "theta_mae_max_deg", "theta_p95_abs_max_deg", "A_mae_max_D", "A_p95_abs_max_D",
    "success_coverage_min",
    "failure_rate_max", "reference_theta_uncertainty_max_deg",
    "reference_A_uncertainty_max_D", "min_recordings_per_condition",
    "theta_mae_improvement_min_deg", "A_mae_improvement_min_D",
    "reference_uncertainty_margin_multiplier", "recording_variability_margin_multiplier",
    "ambiguity_rate_max", "ambiguity_rate_increase_max")


def validation_contract():
    return {
        "schema_version": "independent_validation_v1",
        "status": "not_evaluable_without_independent_references",
        "accepted": False,
        "reason": "Current captures provide nominal targets and recording-confounded demands, not independent per-frame gaze/accommodation references.",
        "prediction_csv_columns": list(PREDICTION_COLUMNS),
        "optional_prediction_interval_columns": list(INTERVAL_COLUMNS),
        "reference_csv_columns": list(REFERENCE_COLUMNS),
        "prediction_success_encoding": "true/false or 1/0; include failed rows with prediction_success=false",
        "protocol_json": {
            "predeclared": "must be true; thresholds and withheld keys declared before evaluating predictions",
            "predeclaration_id": "nonempty auditable source identifier or hash",
            "reference_provenance": {
                "gaze": {"independent": "must be true", "source": "independent instrument/method and source identifier"},
                "accommodation": {"independent": "must be true", "source": "independent instrument/method and source identifier"},
                "uncertainty_method": "nonempty description of reference uncertainty and optional prediction interval construction",
                "head_condition_measurement": "nonempty instrument/method and source identifier for measured head conditions"},
            "withheld_rows": "list of {capture,fixation_index,frame_index}; both CSVs must match this exact set",
            "required_head_conditions": "at least two distinct measured labels, all represented with at least min_recordings_per_condition recordings",
            "no_H0_state_prior_attestation": "must be true, with nonempty no_H0_state_prior_source; treated as provider attestation unless independently audited",
            "prediction_intervals_required": "optional boolean; if true, supply interval columns, nominal coverage, and theta/A interval_coverage_min thresholds",
            "thresholds": {key: "predeclared nonnegative finite number" for key in THRESHOLDS}},
        "metric_policy": "Evaluate theta and A separately: MAE/RMSE/p95/bias, paired baseline MAE improvements, failure/success/ambiguity rates overall and per recording/head condition. Both states must improve by the predeclared practical margin, exceed reference-uncertainty and recording-variability margins, and show no recording-level deterioration. Require repeated recordings in every repeat condition.",
        "reference_policy": "Nominal gaze/demand labels, model-derived values and fits from this detector do not establish independent reference truth. Reference-provenance declarations remain auditable assertions from the data provider.",
        "uncertainty_policy": "Reference uncertainty is required for the improvement margin. Optional prediction interval coverage is a separate calibration diagnostic; local detector/noise conditioning alone is not a validated prediction interval.",
        "head_condition_policy": "A label alone does not establish measured head motion/distance; document its independent measurement in protocol provenance.",
        "current_evidence": "Capture1..4 have one demand per recording and five nominal fixations each; capture5 has no fixation/protocol annotation. Existing observations cannot establish independent recording repeats or physiological accuracy.",
    }


def _write(payload, output_dir, filename):
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    if path.exists():
        payload = dict(payload)
        payload["artifact_write_blocked_existing_path"] = str(path)
        return payload
    with path.open("x") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
    payload = dict(payload)
    payload["artifact_path"] = str(path)
    return payload


def write_validation_contract(output_dir):
    return _write(validation_contract(), output_dir, "validation_contract.json")


def _key(row):
    capture = str(row["capture"]).strip()
    if not capture:
        raise ValueError("capture is empty")
    # Canonicalize CSV and JSON integer keys without silently truncating them.
    numeric = [float(row[name]) for name in KEY_COLUMNS[1:]]
    if any(not np.isfinite(value) or value != int(value) for value in numeric):
        raise ValueError("fixation_index/frame_index must be finite integers")
    return (capture, *(str(int(value)) for value in numeric))


def _read_csv(path, columns):
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(columns) - set(reader.fieldnames or [])
        if missing:
            raise ValueError("Missing CSV columns: " + ", ".join(sorted(missing)))
        rows = list(reader)
    keyed = {}
    for row in rows:
        key = _key(row)
        if key in keyed:
            raise ValueError("Duplicate row key: " + repr(key))
        keyed[key] = row
    if not keyed:
        raise ValueError("CSV contains no rows")
    return keyed


def _finite(row, name):
    value = float(row[name])
    if not np.isfinite(value):
        raise ValueError("Nonfinite " + name)
    return value


def _boolean(row, name):
    value = str(row[name]).strip().lower()
    if value not in ("true", "false", "1", "0"):
        raise ValueError("Invalid boolean encoding: " + name)
    return value in ("true", "1")


def _source(value):
    return isinstance(value, str) and bool(value.strip())


def _metrics(rows):
    n = len(rows)
    success = [row for row in rows if row["success"]]
    result = {"n": n, "success_n": len(success), "success_coverage": len(success) / n,
              "failure_rate": 1 - len(success) / n,
              "ambiguity_rate": float(np.mean([row["ambiguous"] for row in rows])),
              "baseline_ambiguity_rate": float(np.mean([row["baseline_ambiguous"] for row in rows])),
              "paired_success_coverage": float(np.mean([row["success"] and row["baseline_success"] for row in rows]))}
    for name in ("theta", "A"):
        if not success:
            result[name] = {"mae": None, "rmse": None, "p95_abs": None, "bias": None,
                            "interval_coverage": None, "mean_interval_width": None}
            continue
        error = np.array([row[name + "_error"] for row in success])
        interval_rows = [row for row in success if row.get(name + "_interval_width") is not None]
        paired = [row for row in success if row["baseline_success"]]
        baseline = [row for row in rows if row["baseline_success"]]
        result[name] = {"mae": float(np.mean(np.abs(error))), "rmse": float(np.sqrt(np.mean(error**2))),
                        "p95_abs": float(np.percentile(np.abs(error), 95)), "bias": float(np.mean(error)),
                        "baseline_mae": float(np.mean([abs(row[name + "_baseline_error"]) for row in baseline])) if baseline else None,
                        "paired_mae_improvement": float(np.mean([abs(row[name + "_baseline_error"]) - abs(row[name + "_error"]) for row in paired])) if paired else None,
                        "interval_coverage": float(np.mean([row[name + "_covered"] for row in interval_rows])) if interval_rows else None,
                        "interval_n": len(interval_rows),
                        "mean_interval_width": float(np.mean([row[name + "_interval_width"] for row in interval_rows])) if interval_rows else None}
    return result


def evaluate_validation(prediction_csv, reference_csv, protocol_json, output_dir):
    """Evaluate supplied evidence; missing prerequisites produce explicit reasons.

    No threshold, reference, withheld support or interval is inferred from the
    current experiment. Artifact writes never replace previous files.
    """
    result = {"schema_version": "independent_validation_v1", "accepted": False,
              "status": "not_evaluable_without_independent_references", "reasons": []}
    paths = {"prediction_csv": prediction_csv, "reference_csv": reference_csv,
             "protocol_json": protocol_json}
    result["inputs"] = {name: str(path) if path is not None else None for name, path in paths.items()}
    for name, path in paths.items():
        if path is None or not Path(path).is_file():
            result["reasons"].append("Missing " + name)
    if result["reasons"]:
        return _write(result, output_dir, "validation_result.json")
    try:
        result["input_sha256"] = {name: hashlib.sha256(Path(path).read_bytes()).hexdigest() for name, path in paths.items()}
        with Path(protocol_json).open() as handle:
            protocol = json.load(handle)
        if protocol.get("predeclared") is not True or not _source(protocol.get("predeclaration_id")):
            result["reasons"].append("Thresholds/withheld rows lack auditable predeclaration")
        provenance = protocol.get("reference_provenance", {})
        for axis in ("gaze", "accommodation"):
            source = provenance.get(axis, {})
            if source.get("independent") is not True or not _source(source.get("source")):
                result["reasons"].append("Missing independently sourced " + axis + " reference provenance")
        if not _source(provenance.get("uncertainty_method")):
            result["reasons"].append("Missing uncertainty method provenance")
        if not _source(provenance.get("head_condition_measurement")):
            result["reasons"].append("Missing measured head-condition provenance")
        limits = protocol.get("thresholds", {})
        for name in THRESHOLDS:
            if name not in limits:
                result["reasons"].append("Missing predeclared threshold " + name)
                continue
            if isinstance(limits[name], bool):
                raise ValueError("Threshold must be numeric, not boolean: " + name)
            value = float(limits[name])
            if not np.isfinite(value) or value < 0:
                result["reasons"].append("Invalid threshold " + name)
            limits[name] = value
        for name in ("success_coverage_min", "failure_rate_max", "ambiguity_rate_max", "ambiguity_rate_increase_max"):
            if name in limits and limits[name] > 1:
                result["reasons"].append("Fraction threshold outside [0,1]: " + name)
        repeat_min = limits.get("min_recordings_per_condition", 0)
        if repeat_min < 2 or repeat_min != int(repeat_min):
            result["reasons"].append("At least two independent recordings per repeat condition must be predeclared")
        if limits.get("reference_uncertainty_margin_multiplier", 0) < 2 or limits.get("recording_variability_margin_multiplier", 0) < 1:
            result["reasons"].append("Improvement margins must cover at least twice reference uncertainty and one recording-level standard deviation")
        if protocol.get("no_H0_state_prior_attestation") is not True or not _source(protocol.get("no_H0_state_prior_source")):
            result["reasons"].append("Missing auditable no-H0-state-prior attestation")
        intervals_required = protocol.get("prediction_intervals_required", False)
        if not isinstance(intervals_required, bool):
            raise ValueError("prediction_intervals_required must be boolean")
        nominal = None
        if intervals_required:
            nominal = float(protocol.get("prediction_interval_nominal_coverage", np.nan))
            if not np.isfinite(nominal) or not 0 < nominal < 1:
                result["reasons"].append("Missing valid predeclared prediction interval nominal coverage")
            for axis in ("theta", "A"):
                name = axis + "_interval_coverage_min"
                value = float(limits.get(name, np.nan))
                if not np.isfinite(value) or not 0 <= value <= 1:
                    result["reasons"].append("Missing valid interval coverage threshold " + name)
                limits[name] = value
        head_conditions = protocol.get("required_head_conditions", [])
        if not isinstance(head_conditions, list) or not head_conditions or any(not str(value).strip() for value in head_conditions):
            result["reasons"].append("Missing predeclared measured head conditions")
        withheld = protocol.get("withheld_rows", [])
        if not isinstance(withheld, list) or not withheld:
            result["reasons"].append("Missing predeclared withheld row keys")
        if result["reasons"]:
            return _write(result, output_dir, "validation_result.json")
        expected_keys = {_key(row) for row in withheld}
        if len(expected_keys) != len(withheld):
            raise ValueError("Protocol contains duplicate withheld row keys")
        predictions = _read_csv(prediction_csv, PREDICTION_COLUMNS + (INTERVAL_COLUMNS if intervals_required else ()))
        references = _read_csv(reference_csv, REFERENCE_COLUMNS)
        if set(predictions) != expected_keys or set(references) != expected_keys:
            result["status"] = "invalid_validation_input"
            result["reasons"].append("Prediction/reference row sets must both exactly match the predeclared withheld row set")
            result["row_set_counts"] = {"withheld": len(expected_keys), "predictions": len(predictions), "references": len(references),
                                        "prediction_missing": len(expected_keys - set(predictions)),
                                        "reference_missing": len(expected_keys - set(references))}
            return _write(result, output_dir, "validation_result.json")
        evaluated, conditions, seen_heads, head_recordings = [], {}, set(), {}
        uncertainty_theta, uncertainty_A = [], []
        for key in sorted(expected_keys):
            pred, ref = predictions[key], references[key]
            reference_theta = _finite(ref, "reference_theta_deg")
            reference_A = _finite(ref, "reference_A_D")
            u_theta = _finite(ref, "reference_theta_uncertainty_deg")
            u_A = _finite(ref, "reference_A_uncertainty_D")
            if min(u_theta, u_A) < 0:
                raise ValueError("Reference uncertainties must be nonnegative")
            uncertainty_theta.append(u_theta)
            uncertainty_A.append(u_A)
            condition, head = ref["repeat_condition"].strip(), ref["head_condition"].strip()
            if not condition or not head:
                raise ValueError("Every reference row needs repeat/head condition labels")
            conditions.setdefault(condition, set()).add(key[0])
            seen_heads.add(head)
            head_recordings.setdefault(head, set()).add(key[0])
            row = {"capture": key[0], "repeat_condition": condition, "head_condition": head,
                   "success": _boolean(pred, "prediction_success"),
                   "baseline_success": _boolean(pred, "baseline_success"),
                   "ambiguous": _boolean(pred, "ambiguity_flag"),
                   "baseline_ambiguous": _boolean(pred, "baseline_ambiguity_flag")}
            if row["baseline_success"]:
                row["theta_baseline_error"] = _finite(pred, "baseline_theta_deg") - reference_theta
                row["A_baseline_error"] = _finite(pred, "baseline_A_D") - reference_A
            if row["success"]:
                for axis, prediction_name, reference, lower_name, upper_name in (
                    ("theta", "theta_deg", reference_theta, "theta_lower_deg", "theta_upper_deg"),
                    ("A", "A_D", reference_A, "A_lower_D", "A_upper_D")):
                    value = _finite(pred, prediction_name)
                    row[axis + "_error"] = value - reference
                    if all(str(pred.get(name, "")).strip() for name in (lower_name, upper_name)):
                        lower, upper = (_finite(pred, name) for name in (lower_name, upper_name))
                        if not lower <= value <= upper:
                            raise ValueError("Prediction interval must be ordered and contain its point estimate")
                        row[axis + "_covered"] = lower <= reference <= upper
                        row[axis + "_interval_width"] = upper - lower
                    elif intervals_required:
                        raise ValueError("Missing required prediction interval")
            evaluated.append(row)
        failures = []
        if max(uncertainty_theta) > limits["reference_theta_uncertainty_max_deg"]:
            failures.append("Gaze reference uncertainty exceeds predeclared maximum")
        if max(uncertainty_A) > limits["reference_A_uncertainty_max_D"]:
            failures.append("Accommodation reference uncertainty exceeds predeclared maximum")
        if any(len(recordings) < repeat_min for recordings in conditions.values()):
            failures.append("Insufficient independent recording repeats for at least one repeat condition")
        limited_head_coverage = len(set(map(str, head_conditions))) < 2 or len(seen_heads) < 2
        if limited_head_coverage:
            failures.append("At least two distinct measured head conditions are required for full acceptance")
        if not set(map(str, head_conditions)).issubset(seen_heads):
            limited_head_coverage = True
            failures.append("Missing required measured head conditions")
        if any(len(recordings) < repeat_min for recordings in head_recordings.values()):
            limited_head_coverage = True
            failures.append("Insufficient independent recording repeats within at least one head condition")
        summaries = {"overall": _metrics(evaluated)}
        for capture in sorted({row["capture"] for row in evaluated}):
            summaries["recording:" + capture] = _metrics([row for row in evaluated if row["capture"] == capture])
        for head in sorted(seen_heads):
            summaries["head_condition:" + head] = _metrics([row for row in evaluated if row["head_condition"] == head])
        for label, metrics in summaries.items():
            if metrics["success_coverage"] < limits["success_coverage_min"]:
                failures.append(label + ": success coverage below threshold")
            if metrics["failure_rate"] > limits["failure_rate_max"]:
                failures.append(label + ": failure rate above threshold")
            if metrics["paired_success_coverage"] < limits["success_coverage_min"]:
                failures.append(label + ": paired baseline success coverage below threshold")
            if metrics["ambiguity_rate"] > limits["ambiguity_rate_max"]:
                failures.append(label + ": ambiguity rate above threshold")
            if metrics["ambiguity_rate"] - metrics["baseline_ambiguity_rate"] > limits["ambiguity_rate_increase_max"]:
                failures.append(label + ": ambiguity increase exceeds acceptable tradeoff")
            for axis, unit in (("theta", "deg"), ("A", "D")):
                for metric, threshold in (("mae", axis + "_mae_max_" + unit),
                                          ("p95_abs", axis + "_p95_abs_max_" + unit)):
                    if metrics[axis][metric] is None or metrics[axis][metric] > limits[threshold]:
                        failures.append(label + ": " + axis + " " + metric + " fails threshold")
                if intervals_required:
                    coverage = metrics[axis]["interval_coverage"]
                    if coverage is None or coverage < limits[axis + "_interval_coverage_min"]:
                        failures.append(label + ": " + axis + " interval coverage fails threshold")
        improvement = {}
        for axis, unit, uncertainties in (("theta", "deg", uncertainty_theta), ("A", "D", uncertainty_A)):
            record_changes = [metrics[axis].get("paired_mae_improvement") for label, metrics in summaries.items() if label.startswith("recording:")]
            valid_changes = [value for value in record_changes if value is not None]
            variability = float(np.std(valid_changes, ddof=1)) if len(valid_changes) >= 2 else None
            practical = limits[axis + "_mae_improvement_min_" + unit]
            reference_margin = limits["reference_uncertainty_margin_multiplier"] * max(uncertainties)
            variability_margin = limits["recording_variability_margin_multiplier"] * variability if variability is not None else None
            required = max(practical, reference_margin, variability_margin) if variability_margin is not None else None
            observed = summaries["overall"][axis].get("paired_mae_improvement")
            improvement[axis] = {"paired_mae_improvement": observed, "recording_improvement_std": variability,
                                 "reference_uncertainty_margin": reference_margin,
                                 "recording_variability_margin": variability_margin, "required_improvement": required}
            if required is None or observed is None or observed <= required:
                failures.append(axis + ": paired improvement does not exceed practical/reference/recording margins")
            if len(valid_changes) != len(record_changes) or any(value < 0 for value in valid_changes):
                failures.append(axis + ": missing paired support or recording-level deterioration")
        result.update(status="passed_independent_validation" if not failures else
                      ("limited_head_coverage" if limited_head_coverage else "failed_independent_validation"),
                      accepted=not failures, reasons=failures, metrics=summaries, thresholds=limits,
                      predeclaration_id=protocol["predeclaration_id"], reference_provenance=provenance,
                      prediction_interval_nominal_coverage=nominal,
                      paired_improvement=improvement,
                      no_H0_state_prior_attestation={"declared": True, "source": protocol["no_H0_state_prior_source"], "independently_verified": False},
                      repeat_recording_counts={condition: len(recordings) for condition, recordings in conditions.items()},
                      head_recording_counts={condition: len(recordings) for condition, recordings in head_recordings.items()},
                      reference_uncertainty_max={"theta_deg": max(uncertainty_theta), "A_D": max(uncertainty_A)},
                      evaluation_scope="Supplied independent withheld evidence and declared head/repeat conditions only; no extrapolation to unrepresented conditions.")
    except (OSError, ValueError, KeyError, TypeError, AttributeError, OverflowError, csv.Error) as exc:
        result["status"] = "invalid_validation_input"
        result["reasons"].append(str(exc))
    return _write(result, output_dir, "validation_result.json")
