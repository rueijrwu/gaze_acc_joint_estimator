"""Apply the frozen exp5 converged quadratic M2 model to every valid row in captures 5 and 6."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE
sys.path.insert(0, str(HERE))
from joint_m2 import (  # noqa: E402
    OBSERVABLE_SCHEMA, _invert_batch_m2_fast, measurements,
)

MODEL_PATH = HERE / "models/quadratic_model.json"
DEFAULT_OUT = HERE / "experiments/captures_5_6_direct"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_capture(capture: int, coef: np.ndarray, W: np.ndarray, model: dict,
                  anchor_support: dict[str, list[float]], out: Path,
                  expected_cache: bool):
    source = HERE / "data/detections" / f"capture_{capture}_detections.pkl"
    payload = source.read_bytes()
    data = pickle.loads(payload)
    arrays = data["arrays"]
    y, valid, components = measurements(arrays)
    frames = np.asarray(arrays["frame_index"], dtype=np.int64)
    timestamps = np.asarray(arrays["timestamp_ms"], dtype=float)
    if len(frames) != len(valid) or np.any(np.diff(frames) <= 0):
        raise ValueError(f"Invalid frame index sequence for capture {capture}")

    valid_rows = np.flatnonzero(valid)
    states = np.full((len(frames), 2), np.nan)
    diag_names = ("weighted_cost", "minimum_candidate_cost", "equivalent_minima_count",
                  "second_branch_distance", "second_branch_theta_delta_deg",
                  "second_branch_A_delta_D", "equivalent_theta_range_deg",
                  "equivalent_A_range_D", "segment_projected_stationarity", "candidate_cost_max")
    diagnostics = {name: np.full(len(frames), np.nan) for name in diag_names}
    csv_path = out / f"capture_{capture}_states.csv"
    old_cache_path = HERE / "experiments/captures_5_6_quadratic" / csv_path.name
    cache_used = False
    if expected_cache and old_cache_path.exists():
        names = ("gaze_theta_deg", "accommodation_diopters", *diag_names)
        cached = pd.read_csv(old_cache_path, usecols=names)
        if (len(cached) == len(frames) and
                np.isfinite(cached["gaze_theta_deg"].to_numpy()[valid]).all() and
                np.isfinite(cached["accommodation_diopters"].to_numpy()[valid]).all() and
                np.isnan(cached["gaze_theta_deg"].to_numpy()[~valid]).all() and
                np.isnan(cached["accommodation_diopters"].to_numpy()[~valid]).all()):
            states[:, 0] = cached["gaze_theta_deg"].to_numpy()
            states[:, 1] = cached["accommodation_diopters"].to_numpy()
            for name in diag_names:
                diagnostics[name][:] = cached[name].to_numpy()
            cache_used = True
    if not cache_used:
        for lo in range(0, len(valid_rows), 512):
            ix = valid_rows[lo:lo + 512]
            batch_state, batch_diag = _invert_batch_m2_fast(y[ix], coef, W, 100)
            states[ix] = batch_state
            for name in diag_names:
                diagnostics[name][ix] = batch_diag[name]

    status = np.full(len(frames), "invalid_observation", dtype=object)
    status[valid_rows] = "estimated"
    arrays_out = {
        "row_index": np.arange(len(frames)), "frame_index": frames,
        "timestamp_ms": timestamps, "valid_observation": valid,
        "p1_xy": [json.dumps(x.tolist()) for x in np.asarray(arrays["p1_xy"])],
        "p4_xy": [json.dumps(x.tolist()) for x in np.asarray(arrays["p4_xy"])],
        "p1_valid": [json.dumps(x.tolist()) for x in np.asarray(arrays["p1_valid"])],
        "p4_found": [json.dumps(x.tolist()) for x in np.asarray(arrays["p4_found"])],
        "d": y[:, 0], "rho": y[:, 1],
        "area_p1_px2": components["area_p1_px2"], "area_p4_px2": components["area_p4_px2"],
        "sqrt_area_p1_px": components["sqrt_area_p1_px"],
        "centroid_dx_px": components["centroid_dx_px"],
        "gaze_theta_deg": states[:, 0], "accommodation_diopters": states[:, 1],
        "outside_training_anchor_theta_range": valid & (
            (states[:, 0] < anchor_support["theta_deg"][0]) |
            (states[:, 0] > anchor_support["theta_deg"][1])),
        "outside_training_anchor_accommodation_range": valid & (
            (states[:, 1] < anchor_support["accommodation_diopters"][0]) |
            (states[:, 1] > anchor_support["accommodation_diopters"][1])),
        "status": status,
    }
    arrays_out.update(diagnostics)
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        keys = list(arrays_out)
        writer.writerow(keys)
        for row in zip(*(arrays_out[k] for k in keys)):
            writer.writerow([v.item() if isinstance(v, np.generic) else v for v in row])

    retained = valid
    theta = states[retained, 0]
    accommodation = states[retained, 1]
    equiv = diagnostics["equivalent_minima_count"][retained]
    stationarity = diagnostics["segment_projected_stationarity"][retained]
    lo_t, hi_t = model["theta_bounds"]
    lo_a, hi_a = model["A_bounds"]
    summary = {
        "capture": capture,
        "source_file": source.name,
        "source_sha256": hashlib.sha256(payload).hexdigest(),
        "row_count": int(len(frames)), "valid_detection_count": int(valid.sum()),
        "invalid_detection_count": int((~valid).sum()),
        "valid_rows_inverted": int(len(valid_rows)),
        "cached_inverse_reused": cache_used,
        "retained_estimate_count": int(retained.sum()),
        "ambiguous_count_equivalent_minima_gt_1": int(np.sum(equiv > 1)),
        "ambiguous_fraction": float(np.mean(equiv > 1)) if len(equiv) else None,
        "outside_training_anchor_support_count": {
            "theta": int(np.sum((theta < anchor_support["theta_deg"][0]) |
                                 (theta > anchor_support["theta_deg"][1]))),
            "accommodation": int(np.sum((accommodation < anchor_support["accommodation_diopters"][0]) |
                                         (accommodation > anchor_support["accommodation_diopters"][1]))),
        },
        "gaze_theta_deg": {
            "min": float(np.min(theta)), "q05": float(np.quantile(theta, .05)),
            "median": float(np.median(theta)), "mean": float(np.mean(theta)),
            "q95": float(np.quantile(theta, .95)), "max": float(np.max(theta)),
        },
        "accommodation_diopters": {
            "min": float(np.min(accommodation)), "q05": float(np.quantile(accommodation, .05)),
            "median": float(np.median(accommodation)), "mean": float(np.mean(accommodation)),
            "q95": float(np.quantile(accommodation, .95)), "max": float(np.max(accommodation)),
        },
        "bound_estimates_count": {
            "theta_lower": int(np.sum(np.isclose(theta, lo_t))),
            "theta_upper": int(np.sum(np.isclose(theta, hi_t))),
            "accommodation_lower": int(np.sum(np.isclose(accommodation, lo_a))),
            "accommodation_upper": int(np.sum(np.isclose(accommodation, hi_a))),
        },
        "stationarity_projected": {
            "median": float(np.nanmedian(stationarity)),
            "q95": float(np.nanquantile(stationarity, .95)),
            "max": float(np.nanmax(stationarity)),
            "count_above_1e-3": int(np.sum(stationarity > 1e-3)),
        },
        "input_observable_range": {
            "d_min": float(np.nanmin(y[valid, 0])), "d_max": float(np.nanmax(y[valid, 0])),
            "rho_min": float(np.nanmin(y[valid, 1])), "rho_max": float(np.nanmax(y[valid, 1])),
        },
        "fixation_label": "unknown; no reviewed target intervals for captures 5 and 6",
        "nominal_reference": "none available; plots show physical inverse bounds only",
        "training_anchor_support": anchor_support,
    }
    (out / f"capture_{capture}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    # Per-capture trace: every valid row is plotted; invalid rows remain absent.
    fig, axes = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True, constrained_layout=True)
    for ax, col, label, bounds in (
        (axes[0], 0, "Gaze θ (degrees)", (lo_t, hi_t)),
        (axes[1], 1, "Accommodation A (D)", (lo_a, hi_a)),
    ):
        ax.plot(frames[valid], states[valid, col], ".", ms=1.1, alpha=.65, rasterized=True,
                label="valid, estimated")
        ax.axhline(bounds[0], color="black", ls="--", lw=.8, label="inverse bound")
        ax.axhline(bounds[1], color="black", ls="--", lw=.8)
        ax.set_ylabel(label)
        ax.grid(alpha=.2)
        ax.legend(loc="best", markerscale=2)
    axes[1].set_xlabel("Frame index")
    fig.suptitle(f"exp5 capture {capture}: bounded quadratic M2 inverse (fixation unknown)")
    fig.savefig(out / f"capture_{capture}_trace.png", dpi=160)
    plt.close(fig)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    model_raw = MODEL_PATH.read_bytes()
    model = json.loads(model_raw)
    if model.get("observable_schema") != OBSERVABLE_SCHEMA:
        raise ValueError("Model observable schema does not match exp5 triangle measurements")
    if not model.get("diagnostics", {}).get("converged"):
        raise ValueError("Refusing to apply a nonconverged model")
    coef = np.asarray(model["coefficients"], dtype=float)
    W = np.asarray(model["precision"], dtype=float)
    intervals_path = HERE / "data/fixations/fixation_intervals.json"
    interval_raw = intervals_path.read_bytes()
    interval_report = json.loads(interval_raw)
    if hashlib.sha256(interval_raw).hexdigest() != model["diagnostics"]["provenance"]["selected_interval_sha256"]:
        raise ValueError("Training anchor support source hash differs from fitted model provenance")
    training_rows = interval_report["fixations"]
    anchor_support = {
        "theta_deg": [min(r["target_theta_deg"] for r in training_rows),
                      max(r["target_theta_deg"] for r in training_rows)],
        "accommodation_diopters": [min(r["demand_diopters_label"] for r in training_rows),
                                    max(r["demand_diopters_label"] for r in training_rows)],
        "source": "min/max of reviewed c1-c4 nominal fixation labels; coverage indicator, not c5/c6 reference labels",
    }
    cache_manifest_path = HERE / "experiments/captures_5_6_quadratic/manifest.json"
    cache_manifest = json.loads(cache_manifest_path.read_text()) if cache_manifest_path.exists() else {}
    expected_model_hash = hashlib.sha256(model_raw).hexdigest()
    summaries = []
    for i in (5, 6):
        source = HERE / "data/detections" / f"capture_{i}_detections.pkl"
        expected_cache = (
            cache_manifest.get("model_sha256") == expected_model_hash and
            cache_manifest.get("source_sha256", {}).get(source.name) == digest(source))
        summaries.append(write_capture(i, coef, W, model, anchor_support, out, expected_cache))
    (out / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    manifest = {
        "experiment": "exp5_quadratic_application_captures_5_6",
        "model_file": str(MODEL_PATH.relative_to(ROOT)),
        "model_sha256": hashlib.sha256(model_raw).hexdigest(),
        "model_schema": model.get("schema"), "observable_schema": model["observable_schema"],
        "coefficients": coef.tolist(), "precision": W.tolist(),
        "covariance": model["covariance"], "bounds": {
            "theta_deg": model["theta_bounds"], "accommodation_diopters": model["A_bounds"]},
        "inverse": {"method": "exp5 multistart bounded Gauss-Newton", "iterations": 100,
                    "batch_size": 512,
                    "branch_rule": "among equivalent minima choose smallest accommodation then gaze",
                    "temporal_inputs": False},
        "observable_description": model["observable_description"],
        "source_sha256": {f"capture_{i}_detections.pkl": s["source_sha256"]
                          for i, s in zip((5, 6), summaries)},
        "fixation_support": "none; all valid detections across each full capture trace",
        "nominal_target_or_demand_labels": "unavailable; all fixation labels unknown",
        "training_anchor_support": anchor_support,
        "training_anchor_source_sha256": hashlib.sha256(interval_raw).hexdigest(),
        "application": "bounded inverse applied independently to every valid detection; invalid rows retained and flagged",
        "cache_provenance": "reused prior inverse only when model hash, source hash, row count, and valid/invalid state completeness matched",
        "summaries": [f"capture_{i}_summary.json" for i in (5, 6)],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
