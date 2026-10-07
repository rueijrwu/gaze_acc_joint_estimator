#!/usr/bin/env python3
"""Compare a linear P4−P1 gaze calibration with corrected model estimates.

The linear calibration uses the five full capture-1 fixation intervals. For
each fixation, it averages valid per-frame mean(P4_x)−mean(P1_x) in pixels,
then fits theta = slope * displacement + intercept to the nominal target
angles with equal weight per fixation. No accommodation term is used.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
ROOT = HERE
FIXATIONS = HERE / "data/fixations/fixation_intervals.json"
CAPTURE_DIR = HERE / "experiments/captures_5_6_direct"
DEFAULT_OUT = HERE / "experiments/gaze_linear_vs_corrected"


def read_capture1_fixations() -> tuple[list[dict], list[dict]]:
    """Return per-fixation means and target labels using full reviewed windows."""
    report = json.loads(FIXATIONS.read_text())
    intervals = [r for r in report["fixations"]
                 if r["capture"] == "capture_1_detections.pkl"
                 and r["target_theta_deg"] in (-10, -5, 0, 5, 10)]
    intervals.sort(key=lambda r: r["target_theta_deg"])
    if len(intervals) != 5:
        raise ValueError(f"Expected 5 capture-1 target fixations, found {len(intervals)}")

    with (HERE / "data/detections/capture_1_detections.pkl").open("rb") as stream:
        arrays = pickle.load(stream)["arrays"]
    p1 = np.asarray(arrays["p1_xy"], dtype=float)
    p4 = np.asarray(arrays["p4_xy"], dtype=float)
    p1_valid = np.asarray(arrays["p1_valid"], dtype=bool).all(axis=1)
    p4_found = np.asarray(arrays["p4_found"], dtype=bool).all(axis=1)
    good = p1_valid & p4_found & np.isfinite(p1).all(axis=(1, 2)) & np.isfinite(p4).all(axis=(1, 2))
    # The direct pixel displacement explicitly requested by the user.
    raw = p4[:, :, 0].mean(axis=1) - p1[:, :, 0].mean(axis=1)
    # Retained as a diagnostic only; it is not used by the calibration.
    edge_a = p1[:, 1] - p1[:, 0]
    edge_b = p1[:, 2] - p1[:, 0]
    area = np.abs(edge_a[:, 0] * edge_b[:, 1] - edge_a[:, 1] * edge_b[:, 0]) / 2
    normalized = raw / np.sqrt(area)
    rows = []
    for i, interval in enumerate(intervals):
        start, end = int(interval["start_row"]), int(interval["end_row_exclusive"])
        valid = good[start:end] & np.isfinite(raw[start:end])
        n = int(valid.sum())
        if not n:
            raise ValueError(f"No valid observations in fixation {interval['target_theta_deg']}")
        rows.append({
            "fixation_index": i,
            "target_theta_deg": float(interval["target_theta_deg"]),
            "start_row": start,
            "end_row_exclusive": end,
            "valid_count": n,
            "raw_displacement_mean_px": float(raw[start:end][valid].mean()),
            "normalized_diagnostic_mean": float(normalized[start:end][valid].mean()),
        })
    return rows, intervals


def read_state_rows(capture: int) -> list[dict]:
    with (CAPTURE_DIR / f"capture_{capture}_states.csv").open(newline="") as stream:
        return list(csv.DictReader(stream))


def local_median(values: np.ndarray, window: int = 501) -> np.ndarray:
    """Median over nearby frames, interpolating missing values for detection only."""
    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError("No finite comparison observations")
    filled = np.interp(np.arange(len(values)), np.flatnonzero(finite), values[finite])
    padded = np.pad(filled, window // 2, mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(padded, window), axis=1)


def fitted_ylim(values: np.ndarray, step: float) -> tuple[float, float]:
    low, high = float(np.min(values)), float(np.max(values))
    padding = max((high - low) * .05, step)
    return (float(np.floor((low - padding) / step) * step),
            float(np.ceil((high + padding) / step) * step))


def plot_capture(capture: int, rows: list[dict], slope: float, intercept: float,
                 out: Path) -> dict:
    parsed = []
    for row in rows:
        valid = row["valid_observation"].lower() == "true"
        frame = int(row["frame_index"])
        corrected = float(row["gaze_theta_deg"])
        displacement = float(row["centroid_dx_px"])
        linear = slope * displacement + intercept
        parsed.append({
            "frame_index": frame,
            "timestamp_ms": float(row["timestamp_ms"]),
            "valid_observation": valid,
            "raw_displacement_px": displacement,
            "uncorrected_linear_theta_deg": linear if valid else np.nan,
            "corrected_theta_deg": corrected if valid else np.nan,
            "difference_uncorrected_minus_corrected_deg": linear - corrected if valid else np.nan,
        })

    frame = np.asarray([r["frame_index"] for r in parsed])
    valid = np.asarray([r["valid_observation"] for r in parsed])
    linear = np.asarray([r["uncorrected_linear_theta_deg"] for r in parsed])
    corrected = np.asarray([r["corrected_theta_deg"] for r in parsed])
    difference = np.asarray([r["difference_uncorrected_minus_corrected_deg"] for r in parsed])
    valid &= np.isfinite(linear) & np.isfinite(corrected)
    # Exclude transient spikes from both panels using one common mask.
    # The underlying estimates and capture-1 calibration remain unchanged.
    outlier = valid & ((np.abs(linear - local_median(linear)) > 1.0)
                       | (np.abs(corrected - local_median(corrected)) > 1.0)
                       | (np.abs(difference - local_median(difference)) > .5))
    retained = valid & ~outlier
    for i, row in enumerate(parsed):
        row["plot_outlier"] = bool(outlier[i])
        row["included_in_plot"] = bool(retained[i])
    csv_path = out / f"capture_{capture}_comparison.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(parsed[0]))
        writer.writeheader()
        writer.writerows(parsed)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True,
                             sharex=True)
    # Convert plotted angles to arcminutes (60 arcmin per degree).
    linear_arcmin = linear * 60
    corrected_arcmin = corrected * 60
    difference_arcmin = difference * 60
    axes[0].plot(frame[retained], linear_arcmin[retained], ".", ms=1.0, alpha=.65, rasterized=True,
                 label="Linear P4−P1 gaze (no accommodation correction)")
    axes[0].plot(frame[retained], corrected_arcmin[retained], ".", ms=1.0, alpha=.65, rasterized=True,
                 label="Joint-model gaze")
    axes[0].set_ylabel("Gaze θ (arcminutes)")
    axes[0].set_title("Uncorrected and corrected gaze")
    axes[0].legend(markerscale=3, loc="best")
    axes[1].plot(frame[retained], difference_arcmin[retained], ".", ms=1.0, alpha=.65, rasterized=True,
                 color="tab:purple")
    axes[1].axhline(0, color="black", lw=.8)
    axes[1].set_ylabel("Difference: linear − joint-model (arcminutes)")
    axes[1].set_title("Difference")
    gaze_ylim = fitted_ylim(np.concatenate((linear_arcmin[retained], corrected_arcmin[retained])), 30)
    difference_ylim = (-40.0, 40.0)
    axes[0].set_ylim(*gaze_ylim)
    axes[1].set_ylim(*difference_ylim)
    for ax in axes:
        ax.set_xlabel("Frame index")
        ax.grid(alpha=.2)
    fig.suptitle(f"exp5 capture {capture}: linear baseline vs joint model "
                 f"({int(outlier.sum())} outlier frames excluded)")
    fig.savefig(out / f"capture_{capture}_comparison.png", dpi=160)
    plt.close(fig)
    return {"capture": capture, "row_count": len(parsed), "valid_count": int(valid.sum()),
            "outlier_count": int(outlier.sum()), "plotted_count": int(retained.sum()),
            "outlier_rule": "501-frame local median; gaze deviation > 1 deg or difference deviation > 0.5 deg",
            "gaze_ylim_arcmin": gaze_ylim, "difference_ylim_arcmin": difference_ylim,
            "plot_units": "arcminutes (60 arcmin per degree)",
            "comparison_csv": csv_path.name,
            "plot": f"capture_{capture}_comparison.png",
            "difference_definition": "linear gaze minus joint-model gaze; not physiological error"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--captures", type=int, nargs="+", default=[5, 6])
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    calibration_rows, intervals = read_capture1_fixations()
    x = np.asarray([r["raw_displacement_mean_px"] for r in calibration_rows])
    y = np.asarray([r["target_theta_deg"] for r in calibration_rows])
    slope, intercept = np.polyfit(x, y, deg=1)
    with (out / "capture_1_linear_calibration.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(calibration_rows[0]))
        writer.writeheader()
        writer.writerows(calibration_rows)

    comparison_results = []
    for capture in args.captures:
        if capture not in (5, 6):
            raise ValueError("Cached corrected state CSVs are available for captures 5 and 6")
        comparison_results.append(plot_capture(capture, read_state_rows(capture),
                                               float(slope), float(intercept), out))
    report = {
        "experiment": "exp5_capture1_linear_p4_minus_p1_vs_corrected_gaze",
        "calibration": {
            "source": "capture 1 full stored target fixation intervals",
            "observable": "mean(P4_x)-mean(P1_x), raw pixels",
            "formula": "theta_deg = slope * displacement_px + intercept",
            "slope_deg_per_px": float(slope), "intercept_deg": float(intercept),
            "fixation_weighting": "one equal-weight mean point per fixation",
            "fit_r2": float(1 - np.sum((y - (slope * x + intercept)) ** 2) /
                             np.sum((y - y.mean()) ** 2)),
            "includes_accommodation": False,
            "normalization": "sqrt(area(P1)) normalized displacement recorded as diagnostic only",
        },
        "comparison": "cached direct quadratic-model joint-model gaze; common outlier mask for both panels",
        "captures": comparison_results,
        "fixation_intervals": [{"target_theta_deg": r["target_theta_deg"],
                                "start_row": r["start_row"],
                                "end_row_exclusive": r["end_row_exclusive"]} for r in intervals],
    }
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
