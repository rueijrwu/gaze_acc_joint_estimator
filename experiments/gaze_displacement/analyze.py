#!/usr/bin/env python3
"""Plot P1/P4 horizontal displacement versus gaze for reviewed captures 1--4.

This standalone analysis reads the repository's frozen detections and reviewed
fixation intervals. It does not import the estimator package or fit a model.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "gaze_displacement_mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import numpy as np

plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 17})

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
FIXATIONS = ROOT / "data/fixations/fixation_intervals.json"
HAMPEL_WINDOW_VALID_FRAMES = 101
HAMPEL_MIN_JUMP_PX = 2.0
HAMPEL_MAD_MULTIPLIER = 6.0


def despike_mask(values):
    """Reject isolated local jumps with a rolling Hampel rule."""
    values = np.asarray(values, dtype=float)
    keep = np.ones(len(values), dtype=bool)
    radius = HAMPEL_WINDOW_VALID_FRAMES // 2
    for index, value in enumerate(values):
        local = values[max(0, index-radius):min(len(values), index+radius+1)]
        median = float(np.median(local))
        mad = float(np.median(np.abs(local-median)))
        cutoff = max(HAMPEL_MIN_JUMP_PX, HAMPEL_MAD_MULTIPLIER * 1.4826 * mad)
        if abs(value-median) > cutoff:
            keep[index] = False
    return keep


def load_rows():
    report = json.loads(FIXATIONS.read_text(encoding="utf-8"))
    report_sources = {row["capture"]: row for row in report["sources"]}
    groups = [g for g in report["fixations"]
              if g["capture"] in {f"capture_{i}_detections.pkl" for i in range(1, 5)}]
    if len(groups) != 20:
        raise ValueError(f"Expected 20 reviewed fixation periods in captures 1--4; found {len(groups)}")

    summaries = []
    for capture_number in range(1, 5):
        name = f"capture_{capture_number}_detections.pkl"
        source = ROOT / "data/detections" / name
        expected_hash = report_sources[name]["sha256"]
        actual_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(f"Detection file hash does not match reviewed intervals: {source}")
        with source.open("rb") as stream:
            arrays = pickle.load(stream)["arrays"]
        p1 = np.asarray(arrays["p1_xy"], dtype=float)
        p4 = np.asarray(arrays["p4_xy"], dtype=float)
        p1_valid = np.asarray(arrays["p1_valid"], dtype=bool)
        p4_found = np.asarray(arrays["p4_found"], dtype=bool)
        if p1.ndim != 3 or p1.shape[1:] != (3, 2) or p4.shape != p1.shape:
            raise ValueError(f"Unexpected point-array shape for {name}: {p1.shape}, {p4.shape}")
        valid = (p1_valid.all(axis=1) & p4_found.all(axis=1)
                 & np.isfinite(p1).all(axis=(1, 2)) & np.isfinite(p4).all(axis=(1, 2)))
        # Horizontal span between the rightmost and leftmost point, frame by frame.
        p1_span = p1[:, :, 0].max(axis=1) - p1[:, :, 0].min(axis=1)
        p4_span = p4[:, :, 0].max(axis=1) - p4[:, :, 0].min(axis=1)
        capture_groups = sorted((g for g in groups if g["capture"] == name),
                               key=lambda g: float(g["target_theta_deg"]))
        if len(capture_groups) != 5:
            raise ValueError(f"Expected five gaze fixations for {name}")
        for g in capture_groups:
            start, end = int(g["start_row"]), int(g["end_row_exclusive"])
            mask = valid[start:end]
            n = int(mask.sum())
            if n < 2:
                raise ValueError(f"Need at least two valid rows in fixation {name} {g['target_theta_deg']} deg; got {n}")
            p1_raw, p4_raw = p1_span[start:end][mask], p4_span[start:end][mask]
            p1_keep, p4_keep = despike_mask(p1_raw), despike_mask(p4_raw)
            p1_vals, p4_vals = p1_raw[p1_keep], p4_raw[p4_keep]
            if len(p1_vals) < 2 or len(p4_vals) < 2:
                raise ValueError(f"Despiking left too few frames in {name} {g['target_theta_deg']} deg")
            summaries.append({
                "capture": capture_number,
                "target_gaze_deg": float(g["target_theta_deg"]),
                "accommodation_demand_D": float(g["demand_diopters_label"]),
                "start_row": start,
                "end_row_exclusive": end,
                "valid_frame_count": n,
                "p1_filtered_frame_count": len(p1_vals),
                "p1_removed_jump_count": int((~p1_keep).sum()),
                "p1_abs_right_left_x_mean_px": float(p1_vals.mean()),
                "p1_abs_right_left_x_std_px": float(p1_vals.std(ddof=1)),
                "p1_unfiltered_std_px": float(p1_raw.std(ddof=1)),
                "p4_filtered_frame_count": len(p4_vals),
                "p4_removed_jump_count": int((~p4_keep).sum()),
                "p4_abs_right_left_x_mean_px": float(p4_vals.mean()),
                "p4_abs_right_left_x_std_px": float(p4_vals.std(ddof=1)),
                "p4_unfiltered_std_px": float(p4_raw.std(ddof=1)),
            })
    return summaries, report


def plot_panel(ax, rows, mean_key, std_key, ylabel, measurement):
    demands = sorted({r["accommodation_demand_D"] for r in rows})
    cmap = plt.get_cmap("viridis", len(demands))
    fit_results = []
    for index, demand in enumerate(demands):
        series = sorted((r for r in rows if r["accommodation_demand_D"] == demand),
                        key=lambda r: r["target_gaze_deg"])
        x = np.asarray([r["target_gaze_deg"] for r in series])
        y = np.asarray([r[mean_key] for r in series])
        err = np.asarray([r[std_key] for r in series])
        label = f"{demand:.2f}".rstrip("0").rstrip(".") + "D"
        excluded = np.asarray([
            measurement == "P1" and r["capture"] == 2
            and np.isclose(r["target_gaze_deg"], 5.0) for r in series
        ], dtype=bool)
        ax.scatter(x[~excluded], y[~excluded], color=cmap(index), s=56, label=label, zorder=3)
        ax.errorbar(x, y, yerr=err, fmt="none", capsize=4, linewidth=1.5,
                    color=cmap(index), zorder=2)
        if excluded.any():
            ax.scatter(x[excluded], y[excluded], marker="x", s=150, linewidth=3,
                       color="red", zorder=5)
        fit_x, fit_y = x[~excluded], y[~excluded]
        coefficients = np.polyfit(fit_x, fit_y, deg=2)
        curve_x = np.linspace(float(x.min()), float(x.max()), 200)
        ax.plot(curve_x, np.polyval(coefficients, curve_x), linestyle="--",
                linewidth=2.2, color=cmap(index), zorder=1)
        predicted = np.polyval(coefficients, fit_x)
        ss_res = float(np.sum((fit_y-predicted)**2))
        ss_tot = float(np.sum((fit_y-fit_y.mean())**2))
        fit_results.append({
            "measurement": measurement,
            "accommodation_demand_D": float(demand),
            "degree": 2,
            "fit_point_count": int(len(fit_x)),
            "excluded_outlier_count": int(excluded.sum()),
            "excluded_target_gaze_deg": [float(v) for v in x[excluded]],
            "coefficient_deg2": float(coefficients[0]),
            "coefficient_deg1": float(coefficients[1]),
            "intercept_px": float(coefficients[2]),
            "r_squared": float(1-ss_res/ss_tot) if ss_tot else None,
        })
    ax.axvline(0, color="0.75", linewidth=0.8)
    ax.set_xlabel("Eye Rotation (deg)", fontsize=22)
    ax.set_ylabel(ylabel, fontsize=22)
    ax.set_xticks([-10, -5, 0, 5, 10])
    ax.tick_params(axis="both", labelsize=18)
    ax.grid(True, alpha=0.25)
    return fit_results


def write_outputs(rows, report):
    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "fixation_displacements.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    axis_limits = {}
    p1_mean = float(np.mean([r["p1_abs_right_left_x_mean_px"] for r in rows]))
    p4_lower = min(r["p4_abs_right_left_x_mean_px"] - r["p4_abs_right_left_x_std_px"] for r in rows)
    p4_upper = max(r["p4_abs_right_left_x_mean_px"] + r["p4_abs_right_left_x_std_px"] for r in rows)
    p4_min = float(np.floor((p4_lower - 1) / 5) * 5)
    p4_max = float(np.ceil((p4_upper + 1) / 5) * 5)
    p4_range = p4_max - p4_min
    axis_limits["p4"] = [p4_min, p4_max]
    axis_limits["p1"] = [p1_mean - p4_range/2, p1_mean + p4_range/2]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.8))
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.16, top=0.77, wspace=0.34)
    fit_results = plot_panel(axes[0], rows, "p1_abs_right_left_x_mean_px", "p1_abs_right_left_x_std_px",
                             r"$|P_{1a}-P_{1b}|$ (pixels)", "P1")
    fit_results += plot_panel(axes[1], rows, "p4_abs_right_left_x_mean_px", "p4_abs_right_left_x_std_px",
                              r"$|P_{4a}-P_{4b}|$ (pixels)", "P4")
    handles, labels = axes[0].get_legend_handles_labels()
    handles.extend([
        Line2D([0], [0], color="black", linestyle="--", linewidth=2.2),
        Line2D([0], [0], color="red", marker="x", linestyle="none", markersize=10, markeredgewidth=2.5),
    ])
    labels.extend(["Quadratic fit", "P1 outlier (excluded from fit)"])
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.57, 0.98),
               ncol=3, frameon=False, fontsize=16)
    for ax, name in zip(axes, ("p1", "p4")):
        ax.set_ylim(*axis_limits[name])
        ax.yaxis.set_major_locator(MultipleLocator(5))
    png_path = OUT / "p1_p4_displacement_vs_gaze.png"
    svg_path = OUT / "p1_p4_displacement_vs_gaze.svg"
    pdf_path = OUT / "p1_p4_displacement_vs_gaze.pdf"
    fits_path = OUT / "polynomial_fits.csv"
    with fits_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fit_results[0]))
        writer.writeheader()
        writer.writerows(fit_results)
    fig.savefig(png_path, dpi=180)
    fig.savefig(svg_path)
    fig.savefig(pdf_path)
    plt.close(fig)

    result = {
        "schema": "gaze_displacement_plot_v5",
        "captures": [1, 2, 3, 4],
        "fixation_periods": len(rows),
        "source_fixation_file_sha256": hashlib.sha256(FIXATIONS.read_bytes()).hexdigest(),
        "validity": "paired P1 and P4 rows with all P1 points valid, all P4 points found, and finite coordinates",
        "displacement_definition": "per frame, absolute horizontal difference between the rightmost and leftmost x coordinates in each three-point P1 or P4 set; fixation value is the mean of these nonnegative spans",
        "error_bar_definition": "sample standard deviation of the valid framewise nonnegative horizontal spans within each fixation",
        "noise_filter": {"method": "rolling Hampel despiking per reflection within each fixation",
                         "window_valid_frames": HAMPEL_WINDOW_VALID_FRAMES,
                         "threshold": f"max({HAMPEL_MIN_JUMP_PX:g} px, {HAMPEL_MAD_MULTIPLIER:g} * 1.4826 * local MAD)"},
        "accommodation_labels": "reviewed demand_diopters_label values, one fixed accommodation condition per capture",
        "y_axis_scales": {"limits_px": axis_limits, "tick_interval_px": 5},
        "axis_labels": {"x": "Eye Rotation (deg)", "p1_y": "$|P_{1a}-P_{1b}|$ (pixels)", "p4_y": "$|P_{4a}-P_{4b}|$ (pixels)"},
        "vector_editing": "SVG keeps text as text for editing in Inkscape; labels use LaTeX-style mathtext",
        "polynomial_fits": {"degree": 2, "count": len(fit_results), "p1_outlier": {"capture": 2, "accommodation_D": 4.0, "target_gaze_deg": 5.0, "excluded_from_fit": True}},
        "layout": "1x2: P1 and P4 panels side by side",
        "outputs": [csv_path.name, fits_path.name, png_path.name, svg_path.name, pdf_path.name],
    }
    (OUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {png_path}")
    print(f"Wrote {csv_path}")


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT,
                        help="Output directory (default: this experiment folder)")
    args = parser.parse_args()
    OUT = args.output.resolve()
    rows, report = load_rows()
    write_outputs(rows, report)


if __name__ == "__main__":
    main()
