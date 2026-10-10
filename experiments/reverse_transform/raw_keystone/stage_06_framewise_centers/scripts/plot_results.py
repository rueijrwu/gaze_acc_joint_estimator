#!/usr/bin/env python3
"""Render Stage 06 diagnostics from saved summary.json and frames.npz only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


STAGE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_REL = "experiments/reverse_transform/raw_keystone/stage_06_framewise_centers/scripts/plot_results.py"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def stats(values: np.ndarray) -> tuple[float, float, float]:
    v = np.asarray(values, dtype=float).reshape(-1)
    v = v[np.isfinite(v)]
    if not len(v):
        return float("nan"), float("nan"), float("nan")
    return float(np.median(v)), float(np.percentile(v, 95)), float(np.sqrt(np.mean(v * v)))


def sampled_indices(mask: np.ndarray, limit: int = 8000) -> np.ndarray:
    ids = np.flatnonzero(mask)
    if len(ids) <= limit:
        return ids
    return ids[np.linspace(0, len(ids) - 1, limit, dtype=int)]


def save(fig: plt.Figure, path: Path, outputs: dict[str, str]) -> None:
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    outputs[path.name] = sha256(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True, help="Stage 06 results directory containing summary.json and frames.npz")
    args = parser.parse_args()
    results = args.results.resolve()
    summary_path, frames_path = results / "summary.json", results / "frames.npz"
    if not summary_path.is_file() or not frames_path.is_file():
        raise SystemExit(f"expected {summary_path} and {frames_path}")
    summary = json.loads(summary_path.read_text())
    z = np.load(frames_path)
    required = {
        "row", "source_frame", "capture_index", "exposure", "initial_states", "refined_states",
        "initial_magnification", "refined_magnification", "initial_center_p1_xy", "refined_center_p1_xy",
        "initial_center_p4_xy", "refined_center_p4_xy", "centroid_p1_xy", "centroid_p4_xy",
        "initial_corrected_separation_xy", "refined_corrected_separation_xy",
        "initial_forward_residual_p1", "initial_forward_residual_p4",
        "refined_forward_residual_p1", "refined_forward_residual_p4", "expected_A_D",
    }
    missing = sorted(required - set(z.files))
    if missing:
        raise SystemExit(f"frames.npz is missing required fields: {', '.join(missing)}")
    row = np.asarray(z["row"])
    cap = np.asarray(z["capture_index"], dtype=int)
    exposure = np.asarray(z["exposure"], dtype=int)
    if int(np.min(cap)) == 0:
        cap_label = cap + 1
    else:
        cap_label = cap
    colors = plt.get_cmap("tab10")
    outputs: dict[str, str] = {}

    # A state against source row; dashed expected values are observation-specific.
    expected = np.asarray(z["expected_A_D"], dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharey=True)
    for ax, c in zip(axes.flat, sorted(np.unique(cap_label))):
        mask = cap_label == c
        ids = sampled_indices(mask)
        ax.scatter(row[ids], z["initial_states"][ids, 2], s=4, alpha=.2, color="0.45", label="Initial A")
        ax.scatter(row[ids], z["refined_states"][ids, 2], s=4, alpha=.25, color="#1b9e77", label="Refined A")
        for e in sorted(np.unique(exposure[mask])):
            fixation = mask & (exposure == e)
            if not np.any(fixation):
                continue
            xmin, xmax = float(np.min(row[fixation])), float(np.max(row[fixation]))
            color = colors(int(e) % 5)
            exp_a = float(np.mean(expected[fixation]))
            mean_initial = float(np.mean(z["initial_states"][fixation, 2]))
            mean_refined = float(np.mean(z["refined_states"][fixation, 2]))
            ax.hlines(exp_a, xmin, xmax, color=color, linestyle="--", linewidth=1, alpha=.75,
                      label="Expected A" if e == sorted(np.unique(exposure[mask]))[0] else None)
            ax.hlines(mean_initial, xmin, xmax, color="0.2", linestyle=":", linewidth=1.1,
                      label="Initial fixation mean" if e == sorted(np.unique(exposure[mask]))[0] else None)
            ax.hlines(mean_refined, xmin, xmax, color="#1b9e77", linestyle="-.", linewidth=1.1,
                      label="Refined fixation mean" if e == sorted(np.unique(exposure[mask]))[0] else None)
        ax.set(title=f"Capture {c}", xlabel="Source row (sequence index; not elapsed time)", ylabel="A (D)")
        ax.grid(alpha=.2)
    axes[0, 0].legend(ncol=2, fontsize=8)
    fig.suptitle("Framewise accommodation by capture: source order and fixation means")
    fig.subplots_adjust(hspace=.45, top=.9)
    save(fig, results / "accommodation_by_source_order.png", outputs)

    # Center coordinates by capture and camera point pattern.
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex="col")
    center_names = [("initial_center_p1_xy", "refined_center_p1_xy", "P1 center"),
                    ("initial_center_p4_xy", "refined_center_p4_xy", "P4 center")]
    for j, (ik, rk, title) in enumerate(center_names):
        for c in sorted(np.unique(cap_label)):
            ids = sampled_indices(cap_label == c)
            col = colors((int(c) - 1) % 10)
            for coord, suffix in enumerate(("x", "y")):
                ax = axes[coord, j]
                ax.scatter(row[ids], z[ik][ids, coord], s=4, alpha=.16, color=col)
                ax.scatter(row[ids], z[rk][ids, coord], s=4, alpha=.38, color=col, marker=".")
                ax.set_ylabel(f"{title} {suffix} (px)")
                ax.grid(alpha=.2)
        axes[0, j].set_title(title + " (initial and refined)")
        axes[1, j].set_xlabel("Source row (sequence index; not elapsed time)")
    save(fig, results / "centers_by_source_order.png", outputs)

    # Measured and corrected relative P4-to-P1 centers, per capture.
    measured = np.asarray(z["centroid_p4_xy"] - z["centroid_p1_xy"], dtype=float)
    initial_corr = np.asarray(z["initial_corrected_separation_xy"], dtype=float)
    refined_corr = np.asarray(z["refined_corrected_separation_xy"], dtype=float)
    fig, axes = plt.subplots(1, max(1, len(np.unique(cap_label))), figsize=(4.2 * max(1, len(np.unique(cap_label))), 4.2), squeeze=False)
    for ax, c in zip(axes[0], sorted(np.unique(cap_label))):
        ids = sampled_indices(cap_label == c, limit=5000)
        ax.scatter(measured[ids, 0], measured[ids, 1], s=4, alpha=.16, color="0.5", label="Measured P4−P1")
        ax.scatter(initial_corr[ids, 0], initial_corr[ids, 1], s=4, alpha=.25, color="#d95f02", label="Initial corrected")
        ax.scatter(refined_corr[ids, 0], refined_corr[ids, 1], s=4, alpha=.3, color="#1b9e77", label="Refined corrected")
        ax.set(title=f"Capture {c}", xlabel="Relative center x (px)", ylabel="Relative center y (px)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Measured and model-corrected P4−P1 center separation")
    save(fig, results / "corrected_center_separation_by_capture.png", outputs)

    # Last outer-cycle fixation means and the saved quadratic/shared-slope gaze laws.
    cycles = summary.get("cycles", [])
    if cycles:
        calibration = summary.get("next_calibration", cycles[-1].get("calibration", []))
        if calibration:
            fig, axes = plt.subplots(2, len(calibration), figsize=(4.2 * len(calibration), 8), squeeze=False)
            for j, rec in enumerate(calibration):
                means = np.asarray(rec["fixation_mean_corrected_separation_xy_px"], dtype=float)
                target = np.asarray(rec["nominal_targets_deg"], dtype=float)
                poly_mean = np.asarray(rec["polynomial_at_fixation_means_deg"], dtype=float)
                frame_mean = np.asarray(rec["fixation_mean_estimated_gaze_deg"], dtype=float)
                coeff = np.asarray(rec["coefficients_native_ascending_deg"], dtype=float)
                xgrid = np.linspace(float(means[:, 0].min()), float(means[:, 0].max()), 300)
                axes[0, j].scatter(means[:, 0], target, color="#1b9e77", label="Nominal fixation labels")
                axes[0, j].plot(xgrid, coeff[0] + coeff[1] * xgrid + coeff[2] * xgrid**2,
                                color="#d95f02", label="Quadratic x calibration")
                axes[0, j].scatter(means[:, 0], poly_mean, marker="s", facecolors="none", edgecolors="#7570b3",
                                   label="Polynomial at mean input")
                axes[0, j].scatter(means[:, 0], frame_mean, marker="D", s=23, color="#e7298a",
                                   label="Mean framewise output")
                axes[0, j].set(title=f"Capture {rec['capture']}: horizontal", xlabel="Mean corrected Δx (px)", ylabel="Nominal gaze θx (deg)")
                slope = float(rec["shared_first_order_slope_deg_per_px"])
                origin = float(rec["delta_y_origin_px"])
                ymeans = means[:, 1]
                vertical = slope * (ymeans - origin)
                ygrid = np.linspace(float(ymeans.min()), float(ymeans.max()), 200)
                axes[1, j].scatter(ymeans, vertical, color="#7570b3", label="Fixation means")
                axes[1, j].plot(ygrid, slope * (ygrid - origin), color="#e7298a", label="Shared first-order slope")
                axes[1, j].set(xlabel="Mean corrected Δy (px)", ylabel="Assumed mapped θy (deg)", title=f"Capture {rec['capture']}: vertical")
                for ax in (axes[0, j], axes[1, j]):
                    ax.grid(alpha=.2)
                    ax.legend(fontsize=7)
            fig.suptitle("Corrected-center fixation means and gaze calibration (last outer cycle)")
            fig.subplots_adjust(hspace=.45, wspace=.35, top=.9)
            save(fig, results / "corrected_center_gaze_calibration.png", outputs)

    # Initial/refined horizontal and vertical gaze, direct framewise scatter plus fixation means.
    st0, st1 = np.asarray(z["initial_states"]), np.asarray(z["refined_states"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for k, label in enumerate(("Horizontal gaze θx (deg)", "Vertical gaze θy (deg)")):
        ax = axes[k]
        for c in sorted(np.unique(cap_label)):
            col = colors((int(c) - 1) % 10)
            ids = sampled_indices(cap_label == c, limit=5000)
            ax.scatter(st0[ids, k], st1[ids, k], s=5, alpha=.2, color=col, label=f"C{c}")
            for e in sorted(np.unique(exposure[cap_label == c])):
                sub = (cap_label == c) & (exposure == e)
                if np.any(sub):
                    ax.scatter(np.mean(st0[sub, k]), np.mean(st1[sub, k]), s=35, marker="D", color=col, edgecolor="black", linewidth=.4)
        lo = float(min(np.nanmin(st0[:, k]), np.nanmin(st1[:, k])))
        hi = float(max(np.nanmax(st0[:, k]), np.nanmax(st1[:, k])))
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set(xlabel="Initial", ylabel="Refined", title=label)
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Framewise gaze changes (diamonds mark capture/fixation means)")
    save(fig, results / "gaze_initial_vs_refined.png", outputs)

    # Corresponding-point forward residual magnitudes, native camera pixels.
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    table = {}
    for j, pat in enumerate(("p1", "p4")):
        ax = axes[j]
        for c in sorted(np.unique(cap_label)):
            mask = cap_label == c
            col = colors((int(c) - 1) % 10)
            initial = np.linalg.norm(np.asarray(z[f"initial_forward_residual_{pat}"]), axis=-1)[mask].reshape(-1)
            refined = np.linalg.norm(np.asarray(z[f"refined_forward_residual_{pat}"]), axis=-1)[mask].reshape(-1)
            im, ip, ir = stats(initial)
            rm, rp, rr = stats(refined)
            table[f"C{c}_{pat}"] = {"initial_median_p95_rms_px": [im, ip, ir], "refined_median_p95_rms_px": [rm, rp, rr]}
            x = int(c) - 1
            ax.plot([x - .08, x + .08], [im, rm], marker="o", color=col, label=f"C{c} median" if j == 0 else None)
            ax.plot([x - .08, x + .08], [ip, rp], marker="s", linestyle="--", color=col, alpha=.7, label=f"C{c} P95" if j == 0 else None)
            ax.plot([x - .08, x + .08], [ir, rr], marker="^", linestyle=":", color=col, alpha=.7, label=f"C{c} RMS" if j == 0 else None)
        ax.set_xticks(range(len(np.unique(cap_label))), [f"C{c}" for c in sorted(np.unique(cap_label))])
        ax.set_title("P1" if pat == "p1" else "P4")
        ax.set_ylabel("Corresponding-vertex residual (px)")
        ax.grid(alpha=.2)
    axes[1].set_ylabel("")
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle("Forward point residuals: median, P95, and RMS; initial versus refined")
    save(fig, results / "forward_point_metrics_by_capture.png", outputs)

    # Outer-loop fixed-point changes, separate from the independent A-only starts.
    if cycles:
        state_steps = np.asarray([cycle["maximum_state_step"] for cycle in cycles], dtype=float)
        center_steps = np.asarray([cycle["maximum_center_step_px"] for cycle in cycles], dtype=float)
        fig, axes = plt.subplots(1, 4, figsize=(17, 4))
        cycle_number = np.arange(1, len(cycles) + 1)
        for k, name in enumerate(("θx", "θy")):
            axes[0].plot(cycle_number, state_steps[:, k], marker="o", ms=3, label=name)
        axes[0].set(title="Gaze fixed-point update", xlabel="Outer cycle", ylabel="Maximum change (deg)")
        axes[0].legend()
        axes[1].plot(cycle_number, state_steps[:, 2], marker="o", ms=3, color="#7570b3")
        axes[1].set(title="A-only state update", xlabel="Outer cycle", ylabel="Maximum change (D)")
        for k, name in enumerate(("P1 center", "P4 center")):
            axes[2].plot(cycle_number, center_steps[:, k], marker="o", ms=3, label=name)
        axes[2].set(title="Derived-center update", xlabel="Outer cycle", ylabel="Maximum change (px)")
        axes[2].legend()
        for i, cycle in enumerate(cycles):
            for j, start in enumerate(cycle.get("starts", [])):
                axes[3].scatter(i + 1, start.get("objective_equal_fixation_mean_px2", np.nan),
                                marker=("o", "s", "^")[j % 3], s=24,
                                label=f"A start {j}" if i == 0 else None)
        axes[3].set(title="Fixed-gaze A-only start objectives", xlabel="Outer cycle", ylabel="Objective (px²)")
        axes[3].legend(fontsize=7)
        for ax in axes:
            ax.grid(alpha=.2)
        fig.suptitle("Center→mean→gaze recalibration, then fixed-gaze A-only fit")
        save(fig, results / "outer_cycle_convergence.png", outputs)

    metadata = {
        "plot_scope": "Saved summary.json and frames.npz only; no fitting or refitting.",
        "summary_sha256": sha256(summary_path),
        "frames_sha256": sha256(frames_path),
        "source_sha256": sha256(Path(__file__).resolve()),
        "source_path": SOURCE_REL,
        "outputs_sha256": outputs,
        "forward_residual_metrics": table,
        "outer_cycle_state_changes": [cycle.get("maximum_state_step") for cycle in cycles],
        "outer_cycle_center_changes_px": [cycle.get("maximum_center_step_px") for cycle in cycles],
        "A_only_start_certificates": [[{"objective_equal_fixation_mean_px2": start.get("objective_equal_fixation_mean_px2"),
            "certificate": start.get("certificate"), "CPU_certificate": start.get("CPU_certificate")}
            for start in cycle.get("starts", [])] for cycle in cycles],
        "calibration_zero_reference_policy": [[record.get("zero_reference_policy") for record in cycle.get("calibration", [])]
                                                for cycle in cycles],
        "final_polynomial_at_fixation_means_deg": [[record.get("polynomial_at_fixation_means_deg")
            for record in cycle.get("calibration", [])] for cycle in cycles],
        "final_calibration_source": "summary.next_calibration; computed from final refined framewise centers" if summary.get("next_calibration") else "last saved outer-cycle calibration",
        "source_order_note": "Source row is acquisition sequence order, not elapsed time.",
        "normalization": "No RMS/area normalization or independent P4 scale is applied.",
        "outer_cycles": len(cycles),
    }
    (results / "plot_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"plots": sorted(outputs), "results": str(results)}, indent=2))


if __name__ == "__main__":
    main()
