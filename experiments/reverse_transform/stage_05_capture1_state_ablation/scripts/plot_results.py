"""Plot saved Capture-1 A0/G/GA states and diagnostics; performs no fitting."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODELS = ("A0", "G", "GA")
LABELS = {"A0": "Stage04 A-only", "G": "Gaze-only", "GA": "Joint gaze + A"}
COLORS = {"A0": "#4c78a8", "G": "#59a14f", "GA": "#d45087"}
FIX_COLORS = plt.cm.viridis(np.linspace(.08, .92, 5))
GAZE_LABELS = ("−10°", "−5°", "0°", "+5°", "+10°")
METRICS = ("median", "p95", "rms")
PLOT_NAMES = ("state_means_by_fixation.png", "forward_inverse_metrics_by_fixation.png",
              "corrected_triangle_means_clouds.png", "conditioning_by_fixation.png",
              "states_by_source_row.png", "delta_gaze_vs_delta_A.png")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_results(folder: Path):
    summary = json.loads((folder / "summary.json").read_text())
    with np.load(folder / "frames.npz", allow_pickle=False) as archive:
        frames = {key: archive[key] for key in archive.files}
    missing = [f"{name}_states" for name in MODELS if f"{name}_states" not in frames]
    if missing:
        raise ValueError(f"Missing state arrays in {folder / 'frames.npz'}: {missing}")
    if set(summary["comparisons"]) != set(MODELS):
        raise ValueError("Expected exactly A0, G, and GA comparison records")
    return summary, frames


def plot_states(summary, frames, results):
    exposures = frames["exposure"]
    nominal_gaze = np.array([r["nominal_gaze_deg"] for r in summary["comparisons"]["GA"]["fixations"]])
    reference_a = float(summary["protocol"]["kappa_law"]["reference_A"])
    offsets = {"A0": -.12, "G": 0., "GA": .12}
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    for name in MODELS:
        means_x, sd_x, means_a, sd_a = [], [], [], []
        states = frames[f"{name}_states"]
        for exposure in range(5):
            values = states[exposures == exposure]
            means_x.append(float(values[:, 0].mean()))
            sd_x.append(float(values[:, 0].std()))
            means_a.append(float(values[:, 1].mean()))
            sd_a.append(float(values[:, 1].std()))
        x = nominal_gaze + offsets[name]
        axes[0].errorbar(x, means_x, yerr=sd_x, fmt="o-", capsize=3, lw=1.5,
                         color=COLORS[name], label=LABELS[name])
        axes[1].errorbar(x, means_a, yerr=sd_a, fmt="o-", capsize=3, lw=1.5,
                         color=COLORS[name], label=LABELS[name])
    axes[0].plot(nominal_gaze, nominal_gaze, marker="x", ls="none", ms=8, color="black",
                 label="Nominal gaze labels")
    axes[0].set_ylabel("Fixation mean θx (deg) ± within-fixation SD")
    axes[0].set_title("Horizontal gaze by nominal fixation")
    axes[1].axhline(reference_a, color="black", ls="--", lw=1.2,
                    label=f"A reference / nominal demand ({reference_a:.5f} D)")
    axes[1].set_ylabel("Fixation mean A (D) ± within-fixation SD")
    axes[1].set_title("Accommodation state by nominal fixation")
    for ax in axes:
        ax.set_xticks(nominal_gaze, GAZE_LABELS)
        ax.set_xlabel("Nominal horizontal gaze label")
        ax.grid(alpha=.25)
        ax.legend(ncol=2, fontsize=8)
    fig.suptitle("Capture 1 fitted state means; gaze/A fixation-mean anchor widths 0.5° / 0.25 D\n"
                 "Error bars are within-fixation SD, not uncertainty of the mean")
    fig.savefig(results / "state_means_by_fixation.png", dpi=170)
    plt.close(fig)


def plot_error_grids(summary, results):
    rows = (
        ("P1 forward", "p1_forward_px"),
        ("P4 forward", "p4_forward_px"),
        ("P1 inverse, common-valid", "p1_inverse_common_px"),
        ("P4 inverse, common-valid", "p4_inverse_common_px"),
    )
    fig, axes = plt.subplots(len(rows), len(METRICS), figsize=(14, 14), sharex=True,
                             constrained_layout=True)
    for ri, (row_label, field) in enumerate(rows):
        for ci, metric in enumerate(METRICS):
            ax = axes[ri, ci]
            for name in MODELS:
                records = summary["comparisons"][name]["fixations"]
                values = [record[field][metric] for record in records]
                ax.plot(np.arange(5), values, marker="o", lw=1.5, ms=4,
                        color=COLORS[name], label=LABELS[name])
            ax.set_title(f"{row_label}: {metric.upper() if metric == 'p95' else metric}")
            ax.set_ylabel("Corresponding-vertex distance (px)")
            ax.set_xticks(np.arange(5), GAZE_LABELS)
            ax.grid(alpha=.25)
            if ri == 0 and ci == 0:
                ax.legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("Nominal fixation")
    fig.suptitle("A0 / G / GA forward and inverse errors by fixation\n"
                 "G/GA use equal P1/P4 weights; inverse panels share common-valid frames across models")
    fig.savefig(results / "forward_inverse_metrics_by_fixation.png", dpi=170)
    plt.close(fig)


def _sample_indices(mask, exposure, rng, per_fixation=120):
    chosen = []
    for fixation in range(5):
        candidates = np.flatnonzero(mask & (exposure == fixation))
        if candidates.size > per_fixation:
            candidates = np.sort(rng.choice(candidates, per_fixation, replace=False))
        chosen.extend(candidates.tolist())
    return np.asarray(chosen, dtype=np.int64)


def plot_corrected_triangles(frames, results):
    refs = {"P1": frames["reference_p1"], "P4": frames["reference_p4"]}
    common = {"P1": frames["common_inverse_valid_p1"],
              "P4": frames["common_inverse_valid_p4"]}
    exposure = frames["exposure"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    sample_record = {}
    vertices = ("#e15759", "#4e79a7", "#59a14f")
    for row, pattern in enumerate(("P1", "P4")):
        seed = 5705 + row
        idx = _sample_indices(common[pattern], exposure, np.random.default_rng(seed))
        for col, name in enumerate(MODELS):
            ax = axes[row, col]
            ref = refs[pattern]
            mask = common[pattern]
            points = frames[f"{name}_recovered_{pattern.lower()}"]
            for vertex in range(3):
                cloud = points[idx, vertex]
                ax.scatter(cloud[:, 0], cloud[:, 1], s=5, alpha=.12,
                           color=vertices[vertex], rasterized=True,
                           label=f"Recovered vertex {vertex + 1}" if row == 0 and col == 0 else None)
            mean_triangle = points[mask].mean(axis=0)
            for triangle, style, label in ((ref, "k--", "Empirical reference"),
                                           (mean_triangle, "k-", "Recovered mean")):
                closed = np.vstack((triangle, triangle[0]))
                ax.plot(closed[:, 0], closed[:, 1], style, marker="o", ms=3,
                        lw=1.2, label=label if row == 0 and col == 0 else None)
            margin = 7.
            ax.set_xlim(float(ref[:, 0].min() - margin), float(ref[:, 0].max() + margin))
            ax.set_ylim(float(ref[:, 1].min() - margin), float(ref[:, 1].max() + margin))
            ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=.2)
            ax.set_title(f"{pattern}, {LABELS[name]}\n{int(mask.sum()):,} common-valid frames; {len(idx):,} sampled")
            ax.set_xlabel("Camera x (px)")
            ax.set_ylabel("Camera y (px)")
            sample_record[f"{name}_{pattern}"] = {"seed": seed,
                                                  "frames": int(len(idx)),
                                                  "common_valid": int(mask.sum())}
    axes[0, 0].legend(fontsize=7, loc="best")
    fig.suptitle("Recovered corresponding triangles in native reference coordinates\n"
                 "No rotation alignment or radius normalization; clouds are deterministic samples")
    fig.savefig(results / "corrected_triangle_means_clouds.png", dpi=170)
    plt.close(fig)
    return sample_record


def plot_conditioning(frames, results):
    fig, axes = plt.subplots(4, 3, figsize=(15, 13), sharex=True, sharey="row",
                             constrained_layout=True)
    rows = (("P4-only acute angle", "p4", "acute_angle_deg", False),
            ("Joint P1+P4 acute angle", "joint", "acute_angle_deg", False),
            ("P4-only condition number", "p4", "condition", True),
            ("Joint P1+P4 condition number", "joint", "condition", True))
    for ri, (label, family, field, log_scale) in enumerate(rows):
        for ci, name in enumerate(MODELS):
            ax = axes[ri, ci]
            key = f"{name}_{family}_{field}"
            values = frames[key]
            groups = [values[frames["exposure"] == fixation] for fixation in range(5)]
            finite_groups = [group[np.isfinite(group)] for group in groups]
            bp = ax.boxplot(finite_groups, positions=np.arange(5), widths=.58,
                            showfliers=False, patch_artist=True,
                            medianprops={"color": "black", "linewidth": 1.2})
            for box in bp["boxes"]:
                box.set_facecolor(FIX_COLORS[2]); box.set_alpha(.48)
            if log_scale:
                ax.set_yscale("log")
            if ri == 0:
                ax.set_title(LABELS[name])
            ax.set_ylabel(label + (" (log scale)" if log_scale else " (deg)"))
            ax.set_xticks(np.arange(5), GAZE_LABELS)
            ax.grid(axis="y", alpha=.25)
    for ax in axes[-1]:
        ax.set_xlabel("Nominal horizontal fixation")
    fig.suptitle("Framewise gaze/A conditioning by nominal gaze; mean-anchor widths 0.5° / 0.25 D\n"
                 "Joint raw condition number mixes degree and diopter units; acute angle is unit-invariant")
    fig.savefig(results / "conditioning_by_fixation.png", dpi=170)
    plt.close(fig)


def plot_states_by_row(frames, results):
    row = frames["row"]
    exposure = frames["exposure"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True, sharey="row",
                             constrained_layout=True)
    for col, name in enumerate(MODELS):
        states = frames[f"{name}_states"]
        for exposure_id in range(5):
            sel = exposure == exposure_id
            axes[0, col].scatter(row[sel], states[sel, 0], s=3, alpha=.32,
                                 color=FIX_COLORS[exposure_id], label=GAZE_LABELS[exposure_id])
            axes[1, col].scatter(row[sel], states[sel, 1], s=3, alpha=.32,
                                 color=FIX_COLORS[exposure_id])
        axes[0, col].set_title(LABELS[name])
        axes[0, col].set_ylabel("θx (deg)")
        axes[1, col].set_ylabel("A (D)")
        axes[1, col].set_xlabel("Source row (points only; no connecting lines)")
        for ax in axes[:, col]:
            ax.grid(alpha=.2)
    axes[0, 0].legend(title="Nominal gaze", fontsize=7, title_fontsize=8)
    fig.suptitle("Estimated states by source row; gaps and source order are not interpolated\n"
                 "Shared y scales within each state row; fixation-mean anchor widths 0.5° / 0.25 D")
    fig.savefig(results / "states_by_source_row.png", dpi=170)
    plt.close(fig)


def plot_delta_state(summary, frames, results):
    initial = frames["gaze_xy_deg"][:, 0]
    baseline_a = frames["baseline_A"]
    states = frames["GA_states"]
    dx = states[:, 0] - initial
    da = states[:, 1] - baseline_a
    exposure = frames["exposure"]
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    for fixation in range(5):
        sel = exposure == fixation
        ax.scatter(dx[sel], da[sel], s=6, alpha=.15, rasterized=True,
                   color=FIX_COLORS[fixation], label=GAZE_LABELS[fixation])
        ax.scatter(float(dx[sel].mean()), float(da[sel].mean()), marker="D", s=40,
                   edgecolor="black", linewidth=.5, color=FIX_COLORS[fixation])
    corr = summary["comparisons"]["GA"]["all"]["delta_gaze_delta_A_correlation"]
    ax.axhline(0, color="black", lw=.7); ax.axvline(0, color="black", lw=.7)
    ax.set_xlabel("GA θx − Stage03 θx (deg)")
    ax.set_ylabel("GA A − saved Stage04 A0 (D)")
    ax.set_title((f"Joint-state changes by fixation; pooled correlation={corr:.3f}" if corr is not None
                  else "Joint-state changes by fixation; pooled correlation undefined") +
                 "\nMean-anchor widths: gaze 0.5°, A 0.25 D")
    ax.grid(alpha=.25); ax.legend(title="Nominal gaze", fontsize=8)
    fig.savefig(results / "delta_gaze_vs_delta_A.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True,
                        help="Saved run directory containing summary.json and frames.npz")
    results = parser.parse_args().results.resolve()
    summary, frames = load_results(results)
    plot_states(summary, frames, results)
    plot_error_grids(summary, results)
    sample_record = plot_corrected_triangles(frames, results)
    plot_conditioning(frames, results)
    plot_states_by_row(frames, results)
    plot_delta_state(summary, frames, results)
    source = Path(__file__).resolve()
    snapshot_rel = Path("experiments/reverse_transform/stage_05_capture1_state_ablation/scripts/plot_results.py")
    snapshot = results / "source_snapshot" / snapshot_rel
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, snapshot)
    fit_files = ("summary.json", "frames.npz", "audit.json", "protocol.json", "provenance.json")
    metadata = {"models": list(MODELS), "sampled_triangle_clouds": sample_record,
                "triangle_cloud_seed_policy": "5705 + pattern_index, shared across A0/G/GA; up to 120 rows per fixation",
                "triangle_coordinates": "Recovered native reference coordinates; no rotation alignment or radius normalization.",
                "state_error_bars": "Within-fixation standard deviation; not uncertainty of the mean.",
                "conditioning": "Raw condition numbers use degree and diopter coordinates; acute angles are unit-invariant.",
                "anchor_widths": {"gaze_mean_deg": summary["protocol"]["gaze_mean_anchor_width_deg"],
                                  "A_mean_D": summary["protocol"]["A_mean_anchor_width_D"]},
                "G_GA_P1_P4_weights": summary["protocol"]["weights"],
                "fit_artifact_sha256": {name: sha256(results / name) for name in fit_files},
                "plot_script_sha256": sha256(source),
                "plot_script_snapshot": str(snapshot_rel),
                "plot_output_sha256": {name: sha256(results / name) for name in PLOT_NAMES}}
    (results / "plot_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
