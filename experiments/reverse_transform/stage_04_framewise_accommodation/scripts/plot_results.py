"""Plot requested diagnostics from a saved stage04 framewise-A run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"capture_constant": "#4c78a8", "nominal_A": "#f2a541", "framewise_A": "#d45087"}
LABELS = {"capture_constant": "Parent constant κ", "nominal_A": "Frozen nominal A law", "framewise_A": "Framewise fitted A"}


def stats(x):
    x = np.asarray(x, float)
    return {"count": int(x.size), "median": float(np.median(x)), "p95": float(np.percentile(x, 95)),
            "rms": float(np.sqrt(np.mean(x*x)))}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, required=True)
    args = ap.parse_args()
    results = args.results.resolve()
    summary = json.loads((results / "summary.json").read_text())
    with np.load(results / "frames.npz", allow_pickle=False) as z:
        a = {k: z[k] for k in z.files}
    ref = a["reference_p4"]
    caps = a["capture_index"] + 1
    exposures = a["exposure"]
    errors = {}
    for name, points in (("capture_constant", a["parent_recovered_p4"]),
                         ("nominal_A", a["nominal_A_recovered_p4"]),
                         ("framewise_A", a["recovered_p4"])):
        errors[name] = np.linalg.norm(points - ref[None, :, :], axis=2)

    # Per-fixation distributions use the common-valid set for like-for-like methods.
    records = summary["fixations"]
    x = np.arange(len(records))
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True, constrained_layout=True)
    for name in LABELS:
        med, p95, rms = [], [], []
        for rec in records:
            keep = (exposures == rec["exposure"]) & a["common_inverse_valid"]
            vals = errors[name][keep].ravel()
            med.append(np.median(vals) if vals.size else np.nan)
            p95.append(np.percentile(vals, 95) if vals.size else np.nan)
            rms.append(np.sqrt(np.mean(vals*vals)) if vals.size else np.nan)
        axes[0].plot(x, med, marker="o", ms=3, lw=1.5, color=COLORS[name], label=LABELS[name])
        axes[1].plot(x, p95, marker="o", ms=3, lw=1.5, color=COLORS[name], label=LABELS[name])
        axes[2].plot(x, rms, marker="o", ms=3, lw=1.5, color=COLORS[name])
    axes[0].set_ylabel("Median vertex error (px)")
    axes[1].set_ylabel("95th percentile (px)")
    axes[2].set_ylabel("Vertex RMS (px)")
    axes[2].set_xlabel("Fixation (capture blocks × nominal gaze −10, −5, 0, 5, 10°)")
    axes[2].set_xticks(x, [f"C{r['capture']}\n{r['nominal_gaze_deg']}°" for r in records], fontsize=8)
    for ax in axes:
        ax.grid(alpha=.25)
    axes[0].legend(ncol=3, fontsize=9)
    fig.suptitle(f"Common-valid inverse vertex error (n={summary['common_valid']:,} frames; pooled over 3 vertices)")
    fig.savefig(results / "inverse_error_by_fixation.png", dpi=170)
    plt.close(fig)

    # Show all-complete framewise errors, including rows outside the old common set.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for ci, cap in enumerate(range(1, 5)):
        sel = caps == cap
        perfix_med, perfix_p95 = [], []
        for exp in range((cap-1)*5, cap*5):
            vals = errors["framewise_A"][exposures == exp].ravel()
            perfix_med.append(np.median(vals)); perfix_p95.append(np.percentile(vals, 95))
        axes[0].plot(np.arange(5) + ci*5, perfix_med, "o-", color="#d45087", label="Median" if ci == 0 else None)
        axes[1].plot(np.arange(5) + ci*5, perfix_p95, "o-", color="#d45087", label="P95" if ci == 0 else None)
    axes[0].set_ylabel("Median vertex error (px)"); axes[1].set_ylabel("P95 vertex error (px)")
    for ax in axes:
        ax.set_xticks(np.arange(20), [f"C{r['capture']}\n{r['nominal_gaze_deg']}°" for r in records], fontsize=7)
        ax.grid(alpha=.25)
    fig.suptitle("Framewise A errors, all complete frames (no common-valid exclusions)")
    fig.savefig(results / "framewise_error_all_complete.png", dpi=170)
    plt.close(fig)

    # Means and vertex residual clouds, in the same unnormalized coordinate frame.
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    fixation_colors = plt.cm.viridis(np.linspace(.08, .92, 5))
    cloud_tail_counts = {}
    for ci, cap in enumerate(range(1, 5)):
        selcap = caps == cap
        ax = axes[0, ci]
        ax.plot(*np.vstack([ref, ref[0]]).T, "k-", lw=2, label="Reference")
        for exp in range((cap-1)*5, cap*5):
            sel = (exposures == exp) & a["common_inverse_valid"]
            col = fixation_colors[exp % 5]
            for model_name, ls in (("parent_recovered_p4", "--"), ("recovered_p4", "-")):
                pts = a[model_name][sel].mean(axis=0)
                ax.plot(*np.vstack([pts, pts[0]]).T, color=col, marker="o", ms=2.5, lw=1.0, ls=ls)
        ax.set_title(f"C{cap} means")
        ax.set_aspect("equal", adjustable="datalim"); ax.grid(alpha=.2)
        if ci == 0: ax.legend(fontsize=8)

        ax = axes[1, ci]
        rng = np.random.default_rng(9030 + cap)
        shown = 0
        outside = {"parent": 0, "framewise": 0}
        for exp in range((cap-1)*5, cap*5):
            sel = np.flatnonzero((exposures == exp) & a["common_inverse_valid"])
            if sel.size > 150:
                sel = np.sort(rng.choice(sel, 150, replace=False))
            mean_sel = (exposures == exp) & a["common_inverse_valid"]
            for key, shade, label in (("parent_recovered_p4", "#777777", "parent"),
                                      ("recovered_p4", fixation_colors[exp % 5], "framewise")):
                delta = a[key][sel] - ref[None, :, :]
                outside[label] += int(np.sum(np.any((np.abs(delta) > 7).any(axis=2), axis=1)))
                for j in range(3):
                    ax.scatter(delta[:, j, 0], delta[:, j, 1], s=4, alpha=.09 if label == "parent" else .14,
                               color=shade, rasterized=True)
                    mean_delta = a[key][mean_sel].mean(axis=0)[j] - ref[j]
                    ax.scatter(mean_delta[0], mean_delta[1], s=20, marker="o" if label == "framewise" else "x",
                               color=shade, edgecolor="black" if label == "framewise" else None, linewidth=.35)
            shown += len(sel)
        cloud_tail_counts[str(cap)] = {"common_valid_frames": int(np.sum(selcap & a["common_inverse_valid"])),
                                       "subsampled_frames": int(shown), "sampled_frames_with_any_vertex_outside_7px_view": outside}
        ax.axhline(0, color="black", lw=.6); ax.axvline(0, color="black", lw=.6)
        ax.set_xlim(-7, 7); ax.set_ylim(-7, 7); ax.set_aspect("equal")
        ax.set_title(f"C{cap} vertex deviations")
        ax.text(.02, .98,
                f"n={int(np.sum(selcap & a['common_inverse_valid'])):,} valid frames; sampled {shown}\n"
                f"sampled frames with a vertex outside ±7 px: parent {outside['parent']}, fitted {outside['framewise']}",
                transform=ax.transAxes, ha="left", va="top", fontsize=7,
                bbox={"facecolor": "white", "alpha": .82, "edgecolor": "none", "pad": 2})
        ax.grid(alpha=.2)
        if ci == 0:
            ax.set_ylabel("Δy (px)")
        ax.set_xlabel("Δx (px)")
    fig.suptitle("Inverse triangle means and vertex deviations from reference\n"
                 "Common-valid set; hues map −10, −5, 0, 5, 10° dark-to-yellow; "
                 "means parent dashed / fitted solid, clouds parent × / fitted circles",
                 fontsize=11)
    fig.savefig(results / "triangle_means_and_vertex_clouds.png", dpi=170)
    plt.close(fig)

    # A distributions and soft mean anchors.
    fig, ax = plt.subplots(figsize=(13, 5.2), constrained_layout=True)
    groups = [a["A_D"][exposures == r["exposure"]] for r in records]
    bp = ax.boxplot(groups, positions=x, widths=.55, showfliers=False, patch_artist=True,
                    medianprops={"color": "black", "linewidth": 1.2})
    for i, box in enumerate(bp["boxes"]):
        box.set_facecolor(fixation_colors[i % 5]); box.set_alpha(.48)
    means = [r["A_D"]["mean"] for r in records]
    expected = [r["expected_A_D"] for r in records]
    ax.scatter(x, means, color="#202020", s=24, marker="D", label="Fitted fixation mean")
    ax.scatter(x, expected, color="#d45087", s=30, marker="x", label="Expected soft anchor")
    ax.set_xticks(x, [f"C{r['capture']}\n{r['nominal_gaze_deg']}°" for r in records], fontsize=8)
    ax.set_ylabel("A (diopters)"); ax.set_xlabel("Fixation")
    anchor_width=summary['protocol']['anchor_width_D']
    ax.set_title(f"Framewise A distributions and expected fixation means (anchor width {anchor_width:g} D)")
    ax.grid(axis="y", alpha=.25); ax.legend(ncol=2)
    fig.savefig(results / "accommodation_by_fixation.png", dpi=170)
    plt.close(fig)

    # Source-row sequence: scatter only, so gaps and reversed clock regions are not joined.
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=False, constrained_layout=True)
    row_stats = {}
    for ci, (ax, cap) in enumerate(zip(axes, range(1, 5))):
        selcap = caps == cap
        for exp in range((cap-1)*5, cap*5):
            sel = selcap & (exposures == exp)
            ax.scatter(a["row"][sel], a["A_D"][sel], s=3, alpha=.38,
                       color=fixation_colors[exp % 5], label=f"{records[exp]['nominal_gaze_deg']}°")
        ax.set_ylabel(f"C{cap}\nA (D)"); ax.grid(alpha=.2)
        ax.legend(title="Gaze", ncol=5, fontsize=7, title_fontsize=8, loc="upper right")
        row_stats[str(cap)] = int(selcap.sum())
    axes[-1].set_xlabel("Source row (points only; sequence gaps are not connected)")
    fig.suptitle("Framewise A by source row")
    fig.savefig(results / "accommodation_by_source_row.png", dpi=170)
    plt.close(fig)

    # Radius ratios are a diagnostic of residual scale variation, not a fitted or
    # applied normalization. Compute RMS centroid radius against the reference.
    ref_rad = np.sqrt(np.mean(np.sum((ref - ref.mean(axis=0))**2, axis=1)))
    center = a["recovered_p4"].mean(axis=1, keepdims=True)
    frame_rad = np.sqrt(np.mean(np.sum((a["recovered_p4"] - center)**2, axis=2), axis=1))
    ratio = frame_rad / ref_rad
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    byfix = [ratio[exposures == r["exposure"]] for r in records]
    bp = ax.boxplot(byfix, positions=x, widths=.55, showfliers=False, patch_artist=True,
                    medianprops={"color": "black", "linewidth": 1.2})
    for i, box in enumerate(bp["boxes"]):
        box.set_facecolor(fixation_colors[i % 5]); box.set_alpha(.48)
    ax.axhline(1, color="black", lw=1, ls="--", label="Reference radius")
    ax.set_xticks(x, [f"C{r['capture']}\n{r['nominal_gaze_deg']}°" for r in records], fontsize=8)
    ax.set_ylabel("Recovered / reference RMS radius")
    ax.set_xlabel("Fixation"); ax.grid(axis="y", alpha=.25)
    ax.set_title("Remaining recovered-radius variability (diagnostic only; no fit normalization)")
    ax.legend()
    fig.savefig(results / "remaining_radius_variability.png", dpi=170)
    plt.close(fig)

    # Machine-readable plot provenance: metric populations and sampled cloud sizes.
    write = {"common_valid_frames": int(summary["common_valid"]), "complete_frames": int(summary["complete"]),
             "vertex_cloud_frames": cloud_tail_counts,
             "radius_ratio_definition": "RMS radius of each recovered three-vertex triangle about its centroid divided by reference RMS radius.",
             "radius_ratio_summary_by_capture": {str(c): stats(ratio[caps == c]) for c in range(1, 5)},
             "source_row_counts": row_stats}
    (results / "plot_metadata.json").write_text(json.dumps(write, indent=2) + "\n")


if __name__ == "__main__":
    main()
