"""Summarize saved grouped results and export standalone research plots."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .schema import write_json
from .validate import stats


def generate(path):
    path = Path(path)
    saved = json.loads((path/"summary.json").read_text())
    metrics, errors = [], []
    training_means = []
    for key, summary in saved.items():
        fold, name = key.split("/")
        dest = path/fold/name
        calibration = summary["calibration"]
        row = dict(fold=fold, model=name, calibration_converged=summary["calibration_converged"],
                   calibration_cost=min(s["cost"] for s in calibration["alternatives"]),
                   starts_converged=sum(s["converged"] for s in calibration["alternatives"]))
        if summary["calibration_converged"] and (dest/"frames.json").exists():
            for key2 in ["evaluation_rows", "baseline_valid_rows", "estimated_rows", "inverse_failure_rows",
                         "ambiguous_rows", "bound_rows", "weak_rank_rows", "holdout_tests", "testable_holdouts"]:
                row[key2] = summary[key2]
            for key2 in ["heldout_error_px", "geometry_cost", "gaze_mean_anchor_discrepancy_deg",
                         "accommodation_mean_demand_discrepancy_D"]:
                for statistic in ["mean", "median", "rms", "p95"]:
                    row[f"{key2}_{statistic}"] = summary[key2].get(statistic)
            frames = json.loads((dest/"frames.json").read_text())
            for gi in sorted({r["fixation"] for r in frames}):
                selected = [r for r in frames if r["fixation"] == gi and r.get("estimated")]
                if selected:
                    training_means.append(dict(fold=fold, model=name, fixation=gi, role="evaluation",
                        calibration_converged=True,
                        theta=np.mean([r["theta"] for r in selected]), A=np.mean([r["A"] for r in selected]),
                        theta_std=np.std([r["theta"] for r in selected]), A_std=np.std([r["A"] for r in selected]),
                        nominal_theta=selected[0]["nominal_theta"], demand=selected[0]["demand"]))
            holdout_path = dest/"holdouts.jsonl"
            if summary["holdout_tests"] and not holdout_path.exists():
                raise ValueError(f"Missing recorded holdout results: {holdout_path}")
            for line in holdout_path.read_text().splitlines() if holdout_path.exists() else []:
                h = json.loads(line)
                if h.get("score_available"):
                    errors.append(dict(fold=fold, model=name, fixation=h["fixation"], row=h["row"],
                        point=h["held_point"], gaze=h["nominal_theta"], demand=h["demand"],
                        capture=h["capture"], dx=h["error_px"][0], dy=h["error_px"][1],
                        error=float(np.linalg.norm(h["error_px"])), mahalanobis=h.get("mahalanobis")))
        metrics.append(row)
        t = json.loads((dest/"training_states.json").read_text())
        x, groups = np.array(t["states"]), np.array(t["groups"])
        for gi, anchor in enumerate(t["nominal_anchors"]):
            selected = x[groups == gi]
            training_means.append(dict(fold=fold, model=name, fixation=gi, role="training",
                calibration_converged=summary["calibration_converged"],
                theta=selected[:, 0].mean(), A=selected[:, 1].mean(),
                theta_std=selected[:, 0].std(), A_std=selected[:, 1].std(),
                nominal_theta=anchor[0], demand=anchor[1]))
    for filename, rows in [("metrics.csv", metrics), ("heldout_errors.csv", errors), ("fixation_means.csv", training_means)]:
        if rows:
            keys = sorted(set().union(*(r.keys() for r in rows)))
            with (path/filename).open("w") as f:
                writer = csv.DictWriter(f, fieldnames=keys); writer.writeheader(); writer.writerows(rows)
    # Secondary matched support comparison; never replace full coverage counts.
    matched = {}
    for family in ["gaze", "capture"]:
        a = {(e["fold"], e["fixation"], e["row"], e["point"]): e for e in errors
             if e["fold"].startswith(family+"_") and e["model"] == "conditional27"}
        b = {(e["fold"], e["fixation"], e["row"], e["point"]): e for e in errors
             if e["fold"].startswith(family+"_") and e["model"] == "conditional37"}
        common = sorted(a.keys() & b.keys())
        matched[family] = dict(count=len(common), conditional27=stats([a[k]["error"] for k in common]),
                              conditional37=stats([b[k]["error"] for k in common]))
    if errors:
        write_json(path/"matched_holdout_summary.json", matched)
    colors = {"conditional27": "#2674a8", "conditional37": "#db7a29"}
    if errors:
        prediction_plot(path, metrics, errors, colors)
    if any(r["calibration_converged"] for r in training_means):
        anchor_plot(path, training_means, colors)
    return metrics, matched


def prediction_plot(path, metrics, errors, colors):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for ax, family in zip(axes, ["gaze", "capture"]):
        labels = sorted({r["fold"] for r in metrics if r["fold"].startswith(family+"_")},
                        key=lambda s: float(s.split("_")[1]))
        for offset, name in [(-.13, "conditional27"), (.13, "conditional37")]:
            medians, lo, hi = [], [], []
            for label in labels:
                vals = [e["error"] for e in errors if e["fold"] == label and e["model"] == name]
                medians.append(np.median(vals) if vals else np.nan)
                lo.append(np.percentile(vals, 25) if vals else np.nan)
                hi.append(np.percentile(vals, 75) if vals else np.nan)
            ax.errorbar(np.arange(len(labels))+offset, medians,
                        yerr=[np.array(medians)-lo, np.array(hi)-medians], fmt="o", capsize=4,
                        color=colors[name], label=name)
        ax.set_xticks(np.arange(len(labels)), [s.split("_")[1] for s in labels])
        ax.set_xlabel("Held-out nominal gaze (degrees)" if family == "gaze" else "Held-out capture")
        ax.set_ylabel("Held-out point error (pixels), median and IQR")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=.2); ax.legend()
    fig.savefig(path/"heldout_prediction.png", dpi=180); plt.close(fig)


def anchor_plot(path, training_means, colors):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, role in zip(axes, ["training", "evaluation"]):
        for name, color in {**colors, "two_channel13": "#55864d"}.items():
            data = [r for r in training_means if r["model"] == name and r["role"] == role and r["calibration_converged"]]
            ax.scatter([r["demand"] for r in data], [r["A"] for r in data], s=15, alpha=.6, label=name, color=color)
        ax.plot([0, 6], [0, 6], color="gray", ls="--", lw=1)
        ax.set(xlabel="Nominal accommodation demand (D)", ylabel="Fixation-mean estimate (D)",
               title=role.capitalize()+" means; demand is a soft anchor")
        ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.savefig(path/"accommodation_anchors.png", dpi=180); plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    args = p.parse_args()
    generate(args.run)
