"""Plot exact-intersection interior metrics against the fresh joint baseline.

Run from the repository root with:
    rtk proxy python experiments/full_position/joint_sensitivity_v1/plot_results.py

Each baseline/candidate pair uses its own shared complete-interior frame IDs.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


HERE = Path(__file__).resolve().parent
VARIANTS = (
    "anchor_weak", "anchor_strong", "prior_weak", "prior_strong",
    "covariance_4x", "reference_q25", "reference_q75",
)
ROWS = (
    ("gaze", "conditional27", "Gaze / 27"),
    ("gaze", "conditional37", "Gaze / 37"),
    ("capture", "conditional27", "Capture / 27"),
    ("capture", "conditional37", "Capture / 37"),
)
METRICS = (
    ("E_px", "Complete-frame error, px"),
    ("G_theta_deg", "Gaze disagreement, degrees"),
    ("G_A_D", "Accommodation disagreement, D"),
)
COLORS = dict(zip(VARIANTS, plt.get_cmap("tab10").colors))
LABELS = {
    "anchor_weak": "Anchor weak",
    "anchor_strong": "Anchor strong",
    "prior_weak": "Prior weak",
    "prior_strong": "Prior strong",
    "covariance_4x": "Covariance 4×",
    "reference_q25": "Reference Q25",
    "reference_q75": "Reference Q75",
}


def usable_folds(summary: dict, folds: list[str], variant: str, model: str) -> tuple[int, int]:
    outcomes = summary["calibrations"]
    candidates = [outcomes.get(f"{variant}/{fold}/{model}", {}) for fold in folds]
    usable = sum(bool(x.get("calibration_converged") and x.get("evaluation_complete"))
                 for x in candidates)
    return usable, len(folds)


def main() -> None:
    summary = json.loads((HERE / "summary.json").read_text())
    with gzip.open(HERE / "paired_comparisons.json.gz", "rt") as stream:
        comparisons = json.load(stream)
    folds = json.loads((HERE / "config.json").read_text())["folds"]

    fig, axes = plt.subplots(4, 3, figsize=(15.5, 12.5), sharey="row")
    y_base = np.arange(len(VARIANTS))
    for row_i, (family, model, row_title) in enumerate(ROWS):
        family_folds = [fold for fold in folds if fold.startswith(family + "_")]
        row_labels = []
        for variant in VARIANTS:
            comparison = comparisons[f"{family}/{variant}/{model}"]["versus_joint_baseline"]
            n = len(comparison["shared_complete_interior_frame_ids"])
            usable, scheduled = usable_folds(summary, family_folds, variant, model)
            row_labels.append(f"{LABELS[variant]}  (n={n}; usable folds={usable}/{scheduled})")
        for col_i, (metric, metric_title) in enumerate(METRICS):
            ax = axes[row_i, col_i]
            ax.set_yticks(y_base)
            ax.set_yticklabels(row_labels)
            for variant_i, variant in enumerate(VARIANTS):
                comparison = comparisons[f"{family}/{variant}/{model}"]["versus_joint_baseline"]
                ids = comparison["shared_complete_interior_frame_ids"]
                n = len(ids)
                y = y_base[variant_i]

                # Empty support or undefined metric stays visibly unavailable.
                values = []
                for side in ("reference", "candidate"):
                    value = (comparison["interior"][side][metric]
                             ["equal_fixation"]["rms"])
                    values.append(value if value is not None and np.isfinite(value) and n > 0 else None)
                baseline, candidate = values
                if baseline is not None and candidate is not None:
                    ax.plot([baseline, candidate], [y, y], color=COLORS[variant],
                            linewidth=1.1, alpha=.75, zorder=1)
                if baseline is not None:
                    ax.scatter(baseline, y, marker="o", s=35, facecolors="white",
                               edgecolors=COLORS[variant], linewidths=1.5, zorder=3)
                if candidate is not None:
                    ax.scatter(candidate, y, marker="D", s=32, color=COLORS[variant],
                               zorder=4)
                if baseline is None or candidate is None:
                    ax.text(.98, y, "unavailable", transform=ax.get_yaxis_transform(),
                            ha="right", va="center", fontsize=7, color="0.4")

            ax.invert_yaxis()
            ax.grid(axis="x", color="0.88", linewidth=.7)
            ax.set_title(metric_title)
            if col_i == 0:
                ax.set_ylabel(row_title)
            else:
                ax.tick_params(axis="y", labelleft=False)
            ax.set_xlabel("Equal-fixation RMS")

    legend = [
        Line2D([0], [0], marker="o", color="0.45", markerfacecolor="white",
               markeredgecolor="0.45", linestyle="None", label="Fresh baseline"),
        Line2D([0], [0], marker="D", color="0.2", markerfacecolor="0.2",
               linestyle="None", label="Sensitivity fit"),
    ]
    fig.suptitle("Joint-training sensitivity vs fresh baseline\n"
                 "Exact shared complete-interior frames; each pair has its own support",
                 y=.99, fontsize=14)
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(.5, .955),
               ncol=2, frameon=False)
    fig.text(.5, .012,
             "Labels give matched frame count and usable calibration+evaluation folds. "
             "An unavailable value is not zero. Full-population coverage and boundary counts "
             "are reported in RESULTS.md and summary.json.",
             ha="center", va="bottom", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .91), h_pad=1.3, w_pad=1.1)
    fig.savefig(HERE / "paired_interior_vs_baseline.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
