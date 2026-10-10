#!/usr/bin/env python3
"""Plot saved framewise A estimates in source-row order for each capture."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "run"
INPUT = RESULTS / "frames.npz"
OUTPUT = RESULTS / "accommodation_by_source_order.png"


def main():
    z = np.load(INPUT)
    rows = z["row"]
    captures = z["capture_index"]
    gaze = z["gaze_xy_deg"]
    accommodation = z["A_D"]
    expected = z["expected_A_D"]
    colors = plt.get_cmap("tab10").colors
    fig, axes = plt.subplots(4, 1, figsize=(13, 12), sharex=False, sharey=True)
    legend_handles = {}

    # Exposure is the scheduled fixation label. Gaze x is used only to label
    # the five fixation groups; displayed estimates remain the saved A values.
    for cap, ax in enumerate(axes):
        cmask = captures == cap
        exp_ids = np.unique(z["exposure"][cmask])
        for exp in exp_ids:
            mask = cmask & (z["exposure"] == exp)
            nominal_gaze = float(np.median(gaze[mask, 0]))
            color = colors[int(np.argmin(np.abs(np.array([-10, -5, 0, 5, 10]) - nominal_gaze)))]
            order = np.argsort(rows[mask])
            x = rows[mask][order]
            y = accommodation[mask][order]
            expected_a = float(np.median(expected[mask]))
            label = f"{nominal_gaze:+.0f}° gaze"
            points = ax.scatter(x, y, s=3, alpha=0.35, color=color, rasterized=True, label=label)
            legend_handles.setdefault(label, points)
            ax.hlines(expected_a, x.min(), x.max(), color="0.25", lw=1.2, ls="--", alpha=0.8)
        nominal_a = float(np.median(expected[cmask]))
        ax.set_title(f"Capture {cap + 1} — nominal A={nominal_a:g} D", loc="left")
        ax.set_ylabel("Estimated A (D)")
        ax.grid(True, alpha=0.2)
    axes[-1].set_xlabel("Source row / frame order (not elapsed time)")
    fig.suptitle("Framewise accommodation by capture and fixation", y=0.99)
    fig.legend(legend_handles.values(), legend_handles.keys(), loc="upper center",
               bbox_to_anchor=(0.5, 0.965), ncol=5, fontsize=9, frameon=False)
    fig.text(0.5, 0.94, "Soft fixation-mean anchors: width 0.25 D; dashed lines show nominal A",
             ha="center", va="center", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUTPUT, dpi=180)
    print(OUTPUT)


if __name__ == "__main__":
    main()
