"""Plot full-video pair angles and scale from detections.pkl."""

import argparse
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


experiment_dir = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description="Plot pair angle and scale metrics.")
parser.add_argument("input_pickle", nargs="?", type=Path,
                    default=experiment_dir / "detections.pkl",
                    help="tracking result pickle (default: exp2/detections.pkl)")
args = parser.parse_args()
results_path = args.input_pickle.expanduser().resolve()
angles_plot_path = experiment_dir / "pair_angles_full.png"
histogram_path = experiment_dir / "pair_metrics_hist.png"
position_plot_path = experiment_dir / "pair_vs_pupil_position.png"

with results_path.open("rb") as file:
    payload = pickle.load(file)
arrays = payload["arrays"]
valid_pairs = arrays["pair_valid"]
frame_indices = arrays["frame_index"]

fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True, layout="constrained")
axes[0].plot(frame_indices, arrays["p1_angle_deg"], label="P1", linewidth=0.8)
axes[0].plot(frame_indices, arrays["p4_angle_deg"], label="P4", linewidth=0.8)
axes[0].set_ylabel("Pair angle (degrees)")
axes[0].legend(loc="upper right")
axes[1].plot(frame_indices, arrays["angle_difference_deg"], color="tab:purple",
             linewidth=0.8)
axes[1].set_ylabel("P4 − P1 (degrees)")
axes[2].plot(frame_indices, arrays["scale_difference_percent"], color="tab:green",
             linewidth=0.8)
axes[2].set_ylabel("Scale change (%)")
axes[2].set_xlabel("Frame index")
for axis in axes:
    axis.grid(alpha=0.25)
axes[2].set_xlim(0, len(frame_indices) - 1)
fig.savefig(angles_plot_path, dpi=150)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
axes[0].hist(arrays["angle_difference_deg"][valid_pairs], bins=60, color="tab:purple")
axes[0].set_xlabel("P4 − P1 angle (degrees)")
axes[0].set_ylabel("Frames")
axes[1].hist(arrays["scale_difference_percent"][valid_pairs], bins=60,
             color="tab:green")
axes[1].set_xlabel("Scale change from seed (%)")
axes[1].set_ylabel("Frames")
for axis in axes:
    axis.grid(alpha=0.2)
fig.savefig(histogram_path, dpi=150)
plt.close(fig)

fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
for row, metric in enumerate(("angle_difference_deg", "scale_difference_percent")):
    for column in range(2):
        pupil_coordinate = arrays["pupil_center"][valid_pairs, column]
        measurement = arrays[metric][valid_pairs]
        plot = axes[row, column].hexbin(pupil_coordinate, measurement, gridsize=65,
                                        bins="log", mincnt=1, cmap="viridis")
        correlation = np.corrcoef(pupil_coordinate, measurement)[0, 1]
        axes[row, column].set_title(f"r = {correlation:.3f}")
        axes[row, column].set_xlabel(f"Pupil center {'x' if column == 0 else 'y'} (pixels)")
        axes[row, column].set_ylabel("P4 − P1 angle (degrees)" if row == 0
                                     else "Scale change from seed (%)")
        fig.colorbar(plot, ax=axes[row, column], label="Frames (log color scale)")
fig.savefig(position_plot_path, dpi=150)
plt.close(fig)

angle = arrays["angle_error_deg"][valid_pairs]
scale = arrays["scale_difference_percent"][valid_pairs]
print(
    f"Analyzed {len(frame_indices)} frames; {int(valid_pairs.sum())} selected pairs; "
    f"max angle error={angle.max():.3f} deg; "
    f"scale change range={scale.min():.3f}% to {scale.max():.3f}%. "
    f"Saved {angles_plot_path}, {histogram_path}, and {position_plot_path}"
)
