"""Plot horizontal pair separations against their mean-position offset."""

import argparse
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


experiment_dir = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description="Plot horizontal P1/P4 pair metrics.")
parser.add_argument("input_pickle", nargs="?", type=Path,
                    default=experiment_dir / "detections.pkl",
                    help="tracking result pickle (default: exp2/detections.pkl)")
args = parser.parse_args()
results_path = args.input_pickle.expanduser().resolve()
plot_path = experiment_dir / "pair_x_distances.png"
time_plot_path = experiment_dir / "pair_x_over_time.png"

with results_path.open("rb") as file:
    payload = pickle.load(file)
arrays = payload["arrays"]
valid_pairs = arrays["pair_valid"]

p1_x_distance = np.abs(arrays["p1_vector"][:, 0])
p4_x_distance = np.abs(arrays["p4_vector"][:, 0])
p4_to_p1_x_distance_ratio = p4_x_distance / p1_x_distance
p4_minus_p1_mean_x = (arrays["p4_points"].mean(axis=1)[:, 0] -
                      arrays["p1_points"].mean(axis=1)[:, 0])

fig, axes = plt.subplots(1, 3, figsize=(18, 5), layout="constrained")
for axis, distance, label, color in (
        (axes[0], p1_x_distance, "|P1 right x − P1 left x| (pixels)", "tab:blue"),
        (axes[1], p4_x_distance, "|P4 right x − P4 left x| (pixels)", "tab:orange"),
        (axes[2], p4_to_p1_x_distance_ratio,
         "|P4 right x − P4 left x| / |P1 right x − P1 left x|", "tab:green")):
    x = p4_minus_p1_mean_x[valid_pairs]
    y = distance[valid_pairs]
    axis.scatter(x, y, s=5, alpha=0.12, color=color, edgecolors="none",
                 rasterized=True)
    correlation = np.corrcoef(x, y)[0, 1]
    axis.set_title(f"r = {correlation:.3f}")
    axis.set_xlabel("Mean P4 x − mean P1 x (pixels)")
    axis.set_ylabel(label)
    axis.grid(alpha=0.2)

fig.savefig(plot_path, dpi=150)
plt.close(fig)

frame_index = arrays["frame_index"]
fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True, layout="constrained")
axes[0].plot(frame_index, p1_x_distance, color="tab:blue", linewidth=0.8)
axes[0].set_ylabel("|P1 pair Δx| (pixels)")
axes[1].plot(frame_index, p4_x_distance, color="tab:orange", linewidth=0.8)
axes[1].set_ylabel("|P4 pair Δx| (pixels)")
axes[2].plot(frame_index, p4_to_p1_x_distance_ratio, color="tab:green", linewidth=0.8)
axes[2].set_ylabel("|P4 pair Δx| / |P1 pair Δx|")
axes[3].plot(frame_index, p4_minus_p1_mean_x, color="tab:purple", linewidth=0.8)
axes[3].set_ylabel("Mean P4 x − mean P1 x (pixels)")
axes[3].set_xlabel("Frame index")
for axis in axes:
    axis.grid(alpha=0.25)
axes[3].set_xlim(0, len(frame_index) - 1)
fig.savefig(time_plot_path, dpi=150)
plt.close(fig)
print(f"Plotted {int(valid_pairs.sum())} selected pairs; saved {plot_path} and {time_plot_path}")
