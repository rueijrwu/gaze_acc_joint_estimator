#!/usr/bin/env python3
"""
Timeline visualization: Gaze and accommodation estimates over time within fixations.
Shows frame-level estimation traces for held-out fixations.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def load_csv(filename):
    """Simple CSV loader."""
    with open(filename) as f:
        header = f.readline().strip().split(',')
        data = {}
        for col in header:
            data[col] = []

        for line in f:
            values = line.strip().split(',')
            if len(values) == len(header):
                for i, col in enumerate(header):
                    try:
                        val = float(values[i]) if values[i] else np.nan
                    except ValueError:
                        val = values[i]
                    data[col].append(val)

    for col in data:
        try:
            data[col] = np.array(data[col], dtype=float)
        except (ValueError, TypeError):
            data[col] = np.array(data[col])

    return data

print("Loading frame-level predictions...")
h3_pred = load_csv('predictions/holdout3/trained_predictions.csv')
h3_frames = load_csv('comparison/holdout3/all_frames.csv')
h3_pf = load_csv('comparison/holdout3/per_fixation.csv')

# Get held-out fixations (where predictions exist)
held_out_fixations = sorted(np.unique(h3_pred['fixation_index'].astype(int)))

print(f"Found {len(held_out_fixations)} held-out fixations with {len(h3_pred)} total frames")

# Build lookup for targets and demands from all_frames
target_demand_dict = {}
for i in range(len(h3_frames['fixation_index'])):
    idx = int(h3_frames['fixation_index'][i])
    if idx not in target_demand_dict:
        target_demand_dict[idx] = {
            'target': h3_frames['target'][i],
            'demand': h3_frames['demand'][i]
        }

# Create figure with gridspec
n_fixations = len(held_out_fixations)
n_cols = 2  # Gaze and Accommodation side by side
n_rows = n_fixations

fig = plt.figure(figsize=(14, 4 * n_fixations))
gs = GridSpec(n_fixations, 2, figure=fig, wspace=0.3, hspace=0.4)

fig.suptitle('Joint Algorithm: Frame-Level Gaze and Accommodation Over Time (Held-Out Fixations)',
             fontsize=14, fontweight='bold', y=0.995)

# Process each held-out fixation
for row_idx, fixation in enumerate(held_out_fixations):
    fixation = int(fixation)

    # Get data for this fixation
    pred_mask = (h3_pred['fixation_index'] == fixation)
    frame_mask = (h3_frames['fixation_index'] == fixation)

    if fixation in target_demand_dict:
        target = target_demand_dict[fixation]['target']
        demand = target_demand_dict[fixation]['demand']
    else:
        target = np.nan
        demand = np.nan

    pred_frames = h3_pred['frame_index'][pred_mask]
    theta = h3_pred['theta_deg'][pred_mask]
    A = h3_pred['A_diopters'][pred_mask]

    frame_theta = h3_frames['joint_trained_theta_deg'][frame_mask]

    # Normalize frame indices to start at 0
    if len(pred_frames) > 0:
        frames_norm = pred_frames - pred_frames.min()
    else:
        frames_norm = np.array([])

    # Plot 1: Gaze over time
    ax1 = fig.add_subplot(gs[row_idx, 0])
    valid_theta = ~np.isnan(theta)
    n_valid = np.sum(valid_theta)

    if np.any(valid_theta):
        # Plot trace with points
        ax1.plot(frames_norm[valid_theta], theta[valid_theta],
                'o-', markersize=5, linewidth=2, color='#1f77b4', alpha=0.8, label=f'Estimated ({n_valid} frames)')

        # Add uncertainty band
        theta_std = np.std(theta[valid_theta])
        theta_mean = np.mean(theta[valid_theta])
        ax1.fill_between(frames_norm[valid_theta],
                         theta_mean - theta_std, theta_mean + theta_std,
                         alpha=0.15, color='#1f77b4', label=f'±1 SD: {theta_std:.3f}°')

    if not np.isnan(target):
        ax1.axhline(target, color='#ff7f0e', linestyle='--', linewidth=2.5, alpha=0.9,
                   label=f'Target: {target:.2f}°')
        # Add gaze bias
        if np.any(valid_theta):
            bias = np.mean(theta[valid_theta]) - target
            ax1.text(0.98, 0.05, f'Bias: {bias:+.3f}°',
                    transform=ax1.transAxes, ha='right', va='bottom',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=10, fontweight='bold')

    ax1.set_ylabel('Gaze (θ) [deg]', fontsize=11, fontweight='bold')
    ax1.set_title(f'Fixation {fixation}: Gaze (Target={target:.2f}°, Demand={demand:.2f}D)',
                  fontsize=11, fontweight='bold')
    ax1.grid(alpha=0.3, linestyle=':')
    ax1.legend(fontsize=9, loc='upper left')
    ax1.set_xlabel('Frame Index (relative)', fontsize=10)

    # Plot 2: Accommodation over time
    ax2 = fig.add_subplot(gs[row_idx, 1])
    valid_A = ~np.isnan(A)
    n_valid_a = np.sum(valid_A)

    if np.any(valid_A):
        # Plot trace with points
        ax2.plot(frames_norm[valid_A], A[valid_A],
                'o-', markersize=5, linewidth=2, color='#2ca02c', alpha=0.8, label=f'Estimated ({n_valid_a} frames)')

        # Add uncertainty band
        A_std = np.std(A[valid_A])
        A_mean = np.mean(A[valid_A])
        ax2.fill_between(frames_norm[valid_A],
                         A_mean - A_std, A_mean + A_std,
                         alpha=0.15, color='#2ca02c', label=f'±1 SD: {A_std:.3f}D')

    if not np.isnan(demand):
        ax2.axhline(demand, color='#d62728', linestyle='--', linewidth=2.5, alpha=0.9,
                   label=f'Target: {demand:.2f}D')
        # Add accommodation bias
        if np.any(valid_A):
            bias_a = np.mean(A[valid_A]) - demand
            ax2.text(0.98, 0.05, f'Bias: {bias_a:+.3f}D',
                    transform=ax2.transAxes, ha='right', va='bottom',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5), fontsize=10, fontweight='bold')

    ax2.set_ylabel('Accommodation (A) [D]', fontsize=11, fontweight='bold')
    ax2.set_title(f'Fixation {fixation}: Accommodation (Target={demand:.2f}D)',
                  fontsize=11, fontweight='bold')
    ax2.grid(alpha=0.3, linestyle=':')
    ax2.legend(fontsize=9, loc='upper left')
    ax2.set_xlabel('Frame Index (relative)', fontsize=10)

plt.savefig('estimation_timeline.png', dpi=150, bbox_inches='tight')
print("Saved: estimation_timeline.png")

# Print summary statistics
print("\n" + "="*100)
print("HELD-OUT FIXATION ESTIMATION STATISTICS (Frame-Level)")
print("="*100)

for fixation in held_out_fixations:
    fixation = int(fixation)

    mask = (h3_pred['fixation_index'] == fixation)

    theta = h3_pred['theta_deg'][mask]
    A = h3_pred['A_diopters'][mask]

    if fixation in target_demand_dict:
        target = target_demand_dict[fixation]['target']
        demand = target_demand_dict[fixation]['demand']
    else:
        target = np.nan
        demand = np.nan

    theta_valid = theta[~np.isnan(theta)]
    A_valid = A[~np.isnan(A)]

    print(f"\nFixation {fixation}:")
    print(f"  Target: {target:.2f}°  | Demand: {demand:.2f}D")

    if len(theta_valid) > 0:
        gaze_error = theta_valid - target
        print(f"  Gaze ({len(theta_valid)} frames):")
        print(f"    Mean: {np.mean(theta_valid):+.4f}° | Bias: {np.mean(gaze_error):+.4f}° | SD: {np.std(theta_valid):.4f}°")
        print(f"    RMSE: {np.sqrt(np.mean(gaze_error**2)):.4f}° | Range: [{np.min(theta_valid):.4f}, {np.max(theta_valid):.4f}]°")

    if len(A_valid) > 0:
        accom_error = A_valid - demand
        print(f"  Accommodation ({len(A_valid)} frames):")
        print(f"    Mean: {np.mean(A_valid):+.4f}D | Bias: {np.mean(accom_error):+.4f}D | SD: {np.std(A_valid):.4f}D")
        print(f"    RMSE: {np.sqrt(np.mean(accom_error**2)):.4f}D | Range: [{np.min(A_valid):.4f}, {np.max(A_valid):.4f}]D")

print("\n" + "="*100)
