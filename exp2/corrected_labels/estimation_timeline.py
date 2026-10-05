#!/usr/bin/env python3
"""
Timeline comparison: Gaze and accommodation estimates over time.
Shows holdout-3 and full models overlaid on same plots.
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

print("Loading frame-level predictions for both models...")
h3_pred = load_csv('predictions/holdout3/trained_predictions.csv')
h3_frames = load_csv('comparison/holdout3/all_frames.csv')

# For full model, we need to load from the full folder if available
# For now, we'll use the all_frames data which has both models' estimates
full_frames = load_csv('comparison/full/all_frames.csv')

# Get held-out fixations
held_out_fixations = sorted(np.unique(h3_pred['fixation_index'].astype(int)))

print(f"Found {len(held_out_fixations)} held-out fixations")

# Build lookup for targets and demands from all_frames
target_demand_dict = {}
for i in range(len(h3_frames['fixation_index'])):
    idx = int(h3_frames['fixation_index'][i])
    if idx not in target_demand_dict:
        target_demand_dict[idx] = {
            'target': h3_frames['target'][i],
            'demand': h3_frames['demand'][i]
        }

# Create figure with gridspec - 2 columns (Gaze, Accommodation)
n_fixations = len(held_out_fixations)
n_cols = 2

fig = plt.figure(figsize=(15, 3.5 * n_fixations))
gs = GridSpec(n_fixations, 2, figure=fig, wspace=0.3, hspace=0.4)

fig.suptitle('Model Comparison: Holdout-3 vs Full (Line Plots, No Markers)',
             fontsize=14, fontweight='bold', y=0.995)

# Process each held-out fixation
for row_idx, fixation in enumerate(held_out_fixations):
    fixation = int(fixation)

    # Get data for this fixation from holdout-3 predictions
    h3_mask = (h3_pred['fixation_index'] == fixation)
    h3_frames_mask = (h3_frames['fixation_index'] == fixation)
    full_frames_mask = (full_frames['fixation_index'] == fixation)

    # Holdout-3 frame-level predictions
    h3_pred_frames = h3_pred['frame_index'][h3_mask]
    h3_theta = h3_pred['theta_deg'][h3_mask]
    h3_A = h3_pred['A_diopters'][h3_mask]

    # Full model predictions from all_frames
    full_theta = full_frames['joint_trained_theta_deg'][full_frames_mask]

    # Target and demand
    if fixation in target_demand_dict:
        target = target_demand_dict[fixation]['target']
        demand = target_demand_dict[fixation]['demand']
    else:
        target = np.nan
        demand = np.nan

    # Normalize frame indices
    if len(h3_pred_frames) > 0:
        h3_frames_norm = h3_pred_frames - h3_pred_frames.min()
    else:
        h3_frames_norm = np.array([])

    # For full model, use all_frames data
    full_frame_indices = full_frames['frame_index'][full_frames_mask]
    if len(full_frame_indices) > 0:
        full_frames_norm = full_frame_indices - full_frame_indices.min()
    else:
        full_frames_norm = np.array([])

    # Plot 1: Gaze comparison
    ax1 = fig.add_subplot(gs[row_idx, 0])

    # Holdout-3 gaze line
    valid_h3_theta = ~np.isnan(h3_theta)
    if np.any(valid_h3_theta):
        ax1.plot(h3_frames_norm[valid_h3_theta], h3_theta[valid_h3_theta],
                linewidth=2.5, color='#1f77b4', alpha=0.8, label='Holdout-3')

    # Full gaze line
    valid_full_theta = ~np.isnan(full_theta)
    if np.any(valid_full_theta):
        ax1.plot(full_frames_norm[valid_full_theta], full_theta[valid_full_theta],
                linewidth=2.5, color='#ff7f0e', alpha=0.8, label='Full', linestyle='--')

    # Target line
    if not np.isnan(target):
        ax1.axhline(target, color='#2ca02c', linestyle=':', linewidth=2.5, alpha=0.9,
                   label=f'Target: {target:.2f}°')

    ax1.set_ylabel('Gaze (θ) [deg]', fontsize=11, fontweight='bold')
    ax1.set_title(f'Fixation {fixation}: Gaze (Target={target:.2f}°, Demand={demand:.2f}D)',
                  fontsize=11, fontweight='bold')
    ax1.grid(alpha=0.3, linestyle=':')
    ax1.legend(fontsize=10, loc='best')
    ax1.set_xlabel('Frame Index (relative)', fontsize=10)

    # Plot 2: Accommodation comparison
    ax2 = fig.add_subplot(gs[row_idx, 1])

    # Holdout-3 accommodation line
    valid_h3_A = ~np.isnan(h3_A)
    if np.any(valid_h3_A):
        ax2.plot(h3_frames_norm[valid_h3_A], h3_A[valid_h3_A],
                linewidth=2.5, color='#2ca02c', alpha=0.8, label='Holdout-3')

    # Full accommodation line (if available in all_frames)
    # Note: all_frames might not have accommodation, so we'll skip it if not present

    # Target line
    if not np.isnan(demand):
        ax2.axhline(demand, color='#d62728', linestyle=':', linewidth=2.5, alpha=0.9,
                   label=f'Target: {demand:.2f}D')

    ax2.set_ylabel('Accommodation (A) [D]', fontsize=11, fontweight='bold')
    ax2.set_title(f'Fixation {fixation}: Accommodation (Target={demand:.2f}D)',
                  fontsize=11, fontweight='bold')
    ax2.grid(alpha=0.3, linestyle=':')
    ax2.legend(fontsize=10, loc='best')
    ax2.set_xlabel('Frame Index (relative)', fontsize=10)

plt.savefig('estimation_timeline.png', dpi=150, bbox_inches='tight')
print("Saved: estimation_timeline.png")

# Print summary statistics
print("\n" + "="*100)
print("MODEL COMPARISON: HELD-OUT FIXATION STATISTICS")
print("="*100)

for fixation in held_out_fixations:
    fixation = int(fixation)

    h3_mask = (h3_pred['fixation_index'] == fixation)
    full_mask = (full_frames['fixation_index'] == fixation)

    h3_theta = h3_pred['theta_deg'][h3_mask]
    h3_A = h3_pred['A_diopters'][h3_mask]
    full_theta = full_frames['joint_trained_theta_deg'][full_mask]

    if fixation in target_demand_dict:
        target = target_demand_dict[fixation]['target']
        demand = target_demand_dict[fixation]['demand']
    else:
        target = np.nan
        demand = np.nan

    h3_theta_valid = h3_theta[~np.isnan(h3_theta)]
    h3_A_valid = h3_A[~np.isnan(h3_A)]
    full_theta_valid = full_theta[~np.isnan(full_theta)]

    print(f"\nFixation {fixation} (Target: {target:.2f}°, Demand: {demand:.2f}D)")
    print(f"  {'Gaze (θ)':20s} | {'Holdout-3':25s} | {'Full':25s}")
    print(f"  {'-'*20} | {'-'*25} | {'-'*25}")

    if len(h3_theta_valid) > 0 and len(full_theta_valid) > 0:
        h3_bias = np.mean(h3_theta_valid) - target
        full_bias = np.mean(full_theta_valid) - target
        h3_rmse = np.sqrt(np.mean((h3_theta_valid - target)**2))
        full_rmse = np.sqrt(np.mean((full_theta_valid - target)**2))

        print(f"  {'Mean':20s} | {np.mean(h3_theta_valid):+.4f}°         | {np.mean(full_theta_valid):+.4f}°")
        print(f"  {'Bias':20s} | {h3_bias:+.4f}°         | {full_bias:+.4f}°")
        print(f"  {'RMSE':20s} | {h3_rmse:.4f}°         | {full_rmse:.4f}°")
        print(f"  {'Std Dev':20s} | {np.std(h3_theta_valid):.4f}°         | {np.std(full_theta_valid):.4f}°")
        print(f"  {'N frames':20s} | {len(h3_theta_valid):6d}         | {len(full_theta_valid):6d}")

    print(f"\n  {'Accommodation (A)':20s} | {'Holdout-3':25s}")
    print(f"  {'-'*20} | {'-'*25}")

    if len(h3_A_valid) > 0:
        h3_A_bias = np.mean(h3_A_valid) - demand
        h3_A_rmse = np.sqrt(np.mean((h3_A_valid - demand)**2))

        print(f"  {'Mean':20s} | {np.mean(h3_A_valid):+.4f}D")
        print(f"  {'Bias':20s} | {h3_A_bias:+.4f}D")
        print(f"  {'RMSE':20s} | {h3_A_rmse:.4f}D")
        print(f"  {'Std Dev':20s} | {np.std(h3_A_valid):.4f}D")
        print(f"  {'N frames':20s} | {len(h3_A_valid):6d}")

print("\n" + "="*100)
