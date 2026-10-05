#!/usr/bin/env python3
"""
Comparison plot: Holdout-3 model (on held-out 3D fixations) vs Full model (all fixations).
"""

import numpy as np
import matplotlib.pyplot as plt

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

# Load frame-level data
print("Loading frame-level predictions...")
h3_frames = load_csv('comparison/holdout3/all_frames.csv')
full_frames = load_csv('comparison/full/all_frames.csv')

# Extract held-out data for holdout-3 (cohort=='heldout')
# cohort is a string array, check for 'heldout'
h3_cohort = h3_frames['cohort']
h3_heldout_mask = np.array([str(c) == 'heldout' for c in h3_cohort])

# For full, use all frames
full_mask = ~np.isnan(full_frames['joint_trained_theta_deg'])

# Get predictions and targets
h3_target = h3_frames['target'][h3_heldout_mask]
h3_pred = h3_frames['joint_trained_theta_deg'][h3_heldout_mask]
h3_demand = h3_frames['demand'][h3_heldout_mask]
h3_fixation = h3_frames['fixation_index'][h3_heldout_mask]

full_target = full_frames['target'][full_mask]
full_pred = full_frames['joint_trained_theta_deg'][full_mask]
full_demand = full_frames['demand'][full_mask]
full_fixation = full_frames['fixation_index'][full_mask]

# Remove NaNs
h3_valid = ~np.isnan(h3_pred)
full_valid = ~np.isnan(full_pred)

h3_target = h3_target[h3_valid]
h3_pred = h3_pred[h3_valid]
h3_demand = h3_demand[h3_valid]
h3_fixation = h3_fixation[h3_valid]

full_target = full_target[full_valid]
full_pred = full_pred[full_valid]
full_demand = full_demand[full_valid]
full_fixation = full_fixation[full_valid]

print(f"Holdout-3: {len(h3_pred)} heldout frames")
print(f"Full: {len(full_pred)} frames")

# Compute metrics
h3_residual = h3_pred - h3_target
full_residual = full_pred - full_target

h3_rmse = np.sqrt(np.mean(h3_residual**2))
full_rmse = np.sqrt(np.mean(full_residual**2))
h3_bias_rms = np.sqrt(np.mean(h3_residual**2))
full_bias_rms = np.sqrt(np.mean(full_residual**2))

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle('Joint Algorithm Baseline Comparison\nHoldout-3 (held-out) vs Full (in-sample)',
             fontsize=14, fontweight='bold')

# Plot 1: Prediction scatter
ax = axes[0, 0]
ax.scatter(h3_target, h3_pred, alpha=0.4, s=20, label=f'Holdout-3 (N={len(h3_pred)})', color='#1f77b4')
ax.scatter(full_target, full_pred, alpha=0.2, s=5, label=f'Full (N={len(full_pred)})', color='#ff7f0e')
target_range = np.array([min(h3_target.min(), full_target.min()),
                         max(h3_target.max(), full_target.max())])
ax.plot(target_range, target_range, 'k--', alpha=0.5, linewidth=2, label='Perfect agreement')
ax.set_xlabel('Target Gaze [deg]', fontsize=11)
ax.set_ylabel('Predicted Gaze [deg]', fontsize=11)
ax.set_title('Frame-Level Prediction Accuracy')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax.set_aspect('equal', adjustable='box')

# Plot 2: Residuals histogram
ax = axes[0, 1]
bins = np.linspace(-2, 2, 41)
ax.hist(h3_residual, bins=bins, alpha=0.6, label=f'Holdout-3 (σ={np.std(h3_residual):.3f}°)',
        color='#1f77b4', density=True)
ax.hist(full_residual[::10], bins=bins, alpha=0.4, label=f'Full (σ={np.std(full_residual):.3f}°)',
        color='#ff7f0e', density=True)
ax.axvline(0, color='k', linestyle='--', linewidth=1, alpha=0.5)
ax.set_xlabel('Prediction Error [deg]', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Error Distribution')
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)

# Plot 3: Per-demand RMSE
ax = axes[1, 0]
unique_demands = sorted(np.unique(h3_demand))
h3_rmse_by_demand = []
full_rmse_by_demand = []
labels = []

for dem in unique_demands:
    h3_mask = h3_demand == dem
    full_mask = full_demand == dem

    h3_rmse_dem = np.sqrt(np.mean((h3_pred[h3_mask] - h3_target[h3_mask])**2)) if np.any(h3_mask) else np.nan
    full_rmse_dem = np.sqrt(np.mean((full_pred[full_mask] - full_target[full_mask])**2)) if np.any(full_mask) else np.nan

    h3_rmse_by_demand.append(h3_rmse_dem)
    full_rmse_by_demand.append(full_rmse_dem)
    labels.append(f'{dem:.2f}D')

x_pos = np.arange(len(labels))
width = 0.35
ax.bar(x_pos - width/2, h3_rmse_by_demand, width, label='Holdout-3', color='#1f77b4', alpha=0.8)
ax.bar(x_pos + width/2, full_rmse_by_demand, width, label='Full', color='#ff7f0e', alpha=0.8)
ax.set_ylabel('RMSE [deg]', fontsize=11)
ax.set_title('RMSE per Accommodation Demand')
ax.set_xticks(x_pos)
ax.set_xticklabels(labels)
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)

# Plot 4: Statistics table
ax = axes[1, 1]
ax.axis('off')

table_data = [
    ['Metric', 'Holdout-3', 'Full', 'Unit'],
    ['Sample count', f'{len(h3_pred)}', f'{len(full_pred)}', 'frames'],
    ['RMSE', f'{h3_rmse:.4f}', f'{full_rmse:.4f}', 'deg'],
    ['Mean bias', f'{np.mean(h3_residual):+.4f}', f'{np.mean(full_residual):+.4f}', 'deg'],
    ['Bias RMS', f'{np.sqrt(np.mean(h3_residual**2)):+.4f}', f'{np.sqrt(np.mean(full_residual**2)):+.4f}', 'deg'],
    ['Std dev', f'{np.std(h3_residual):.4f}', f'{np.std(full_residual):.4f}', 'deg'],
    ['Min error', f'{np.min(h3_residual):+.4f}', f'{np.min(full_residual):+.4f}', 'deg'],
    ['Max error', f'{np.max(h3_residual):+.4f}', f'{np.max(full_residual):+.4f}', 'deg'],
]

table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                colWidths=[0.2, 0.25, 0.25, 0.2])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2.0)

for i in range(4):
    table[(0, i)].set_facecolor('#40466e')
    table[(0, i)].set_text_props(weight='bold', color='white')

for i in range(1, len(table_data)):
    for j in range(4):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#f0f0f0')

plt.tight_layout()
plt.savefig('comparison_baseline_plot.png', dpi=150, bbox_inches='tight')
print("\nSaved: comparison_baseline_plot.png")

# Print summary
print("\n" + "="*70)
print("BASELINE PERFORMANCE COMPARISON")
print("="*70)
print(f"\nHoldout-3 (Held-out 3D fixations only):")
print(f"  Frames analyzed: {len(h3_pred)}")
print(f"  RMSE: {h3_rmse:.4f}°")
print(f"  Mean bias: {np.mean(h3_residual):+.4f}°")
print(f"  Bias RMS: {np.sqrt(np.mean(h3_residual**2)):+.4f}°")
print(f"  Std dev: {np.std(h3_residual):.4f}°")

print(f"\nFull (All in-sample fixations):")
print(f"  Frames analyzed: {len(full_pred)}")
print(f"  RMSE: {full_rmse:.4f}°")
print(f"  Mean bias: {np.mean(full_residual):+.4f}°")
print(f"  Bias RMS: {np.sqrt(np.mean(full_residual**2)):+.4f}°")
print(f"  Std dev: {np.std(full_residual):.4f}°")

print(f"\nPer-Demand Performance:")
for i, dem in enumerate(unique_demands):
    if not np.isnan(h3_rmse_by_demand[i]):
        print(f"\n  Demand {dem:.2f} D:")
        print(f"    Holdout-3 RMSE: {h3_rmse_by_demand[i]:.4f}°")
        print(f"    Full RMSE:      {full_rmse_by_demand[i]:.4f}°")

print("\n" + "="*70)
print("Note: Holdout-3 comparison is on held-out test data (fair assessment)")
print("      Full comparison is on training data (will show lower RMSE)")
print("="*70)
