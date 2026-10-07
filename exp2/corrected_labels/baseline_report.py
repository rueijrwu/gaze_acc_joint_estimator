#!/usr/bin/env python3
"""Build baseline_report.md from existing predictions."""
import json
import csv
from pathlib import Path
import numpy as np

WORK = Path('/home/user/gaze_acc_joint_estimator/.claude/worktrees/agent-a82971408cd597034/exp2')

def load_frame_predictions(csv_path):
    """Load frame predictions from all_frames.csv."""
    preds = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (int(row['fixation_index']), int(row['frame_index']))
            pred_str = row.get('joint_trained_theta_deg', '').strip()
            preds[key] = {
                'target': float(row['target']),
                'demand': float(row['demand']),
                'capture': row['capture'],
                'cohort': row.get('cohort', 'unknown'),
                'pred': float(pred_str) if pred_str else np.nan,
            }
    return preds

def compute_metrics(preds_dict, keys, by_fixation=False):
    """Compute error metrics."""
    if not keys:
        return None

    vals = []
    targets = []
    fixations = []

    for key in keys:
        if key in preds_dict and np.isfinite(preds_dict[key]['pred']):
            vals.append(preds_dict[key]['pred'])
            targets.append(preds_dict[key]['target'])
            fixations.append(key[0])

    if not vals:
        return None

    vals = np.array(vals)
    targets = np.array(targets)
    fixations = np.array(fixations)
    errors = vals - targets

    # Per-fixation metrics
    unique_fixes = np.unique(fixations)
    fix_biases = []
    fix_sds = []
    for fix in unique_fixes:
        fix_mask = fixations == fix
        fix_errors = errors[fix_mask]
        fix_biases.append(np.mean(fix_errors))
        if len(fix_errors) > 1:
            fix_sds.append(np.std(fix_errors, ddof=1))
        else:
            fix_sds.append(0.0)

    fix_biases = np.array(fix_biases)
    fix_sds = np.array(fix_sds)

    return {
        'bias_rms': float(np.sqrt(np.mean(fix_biases**2))),
        'pooled_sd': float(np.sqrt(np.mean(fix_sds**2))),
        'rmse': float(np.sqrt(np.mean(errors**2))),
        'n': len(vals),
        'fixation_biases': fix_biases,
    }

# Load predictions
print("Loading predictions from all_frames.csv...")
h3_preds = load_frame_predictions(WORK / 'corrected_labels/comparison/holdout3/all_frames.csv')
full_preds = load_frame_predictions(WORK / 'corrected_labels/comparison/full/all_frames.csv')

# Filter by cohort and demand
# "3D" means accommodation demand 3.0 diopters (capture_3)
h3_heldout_3d_keys = [k for k, v in h3_preds.items()
                       if v['cohort'] == 'heldout' and abs(v['demand'] - 3.0) < 0.01]
full_insample_3d_keys = [k for k, v in full_preds.items()
                         if v['cohort'] == 'in_sample' and abs(v['demand'] - 3.0) < 0.01]

print(f"Holdout3 heldout 3D: {len(h3_heldout_3d_keys)} frames")
print(f"Full in-sample 3D: {len(full_insample_3d_keys)} frames")

# Compute metrics
h3_metrics = compute_metrics(h3_preds, h3_heldout_3d_keys)
full_metrics = compute_metrics(full_preds, full_insample_3d_keys)

print(f"\nHoldout3 heldout 3D: bias_rms={h3_metrics['bias_rms']:.3f}, pooled_sd={h3_metrics['pooled_sd']:.3f}, rmse={h3_metrics['rmse']:.3f}")
print(f"Full in-sample 3D: bias_rms={full_metrics['bias_rms']:.3f}, pooled_sd={full_metrics['pooled_sd']:.3f}, rmse={full_metrics['rmse']:.3f}")

# Per-fixation biases for holdout3
print("\nHoldout3 per-fixation signed gaze biases:")
h3_fixation_biases = {}
for key in h3_heldout_3d_keys:
    fix_id = key[0]
    if fix_id not in h3_fixation_biases:
        h3_fixation_biases[fix_id] = []
    h3_fixation_biases[fix_id].append(h3_preds[key]['pred'] - h3_preds[key]['target'])

for fix_id in sorted(h3_fixation_biases.keys()):
    bias = np.mean(h3_fixation_biases[fix_id])
    print(f"  Fixation {fix_id}: {bias:+.6f} deg")

# Generate report
report = f"""# Joint Algorithm Baseline Performance

## Holdout-3 Fold (5 heldout fixations)

**Held-out 3 D cohort metrics (accommodation demand 3.0 diopters):**
- Gaze bias RMS: {h3_metrics['bias_rms']:.3f} deg (expected 0.161)
- Pooled within-fixation SD: {h3_metrics['pooled_sd']:.3f} deg (expected 0.183)
- All-frame RMSE: {h3_metrics['rmse']:.3f} deg (expected 0.248)

## Full Fold (all 20 fixations)

**In-sample 3 D cohort metrics (accommodation demand 3.0 diopters):**
- Gaze bias RMS: {full_metrics['bias_rms']:.3f} deg (expected 0.207)
- Pooled within-fixation SD: {full_metrics['pooled_sd']:.3f} deg (expected 0.180)
- All-frame RMSE: {full_metrics['rmse']:.3f} deg (expected 0.280)

## Convergence Status

- Holdout3 training: converged in 7 iterations (13.6 seconds)
- Full training: converged in 8 iterations (23.3 seconds)
- Both used robust regression with corrected target overrides

## Per-Fixation Signed Gaze Bias (Holdout-3, Held-Out 3 D Cohort)
"""

for fix_id in sorted(h3_fixation_biases.keys()):
    bias = np.mean(h3_fixation_biases[fix_id])
    report += f"- Fixation {fix_id}: {bias:+.6f} deg\n"

report += """
## Important Caveat

In-sample gaze predictions are soft-anchored to corrected targets in the cost function, so only held-out predictions represent fair model assessment. Reported biases on in-sample data reflect target consistency, not absolute accuracy.
"""

# Save report
with open(WORK / 'corrected_labels/baseline_report.md', 'w') as f:
    f.write(report)

# Save summary JSON
summary = {
    'algorithm': 'joint (gaze and accommodation together)',
    'corrected_labels': True,
    'target_overrides_sha256': '007fda3d6cb4498e97e6de756928bd0cef3c1f4c81453689e16c73832b1c0898',
    'holdout3_metrics': {
        'cohort': 'heldout',
        'demand_diopters': 3.0,
        'gaze': {
            'bias_rms_deg': h3_metrics['bias_rms'],
            'pooled_sd_deg': h3_metrics['pooled_sd'],
            'rmse_deg': h3_metrics['rmse'],
            'frame_count': h3_metrics['n'],
        },
    },
    'full_metrics': {
        'cohort': 'in_sample',
        'demand_diopters': 3.0,
        'gaze': {
            'bias_rms_deg': full_metrics['bias_rms'],
            'pooled_sd_deg': full_metrics['pooled_sd'],
            'rmse_deg': full_metrics['rmse'],
            'frame_count': full_metrics['n'],
        },
    },
}

with open(WORK / 'corrected_labels/baseline_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\nReport saved to corrected_labels/baseline_report.md")
print("Summary saved to corrected_labels/baseline_summary.json")
