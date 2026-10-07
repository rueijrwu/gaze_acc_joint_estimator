"""Gaze baseline models (B1, B2, B3) and comparison with joint algorithm."""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP))

from calibrate_profiled import load_data
from observations import relative_measurements
from reduced_calibration.experiment import FOLDS

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load_joint_predictions(model_dir, prediction_dir, fold):
    """Load joint algorithm predictions."""
    joint_data = {}

    # Load trained model predictions
    if prediction_dir is not None:
        trained_csv = prediction_dir / 'trained_predictions.csv'
        if trained_csv.exists():
            with trained_csv.open(newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    key = (int(row['fixation_index']), int(row['frame_index']))
                    joint_data[('trained', key)] = float(row['theta_deg'])

        initial_csv = prediction_dir / 'initial_predictions.csv'
        if initial_csv.exists():
            with initial_csv.open(newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    key = (int(row['fixation_index']), int(row['frame_index']))
                    joint_data[('initial', key)] = float(row['theta_deg'])

    # Load in-sample predictions from model states
    if model_dir is not None and fold == 'full':
        for capture_idx in range(1, 5):
            states_csv = model_dir / f'capture_{capture_idx}_states.csv'
            if states_csv.exists():
                with states_csv.open(newline='') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if row.get('status') == 'calibration_core':
                            key = (int(row['fixation_index']), int(row['frame_index']))
                            joint_data[('trained', key)] = float(row['theta_deg'])

    return joint_data

def build_frame_data(experiment, interval_path, target_overrides, fold):
    """Build per-frame m and S1 arrays with corrected z_lookup keying."""
    y, frames, groups, meta, records, prov = load_data(experiment, interval_path, target_overrides=target_overrides)

    # Build lookup table: (group_idx, absolute_frame_index) -> z_row
    z_lookup = {}
    for record in records:
        z = relative_measurements(record['arrays'])['z']  # columns [m, S1, S4]
        grp = record['groups']
        frame_indices = record['arrays']['frame_index']
        for i in np.flatnonzero(grp >= 0):
            key = (int(grp[i]), int(frame_indices[i]))
            z_lookup[key] = z[i]

    # Build per-frame data using original order (no lexsort)
    heldout_fixations = set(FOLDS[fold])
    frame_data = []
    training_fixations = [j for j in range(len(meta)) if j not in heldout_fixations]

    for n in range(len(groups)):
        g = int(groups[n])
        if g < 0:
            continue

        key = (g, int(frames[n]))
        if key not in z_lookup:
            raise ValueError(f"Missing z_lookup entry for (group={g}, frame_idx={int(frames[n])})")

        z_row = z_lookup[key]
        m = z_row[0]
        S1 = abs(z_row[1])
        S4 = z_row[2]

        # Verify spec assertion: max |m/S1 - y[:,0]| < 1e-9
        if S1 > 0:
            d_from_z = m / S1
            d_from_y = y[n, 0]
            assert np.abs(d_from_z - d_from_y) < 1e-9, f"Frame {n}: |m/S1 - y[:,0]| = {np.abs(d_from_z - d_from_y)}"

        target = meta[g]['target_theta_deg']
        demand = meta[g]['demand_diopters_label']
        is_heldout = g in heldout_fixations

        frame_data.append({
            'fixation_index': g,
            'frame_index': int(frames[n]),
            'local_index': n,
            'm': m,
            'S1': S1,
            'S4': S4,
            'd': m / S1 if S1 > 0 else np.nan,
            'target': target,
            'demand': demand,
            'is_heldout': is_heldout,
            'is_training': g in training_fixations,
            'capture': meta[g]['capture'],
            'capture_idx': int(meta[g]['capture'].split('_')[1])
        })

    return frame_data, meta, training_fixations, heldout_fixations

def fit_baselines(frame_data, training_fixations):
    """Fit B1, B2, B3 baselines using fixation means."""
    # Filter to training frames only
    training_frames = [f for f in frame_data if f['is_training']]

    # Compute fixation-level statistics from training frames
    fixation_stats = {}
    for f in training_frames:
        g = f['fixation_index']
        if g not in fixation_stats:
            fixation_stats[g] = {'m_vals': [], 'd_vals': [], 'target': f['target'], 'demand': f['demand']}
        fixation_stats[g]['m_vals'].append(f['m'])
        fixation_stats[g]['d_vals'].append(f['d'])

    # Compute fixation means
    fixation_means = {}
    for g, stats in fixation_stats.items():
        m_mean = np.mean(stats['m_vals'])
        d_mean = np.mean(stats['d_vals'])
        target = stats['target']
        demand = stats['demand']
        fixation_means[g] = {
            'm': m_mean,
            'd': d_mean,
            'target': target,
            'demand': demand,
        }

    # B1: theta = alpha + beta * m (using fixation means)
    m_mean_vals = np.array([fixation_means[g]['m'] for g in sorted(fixation_means.keys())])
    target_vals = np.array([fixation_means[g]['target'] for g in sorted(fixation_means.keys())])

    b1_coef = np.polyfit(m_mean_vals, target_vals, 1)[::-1]  # [intercept, slope]

    # B2: theta = alpha + beta * d (using fixation means via OLS)
    d_mean_vals = np.array([fixation_means[g]['d'] for g in sorted(fixation_means.keys())])

    b2_coef = np.polyfit(d_mean_vals, target_vals, 1)[::-1]  # [intercept, slope]

    # B3: per-demand OLS on fixation means
    b3_coefs = {}
    demands_in_training = sorted(set(fixation_means[g]['demand'] for g in fixation_means))

    for demand_val in demands_in_training:
        demand_fixations = [g for g in sorted(fixation_means.keys())
                           if fixation_means[g]['demand'] == demand_val]
        if demand_fixations:
            xmean = np.array([fixation_means[g]['d'] for g in demand_fixations])
            targetmean = np.array([fixation_means[g]['target'] for g in demand_fixations])
            coef = np.polyfit(xmean, targetmean, 1)
            b3_coefs[demand_val] = coef[::-1]  # [alpha, beta]

    return b1_coef, b2_coef, b3_coefs, demands_in_training, fixation_means

def predict_b1(frame_data, b1_coef):
    """Predict B1 model."""
    predictions = {}
    for f in frame_data:
        pred = b1_coef[0] + b1_coef[1] * f['m']
        key = (f['fixation_index'], f['frame_index'])
        predictions[key] = pred
    return predictions

def predict_b2(frame_data, b2_coef):
    """Predict B2 model."""
    predictions = {}
    for f in frame_data:
        pred = b2_coef[0] + b2_coef[1] * f['d']
        key = (f['fixation_index'], f['frame_index'])
        predictions[key] = pred
    return predictions

def predict_b3(frame_data, b3_coefs, demands_in_training, fold):
    """Predict B3 model with demand-based interpolation."""
    predictions = {}
    demands_sorted = sorted(demands_in_training)

    for f in frame_data:
        demand = f['demand']

        if demand in b3_coefs:
            coef = b3_coefs[demand]
        else:
            # Interpolate or clamp extrapolate
            idx = np.searchsorted(demands_sorted, demand)
            if idx == 0:
                coef = b3_coefs[demands_sorted[0]]
            elif idx == len(demands_sorted):
                coef = b3_coefs[demands_sorted[-1]]
            else:
                d1, d2 = demands_sorted[idx-1], demands_sorted[idx]
                c1, c2 = b3_coefs[d1], b3_coefs[d2]
                alpha = (demand - d1) / (d2 - d1)
                coef = (1 - alpha) * c1 + alpha * c2

        pred = coef[0] + coef[1] * f['d']
        key = (f['fixation_index'], f['frame_index'])
        predictions[key] = pred

    return predictions

def compute_metrics(predictions, targets, name, cohort_label):
    """Compute error metrics."""
    if not predictions:
        return {
            'method': name,
            'cohort': cohort_label,
            'count': 0,
            'rmse': None,
            'bias': None,
            'mae': None,
            'median_absolute': None,
            'p95_absolute': None
        }

    pred_array = np.array(predictions)
    target_array = np.array(targets)
    errors = pred_array - target_array

    return {
        'method': name,
        'cohort': cohort_label,
        'count': len(errors),
        'rmse': float(np.sqrt(np.mean(errors**2))),
        'bias': float(np.mean(errors)),
        'mae': float(np.mean(np.abs(errors))),
        'median_absolute': float(np.median(np.abs(errors))),
        'p95_absolute': float(np.quantile(np.abs(errors), 0.95))
    }

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fold', choices=['holdout3', 'full'], required=True)
    parser.add_argument('--target-overrides', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--prediction-dir', type=Path, default=None)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--experiment-dir', type=Path, default=EXP)
    parser.add_argument('--intervals', type=Path, default=EXP/'fixations/fixation_intervals.json')

    args = parser.parse_args()

    # Refuse existing output dir
    if args.output_dir.exists():
        parser.error('Output directory already exists')

    args.output_dir.mkdir(parents=True)

    # Load frame data
    frame_data, meta, training_fixations, heldout_fixations = build_frame_data(
        args.experiment_dir, args.intervals, args.target_overrides, args.fold)

    # Fit baselines
    b1_coef, b2_coef, b3_coefs, demands_in_training, fixation_means = fit_baselines(frame_data, training_fixations)

    # Generate predictions
    b1_pred = predict_b1(frame_data, b1_coef)
    b2_pred = predict_b2(frame_data, b2_coef)
    b3_pred = predict_b3(frame_data, b3_coefs, demands_in_training, args.fold)

    # Load joint algorithm predictions
    joint_pred = load_joint_predictions(args.model_dir, args.prediction_dir, args.fold)

    # Filter by cohort
    heldout_frames = [f for f in frame_data if f['is_heldout']]

    # Build comparison output
    comparison_output = []

    for f in frame_data:
        key = (f['fixation_index'], f['frame_index'])
        cohort = 'heldout' if f['is_heldout'] else 'in_sample'

        row = {
            'fixation_index': f['fixation_index'],
            'frame_index': f['frame_index'],
            'target': f['target'],
            'demand': f['demand'],
            'capture': f['capture'],
            'cohort': cohort
        }

        # Add baseline predictions
        row['b1_theta_deg'] = b1_pred.get(key)
        row['b2_theta_deg'] = b2_pred.get(key)
        row['b3_theta_deg'] = b3_pred.get(key)

        # Add joint predictions
        if ('trained', key) in joint_pred:
            row['joint_trained_theta_deg'] = joint_pred[('trained', key)]
        if ('initial', key) in joint_pred:
            row['joint_initial_theta_deg'] = joint_pred[('initial', key)]

        comparison_output.append(row)

    # Write detailed comparison CSV
    if comparison_output:
        fieldnames = ['fixation_index', 'frame_index', 'target', 'demand', 'capture', 'cohort',
                     'b1_theta_deg', 'b2_theta_deg', 'b3_theta_deg', 'joint_trained_theta_deg', 'joint_initial_theta_deg']
        comparison_csv = args.output_dir / 'all_frames.csv'
        with comparison_csv.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(comparison_output)

    # Write per-fixation CSV (fixation means)
    per_fixation_rows = []
    for g in sorted(fixation_means.keys()):
        fm = fixation_means[g]
        row = {
            'fixation_index': g,
            'd': fm['d'],
            'm': fm['m'],
            'target': fm['target'],
            'demand': fm['demand'],
            'is_training': g in training_fixations
        }
        per_fixation_rows.append(row)

    if per_fixation_rows:
        per_fixation_csv = args.output_dir / 'per_fixation.csv'
        with per_fixation_csv.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['fixation_index', 'd', 'm', 'target', 'demand', 'is_training'])
            writer.writeheader()
            writer.writerows(per_fixation_rows)

    # Write per-capture CSV (for full fold only)
    if args.fold == 'full':
        per_capture_rows = []
        captures = sorted(set(f['capture'] for f in frame_data if f['is_heldout']))

        for capture in captures:
            cap_frames = [f for f in heldout_frames if f['capture'] == capture]
            if not cap_frames:
                continue

            # Compute metrics per method per capture
            for method_name, pred_dict in [('B1', b1_pred), ('B2', b2_pred), ('B3', b3_pred), ('joint_trained', joint_pred)]:
                preds = []
                targets = []
                for f in cap_frames:
                    key = (f['fixation_index'], f['frame_index'])
                    if method_name == 'joint_trained':
                        if ('trained', key) in pred_dict:
                            preds.append(pred_dict[('trained', key)])
                            targets.append(f['target'])
                    else:
                        if pred_dict.get(key) is not None:
                            preds.append(pred_dict[key])
                            targets.append(f['target'])

                if preds:
                    errors = np.array(preds) - np.array(targets)
                    row = {
                        'capture': capture,
                        'method': method_name,
                        'count': len(preds),
                        'rmse': float(np.sqrt(np.mean(errors**2))),
                        'bias': float(np.mean(errors)),
                    }
                    per_capture_rows.append(row)

        if per_capture_rows:
            per_capture_csv = args.output_dir / 'per_capture.csv'
            with per_capture_csv.open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['capture', 'method', 'count', 'rmse', 'bias'])
                writer.writeheader()
                writer.writerows(per_capture_rows)

    # Write accommodation CSV (per-demand metrics for full fold)
    if args.fold == 'full':
        accom_rows = []
        for demand in demands_in_training:
            demand_frames = [f for f in heldout_frames if f['demand'] == demand]
            if not demand_frames:
                continue

            for method_name, pred_dict in [('B1', b1_pred), ('B2', b2_pred), ('B3', b3_pred), ('joint_trained', joint_pred)]:
                preds = []
                targets = []
                for f in demand_frames:
                    key = (f['fixation_index'], f['frame_index'])
                    if method_name == 'joint_trained':
                        if ('trained', key) in pred_dict:
                            preds.append(pred_dict[('trained', key)])
                            targets.append(f['target'])
                    else:
                        if pred_dict.get(key) is not None:
                            preds.append(pred_dict[key])
                            targets.append(f['target'])

                if preds:
                    errors = np.array(preds) - np.array(targets)
                    row = {
                        'demand_diopters': demand,
                        'method': method_name,
                        'count': len(preds),
                        'rmse': float(np.sqrt(np.mean(errors**2))),
                        'bias': float(np.mean(errors)),
                    }
                    accom_rows.append(row)

        if accom_rows:
            accom_csv = args.output_dir / 'accommodation.csv'
            with accom_csv.open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['demand_diopters', 'method', 'count', 'rmse', 'bias'])
                writer.writeheader()
                writer.writerows(accom_rows)

    # Write JSON summary
    summary_json = {
        'fold': args.fold,
        'target_overrides_sha256': sha(args.target_overrides),
        'interval_sha256': sha(args.intervals),
        'model_sha256': sha(args.model_dir / 'model.json'),
        'baseline_coefficients': {
            'b1': {'intercept': float(b1_coef[0]), 'slope': float(b1_coef[1])},
            'b2': {'intercept': float(b2_coef[0]), 'slope': float(b2_coef[1])},
            'b3_demands': list(demands_in_training),
            'b3_coefficients': {str(d): {'intercept': float(c[0]), 'slope': float(c[1])}
                               for d, c in b3_coefs.items()}
        },
        'frame_count_total': len(frame_data),
        'frame_count_heldout': len(heldout_frames),
        'frame_count_training': len([f for f in frame_data if f['is_training']])
    }

    with (args.output_dir / 'summary.json').open('w') as f:
        json.dump(summary_json, f, indent=2)

    # Write report.md
    report_md = f"""# Baseline Comparison Report

## Fold: {args.fold}

### B1 Model (WLS on fixation means)
- Intercept: {b1_coef[0]:.6f} deg
- Slope: {b1_coef[1]:.6f} deg/px

### B2 Model (WLS on fixation means)
- Intercept: {b2_coef[0]:.6f} deg
- Slope: {b2_coef[1]:.6f} deg per unit d

### B3 Model (per-demand OLS)
Demands in training: {demands_in_training}
"""

    for demand, coef in b3_coefs.items():
        report_md += f"- Demand {demand}: intercept {coef[0]:.6f}, slope {coef[1]:.6f}\n"

    with (args.output_dir / 'report.md').open('w') as f:
        f.write(report_md)

    print(f"Baseline comparison complete. Output in {args.output_dir}")
    print(f"B1 coef: intercept={b1_coef[0]:.6f}, slope={b1_coef[1]:.6f}")
    print(f"B2 coef: intercept={b2_coef[0]:.6f}, slope={b2_coef[1]:.6f}")

if __name__ == '__main__':
    main()
