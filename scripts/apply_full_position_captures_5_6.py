#!/usr/bin/env python3
"""Apply a saved joint model to captures 5/6 and compare capture-1 linear gaze.

Run: python scripts/apply_full_position_captures_5_6.py
The default is the frozen quadratic M2 full-calibration fit on captures 1–4.
No model is refitted. All original rows are exported; invalid inputs remain NaN.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from compare_gaze_corrections import read_capture1_fixations
from full_position.accelerated import solve_batch
from full_position.data import load_capture
from full_position.invert import polish, invert
from full_position.model import PositionModel, STATE_SCALE, LOWER, UPPER
from full_position.noise import reference_covariance, whitening
from full_position.schema import load_model

DEFAULT_MODEL = ROOT/'models/quadratic_model.json'


def estimate(cap, model, meta, backend, batch_size):
    """Batched 49-start seeds, exact scalar certification of the cheapest seed.

    This is an accelerated application, rather than exhaustive scalar branch
    enumeration. Failed certification uses the scalar multistart reference.
    Two valid P4 points suffice; their exact covariance marginal is used.
    """
    n = len(cap.frame)
    states = np.full((n, 2), np.nan)
    reasons = np.full(n, 'invalid_input', dtype=object)
    costs = np.full(n, np.nan)
    bound = np.zeros(n, bool)
    pilot = PositionModel(int(meta['pilot_model'].removeprefix('conditional')), meta['pilot_coefficients'])
    eligible = cap.ctx.valid & (cap.point_valid.sum(1) >= 2)
    patterns = np.unique(cap.point_valid[eligible], axis=0)
    fallback_count = 0
    for pattern in patterns:
        rows = np.flatnonzero(eligible & np.all(cap.point_valid == pattern, axis=1))
        ix = np.flatnonzero(np.repeat(pattern, 2))
        for start in range(0, len(rows), batch_size):
            rr = rows[start:start+batch_size]
            cov = reference_covariance(cap.p[rr], pilot, np.asarray(meta['reference_state']), np.asarray(meta['coordinate_covariance']))
            cov = cov[:, ix][:, :, ix]
            y = cap.v[rr].reshape(-1, 6)[:, ix]
            seeds = solve_batch(model, cap.ctx.r[rr], y, cov, ix, backend=backend)
            W = whitening(cov)
            best = np.argmin(seeds['costs'], axis=1)
            for j, row in enumerate(rr):
                z, cert = polish(model, cap.ctx.r[row], y[j], W[j], ix,
                                 seeds['states'][j, best[j]]/STATE_SCALE)
                if cert['certified']:
                    x = z*STATE_SCALE
                    _, jac = model.predict(x, cap.ctx.r[row], True)
                    sv = np.linalg.svd(W[j]@jac[ix], compute_uv=False)
                    reasons[row] = 'ok' if sv[-1] > max(sv[0]*1e-6, 1e-8) else 'weak_rank'
                else:
                    fallback_count += 1
                    result = invert(model, cap.ctx.r[row], y[j], cov[j], ix)
                    reasons[row] = result['reason']
                    if not result['available']:
                        continue
                    x = np.asarray(result['state'])
                states[row] = x
                residual = W[j]@(model.predict(x, cap.ctx.r[row])[ix]-y[j])
                costs[row] = residual@residual
                bound[row] = np.any((x-LOWER < 1e-5) | (UPPER-x < 1e-5))
            print(f'{cap.name}: {start+len(rr)}/{len(rows)} rows for P4 mask {pattern.tolist()}', flush=True)
    return states, reasons, costs, bound, fallback_count


def estimate_quadratic(cap, meta, batch_size):
    """Apply frozen M2 coefficients to fresh observations, without calibration."""
    import pickle
    from joint_m2 import measurements, _invert_batch_m2_fast, OBSERVABLE_SCHEMA
    if meta.get('observable_schema') != OBSERVABLE_SCHEMA or not meta.get('diagnostics', {}).get('converged'):
        raise ValueError('Incompatible or nonconverged M2 model')
    with (ROOT/'data/detections'/cap.name).open('rb') as stream:
        arrays = pickle.load(stream)['arrays']
    y, valid, _ = measurements(arrays)
    states = np.full((len(cap.frame), 2), np.nan)
    costs = np.full(len(cap.frame), np.nan)
    reasons = np.full(len(cap.frame), 'invalid_input', dtype=object)
    bound = np.zeros(len(cap.frame), bool)
    rows = np.flatnonzero(valid)
    for start in range(0, len(rows), batch_size):
        rr = rows[start:start+batch_size]
        x, diag = _invert_batch_m2_fast(y[rr], np.asarray(meta['coefficients']), np.asarray(meta['precision']), 100)
        states[rr] = x
        costs[rr] = diag['weighted_cost']
        reasons[rr] = np.where(diag['segment_projected_stationarity'] <= 1e-3, 'ok', 'stationarity_unverified')
        bound[rr] = np.any((x-np.asarray([meta['theta_bounds'][0], meta['A_bounds'][0]]) < 1e-5) |
                           (np.asarray([meta['theta_bounds'][1], meta['A_bounds'][1]])-x < 1e-5), axis=1)
        print(f'{cap.name}: {start+len(rr)}/{len(rows)} M2 rows', flush=True)
    return states, reasons, costs, bound, 0


def remove_plot_spikes(series, out):
    """Mask brief deviations from a centered 0.3-second local median."""
    from scipy.ndimage import median_filter
    filtered, counts = {}, {}
    for c, original in series.items():
        s = {key: np.asarray(value).copy() for key, value in original.items()}
        time = s['time']
        dt = np.median(np.diff(time)[np.diff(time) > 0])
        window = max(3, int(round(.3/dt)) | 1)
        bad = np.zeros(len(time), dtype=bool)
        for values, threshold in ((s['states'][:, 0], 1.0),
                                  (s['states'][:, 1], .5),
                                  (s['linear'], 1.0),
                                  (s['difference'], 30.0)):
            valid = np.isfinite(values)
            if not valid.any():
                continue
            filled = np.interp(time, time[valid], values[valid])
            local = median_filter(filled, size=window, mode='nearest')
            bad |= valid & (np.abs(values-local) > threshold)
        s['states'][bad] = np.nan
        s['linear'][bad] = np.nan
        s['difference'][bad] = np.nan
        filtered[c] = s
        counts[c] = int(bad.sum())
        with (out/f'capture_{c}_plot_mask.csv').open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['row', 'elapsed_s', 'excluded_spike'])
            writer.writerows(zip(range(len(time)), time, bad))
    return filtered, dict(method='centered 0.3-second local median; common mask across panels',
                         thresholds=dict(gaze_deg=1.0, accommodation_D=.5, difference_arcmin=30.0),
                         excluded_rows=counts, removed_samples='NaN gaps; no interpolation of displayed values')


def make_figures(series, out):
    series, filtering = remove_plot_spikes(series, out)
    plt.rcParams.update({'font.size': 18, 'axes.titlesize': 20, 'axes.labelsize': 18,
                         'xtick.labelsize': 18, 'ytick.labelsize': 18})
    trials = {5: 1, 6: 2}
    fig, axes = plt.subplots(2, 2, figsize=(20, 9), constrained_layout=True, sharex='col')
    for col, c in enumerate((5, 6)):
        s = series[c]
        axes[0, col].plot(s['time'], s['states'][:, 0], '-', lw=1.0)
        axes[1, col].plot(s['time'], s['states'][:, 1], '-', lw=1.0, color='tab:orange')
        axes[0, col].set_title(f"Trial {trials[c]} - Gaze")
        axes[0, col].set_ylim(-12.5, 7.5)
        axes[1, col].set_title(f"Trial {trials[c]} - Accommodation")
        axes[1, col].set_ylim(0, 4.5)
        axes[0, col].set_ylabel('Horizontal gaze (deg)')
        axes[1, col].set_ylabel('Accommodation (D)')
    for ax in axes.flat:
        ax.set_xlabel('Time (s)'); ax.grid(alpha=.25)
    fig.savefig(out/'joint_gaze_accommodation.png', dpi=180)
    fig.savefig(out/'joint_gaze_accommodation.svg', format='svg')
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(20, 9), constrained_layout=True, sharex='row')
    for row, c in enumerate((5, 6)):
        s = series[c]
        axes[row, 0].plot(s['time'], s['linear'], '-', lw=1.0, color='tab:green')
        axes[row, 1].plot(s['time'], s['difference'], '-', lw=1.0, color='tab:purple')
        axes[row, 0].set_title(f"Trial {trials[c]} - Gaze")
        axes[row, 1].set_title(f"Trial {trials[c]} - Error")
        axes[row, 0].set_ylabel('Horizontal gaze (deg)')
        axes[row, 0].set_ylim(-12.5, 7.5)
        axes[row, 1].set_ylabel('Gaze difference (arcmin)')
        axes[row, 1].axhline(0, color='black', lw=.8)
        axes[row, 1].set_ylim(-50, 50)
    for ax in axes.flat:
        ax.set_xlabel('Time (s)'); ax.grid(alpha=.25)
    fig.savefig(out/'linear_gaze_comparison.png', dpi=180)
    fig.savefig(out/'linear_gaze_comparison.svg', format='svg')
    plt.close(fig)
    (out/'plot_filter.json').write_text(json.dumps(filtering, indent=2)+'\n')
    return filtering


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--output', type=Path, default=ROOT/'experiments/full_position/captures_5_6_quadratic_application')
    parser.add_argument('--backend', choices=['numpy', 'cupy'], default='cupy')
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--plots-only', action='store_true', help='Regenerate figures from this output directory\'s CSVs')
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    series = {}
    if args.plots_only:
        for c in (5, 6):
            a = np.genfromtxt(args.output/f'capture_{c}_estimates.csv', delimiter=',', names=True)
            series[c] = dict(time=a['elapsed_s'], states=np.column_stack((a['joint_gaze_deg'], a['accommodation_D'])),
                             linear=a['linear_gaze_deg'], difference=a['linear_minus_joint_arcmin'])
        filtering = make_figures(series, args.output)
        summary_path = args.output/'summary.json'
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            summary['plotting'] = filtering
            summary_path.write_text(json.dumps(summary, indent=2)+'\n')
        print(json.dumps(filtering, indent=2))
        return
    meta = json.loads(args.model.read_text())
    quadratic = meta.get('model_type') == 'm2'
    if quadratic:
        model = None
    else:
        model, meta = load_model(args.model)
        if model.name not in ('conditional27', 'conditional37'):
            parser.error('Choose a frozen quadratic M2 or conditional27/37 model')
    calibration, _ = read_capture1_fixations()
    displacement = np.array([r['raw_displacement_mean_px'] for r in calibration])
    targets = np.array([r['target_theta_deg'] for r in calibration])
    slope, intercept = np.polyfit(displacement, targets, 1)
    with (args.output/'capture_1_linear_calibration.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(calibration[0]))
        writer.writeheader(); writer.writerows(calibration)
    report = dict(model=str(args.model.resolve()), model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
                  solver=('frozen M2 bounded multistart Gauss-Newton, 100 iterations' if quadratic else
                          '49 batched starts; cheapest seed exact-polished; scalar fallback on certificate failure'),
                  backend='numpy' if quadratic else args.backend, model_family=meta.get('model_type', meta.get('model')), slope_deg_per_px=float(slope), intercept_deg=float(intercept),
                  linear_formula='gaze_deg = slope * (mean(P4_x) - mean(P1_x)) + intercept',
                  calibration='capture 1 full fixation intervals; equal-weight fixation means; no accommodation term',
                  difference='60 * (linear gaze deg - joint gaze deg); model difference, not ground-truth error',
                  plotting='all available estimates; no smoothing or outlier exclusion', captures={})
    for c in (5, 6):
        cap = load_capture(ROOT/f'data/detections/capture_{c}_detections.pkl')
        if not quadratic and cap.metadata['pair_index'] != next(iter(meta['provenance']['correspondence'].values())):
            raise ValueError('P1/P4 correspondence differs from the fitted model')
        states, reasons, costs, bound, fallback = (estimate_quadratic(cap, meta, min(args.batch_size, 512)) if quadratic else
                                                 estimate(cap, model, meta, args.backend, args.batch_size))
        dx = cap.q[:, :, 0].mean(1)-cap.p[:, :, 0].mean(1)
        linear = np.where(cap.baseline_valid, slope*dx+intercept, np.nan)
        time = (cap.timestamp-cap.timestamp[0])/1000
        difference = 60*(linear-states[:, 0])
        series[c] = dict(time=time, states=states, linear=linear, difference=difference)
        with (args.output/f'capture_{c}_estimates.csv').open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['row','frame','timestamp_ms','elapsed_s','joint_gaze_deg','accommodation_D',
                             'linear_gaze_deg','linear_minus_joint_arcmin','reason','at_bound','weighted_cost'])
            for i in range(len(time)):
                writer.writerow([i, cap.frame[i], cap.timestamp[i], time[i], *states[i], linear[i], difference[i],
                                 reasons[i], bound[i], costs[i]])
        finite = np.isfinite(difference)
        report['captures'][c] = dict(rows=len(time), joint_estimates=int(np.isfinite(states[:, 0]).sum()),
                                    linear_estimates=int(np.isfinite(linear).sum()), compared=int(finite.sum()),
                                    bound_estimates=int(bound.sum()), scalar_fallbacks=fallback, source_sha256=cap.sha256,
                                    stationarity_unverified=int(np.sum(reasons == 'stationarity_unverified')),
                                    accommodation_quantiles_D=np.nanquantile(states[:, 1], [.05,.5,.95]).tolist(),
                                    difference_rms_arcmin=float(np.sqrt(np.mean(difference[finite]**2))))
    report['plotting'] = make_figures(series, args.output)
    (args.output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
