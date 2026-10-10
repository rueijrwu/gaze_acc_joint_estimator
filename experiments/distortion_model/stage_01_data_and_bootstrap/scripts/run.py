#!/usr/bin/env python3
"""Reproduce S0/G0 and S1/G1; no full optical calibration is performed."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "docs/Theory.md").is_file()
    and (parent / "data/fixations/fixation_intervals.json").is_file()
)
sys.path.insert(0, str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from distortion_model.data import load_reviewed, make_population, make_slot_manifest, validate_slots, array_hash
from distortion_model.geometry import relative_coordinates, relative_covariance
from distortion_model.gaze import centroid_displacement, fit_bootstrap, apply_bootstrap
from distortion_model.optics import Parameters, predict_relative, p1_reference

CONFIG = {
    'stage': 'S0-S1/G0-G1', 'candidate': 'bootstrap_only', 'captures': [1, 2, 3, 4],
    'interval_policy': '[start_row,end_row_exclusive), full reviewed interval, no trimming',
    'validity_policy': 'finite and present P1 and all three mapped P4; no pupil, shape or residual gate',
    'pairing_policy': 'read stored pair_index; require reviewed [2,1,0]; never re-sort',
    'camera_units': 'native pixels; x right, y down', 'sign': 'P4-P1',
    'bootstrap_degree': 1, 'bootstrap_weighting': 'equal five reference fixation means',
    'provisional_reference_scale': 1.0, 'blocks_per_interval': 5, 'boundary_window_rows': 100,
    'theta_search_bounds_deg': [-20., 20.], 'accommodation_search_bounds_D': [0., 6.],
    'out_of_domain_policy': 'record; never clip bootstrap states',
    'noise_policy': 'full native sample covariance of adjacent differences / sqrt(2); '
                    'only timestamp-reliable reviewed intervals, complete adjacent rows and source frames; '
                    'positive finite dt; no interval/gap bridging; real eye motion may contribute',
    'covariance_shrinkage': 0.05, 'covariance_diagonal_floor_px2': 1e-8,
    'numerical_tolerances': {'rtol': 1e-10, 'atol': 1e-9},
    'forward_model': 'DM0-M1 synthetic fixture only; independent fixed optical zeros; visual degree 2',
    'threads': 4, 'array_backend': 'numpy float64', 'optimizer': 'five-by-two QR/SVD least squares',
    'gpu': 'not required for stage-one linear bootstrap', 'plot_stride': 10,
}


def write_json(path, content):
    path.write_text(json.dumps(content, indent=2, sort_keys=True, allow_nan=False) + '\n')


def command_output(*args):
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def finite_summary(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if not len(values):
        return {'count': 0, 'mean': None, 'std': None, 'min': None, 'max': None, 'p05': None, 'p95': None}
    return {'count': len(values), 'mean': float(values.mean()), 'std': float(values.std()),
            'min': float(values.min()), 'max': float(values.max()),
            'p05': float(np.percentile(values, 5)), 'p95': float(np.percentile(values, 95))}


def census(captures, intervals, population):
    exposures, source_records = [], []
    for name, capture in captures.items():
        arrays = capture.arrays
        dt = np.diff(arrays['timestamp_ms'])
        frames = arrays['frame_index']
        finite_valid = np.isfinite(capture.p1).all(axis=(1, 2)) & np.isfinite(capture.p4).all(axis=(1, 2))
        historical_valid = finite_valid & arrays['p4_found'].all(axis=1)
        valid = capture.p1_available.all(axis=1) & capture.p4_available.all(axis=1)
        flags_disagree = ((np.isfinite(capture.p1).all(axis=-1) != arrays['p1_valid']).any(axis=1)
                          | (np.isfinite(arrays['p4_xy']).all(axis=-1) != arrays['p4_found']).any(axis=1))
        source_records.append({'capture': name, 'sha256': capture.digest, 'rows': len(frames),
                               'source_frame_gap_count': int((np.diff(frames) != 1).sum()),
                               'backward_timestamp_jumps': int((dt < 0).sum()),
                               'nonpositive_timestamp_differences': int((dt <= 0).sum()),
                               'nonfinite_timestamps': int((~np.isfinite(arrays['timestamp_ms'])).sum()),
                               'complete_valid': int(valid.sum()), 'review_policy_valid': int(historical_valid.sum()),
                               'finite_flag_disagreements': int(flags_disagree.sum()),
                               'p1_available_by_point': capture.p1_available.sum(axis=0).tolist(),
                               'p4_available_by_corresponding_point': capture.p4_available.sum(axis=0).tolist(),
                               'pair_index': capture.meta['pair_index'], 'units': capture.meta['array_layout']['coordinates'],
                               'status_counts': {capture.meta['status_names'].get(int(s), str(s)): int(n)
                                                 for s, n in zip(*np.unique(arrays['status'], return_counts=True))}})
    displacement = centroid_displacement(population['p1'], population['p4'])
    for exposure, interval in enumerate(intervals):
        select = population['exposure'] == exposure
        valid = population['complete_valid'][select]
        rows = population['row'][select]
        d = displacement[select]
        capture = captures[interval['capture']]
        timestamps = population['timestamp_ms'][select]
        edge_summaries = []
        for label, edge in [('first', rows < rows[0] + 100), ('last', rows >= rows[-1] - 99)]:
            edge_summaries.append({'edge': label, 'rows': int(edge.sum()), 'valid': int((edge & valid).sum()),
                                   'displacement_x_px': finite_summary(d[edge & valid, 0])})
        blocks = []
        for block_id, block in enumerate(np.array_split(np.arange(len(rows)), CONFIG['blocks_per_interval'])):
            ok = block[valid[block]]
            blocks.append({'block': block_id, 'start_row': int(rows[block[0]]),
                           'end_row_exclusive': int(rows[block[-1]] + 1), 'scheduled': len(block),
                           'valid': len(ok), 'displacement_x_px': finite_summary(d[ok, 0]),
                           'displacement_y_px': finite_summary(d[ok, 1])})
        actual_backjumps = int((np.diff(timestamps) < 0).sum())
        actual_valid = int(valid.sum())
        exposures.append({'exposure': exposure, **interval, 'actual_scheduled': int(select.sum()),
                          'actual_valid': actual_valid, 'actual_invalid': int((~valid).sum()),
                          'review_valid_count_agrees': actual_valid == interval['full_valid_count'],
                          'actual_backward_timestamp_jumps': actual_backjumps,
                          'review_timestamp_count_agrees': actual_backjumps == interval['timestamp_backward_jumps_in_interval'],
                          'source_frame_gap_count': int((np.diff(population['source_frame'][select]) != 1).sum()),
                          'displacement_x_px': finite_summary(d[valid, 0]),
                          'displacement_y_px': finite_summary(d[valid, 1]),
                          'p1_available_by_point': population['p1_available'][select].sum(axis=0).tolist(),
                          'p4_available_by_corresponding_point': population['p4_available'][select].sum(axis=0).tolist(),
                          'boundary_summaries': edge_summaries, 'blocks': blocks,
                          'first_row_valid': bool(valid[0]), 'last_row_valid': bool(valid[-1])})
    return {'sources': source_records, 'exposures': exposures,
            'scheduled': len(population['row']), 'valid': int(population['complete_valid'].sum()),
            'invalid': int((~population['complete_valid']).sum()), 'expected_slots': 3 * len(population['row'])}


def estimate_covariance(population, intervals):
    z = np.concatenate((population['p1'].reshape(-1, 6), population['p4'].reshape(-1, 6)), axis=1)
    differences, counts = [], []
    for exposure, interval in enumerate(intervals):
        indices = np.flatnonzero(population['exposure'] == exposure)
        left, right = indices[:-1], indices[1:]
        dt = population['timestamp_ms'][right] - population['timestamp_ms'][left]
        reliable = bool(interval['timestamp_duration_reliable']) and np.isfinite(dt).all() and np.all(dt > 0)
        eligible = (population['complete_valid'][left] & population['complete_valid'][right]
                    & (population['row'][right] - population['row'][left] == 1)
                    & (population['source_frame'][right] - population['source_frame'][left] == 1)
                    & np.isfinite(dt) & (dt > 0) & reliable)
        differences.append((z[right[eligible]] - z[left[eligible]]) / np.sqrt(2))
        counts.append({'exposure': exposure, 'adjacent_pairs': len(left), 'used_pairs': int(eligible.sum()),
                       'excluded_unreliable_timestamps': not reliable})
    differences = np.concatenate(differences)
    if len(differences) < 13:
        raise ValueError('insufficient contiguous native differences for covariance')
    empirical = np.cov(differences, rowvar=False, ddof=1)
    diagonal = np.maximum(np.diag(empirical), CONFIG['covariance_diagonal_floor_px2'])
    shrink = CONFIG['covariance_shrinkage']
    native = (1-shrink)*empirical + shrink*np.diag(diagonal)
    relative = relative_covariance(native)
    np.linalg.cholesky(native)
    np.linalg.cholesky(relative)
    return {'native': native, 'relative': relative, 'R11': relative[:4, :4], 'native_unshrunk': empirical}, {
        'policy': CONFIG['noise_policy'], 'shrinkage': shrink, 'pairs': len(differences), 'by_exposure': counts,
        'native_eigenvalues': np.linalg.eigvalsh(native).tolist(),
        'relative_eigenvalues': np.linalg.eigvalsh(relative).tolist(),
        'interpretation': 'frozen weighting approximation; contains eye motion and detector error; '
                          'not independently measured hardware noise or plug-in residual covariance'}


def synthetic_forward(output, covariance):
    fixture = {'b1': [[-10., -3.], [0., 6.], [11., -2.]],
               'b4': [[-7., -2.], [0.5, 4.], [7., -1.]],
               'omega1': -3., 'omega4': 4., 'k1': [0.0002, -0.0001, 0.0002],
               'k4': [0.0006, -0.0002, -0.0003], 'aref': 0.36036036036036034, 'm1': 0.06,
               'center': [[2., 1.], [0.2, -0.1], [40., 0.3], [1., 0.1], [0.4, -0.2], [0., 0.]]}
    params = Parameters(**fixture)
    theta, accommodation = np.meshgrid(np.linspace(-20, 20, 41), np.linspace(0, 6, 13), indexing='ij')
    theta, accommodation = theta.ravel(), accommodation.ravel()
    f1, _, edges, _ = p1_reference(theta, params)
    g_true = np.linspace(0.7, 1.3, len(theta))
    prediction, g, valid = predict_relative(theta, accommodation, g_true[:, None]*edges, covariance['R11'], params)
    np.savez_compressed(output / 'synthetic_forward.npz', theta_visual_deg=theta, accommodation_D=accommodation,
                        predicted_relative=prediction, scale=g, validity=valid, p1_reference=f1)
    write_json(output / 'synthetic_fixture.json', fixture)
    return {'kind': 'synthetic numerical, not empirical optical calibration', 'proposals': len(theta),
            'valid': int(valid.sum()), 'max_scale_recovery_error': float(np.max(np.abs(g-g_true))),
            'local_angle_ranges_deg': {'P1': [float(theta.min()-params.omega1), float(theta.max()-params.omega1)],
                                       'P4': [float(theta.min()-params.omega4), float(theta.max()-params.omega4)]}}


def plots(output, captures, intervals, population, bootstrap, states, census_record):
    fig, axes = plt.subplots(4, 2, figsize=(14, 12), constrained_layout=True)
    for row, (name, capture) in enumerate(captures.items()):
        display = np.arange(0, len(capture.p1), CONFIG['plot_stride'])
        valid = capture.p1_available.all(axis=1) & capture.p4_available.all(axis=1)
        for axis in (0, 1):
            ax = axes[row, axis]
            for reflection, points, linestyle in [('P1', capture.p1, '-'), ('P4 mapped', capture.p4, '--')]:
                for point in range(3):
                    ax.plot(display, points[display, point, axis], linestyle, linewidth=.6,
                            label=f'{reflection} {point}', alpha=.8)
            for interval in intervals:
                if interval['capture'] == name:
                    ax.axvline(interval['start_row'], color='black', alpha=.4, linewidth=.6)
                    ax.axvline(interval['end_row_exclusive'], color='black', alpha=.4, linewidth=.6)
                    ax.axvspan(interval['start_row'], interval['end_row_exclusive'], alpha=.05, color='blue')
            invalid_rows = np.flatnonzero(~valid)
            ax.plot(invalid_rows, np.full(len(invalid_rows), .02), '|', color='red', markersize=3,
                    transform=ax.get_xaxis_transform(), label='unavailable pair')
            ax.set(title=f'{name}: {"x" if axis == 0 else "y"}', xlabel='Original row', ylabel='Native pixels')
    axes[0, 0].legend(fontsize=7, ncols=3)
    fig.savefig(output / 'native_tracks.png', dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for name in captures:
        records = [r for r in census_record['exposures'] if r['capture'] == name]
        means = [r['displacement_x_px']['mean'] for r in records]
        targets = [r['target_theta_deg'] for r in records]
        axes[0, 0].plot(targets, means, 'o-', label=name)
        for record in records:
            blocks = record['blocks']
            axes[0, 1].scatter([record['target_theta_deg']]*len(blocks),
                               [b['displacement_x_px']['mean'] for b in blocks], s=15, alpha=.5)
    axes[0, 0].set(xlabel='Nominal visual target (degrees)', ylabel='Mean P4−P1 x (px)', title='Raw capture transfer')
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].set(xlabel='Nominal visual target (degrees)', ylabel='Block mean displacement (px)',
                   title='Five contiguous row blocks per exposure')
    reference = [r for r in census_record['exposures'] if r['capture'] == bootstrap['reference_capture']]
    means = np.array([r['displacement_x_px']['mean'] for r in reference])
    targets = np.array([r['target_theta_deg'] for r in reference])
    domain = np.linspace(means.min(), means.max(), 100)
    axes[1, 0].scatter(means, targets, label='Reference means')
    axes[1, 0].plot(domain, apply_bootstrap(domain, bootstrap), label='Linear inverse')
    axes[1, 0].set(xlabel='P4−P1 x (px), provisional g=1', ylabel='Visual gaze start (degrees)',
                   title='Five means calibrate this initializer')
    axes[1, 0].legend()
    for exposure, record in enumerate(census_record['exposures']):
        keep = np.flatnonzero(population['exposure'] == exposure)[::CONFIG['plot_stride']]
        axes[1, 1].plot(np.arange(len(keep))*CONFIG['plot_stride'], states[keep], linewidth=.6,
                        label=f"c{record['capture'][8]} {record['target_theta_deg']:g}°")
    axes[1, 1].set(xlabel='Row offset within full interval', ylabel='Framewise gaze start (degrees)',
                   title='Dynamic starts; no target clamping')
    fig.savefig(output / 'bootstrap.png', dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(5, 4, figsize=(16, 14), constrained_layout=True)
    for record in census_record['exposures']:
        i = record['exposure']
        ax = axes.flat[i]
        select = np.flatnonzero(population['exposure'] == i)
        display = select[::CONFIG['plot_stride']]
        displacement = centroid_displacement(population['p1'][display], population['p4'][display])[:, 0]
        ax.plot(population['row'][display], displacement, linewidth=.6)
        start, end = record['start_row'], record['end_row_exclusive']
        ax.axvspan(start, start+100, color='orange', alpha=.2)
        ax.axvspan(end-100, end, color='orange', alpha=.2)
        bad = select[~population['complete_valid'][select]]
        ax.plot(population['row'][bad], np.full(len(bad), .02), '|', color='red', markersize=3,
                transform=ax.get_xaxis_transform())
        ax.set(title=f"{record['capture']} / {record['target_theta_deg']:g}°", xlabel='Original row', ylabel='P4−P1 x (px)')
    fig.savefig(output / 'full_interval_boundaries.png', dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New attempt directory; never overwritten')
    parser.add_argument('--test-report', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit('Output already exists; choose a new attempt directory')
    test_report = json.loads(args.test_report.read_text())
    if test_report['failed'] or test_report['errors'] or not test_report['passed'] or test_report['skipped']:
        raise SystemExit('A complete passing stage-one test report is required')
    started = time.perf_counter()
    output.mkdir(parents=True)
    write_json(output / 'config.json', CONFIG)
    source_files = sorted(set(list((ROOT/'distortion_model').glob('*.py')) + list((ROOT/'tests').glob('*.py'))
                              + [Path(__file__).resolve(), ROOT/'requirements-stage1.txt']
                              + list((ROOT/'docs').glob('*.md')) + list((ROOT/'docs/stages').glob('*.md'))))
    hashes = {str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for p in source_files}
    current_test_hashes = {k:v for k,v in hashes.items() if k.startswith(('distortion_model/', 'tests/'))}
    if test_report['source_hashes'] != current_test_hashes:
        raise SystemExit('Test report source hashes do not match this implementation')
    dirty_hash = sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    provenance = {'created_utc': datetime.now(timezone.utc).isoformat(), 'source_commit': command_output('git', 'rev-parse', 'HEAD'),
                  'branch': command_output('git', 'branch', '--show-current'), 'dirty_source_hash': dirty_hash,
                  'source_hashes': hashes, 'git_status': command_output('git', 'status', '--porcelain'),
                  'config_hash': sha256((output/'config.json').read_bytes()).hexdigest(),
                  'python': sys.version, 'platform': platform.platform(), 'numpy': np.__version__,
                  'matplotlib': matplotlib.__version__, 'cpu_count': os.cpu_count(),
                  'runtime_threads': {k: os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')},
                  'remote_verification': 'unverified: git ls-remote failed to resolve github.com; local implementation only',
                  'gpu_observation': 'CuPy 14.2.0 imports with /tmp cache; cudaErrorNoDevice',
                  'hardware_observation_source': 'mechanical inventory, 2026-10-09'}
    write_json(output / 'provenance.json', provenance)
    write_json(output / 'tests.json', test_report)
    captures, intervals, reviewed, interval_hash = load_reviewed(ROOT, workers=CONFIG['threads'])
    population = make_population(captures, intervals)
    slots = make_slot_manifest(population)
    validate_slots(population, slots)
    manifest_keys = ('capture', 'exposure', 'row', 'source_frame', 'timestamp_ms', 'target_theta_deg', 'demand_diopters',
                     'p1_available', 'p4_available', 'complete_valid', 'invalid_reason', 'detector_status')
    manifest = {key: population[key] for key in manifest_keys}
    np.savez_compressed(output / 'population.npz', **manifest)
    np.savez_compressed(output / 'slots.npz', **slots)
    manifest_hash, slot_hash = array_hash(manifest), array_hash(slots)
    write_json(output/'population.json', {'schema': 'reviewed_population_v1', 'frame_records': 'population.npz',
                                         'slot_records': 'slots.npz', 'frame_hash': manifest_hash, 'slot_hash': slot_hash,
                                         'interval_sha256': interval_hash, 'expected_exposures': 20,
                                         'scheduled': len(population['row']), 'expected_slots': len(slots['held_point']),
                                         'captures': {name: c.digest for name,c in captures.items()},
                                         'reviewed_intervals': intervals, 'slot_point_order': 'corresponding P4, 0/1/2',
                                         'candidate_independent': True})
    write_json(output/'source_metadata.json', {name: capture.meta for name,capture in captures.items()})
    record = census(captures, intervals, population)
    write_json(output / 'census.json', record)
    measurements = relative_coordinates(population['p1'], population['p4'])
    displacement = centroid_displacement(population['p1'], population['p4'])
    np.savez_compressed(output/'measurements.npz', p1=population['p1'], p4_corresponding=population['p4'],
                        relative=measurements, centroid_displacement=displacement)
    covariance, covariance_record = estimate_covariance(population, intervals)
    np.savez_compressed(output / 'covariance.npz', **covariance)
    covariance_record['arrays_hash'] = array_hash(covariance)
    write_json(output / 'covariance.json', covariance_record)
    fixture_record = synthetic_forward(output, covariance)
    reference_name = min(captures, key=lambda name: next(i['demand_diopters_label'] for i in intervals if i['capture'] == name))
    reference = [r for r in record['exposures'] if r['capture'] == reference_name]
    bootstrap = fit_bootstrap([r['displacement_x_px']['mean'] for r in reference],
                              [r['target_theta_deg'] for r in reference])
    bootstrap.update({'reference_capture': reference_name, 'Aref_D': reference[0]['demand_diopters_label'],
                      'population_hash': manifest_hash, 'source_hash': dirty_hash,
                      'reference_table': reference, 'free_parameter_count': 2,
                      'coefficient_order': ['intercept', 'standardized_displacement_gain']})
    theta = apply_bootstrap(displacement[:, 0], bootstrap)
    theta = np.where(population['complete_valid'], theta, np.nan)
    lower, upper = CONFIG['theta_search_bounds_deg']
    state_valid = np.isfinite(theta)
    within_bounds = state_valid & (theta >= lower) & (theta <= upper)
    xlo, xhi = bootstrap['reference_displacement_domain_px']
    extrapolated = state_valid & ((displacement[:, 0] < xlo) | (displacement[:, 0] > xhi))
    states = {'theta_visual_deg': theta, 'accommodation_start_D': np.where(state_valid, population['demand_diopters'], np.nan),
              'initialization_valid': state_valid, 'within_proposed_theta_bounds': within_bounds,
              'outside_reference_mean_domain': extrapolated,
              'status': np.where(state_valid, 'provisional_centroid_g1', population['invalid_reason'])}
    np.savez_compressed(output/'initial_states.npz', **states)
    bootstrap['framewise_start_hash'] = array_hash(states)
    bootstrap['framewise_domain_by_exposure'] = [
        {'exposure': i, 'finite_starts': int((state_valid & (population['exposure']==i)).sum()),
         'outside_proposed_theta_bounds': int((state_valid & ~within_bounds & (population['exposure']==i)).sum()),
         'outside_reference_mean_domain': int((extrapolated & (population['exposure']==i)).sum()),
         'theta_start_deg': finite_summary(theta[population['exposure']==i])} for i in range(20)]
    write_json(output / 'bootstrap.json', bootstrap)
    plots(output, captures, intervals, population, bootstrap, theta, record)
    elapsed = time.perf_counter()-started
    summary = {'status': 'COMPLETE_PENDING_REVIEW', 'candidate': 'bootstrap_only', 'scheduled': record['scheduled'],
               'valid': record['valid'], 'invalid': record['invalid'], 'slots': record['expected_slots'],
               'manifest_hash': manifest_hash, 'source_hash': dirty_hash,
               'tests': {k: test_report[k] for k in ('passed', 'failed', 'errors', 'skipped')},
               'synthetic_forward': fixture_record, 'bootstrap_gain_deg_per_px': bootstrap['gain_deg_per_px'],
               'reference_ordered': bootstrap['ordered'], 'reference_mean_residuals_deg': bootstrap['reference_mean_residuals_deg'],
               'outside_proposed_theta_bounds': int((state_valid & ~within_bounds).sum()),
               'fit_complete': False, 'fit_certified': False, 'crosscheck_complete': False, 'comparison_complete': False,
               'elapsed_seconds': elapsed, 'decision': 'none; execution owner must review census, tests and plots'}
    write_json(output/'summary.json', summary)
    (output/'STAGE_REPORT.md').write_text('# Stage 1 evidence awaiting review\n\n'
        'G0 and G1 status: COMPLETE_PENDING_REVIEW. Decision: none.\n\n'
        'See census.json, tests.json, bootstrap.json, covariance.json, population.json and plots.\n'
        'A successful script exit does not authorize a gate. No optical calibration is claimed.\n')
    (output/'PROGRESS.md').write_text('# Progress\n\nCurrent gate: G0/G1 review.\n'
        'Last checkpoint: population.npz and initial_states.npz (provisional).\n'
        'One next action: execution owner audits the evidence and records gate decisions.\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
