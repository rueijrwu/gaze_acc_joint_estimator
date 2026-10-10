#!/usr/bin/env python3
"""S2/G2: empirical P1 reference, constrained distortion, and common scale."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = next(parent for parent in Path(__file__).resolve().parents
            if (parent/'docs/Theory.md').is_file() and (parent/'data/fixations/fixation_intervals.json').is_file())
sys.path.insert(0, str(ROOT))
import numpy as np
import scipy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from distortion_model.data import load_reviewed, make_population, array_hash
from distortion_model.alignment import p1_balance, empirical_template
from distortion_model.p1 import (P1Model, ProfileObjective, fit_p1, evaluate_p1, scaled_bounds, domain_margins)
from distortion_model.gaze import fit_bootstrap, apply_bootstrap
from distortion_model.geometry import relative_coordinates

CONFIG = {
    'stage': 'S2/G2', 'candidate': 'p1_initialization_only', 'backend': 'NumPy/SciPy float64 CPU',
    'workers': 4, 'population': 'unchanged stage-one full scheduled rows and complete-pair fit policy',
    'reference_selection': 'smallest mean fixed-camera-axis P1 balance score in reference capture; '
                           'fixed operational reference; physical source symmetry unknown',
    'source_geometry_status': 'unknown; stored L/M/R and P1-P4 pairing do not establish physical mirror-source geometry',
    'symmetry_score': 'norm([(Lx+Rx-2*Mx),(Ry-Ly)])/norm(R-L); observed pattern diagnostic, not anatomical alignment',
    'axis_policy': 'fixed native camera x right, y down; optical/camera alignment unknown; no fitted rotation/affine',
    'omega1_policy': 'fixed at mean stage-one visual gaze in selected reference exposure; no continuous zero extrapolation',
    'template_policy': 'mean of framewise centered P1 triples in selected complete-valid reference exposure; no symmetrizing',
    'length_policy': 'fixed empirical native-pixel reference length',
    'origin_policy': 'fixed reference-template centroid; true spatial optical origin unknown',
    'isotropic_gaze_scale_policy': 'alpha1+beta1=0; trace-free quadratic anisotropy initially free, no free isotropic term',
    'free_parameters': ['delta1=100*alpha1=-100*beta1', 'keystone1=10*reference_radius_px*gamma1'],
    'parameter_count': 2, 'scale_policy': 'p1_profile_v1', 'covariance_policy': 'unchanged frozen stage-one R11',
    'theta_bounds_deg': [-20., 20.], 'starts': 'zero plus fixed 15%/-10% of admissible positive bounds',
    'bounds_policy': 'actual reference coordinates and shifted full visual domain; scales and denominators >=0.5',
    'optimizer': {'method': 'SciPy least_squares trf; exact analytic profile Jacobian', 'max_nfev': 100,
                  'ftol': 1e-10, 'xtol': 1e-10, 'gtol': 1e-8},
    'certificate': {'conditional_rank_rtol': 1e-10, 'scaled_projected_gradient_inf': 1e-6,
                    'kind': 'conditional two-global-parameter fit, not full-model certification'},
    'derivative_test_tolerances': {'rtol': 3e-6, 'atol': 1e-5, 'finite_difference_steps': 'theta 1e-4; alpha/beta 1e-8; gamma 1e-10; scaled params 1e-6'},
    'alternative_policy': 'retain all five balance profiles; fit selected reference and adjacent sampled reference(s); never choose omega by training cost',
    'block_policy': 'five contiguous original-row partitions per exposure; diagnostic fits only; full fit keeps all rows',
    'bootstrap_refresh': 'one framewise displacement/g refresh using fit-snapshot scale, then recompute scale/means at refreshed visual theta',
    'p4_policy': 'no P4-dependent scale; P4 used only through the upstream approximate gaze and the bootstrap refresh',
    'plot_stride': 10,
}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


def values_summary(values):
    x = np.asarray(values)
    x = x[np.isfinite(x)]
    return {'count': len(x), 'mean': float(x.mean()) if len(x) else None,
            'std': float(x.std()) if len(x) else None, 'min': float(x.min()) if len(x) else None,
            'max': float(x.max()) if len(x) else None,
            'p05': float(np.percentile(x, 5)) if len(x) else None, 'p95': float(np.percentile(x, 95)) if len(x) else None}


def runtime_observations():
    record = {'python': sys.version, 'numpy': np.__version__, 'scipy': scipy.__version__,
              'matplotlib': matplotlib.__version__, 'cpu_count': os.cpu_count(),
              'threads': {k: os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')}}
    try:
        os.environ.setdefault('CUPY_CACHE_DIR', '/tmp/acc-cupy-stage2-cache')
        import cupy as cp
        record['cupy'] = {'version': cp.__version__, 'cuda_device_count': cp.cuda.runtime.getDeviceCount()}
    except Exception as error:
        record['cupy'] = {'available': False, 'error': str(error)}
    try:
        remote = subprocess.run(['git','ls-remote','origin','refs/heads/exp5_distortion_model'], cwd=ROOT,
                                text=True, capture_output=True, timeout=20)
        record['remote'] = {'returncode': remote.returncode, 'stdout': remote.stdout.strip(), 'stderr': remote.stderr.strip()}
    except subprocess.TimeoutExpired:
        record['remote'] = {'error': 'timeout after 20s; remote unverified'}
    return record


def balance_records(population, intervals, score, components, score_valid, theta):
    records = []
    blocks = np.full(len(theta), -1, dtype=np.int8)
    for exposure, interval in enumerate(intervals):
        scheduled = np.flatnonzero(population['exposure']==exposure)
        all_p1 = population['p1_available'][scheduled].all(axis=1)&score_valid[scheduled]
        full = population['complete_valid'][scheduled]&score_valid[scheduled]&np.isfinite(theta[scheduled])
        block_records = []
        for block_id, indices in enumerate(np.array_split(scheduled, 5)):
            blocks[indices] = block_id
            valid = indices[population['complete_valid'][indices]&score_valid[indices]&np.isfinite(theta[indices])]
            block_records.append({'block': block_id, 'start_row': int(population['row'][indices[0]]),
                                  'end_row_exclusive': int(population['row'][indices[-1]]+1),
                                  'scheduled': len(indices), 'valid': len(valid),
                                  'score': values_summary(score[valid]),
                                  'mean_visual_gaze_start_deg': values_summary(theta[valid])})
        records.append({'exposure': exposure, 'capture': interval['capture'],
                        'nominal_target_deg': interval['target_theta_deg'], 'demand_D': interval['demand_diopters_label'],
                        'scheduled': len(scheduled), 'all_p1_available': int(all_p1.sum()), 'complete_valid': int(full.sum()),
                        'collapsed_or_nonfinite_balance_rows': int((~score_valid[scheduled]&population['p1_available'][scheduled].all(axis=1)).sum()),
                        'all_p1_score': values_summary(score[scheduled[all_p1]]),
                        'calibration_population_score': values_summary(score[scheduled[full]]),
                        'horizontal_imbalance': values_summary(components[scheduled[full], 0]),
                        'vertical_imbalance': values_summary(components[scheduled[full], 1]),
                        'visual_gaze_start_deg': values_summary(theta[scheduled[full]]), 'blocks': block_records})
    return records, blocks


def residual_records(population, intervals, evaluation, covariance, theta, block_ids):
    output = []
    for exposure, interval in enumerate(intervals):
        keep = population['exposure']==exposure
        valid = keep&evaluation['valid']
        r = evaluation['edge_residual'][valid]
        norm = np.sum(r*np.linalg.solve(covariance, r.T).T, axis=1) if len(r) else np.array([])
        blocks = []
        for block_id in range(5):
            select = valid&(block_ids==block_id)
            residual = evaluation['edge_residual'][select]
            blocks.append({'block': block_id, 'valid_scale': int(select.sum()),
                           'g': values_summary(evaluation['g'][select]),
                           'edge_coordinate_rmse_px': float(np.sqrt(np.mean(residual**2))) if len(residual) else None})
        output.append({'exposure': exposure, 'capture': interval['capture'], 'nominal_target_deg': interval['target_theta_deg'],
                       'demand_D': interval['demand_diopters_label'], 'scheduled': int(keep.sum()),
                       'scale_valid': int(valid.sum()), 'unavailable': int((keep&~evaluation['valid']).sum()),
                       'g': values_summary(evaluation['g'][valid]), 'visual_theta_snapshot_deg': values_summary(theta[valid]),
                       'signed_edge_residual_means_px': r.mean(axis=0).tolist() if len(r) else None,
                       'edge_coordinate_rmse_px': float(np.sqrt(np.mean(r**2))) if len(r) else None,
                       'mean_weighted_edge_squared_error': float(norm.mean()) if len(norm) else None, 'blocks': blocks})
    return output


def model_record(model, best, template_count, template_exposure, parent, radius):
    return {'schema': 'p1_reference_checkpoint_v1', 'stage': 'S2/G2', 'scope': 'partial optical initialization; no P4 model',
            'b1_reference_px': model.b1.tolist(), 'omega1_visual_deg': model.omega1,
            'alpha1_deg_minus2': model.k1[0], 'beta1_deg_minus2': model.k1[1], 'gamma1_px_minus1_deg_minus1': model.k1[2],
            'template_frame_count': template_count, 'template_exposure': template_exposure, 'reference_radius_px': radius,
            'theta_bounds_deg': list(model.theta_bounds), 'scale_policy': 'p1_profile_v1',
            'parameter_roster': {'fitted': CONFIG['free_parameters'], 'derived': ['alpha1', 'beta1=-alpha1', 'gamma1', 'per-frame g'],
                                 'fixed': ['omega1','b1 template','native-pixel length','centroid origin','camera axes','alpha1+beta1=0'],
                                 'free_count': 2},
            'reference_evidence_kind': 'empirical observed-pattern balance; source symmetry/physical zero unresolved',
            'physical_origin_status': 'unknown; fixed empirical centroid convention',
            'parent_population_hash': parent['frame_hash'], 'conditional_fit': best, 'domain': domain_margins(model)}


def make_plots(output, population, records, reference_records, chosen, identity, fitted, refreshed, block_fits):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    for name in dict.fromkeys(population['capture'].tolist()):
        selected = [r for r in records if r['capture']==name]
        targets = [r['nominal_target_deg'] for r in selected]
        axes[0].plot(targets, [r['calibration_population_score']['mean'] for r in selected], 'o-', label=name)
        axes[1].plot(targets, [r['horizontal_imbalance']['mean'] for r in selected], 'o-')
        axes[2].plot(targets, [r['vertical_imbalance']['mean'] for r in selected], 'o-')
        for r in selected:
            axes[0].scatter([r['nominal_target_deg']]*5, [b['score']['mean'] for b in r['blocks']], s=12, alpha=.4)
    axes[0].axvline(chosen['nominal_target_deg'], color='black', linestyle='--', label='Operational reference condition')
    axes[0].legend(fontsize=7)
    for ax,title in zip(axes, ['Observed balance + contiguous blocks','Horizontal imbalance','Vertical imbalance']):
        ax.set(xlabel='Nominal visual target (degrees)', ylabel='Scale invariant pattern imbalance', title=title)
    fig.suptitle('Fixed camera axes; physical illuminator symmetry and optical zero unknown')
    fig.savefig(output/'p1_balance.png', dpi=150); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for name in dict.fromkeys(population['capture'].tolist()):
        selected = [r for r in records if r['capture']==name]
        targets = [r['nominal_target_deg'] for r in selected]
        means_g, identity_rms, fitted_rms = [], [], []
        for r in selected:
            rows = population['exposure']==r['exposure']
            valid = rows&fitted['valid']
            means_g.append(float(np.mean(fitted['g'][valid])))
            identity_rms.append(float(np.sqrt(np.mean(identity['edge_residual'][valid]**2))))
            fitted_rms.append(float(np.sqrt(np.mean(fitted['edge_residual'][valid]**2))))
            indices = np.flatnonzero(rows)[::10]
            axes[1, 0].plot(population['row'][indices], fitted['g'][indices], linewidth=.6)
            axes[1, 1].plot(population['row'][indices], np.sqrt(np.mean(fitted['edge_residual'][indices]**2, axis=1)), linewidth=.6)
        axes[0, 0].plot(targets, means_g, 'o-', label=name)
        axes[0, 1].plot(targets, identity_rms, '--o', alpha=.6)
        axes[0, 1].plot(targets, fitted_rms, '-o', label=name)
    axes[0, 0].set(xlabel='Nominal gaze bin (degrees)', ylabel='P1-only g', title='Effective reference magnification')
    axes[0, 0].legend(fontsize=7)
    axes[0, 1].set(xlabel='Nominal gaze bin (degrees)', ylabel='Edge coordinate RMSE (native px)', title='Dashed identity / solid constrained K1')
    axes[0, 1].legend(fontsize=7)
    axes[1, 0].set(xlabel='Original row', ylabel='P1-only g', title='Scale trajectories at frozen stage-one theta')
    axes[1, 1].set(xlabel='Original row', ylabel='Edge coordinate RMSE (px)', title='Residual trajectories; unavailable rows preserved')
    fig.savefig(output/'p1_scale_and_residuals.png', dpi=150); plt.close(fig)
    fig, axes = plt.subplots(4, 5, figsize=(17, 12), constrained_layout=True)
    for r,ax in zip(records, axes.flat):
        select = (population['exposure']==r['exposure'])&fitted['valid']
        indices = np.flatnonzero(select)
        index = indices[len(indices)//2]
        p = population['p1'][index]
        measured = p-p.mean(axis=0)
        f = fitted['F1'][index]
        predicted = fitted['g'][index]*(f-f.mean(axis=0))
        for points,label,style in [(measured,'measured','o-'), (predicted,'predicted','x--')]:
            ax.plot(points[[0,1,2,0], 0], points[[0,1,2,0], 1], style, markersize=4, linewidth=.8, label=label)
        ax.set_aspect('equal'); ax.invert_yaxis()
        ax.set(title=f"{r['capture'][8]} / {r['nominal_target_deg']:g}° / row {population['row'][index]}", xlabel='Centered x (px)', ylabel='Centered y (px)')
    axes.flat[0].legend(fontsize=7)
    fig.savefig(output/'p1_patterns.png', dpi=120); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for name in dict.fromkeys(population['capture'].tolist()):
        selected = [r for r in records if r['capture']==name]
        means = []
        for r in selected:
            select = (population['exposure']==r['exposure'])&np.isfinite(refreshed)
            means.append(float(refreshed[select].mean()))
        axes[0].plot([r['nominal_target_deg'] for r in selected], means, 'o-', label=name)
    axes[0].set(xlabel='Nominal gaze bin (degrees)', ylabel='Refreshed mean gaze start (degrees)', title='Framewise displacement/g bootstrap refresh')
    axes[0].legend(fontsize=7)
    for i, record in enumerate(block_fits):
        axes[1].scatter([0,1], record['best']['scaled_parameters'], label=f'Contiguous block {i}')
    axes[1].set_xticks([0,1], ['delta1', 'keystone1'])
    axes[1].set(ylabel='Scaled coefficient', title='Block stability diagnostic; no selection of blocks')
    axes[1].legend(fontsize=7)
    fig.savefig(output/'bootstrap_refresh_and_blocks.png', dpi=150); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, default=ROOT/'experiments/distortion_model/stage_01_data_and_bootstrap/results/attempt_02')
    parser.add_argument('--test-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    parent, output = args.parent.resolve(), args.output.resolve()
    if output.exists():
        raise SystemExit('Choose a new attempt directory; previous attempts are preserved')
    previous_summary = json.loads((parent/'summary.json').read_text())
    if previous_summary.get('G0_decision') not in ('GO','GO_WITH_LIMIT') or previous_summary.get('G1_decision') not in ('GO','GO_WITH_LIMIT'):
        raise SystemExit('Reviewed compatible G0 and G1 evidence is required')
    tests = json.loads(args.test_report.read_text())
    if tests['failed'] or tests['errors'] or tests['skipped'] or tests['passed']<30:
        raise SystemExit('Complete passing stage-one and stage-two contract evidence is required')
    test_sources = {str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest()
                    for p in sorted(list((ROOT/'distortion_model').glob('*.py'))+list((ROOT/'tests').glob('*.py')))}
    if tests['source_hashes'] != test_sources:
        raise SystemExit('Test source hashes do not match the current implementation')
    output.mkdir(parents=True)
    write_json(output/'config.json', CONFIG)
    write_json(output/'tests.json', tests)
    source_files = sorted(set(list((ROOT/'distortion_model').glob('*.py'))+list((ROOT/'tests').glob('*.py'))
                              +list((ROOT/'docs').glob('*.md'))+list((ROOT/'docs/stages').glob('*.md'))
                              +[Path(__file__).resolve(), ROOT/'requirements-stage2.txt']))
    source_hashes = {str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for p in source_files}
    for p in source_files:
        archive = output/'source_snapshot'/p.relative_to(ROOT)
        archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, archive)
    source_hash = sha256(json.dumps(source_hashes,sort_keys=True).encode()).hexdigest()
    head = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()
    provenance = {'created_utc': datetime.now(timezone.utc).isoformat(), 'source_commit': head,
                  'source_hash': source_hash, 'source_hashes': source_hashes,
                  'config_hash': sha256((output/'config.json').read_bytes()).hexdigest(),
                  'runtime': runtime_observations(), 'parent_path': str(parent.relative_to(ROOT)),
                  'parent_summary_sha256': sha256((parent/'summary.json').read_bytes()).hexdigest()}
    write_json(output/'provenance.json', provenance)
    manifest_record = json.loads((parent/'population.json').read_text())
    manifest = dict(np.load(parent/'population.npz', allow_pickle=False))
    parent_states = dict(np.load(parent/'initial_states.npz', allow_pickle=False))
    parent_measurements = dict(np.load(parent/'measurements.npz', allow_pickle=False))
    covariance_arrays = dict(np.load(parent/'covariance.npz', allow_pickle=False))
    covariance_record = json.loads((parent/'covariance.json').read_text())
    parent_bootstrap = json.loads((parent/'bootstrap.json').read_text())
    if array_hash(manifest)!=manifest_record['frame_hash'] or array_hash(covariance_arrays)!=covariance_record['arrays_hash']:
        raise ValueError('changed parent population or covariance')
    if array_hash(parent_states)!=parent_bootstrap['framewise_start_hash']:
        raise ValueError('changed parent frame starts')
    captures, intervals, reviewed, interval_sha = load_reviewed(ROOT, workers=4)
    population = make_population(captures, intervals)
    reconstructed_manifest = {key: population[key] for key in manifest}
    if array_hash(reconstructed_manifest)!=manifest_record['frame_hash'] or interval_sha!=manifest_record['interval_sha256']:
        raise ValueError('current reviewed sources/population differ from parent checkpoint')
    if not np.array_equal(population['p1'], parent_measurements['p1'], equal_nan=True) or not np.array_equal(population['p4'], parent_measurements['p4_corresponding'], equal_nan=True):
        raise ValueError('saved native measurements differ from original source bytes')
    theta = parent_states['theta_visual_deg']
    complete = population['complete_valid']&np.isfinite(theta)
    if not np.array_equal(complete, population['complete_valid']):
        raise ValueError('parent starts unavailable for complete calibration frames')
    covariance = covariance_arrays['R11']
    scores, components, score_valid = p1_balance(population['p1'])
    records, block_ids = balance_records(population, intervals, scores, components, score_valid, theta)
    reference_records = sorted([r for r in records if r['capture']==parent_bootstrap['reference_capture']], key=lambda r:r['nominal_target_deg'])
    if any(r['calibration_population_score']['mean'] is None for r in reference_records):
        raise ValueError('reference evidence absent for a required gaze condition')
    chosen_index = int(np.argmin([r['calibration_population_score']['mean'] for r in reference_records]))
    chosen_record = reference_records[chosen_index]
    candidate_indices = sorted({chosen_index, max(0,chosen_index-1), min(4,chosen_index+1)})
    candidate_references = [reference_records[index] for index in candidate_indices]
    write_json(output/'p1_balance.json', {'metric': CONFIG['symmetry_score'], 'physical_source_geometry': 'unknown',
                                         'reference_zero_interpretation': 'fixed empirical operational reference; physical symmetry zero unresolved',
                                         'chosen_reference_exposure': chosen_record['exposure'],
                                         'chosen_reference_is_endpoint': chosen_index in (0,4),
                                         'all_reference_alternatives': reference_records, 'exposures': records})
    fit_indices = np.flatnonzero(complete)
    observed_edges = parent_measurements['relative'][:, :4]
    fit_theta, fit_edges = theta[complete], observed_edges[complete]
    fit_exposure = population['exposure'][complete]
    print(f'Fitting all {len(fit_theta)} complete frames across {len(np.unique(fit_exposure))} exposures', flush=True)
    fit_started = time.perf_counter()
    def fit_reference(record):
        select = complete&(population['exposure']==record['exposure'])
        template, radius = empirical_template(population['p1'][select])
        omega1 = float(theta[select].mean())
        objective = ProfileObjective(fit_theta, fit_edges, fit_exposure, covariance, template, omega1)
        lower, upper = scaled_bounds(template, omega1)
        starts = [[0.,0.], (upper*np.array([.15,-.10])).tolist()]
        model, best, outcomes = fit_p1(objective, starts, max_evaluations=100)
        identity_residual = objective.fun([0.,0.])
        return {'record': record, 'model': model, 'radius': radius, 'count': int(select.sum()), 'best': best,
                'outcomes': outcomes, 'identity_cost': float(identity_residual@identity_residual/2)}
    with ThreadPoolExecutor(max_workers=min(4,len(candidate_references))) as executor:
        references = list(executor.map(fit_reference, candidate_references))
    chosen = next(r for r in references if r['record']['exposure']==chosen_record['exposure'])
    model = chosen['model']
    conditional_fit_seconds = time.perf_counter()-fit_started
    reference_json = []
    for r in references:
        name = f"reference_{r['record']['exposure']:02d}"
        checkpoint = model_record(r['model'], r['best'], r['count'], r['record']['exposure'], manifest_record, r['radius'])
        checkpoint['source_hash'] = source_hash
        checkpoint['parent_covariance_hash'] = covariance_record['arrays_hash']
        write_json(output/f'{name}.json', checkpoint)
        reference_json.append({'exposure': r['record']['exposure'], 'nominal_target_deg': r['record']['nominal_target_deg'],
                               'omega1': r['model'].omega1, 'model_file': f'{name}.json', 'best': r['best'],
                               'outcomes': r['outcomes'], 'identity_cost': r['identity_cost'],
                               'selection': 'chosen_by_observed_balance' if r is chosen else 'bounded_reference_alternative'})
    write_json(output/'reference_fits.json', reference_json)
    if not chosen['best']['conditional_fit_certified']:
        write_json(output/'summary.json', {'status':'COMPLETE_PENDING_REVIEW','G2_decision':'none',
                                          'conditional_fit_available':False,'reason':'selected P1 fit uncertified',
                                          'scheduled':len(theta), 'complete_valid':int(complete.sum()),
                                          'source_hash':source_hash,'parent_population_hash':manifest_record['frame_hash']})
        print('Selected conditional fit uncertified; checkpoint preserved. No bootstrap refresh.', flush=True)
        return
    fitted = evaluate_p1(theta, observed_edges, covariance, model)
    identity_model = P1Model(model.b1, model.omega1)
    identity = evaluate_p1(theta, observed_edges, covariance, identity_model)
    if not fitted['valid'][complete].all() or not identity['valid'][complete].all():
        raise ValueError('scale unavailable for an expected complete fitting row')
    reference_exposures = [r['exposure'] for r in reference_records]
    normalized_displacement = parent_measurements['centroid_displacement'][:, 0]/fitted['g']
    corrected_means = [float(np.mean(normalized_displacement[complete&(population['exposure']==exposure)])) for exposure in reference_exposures]
    bootstrap = fit_bootstrap(corrected_means, [r['nominal_target_deg'] for r in reference_records])
    refreshed_theta = apply_bootstrap(normalized_displacement, bootstrap)
    refreshed_theta = np.where(fitted['valid'], refreshed_theta, np.nan)
    exported = evaluate_p1(refreshed_theta, observed_edges, covariance, model)
    bootstrap.update({'reference_capture':parent_bootstrap['reference_capture'], 'Aref_D':parent_bootstrap['Aref_D'],
                      'reference_exposures':reference_exposures, 'reference_mean_framewise_displacement_over_g_px':corrected_means,
                      'parent_bootstrap_sha256':sha256((parent/'bootstrap.json').read_bytes()).hexdigest(),
                      'scale_snapshot':'p1_at_frozen_stage_one_theta.npz; bootstrap used framewise ratios',
                      'export_scale_snapshot':'p1_at_refreshed_theta.npz; recomputed at exported visual states',
                      'source_hash':source_hash,'population_hash':manifest_record['frame_hash']})
    state_arrays = {'theta_visual_deg':refreshed_theta, 'accommodation_start_D':parent_states['accommodation_start_D'],
                    'initialization_valid':np.isfinite(refreshed_theta), 'p1_scale_valid':exported['valid'],
                    'within_proposed_theta_bounds':np.isfinite(refreshed_theta)&(refreshed_theta>=-20)&(refreshed_theta<=20),
                    'status':np.where(exported['valid'],'provisional_scale_corrected_centroid',
                                      np.where(complete,'p1_model_or_domain_unavailable',population['invalid_reason']))}
    bootstrap['state_hash'] = array_hash(state_arrays)
    np.savez_compressed(output/'initial_states.npz', **state_arrays)
    np.savez_compressed(output/'p1_at_frozen_stage_one_theta.npz', **fitted)
    np.savez_compressed(output/'p1_identity_at_frozen_stage_one_theta.npz', **identity)
    np.savez_compressed(output/'p1_at_refreshed_theta.npz', **exported)
    np.savez_compressed(output/'balance.npz', score=scores, components=components, validity=score_valid, block_ids=block_ids)
    write_json(output/'bootstrap.json', bootstrap)
    write_json(output/'p1_reference.json', model_record(model, chosen['best'], chosen['count'], chosen_record['exposure'], manifest_record, chosen['radius']) |
               {'source_hash':source_hash,'parent_covariance_hash':covariance_record['arrays_hash']})
    print('Running five contiguous-block coefficient diagnostics', flush=True)
    def fit_block(block_id):
        select = complete&(block_ids==block_id)
        objective = ProfileObjective(theta[select], observed_edges[select], population['exposure'][select], covariance,
                                     model.b1, model.omega1)
        _, best, outcomes = fit_p1(objective, [chosen['best']['scaled_parameters'],[0.,0.]], max_evaluations=100)
        return {'block':block_id,'frames':int(select.sum()),'exposures':len(objective.groups),'best':best,'outcomes':outcomes}
    with ThreadPoolExecutor(max_workers=4) as executor:
        block_fits = list(executor.map(fit_block, range(5)))
    write_json(output/'block_fits.json', block_fits)
    old_records = residual_records(population, intervals, identity, covariance, theta, block_ids)
    fit_records = residual_records(population, intervals, fitted, covariance, theta, block_ids)
    refreshed_records = residual_records(population, intervals, exported, covariance, refreshed_theta, block_ids)
    evaluation_records = {'identity_at_original_theta':old_records,'constrained_k1_at_original_theta':fit_records,
                          'same_k1_at_refreshed_theta':refreshed_records,
                          'interpretation':'component diagnostics on the same original population; not physiological accuracy or crosscheck'}
    write_json(output/'p1_residuals.json', evaluation_records)
    alternative_arrays = {}
    for r in references:
        evaluated = evaluate_p1(theta, observed_edges, covariance, r['model'])
        alternative_arrays[f"g_reference_{r['record']['exposure']:02d}"] = evaluated['g']
    np.savez_compressed(output/'reference_alternative_scales.npz', **alternative_arrays)
    write_json(output/'checkpoint.json', {'schema':'stage_02_p1_initialization_checkpoint_v1',
        'source_commit':head,'source_hash':source_hash,'config_hash':provenance['config_hash'],
        'parent_population_path':str((parent/'population.npz').relative_to(ROOT)), 'parent_population_hash':manifest_record['frame_hash'],
        'parent_slot_path':str((parent/'slots.npz').relative_to(ROOT)), 'parent_slot_hash':manifest_record['slot_hash'],
        'parent_covariance_path':str((parent/'covariance.npz').relative_to(ROOT)), 'parent_covariance_hash':covariance_record['arrays_hash'],
        'p1_reference_file':'p1_reference.json', 'p1_reference_sha256':sha256((output/'p1_reference.json').read_bytes()).hexdigest(),
        'bootstrap_file':'bootstrap.json','bootstrap_sha256':sha256((output/'bootstrap.json').read_bytes()).hexdigest(),
        'frame_states_file':'initial_states.npz','frame_states_hash':array_hash(state_arrays),
        'consistent_exported_p1_file':'p1_at_refreshed_theta.npz','consistent_exported_p1_hash':array_hash(exported),
        'fit_theta_file':'parent initial_states.npz','fit_population':'all complete rows, all twenty exposures',
        'unknown':['physical source symmetry','physical omega1','camera/optical-axis alignment','optical spatial origin'],
        'full_optical_fit_certified':False,'next_gate_requires_review':True})
    make_plots(output, population, records, reference_records, chosen_record, identity, fitted, refreshed_theta, block_fits)
    def aggregate(records):
        return {'equal_exposure_edge_coordinate_rmse_px':float(np.sqrt(np.mean([r['edge_coordinate_rmse_px']**2 for r in records]))),
                'equal_exposure_mean_weighted_edge_squared_error':float(np.mean([r['mean_weighted_edge_squared_error'] for r in records]))}
    summary = {'status':'COMPLETE_PENDING_REVIEW','G2_decision':'none','source_hash':source_hash,
        'parent_population_hash':manifest_record['frame_hash'],'scheduled':len(theta),'complete_valid':int(complete.sum()),
        'unavailable':int((~complete).sum()), 'expected_slots_unchanged':manifest_record['expected_slots'],
        'scale_valid_at_fit_theta':int(fitted['valid'].sum()),'scale_valid_at_export_theta':int(exported['valid'].sum()),
        'chosen_reference_nominal_target_deg':chosen_record['nominal_target_deg'],'omega1_fixed_operational_deg':model.omega1,
        'physical_zero_identified':False, 'chosen_reference_is_endpoint':chosen_index in (0,4),
        'k1_coefficients_native_units':list(model.k1),'free_parameter_count':2,
        'identity':aggregate(old_records),'constrained_k1':aggregate(fit_records),'refreshed_theta':aggregate(refreshed_records),
        'g_at_export_theta':values_summary(exported['g']),
        'conditional_fit_certified':chosen['best']['conditional_fit_certified'],
        'scaled_projected_gradient_inf':chosen['best']['scaled_projected_gradient_inf'],
        'conditional_rank':chosen['best']['conditional_rank'],'multistart_outcomes':chosen['outcomes'],
        'block_scaled_coefficients':[r['best']['scaled_parameters'] for r in block_fits],
        'block_conditional_certified':[r['best']['conditional_fit_certified'] for r in block_fits],
        'bootstrap_reference_mean_predictions_deg':bootstrap['reference_mean_predictions_deg'],
        'bootstrap_gain_deg_per_reference_px':bootstrap['gain_deg_per_px'],
        'theta_refresh_change_deg':values_summary(refreshed_theta-theta),
        'finite_refreshed_starts':int(np.isfinite(refreshed_theta).sum()),
        'outside_proposed_theta_bounds':int((np.isfinite(refreshed_theta)&~state_arrays['within_proposed_theta_bounds']).sum()),
        'tests':{k:tests[k] for k in ('passed','failed','errors','skipped')},
        'conditional_fit_seconds':conditional_fit_seconds,'elapsed_seconds':time.perf_counter()-started,
        'fit_complete':False,'fit_certified':False,'crosscheck_complete':False,'comparison_complete':False,
        'decision':'execution owner must review reference support, coefficient identifiability, residuals and snapshots'}
    write_json(output/'summary.json', summary)
    (output/'STAGE_REPORT.md').write_text('# G2 evidence pending execution-owner review\n\nStatus: COMPLETE_PENDING_REVIEW. Decision: none.\nNo full optical calibration or physical symmetry zero is claimed.\n')
    (output/'PROGRESS.md').write_text('Current gate: G2 review. Last candidate checkpoint: checkpoint.json.\nOne next action: execution-owner audit of G2 evidence.\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
