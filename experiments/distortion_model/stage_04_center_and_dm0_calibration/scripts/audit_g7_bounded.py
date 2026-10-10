"""Reconstruct G7 outputs and local curvature without changing the fit."""
from hashlib import sha256
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys

import cupy as cp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'docs/Theory.md').is_file())
STAGE = ROOT / 'experiments/distortion_model/stage_04_center_and_dm0_calibration'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments/distortion_model/stage_03_p4_baseline_and_deformation/scripts'))
from g45_common import load_npz
from distortion_model.data import array_hash
from distortion_model.geometry import relative_coordinates
from distortion_model.joint import JointDM0, JointSpec
from distortion_model.optics import p1_reference, p4_reference, center_polynomial, predict_relative
from distortion_model.centroid_bound import CentroidBound, BoundedJointDM0


def load_runner():
    path = STAGE / 'scripts/run_joint.py'
    spec = importlib.util.spec_from_file_location('g7_runner_audit', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def main(attempt):
    runner = load_runner()
    inputs, parent_arrays, model = runner.check_parent(STAGE / 'results/g6_attempt_02')
    population = inputs['population']
    valid = inputs['valid']
    exposure = population['exposure']
    y = relative_coordinates(population['p1'][valid], population['p4'][valid])
    e = inputs['exposure']
    targets = np.array([[population['target_theta_deg'][np.flatnonzero(exposure == k)[0]],
                         population['demand_diopters'][np.flatnonzero(exposure == k)[0]]]
                        for k in range(20)])
    spec = JointSpec(np.array(model['b1_reference_px']), np.array(model['b4_reference_px']),
                     model['omega1_visual_deg'], model['omega4_visual_deg'], model['Aref_D'])
    objective = JointDM0(spec, y, e, inputs['covariance']['relative'], targets,
                         xp=cp, chunk_size=32768)
    physical_bound = CentroidBound(spec, spec.pack(model), 1.0)
    bounded_objective = BoundedJointDM0(spec, y, e, inputs['covariance']['relative'], targets,
                         xp=cp, chunk_size=32768, physical_bound=physical_bound)
    output = Path(attempt).resolve()
    summary = json.loads((output / 'summary.json').read_text())
    label = summary['selected_start']
    fit = json.loads((output / label / 'fit.json').read_text())
    with np.load(output / label / 'solution.npz', allow_pickle=False) as z:
        x = np.asarray(z['states'])
        p = np.asarray(z['scaled_globals'])
    xg, pg = cp.asarray(x), cp.asarray(p)

    # Re-evaluate exact full objective and snapshot in chunks on the saved fit.
    eval_saved = objective.evaluate(xg, pg, hessian=True)
    snapshot = objective.snapshot(xg, pg)
    parent_fitted = load_npz(output / 'fitted.npz')
    fit_comparison = {}
    for key in ('g', 'F1', 'F4', 'mu1', 'mu4', 'M', 'D', 'prediction'):
        stored = parent_fitted[key][valid]
        recreated = snapshot[key]
        fit_comparison[key] = {'shape': list(stored.shape),
                               'max_abs_difference': float(np.max(np.abs(stored - recreated))),
                               'array_equal': bool(np.array_equal(stored, recreated))}
    residual = y - snapshot['prediction']
    objective_components = {
        'cost': objective.scalar(eval_saved['cost']),
        'point': objective.scalar(eval_saved['point']),
        'theta_anchor': objective.scalar(eval_saved['theta_anchor']),
        'A_anchor': objective.scalar(eval_saved['A_anchor']),
        'regularization': objective.scalar(eval_saved['regularization']),
        'temporal': objective.scalar(eval_saved['temporal']),
    }
    component_diff = {k: objective_components[k] - fit['components'][k] for k in objective_components}
    # Independent NumPy forward/pixel-loss reconstruction, using the public
    # optics adapter rather than JointDM0.evaluate/batch.
    params = spec.parameters(p)
    native_p1 = population['p1'][valid]
    observed_p1_edges = np.concatenate((native_p1[:, 1] - native_p1[:, 0],
                                        native_p1[:, 2] - native_p1[:, 0]), axis=1)
    independent_prediction, independent_g, independent_valid = predict_relative(
        x[:, 0], x[:, 1], observed_p1_edges, inputs['covariance']['R11'], params)
    independent_residual = y - independent_prediction
    precision10 = np.linalg.solve(inputs['covariance']['relative'], np.eye(10))
    counts = np.bincount(e, minlength=20)
    weights = 1. / (20. * counts[e])
    independent_point = .5 * np.sum(weights * np.einsum(
        'ni,ij,nj->n', independent_residual, precision10, independent_residual))
    independent_means = np.stack([x[e == k].mean(axis=0) for k in range(20)])
    independent_deviation = independent_means - targets
    independent_theta_anchor = .5 * np.sum(independent_deviation[:, 0]**2 / .10**2) / 20
    independent_A_anchor = .5 * np.sum(independent_deviation[:, 1]**2 / .25**2) / 20
    independent_regularization = .5 * np.sum((p[3:6] / spec.template_sigma)**2) + .5 * np.sum(p[17:19]**2)
    independent_components = {
        'point': float(independent_point), 'theta_anchor': float(independent_theta_anchor),
        'A_anchor': float(independent_A_anchor), 'regularization': float(independent_regularization),
        'temporal': 0.0,
    }
    independent_components['cost'] = sum(independent_components.values())
    independent_differences = {k: independent_components[k] - fit['components'][k]
                               for k in independent_components}
    independent_forward_check = {
        'prediction_max_abs_difference_vs_saved': float(np.max(np.abs(independent_prediction - parent_fitted['prediction'][valid]))),
        'g_max_abs_difference_vs_saved': float(np.max(np.abs(independent_g - parent_fitted['g'][valid]))),
        'all_valid': bool(np.all(independent_valid)),
        'component_reconstruction': independent_components,
        'component_differences_vs_saved_fit': independent_differences,
        'full_mean_max_abs_difference_vs_saved': float(np.max(np.abs(independent_means - np.asarray(fit['means'])))),
    }
    # Independently recompute active physical KKT quantities using the archived bounded model.
    physical_slack = physical_bound.values(p)
    physical_kkt = bounded_objective.kkt(pg, eval_saved['gp'])
    physical_certificate = {'bound_record': physical_bound.record(p),
        'inequality_count': len(physical_slack), 'minimum_normalized_slack': float(physical_slack.min()),
        'active_within_1e-7': int(np.sum(physical_slack <= 1e-7)),
        'KKT_residual_inf': physical_kkt['norm'], 'complementarity_inf': physical_kkt['complementarity_inf'],
        'fit_certificate_KKT_residual_inf': fit['certificate']['global_projected_gradient_inf'],
        'fit_certificate_complementarity_inf': fit['certificate']['KKT_complementarity_inf']}
    theta_dense = np.linspace(*spec.theta_bounds, 81)
    params_saved = spec.parameters(p)
    def H_horizontal(theta, accommodation):
        _, mu1, _, ok1 = p1_reference(theta, params_saved)
        _, mu4, ok4 = p4_reference(theta, np.full_like(theta, accommodation), params_saved)
        return center_polynomial(theta, np.full_like(theta, accommodation), params_saved)[:,0] + mu4[:,0] - mu1[:,0], bool(np.all(ok1 & ok4))
    h0, ok0 = H_horizontal(theta_dense, 0.)
    h4, ok4 = H_horizontal(theta_dense, 4.)
    physical_certificate['dense_public_forward_crosscheck'] = {'theta_points':len(theta_dense), 'all_optics_valid':ok0 and ok4,
        'max_abs_0_to_4D_horizontal_shift_px':float(np.max(np.abs(h4-h0))),
        'user_authorized_pixel_envelope_px':physical_bound.slope_floor*1.0,
        'finite_grid_within_envelope':bool(np.max(np.abs(h4-h0)) <= physical_bound.slope_floor + 1e-7)}
    rms_by_exposure = []
    for k in range(20):
        select = e == k
        rms_by_exposure.append({'exposure': k, 'count': int(select.sum()),
                                'native_relative_coordinate_rms_px': float(np.sqrt(np.mean(residual[select] ** 2)))})

    # Recompute certificate's observed finite-difference state Hessian and
    # apply the same active-state projection before local eigensummaries.
    observed = objective.observed_hessian(xg, pg)
    active = objective.active(xg, eval_saved['gx'], objective.state_lower, objective.state_upper)
    hx = cp.asnumpy(observed['hx'])
    active_host = cp.asnumpy(active)
    exposure_counts = []
    total_negative = 0
    total_free_eigenvalues = 0
    global_min = float('inf')
    global_max = -float('inf')
    for k in range(20):
        row_ids = np.flatnonzero(e == k)
        negative = zero = free_dims = two_free = 0
        minimum = float('inf')
        maximum = -float('inf')
        for i in row_ids:
            free = np.flatnonzero(~active_host[i])
            free_dims += len(free)
            if not len(free):
                continue
            block = hx[i][np.ix_(free, free)]
            eigen = np.linalg.eigvalsh(block)
            two_free += int(len(free) == 2)
            negative += int(np.sum(eigen < 0.0))
            zero += int(np.sum(eigen == 0.0))
            total_free_eigenvalues += len(eigen)
            minimum = min(minimum, float(eigen.min()))
            maximum = max(maximum, float(eigen.max()))
        total_negative += negative
        if free_dims:
            global_min = min(global_min, minimum)
            global_max = max(global_max, maximum)
        exposure_counts.append({'exposure': k, 'capture': str(population['capture'][np.flatnonzero(exposure == k)[0]]),
                                'rows': len(row_ids), 'free_state_dimensions': free_dims,
                                'two_free_state_rows': two_free, 'negative_free_local_eigenvalues': negative,
                                'zero_free_local_eigenvalues': zero,
                                'minimum_free_local_eigenvalue': None if not free_dims else minimum,
                                'maximum_free_local_eigenvalue': None if not free_dims else maximum})

    # Rebuild the predeclared 5x20 population schedule and compare all 300 slots
    # in every progress/final compact record.
    scheduled_rows = [np.flatnonzero(exposure_all == k) for exposure_all in [population['exposure']] for k in range(20)]
    schedule = np.concatenate([rows[np.linspace(0, len(rows)-1, 5, dtype=int)] for rows in scheduled_rows])
    with np.load(output / 'compact_schedule.npz', allow_pickle=False) as z:
        schedule_saved = np.asarray(z['scheduled_population_index'])
        schedule_hash_saved = array_hash({k: np.asarray(z[k]) for k in z.files})
    compact_records = []
    for path in sorted((output / 'compact').glob('*.json')):
        data = json.loads(path.read_text())
        slots = data['slots']
        compact_records.append({'path': str(path.relative_to(output)), 'slots': len(slots),
                                'exposures': len({v['exposure'] for v in slots}),
                                'held_points': len({v['held_point'] for v in slots}),
                                'retained_input_valid': sum(bool(v['retained_input_valid']) for v in slots),
                                'held_input_valid': sum(bool(v['held_input_valid']) for v in slots),
                                'scored': sum('held_native_error_px' in v for v in slots),
                                'certified_inference': sum(bool(v.get('certified')) for v in slots),
                                'unresolved_inference': sum(v.get('inference_status') == 'unresolved' for v in slots),
                                'input_unavailable': sum(v.get('inference_status') == 'input_unavailable' for v in slots),
                                'summary': data['summary']})

    checkpoint = json.loads((output / 'checkpoint.json').read_text())
    g6dir = STAGE / 'results/g6_attempt_02'
    g6cp = json.loads((g6dir / 'checkpoint.json').read_text())
    g5cp = ROOT / g6cp['parent_checkpoint_path']
    parent_hashes = {
        'g6_checkpoint': {'recorded': checkpoint['parent_checkpoint_sha256'], 'actual': digest(g6dir / 'checkpoint.json')},
        'g6_summary': {'recorded': checkpoint['parent_summary_sha256'], 'actual': digest(g6dir / 'summary.json')},
        'g5_checkpoint': {'recorded': g6cp['parent_checkpoint_sha256'], 'actual': digest(g5cp)},
        'g5_summary': {'recorded': g6cp['parent_summary_sha256'], 'actual': digest(g5cp.parent / 'summary.json')},
    }
    source_files = json.loads((output / 'provenance.json').read_text())['source_hashes']
    source_checks = {rel: {'recorded': expected, 'actual': digest(ROOT / rel), 'matches': digest(ROOT / rel) == expected}
                     for rel, expected in source_files.items()}
    snapshot_checks = {rel: {'recorded': expected,
                             'actual': digest(output / 'source_snapshot' / rel),
                             'matches': digest(output / 'source_snapshot' / rel) == expected}
                       for rel, expected in source_files.items()}
    audit_only_live_change = 'experiments/distortion_model/stage_04_center_and_dm0_calibration/scripts/audit_g7_bounded.py'
    fitting_paths = [rel for rel in source_files if
        rel.startswith('distortion_model/') or
        rel.startswith('experiments/distortion_model/stage_03_p4_reference') or
        rel.startswith('experiments/distortion_model/stage_03_p4_baseline_and_deformation/') or
        rel in ('experiments/distortion_model/stage_04_center_and_dm0_calibration/scripts/run.py',
                'experiments/distortion_model/stage_04_center_and_dm0_calibration/scripts/run_joint.py',
                'experiments/distortion_model/stage_04_center_and_dm0_calibration/scripts/run_joint_bounded.py',
                'experiments/distortion_model/stage_04_center_and_dm0_calibration/scripts/continue_g7_repaired.py',
                'requirements-stage7.txt')]
    start_details = {}
    start_labels = [v['label'] for v in summary.get('all_start_outcomes', [])]
    if not start_labels:
        start_labels = [summary['selected_start']]
    if len(start_labels) != len(set(start_labels)):
        raise ValueError('duplicate start labels in saved summary')
    for start in start_labels:
        history = json.loads((output / start / 'history.json').read_text())
        accepted = [v for v in history if v.get('accepted')]
        rejected = [v for v in history if not v.get('accepted')]
        outer_joint = [v for v in accepted if v['block'] == 'joint' and not v.get('observed_curvature')]
        polish = [v for v in history if v.get('polish') is not None]
        start_details[start] = {
            'history_events': len(history), 'accepted': len(accepted), 'rejected': len(rejected),
            'accepted_by_block': {b: sum(v['block'] == b for v in accepted) for b in ('A','theta','P1','baseline','K4','D','joint')},
            'rejected_by_block': {b: sum(v['block'] == b for v in rejected) for b in ('A','theta','P1','baseline','K4','D','joint')},
            'outer_joint_accepted': len(outer_joint),
            'max_joint_trust_cap': max((v['outcome'].get('max_frame_trust_cap', v['outcome'].get('trust_cap', 0.)) for v in outer_joint), default=None),
            'joint_accepted_damping_median': float(np.median([v['outcome']['damping'] for v in outer_joint])) if outer_joint else None,
            'joint_accepted_damping_max': max((v['outcome']['damping'] for v in outer_joint), default=None),
            'joint_rejection_counts': {reason:sum(v.get('rejection_counts',{}).get(reason,0) for v in history if v.get('block')=='joint') for reason in sorted({reason for v in history if v.get('block')=='joint' for reason in v.get('rejection_counts',{})})},
            'polish_events': [{'iteration': v['polish'], 'accepted': v['accepted'], 'reason': v.get('reason')}
                              for v in polish],
            'cost_start': history[0]['cost_before'] if history else None,
            'cost_last_accepted': next((v['outcome']['cost_after'] for v in reversed(history) if v.get('accepted')), None),
            'last_event': history[-1] if history else None,
        }

    # Native P4-local angle supplement, using only selected immutable globals.
    theta = np.linspace(*spec.theta_bounds, 401)
    xi4 = theta - spec.omega4
    fig, ax = plt.subplots(figsize=(8, 5))
    for accommodation in (0., 2., 4., 6.):
        f4, mu4, ok = p4_reference(theta, np.full_like(theta, accommodation), params)
        centered = f4 - mu4[:, None, :]
        rms = np.sqrt(np.mean(centered**2, axis=(1, 2)))
        ax.plot(xi4[ok], rms[ok], label=f'A={accommodation:g} D')
    ax.set(xlabel=r'P4 local angle $\xi_4=\theta-\omega_4$ (degrees)',
           ylabel='Centered P4 coordinate RMS (reference px)',
           title='Selected G7 snapshot response over valid visual-theta domain')
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / 'response_curves_native_xi4.png', dpi=150)
    plt.close(fig)

    # Verify the immutable repair input manifest and the exact compatible warm
    # start used by this continuation. These checks are mechanical; they do
    # not change the checkpoint or its scientific status.
    repair_dir = STAGE / 'results/g7_repair_01'
    repair_provenance = json.loads((repair_dir / 'provenance.json').read_text())
    repair_input_checks = {}
    for rel, expected in repair_provenance.get('input_sha256', {}).items():
        actual_path = STAGE / 'results/g7_attempt_02' / rel
        actual = digest(actual_path) if actual_path.is_file() else None
        repair_input_checks[rel] = {'expected': expected, 'actual': actual, 'matches': actual == expected}
    continuation_config = json.loads((output / 'config.json').read_text())
    warm_attempt = ROOT / continuation_config['compatible_warm_start_attempt']
    warm_label = continuation_config['compatible_warm_start_selected']
    warm_solution_path = warm_attempt / warm_label / 'solution.npz'
    continuation_initial_path = output / 'continued/initial.npz'
    with np.load(warm_solution_path, allow_pickle=False) as old_warm, np.load(continuation_initial_path, allow_pickle=False) as new_warm:
        warm_state_equal = bool(np.array_equal(old_warm['states'], new_warm['states']))
        warm_globals_equal = bool(np.array_equal(old_warm['scaled_globals'], new_warm['scaled_globals']))
    warm_start_integrity = {
        'attempt': continuation_config['compatible_warm_start_attempt'],
        'selected_start': warm_label,
        'recorded_solution_sha256': continuation_config['compatible_warm_start_solution_sha256'],
        'actual_solution_sha256': digest(warm_solution_path),
        'solution_sha256_matches': digest(warm_solution_path) == continuation_config['compatible_warm_start_solution_sha256'],
        'states_bitwise_equal_to_continuation_initial': warm_state_equal,
        'globals_bitwise_equal_to_continuation_initial': warm_globals_equal,
    }

    report = {
        'source': 'mechanical_inventory', 'method': 'reconstruct saved selected G7 state with same objective implementation; no refit',
        'audit_command': 'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python experiments/distortion_model/stage_04_center_and_dm0_calibration/scripts/audit_g7_bounded.py --attempt experiments/distortion_model/stage_04_center_and_dm0_calibration/results/g7_attempt_02',
        'audit_created_utc': datetime.now(timezone.utc).isoformat(),
        'postfit_audit_script_sha256': digest(Path(__file__).resolve()),
        'postfit_aggregation_script_sha256': digest(STAGE / 'scripts/summarize_g7_compact.py'),
        'audit_schema_adaptation': 'After fitting, audit start labels were generalized to read distinct labels from the saved summary; no fitting source or result was changed.',
        'selected_start': label, 'attempt_summary_sha256': digest(output / 'summary.json'), 'population': {'scheduled': len(valid), 'complete_valid': int(valid.sum()),
                                                'unavailable': int((~valid).sum()), 'evaluated': len(x),
                                                'counts_by_exposure': [int(np.sum(e == k)) for k in range(20)]},
        'objective_reconstruction': {'components': objective_components, 'saved_fit_components': fit['components'],
                                     'differences': component_diff, 'prediction_arrays': fit_comparison,
                                     'independent_numpy_forward_and_loss': independent_forward_check,
                                     'native_relative_coordinate_rms_by_exposure': rms_by_exposure},
        'physical_constraint_reconstruction': physical_certificate,
        'curvature': {'method': 'finite differences of analytic gradients; state coordinates active exactly per JointDM0.active; free local state subblocks only',
                      'negative_free_local_eigenvalue_count': total_negative,
                      'free_local_eigenvalue_count': total_free_eigenvalues,
                      'minimum_free_local_eigenvalue': None if not np.isfinite(global_min) else global_min,
                      'maximum_free_local_eigenvalue': None if not np.isfinite(global_max) else global_max,
                      'by_exposure': exposure_counts,
                      'profile_rank_interpretation': ('Observed constrained profile rank/curvature are reconstructed for this saved fit; any serialized zero after a local-curvature failure is a NOT_EVALUATED sentinel, not evidence of global rank deficiency.')},
        'starts': start_details,
        'compact_schedule': {'rows': len(schedule), 'slots_per_compact_record': 300,
                             'schedule_matches_saved_indices': bool(np.array_equal(schedule, schedule_saved)),
                             'semantic_hash': array_hash({'scheduled_population_index': schedule}),
                             'saved_schedule_semantic_hash': schedule_hash_saved,
                             'records': compact_records,
                             'input_masking_contract': 'infer_retained accepts only N x 8 retained coordinates and the 8x8 retained covariance marginal; held-point score is read after state/branch solve.'},
        'parent_hashes': parent_hashes,
        'repair01_input_hashes': {'count': len(repair_input_checks),
                                  'all_match': all(v['matches'] for v in repair_input_checks.values()),
                                  'checks': repair_input_checks},
        'compatible_warm_start_integrity': warm_start_integrity,
        'source_hashes_match': all(v['matches'] for v in source_checks.values()),
        'source_hash_checks': source_checks,
        'source_snapshot_hashes_match': all(v['matches'] for v in snapshot_checks.values()),
        'source_snapshot_hash_checks': snapshot_checks,
        'source_change_classification': {
            'postrun_live_only_changes': [k for k, v in source_checks.items() if not v['matches']],
            'expected_postrun_live_only_changes': [audit_only_live_change,
                'docs/stages/04_CENTER_AND_DM0_CALIBRATION.md'],
            'fitting_code_paths_checked': fitting_paths,
            'fitting_code_current_hashes_match': all(source_checks[k]['matches'] for k in fitting_paths),
            'all_archived_bytes_match_original_provenance': all(v['matches'] for v in snapshot_checks.values()),
            'interpretation': 'Post-run changes are limited to the audit-only start-label adaptation and current stage-authority status text. The source snapshot still matches every archived hash, and fitting source hashes match runtime provenance.'},
        'review_state': {'status': summary.get('status'), 'G7_decision': summary.get('G7_decision'),
                         'fit_complete': bool(checkpoint.get('fit_complete')),
                         'fit_certified': bool(checkpoint.get('fit_certified')),
                         'crosscheck_complete': bool(checkpoint.get('crosscheck_complete')),
                         'comparison_complete': bool(checkpoint.get('comparison_complete')),
                         'G8_authorized': False},
        'native_local_angle_response_plot': 'response_curves_native_xi4.png',
    }
    (output / 'mechanical_audit.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'selected_start':label,'components':objective_components,'component_differences':component_diff,
                      'negative_free_local_eigenvalue_count':total_negative,
                      'minimum_free_local_eigenvalue':report['curvature']['minimum_free_local_eigenvalue'],
                      'by_exposure':exposure_counts,'compact_records':compact_records,
                      'source_hashes_match':report['source_hashes_match'],'parent_hashes':parent_hashes},indent=2))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--attempt', type=Path, required=True)
    main(parser.parse_args().attempt)
