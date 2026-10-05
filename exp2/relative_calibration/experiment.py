"""Isolated curvature/frozen-gaze-prior calibration and relative quality reports."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP))
from observations import relative_measurements, RELATIVE_SCHEMA, CROSSED_TRANSFORM
from calibrate_profiled import ProfiledProblem, KNOTS, basis, load_data, save
from calibrate_continuation import atomic_json, array_hash, make_manifest, read_checkpoint, continue_fit
from estimate_profiled import invert_batch, physical_optimality, forward_jac

spec = importlib.util.spec_from_file_location('reduced_protocol', EXP/'reduced_calibration/experiment.py')
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)
EXPERIMENT = 'relative_curvature_frozen_mean_v1'
FOLDS = {'holdout3': list(range(10, 15)), 'holdout2': list(range(15, 20)),
         'full_constrained': [], 'full_flexible': []}
PRIOR_RULE = legacy.PRIOR_RULE


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def coefficient_map(fold, curvature=False):
    active = KNOTS if fold == 'full_flexible' else np.delete(KNOTS, 1 if fold == 'holdout2' else 2)
    E = legacy.coefficient_map(active)
    if curvature:
        expanded = np.zeros((15, E.shape[1]+1))
        expanded[:14, :-1], expanded[14, -1] = E, 1.
        E = expanded
    return E


def training_base(y, frames, groups, meta, fold):
    """All20 initial nominal means, then Euclidean projection for full constrained.

    Balanced five-gaze support makes projection of the per-knot line fits agree
    with the constrained equal-fixation-mean displacement least-squares fit.
    Frame noise weighting is deliberately absent from this initial prior policy.
    """
    old_fold = fold if fold.startswith('holdout') else 'full'
    base, initial, selected, global_groups, covariance, detail = legacy.build_training(
        y, frames, groups, meta, old_fold)
    E = coefficient_map(fold)
    coef = E@np.linalg.lstsq(E, base.initial_coef, rcond=None)[0]
    theta, _, A, _ = base.decode(initial)
    detail.update(coefficient_map=E.tolist(), free_coefficient_count=E.shape[1],
        initial_function_policy='training ordinary fixation means; per-demand displacement lines; '
            'Euclidean coefficient projection onto declared map; ratio fit on all training means',
        initial_unprojected_coefficients=base.initial_coef.tolist(),
        initial_projected_coefficients=coef.tolist())
    return base, theta, A, coef, selected, global_groups, covariance, detail


def previous_source(directory, base, selected, global_groups, provenance):
    """Verify strict training identity before opening source states as information."""
    directory = Path(directory)
    with np.load(directory/'checkpoint.npz', allow_pickle=False) as data:
        identity = json.loads(str(data['manifest_json']))
        states = data['states'].copy()
    required = dict(training_fixations=selected, states_shape=[base.n, 2],
        frame_hash=array_hash(base.frames), group_hash=array_hash(base.groups),
        global_group_hash=array_hash(global_groups), observation_hash=array_hash(base.y),
        source_sha256=provenance['source_sha256'], interval_sha256=provenance['selected_interval_sha256'],
        validity_policy=provenance['validity_policy'], targets=base.targets.tolist(), demands=base.demands.tolist(),
        capture5_used=False, knots=KNOTS.tolist())
    mismatch = [key for key, value in required.items() if identity.get(key) != value]
    if mismatch:
        raise ValueError('Previous source violates exact training support: '+', '.join(mismatch))
    model = json.loads((directory/'model.json').read_text())
    expected_schema = ('profiled_curved_displacement_power_ratio_v2' if
        len(identity['initial_coefficients'])==15 else 'profiled_piecewise_displacement_power_ratio_v1')
    if model.get('schema') != expected_schema or identity.get('schema') != expected_schema:
        raise ValueError('Unsupported previous model schema')
    if identity['encoding'].get('theta_scale') != 15. or not 0 < identity['p'] <= 1:
        raise ValueError('Unsupported previous source state units/exponent')
    model_prov = model['diagnostics']['provenance']
    if (model_prov.get('training_fixations') != selected or
            model_prov.get('source_sha256') != provenance['source_sha256'] or
            model_prov.get('selected_interval_sha256') != provenance['selected_interval_sha256'] or
            model.get('p') != identity['p'] or
            not np.array_equal(model['coefficients'], data_coefficients(directory))):
        raise ValueError('Previous model/checkpoint provenance mismatch')
    for model_key, manifest_key in [('initial_coefficients', 'initial_coefficients'),
            ('precision', 'precision'), ('prior_precision', 'prior_precision'),
            ('coefficient_map', 'coefficient_map')]:
        if model.get(model_key) != identity.get(manifest_key):
            raise ValueError('Previous model identity differs from checkpoint: '+model_key)
    source_prior = identity.get('previous_mean_prior') or {}
    source_problem = ProfiledProblem(base.y, base.frames, base.groups, base.targets, base.demands,
        identity['p'], np.asarray(identity['initial_coefficients']), np.asarray(identity['precision']),
        np.asarray(identity['prior_precision']), coefficient_map=identity.get('coefficient_map'),
        curvature=len(identity['initial_coefficients'])==15,
        curvature_strength=identity.get('curvature_strength', 0.),
        curvature_scale_output=identity.get('curvature_scale_output', 1.),
        previous_means=source_prior.get('means_deg'), previous_mean_strength=source_prior.get('strength', 0.),
        previous_mean_scale_deg=source_prior.get('scale_deg', 1.), previous_mean_provenance=source_prior.get('source'))
    if (array_hash(source_problem.prior_M) != identity['prior_matrix_hash'] or
            array_hash(source_problem.prior_v) != identity['prior_value_hash']):
        raise ValueError('Previous source initial-function prior reconstruction failed')
    rebuilt_identity = make_manifest(source_problem, model_prov, global_groups, identity['kappa'])
    differing = [key for key, value in rebuilt_identity.items() if identity.get(key) != value]
    if differing:
        raise ValueError('Previous objective manifest reconstruction failed: '+', '.join(differing))
    read_checkpoint(directory, identity, source_problem, global_groups, False)
    if states.shape != (2*base.n,) or not np.isfinite(states).all():
        raise ValueError('Invalid previous source states')
    encoded = states.reshape(-1, 2)
    theta = 15*encoded[:, 0]
    A = (identity['encoding']['power_scale']*encoded[:, 1])**(1/identity['p'])
    if not np.isfinite(A).all() or np.any(abs(theta)>20) or np.any((A<0)|(A>6)):
        raise ValueError('Previous physical states violate bounds')
    means = np.bincount(base.groups, weights=theta)/base.counts
    frozen = dict(source_directory=str(directory.resolve()), model_sha256=sha(directory/'model.json'),
        checkpoint_sha256=sha(directory/'checkpoint.npz'), training_fixations=selected,
        source_sha256=provenance['source_sha256'], interval_sha256=provenance['selected_interval_sha256'],
        frame_hash=required['frame_hash'], group_hash=required['group_hash'],
        global_group_hash=required['global_group_hash'], observation_hash=required['observation_hash'],
        means_deg=means.tolist(), means_hash=array_hash(means), units='degrees',
        center_policy='frozen ordinary fitted gaze mean on exact retained training frame support',
        interpretation='model-derived regularization, not independent state measurements',
        source_fold=identity.get('fold'), source_experiment=identity.get('experiment'))
    return theta, A, means, frozen


def data_coefficients(directory):
    with np.load(Path(directory)/'checkpoint.npz', allow_pickle=False) as data:
        return data['coefficients'].copy()


def build_training(y, frames, groups, meta, fold, curvature=False, curvature_strength=.1,
                   previous_mean_strength=0., previous_mean_scale_deg=1., previous_dir=None, provenance=None):
    base, theta, A, coef, selected, global_groups, covariance, detail = training_base(y, frames, groups, meta, fold)
    means, source = None, None
    if previous_dir is not None:
        theta, A, means, source = previous_source(previous_dir, base, selected, global_groups, provenance)
    if previous_mean_strength > 0 and previous_dir is None:
        raise ValueError('Previous mean penalty requires a verified same-training source')
    if curvature: coef = np.r_[coef, 0.]
    range_d = float(np.ptp(base.y[:, 0]))
    problem = ProfiledProblem(base.y, base.frames, base.groups, base.targets, base.demands, base.p,
        coef, base.W, base.prior_W, coefficient_map=coefficient_map(fold, curvature),
        curvature=curvature, curvature_strength=curvature_strength if curvature else 0.,
        curvature_scale_output=range_d, previous_means=means,
        previous_mean_strength=previous_mean_strength, previous_mean_scale_deg=previous_mean_scale_deg,
        previous_mean_provenance=source)
    detail.update(curvature=curvature, curvature_strength=problem.curvature_strength,
        curvature_scale_output=range_d, curvature_scale_policy='ptp(retained training frame d)',
        previous_mean_strength=previous_mean_strength, previous_mean_scale_deg=previous_mean_scale_deg,
        previous_source=source, coefficient_map=problem.coefficient_map.tolist(),
        free_coefficient_count=problem.free_coefficient_count,
        training_demand_levels=sorted(set(problem.demands.tolist())),
        independent_displacement_knots=(KNOTS if fold=='full_flexible' else
            np.delete(KNOTS, 1 if fold=='holdout2' else 2)).tolist())
    return problem, problem.encode(theta, A), selected, global_groups, covariance, detail


def train(args):
    y, frames, groups, meta, records, provenance = load_data(args.experiment_dir, args.intervals)
    problem, initial, selected, global_groups, covariance, detail = build_training(
        y, frames, groups, meta, args.fold, args.curvature, args.curvature_strength,
        args.previous_mean_strength, args.previous_mean_scale_deg, args.previous_dir, provenance)
    heldout = FOLDS[args.fold]
    provenance.update(training_fixations=selected, heldout_fixation=None, heldout_fixations=heldout,
        fold=args.fold, experiment=EXPERIMENT, training_frame_count=problem.n,
        heldout_frame_count=int(np.isin(groups, heldout).sum()),
        full_seed_used=bool(args.previous_dir and selected==list(range(20))),
        historical_reference_used=False, fixed_p=problem.p, initialization=detail,
        prior_rule=PRIOR_RULE, nominal_anchor_policy='retained strengths 1; separate previous-mean strength',
        initialization_mode='explicit new-objective physical-state warm start' if args.previous_dir else 'training nominal means',
        continuation_source=str(args.resume_dir) if args.resume_dir else None,
        measurement_basis='normalized [d,rho4]; relative pixels are quality diagnostics')
    meta = [dict(row, heldout=j in heldout) for j, row in enumerate(meta)]
    records = [r for r in records if not any(meta[j]['capture'] == f"capture_{r['capture']}_detections.pkl" for j in heldout)]
    if args.output_dir.exists():
        raise ValueError('Use a new output directory; preserved artifacts cannot be overwritten')
    args.output_dir.mkdir(parents=True)
    atomic_json(args.output_dir/'provenance.json', provenance)
    (args.output_dir/'selected_intervals.json').write_bytes(args.intervals.read_bytes())
    stages = [(args.loss, args.kappa if args.loss == 'robust' else None)]
    x = initial
    for name, kappa in stages:
        manifest = make_manifest(problem, provenance, global_groups, kappa)
        manifest.update(experiment=EXPERIMENT, fold=args.fold, heldout_fixations=heldout, prior_rule=PRIOR_RULE,
            initialization=detail, nominal_anchor_policy=provenance['nominal_anchor_policy'],
            training_demand_levels=detail['training_demand_levels'],
            independent_displacement_knots=detail['independent_displacement_knots'])
        if args.resume_dir:
            x = read_checkpoint(args.resume_dir, manifest, problem, global_groups, False)[0]
        output = args.output_dir/name
        output.mkdir()
        atomic_json(output/'run_provenance.json', provenance)
        x, coef, history, converged, status = continue_fit(problem, x, output, manifest, kappa=kappa,
            max_nfev=args.max_nfev, wall_seconds=args.wall_seconds, lsmr_maxiter=args.lsmr_maxiter,
            lsmr_atol=args.lsmr_atol, lsmr_btol=args.lsmr_btol, robust_outer=args.robust_outer,
            robust_max_nfev=args.robust_max_nfev, physical_gtol=args.physical_gtol,
            physical_step_tol=args.physical_step_tol, relative_cost_tol=args.relative_cost_tol, weight_tol=1e-7)
        save(output, problem, initial, x, coef, history, converged, meta, records, selected,
             provenance, covariance, optical_loss='quadratic' if kappa is None else 'block soft-L1')
        model = json.loads((output/'model.json').read_text())
        model.update(training_demand_levels=detail['training_demand_levels'],
            independent_displacement_knots=detail['independent_displacement_knots'])
        model['diagnostics'].update(termination=status['termination'], continuation_status=status,
            checkpoint_manifest=manifest, objective_components=history[-1],
            previous_mean_prior=manifest.get('previous_mean_prior'))
        atomic_json(output/'model.json', model)
        atomic_json(output/'diagnostics.json', model['diagnostics'])
        print(json.dumps(dict(fold=args.fold, stage=name, status=status)), flush=True)


def state_diagnostics(states, observation, coef, p, W, active_knots=None):
    theta, A = states.T
    H, dt, da = basis(theta, A**p, p, curvature=len(coef)==15)
    prediction = H@coef
    stationarity, knots = physical_optimality(states, observation, coef, p, W)
    active = KNOTS if active_knots is None else np.asarray(active_knots)
    nonsmooth = np.zeros(len(A), bool)
    for knot in active[1:-1]: nonsmooth |= abs(A-knot) <= 16*np.finfo(float).eps*max(1., knot)
    defined = (A > 0) | (p == 1)
    singular = np.full((len(A), 2), np.nan)
    J = np.stack([dt@coef, da@coef], axis=2)[defined]
    J[:, :, 1] *= (p*A[defined]**(p-1))[:, None]
    singular[defined] = np.linalg.svd(np.einsum('ab,nbc->nac', np.linalg.cholesky(W).T, J)*np.array([1., .25]), compute_uv=False)
    return prediction, dict(physical_projected_stationarity=stationarity, stationarity_verified=stationarity <= 1e-5,
        condition=singular[:, 0]/np.maximum(singular[:, 1], 1e-14), minimum_scaled_singular_value=singular[:, 1],
        physical_derivative_defined=defined, interior_protocol_knot=knots, interior_knot_nondifferentiable=nonsmooth,
        theta_bound=abs(theta)>=20-1e-5, A_bound=(A<=1e-5)|(A>=6-1e-5),
        extrapolation=(A<KNOTS[0])|(A>KNOTS[-1]))


def grid_audit(coef, p, W, active_knots):
    # Reuse the maintained sampling/branch policy with a local generalized basis.
    original_basis = legacy.basis
    legacy.basis = lambda *a, **kw: basis(*a, **kw, curvature=len(coef)==15)
    try: return legacy.grid_audit(coef, p, W, active_knots)
    finally: legacy.basis = original_basis


def plot_trajectory(axis, rows, field, label):
    """Show only observed contiguous support, preserving original-frame gaps."""
    frames, values, previous = [], [], None
    for row in rows:
        frame = int(row['frame_index'])
        if previous is not None and frame-previous != 1:
            frames.append(np.nan); values.append(np.nan)
        frames.append(frame); values.append(float(row[field])); previous = frame
    axis.plot(frames, values, lw=.5, label=label)


def fixation_mean_deviations(statistics, fixations):
    """Equal-fixation descriptive mean deviations, separate from frame RMSE."""
    return dict(fixation_count=len(fixations), weighting='equal fixation weight on exact common frame support',
        **{field:dict(rmse=float(np.sqrt(np.mean([statistics[str(j)][field]['bias']**2 for j in fixations]))),
            mean_absolute_bias=float(np.mean([abs(statistics[str(j)][field]['bias']) for j in fixations])),
            mean_bias=float(np.mean([statistics[str(j)][field]['bias'] for j in fixations])),
            per_fixation_mean_delta={str(j):statistics[str(j)][field]['bias'] for j in fixations})
            for field in ['theta_deg','A_diopters']})


def estimate(args):
    model = json.loads((args.model_dir/'model.json').read_text())
    prov = model['diagnostics']['provenance']
    reference = getattr(args,'reference',False)
    expected_experiment = 'hard_constrained_reduced_demands_v1' if reference else EXPERIMENT
    if (prov.get('experiment') != expected_experiment or prov['fold'] != args.fold or
            (reference and (args.fold!='holdout3' or model.get('schema')!='profiled_piecewise_displacement_power_ratio_v1' or
                args.model_dir.resolve()!=(EXP/'reduced_calibration/training/holdout3/robust').resolve()))):
        raise ValueError('Wrong model experiment/family')
    y, frames, groups, meta, records, current = load_data(args.experiment_dir, args.intervals)
    if current['source_sha256'] != prov['source_sha256'] or current['selected_interval_sha256'] != prov['selected_interval_sha256']:
        raise ValueError('Prediction sources differ from frozen training')
    if reference:
        base,_,selected_training,global_training,_,_=legacy.build_training(y,frames,groups,meta,'holdout3')
        previous_source(args.model_dir,base,selected_training,global_training,current)
    selected = FOLDS[args.fold] if args.scope == 'heldout' else list(range(20))
    if not selected: raise ValueError('This cohort has no heldout support; use --scope all')
    mask = np.isin(groups, selected)
    obs, frame, group = y[mask], frames[mask], groups[mask]
    if args.output_dir.exists(): raise ValueError('Use a new prediction directory')
    args.output_dir.mkdir(parents=True)
    p, W = model['p'], np.asarray(model['precision'])
    active = np.delete(KNOTS, 1 if args.fold == 'holdout2' else 2) if args.fold != 'full_flexible' else KNOTS
    E = coefficient_map(args.fold, len(model['coefficients'])==15)
    if not np.array_equal(E, model['coefficient_map']): raise ValueError('Unexpected exported coefficient map')
    manifests = {}
    model_keys = [('trained','coefficients')] if reference else [('trained','coefficients'),('initial','initial_coefficients')]
    for name, key in model_keys:
        coef = np.asarray(model[key])
        if not np.allclose(E@np.linalg.lstsq(E, coef, rcond=None)[0], coef, atol=1e-12, rtol=1e-12):
            raise ValueError('Exported coefficients violate map')
        rows = []
        for start in range(0, len(obs), args.batch_size):
            stop = min(start+args.batch_size, len(obs))
            states, inverse = invert_batch(obs[start:stop], coef, p, W, args.inverse_iterations)
            prediction, physical = state_diagnostics(states, obs[start:stop], coef, p, W, active)
            for k in range(len(states)):
                i = start+k
                row = dict(frame_index=int(frame[i]), fixation_index=int(group[i]),
                    training_support=bool(group[i] in prov['training_fixations']),
                    theta_deg=float(states[k,0]), A_diopters=float(states[k,1]),
                    d=float(obs[i,0]), rho4=float(obs[i,1]), predicted_d=float(prediction[k,0]),
                    predicted_rho4=float(prediction[k,1]), residual_d=float(prediction[k,0]-obs[i,0]),
                    residual_rho4=float(prediction[k,1]-obs[i,1]))
                row.update(observed_u1=float(obs[i,0]-(obs[i,1]+1)/2),
                    observed_u2=float(obs[i,0]+(obs[i,1]+1)/2),
                    predicted_u1=float(prediction[k,0]-(prediction[k,1]+1)/2),
                    predicted_u2=float(prediction[k,0]+(prediction[k,1]+1)/2),
                    residual_u1=float((prediction[k,0]-obs[i,0])-(prediction[k,1]-obs[i,1])/2),
                    residual_u2=float((prediction[k,0]-obs[i,0])+(prediction[k,1]-obs[i,1])/2))
                for fields in [inverse, physical]:
                    row.update({field: bool(v[k]) if v.dtype.kind=='b' else int(v[k]) if v.dtype.kind in 'iu' else float(v[k]) for field,v in fields.items()})
                rows.append(row)
            print(json.dumps(dict(model=name, completed_frames=stop, total=len(obs))), flush=True)
        path = args.output_dir/(name+'_predictions.csv')
        legacy.write_csv(path, rows)
        manifests[name] = dict(prediction_sha256=sha(path), frame_count=len(rows), frame_hash=array_hash(frame),
            group_hash=array_hash(group), observation_hash=array_hash(obs))
        atomic_json(args.output_dir/(name+'_grid_audit.json'), legacy.json_safe(grid_audit(coef, p, W, active)))
    atomic_json(args.output_dir/'prediction_manifest.json', dict(experiment=expected_experiment, fold=args.fold,
        scope=args.scope, evaluated_fixations=selected, training_fixations=prov['training_fixations'],
        model_dir=str(args.model_dir.resolve()), model_sha256=sha(args.model_dir/'model.json'), models=manifests,
        source_sha256=current['source_sha256'], selected_interval_sha256=current['selected_interval_sha256'],
        training_converged=model['diagnostics']['converged'], training_status=model['diagnostics'].get('continuation_status'),
        batch_size=args.batch_size, inverse_iterations=args.inverse_iterations, baseline_read=False,
        inference='independent bounded optical per-frame inverse; no fixation, target, demand or temporal anchors',
        crossed_pair_diagnostics='u1=d-(rho4+1)/2; u2=d+(rho4+1)/2; exact equivalent normalized basis, no additional measurement or accuracy test',
        reference_role='primary frozen H0 anchor-free inverse' if reference else 'candidate',
        interpretation='all support includes in-sample data for full cohorts; model-state comparison is not physiological validation'))


def estimate_reference(args):
    args.reference = True
    args.fold = 'holdout3'
    args.scope = 'all'
    args.model_dir = EXP/'reduced_calibration/training/holdout3/robust'
    estimate(args)


def measurements_report(args):
    y, frames, groups, meta, records, provenance = load_data(args.experiment_dir, args.intervals)
    if args.output_dir.exists(): raise ValueError('Use a new measurement report directory')
    args.output_dir.mkdir(parents=True)
    selected = [j for j in range(20) if j not in FOLDS[args.fold]]
    rows, ordering = [], []
    for record in records:
        result = relative_measurements(record['arrays'])
        mask = record['core'] & record['valid'] & np.isin(record['groups'], selected)
        for i in np.flatnonzero(mask):
            z = result['z'][i]; crossed = result['crossed'][i]
            rows.append(dict(frame_index=int(record['arrays']['frame_index'][i]), fixation_index=int(record['groups'][i]),
                m=float(z[0]), S1=float(z[1]), S4=float(z[2]), e1=float(crossed[0]), e2=float(crossed[1]),
                d=float(record['y'][i,0]), rho4=float(record['y'][i,1]), ordered=bool(result['ordering_valid'][i])))
        ordering.append(dict(capture=record['capture'], retained_training_count=int(mask.sum()),
            retained_ordering_failures=int((mask & result['ordering_failure']).sum()),
            all_measurement_ordering_failures=int(result['ordering_failure'].sum())))
    legacy.write_csv(args.output_dir/'training_relative_measurements.csv', rows)
    ordered_rows = [r for r in rows if r['ordered']]
    summary = []
    for j in selected:
        local = [r for r in ordered_rows if r['fixation_index']==j]
        if not local: continue
        summary.append(dict(fixation_index=j, nominal_theta_deg=meta[j]['target_theta_deg'],
            nominal_demand_D=meta[j]['demand_diopters_label'], count=len(local),
            S1_mean=float(np.mean([r['S1'] for r in local])), S1_std=float(np.std([r['S1'] for r in local])),
            S1_range=float(np.ptp([r['S1'] for r in local])),
            S4_mean=float(np.mean([r['S4'] for r in local])), m_mean=float(np.mean([r['m'] for r in local]))))
    values = np.array([[r['m'],r['S1'],r['S4']] for r in ordered_rows])
    crossed = np.array([[r['e1'],r['e2'],r['S4']] for r in ordered_rows])
    normalized = np.array([[r['d'],r['rho4']] for r in ordered_rows])
    covariance = legacy.noise_covariance(values, np.array([r['frame_index'] for r in ordered_rows]), np.array([r['fixation_index'] for r in ordered_rows]))
    report = dict(experiment=EXPERIMENT, fold=args.fold, training_fixations=selected, provenance=provenance,
        observation_schema=RELATIVE_SCHEMA, ordering=ordering, fixations=summary,
        retained_frame_count=len(rows), ordered_branch_frame_count=len(ordered_rows), relative_measurements_sha256=sha(args.output_dir/'training_relative_measurements.csv'),
        identity_max_error=float(np.max(abs(crossed-values@CROSSED_TRANSFORM.T))),
        normalized_max_error=float(np.max(abs(normalized-values[:,[0,2]]/values[:,1,None]))),
        relative_covariance=covariance.tolist(), covariance_basis=RELATIVE_SCHEMA['basis'],
        covariance_interpretation='training consecutive differences; actual motion may contaminate localization uncertainty',
        scale_decision='retain normalized estimator; S1 is a quality diagnostic pending evidence',
        interpretation='nominal labels are assumptions; neither S1 invariance nor a head-depth-only signal is established')
    atomic_json(args.output_dir/'measurement_report.json', report)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for demand in sorted(set(r['nominal_demand_D'] for r in summary)):
        local = [r for r in summary if r['nominal_demand_D']==demand]
        ax[0].errorbar([r['nominal_theta_deg'] for r in local],[r['S1_mean'] for r in local],
            yerr=[r['S1_std'] for r in local],label=f'{demand:.3g} D',marker='o')
    ax[0].set(xlabel='Nominal gaze (degrees)',ylabel='S1 mean ± within fixation SD (pixels)');ax[0].legend()
    ax[1].plot([r['S1'] for r in rows],lw=.3);ax[1].set(xlabel='Retained training row',ylabel='S1 (pixels)')
    fig.savefig(args.output_dir/'training_separation.png',dpi=140);plt.close(fig)


def compare(args):
    frozen = json.loads((args.prediction_dir/'prediction_manifest.json').read_text())
    if sha(Path(frozen['model_dir'])/'model.json') != frozen['model_sha256']: raise ValueError('Model changed after prediction freeze')
    selected = frozen['evaluated_fixations']
    # Reference access happens only after frozen candidate validation.
    predictions = {}
    for name in frozen['models']:
        path = args.prediction_dir/(name+'_predictions.csv')
        if sha(path) != frozen['models'][name]['prediction_sha256']: raise ValueError('Prediction changed after freeze')
        predictions[name] = legacy.read_csv(path)
        if len(predictions[name]) != frozen['models'][name]['frame_count']: raise ValueError('Prediction frame count mismatch')
        legacy.exact_join(predictions[name], predictions[name], selected)
    references = {}
    for label, directory in [('holdout3',args.holdout_dir),('matched_full',args.matched_dir)]:
        model = json.loads((directory/'model.json').read_text())
        prov = model['diagnostics']['provenance']
        if prov['source_sha256'] != frozen['source_sha256'] or prov['selected_interval_sha256'] != frozen['selected_interval_sha256']:
            raise ValueError('Reference source identity differs')
        rows = []
        reference_detail = dict(estimate_kind='joint calibration training states')
        if label == 'holdout3':
            reference_dir = args.holdout_prediction_dir or EXP/'reduced_calibration/predictions/holdout3'
            old_predictions = reference_dir/'trained_predictions.csv'
            old_manifest = json.loads((old_predictions.parent/'prediction_manifest.json').read_text())
            if (old_manifest['model_sha256'] != sha(directory/'model.json') or
                    old_manifest['models']['trained']['prediction_sha256'] != sha(old_predictions) or
                    old_manifest['source_sha256']!=frozen['source_sha256'] or
                    old_manifest['selected_interval_sha256']!=frozen['selected_interval_sha256']):
                raise ValueError('Holdout reference prediction identity mismatch')
            rows.extend(legacy.read_csv(old_predictions))
            reference_detail=dict(estimate_kind='anchor-free per-frame optical inverse',
                reference_role='primary H0 frozen holdout3 model',scope=old_manifest.get('scope','heldout'),
                prediction_sha256=sha(old_predictions),prediction_manifest_sha256=sha(reference_dir/'prediction_manifest.json'))
        else:
            for path in sorted(directory.glob('capture_*_states.csv')): rows.extend(legacy.read_csv(path))
            reference_detail['reference_role']='secondary matched full model'
        available = sorted(set(int(r['fixation_index']) for r in rows if r.get('theta_deg') not in ('',None) and int(r['fixation_index']) in selected))
        references[label] = (rows, available, dict(model_sha256=sha(directory/'model.json'),
            converged=model['diagnostics']['converged'], training_fixations=prov['training_fixations'],**reference_detail))
    summary = {}
    import matplotlib.pyplot as plt
    for name, rows in predictions.items():
        entries = {}
        for label, (reference, available, identity) in references.items():
            if not available: continue
            candidate = [r for r in rows if int(r['fixation_index']) in available]
            pairs = legacy.exact_join(candidate, reference, available)
            deltas = [dict(r,reference_theta_deg=float(b['theta_deg']),reference_A_diopters=float(b['A_diopters']),
                delta_theta_deg=float(r['theta_deg'])-float(b['theta_deg']),delta_A_diopters=float(r['A_diopters'])-float(b['A_diopters'])) for r,b in pairs]
            legacy.write_csv(args.prediction_dir/f'{name}_vs_{label}_frames.csv',deltas)
            stats = {}
            for j in [None]+available:
                local = deltas if j is None else [r for r in deltas if int(r['fixation_index'])==j]
                stats['global' if j is None else str(j)] = {field:dict(legacy.metrics([float(r['delta_'+field]) for r in local]),
                    candidate_within_fixation_SD=float(np.std([float(r[field]) for r in local])) if j is not None else None)
                    for field in ['theta_deg','A_diopters']}
            entries[label] = dict(reference=identity, evaluated_fixations=available,metrics=stats,
                fixation_mean_deviation=fixation_mean_deviations(stats,available))
            fig,axes=plt.subplots(len(available),2,figsize=(12,2.4*len(available)),squeeze=False,layout='constrained')
            for i,j in enumerate(available):
                local=[r for r in deltas if int(r['fixation_index'])==j]
                for col,field in enumerate(['theta_deg','A_diopters']):
                    plot_trajectory(axes[i,col],local,field,name)
                    plot_trajectory(axes[i,col],local,'reference_'+field,label)
                    axes[i,col].set_title(f'Fixation {j}: {field}')
                    axes[i,col].set(xlabel='Original frame index',ylabel='Gaze (degrees)' if field=='theta_deg' else 'Accommodation (D)')
            axes[0,0].legend();fig.savefig(args.prediction_dir/f'{name}_vs_{label}_traces.png',dpi=130);plt.close(fig)
        entries['diagnostics'] = dict(frame_count=len(rows),
            ambiguity_count=sum(int(r['equivalent_minima_count'])>1 for r in rows),
            stationarity_unverified_count=sum(r['stationarity_verified']!='True' for r in rows),
            theta_bound_count=sum(r['theta_bound']=='True' for r in rows),A_bound_count=sum(r['A_bound']=='True' for r in rows),
            extrapolation_count=sum(r['extrapolation']=='True' for r in rows))
        summary[name]=entries
    atomic_json(args.prediction_dir/'comparison_summary.json',dict(fold=frozen['fold'],scope=frozen['scope'],metrics=summary,
        primary_reference='H0 frozen holdout3 model anchor-free inverse on exact common support',
        secondary_reference='matched full joint calibration states; estimate methods differ',
        training_converged=frozen['training_converged'],training_status=frozen['training_status'],
        interpretation='Retrospective model-state deviations and descriptive variability; flatter trajectories do not establish accuracy.'))


def labeled_paths(values):
    result = {}
    for value in values:
        label, separator, path = value.partition('=')
        if not separator or not label or label in result:
            raise ValueError('Use unique LABEL=directory arguments')
        result[label] = Path(path)
    return result


def continuation_lineage(directory):
    """Account for all numerical segments of a declared candidate objective."""
    lineage, seen = [], set()
    current = Path(directory).resolve()
    while current not in seen:
        seen.add(current)
        model_path=current/'model.json'
        if not model_path.exists():
            lineage.append(dict(directory=str(current),status='historical source artifact unavailable',
                elapsed_seconds=None,converged=None,solver_settings=None))
            break
        model=json.loads(model_path.read_text())
        diag=model['diagnostics'];status=diag.get('continuation_status') or {}
        history=diag.get('history') or json.loads((current/'history.json').read_text())
        elapsed=status.get('elapsed_seconds',history[-1].get('elapsed_seconds',0.))
        lineage.append(dict(directory=str(current),model_sha256=sha(model_path),converged=diag['converged'],
            termination=diag.get('termination'),elapsed_seconds=elapsed,
            solver_settings=json.loads((current/'solver_settings.json').read_text()) if (current/'solver_settings.json').exists() else None))
        source=diag['provenance'].get('continuation_source')
        if not source: break
        current=Path(source).resolve()
    else: raise ValueError('Continuation lineage contains a cycle')
    lineage.reverse()
    return lineage


def summarize(args):
    """Frozen matrix and exact common-support controlled contrasts."""
    cases = dict(H0=EXP/'reduced_calibration/training/holdout3/robust',
                 FF0=EXP/'full_calibration/matched/robust')
    cases.update(labeled_paths(args.case))
    prediction_dirs = dict(H0=EXP/'reduced_calibration/predictions/holdout3')
    prediction_dirs.update(labeled_paths(args.prediction))
    if args.output_dir.exists(): raise ValueError('Use a new summary directory')
    args.output_dir.mkdir(parents=True)
    matrix, training, inverses = {}, {}, {}
    for label, directory in cases.items():
        model = json.loads((directory/'model.json').read_text())
        diag = model['diagnostics']; provenance = diag['provenance']
        histories = diag.get('history') or json.loads((directory/'history.json').read_text())
        last = histories[-1]
        rows = []
        for path in sorted(directory.glob('capture_*_states.csv')):
            rows.extend(r for r in legacy.read_csv(path) if r.get('theta_deg') not in ('',None))
        training[label] = rows
        fixation_summary = {}
        for j in provenance['training_fixations']:
            local = [r for r in rows if int(r['fixation_index'])==j]
            fixation_summary[str(j)] = {field:dict(mean=float(np.mean([float(r[field]) for r in local])),
                within_fixation_SD=float(np.std([float(r[field]) for r in local]))) for field in ['theta_deg','A_diopters']}
        matrix[label] = dict(model_directory=str(directory.resolve()),model_sha256=sha(directory/'model.json'),
            checkpoint_sha256=sha(directory/'checkpoint.npz'),training_fixations=provenance['training_fixations'],
            training_state_csv_sha256={path.name:sha(path) for path in sorted(directory.glob('capture_*_states.csv'))},
            converged=diag['converged'],termination=diag.get('termination'),
            objective=last['objective'],objective_components={k:last.get(k,0.) for k in
                ['observations','model_prior','curvature_prior','anchors','temporal','previous_mean_prior']},
            q=float(model['coefficients'][14]) if len(model['coefficients'])==15 else 0.,
            q_units='normalized displacement d; contribution equals q at abs(theta)=15 degrees',
            curvature_strength=model.get('curvature_strength',0.),previous_mean_prior=model.get('previous_mean_prior'),
            physical_projected_optimality=last.get('physical_projected_optimality'),
            elapsed_seconds=diag.get('continuation_status',{}).get('elapsed_seconds',last.get('elapsed_seconds')),
            p=model['p'], precision=model['precision'],prior_precision=model['prior_precision'],
            coefficient_count=len(model['coefficients']),free_coefficient_count=model.get('free_coefficient_count',len(model['coefficients'])),
            training_state_fixations=fixation_summary,conditioning=diag.get('calibrated_grid_conditioning'),
            training_bound_fractions=dict(theta=diag.get('theta_bound_fraction'),A=diag.get('A_bound_fraction')),
            training_extrapolation_fraction=diag.get('A_extrapolation_fraction'))
        lineage=continuation_lineage(directory)
        matrix[label]['continuation_history']=lineage
        matrix[label]['total_elapsed_seconds']=sum(segment['elapsed_seconds'] or 0. for segment in lineage)
        matrix[label]['runtime_history_complete']=all(segment['elapsed_seconds'] is not None for segment in lineage)
        if label in prediction_dirs:
            pred_dir = prediction_dirs[label]
            manifest=json.loads((pred_dir/'prediction_manifest.json').read_text())
            path=pred_dir/'trained_predictions.csv'
            if manifest['model_sha256']!=sha(directory/'model.json') or manifest['models']['trained']['prediction_sha256']!=sha(path):
                raise ValueError('Frozen prediction/model mismatch: '+label)
            inverse=legacy.read_csv(path);inverses[label]=inverse
            matrix[label]['fixed_inverse_diagnostics']=dict(frame_count=len(inverse),
                ambiguity_count=sum(int(r['equivalent_minima_count'])>1 for r in inverse),
                stationarity_unverified_count=sum(r['stationarity_verified']!='True' for r in inverse),
                theta_bound_count=sum(r['theta_bound']=='True' for r in inverse),A_bound_count=sum(r['A_bound']=='True' for r in inverse),
                extrapolation_count=sum(r['extrapolation']=='True' for r in inverse),
                condition_quantiles=np.nanquantile([float(r['condition']) for r in inverse],[.5,.95,1.]).tolist(),
                prediction_sha256=sha(path),scope=manifest.get('scope','heldout'))
    # Reuse the immutable matched-full optical inverse on holdout3 support.
    # Its scope remains heldout3 even when new full models export all20.
    if 'FF0' in cases and 'FF0' not in prediction_dirs:
        old_dir=EXP/'reduced_calibration/predictions/holdout3'
        old_comparison=json.loads((old_dir/'comparison_summary.json').read_text())
        old_frozen=json.loads((old_dir/'prediction_manifest.json').read_text())
        control=old_dir/'fresh_matched_full_fixed_inverse.csv'
        control_diag=old_comparison['fresh_matched_full_fixed_inverse']['diagnostics']
        ff_model=json.loads((cases['FF0']/'model.json').read_text())
        ff_prov=ff_model['diagnostics']['provenance']
        if (old_comparison['reference_status']['fresh_matched_full']['model_sha256']!=matrix['FF0']['model_sha256'] or
                sha(control)!=control_diag['prediction_sha256'] or
                ff_prov['source_sha256']!=old_frozen['source_sha256'] or
                ff_prov['selected_interval_sha256']!=old_frozen['selected_interval_sha256']):
            raise ValueError('Frozen FF0 anchor-free control identity mismatch')
        inverse=legacy.read_csv(control)
        if len(inverse)!=control_diag['frame_count']: raise ValueError('FF0 control frame count mismatch')
        if 'H0' in inverses: legacy.exact_join(inverse,inverses['H0'],list(range(10,15)))
        inverses['FF0']=inverse
        matrix['FF0']['fixed_inverse_diagnostics']=dict(control_diag,scope='heldout3 only (fixations10–14)',
            reference_source=str(control.resolve()),comparison_manifest_sha256=sha(old_dir/'comparison_summary.json'),
            inference='immutable independent bounded optical inversion, same batch512/iterations100 policy')
    contrasts = {}
    planned = [('H0','Hq'),('H0','HqP'),('H0','H0P'),('H0','FC0'),('H0','FCq'),('H0','FFq'),
        ('Hq','HqP'),('FC0','FCq'),('FCq','FFq'),('Hq','FCq'),('FC0','FF0'),('H0','FF0'),('FF0','FFq')]
    import matplotlib.pyplot as plt
    for left,right in planned:
        if left not in cases or right not in cases: continue
        comparison = {}
        for state_kind, collection in [('joint_training_states',training),('anchor_free_inverse',inverses)]:
            if left not in collection or right not in collection: continue
            a,b=collection[left],collection[right]
            aset=set(int(r['fixation_index']) for r in a);bset=set(int(r['fixation_index']) for r in b)
            common=sorted(aset&bset)
            if not common: continue
            pairs=legacy.exact_join([r for r in b if int(r['fixation_index']) in common],a,common)
            deltas=[dict(r,reference_theta_deg=float(ref['theta_deg']),reference_A_diopters=float(ref['A_diopters']),
                delta_theta_deg=float(r['theta_deg'])-float(ref['theta_deg']),
                delta_A_diopters=float(r['A_diopters'])-float(ref['A_diopters'])) for r,ref in pairs]
            prefix=f'{right}_minus_{left}_{state_kind}'
            legacy.write_csv(args.output_dir/(prefix+'.csv'),deltas)
            statistics={}
            for j in [None]+common:
                local=deltas if j is None else [r for r in deltas if int(r['fixation_index'])==j]
                statistics['global' if j is None else str(j)]={field:dict(legacy.metrics([float(r['delta_'+field]) for r in local]),
                    candidate_within_fixation_SD=float(np.std([float(r[field]) for r in local])) if j is not None else None,
                    reference_within_fixation_SD=float(np.std([float(r['reference_'+field]) for r in local])) if j is not None else None)
                    for field in ['theta_deg','A_diopters']}
            comparison[state_kind]=dict(frame_count=len(deltas),fixations=common,delta_convention=right+' minus '+left,metrics=statistics,
                fixation_mean_deviation=fixation_mean_deviations(statistics,common),
                support_interpretation='exact common original frame/fixation support; joint training and anchor-free inversion are distinct estimates')
            if left=='H0' and state_kind=='anchor_free_inverse':
                matrix[right]['primary_H0_inverse_comparison']=comparison[state_kind]
                held_fixations=sorted(set(common)&set(range(10,15)))
                held_rows=[r for r in deltas if int(r['fixation_index']) in held_fixations]
                held_statistics={'global':{field:legacy.metrics([float(r['delta_'+field]) for r in held_rows])
                    for field in ['theta_deg','A_diopters']}}
                for j in held_fixations: held_statistics[str(j)]=statistics[str(j)]
                held_comparison=dict(frame_count=len(held_rows),fixations=held_fixations,
                    metrics=held_statistics,fixation_mean_deviation=fixation_mean_deviations(held_statistics,held_fixations),
                    interpretation='same five 3D fixations and exact common original frames; held out for holdout3 models, in sample for full models')
                matrix[right]['primary_H0_heldout3_comparison']=held_comparison
                comparison[state_kind]['heldout3_subset']=held_comparison
            fig,axes=plt.subplots(len(common),2,figsize=(12,2.4*len(common)),squeeze=False,layout='constrained')
            for i,j in enumerate(common):
                local=[r for r in deltas if int(r['fixation_index'])==j]
                for col,field in enumerate(['theta_deg','A_diopters']):
                    plot_trajectory(axes[i,col],local,field,right)
                    plot_trajectory(axes[i,col],local,'reference_'+field,left)
                    axes[i,col].set_title(f'Fixation {j}: {field} ({state_kind})')
                    axes[i,col].set(xlabel='Original frame index',ylabel='Gaze (degrees)' if field=='theta_deg' else 'Accommodation (D)')
            axes[0,0].legend();fig.savefig(args.output_dir/(prefix+'.png'),dpi=130);plt.close(fig)
        contrasts[right+'_minus_'+left]=comparison
    atomic_json(args.output_dir/'matrix_summary.json',dict(cases=matrix,controlled_contrasts=contrasts,
        predeclared_cases=['H0','Hq','HqP','H0P','FC0','FCq','FF0','FFq'],
        primary_reference='H0 frozen holdout3 model; anchor-free inverse comparisons use exact common support',
        secondary_reference='FF0 matched full model; joint states and heldout3 inverse labeled separately',
        interpretation='All predeclared cases retained, including nonconvergence. Model-derived state deviations and variability do not measure physiological accuracy. Different training supports/priors prevent naive objective ranking.'))
    flat=[]
    for label,row in matrix.items():
        primary=row.get('primary_H0_inverse_comparison')
        mean=primary['fixation_mean_deviation'] if primary else None
        frame_metrics=primary['metrics']['global'] if primary else None
        reference_fixations=len(set(int(r['fixation_index']) for r in inverses.get('H0',[])))
        flat.append(dict(case=label,converged=row['converged'],objective=row['objective'],q_output_d=row['q'],
            physical_projected_optimality=row['physical_projected_optimality'],elapsed_seconds=row['elapsed_seconds'],
            total_elapsed_seconds=row['total_elapsed_seconds'],numerical_segments=len(row['continuation_history']),
            runtime_history_complete=row['runtime_history_complete'],
            coefficient_count=row['coefficient_count'],free_coefficient_count=row['free_coefficient_count'],
            H0_common_fixation_count=mean['fixation_count'] if mean else reference_fixations if label=='H0' else '',
            H0_mean_gaze_RMSE_deg=mean['theta_deg']['rmse'] if mean else 0. if label=='H0' else '',
            H0_mean_gaze_mean_absolute_bias_deg=mean['theta_deg']['mean_absolute_bias'] if mean else 0. if label=='H0' else '',
            H0_frame_gaze_RMSE_deg=frame_metrics['theta_deg']['rmse'] if frame_metrics else 0. if label=='H0' else '',
            H0_frame_A_RMSE_D=frame_metrics['A_diopters']['rmse'] if frame_metrics else 0. if label=='H0' else '',
            **row['objective_components']))
    legacy.write_csv(args.output_dir/'matrix_summary.csv',flat)
    held_table=[]
    for label,row in matrix.items():
        held=row.get('primary_H0_heldout3_comparison')
        if held is None and label!='H0': continue
        global_metrics=held['metrics']['global'] if held else None
        means=held['fixation_mean_deviation'] if held else None
        held_table.append(dict(case=label,training_fixation_count=len(row['training_fixations']),
            evaluated_fixation_count=means['fixation_count'] if means else 5,
            evaluated_frame_count=held['frame_count'] if held else sum(int(r['fixation_index']) in range(10,15) for r in inverses['H0']),
            demand3_training_status='reference heldout' if label=='H0' else 'heldout' if not set(range(10,15))&set(row['training_fixations']) else 'in_sample',
            converged=row['converged'],q_output_d=row['q'],
            mean_gaze_RMSE_vs_H0_deg=means['theta_deg']['rmse'] if means else 0.,
            mean_gaze_absolute_bias_vs_H0_deg=means['theta_deg']['mean_absolute_bias'] if means else 0.,
            frame_gaze_RMSE_vs_H0_deg=global_metrics['theta_deg']['rmse'] if global_metrics else 0.,
            frame_A_RMSE_vs_H0_D=global_metrics['A_diopters']['rmse'] if global_metrics else 0.))
    legacy.write_csv(args.output_dir/'common_holdout3_summary.csv',held_table)
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained',sharey=True)
    for axis,labels,title in [(axes[0],['Hq','HqP','H0P'],'Holdout3 calibration (15 training fixations)'),
                             (axes[1],['FF0','FC0','FCq','FFq'],'Full calibration (20 training fixations)')]:
        axis.axhline(0.,color='black',lw=.7,label='H0 reference')
        for label in labels:
            held=matrix.get(label,{}).get('primary_H0_heldout3_comparison')
            if not held: continue
            means=held['fixation_mean_deviation']['theta_deg']['per_fixation_mean_delta']
            style=dict(color='gray',linestyle='--',label='FF0 existing full (inverse)') if label=='FF0' else dict(label=label)
            axis.plot(held['fixations'],[means[str(j)] for j in held['fixations']],marker='o',lw=1.,**style)
        axis.set(title=title,xlabel='3 D fixation ID (same five evaluation fixations)',
            ylabel='Mean gaze difference from H0 (degrees)',xticks=list(range(10,15)))
        axis.legend(fontsize=8)
    fig.savefig(args.output_dir/'common_holdout3_mean_gaze.png',dpi=140);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='stage',required=True)
    for stage in ['train','estimate','measurements']:
        cmd=sub.add_parser(stage);cmd.add_argument('--fold',choices=list(FOLDS),required=True)
        cmd.add_argument('--experiment-dir',type=Path,default=EXP)
        cmd.add_argument('--intervals',type=Path,default=EXP/'fixations/fixation_intervals.json')
        cmd.add_argument('--output-dir',type=Path,required=True)
    cmd=sub.choices['train'];cmd.add_argument('--curvature',action='store_true')
    cmd.add_argument('--curvature-strength',type=float,default=.1)
    cmd.add_argument('--previous-mean-strength',type=float,default=0.)
    cmd.add_argument('--previous-mean-scale-deg',type=float,default=1.)
    cmd.add_argument('--previous-dir',type=Path,help='Verified training source for explicit new-objective initialization and optional mean prior')
    cmd.add_argument('--resume-dir',type=Path,help='Strict same-objective checkpoint; repeat all frozen source/settings arguments')
    cmd.add_argument('--loss',choices=['quadratic','robust'],default='robust')
    for name,default,kind in [('max-nfev',300,int),('wall-seconds',600.,float),('lsmr-maxiter',100,int),
        ('lsmr-atol',1e-7,float),('lsmr-btol',1e-7,float),('robust-outer',12,int),('robust-max-nfev',25,int),
        ('physical-gtol',1e-5,float),('physical-step-tol',1e-5,float),('relative-cost-tol',1e-10,float),('kappa',2.,float)]:
        cmd.add_argument('--'+name,default=default,type=kind)
    cmd=sub.choices['estimate'];cmd.add_argument('--model-dir',type=Path,required=True)
    cmd.add_argument('--scope',choices=['heldout','all'],default='heldout')
    cmd.add_argument('--batch-size',type=int,default=512);cmd.add_argument('--inverse-iterations',type=int,default=100)
    cmd=sub.add_parser('estimate-reference',help='Verified immutable H0 anchor-free inverse on all20; no fitting or anchors')
    cmd.add_argument('--experiment-dir',type=Path,default=EXP)
    cmd.add_argument('--intervals',type=Path,default=EXP/'fixations/fixation_intervals.json')
    cmd.add_argument('--output-dir',type=Path,required=True)
    cmd.add_argument('--batch-size',type=int,default=512);cmd.add_argument('--inverse-iterations',type=int,default=100)
    cmd=sub.add_parser('compare');cmd.add_argument('--prediction-dir',type=Path,required=True)
    cmd.add_argument('--matched-dir',type=Path,default=EXP/'full_calibration/matched/robust')
    cmd.add_argument('--holdout-dir',type=Path,default=EXP/'reduced_calibration/training/holdout3/robust')
    cmd.add_argument('--holdout-prediction-dir',type=Path,help='Frozen H0 all20 anchor-free predictions; old heldout-only inverse used if omitted')
    cmd=sub.add_parser('summarize');cmd.add_argument('--case',action='append',default=[],help='LABEL=model directory; H0 and FF0 included by default')
    cmd.add_argument('--prediction',action='append',default=[],help='LABEL=frozen prediction directory')
    cmd.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    if args.stage=='train' and (args.curvature_strength<0 or args.previous_mean_strength<0 or args.previous_mean_scale_deg<=0):
        parser.error('Prior strengths must be nonnegative; scales positive')
    for field in ['max_nfev','wall_seconds','lsmr_maxiter','lsmr_atol','lsmr_btol','robust_outer','robust_max_nfev',
                  'physical_gtol','physical_step_tol','relative_cost_tol','kappa','batch_size','inverse_iterations']:
        if hasattr(args,field) and getattr(args,field)<=0: parser.error(field+' must be positive')
    dispatch={'measurements':'measurements_report','estimate-reference':'estimate_reference'}
    globals()[dispatch.get(args.stage,args.stage)](args)


if __name__=='__main__': main()
