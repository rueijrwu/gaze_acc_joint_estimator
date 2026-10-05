"""Hard-constrained demand holdouts and a freshly matched full calibration."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP))
from calibrate_profiled import (ProfiledProblem, KNOTS, basis, design, interpolation,
                               load_data, noise_covariance, save)
from calibrate_continuation import (atomic_json, array_hash, make_manifest,
                                   read_checkpoint, continue_fit)
from estimate_profiled import forward_jac, invert_batch, physical_optimality

FOLDS = {'full': [], 'holdout3': list(range(10, 15)), 'holdout2': list(range(15, 20))}
PRIOR_RULE = 'diag(1/ptp(retained_training_frame_observations,axis=0)**2); strength0.1'


def coefficient_map(active, curvature=False):
    active = np.asarray(sorted(active), dtype=float)
    if len(active) not in (3, 4) or not np.all(np.isin(active, KNOTS)):
        raise ValueError('Use three or four protocol demand knots')
    if active[0] != KNOTS[0] or active[-1] != KNOTS[-1]:
        raise ValueError('Only an internal demand may be withheld')
    weights = interpolation(KNOTS, active)[0]
    m = len(active)
    E = np.zeros((14, 2*m+6))
    E[:4, :m], E[4:8, m:2*m], E[8:, 2*m:] = weights, weights, np.eye(6)
    if curvature:  # the d-channel curvature coefficient q is always free
        E = np.pad(E, ((0, 1), (0, 1)))
        E[14, -1] = 1.
    return E


def build_training(y, frames, groups, meta, fold, fixed_p=None, curvature=False, theta_anchor_scale_deg=1.):
    heldout = FOLDS[fold]
    selected = [j for j in range(20) if j not in heldout]
    mask = np.isin(groups, selected)
    ty, tf, global_groups = y[mask].copy(), frames[mask].copy(), groups[mask].copy()
    tg = np.searchsorted(selected, global_groups)
    targets = np.array([meta[j]['target_theta_deg'] for j in selected])
    demands = np.array([meta[j]['demand_diopters_label'] for j in selected])
    means = np.array([ty[tg == j].mean(axis=0) for j in range(len(selected))])
    def exponent_loss(p):
        X = design(targets, demands**p)
        c = np.linalg.lstsq(X, means[:, 1], rcond=None)[0]
        return np.sum((X@c-means[:, 1])**2)
    p = float(minimize_scalar(exponent_loss, bounds=(.15, 1.), method='bounded').x) if fixed_p is None else float(fixed_p)
    if not .15 <= p <= 1:
        raise ValueError('Exponent must remain in the protocol .15..1 range')
    active = np.unique(demands)
    E = coefficient_map(active, curvature)
    b, s, theta = [], [], np.empty(len(ty))
    for demand in active:
        slots = np.flatnonzero(demands == demand)
        line = np.linalg.lstsq(np.column_stack([np.ones(len(slots)), targets[slots]]), means[slots, 0], rcond=None)[0]
        if abs(line[1]) < 1e-8:
            raise ValueError('Insufficient training displacement gain')
        b.append(line[0]); s.append(line[1])
        rows = np.isin(tg, slots)
        theta[rows] = np.clip((ty[rows, 0]-line[0])/line[1], -20, 20)
    rho = np.linalg.lstsq(design(targets, demands**p), means[:, 1], rcond=None)[0]
    coef = E@np.r_[b, s, rho, [0.] if curvature else []]  # curvature starts at q=0
    covariance = noise_covariance(ty, tf, tg)
    ranges = np.ptp(ty, axis=0)
    if np.any(ranges <= 0):
        raise ValueError('Degenerate training observable range')
    prior_W = np.diag(1/ranges**2)
    problem = ProfiledProblem(ty, tf, tg, targets, demands, p, coef, np.linalg.inv(covariance), prior_W, coefficient_map=E,
                              curvature=curvature, curvature_strength=0., theta_anchor_scale_deg=theta_anchor_scale_deg)
    initial = problem.encode(theta, demands[tg])
    return problem, initial, selected, global_groups, covariance, dict(
        training_mean_observations=means.tolist(), active_demand_knots=active.tolist(),
        training_observation_hash=array_hash(ty), exponent_training_mean_SSE=float(exponent_loss(p)),
        prior_rule=PRIOR_RULE, coefficient_map=E.tolist(), free_coefficient_count=E.shape[1])


def train(args):
    source = args.resume_dir or args.warm_start_dir
    source_identity = None
    if source is not None:
        with np.load(source/'checkpoint.npz', allow_pickle=False) as checkpoint:
            source_identity = json.loads(str(checkpoint['manifest_json']))
        if source_identity.get('fold') != args.fold or source_identity.get('experiment') != 'hard_constrained_reduced_demands_v1':
            raise ValueError('Source checkpoint is not this experiment/fold')
    y, frames, groups, meta, records, provenance = load_data(args.experiment_dir, args.intervals, target_overrides=args.target_overrides)
    problem, initial, selected, global_groups, covariance, detail = build_training(
        y, frames, groups, meta, args.fold, source_identity['p'] if source_identity else None,
        curvature=args.curvature, theta_anchor_scale_deg=args.theta_anchor_scale_deg)
    heldout = FOLDS[args.fold]
    provenance.update(training_fixations=selected, heldout_fixation=None, heldout_fixations=heldout,
        fold=args.fold, experiment='hard_constrained_reduced_demands_v1', capture5_used=False,
        training_frame_count=problem.n, heldout_frame_count=int(np.isin(groups, heldout).sum()),
        full_seed_used=False, historical_reference_used=False, fixed_p=problem.p, initialization=detail,
        prior_rule=PRIOR_RULE, continuation_source=str(source) if source else None)
    if args.curvature or args.theta_anchor_scale_deg != 1.:  # defaults leave provenance byte-identical
        provenance.update(curvature=bool(args.curvature), curvature_strength=0.,
                          theta_anchor_scale_deg=float(args.theta_anchor_scale_deg))
    meta = [dict(row, heldout=j in heldout) for j, row in enumerate(meta)]
    held_capture = 3 if args.fold == 'holdout3' else 4 if args.fold == 'holdout2' else None
    records = [record for record in records if record['capture'] != held_capture]
    if args.output_dir is None:
        args.output_dir = HERE/'training'/args.fold
    if args.output_dir.exists():
        raise ValueError('Use a new output directory; preserve earlier checkpoints')
    if source and (source.resolve() == args.output_dir.resolve() or source.resolve() in args.output_dir.resolve().parents):
        raise ValueError('Output must be outside source checkpoint directory')
    args.output_dir.mkdir(parents=True)
    atomic_json(args.output_dir/'provenance.json', provenance)
    (args.output_dir/'selected_intervals.json').write_bytes(args.intervals.read_bytes())
    if source:
        stages = [(args.loss, args.kappa if args.loss == 'robust' else None)]
        if args.warm_start_dir and (args.loss != 'robust' or source_identity['loss'] != 'quadratic'):
            raise ValueError('Warm start must change quadratic to robust')
    else:
        stages = [('quadratic', None), ('robust', args.kappa)]
    x = initial
    for name, kappa in stages:
        manifest = make_manifest(problem, provenance, global_groups, kappa)
        manifest.update(experiment='hard_constrained_reduced_demands_v1', fold=args.fold,
                        heldout_fixations=heldout, prior_rule=PRIOR_RULE)
        if source:
            x = read_checkpoint(source, manifest, problem, global_groups, bool(args.warm_start_dir))[0]
        output = args.output_dir/name
        output.mkdir()
        atomic_json(output/'run_provenance.json', provenance)
        x, coef, history, converged, status = continue_fit(problem, x, output, manifest,
            kappa=kappa, max_nfev=args.max_nfev, wall_seconds=args.wall_seconds,
            lsmr_maxiter=args.lsmr_maxiter, lsmr_atol=args.lsmr_atol, lsmr_btol=args.lsmr_btol,
            robust_outer=args.robust_outer, robust_max_nfev=args.robust_max_nfev,
            physical_gtol=args.physical_gtol, physical_step_tol=args.physical_step_tol,
            relative_cost_tol=args.relative_cost_tol)
        save(output, problem, initial, x, coef, history, converged, meta, records, selected,
             provenance, covariance, optical_loss='quadratic' if kappa is None else 'block soft-L1')
        saved = json.loads((output/'model.json').read_text())
        saved['diagnostics'].update(termination=status['termination'], continuation_status=status,
                                    checkpoint_manifest=manifest)
        atomic_json(output/'model.json', saved)
        atomic_json(output/'diagnostics.json', saved['diagnostics'])
        print(json.dumps(dict(fold=args.fold, stage=name, status=status, free_coefficients=problem.free_coefficient_count)), flush=True)


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def state_diagnostics(states, observation, coef, p, W, active_knots=None):
    theta, A = states.T
    curved = len(coef) == 15
    prediction = basis(theta, A**p, p, derivatives=False, curvature=curved)[0]@coef
    stationarity, knots = physical_optimality(states, observation, coef, p, W)
    active = KNOTS if active_knots is None else np.asarray(active_knots)
    nonsmooth = np.zeros(len(A), dtype=bool)
    for knot in active[1:-1]:
        nonsmooth |= abs(A-knot) <= 16*np.finfo(float).eps*max(1., knot)
    w, dw = interpolation(A, KNOTS)
    r, t, a = coef[8:14], theta/15, A**p
    defined = (A > 0) | (p == 1)
    jac = np.empty((len(A), 2, 2))
    jac[:, 0, 0] = w@coef[4:8]
    if curved: jac[:, 0, 0] += 2*coef[14]*theta/225
    jac[:, 1, 0] = (r[1]+2*r[2]*t+a*(r[4]+2*r[5]*t))/15
    jac[:, 0, 1] = dw@coef[:4]+(dw@coef[4:8])*theta
    jac[:, 1, 1] = np.nan
    jac[defined, 1, 1] = (r[3]+r[4]*t[defined]+r[5]*t[defined]**2)*p*A[defined]**(p-1)
    singular = np.full((len(A), 2), np.nan)
    singular[defined] = np.linalg.svd(np.einsum('ab,nbc->nac', np.linalg.cholesky(W).T, jac[defined])*np.array([1., .25]), compute_uv=False)
    return prediction, dict(physical_projected_stationarity=stationarity, stationarity_verified=stationarity <= 1e-5,
        condition=singular[:, 0]/np.maximum(singular[:, 1], 1e-14), minimum_scaled_singular_value=singular[:, 1],
        physical_derivative_defined=defined, interior_protocol_knot=knots, interior_knot_nondifferentiable=nonsmooth,
        theta_bound=abs(theta) >= 20-1e-5, A_bound=(A <= 1e-5) | (A >= 6-1e-5),
        extrapolation=(A < KNOTS[0]) | (A > KNOTS[-1]))


def grid_audit(coef, p, W, active_knots=None):
    """Sampled optical geometry/branches; no global uniqueness certificate."""
    L = np.linalg.cholesky(W).T
    rows = []
    for scope, th_values, A_values in [('calibrated_grid', np.linspace(-15, 15, 17), np.linspace(KNOTS[0], 4., 25)),
                                     ('full_bounds_grid', np.linspace(-20, 20, 17), np.linspace(0., 6., 25))]:
        for theta in th_values:
            for A in A_values:
                if A == 0 and p < 1:
                    rows.append(dict(scope=scope, theta_deg=float(theta), A_diopters=0., side='singular_endpoint', derivative_defined=False))
                    continue
                segment = int(np.clip(np.searchsorted(KNOTS, A, side='right')-1, 0, 2))
                _, jac = forward_jac(np.array([theta]), np.array([A**p]), coef, p, float(KNOTS[segment]), float(KNOTS[segment+1]))
                J = jac[0]; J[:, 1] *= p*A**(p-1)
                sv = np.linalg.svd(L@J@np.diag([1., .25]), compute_uv=False)
                slope = J[1, 1]-J[1, 0]*J[0, 1]/J[0, 0] if abs(J[0, 0]) > 1e-14 else np.nan
                rows.append(dict(scope=scope, theta_deg=float(theta), A_diopters=float(A), side='ordinary_or_right', derivative_defined=True,
                    determinant=float(np.linalg.det(J)), fixed_displacement_ratio_A_slope=float(slope),
                    condition=float(sv[0]/max(sv[1], 1e-14)), minimum_scaled_singular_value=float(sv[1])))
    for theta in np.linspace(-15, 15, 9):
        for knot in KNOTS[1:-1]:
            k = int(np.flatnonzero(KNOTS == knot)[0])
            for side, segment in [('left', k-1), ('right', k)]:
                _, jac = forward_jac(np.array([theta]), np.array([knot**p]), coef, p, KNOTS[segment], KNOTS[segment+1])
                J = jac[0]; J[:, 1] *= p*knot**(p-1)
                sv = np.linalg.svd(L@J@np.diag([1., .25]), compute_uv=False)
                rows.append(dict(scope='calibrated_one_sided_knots', theta_deg=float(theta), A_diopters=float(knot), side=side,
                    derivative_defined=True, determinant=float(np.linalg.det(J)),
                    fixed_displacement_ratio_A_slope=float(J[1, 1]-J[1, 0]*J[0, 1]/J[0, 0]) if abs(J[0, 0]) > 1e-14 else np.nan,
                    condition=float(sv[0]/max(sv[1], 1e-14)), minimum_scaled_singular_value=float(sv[1])))
    # Calibrated-support interior probes; inversion still searches full bounds.
    theta, A = np.meshgrid(np.linspace(-15, 15, 9), np.unique(np.r_[np.linspace(KNOTS[0], 4., 13), KNOTS]))
    true_states = np.column_stack([theta.ravel(), A.ravel()])
    obs = basis(true_states[:, 0], true_states[:, 1]**p, p, derivatives=False, curvature=len(coef) == 15)[0]@coef
    states, diagnostics = invert_batch(obs, coef, p, W, maxiter=100)
    stationarity, _ = physical_optimality(states, obs, coef, p, W)
    roundtrip_error = np.linalg.norm((states-true_states)/np.array([1., .25]), axis=1)
    return dict(scope=f'finite sampled Jacobian grids and{len(states)} calibrated-support roundtrip probes; inverse searches full bounds; not global coverage or uniqueness proof',
        active_demand_knots=KNOTS.tolist() if active_knots is None else list(active_knots),
        sample_counts_by_scope={scope: sum(row['scope'] == scope for row in rows) for scope in {row['scope'] for row in rows}},
        branch_probe_count=len(states), roundtrip_selected_state_mismatch_count=int(np.sum(roundtrip_error > 1e-4)),
        branch_probe_missing_root_count=int(np.sum(diagnostics['minimum_candidate_cost'] > 1e-10)),
        branch_probe_ambiguous_count=int(np.sum(diagnostics['equivalent_minima_count'] > 1)),
        branch_probe_stationarity_unverified_count=int(np.sum(stationarity > 1e-5)),
        physical_jacobian_samples=rows,
        synthetic_branch_probes=[dict(input_theta_deg=float(t), input_A_diopters=float(a), inferred_theta_deg=float(s[0]),
            inferred_A_diopters=float(s[1]), equivalent_minima_count=int(diagnostics['equivalent_minima_count'][i]),
            weighted_cost=float(diagnostics['weighted_cost'][i]), physical_projected_stationarity=float(stationarity[i]),
            reference_scaled_roundtrip_distance=float(roundtrip_error[i])) for i, ((t, a), s) in enumerate(zip(true_states, states))])


def estimate(args):
    if args.fold == 'full':
        raise ValueError('Estimate is for a heldout fold; full trajectories are exported by training')
    model = json.loads((args.model_dir/'model.json').read_text())
    prov = model['diagnostics']['provenance']
    if prov['fold'] != args.fold or prov.get('experiment') != 'hard_constrained_reduced_demands_v1':
        raise ValueError('Wrong model fold/experiment')
    y, frames, groups, meta, records, current = load_data(args.experiment_dir, args.intervals, target_overrides=args.target_overrides)
    if current['source_sha256'] != prov['source_sha256'] or current['selected_interval_sha256'] != prov['selected_interval_sha256']:
        raise ValueError('Prediction sources differ from training')
    if current.get('target_override_sha256') != prov.get('target_override_sha256'):
        raise ValueError('Target override mismatch')
    heldout = FOLDS[args.fold]
    mask = np.isin(groups, heldout)
    hy, hf, hg = y[mask], frames[mask], groups[mask]
    if args.output_dir is None:
        args.output_dir = HERE/'predictions'/args.fold
    if args.output_dir.exists():
        raise ValueError('Use a new prediction directory; preserve frozen predictions')
    args.output_dir.mkdir(parents=True)
    p, W = model['p'], np.asarray(model['precision'])
    E = np.asarray(model['coefficient_map'])
    active = np.unique([meta[j]['demand_diopters_label'] for j in prov['training_fixations']])
    curved = len(model['coefficients']) == 15
    if curved != bool(prov.get('curvature', False)):
        raise ValueError('Model curvature schema differs from its provenance')
    if not np.array_equal(E, coefficient_map(active, curved)):
        raise ValueError('Model map is not this fold hard constraint')
    manifests = {}
    for name, key in [('trained', 'coefficients'), ('initial', 'initial_coefficients')]:
        coef = np.asarray(model[key])
        gamma = np.linalg.lstsq(E, coef, rcond=None)[0]
        if not np.allclose(E@gamma, coef, atol=1e-12, rtol=1e-12):
            raise ValueError('Exported model violates coefficient map')
        rows = []
        for start in range(0, len(hy), args.batch_size):
            stop = min(start+args.batch_size, len(hy))
            states, diagnostics = invert_batch(hy[start:stop], coef, p, W, args.inverse_iterations)
            prediction, physical = state_diagnostics(states, hy[start:stop], coef, p, W, active)
            for k in range(len(states)):
                i = start+k
                row = dict(frame_index=int(hf[i]), fixation_index=int(hg[i]), theta_deg=float(states[k, 0]), A_diopters=float(states[k, 1]),
                    d=float(hy[i, 0]), rho4=float(hy[i, 1]), predicted_d=float(prediction[k, 0]), predicted_rho4=float(prediction[k, 1]),
                    residual_d=float(prediction[k, 0]-hy[i, 0]), residual_rho4=float(prediction[k, 1]-hy[i, 1]))
                for fields in [diagnostics, physical]:
                    row.update({field: bool(value[k]) if value.dtype.kind == 'b' else int(value[k]) if value.dtype.kind in 'iu' else float(value[k])
                                for field, value in fields.items()})
                rows.append(row)
            print(json.dumps(dict(fold=args.fold, model=name, completed_frames=stop, total=len(hy))), flush=True)
        path = args.output_dir/(name+'_predictions.csv')
        write_csv(path, rows)
        manifests[name] = dict(prediction_sha256=sha(path), frame_count=len(rows), frame_hash=array_hash(hf), observation_hash=array_hash(hy))
        atomic_json(args.output_dir/(name+'_grid_audit.json'), json_safe(grid_audit(coef, p, W, active)))
        capture = 3 if args.fold == 'holdout3' else 4
        record = next(record for record in records if record['capture'] == capture)
        lookup = {row['frame_index']: row for row in rows}
        full_rows = []
        for i, frame in enumerate(record['arrays']['frame_index']):
            row = dict(frame_index=int(frame), timestamp_ms=float(record['arrays']['timestamp_ms'][i]), fixation_index=int(record['groups'][i]), status='')
            row.update({field: '' for field in rows[0] if field not in row})
            if int(frame) in lookup:
                row.update(lookup[int(frame)]); row['status'] = 'heldout_core'
            elif not record['valid'][i]:
                row['status'] = 'invalid_detection'
            else:
                row['status'] = 'excluded_fixation_edge' if record['full'][i] else 'freeview'
                row['d'], row['rho4'] = record['y'][i]
            full_rows.append(row)
        write_csv(args.output_dir/(name+f'_capture_{capture}_states.csv'), full_rows)
    atomic_json(args.output_dir/'prediction_manifest.json', dict(
        experiment=prov['experiment'], fold=args.fold, heldout_fixations=heldout, training_fixations=prov['training_fixations'],
        source_sha256=current['source_sha256'], selected_interval_sha256=current['selected_interval_sha256'],
        model_sha256=sha(args.model_dir/'model.json'), models=manifests, model_dir=str(args.model_dir.resolve()),
        training_converged=model['diagnostics']['converged'], training_status=model['diagnostics'].get('continuation_status'),
        inverse_iterations=args.inverse_iterations, batch_size=args.batch_size, baseline_read=False,
        inference='independent bounded per-frame optical inversion; no nominal/state/temporal anchors',
        branch_rule='equivalentcost1e-7; reference-scaleddistance1e-4; smallestA then theta; no global certificate'))


def exact_join(predictions, baseline, heldout):
    def key(row):
        return int(row['fixation_index']), int(row['frame_index'])
    lookup = {}
    for row in baseline:
        if row.get('theta_deg') not in ('', None) and row.get('A_diopters') not in ('', None):
            k = key(row)
            if k in lookup:
                raise ValueError('Duplicate baseline key')
            lookup[k] = row
    keys = [key(row) for row in predictions]
    if len(keys) != len(set(keys)) or set(keys) != {k for k in lookup if k[0] in heldout}:
        raise ValueError('Frame-exact heldout support mismatch')
    pairs = [(row, lookup[key(row)]) for row in predictions]
    for row, baseline in pairs:
        for field in ['theta_deg', 'A_diopters', 'd', 'rho4']:
            if not np.isfinite(float(row[field])) or not np.isfinite(float(baseline[field])):
                raise ValueError('Nonfinite comparison support')
        for field in ['d', 'rho4']:
            if not np.isclose(float(row[field]), float(baseline[field]), atol=1e-12, rtol=1e-12):
                raise ValueError('Baseline observation identity mismatch')
    return pairs


def metrics(values):
    v = np.asarray(values)
    if not len(v):
        return dict(count=0, rmse=None, bias=None, mae=None, median_absolute=None, p95_absolute=None)
    return dict(count=len(v), rmse=float(np.sqrt(np.mean(v*v))), bias=float(np.mean(v)), mae=float(np.mean(abs(v))),
                median_absolute=float(np.median(abs(v))), p95_absolute=float(np.quantile(abs(v), .95)))


def compare(args):
    frozen = json.loads((args.prediction_dir/'prediction_manifest.json').read_text())
    if frozen['fold'] != args.fold:
        raise ValueError('Prediction fold mismatch')
    heldout = FOLDS[args.fold]
    capture = 3 if args.fold == 'holdout3' else 4
    if sha(Path(frozen['model_dir'])/'model.json') != frozen['model_sha256']:
        raise ValueError('Trained model differs from frozen manifest')
    prediction_rows = {}
    for model_name in ['trained', 'initial']:
        path = args.prediction_dir/(model_name+'_predictions.csv')
        if sha(path) != frozen['models'][model_name]['prediction_sha256']:
            raise ValueError('Frozen prediction hash mismatch')
        rows = read_csv(path)
        # Validate both supports and observations before opening reference files.
        exact_join(rows, rows, heldout)
        if len(rows) != frozen['models'][model_name]['frame_count']:
            raise ValueError('Frozen frame count differs')
        prediction_rows[model_name] = rows
    exact_join(prediction_rows['trained'], prediction_rows['initial'], heldout)
    reference_dirs = dict(fresh_matched_full=args.matched_dir)
    if not args.no_legacy:
        reference_dirs['preserved_legacy'] = args.legacy_dir
    refs, reference_hashes = {}, {}
    for name, directory in reference_dirs.items():
        model = json.loads((directory/'model.json').read_text())
        prov = model['diagnostics']['provenance']
        if prov['training_fixations'] != list(range(20)) or prov['source_sha256'] != frozen['source_sha256'] or prov['selected_interval_sha256'] != frozen['selected_interval_sha256']:
            raise ValueError('Reference cohort/source mismatch: '+name)
        if name == 'fresh_matched_full' and (prov.get('experiment') != frozen['experiment'] or prov.get('prior_rule') != PRIOR_RULE or not np.array_equal(np.asarray(model.get('coefficient_map')), np.eye(len(model['coefficients'])))):
            raise ValueError('Matched full reference must use new policy and identity map')
        csv_path = directory/f'capture_{capture}_states.csv'
        refs[name] = (read_csv(csv_path), model)
        reference_hashes[name] = dict(model_sha256=sha(directory/'model.json'), state_csv_sha256=sha(csv_path))
    summary = {}
    matched_model = refs['fresh_matched_full'][1]
    matched_coef = np.asarray(matched_model['coefficients'])
    trained_model = json.loads((Path(frozen['model_dir'])/'model.json').read_text())
    if len(matched_coef) != len(trained_model['coefficients']):
        raise ValueError('Matched full reference and heldout model differ in curvature schema')
    if matched_model['diagnostics']['settings'] != trained_model['diagnostics']['settings']:
        raise ValueError('Matched full reference and heldout model differ in training settings')
    matched_p, matched_W = matched_model['p'], np.asarray(matched_model['precision'])
    input_rows = prediction_rows['trained']
    observation = np.array([[float(r['d']), float(r['rho4'])] for r in input_rows])
    control_rows = []
    for start in range(0, len(observation), frozen['batch_size']):
        stop = min(start+frozen['batch_size'], len(observation))
        states, diagnostics = invert_batch(observation[start:stop], matched_coef, matched_p, matched_W, frozen['inverse_iterations'])
        prediction, physical = state_diagnostics(states, observation[start:stop], matched_coef, matched_p, matched_W, KNOTS)
        for k in range(len(states)):
            i = start+k
            row = dict(frame_index=int(input_rows[i]['frame_index']), fixation_index=int(input_rows[i]['fixation_index']),
                theta_deg=float(states[k, 0]), A_diopters=float(states[k, 1]), d=float(observation[i, 0]), rho4=float(observation[i, 1]),
                predicted_d=float(prediction[k, 0]), predicted_rho4=float(prediction[k, 1]))
            for fields in [diagnostics, physical]:
                row.update({field: bool(value[k]) if value.dtype.kind == 'b' else int(value[k]) if value.dtype.kind in 'iu' else float(value[k])
                            for field, value in fields.items()})
            control_rows.append(row)
        print(json.dumps(dict(control='fresh_matched_full_fixed_inverse', completed_frames=stop, total=len(observation))), flush=True)
    control_path = args.prediction_dir/'fresh_matched_full_fixed_inverse.csv'
    write_csv(control_path, control_rows)
    control_summary = {}
    for name, candidate in [('vs_matched_joint', refs['fresh_matched_full'][0]), ('vs_heldout_trained_inverse', prediction_rows['trained']), ('vs_heldout_initial_inverse', prediction_rows['initial'])]:
        pairs = exact_join(control_rows, candidate, heldout)
        control_summary[name] = {}
        for j in [None]+heldout:
            local = pairs if j is None else [(r, b) for r, b in pairs if int(r['fixation_index']) == j]
            control_summary[name]['global' if j is None else str(j)] = {field: metrics([float(r[field])-float(b[field]) for r, b in local]) for field in ['theta_deg', 'A_diopters']}
    control_summary['diagnostics'] = dict(frame_count=len(control_rows),
        ambiguity_count=sum(int(row['equivalent_minima_count']) > 1 for row in control_rows),
        stationarity_unverified_count=sum(not row['stationarity_verified'] for row in control_rows),
        theta_bound_count=sum(row['theta_bound'] for row in control_rows), A_bound_count=sum(row['A_bound'] for row in control_rows),
        extrapolation_count=sum(row['extrapolation'] for row in control_rows),
        prediction_sha256=sha(control_path), inference_iterations=frozen['inverse_iterations'], batch_size=frozen['batch_size'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for model_name in ['trained', 'initial']:
        predictions = prediction_rows[model_name]
        model_summary = {}
        for ref_name, (baseline, ref_model) in refs.items():
            pairs = exact_join(predictions, baseline, heldout)
            entries = {}
            for j in [None]+heldout:
                local = pairs if j is None else [(r, b) for r, b in pairs if int(r['fixation_index']) == j]
                entries['global' if j is None else str(j)] = {field: metrics([float(r[field])-float(b[field]) for r, b in local]) for field in ['theta_deg', 'A_diopters']}
            reliable_pairs = [(r, b) for r, b in pairs if r['stationarity_verified'] == 'True' and int(r['equivalent_minima_count']) == 1 and float(r['condition']) < 1000 and r['theta_bound'] == 'False' and r['A_bound'] == 'False' and r['interior_knot_nondifferentiable'] == 'False']
            entries['reliable_subset'] = dict(count=len(reliable_pairs), coverage=len(reliable_pairs)/len(pairs),
                metrics={field: metrics([float(r[field])-float(b[field]) for r, b in reliable_pairs]) for field in ['theta_deg', 'A_diopters']})
            model_summary[ref_name] = entries
            deltas = [dict(r, reference_theta_deg=float(b['theta_deg']), reference_A_diopters=float(b['A_diopters']),
                delta_theta_deg=float(r['theta_deg'])-float(b['theta_deg']), delta_A_diopters=float(r['A_diopters'])-float(b['A_diopters'])) for r, b in pairs]
            write_csv(args.prediction_dir/(model_name+'_vs_'+ref_name+'_frames.csv'), deltas)
            fig, axes = plt.subplots(5, 4, figsize=(22, 15), layout='constrained')
            for i, j in enumerate(heldout):
                rows = [r for r in deltas if int(r['fixation_index']) == j]
                frames = [int(r['frame_index']) for r in rows]
                for col, field in enumerate(['theta_deg', 'A_diopters']):
                    axes[i, col].plot(frames, [float(r[field]) for r in rows], lw=.5, label=model_name+' heldout inverse')
                    axes[i, col].plot(frames, [float(r['reference_'+field]) for r in rows], lw=.5, label=ref_name)
                    axes[i, col+2].plot(frames, [float(r['delta_'+field]) for r in rows], lw=.5)
                    axes[i, col+2].axhline(0., lw=.5, color='black')
                    axes[i, col].set_title(f'Fixation{j}: {field}')
                    axes[i, col+2].set_title(f'Fixation{j}: deviation {field}')
            axes[0, 0].legend(fontsize=7)
            fig.savefig(args.prediction_dir/(model_name+'_vs_'+ref_name+'_traces.png'), dpi=130); plt.close(fig)
        reliable = [r for r in predictions if r['stationarity_verified'] == 'True' and int(r['equivalent_minima_count']) == 1 and float(r['condition']) < 1000 and r['theta_bound'] == 'False' and r['A_bound'] == 'False' and r['interior_knot_nondifferentiable'] == 'False']
        model_summary['diagnostics'] = dict(frame_count=len(predictions), ambiguity_count=sum(int(r['equivalent_minima_count']) > 1 for r in predictions),
            stationarity_unverified_count=sum(r['stationarity_verified'] != 'True' for r in predictions),
            theta_bound_count=sum(r['theta_bound'] == 'True' for r in predictions), A_bound_count=sum(r['A_bound'] == 'True' for r in predictions),
            extrapolation_count=sum(r['extrapolation'] == 'True' for r in predictions), reliable_coverage=len(reliable)/len(predictions),
            optical_rmse=np.sqrt(np.mean(np.array([[float(r['residual_d']), float(r['residual_rho4'])] for r in predictions])**2, axis=0)).tolist())
        summary[model_name] = model_summary
    atomic_json(args.prediction_dir/'comparison_summary.json', dict(fold=args.fold, metrics=summary,
        fresh_matched_full_fixed_inverse=control_summary,
        training_converged=frozen['training_converged'], training_status=frozen['training_status'],
        reference_status={name: dict(converged=model['diagnostics']['converged'], termination=model['diagnostics'].get('termination'),
            **reference_hashes[name]) for name, (_, model) in refs.items()},
        interpretation='Retrospective model-state deviations, not measured physiological accuracy; prior-policy and convergence limitations apply'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='stage', required=True)
    for stage in ['train', 'estimate', 'compare']:
        command = sub.add_parser(stage)
        command.add_argument('--fold', choices=list(FOLDS), required=True)
    for stage in ['train', 'estimate']:
        command = sub.choices[stage]
        command.add_argument('--experiment-dir', type=Path, default=EXP)
        command.add_argument('--intervals', type=Path, default=EXP/'fixations/fixation_intervals.json')
        command.add_argument('--output-dir', type=Path)
        command.add_argument('--target-overrides', type=Path, default=None)
    command = sub.choices['train']
    command.add_argument('--theta-anchor-scale-deg', type=float, default=1.,
                         help='Gaze mean-anchor scale in degrees (default 1.0; smaller is a stronger anchor)')
    command.add_argument('--curvature', action='store_true',
                         help='Add the 15th d-channel coefficient q*(theta/15)^2 (zero explicit prior)')
    source = command.add_mutually_exclusive_group()
    source.add_argument('--resume-dir', type=Path)
    source.add_argument('--warm-start-dir', type=Path)
    command.add_argument('--loss', choices=['quadratic', 'robust'], default='robust', help='Resume/warm-start output loss; fresh train runs both')
    for name, default, kind in [('max-nfev', 300, int), ('wall-seconds', 600., float), ('lsmr-maxiter', 100, int),
        ('lsmr-atol', 1e-7, float), ('lsmr-btol', 1e-7, float), ('robust-outer', 12, int), ('robust-max-nfev', 25, int),
        ('physical-gtol', 1e-5, float), ('physical-step-tol', 1e-5, float), ('relative-cost-tol', 1e-10, float), ('kappa', 2., float)]:
        command.add_argument('--'+name, default=default, type=kind)
    command = sub.choices['estimate']
    command.add_argument('--model-dir', type=Path, required=True)
    command.add_argument('--batch-size', type=int, default=512)
    command.add_argument('--inverse-iterations', type=int, default=100)
    command = sub.choices['compare']
    command.add_argument('--prediction-dir', type=Path, required=True)
    command.add_argument('--legacy-dir', type=Path, default=EXP/'full_calibration/legacy')
    command.add_argument('--matched-dir', type=Path, default=EXP/'full_calibration/matched/robust')
    command.add_argument('--no-legacy', action='store_true')
    args = parser.parse_args()
    for field in ['max_nfev', 'wall_seconds', 'lsmr_maxiter', 'lsmr_atol', 'lsmr_btol', 'robust_outer', 'robust_max_nfev',
                  'physical_gtol', 'physical_step_tol', 'relative_cost_tol', 'kappa', 'batch_size', 'inverse_iterations',
                  'theta_anchor_scale_deg']:
        if hasattr(args, field) and getattr(args, field) <= 0:
            parser.error(field+' must be positive')
    if args.stage == 'compare' and args.fold == 'full':
        parser.error('Compare requires a heldout fold')
    globals()[args.stage](args)


if __name__ == '__main__':
    main()
