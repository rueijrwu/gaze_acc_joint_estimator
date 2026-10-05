"""Calibrate, resume, or robustly refine the captures 1–4 profiled estimator."""
import argparse
import contextlib
import hashlib
import importlib
import json
import os
import threading
import time
import tempfile
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OMP_THREAD_LIMIT', os.environ['OMP_NUM_THREADS'].split(',')[0])
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
from scipy.optimize import least_squares

from calibrate_profiled import (ProfiledProblem, SETTINGS, KNOTS, basis, free_inverse,
                               load_data, initial_model, noise_covariance, save, interpolation)

SCHEMA = 'profiled_piecewise_displacement_power_ratio_v1'
ORDER = 'b[4],s[4],rho[1,t,t²,a,at,at²]'
REFERENCE_UNITS = np.array([1., .25])  # degrees, diopters


def array_hash(value):
    a = np.ascontiguousarray(value)
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()


def atomic_json(path, value):
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix=path.name+'.',
                                     suffix='.tmp', delete=False) as stream:
        tmp = Path(stream.name)
        try:
            stream.write(json.dumps(value, indent=2, allow_nan=False)+'\n')
            stream.flush()
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    os.replace(tmp, path)


def make_manifest(problem, provenance, global_groups, kappa):
    """Objective identity includes support/order, initial-function prior and scaling."""
    manifest = dict(version=1, schema=SCHEMA, coefficient_order=ORDER,
                states_shape=[problem.n, 2], frame_hash=array_hash(problem.frames),
                group_hash=array_hash(problem.groups), global_group_hash=array_hash(global_groups),
                observation_hash=array_hash(problem.y),
                source_sha256=provenance['source_sha256'],
                interval_sha256=provenance['selected_interval_sha256'],
                validity_policy=provenance['validity_policy'],
                training_fixations=provenance['training_fixations'],
                heldout_fixation=provenance['heldout_fixation'], capture5_used=False,
                targets=problem.targets.tolist(), demands=problem.demands.tolist(),
                p=problem.p, knots=KNOTS.tolist(), initial_coefficients=problem.initial_coef.tolist(),
                basis_description='d=linear knot b(A)+s(A)*theta; rho=[1,t,t²,a,at,at²], t=theta/15,a=A**p; endpoint-linear extrapolation',
                prior_grid=dict(theta=np.linspace(-15, 15, 9).tolist(),
                                A=np.linspace(KNOTS[0], KNOTS[-1], 9).tolist()),
                prior_matrix_hash=array_hash(problem.prior_M), prior_value_hash=array_hash(problem.prior_v),
                precision=problem.W.tolist(), prior_precision=problem.prior_W.tolist(),
                precision_inverse=np.linalg.inv(problem.W).tolist(), settings=SETTINGS,
                estimated_noise_covariance=(noise_covariance(problem.y, problem.frames, problem.groups).tolist()
                                            if len(problem.left) >= 10 else None),
                temporal_left_hash=array_hash(problem.left), temporal_right_hash=array_hash(problem.right),
                loss='quadratic' if kappa is None else 'block_soft_l1', kappa=kappa,
                encoding=dict(theta_scale=15., power_scale=problem.a_scale,
                              residual_scale=float(np.sqrt(problem.n))),
                bounds=dict(theta=[-20., 20.], A=[0., 6.]))
    if 'target_override_sha256' in provenance:
        manifest['target_override_sha256'] = provenance['target_override_sha256']
    if problem.coefficient_map is not None:
        manifest.update(coefficient_map=problem.coefficient_map.tolist(),
                        free_coefficient_count=problem.free_coefficient_count,
                        active_demand_knots=sorted(set(problem.demands.tolist())))
    if problem.curvature:
        manifest.update(schema='profiled_curved_displacement_power_ratio_v2',
                        coefficient_order=ORDER+',q',
                        basis_description=manifest['basis_description']+'; displacement adds q*t²',
                        curvature_strength=problem.curvature_strength,
                        curvature_scale_output=problem.curvature_scale_output)
        manifest['settings'] = dict(SETTINGS, curvature_prior=problem.curvature_strength,
                                    curvature_scale_output=problem.curvature_scale_output)
    if problem.previous_means is not None:
        manifest.update(previous_mean_prior=dict(strength=problem.previous_mean_strength,
            scale_deg=problem.previous_mean_scale_deg, means_deg=problem.previous_means.tolist(),
            means_hash=array_hash(problem.previous_means), source=problem.previous_mean_provenance,
            nominal_anchor_policy='retained unchanged'))
        manifest['settings'] = dict(manifest['settings'], previous_mean_strength=problem.previous_mean_strength,
                                    previous_mean_scale_deg=problem.previous_mean_scale_deg)
    return manifest


def validate_manifest(source, expected, warm_start=False):
    ignored = {'loss', 'kappa'} if warm_start else set()
    differing = [k for k in set(source)|set(expected)
                 if k not in ignored and source.get(k) != expected.get(k)]
    if differing:
        raise ValueError('Checkpoint identity mismatch: '+', '.join(sorted(differing)))


def write_checkpoint(output, manifest, x, coef, omega, history, status):
    """One atomic NPZ contains manifest plus all numerical state; JSON is advisory."""
    output.mkdir(parents=True, exist_ok=True)
    tmp = output/'checkpoint.tmp.npz'
    np.savez(tmp, states=x, coefficients=coef, weights=omega,
             manifest_json=np.array(json.dumps(manifest, sort_keys=True)),
             history_json=np.array(json.dumps(history, allow_nan=False)))
    os.replace(tmp, output/'checkpoint.npz')
    atomic_json(output/'checkpoint_manifest.json', manifest)
    atomic_json(output/'history.json', history)
    atomic_json(output/'status.json', status)


def read_checkpoint(source, expected, problem, global_groups, warm_start=False):
    with np.load(source/'checkpoint.npz', allow_pickle=False) as data:
        x, coef = data['states'].copy(), data['coefficients'].copy()
        omega = data['weights'].copy() if 'weights' in data else None
        embedded = json.loads(str(data['manifest_json'])) if 'manifest_json' in data else None
    if embedded is None:
        raise ValueError('Checkpoint lacks embedded identity; initialize with --fresh')
    validate_manifest(embedded, expected, warm_start)
    low, high = problem.bounds()
    if (x.shape != low.shape or coef.shape != (problem.coefficient_count,) or not np.isfinite(x).all()
            or not np.isfinite(coef).all() or np.any(x < low) or np.any(x > high)):
        raise ValueError('Invalid checkpoint arrays/bounds')
    rebuilt, weights, info = problem.robust_coefficients(x, expected['kappa'], maxiter=500, tol=1e-10)
    if not info['converged']: raise ValueError('Checkpoint coefficient profile did not converge')
    if not warm_start:
        if not np.allclose(coef, rebuilt, atol=1e-7, rtol=1e-6):
            raise ValueError('Checkpoint coefficients fail same-objective reconstruction')
        if omega is not None and not np.allclose(omega, weights, atol=1e-7, rtol=1e-6):
            raise ValueError('Checkpoint weights fail reconstruction')
    return x, rebuilt, weights, info




def curvature_scale(problem, x, omega, use_problem_hook=True):
    if use_problem_hook:
        hook=getattr(problem,'state_curvature_scale',None)
        if hook is not None: return hook(x,omega)
    d = problem.profile(x, omega)
    # Positive local whitened curvature plus physical anchor/temporal curvature.
    # This is a coordinate preconditioner, not a change to the objective.
    curvature = np.sum(d['local']**2, axis=1)*problem.n
    physical = np.column_stack([np.full(problem.n, 15.), d['dA']/.25])
    curvature += physical**2/(problem.J*problem.counts[problem.groups, None]**2)*problem.n
    if problem.previous_mean_strength > 0:
        curvature[:, 0] += (15**2*problem.previous_mean_strength /
            (problem.J*problem.previous_mean_scale_deg**2*problem.counts[problem.groups]**2))*problem.n
    links = np.zeros(problem.n)
    np.add.at(links, problem.left, problem.link_scale**2)
    np.add.at(links, problem.right, problem.link_scale**2)
    curvature += physical**2*links[:, None]*problem.n
    return np.clip(1/np.sqrt(np.maximum(curvature.ravel(), 1e-16)), 1e-6, 1e6)


def knot_violation(gminus, gplus):
    return np.maximum(np.maximum(gminus, -gplus), 0.)


def envelope_gradients(problem, x, coef, omega, use_problem_hook=True):
    """Physical gradients of the true objective at coefficient stationarity.

    The profile-envelope theorem eliminates d beta*/d x from the objective
    gradient. Optical omega is the true block robust derivative, not stale IRLS.
    """
    if use_problem_hook:
        hook=getattr(problem,'physical_envelope_gradients',None)
        if hook is not None: return hook(x,coef,omega)
    theta, a, A, chain = problem.decode(x)
    H, dt, _ = basis(theta, a, problem.p, curvature=problem.curvature)
    error = H@coef-problem.y
    force = (error@problem.W)* (problem.alpha*omega)[:, None]
    _, dw = interpolation(A, KNOTS)
    displacement_A = dw@coef[:4]+(dw@coef[4:8])*theta
    t = theta/15
    ratio_a = coef[11]+t*coef[12]+t*t*coef[13]
    ratio_A = np.zeros(problem.n)
    positive = A > 0
    ratio_A[positive] = ratio_a[positive]*problem.p*A[positive]**(problem.p-1)
    if problem.p == 1: ratio_A[~positive] = ratio_a[~positive]
    grad = np.column_stack([np.sum(force*(dt@coef), axis=1),
                            force[:, 0]*displacement_A+force[:, 1]*ratio_A])
    mt = np.bincount(problem.groups, weights=theta)/problem.counts
    ma = np.bincount(problem.groups, weights=A)/problem.counts
    grad[:, 0] += (mt-problem.targets)[problem.groups]/(problem.J*problem.counts[problem.groups])
    grad[:, 1] += (ma-problem.demands)[problem.groups]/(.25**2*problem.J*problem.counts[problem.groups])
    if problem.previous_mean_strength > 0:
        grad[:, 0] += (problem.previous_mean_strength*(mt-problem.previous_means)[problem.groups] /
                       (problem.J*problem.previous_mean_scale_deg**2*problem.counts[problem.groups]))
    dif = np.column_stack([theta[problem.right]-theta[problem.left],
                           (A[problem.right]-A[problem.left])/.25**2])*problem.link_scale[:, None]**2
    np.add.at(grad, problem.left, -dif); np.add.at(grad, problem.right, dif)
    minus, plus = grad[:, 1].copy(), grad[:, 1].copy()
    exact = np.zeros(problem.n, bool)
    nearby = np.zeros(problem.n, bool)
    for k in range(1, len(KNOTS)-1):
        distance = np.abs(A-KNOTS[k])
        at = distance <= 16*np.finfo(float).eps*max(1., KNOTS[k])
        nearby |= (distance <= 1e-6) & ~at
        exact |= at
        slopes = []
        for segment in [k-1, k]:
            width = KNOTS[segment+1]-KNOTS[segment]
            slopes.append((coef[segment+1]-coef[segment])/width
                          +(coef[segment+5]-coef[segment+4])*theta/width)
        minus[at] += force[at, 0]*(slopes[0][at]-displacement_A[at])
        plus[at] += force[at, 0]*(slopes[1][at]-displacement_A[at])
    # At A=0 the physical ratio derivative is singular for p<1. Use the
    # exact feasible transformed derivative for its sign; never divide by zero.
    encoded_zero = (force[:, 1]*ratio_a*problem.a_scale if problem.p < 1
                    else grad[:, 1]*chain)
    return grad, minus, plus, exact, nearby, encoded_zero


def physical_diagnostics(problem, x, coef, omega, previous=None, value=None, old_value=None):
    grad, minus, plus, exact, nearby, encoded_zero = envelope_gradients(problem, x, coef, omega)
    theta, _, A, _ = problem.decode(x)
    state = np.column_stack([theta, A])
    pg = grad.copy()
    atlow = state <= np.array([-20., 0.])+1e-12
    athigh = state >= np.array([20., 6.])-1e-12
    pg[(atlow & (pg > 0)) | (athigh & (pg < 0))] = 0
    pg[exact, 1] = knot_violation(minus[exact], plus[exact])
    zero = A == 0
    singular_descent = zero & (encoded_zero < 0) & (problem.p < 1)
    indeterminate_zero = zero & (np.abs(encoded_zero) <= 1e-14) & (problem.p < 1)
    if problem.p < 1: pg[zero, 1] = 0  # magnitude undefined; feasible sign audited separately
    scaled = np.abs(pg*REFERENCE_UNITS)
    row = dict(physical_reference_units=REFERENCE_UNITS.tolist(),
               physical_projected_optimality=float(scaled.max()),
               physical_gradient_max=np.max(np.abs(grad), axis=0).tolist(),
               physical_gradient_rms=np.sqrt(np.mean(grad**2, axis=0)).tolist(),
               physical_gradient_quantiles=np.quantile(np.abs(grad), [.5, .95, 1.], axis=0).tolist(),
               zero_A_physical_derivative_undefined_count=int(zero.sum()) if problem.p < 1 else 0,
               zero_A_feasible_descent_count=int(singular_descent.sum()),
               zero_A_stationarity_unverified_count=int(indeterminate_zero.sum()),
               exact_knot_count=int(exact.sum()), near_not_exact_knot_count=int(nearby.sum()),
               knot_violation_max=float(knot_violation(minus[exact], plus[exact]).max()) if exact.any() else 0.,
               one_sided_knot_minus_max=float(minus[exact].max()) if exact.any() else None,
               one_sided_knot_plus_min=float(plus[exact].min()) if exact.any() else None,
               theta_active_bound_count=int((atlow[:, 0]|athigh[:, 0]).sum()),
               A_active_bound_count=int((atlow[:, 1]|athigh[:, 1]).sum()),
               stationarity_interpretation='Feasible local first-order test; no global optimum certificate')
    if previous is not None:
        ptheta, _, pA, _ = problem.decode(previous)
        step = np.abs(state-np.column_stack([ptheta, pA]))
        row.update(physical_step_max=step.max(axis=0).tolist(),
                   physical_step_rms=np.sqrt(np.mean(step**2, axis=0)).tolist(),
                   physical_step_quantiles=np.quantile(step, [.5, .95, 1.], axis=0).tolist(),
                   reference_step_max=float(np.max(step/REFERENCE_UNITS)))
    if old_value is not None:
        row['relative_cost_change'] = float(abs(value-old_value)/max(abs(old_value), 1e-30))
    return row


@contextlib.contextmanager
def lsmr_telemetry(rows, deadline=None):
    """Patch only SciPy TRF's imported LSMR, restoring it even on exceptions."""
    module = importlib.import_module('scipy.optimize._lsq.trf')
    original = module.lsmr
    def observed(*args, **kwargs):
        if deadline is not None and time.monotonic() >= deadline: raise WallBudget()
        started = time.monotonic()
        result = original(*args, **kwargs)
        rows.append(dict(istop=int(result[1]), iterations=int(result[2]), normr=float(result[3]),
                         normar=float(result[4]), norma=float(result[5]), conda=float(result[6]),
                         normx=float(result[7]), elapsed_seconds=time.monotonic()-started,
                         requested_maxiter=kwargs.get('maxiter'), atol=kwargs.get('atol'), btol=kwargs.get('btol')))
        return result
    module.lsmr = observed
    try: yield
    finally: module.lsmr = original


class WallBudget(Exception):
    pass


def continue_fit(problem, initial, output, manifest, kappa=None, max_nfev=100,
                 wall_seconds=600., lsmr_maxiter=100, lsmr_atol=1e-7, lsmr_btol=1e-7,
                 scaling='curvature', robust_outer=5, physical_gtol=1e-5,
                 physical_step_tol=1e-5, relative_cost_tol=1e-10, weight_tol=1e-7,
                 robust_max_nfev=25, resume_initial=None):
    started = time.monotonic(); deadline = started+wall_seconds
    x = initial.copy()
    if resume_initial is None:
        coef, omega, inner = problem.robust_coefficients(x, kappa, maxiter=500, tol=1e-10)
    else:
        coef = np.asarray(resume_initial['coefficients'],dtype=float).copy()
        omega = np.asarray(resume_initial['weights'],dtype=float).copy()
        if coef.shape != problem.initial_coef.shape or omega.shape != (problem.n,):
            raise ValueError('Verified resume coefficient/weight dimensions differ')
        if not np.all(np.isfinite(coef)) or not np.all(np.isfinite(omega)) or np.any(omega<=0):
            raise ValueError('Verified resume coefficient/weights are invalid')
        inner = dict(converged=True,verified_resume=True)
    if not inner['converged']: raise RuntimeError('Initial coefficient IRLS not converged')
    value, parts = problem.true_objective(x, coef, kappa)
    if resume_initial is not None and not np.isclose(value,resume_initial['objective'],rtol=1e-10,atol=1e-12):
        raise ValueError('Verified resume initial objective differs from previous final objective')
    history = [dict(iteration=0, objective=value, **parts, coefficient_irls=inner,
                    **physical_diagnostics(problem, x, coef, omega))]
    status = dict(status='running', elapsed_seconds=0., accepted_iterations=0, nfev=0)
    telemetry = []; result = None; converged = False; total_nfev = 0
    write_checkpoint(output, manifest, x, coef, omega, history, status)
    stop = threading.Event()
    def heartbeat():
        while not stop.wait(20.):
            snapshot = dict(status, elapsed_seconds=time.monotonic()-started)
            atomic_json(output/'status.json', snapshot)
            print(json.dumps(dict(progress=snapshot)), flush=True)
    thread = threading.Thread(target=heartbeat, daemon=True); thread.start()
    def check_budget():
        if time.monotonic() >= deadline: raise WallBudget()
    try:
        for outer in range(1 if kappa is None else robust_outer):
            fixed_weights = omega.copy(); origin = x.copy(); origin_value = value
            scales = np.ones(len(x)) if scaling == 'none' else curvature_scale(problem, x, fixed_weights)
            atomic_json(output/'solver_settings.json', dict(max_nfev=max_nfev, wall_seconds=wall_seconds,
                        lsmr_maxiter=lsmr_maxiter, lsmr_atol=lsmr_atol, lsmr_btol=lsmr_btol,
                        scaling=scaling, x_scale_hash=array_hash(scales),
                        x_scale_quantiles=np.quantile(scales, [0, .5, 1]).tolist(),
                        physical_gtol=physical_gtol, physical_step_tol=physical_step_tol,
                        relative_cost_tol=relative_cost_tol, robust_weight_tol=weight_tol))
            def fun(z):
                check_budget(); status['nfev'] += 1
                return problem.residual(z, fixed_weights)
            def jac(z):
                check_budget(); return problem.jacobian(z, fixed_weights)
            def callback(intermediate_result):
                nonlocal x, coef, omega, value, parts, converged
                check_budget()
                trial_x = intermediate_result.x.copy()
                candidate, weights, info = problem.robust_coefficients(trial_x, kappa, maxiter=500, tol=1e-10)
                if not info['converged']: raise RuntimeError('Coefficient IRLS did not converge')
                trial, trial_parts = problem.true_objective(trial_x, candidate, kappa)
                accepted = trial <= value+1e-13*max(1., value)
                if accepted:
                    previous, old_value, old_weights = x.copy(), value, omega.copy()
                    x, coef, omega, value, parts = trial_x, candidate, weights, trial, trial_parts
                    row = dict(iteration=len(history), outer=outer, objective=value, **parts,
                               accepted=True,
                               fixed_weight_trust_region_accepted=not getattr(intermediate_result, 'backtracking', False),
                               true_objective_backtracking_accepted=bool(getattr(intermediate_result, 'backtracking', False)),
                               coefficient_irls=info,
                               robust_weight_change=float(np.max(np.abs(omega-old_weights))),
                               fixed_weight_change=float(np.max(np.abs(omega-fixed_weights))),
                               elapsed_seconds=time.monotonic()-started,
                               lsmr=telemetry[-1] if telemetry else None,
                               **physical_diagnostics(problem, x, coef, omega, previous, value, old_value))
                    history.append(row); status['accepted_iterations'] = len(history)-1
                    status['elapsed_seconds'] = time.monotonic()-started
                    write_checkpoint(output, manifest, x, coef, omega, history, status)
                    atomic_json(output/'lsmr_history.json', telemetry)
                    print(json.dumps(row), flush=True)
                    converged = (row['physical_projected_optimality'] <= physical_gtol
                        and row['zero_A_feasible_descent_count'] == 0
                        and row['zero_A_stationarity_unverified_count'] == 0
                        and row['reference_step_max'] <= physical_step_tol
                        and row['relative_cost_change'] <= relative_cost_tol
                        and (kappa is None or row['fixed_weight_change'] <= weight_tol))
                    if converged: raise StopIteration
                else:
                    # Do not checkpoint a stale-weight step that raises the true
                    # objective. End this subproblem and backtrack below.
                    raise StopIteration
            with lsmr_telemetry(telemetry, deadline):
                result = least_squares(fun, origin, jac=jac, bounds=problem.bounds(),
                        method='trf', tr_solver='lsmr', x_scale=scales, callback=callback,
                        tr_options=dict(maxiter=lsmr_maxiter, atol=lsmr_atol, btol=lsmr_btol),
                        max_nfev=max(1, min(max_nfev-total_nfev,
                                          max_nfev if kappa is None else robust_max_nfev)),
                        ftol=1e-12, xtol=1e-12, gtol=None)
            total_nfev += result.nfev
            if converged: break
            # Monotone true-objective backtracking if a robust subproblem's last
            # accepted fixed-weight step was rejected by the callback.
            if not np.array_equal(result.x, x):
                for fraction in [1., .5, .25, .125, .0625, .03125]:
                    check_budget(); z = x+fraction*(result.x-x)
                    c, w, info = problem.robust_coefficients(z, kappa, maxiter=500, tol=1e-10)
                    trial = problem.true_objective(z, c, kappa)[0]
                    if info['converged'] and trial < value:
                        # Reuse the same logging/checkpoint path for the accepted point.
                        try:
                            callback(type('Point', (), dict(x=z, backtracking=True))())
                        except StopIteration:
                            if not converged: raise
                        break
            if converged: break
            last = history[-1]
            stationary = (last['physical_projected_optimality'] <= physical_gtol
                          and last['zero_A_feasible_descent_count'] == 0
                          and last['zero_A_stationarity_unverified_count'] == 0)
            small = ('reference_step_max' in last and last['reference_step_max'] <= physical_step_tol
                     and last['relative_cost_change'] <= relative_cost_tol)
            stable = kappa is None or np.max(np.abs(omega-fixed_weights)) <= weight_tol
            if stationary and small and stable and result.status > 0:
                converged = True; break
            if total_nfev >= max_nfev: break
            if np.array_equal(x, origin) and value == origin_value: break
        status.update(status='converged' if converged else 'not_converged',
                      termination=('physical stationarity, step, cost and weight criteria satisfied' if converged
                                   else 'evaluation/outer budget or solver stall; physical criteria not satisfied'),
                      scipy_status=int(result.status) if result is not None else None,
                      scipy_message=str(result.message) if result is not None else None)
    except WallBudget:
        status.update(status='not_converged', termination='wall budget; last verified accepted checkpoint retained')
    except Exception as exc:
        status.update(status='failed', termination=type(exc).__name__+': '+str(exc))
        raise
    finally:
        stop.set(); thread.join()
        status['elapsed_seconds'] = time.monotonic()-started
        write_checkpoint(output, manifest, x, coef, omega, history, status)
        atomic_json(output/'lsmr_history.json', telemetry)
    return x, coef, history, converged, status


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--fresh', action='store_true', help='Initialize from training data and fit continuously')
    mode.add_argument('--resume-dir', type=Path)
    mode.add_argument('--warm-start-dir', type=Path)
    parser.add_argument('--experiment-dir', type=Path, default=root)
    parser.add_argument('--intervals', type=Path, default=root/'fixations/fixation_intervals.json')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--variant', choices=['solver_only', 'quadratic_noise', 'robust_noise'], required=True)
    parser.add_argument('--holdout-fixation', type=int, help='Fresh fit only: exclude fixation index 0..19')
    parser.add_argument('--p', type=float, help='Fresh fit only: external fixed exponent in (0,1]')
    parser.add_argument('--seed-path', type=Path, help='Fresh full fit only: exponent and prior-precision seed')
    parser.add_argument('--kappa', type=float, default=2.)
    parser.add_argument('--max-nfev', type=int, default=100)
    parser.add_argument('--wall-seconds', type=float, default=600.)
    parser.add_argument('--lsmr-maxiter', type=int, default=100)
    parser.add_argument('--lsmr-atol', type=float, default=1e-7)
    parser.add_argument('--lsmr-btol', type=float, default=1e-7)
    parser.add_argument('--scaling', choices=['curvature', 'none'], default='curvature')
    parser.add_argument('--robust-outer', type=int, default=5)
    parser.add_argument('--robust-max-nfev', type=int, default=25,
                        help='Evaluation cap per robust fixed-weight solve; total cap is max-nfev')
    parser.add_argument('--physical-gtol', type=float, default=1e-5)
    parser.add_argument('--physical-step-tol', type=float, default=1e-5)
    parser.add_argument('--relative-cost-tol', type=float, default=1e-10)
    args = parser.parse_args()
    if min(args.max_nfev, args.wall_seconds, args.lsmr_maxiter, args.lsmr_atol,
           args.lsmr_btol, args.kappa, args.robust_outer, args.robust_max_nfev, args.physical_gtol,
           args.physical_step_tol, args.relative_cost_tol) <= 0:
        parser.error('Budgets, tolerances and kappa must be positive')
    if args.holdout_fixation is not None and not 0 <= args.holdout_fixation < 20:
        parser.error('Holdout fixation must be in 0..19')
    if not args.fresh and any(value is not None for value in [args.holdout_fixation, args.p, args.seed_path]):
        parser.error('Fold, exponent and seed options apply only to --fresh')
    if args.holdout_fixation is not None and args.seed_path is not None:
        parser.error('A full-fit seed cannot be used with a held-out fixation')
    if args.p is not None and not 0 < args.p <= 1:
        parser.error('Exponent must be in (0,1]')
    if (args.output_dir/'checkpoint.npz').exists():
        parser.error('Output already has a checkpoint; choose a new output directory')
    source = args.resume_dir or args.warm_start_dir
    model = None
    seed = None
    if source is not None:
        if source.resolve() == args.output_dir.resolve() or source.resolve() in args.output_dir.resolve().parents:
            parser.error('Use an output directory outside the source checkpoint directory')
        with np.load(source/'checkpoint.npz', allow_pickle=False) as checkpoint:
            identity = json.loads(str(checkpoint['manifest_json'])) if 'manifest_json' in checkpoint else None
        if identity is not None:
            # The embedded identity permits resume without completed reports.
            source_provenance = (json.loads((source/'run_provenance.json').read_text())
                                 if (source/'run_provenance.json').exists() else {})
            source_provenance.update(training_fixations=identity['training_fixations'],
                                     heldout_fixation=identity['heldout_fixation'])
            model = dict(p=identity['p'], initial_coefficients=identity['initial_coefficients'],
                         precision=identity['precision'], prior_precision=identity['prior_precision'],
                         diagnostics=dict(provenance=source_provenance))
        else:
            raise ValueError('Checkpoint lacks embedded identity; initialize with --fresh')
        source_prov = model['diagnostics']['provenance']
        heldout = source_prov['heldout_fixation']
    else:
        heldout = args.holdout_fixation
        seed_path = args.seed_path or args.experiment_dir/'calibration_seed.json'
        if heldout is None and seed_path.exists():
            seed = json.loads(seed_path.read_text())
            if seed.get('calibration_captures') != [f'capture_{i}_detections.pkl' for i in range(1,5)]:
                raise ValueError('Seed must be trained on captures 1–4 only')
        elif args.seed_path is not None:
            raise FileNotFoundError(args.seed_path)
    y, frames, groups, meta, records, provenance = load_data(args.experiment_dir, args.intervals, heldout)
    selected = [j for j in range(20) if j != heldout]
    if model is not None and source_prov['training_fixations'] != selected:
        raise ValueError('Source training fold mismatch')
    mask = np.isin(groups, selected); global_groups = groups[mask]
    remap = {j: g for g, j in enumerate(selected)}
    training_groups = np.array([remap[j] for j in global_groups])
    # Rebuild the initial training-only prior, rather than trusting arbitrary
    # source coefficients as a newly invented prior.
    fixed_p = model['p'] if model is not None else args.p if args.p is not None else seed['p'] if seed else None
    p, initial_coef, theta, accommodation = initial_model(y, groups, meta, selected, fixed_p)
    if model is not None:
        if not np.allclose(initial_coef, model['initial_coefficients'], atol=1e-13, rtol=1e-13):
            raise ValueError('Source prior is not the frozen training-only initial model')
        initial_coef = np.asarray(model['initial_coefficients'])
    cov = noise_covariance(y[mask], frames[mask], training_groups)
    prior_W = (np.asarray(model['prior_precision']) if model is not None else
               np.diag([-1.,1.])@np.asarray(seed['weight_matrix'])@np.diag([-1.,1.]) if seed else
               np.diag(1/np.ptp(y[mask], axis=0)**2))
    W = prior_W if args.variant == 'solver_only' else np.linalg.inv(cov)
    if model is not None and not np.allclose(W, model['precision'], rtol=1e-12, atol=1e-12):
        raise ValueError('Continuation/warm-start must preserve W; choose a matching noise variant')
    problem = ProfiledProblem(y[mask], frames[mask], training_groups,
                             [meta[j]['target_theta_deg'] for j in selected],
                             [meta[j]['demand_diopters_label'] for j in selected],
                             p, initial_coef, np.asarray(model['precision']) if model is not None else W, prior_W)
    kappa = args.kappa if args.variant == 'robust_noise' else None
    provenance.update(training_fixations=selected, heldout_fixation=heldout, fixed_p=p,
                      continuation_source=str(source.resolve()) if source is not None else None,
                      continuation_mode='fresh' if args.fresh else 'warm_start_changed_loss' if args.warm_start_dir else 'same_objective_resume',
                      soft_l1_kappa=args.kappa)
    manifest = make_manifest(problem, provenance, global_groups, kappa)
    initial = (problem.encode(theta[mask], accommodation[mask]) if args.fresh else
               read_checkpoint(source, manifest, problem, global_groups, bool(args.warm_start_dir))[0])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir/'checkpoint.npz').exists():
        raise ValueError('Output already has a checkpoint; resume it into another directory')
    (args.output_dir/'selected_fixation_intervals.json').write_bytes(args.intervals.read_bytes())
    atomic_json(args.output_dir/'run_provenance.json', provenance)
    x, coef, history, converged, status = continue_fit(problem, initial, args.output_dir, manifest,
        kappa, args.max_nfev, args.wall_seconds, args.lsmr_maxiter, args.lsmr_atol,
        args.lsmr_btol, args.scaling, args.robust_outer, args.physical_gtol,
        args.physical_step_tol, args.relative_cost_tol, robust_max_nfev=args.robust_max_nfev)
    diag = save(args.output_dir, problem, initial, x, coef, history, converged,
                meta, records, selected, provenance, cov,
                optical_loss='block soft-L1' if kappa is not None else 'quadratic')
    diag.update(optical_loss='block soft-L1' if kappa is not None else 'quadratic',
                termination=status['termination'], continuation_status=status,
                checkpoint_manifest=manifest, physical_stopping=history[-1])
    saved_model = json.loads((args.output_dir/'model.json').read_text())
    saved_model['diagnostics'] = diag
    atomic_json(args.output_dir/'model.json', saved_model)
    atomic_json(args.output_dir/'diagnostics.json', diag)
    if heldout is not None:
        observation = y[groups == heldout].mean(axis=0)
        inverse = free_inverse(problem, coef, observation)
        inverse.update(fixation_index=heldout, mean_observation=observation.tolist(),
                       target_theta_diagnostic_only=meta[heldout]['target_theta_deg'],
                       nominal_demand_diagnostic_only=meta[heldout]['demand_diopters_label'],
                       interpretation='Whole-fixation inversion without anchors; no physiological accuracy claim')
        atomic_json(args.output_dir/'holdout_inverse.json', inverse)
    print(json.dumps(status), flush=True)


if __name__ == '__main__':
    main()
