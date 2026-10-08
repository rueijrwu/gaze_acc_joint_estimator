"""Independent bounded scalar multistart reference, branch diagnostics, holdouts."""
from __future__ import annotations
import itertools
import numpy as np
from scipy.optimize import least_squares
from .model import STATE_SCALE, LOWER, UPPER
from .noise import marginal, whitening, predictive_covariance
from .calibrate import projected_gradient

STARTS = np.array(list(itertools.product([-20, -10, -5, 0, 5, 10, 20], range(7))), float)

STATIONARITY_TOLERANCE = 1e-4
PHYSICAL_CORRECTION_TOLERANCE = 1e-5


def objective_derivatives(model, r, y, W, indices, z):
    """Half squared cost, gradient and exact Hessian in encoded state units."""
    value, J = model.predict(z*STATE_SCALE, r, True)
    e = W@(value[indices]-y)
    J = (W@J[indices])*STATE_SCALE
    second = model.state_hessian(z*STATE_SCALE, r)[indices]
    second = second*STATE_SCALE[:, None]*STATE_SCALE[None, :]
    H = J.T@J+np.einsum('c,cab->ab', W.T@e, second)
    return float(e@e/2), J.T@e, (H+H.T)/2


def polish(model, r, y, W, indices, z, max_steps=20):
    """Bounded, descent-safeguarded exact-Newton refinement and KKT checks.

    Certification uses the original gradient threshold, free/critical-face
    curvature, a physical correction tolerance, and a stable-cost probe.
    Budget termination of the preceding solver is never itself a certificate.
    """
    lower, upper = getattr(model,"lower",LOWER)/STATE_SCALE, getattr(model,"upper",UPPER)/STATE_SCALE
    z = np.clip(np.asarray(z, float), lower, upper)
    last_change = None
    for iteration in range(max_steps+1):
        cost, g, H = objective_derivatives(model, r, y, W, indices, z)
        active = ((z-lower < 1e-9) & (g > 1e-7)) | ((upper-z < 1e-9) & (g < -1e-7))
        free = np.flatnonzero(~active)
        eig = np.linalg.eigvalsh(H[np.ix_(free, free)]) if len(free) else np.array([])
        curvature_scale = max(1., float(np.max(np.abs(eig))) if len(eig) else 1.)
        minimum = bool(not len(eig) or eig[0] >= -1e-10*curvature_scale)
        step = np.zeros(2)
        if len(free):
            h = H[np.ix_(free, free)]
            shift = max(0., 1e-12*curvature_scale-(eig[0] if len(eig) else 0.))
            step[free] = -np.linalg.solve(h+shift*np.eye(len(free)), g[free])
            if not minimum and np.linalg.norm(g[free]) < 1e-8:
                step[free] = .05*np.linalg.eigh(h)[1][:, 0]
        projected_step = np.clip(z+step, lower, upper)-z
        correction = float(np.max(np.abs(projected_step*STATE_SCALE)))
        stationarity = projected_gradient(z, g, lower, upper)
        probe_cost = objective_derivatives(model, r, y, W, indices, z+projected_step)[0] if correction < PHYSICAL_CORRECTION_TOLERANCE else None
        stable = probe_cost is not None and abs(probe_cost-cost) <= 1e-10*(1+cost)
        if stationarity <= STATIONARITY_TOLERANCE and minimum and correction <= PHYSICAL_CORRECTION_TOLERANCE and stable:
            return z, dict(certified=True, steps=iteration, stationarity_encoded=stationarity,
                physical_correction=correction, stable_cost=True, local_minimum=True,
                free_curvature_eigenvalues=eig.tolist(), last_cost_change=last_change,
                reason="certified_stationary_minimum")
        if iteration == max_steps:
            break
        length = np.max(np.abs(step))
        if length > .25:
            step *= .25/length
        accepted = False
        for power in range(24):
            trial = np.clip(z+step*2.**(-power), lower, upper)
            delta = trial-z
            trial_cost, trial_g, _ = objective_derivatives(model, r, y, W, indices, trial)
            slack = 8*np.finfo(float).eps*(1+cost)
            # At a high-residual minimum, the tiny Newton cost reduction can
            # be below floating-point evaluation error. Permit a stable-cost
            # correction only if it improves the unchanged gradient criterion.
            roundoff_correction = (minimum and correction <= PHYSICAL_CORRECTION_TOLERANCE and
                abs(trial_cost-cost) <= 1e-10*(1+cost) and
                projected_gradient(trial,trial_g,lower,upper) < .5*stationarity)
            if trial_cost <= cost+min(0., 1e-4*float(g@delta))+slack or roundoff_correction:
                last_change = float(trial_cost-cost)
                z = trial
                accepted = True
                break
        if not accepted:
            break
    return z, dict(certified=False, steps=iteration, stationarity_encoded=stationarity,
        physical_correction=correction, stable_cost=bool(stable), local_minimum=minimum,
        free_curvature_eigenvalues=eig.tolist(), last_cost_change=last_change,
        reason="negative_curvature" if not minimum else "unresolved_stationarity_or_correction")


def invert(model, r, y, cov, indices=None, starts=STARTS, max_nfev=100,
           tie_tolerance=1e-6, plausible_delta=2., cluster_tolerance=(.01, .01),
           objective_scale=1.):
    """No other-frame state or labels enter starts, weights, or branch selection.

    `y` must contain ONLY the selected coordinates. `cov` is their marginal.
    plausible_delta is a declared heuristic cost difference, not a confidence level.
    objective_scale changes numerical least-squares units only: polishing, costs,
    rank and branch thresholds still use the supplied statistical covariance.
    """
    indices = np.arange(model.channels) if indices is None else np.asarray(indices, int)
    if len(y) != len(indices) or cov.shape != (len(y), len(y)):
        raise ValueError("Input must contain exactly the retained observations and covariance")
    if not np.isfinite(y).all() or not np.isfinite(r).all():
        return dict(available=False, reason="invalid_subset_input", branches=[])
    if model.channels == 6 and len(np.unique(indices//2)) < 2:
        return dict(available=False, reason="fewer_than_two_P4", branches=[])
    if not np.isfinite(objective_scale) or objective_scale <= 0:
        raise ValueError("Numerical objective scale must be finite and positive")
    W = whitening(cov)
    numerical_gain = np.sqrt(objective_scale)
    def fun(z):
        return numerical_gain*(W@(model.predict(z*STATE_SCALE, r)[indices]-y))
    def jac(z):
        return numerical_gain*(W@model.predict(z*STATE_SCALE, r, True)[1][indices]*STATE_SCALE)
    physical_lower,physical_upper=getattr(model,"lower",LOWER),getattr(model,"upper",UPPER)
    lower, upper = physical_lower/STATE_SCALE, physical_upper/STATE_SCALE
    if starts is STARTS and hasattr(model,"lower"):
        starts=np.clip(starts,physical_lower,physical_upper)
    branches, failed, candidates = [], 0, []
    for start_id, physical in enumerate(np.asarray(starts)):
        result = least_squares(fun, physical/STATE_SCALE, jac=jac, bounds=(lower, upper),
            method="trf", ftol=None, xtol=1e-12, gtol=1e-8*objective_scale, max_nfev=max_nfev)
        g = (jac(result.x).T@fun(result.x))/objective_scale
        stationarity = projected_gradient(result.x, g, lower, upper)
        initial = dict(state=(result.x*STATE_SCALE).tolist(), cost=float(2*result.cost/objective_scale),
            status=int(result.status), message=str(result.message), nfev=int(result.nfev),
            success=bool(result.success), stationarity_encoded=stationarity,
            accepted_before_polish=bool(result.success and stationarity <= STATIONARITY_TOLERANCE),
            termination="budget_exhausted" if result.status == 0 else
                "small_step" if result.status == 3 else "gradient_stop" if result.status == 1 else "solver_stop")
        finite = bool(np.isfinite(result.x).all() and np.isfinite(result.cost))
        if finite:
            z, certificate = polish(model, r, y, W, indices, result.x)
        else:
            z, certificate = result.x, dict(certified=False, reason="nonfinite_candidate")
        candidate = dict(start=start_id, starting_state=np.asarray(physical).tolist(), initial=initial, state=(z*STATE_SCALE).tolist(),
            cost=float(2*objective_derivatives(model, r, y, W, indices, z)[0]) if finite else None,
            polish=certificate, accepted=certificate["certified"])
        candidates.append(candidate)
        if not certificate["certified"]:
            failed += 1
            continue
        x = z*STATE_SCALE
        _, J = model.predict(x, r, True)
        sv = np.linalg.svd(W@J[indices]*np.array([1., 1.]), compute_uv=False)
        record = dict(state=x.tolist(), cost=candidate["cost"],
                      stationarity_encoded=certificate["stationarity_encoded"], singular_values_physical=sv.tolist(),
                      certificate=certificate,
                      rank=int(np.sum(sv > max(sv[0]*1e-6, 1e-8))),
                      at_bound=bool(np.any((x-physical_lower < 1e-5) | (physical_upper-x < 1e-5))))
        existing = next((b for b in branches if np.all(np.abs(x-np.array(b["state"])) < cluster_tolerance)), None)
        if existing is None:
            branches.append(record)
        elif record["cost"] < existing["cost"]:
            existing.update(record)
    if not branches:
        return dict(available=False, reason="no_converged_inverse", branches=[], failed_starts=failed,
                    candidates=candidates, start_count=len(starts))
    branches.sort(key=lambda b: b["cost"])
    best_cost = branches[0]["cost"]
    ties = [b for b in branches if b["cost"] <= best_cost+tie_tolerance*(1+best_cost)]
    representative = min(ties, key=lambda b: (b["state"][1], b["state"][0]))
    plausible = [b for b in branches if b["cost"] <= best_cost+plausible_delta]
    identifiable = representative["rank"] == 2
    return dict(available=True, reason="ok" if identifiable else "weak_rank",
                state=representative["state"], cost=representative["cost"],
                stationarity_encoded=representative["stationarity_encoded"],
                at_bound=representative["at_bound"], rank=representative["rank"],
                singular_values_physical=representative["singular_values_physical"],
                ambiguous=len(plausible)>1, numerical_ties=len(ties), branches=branches,
                plausible_branches=plausible, failed_starts=failed, start_count=len(starts),
                plausible_delta=plausible_delta, candidates=candidates,
                recovered_starts=sum(c["accepted"] and not c["initial"]["accepted_before_polish"] for c in candidates),
                acceptance_contract="projected_gradient_1e-4; physical_correction_1e-5; stable_cost; critical_face_curvature")


def predict_holdout(model, ctx, j, retained_v, full_cov, starts=STARTS, channels="xy"):
    """Prediction stage has no parameter for the withheld measurement."""
    held = np.array([2*j, 2*j+1])
    if channels not in ("x", "xy"):
        raise ValueError("Retained channels must be x or xy")
    kept = np.array([i for i in range(6) if i not in held and (channels == "xy" or i % 2 == 0)])
    result = invert(model, ctx.r, retained_v, marginal(full_cov, kept), kept, starts)
    result["held_point"] = int(j)
    if not result["available"]:
        return result
    predictions = []
    for branch in result["plausible_branches"]:
        x = np.asarray(branch["state"])
        v, J = model.predict(x, ctx.r, True)
        q = ctx.c+ctx.ell*v[held]
        predictions.append(dict(state=branch["state"], normalized=v[held].tolist(), pixel=q.tolist()))
    result["predictions"] = predictions
    result["testable"] = bool(result["rank"] == 2 and not result["ambiguous"])
    if result["testable"] and not result["at_bound"]:
        _, J = model.predict(np.asarray(result["state"]), ctx.r, True)
        V = predictive_covariance(full_cov, J, kept, held)
        if np.linalg.eigvalsh(V).min() > 0:
            result["predictive_covariance_normalized"] = V.tolist()
    return result


def score_holdout(prediction, observed_q, ell):
    """Access the excluded coordinate only after state and branches are frozen."""
    result = dict(prediction)
    if not prediction.get("available") or not np.isfinite(observed_q).all():
        result.update(score_available=False, score_reason="missing_prediction_or_measurement")
        return result
    # Keep every plausible prediction. No minimum-error branch selection.
    result["branch_errors_px"] = [(np.asarray(observed_q)-p["pixel"]).tolist()
                                  for p in prediction["predictions"]]
    if not prediction["testable"]:
        result.update(score_available=False, score_reason="ambiguous_or_weak_rank")
        return result
    error = np.asarray(result["branch_errors_px"][0])
    result.update(score_available=True, error_px=error.tolist(), error_normalized=(error/ell).tolist())
    V = prediction.get("predictive_covariance_normalized")
    if V is not None:
        result["mahalanobis"] = float((error/ell)@np.linalg.solve(V, error/ell))
    return result
