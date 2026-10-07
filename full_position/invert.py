"""Independent bounded scalar multistart reference, branch diagnostics, holdouts."""
from __future__ import annotations
import itertools
import numpy as np
from scipy.optimize import least_squares
from .model import STATE_SCALE, LOWER, UPPER
from .noise import marginal, whitening, predictive_covariance
from .calibrate import projected_gradient

STARTS = np.array(list(itertools.product([-20, -10, -5, 0, 5, 10, 20], range(7))), float)


def invert(model, r, y, cov, indices=None, starts=STARTS, max_nfev=100,
           tie_tolerance=1e-6, plausible_delta=2., cluster_tolerance=(.01, .01)):
    """No other-frame state or labels enter starts, weights, or branch selection.

    `y` must contain ONLY the selected coordinates. `cov` is their marginal.
    plausible_delta is a declared heuristic cost difference, not a confidence level.
    """
    indices = np.arange(model.channels) if indices is None else np.asarray(indices, int)
    if len(y) != len(indices) or cov.shape != (len(y), len(y)):
        raise ValueError("Input must contain exactly the retained observations and covariance")
    if not np.isfinite(y).all() or not np.isfinite(r).all():
        return dict(available=False, reason="invalid_subset_input", branches=[])
    if model.channels == 6 and len(indices) < 4:
        return dict(available=False, reason="fewer_than_two_P4", branches=[])
    W = whitening(cov)
    def fun(z):
        return W@(model.predict(z*STATE_SCALE, r)[indices]-y)
    def jac(z):
        return W@model.predict(z*STATE_SCALE, r, True)[1][indices]*STATE_SCALE
    lower, upper = LOWER/STATE_SCALE, UPPER/STATE_SCALE
    branches, failed = [], 0
    for physical in np.asarray(starts):
        result = least_squares(fun, physical/STATE_SCALE, jac=jac, bounds=(lower, upper),
            method="trf", ftol=None, xtol=1e-12, gtol=1e-8, max_nfev=max_nfev)
        g = jac(result.x).T@fun(result.x)
        stationarity = projected_gradient(result.x, g, lower, upper)
        if not result.success or stationarity > 1e-4:
            failed += 1
            continue
        x = result.x*STATE_SCALE
        _, J = model.predict(x, r, True)
        sv = np.linalg.svd(W@J[indices]*np.array([1., 1.]), compute_uv=False)
        record = dict(state=x.tolist(), cost=float(2*result.cost),
                      stationarity_encoded=stationarity, singular_values_physical=sv.tolist(),
                      rank=int(np.sum(sv > max(sv[0]*1e-6, 1e-8))),
                      at_bound=bool(np.any((x-LOWER < 1e-5) | (UPPER-x < 1e-5))))
        existing = next((b for b in branches if np.all(np.abs(x-np.array(b["state"])) < cluster_tolerance)), None)
        if existing is None:
            branches.append(record)
        elif record["cost"] < existing["cost"]:
            existing.update(record)
    if not branches:
        return dict(available=False, reason="no_converged_inverse", branches=[], failed_starts=failed)
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
                plausible_delta=plausible_delta)


def predict_holdout(model, ctx, j, retained_v, full_cov, starts=STARTS):
    """Prediction stage has no parameter for the withheld measurement."""
    held = np.array([2*j, 2*j+1])
    kept = np.array([i for i in range(6) if i not in held])
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
