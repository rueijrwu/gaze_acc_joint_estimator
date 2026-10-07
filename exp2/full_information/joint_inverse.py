"""Bounded multistart held-out inversion without nominal-state anchors.

A single monotone radial robust loss has the same per-frame minimizer as the
Mahalanobis quadratic. The inverse therefore minimizes that quadratic using
the maintained vectorized segment search and preserves its cost-tie policy.
"""

from __future__ import annotations

import numpy as np

from calibrate_profiled import KNOTS, interpolation
from estimate_profiled import forward_jac as relative_forward_jac
from estimate_profiled import invert_batch as relative_invert_batch
if __package__:
    from .joint_problem import joint_basis
else:
    from joint_problem import joint_basis


def forward_jac(theta, a, coef, p, lo, hi):
    theta, a = np.asarray(theta), np.asarray(a)
    relative, derivative = relative_forward_jac(theta, a, np.asarray(coef)[:14], p, lo, hi)
    log_shape = coef[14] + coef[15] * theta / 15 + coef[16] * a
    predicted = np.concatenate((relative, log_shape[..., None]), axis=-1)
    jacobian = np.zeros(theta.shape + (3, 2))
    jacobian[..., :2, :] = derivative
    jacobian[..., 2, 0], jacobian[..., 2, 1] = coef[15] / 15, coef[16]
    return predicted, jacobian


def physical_optimality(state, observation, coef, p, W):
    """First-order bound/knot audit; degenerate_bound_audit checks A=0 further."""
    theta, A = np.asarray(state).T
    a = A**p
    H, dt, _ = joint_basis(theta, a, p)
    force = (H @ coef - observation) @ W
    gt = np.sum(force * (dt @ coef), axis=1)
    _, dw = interpolation(A, KNOTS)
    t = theta / 15
    ratio_a = coef[11] + coef[12]*t + coef[13]*t*t
    multiplier = np.full(len(A), np.nan)
    defined = (A > 0) | (p >= 1)
    multiplier[defined] = p * A[defined]**(p-1)
    displacement_A = dw @ coef[:4] + theta*(dw @ coef[4:8])
    smooth_force = force[:, 1]*ratio_a + force[:, 2]*coef[16]
    ga = force[:, 0]*displacement_A + multiplier*smooth_force
    minus, plus = ga.copy(), ga.copy()
    at_knot = np.zeros(len(A), dtype=bool)
    for k in range(1, len(KNOTS)-1):
        at = abs(A-KNOTS[k]) <= 16*np.finfo(float).eps*max(1., KNOTS[k])
        at_knot |= at
        for segment, side in ((k-1, minus), (k, plus)):
            slope = (coef[segment+1]-coef[segment]+theta*(coef[segment+5]-coef[segment+4]))/(KNOTS[segment+1]-KNOTS[segment])
            side[at] = (force[:, 0]*slope + multiplier*smooth_force)[at]
    pt, pa = gt.copy(), ga.copy()
    pt[((theta <= -20+1e-8) & (gt >= 0)) | ((theta >= 20-1e-8) & (gt <= 0))] = 0
    pa[((A <= 1e-10) & (ga >= 0)) | ((A >= 6-1e-8) & (ga <= 0))] = 0
    pa[at_knot] = np.maximum(np.maximum(minus[at_knot], -plus[at_knot]), 0.)
    zero = (A == 0) & (p < 1)
    # Physical derivatives are singular here; a-space feasible sign includes
    # both rho and logS1. A negative sign is an available descent direction.
    pa[zero] = np.where(smooth_force[zero] >= 0, 0., np.inf)
    undefined = zero & ((ratio_a != 0) | (coef[16] != 0))
    return np.maximum(abs(pt), .25*abs(pa)), at_knot, undefined, smooth_force


def _invert_batch(y, coef, p, W, maxiter=60, tolerance=1e-7):
    L = np.linalg.cholesky(W).T
    candidates, costs, stationarity = [], [], []
    edges = np.r_[0., KNOTS, 6.]
    for lo, hi in zip(edges[:-1], edges[1:]):
        al, ah = lo**p, hi**p
        starts = [(th, aa) for aa in [al, (al+ah)/2, ah] for th in [-20., -10., 0., 10., 20.]]
        z = np.broadcast_to(np.array(starts)[None], (len(y), len(starts), 2)).copy()
        lower, upper = np.array([-20., al]), np.array([20., ah])
        damping = np.full(z.shape[:2], 1e-4)
        for _ in range(maxiter):
            pred, J = forward_jac(z[..., 0], z[..., 1], coef, p, lo, hi)
            error = (pred-y[:, None]) @ L.T
            JW = np.einsum('ab,nkbc->nkac', L, J)
            g = np.einsum('nkac,nka->nkc', JW, error)
            h00 = np.sum(JW[..., 0]**2, axis=-1)
            h11 = np.sum(JW[..., 1]**2, axis=-1)
            h01 = np.sum(JW[..., 0]*JW[..., 1], axis=-1)
            d0, d1 = damping*np.maximum(h00, 1.), damping*np.maximum(h11, 1.)
            determinant = np.maximum((h00+d0)*(h11+d1)-h01*h01, 1e-30)
            step = np.stack([((h11+d1)*g[..., 0]-h01*g[..., 1])/determinant,
                             ((h00+d0)*g[..., 1]-h01*g[..., 0])/determinant], axis=-1)
            trial = np.clip(z-step, lower, upper)
            before = np.sum(error**2, axis=-1)
            predicted = forward_jac(trial[..., 0], trial[..., 1], coef, p, lo, hi)[0]
            after = np.sum(((predicted-y[:, None]) @ L.T)**2, axis=-1)
            accepted = after <= before
            z[accepted] = trial[accepted]
            damping = np.where(accepted, np.maximum(damping/3, 1e-12), np.minimum(damping*10, 1e12))
        pred, J = forward_jac(z[..., 0], z[..., 1], coef, p, lo, hi)
        error = (pred-y[:, None]) @ L.T
        JW = np.einsum('ab,nkbc->nkac', L, J)
        g = np.einsum('nkac,nka->nkc', JW, error)
        pg = z-np.clip(z-g, lower, upper)
        stationarity.append(np.max(np.abs(pg), axis=-1))
        candidates.append(np.stack([z[..., 0], z[..., 1]**(1/p)], axis=-1))
        costs.append(np.sum(error**2, axis=-1))
    candidates, costs = np.concatenate(candidates, axis=1), np.concatenate(costs, axis=1)
    stationarity = np.concatenate(stationarity, axis=1)
    best = costs.min(axis=1)
    near = costs <= best[:, None]+tolerance
    chosen, count = np.empty(len(y), dtype=int), np.empty(len(y), dtype=int)
    distance = np.full(len(y), np.nan)
    second_theta_delta, second_A_delta = np.full(len(y), np.nan), np.full(len(y), np.nan)
    theta_range, A_range = np.empty(len(y)), np.empty(len(y))
    for i in range(len(y)):
        indices = np.flatnonzero(near[i])
        indices = indices[np.lexsort((candidates[i, indices, 0], candidates[i, indices, 1]))]
        unique = []
        for j in indices:
            if not any(np.linalg.norm((candidates[i,j]-candidates[i,k])/np.array([1.,.25])) < 1e-4 for k in unique):
                unique.append(j)
        chosen[i], count[i] = unique[0], len(unique)
        theta_range[i], A_range[i] = np.ptp(candidates[i, unique, 0]), np.ptp(candidates[i, unique, 1])
        if len(unique) > 1:
            delta = candidates[i, unique[1]]-candidates[i, unique[0]]
            second_theta_delta[i], second_A_delta[i] = delta
            distance[i] = np.linalg.norm(delta/np.array([1.,.25]))
    row = np.arange(len(y))
    return candidates[row, chosen], dict(weighted_cost=costs[row, chosen], minimum_candidate_cost=best,
        equivalent_minima_count=count, second_branch_distance=distance,
        second_branch_theta_delta_deg=second_theta_delta, second_branch_A_delta_D=second_A_delta,
        equivalent_theta_range_deg=theta_range, equivalent_A_range_D=A_range,
        segment_projected_stationarity=stationarity[row, chosen], candidate_cost_max=costs.max(axis=1))


def conditional_noise(state, coef, p, covariance):
    """Local full-correlated-noise conditioning; not physiological intervals.

    Unconstrained interior, smooth, nonsingular states only. Bound/knot and
    rank-deficient cases are marked invalid rather than assigned finite CIs.
    """
    theta, A = np.asarray(state).T
    a = A**p
    _, dt, _ = joint_basis(theta, a, p)
    jacobian = np.empty((len(A), 3, 2))
    jacobian[:, :, 0] = dt @ coef
    _, dw = interpolation(A, KNOTS)
    t = theta / 15
    jacobian[:, 0, 1] = dw @ coef[:4] + theta*(dw @ coef[4:8])
    multiplier = np.zeros(len(A))
    positive = A > 0
    multiplier[positive] = p*A[positive]**(p-1)
    if p == 1:
        multiplier[~positive] = 1.
    jacobian[:, 1, 1] = multiplier*(coef[11] + coef[12]*t + coef[13]*t*t)
    jacobian[:, 2, 1] = multiplier*coef[16]
    precision = np.linalg.inv(covariance)
    information = np.einsum("nbi,bc,ncj->nij", jacobian, precision, jacobian)
    eigenvalues = np.linalg.eigvalsh(information)
    knot = np.any(np.isclose(A[:, None], np.asarray(KNOTS)[None, 1:-1], rtol=0, atol=1e-8), axis=1)
    bound = (np.abs(theta) >= 20-1e-8) | (A <= 1e-10) | (A >= 6-1e-8)
    valid = ~knot & ~bound & (eigenvalues[:, 0] > np.maximum(eigenvalues[:, 1]*1e-12, 0))
    state_covariance = np.full((len(A), 2, 2), np.nan)
    state_covariance[valid] = np.linalg.inv(information[valid])
    sd = np.sqrt(np.diagonal(state_covariance, axis1=1, axis2=2))
    correlation = state_covariance[:, 0, 1] / (sd[:, 0]*sd[:, 1])
    return {"local_theta_noise_sd_deg": sd[:, 0], "local_A_noise_sd_D": sd[:, 1],
            "local_noise_correlation": correlation, "local_noise_valid": valid,
            "local_noise_scope": "Linearized conditional variation under supplied covariance/model; excludes coefficients, reference uncertainty and physiological validation."}


def degenerate_bound_audit(state, y, coef, p, W):
    """Audit zero encoded derivatives with nonzero residual cost at A=0."""
    predicted = joint_basis(state[:, 0], state[:, 1]**p, p, derivatives=False)[0] @ coef
    error = predicted - y
    cost = np.einsum("nb,bc,nc->n", error, W, error)
    force = error @ W
    t = state[:, 0] / 15
    ratio_a = coef[11] + coef[12]*t + coef[13]*t*t
    encoded = force[:, 1]*ratio_a + force[:, 2]*coef[16]
    derivative_scale = np.maximum(1., np.abs(force[:, 1]*ratio_a) + np.abs(force[:, 2]*coef[16]))
    degenerate = (state[:, 1] == 0) & (p < 1) & (cost > 1e-12) & (np.abs(encoded) <= 1e-10*derivative_scale)
    right_descent = np.zeros(len(y), dtype=bool)
    if np.any(degenerate):
        for step in (1e-8, 1e-6, 1e-4):
            probe = state[degenerate].copy()
            probe[:, 1] = step
            probe_error = joint_basis(probe[:, 0], probe[:, 1]**p, p, derivatives=False)[0] @ coef - y[degenerate]
            probe_cost = np.einsum("nb,bc,nc->n", probe_error, W, probe_error)
            decrease = probe_cost < cost[degenerate] - 64*np.finfo(float).eps*np.maximum(1., cost[degenerate])
            right_descent[degenerate] |= decrease
    return degenerate, right_descent


def infer_joint(y, coef, p, covariance, scale_hypothesis="known", scale_variance=0.,
                maxiter=60, tolerance=1e-7):
    """Return [theta_deg,A_D] and maintained ambiguity/optimality diagnostics."""
    y, coef, covariance = np.asarray(y, dtype=float), np.asarray(coef, dtype=float), np.asarray(covariance, dtype=float)
    if y.ndim != 2 or y.shape[1] != 3 or not len(y) or not np.isfinite(y).all():
        raise ValueError("Joint inverse requires finite nonempty [d,rho4,logS1] observations")
    free_dispatch = (scale_hypothesis in ("free", "free_scale") or
                     (scale_hypothesis in ("uncertain", "constrained", "uncertain_scale") and
                      float(scale_variance) == np.inf))
    allowed_shapes = ((14,), (17,)) if free_dispatch else ((17,),)
    if coef.shape not in allowed_shapes or not np.isfinite(coef).all() or not np.isfinite(p) or p <= 0:
        raise ValueError("Joint inverse requires 17 finite coefficients (14 allowed for exact free dispatch) and p > 0")
    if covariance.shape != (3, 3) or not np.allclose(covariance, covariance.T):
        raise ValueError("Joint inverse covariance must be symmetric 3x3 SPD")
    np.linalg.cholesky(covariance)
    if not np.isfinite(tolerance) or tolerance < 0 or maxiter < 1:
        raise ValueError("Invalid inverse search settings")
    if scale_hypothesis in ("uncertain", "constrained", "uncertain_scale"):
        scale_variance = float(scale_variance)
        if np.isnan(scale_variance) or scale_variance < 0:
            raise ValueError("Assumed scale variance must be nonnegative")
        if np.isinf(scale_variance):
            scale_hypothesis = "free"
        else:
            covariance = covariance.copy()
            covariance[2, 2] += scale_variance
    if scale_hypothesis in ("free", "free_scale"):
        # Exact function dispatch preserves legacy state/ambiguity/cost ties.
        return relative_invert_batch(y[:, :2], coef[:14], p, np.linalg.inv(covariance[:2, :2]),
                                     maxiter=maxiter, tolerance=tolerance)
    if scale_hypothesis not in ("known", "fixed", "fixed_scale", "uncertain", "constrained", "uncertain_scale"):
        raise ValueError("Unknown scale hypothesis")
    if scale_hypothesis in ("known", "fixed", "fixed_scale") and float(scale_variance) != 0:
        raise ValueError("Fixed-scale inverse requires zero scale_variance")
    W = np.linalg.inv(covariance)
    state, diagnostics = _invert_batch(y, coef, p, W, maxiter, tolerance)
    optimality, knot, undefined, encoded = physical_optimality(state, y, coef, p, W)
    degenerate, right_descent = degenerate_bound_audit(state, y, coef, p, W)
    optimality[right_descent] = np.inf
    diagnostics.update(physical_optimality=optimality, at_interior_knot=knot,
                       physical_derivative_undefined=undefined,
                       encoded_A_zero_derivative=encoded,
                       encoded_A_zero_degenerate=degenerate,
                       A_zero_right_probe_descent=right_descent,
                       physical_optimality_unverified=degenerate & ~right_descent,
                       degenerate_bound_audit="At A=0,p<1: nonzero cost and near-zero encoded derivative require higher-order audit; finite right probes detect descent but cannot certify sufficiency.",
                       scale_hypothesis=scale_hypothesis,
                       assumed_log_scale_variance=float(scale_variance),
                       inverse_state_prior="none; no nominal targets or H0 state anchors",
                       robust_argmin="Any single monotone radial robust loss has this quadratic argmin")
    diagnostics.update(conditional_noise(state, coef, p, covariance))
    return state, diagnostics
