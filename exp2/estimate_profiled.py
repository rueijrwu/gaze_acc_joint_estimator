"""Shared bounded optical inversion, without state or nominal anchors."""
import numpy as np
from calibrate_profiled import KNOTS, basis, interpolation


def forward_jac(theta, a, coef, p, lo, hi):
    """Analytic forward/Jacobian on one smooth protocol segment, any array shape."""
    middle = (lo+hi)/2
    segment = int(np.clip(np.searchsorted(KNOTS, middle, side='right')-1, 0, 2))
    width = KNOTS[segment+1]-KNOTS[segment]
    db = (coef[segment+1]-coef[segment])/width
    ds = (coef[segment+5]-coef[segment+4])/width
    A = a**(1/p)
    b = coef[segment]+db*(A-KNOTS[segment])
    s = coef[segment+4]+ds*(A-KNOTS[segment])
    t = theta/15
    r = coef[8:14]
    rho = r[0]+r[1]*t+r[2]*t*t+a*(r[3]+r[4]*t+r[5]*t*t)
    displacement = b+s*theta
    if len(coef) == 15: displacement = displacement+coef[14]*t*t
    pred = np.stack([displacement, rho], axis=-1)
    jac = np.empty(theta.shape+(2, 2))
    jac[..., 0, 0] = s
    if len(coef) == 15: jac[..., 0, 0] += 2*coef[14]*theta/225
    jac[..., 1, 0] = (r[1]+2*r[2]*t+a*(r[4]+2*r[5]*t))/15
    jac[..., 0, 1] = (db+ds*theta)/p*a**(1/p-1)
    jac[..., 1, 1] = r[3]+r[4]*t+r[5]*t*t
    return pred, jac


def invert_batch(y, coef, p, W, maxiter=60, tolerance=1e-7):
    """Bounded batched multistart GN. No temporal/nominal/state anchors."""
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
    candidates = np.concatenate(candidates, axis=1)
    costs = np.concatenate(costs, axis=1)
    stationarity = np.concatenate(stationarity, axis=1)
    # Equivalence tolerance matches the existing free_inverse rule.
    best = costs.min(axis=1)
    near = costs <= best[:, None]+tolerance
    chosen = np.empty(len(y), dtype=int)
    count = np.empty(len(y), dtype=int)
    distance = np.full(len(y), np.nan)
    second_theta_delta, second_A_delta = np.full(len(y), np.nan), np.full(len(y), np.nan)
    theta_range, A_range = np.empty(len(y)), np.empty(len(y))
    for i in range(len(y)):
        indices = np.flatnonzero(near[i])
        indices = indices[np.lexsort((candidates[i, indices, 0], candidates[i, indices, 1]))]
        unique = []
        for j in indices:
            if not any(np.linalg.norm((candidates[i, j]-candidates[i, k])/np.array([1., .25])) < 1e-4 for k in unique):
                unique.append(j)
        chosen[i], count[i] = unique[0], len(unique)
        theta_range[i] = np.ptp(candidates[i, unique, 0])
        A_range[i] = np.ptp(candidates[i, unique, 1])
        if len(unique) > 1:
            delta = candidates[i, unique[1]]-candidates[i, unique[0]]
            second_theta_delta[i], second_A_delta[i] = delta
            distance[i] = np.linalg.norm(delta/np.array([1., .25]))
    row = np.arange(len(y))
    return candidates[row, chosen], dict(weighted_cost=costs[row, chosen], minimum_candidate_cost=best,
        equivalent_minima_count=count, second_branch_distance=distance,
        second_branch_theta_delta_deg=second_theta_delta, second_branch_A_delta_D=second_A_delta,
        equivalent_theta_range_deg=theta_range, equivalent_A_range_D=A_range,
        segment_projected_stationarity=stationarity[row, chosen],
        candidate_cost_max=costs.max(axis=1))


def physical_optimality(state, observation, coef, p, W):
    """Full-bound physical KKT check, including both derivatives at knots."""
    theta, A = state.T
    a = A**p
    pred = basis(theta, a, p, derivatives=False, curvature=len(coef)==15)[0] @ coef
    force = (pred-observation) @ W
    w, dw = interpolation(A, KNOTS)
    t, r = theta/15, coef[8:14]
    gt = force[:, 0]*(w @ coef[4:8])+force[:, 1]*(r[1]+2*r[2]*t+a*(r[4]+2*r[5]*t))/15
    if len(coef) == 15: gt += force[:, 0]*2*coef[14]*theta/225
    ratio_a = r[3]+r[4]*t+r[5]*t*t
    ratio_physical_derivative = np.full(len(A), np.nan)
    defined = (A > 0) | (p == 1)
    ratio_physical_derivative[defined] = ratio_a[defined]*p*A[defined]**(p-1)
    ga = force[:, 0]*(dw @ coef[:4]+(dw @ coef[4:8])*theta)+force[:, 1]*ratio_physical_derivative
    minus, plus = ga.copy(), ga.copy()
    at_knot = np.zeros(len(A), dtype=bool)
    for k in range(1, 3):
        at = abs(A-KNOTS[k]) <= 16*np.finfo(float).eps*max(1., KNOTS[k])
        at_knot |= at
        for segment, side in [(k-1, minus), (k, plus)]:
            slope = (coef[segment+1]-coef[segment]+theta*(coef[segment+5]-coef[segment+4]))/(KNOTS[segment+1]-KNOTS[segment])
            side[at] = (force[:, 0]*slope+force[:, 1]*ratio_physical_derivative)[at]
    pt, pa = gt.copy(), ga.copy()
    pt[((theta <= -20+1e-8) & (gt >= 0)) | ((theta >= 20-1e-8) & (gt <= 0))] = 0
    pa[((A <= 1e-10) & (ga >= 0)) | ((A >= 6-1e-8) & (ga <= 0))] = 0
    pa[at_knot] = np.maximum(np.maximum(minus[at_knot], -plus[at_knot]), 0.)
    zero = (A == 0) & (p < 1)
    encoded_ga = force[:, 1]*ratio_a
    pa[zero] = np.where(encoded_ga[zero] > 0, 0., np.inf)
    return np.maximum(abs(pt), .25*abs(pa)), at_knot
