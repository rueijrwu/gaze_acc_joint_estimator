"""Optional batched NumPy/CuPy multistart inverse.

This is an independently checked acceleration candidate, not the scalar
trust-region reference. Inputs contain only retained P4 coordinates.
"""
from __future__ import annotations
import numpy as np
from .invert import STARTS
from .model import STATE_SCALE, LOWER, UPPER
from .noise import whitening


def forward(beta, capacity, x, r, xp=np):
    """Direct coordinate response/Jacobian; supports (frames,starts,...) arrays."""
    t, A = x[..., 0]/10., x[..., 1]
    L, La = xp.log1p(A), 1/(1+A)
    one, zero = xp.ones_like(t), xp.zeros_like(t)
    bd = xp.stack((one, A, t, t*L, t*t, t*t*L, t**3), -1)
    dd = xp.stack((xp.stack((zero, zero, one, L, 2*t, 2*t*L, 3*t*t), -1)/10,
                   xp.stack((zero, one, zero, t*La, zero, t*t*La, zero), -1)), -1)
    bs = xp.stack((one, t, L, t*L, t*t, t*t*L), -1)
    ds = xp.stack((xp.stack((zero, one, zero, L, 2*t, 2*t*L), -1)/10,
                   xp.stack((zero, zero, La, t*La, zero, t*t*La), -1)), -1)
    width = 4 if capacity == 27 else 6
    values, deriv = [bd@beta[:7]], [xp.einsum("...pz,p->...z", dd, beta[:7])]
    for j in range(5):
        coef = beta[7+j*width:7+(j+1)*width]
        values.append(bs[..., :width]@coef)
        deriv.append(xp.einsum("...pz,p->...z", ds[..., :width, :], coef))
    D = xp.stack(values[:2], -1)
    T = xp.stack(values[2:], -1).reshape(x.shape[:-1]+(2, 2))
    dD = xp.stack(deriv[:2], -2)
    dT = xp.stack(deriv[2:], -2).reshape(x.shape[:-1]+(2, 2, 2))
    F = D[..., None, :]+xp.einsum("...ab,...jb->...ja", T, r)
    J = dD[..., None, :, :]+xp.einsum("...abz,...jb->...jaz", dT, r)
    return F.reshape(x.shape[:-1]+(6,)), J.reshape(x.shape[:-1]+(6, 2))


def solve_batch(model, r, y, covariance, indices=None, backend="numpy", device=0,
                maxiter=120, tolerance=1e-4, starts=STARTS):
    if model.channels != 6:
        raise ValueError("Batched acceleration currently supports only full-position models")
    if backend == "cupy":
        import cupy as xp
        xp.cuda.Device(device).use()
        to_cpu = xp.asnumpy
    elif backend == "numpy":
        xp, to_cpu = np, np.asarray
    else:
        raise ValueError("backend must be numpy or cupy")
    r, y, covariance = np.asarray(r), np.asarray(y), np.asarray(covariance)
    indices = np.arange(6) if indices is None else np.asarray(indices, int)
    if y.shape != (len(r), len(indices)) or covariance.shape != (len(r), len(indices), len(indices)):
        raise ValueError("Batched input must contain exactly the retained coordinates/marginals")
    if len(indices) < 4 or not np.isfinite(r).all() or not np.isfinite(y).all():
        raise ValueError("Batched inputs require two or more finite P4 points and valid P1")
    beta = xp.asarray(model.beta)
    rg = xp.asarray(r)[:, None, :, :]
    yg = xp.asarray(y)[:, None, :]
    W = xp.asarray(whitening(covariance))[:, None, :, :]
    lower, upper = xp.asarray(LOWER/STATE_SCALE), xp.asarray(UPPER/STATE_SCALE)
    scale = xp.asarray(STATE_SCALE)
    z = xp.broadcast_to(xp.asarray(starts/STATE_SCALE)[None, :, :], (len(r), len(starts), 2)).copy()
    damping = xp.full(z.shape[:-1], 1e-5)
    def evaluate(z):
        pred, jac = forward(beta, model.capacity, z*scale, rg, xp)
        e = xp.einsum("...ij,...j->...i", W, pred[..., indices]-yg)
        J = xp.einsum("...ij,...jz->...iz", W, jac[..., indices, :])*scale
        return xp.sum(e*e, axis=-1), e, J
    for iteration in range(maxiter):
        cost, e, J = evaluate(z)
        g = xp.einsum("...iz,...i->...z", J, e)
        pg = xp.max(xp.abs(z-xp.clip(z-g, lower, upper)), axis=-1)
        active = pg > tolerance
        if not bool(to_cpu(xp.any(active))):
            break
        h00, h11 = xp.sum(J[..., 0]**2, -1), xp.sum(J[..., 1]**2, -1)
        h01 = xp.sum(J[..., 0]*J[..., 1], -1)
        ridge = damping*xp.maximum(h00+h11, 1e-12)
        a, b = h00+ridge, h11+ridge
        det = xp.maximum(a*b-h01*h01, 1e-30)
        step = xp.stack(((-b*g[..., 0]+h01*g[..., 1])/det,
                         (h01*g[..., 0]-a*g[..., 1])/det), -1)
        bound0 = ((z[..., 0] <= lower[0]+1e-8) & (g[..., 0] > 0)) | ((z[..., 0] >= upper[0]-1e-8) & (g[..., 0] < 0))
        bound1 = ((z[..., 1] <= lower[1]+1e-8) & (g[..., 1] > 0)) | ((z[..., 1] >= upper[1]-1e-8) & (g[..., 1] < 0))
        step[..., 0] = xp.where(bound0, 0, xp.where(bound1, -g[..., 0]/a, step[..., 0]))
        step[..., 1] = xp.where(bound1, 0, xp.where(bound0, -g[..., 1]/b, step[..., 1]))
        step /= xp.maximum(xp.max(xp.abs(step), axis=-1), 1)[..., None]
        step *= active[..., None]
        alpha = xp.ones(z.shape[:-1])
        accepted = ~active
        next_z = z.copy()
        for _ in range(12):
            trial = xp.clip(z+alpha[..., None]*step, lower, upper)
            trial_cost = evaluate(trial)[0]
            improve = (~accepted) & (trial_cost < cost)
            next_z = xp.where(improve[..., None], trial, next_z)
            accepted |= improve
            if bool(to_cpu(xp.all(accepted))):
                break
            alpha = xp.where(accepted, alpha, alpha*.5)
        damping = xp.where(accepted, xp.maximum(damping*.3, 1e-10), xp.minimum(damping*10, 1e8))
        z = next_z
    cost, e, J = evaluate(z)
    g = xp.einsum("...iz,...i->...z", J, e)
    stationarity = xp.max(xp.abs(z-xp.clip(z-g, lower, upper)), -1)
    return dict(states=to_cpu(z*scale), costs=to_cpu(cost), stationarity_encoded=to_cpu(stationarity),
                converged=to_cpu(stationarity <= tolerance), iterations=iteration+1,
                backend=backend, device=device if backend == "cupy" else None)


def assemble(model, r, covariance, indices, solutions):
    """Same clustering/rank/tie policy as the scalar reference, per frame."""
    W = whitening(covariance)
    results = []
    for i in range(len(r)):
        branches = []
        for x, cost, stationarity, converged in zip(solutions["states"][i], solutions["costs"][i],
                solutions["stationarity_encoded"][i], solutions["converged"][i]):
            if not converged:
                continue
            _, J = model.predict(x, r[i], True)
            sv = np.linalg.svd(W[i]@J[indices], compute_uv=False)
            record = dict(state=x.tolist(), cost=float(cost), stationarity_encoded=float(stationarity),
                singular_values_physical=sv.tolist(), rank=int(np.sum(sv > max(sv[0]*1e-6, 1e-8))),
                at_bound=bool(np.any((x-LOWER < 1e-5) | (UPPER-x < 1e-5))))
            old = next((b for b in branches if np.all(np.abs(x-b["state"]) < [.01, .01])), None)
            if old is None:
                branches.append(record)
            elif record["cost"] < old["cost"]:
                old.update(record)
        if not branches:
            results.append(dict(available=False, reason="no_converged_batched_inverse", branches=[]))
            continue
        branches.sort(key=lambda b: b["cost"])
        best = branches[0]["cost"]
        ties = [b for b in branches if b["cost"] <= best+1e-6*(1+best)]
        primary = min(ties, key=lambda b: (b["state"][1], b["state"][0]))
        plausible = [b for b in branches if b["cost"] <= best+2.]
        results.append(dict(available=True, reason="ok" if primary["rank"] == 2 else "weak_rank",
            **primary, branches=branches, plausible_branches=plausible, ambiguous=len(plausible)>1,
            numerical_ties=len(ties), failed_starts=int(np.sum(~solutions["converged"][i])),
            start_count=solutions["states"].shape[1], plausible_delta=2., backend=solutions["backend"]))
    return results
