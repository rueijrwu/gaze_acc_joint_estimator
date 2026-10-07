"""Frozen reference residual covariance and correct subset marginals."""
from __future__ import annotations
import numpy as np
from .geometry import context, area_gradient, signed_area


def residual_coordinate_jacobian(p, q, D, T):
    """d[v-D-T r]/d[P1,P4], with state/coefficient values held fixed."""
    ctx = context(p)
    p, q, D, T = map(np.asarray, (p, q, D, T))
    n = (q-ctx.c[..., None, :] - np.einsum("...ab,...jb->...ja", T,
         p-ctx.c[..., None, :])) / ctx.ell[..., None, None]
    g = area_gradient(p)/(2*ctx.ell[..., None, None]**2)
    J = np.zeros(p.shape[:-2]+(6, 12))
    eye = np.eye(2)
    for j in range(3):
        for k in range(3):
            J[..., 2*j:2*j+2, 2*k:2*k+2] = (
                ((T-eye)/3 - (T if j == k else 0))/ctx.ell[..., None, None]
                - n[..., j, :, None]*g[..., k, None, :])
        J[..., 2*j:2*j+2, 6+2*j:8+2*j] = eye/ctx.ell[..., None, None]
    return J


def summary_coordinate_jacobian(p, q):
    ctx = context(p)
    a4 = np.abs(signed_area(q))
    dx = (q[..., :, 0].mean(-1)-ctx.c[..., 0])/ctx.ell
    rho = a4/ctx.ell**2
    J = np.zeros(p.shape[:-2]+(2, 12))
    gp, gq = area_gradient(p).reshape(p.shape[:-2]+(6,)), area_gradient(q).reshape(q.shape[:-2]+(6,))
    J[..., 0, :6] = -dx[..., None]*gp/(2*ctx.ell[..., None]**2)
    J[..., 0, :6:2] -= 1/(3*ctx.ell[..., None])
    J[..., 0, 6::2] = 1/(3*ctx.ell[..., None])
    J[..., 1, :6] = -rho[..., None]*gp/ctx.ell[..., None]**2
    J[..., 1, 6:] = gq/ctx.ell[..., None]**2
    return J


def coordinate_covariance(blocks, shrinkage=.1, floor_px=.02):
    """Robust effective noise from uninterrupted training-only second differences.

    blocks contain complete contiguous runs, never downsampled frame differences.
    Coordinate order is mapped P1 then mapped P4. Motion remains a limitation.
    """
    parts = [np.diff(b, n=2, axis=0) for b in blocks if len(b) >= 3]
    if not parts:
        raise ValueError("No contiguous training triples for coordinate noise")
    delta = np.concatenate(parts)
    center = np.median(delta, axis=0)
    scale = np.maximum(1.4826*np.median(np.abs(delta-center), axis=0), floor_px*np.sqrt(6))
    clipped = np.clip(delta-center, -5*scale, 5*scale)
    cov = np.cov(clipped, rowvar=False)/6
    cov = (1-shrinkage)*cov + shrinkage*np.diag(np.diag(cov)) + floor_px**2*np.eye(12)
    return cov, dict(method="training_contiguous_second_difference_clipped5MAD_div6",
                      effective_noise_not_independent_localization=True,
                      triple_count=len(delta), shrinkage=shrinkage, floor_px=floor_px)


def reference_covariance(p, pilot, reference_state, sigma, summary=False):
    ctx = context(p)
    x = np.broadcast_to(reference_state, p.shape[:-2]+(2,))
    D, T = pilot.components(x)
    v = pilot.predict(x, ctx.r).reshape(p.shape)
    q = ctx.c[..., None, :]+ctx.ell[..., None, None]*v
    J = summary_coordinate_jacobian(p, q) if summary else residual_coordinate_jacobian(p, q, D, T)
    return J @ sigma @ J.swapaxes(-1, -2)


def marginal(cov, indices):
    ix = np.asarray(indices, int)
    return np.asarray(cov)[..., ix[:, None], ix]


def whitening(cov):
    """W such that W.T W = cov^-1; lower-Cholesky inverse convention."""
    return np.linalg.solve(np.linalg.cholesky(cov), np.broadcast_to(np.eye(cov.shape[-1]), cov.shape))


def predictive_covariance(cov, jacobian, kept, held):
    RII = marginal(cov, kept)
    JI, Jj = jacobian[kept], jacobian[held]
    WJ = np.linalg.solve(RII, JI)
    K = np.linalg.solve(JI.T@WJ, WJ.T)
    M = Jj@K
    cross = cov[np.ix_(held, kept)]
    V = marginal(cov, held) + M@RII@M.T - cross@M.T - M@cross.T
    return (V+V.T)/2
