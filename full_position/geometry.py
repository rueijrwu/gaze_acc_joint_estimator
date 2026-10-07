"""Correspondence, area normalization, and mask-safe P1 context."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def signed_area(p):
    p = np.asarray(p, float)
    a, b = p[..., 1, :] - p[..., 0, :], p[..., 2, :] - p[..., 0, :]
    return (a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]) / 2


def area_gradient(p):
    p = np.asarray(p, float)
    x, y = p[..., :, 0], p[..., :, 1]
    g = np.stack((np.stack((y[..., 1]-y[..., 2], y[..., 2]-y[..., 0],
                            y[..., 0]-y[..., 1]), -1),
                  np.stack((x[..., 2]-x[..., 1], x[..., 0]-x[..., 2],
                            x[..., 1]-x[..., 0]), -1)), -1) / 2
    return g * np.sign(signed_area(p))[..., None, None]


@dataclass
class Context:
    p: np.ndarray
    c: np.ndarray
    ell: np.ndarray
    r: np.ndarray
    valid: np.ndarray
    signed_area: np.ndarray
    edge_condition: np.ndarray


def context(p):
    p = np.asarray(p, float)
    if p.shape[-2:] != (3, 2):
        raise ValueError("P1 must have shape (...,3,2)")
    edges = np.stack((p[..., 1, :]-p[..., 0, :], p[..., 2, :]-p[..., 0, :]), -1)
    a = signed_area(p)
    edge_sq = np.maximum.reduce([np.sum((p[..., j, :]-p[..., k, :])**2, -1)
                                 for j, k in [(0, 1), (0, 2), (1, 2)]])
    valid = np.isfinite(p).all(axis=(-2, -1)) & (np.abs(a) >
            32*np.finfo(float).eps*np.maximum(edge_sq, 1))
    c = p.mean(axis=-2)
    ell = np.where(valid, np.sqrt(np.abs(a)), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (p-c[..., None, :]) / ell[..., None, None]
    # Invalid contexts keep explicit missing values; SVD never receives NaN.
    condition = np.linalg.cond(np.where(valid[..., None, None], edges, np.eye(2)))
    condition = np.where(valid, condition, np.nan)
    return Context(p, c, ell, r, valid, a, condition)


def reorder(q, found, pair_index):
    order = np.asarray(pair_index, int)
    if order.shape != (3,) or sorted(order.tolist()) != [0, 1, 2]:
        raise ValueError("Missing/invalid correspondence permutation")
    return np.asarray(q, float)[..., order, :], np.asarray(found, bool)[..., order]


def observations(ctx, q, found):
    """Only individual P4 flags enter validity; no full-P4 geometry gates."""
    q, found = np.asarray(q, float), np.asarray(found, bool)
    valid = ctx.valid[..., None] & found & np.isfinite(q).all(-1)
    v = (q-ctx.c[..., None, :])/ctx.ell[..., None, None]
    return np.where(valid[..., None], v, np.nan), valid


def pixel_prediction(ctx, v):
    return ctx.c[..., None, :] + ctx.ell[..., None, None]*np.asarray(v).reshape(ctx.r.shape)


def summaries(ctx, q):
    v = (np.asarray(q)-ctx.c[..., None, :])/ctx.ell[..., None, None]
    return np.concatenate((v.mean(-2), (np.abs(signed_area(q))/ctx.ell**2)[..., None]), -1)


def observed_map(ctx, q):
    e1 = np.stack((ctx.p[..., 1, :]-ctx.p[..., 0, :],
                   ctx.p[..., 2, :]-ctx.p[..., 0, :]), -1)
    e4 = np.stack((q[..., 1, :]-q[..., 0, :], q[..., 2, :]-q[..., 0, :]), -1)
    return np.linalg.solve(np.swapaxes(e1, -1, -2), np.swapaxes(e4, -1, -2)).swapaxes(-1, -2)
