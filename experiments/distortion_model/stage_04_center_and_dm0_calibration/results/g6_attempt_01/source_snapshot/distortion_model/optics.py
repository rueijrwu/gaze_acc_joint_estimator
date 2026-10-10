"""Vectorized CPU reference forward adapter from Theory sections 4–7.

No real optical parameters are fitted in stage one. Angles are visual degrees;
template coordinates and D use reference pixels. Parameters are immutable.
"""
from dataclasses import dataclass
import numpy as np
from .geometry import p1_scale, retained_indices


@dataclass(frozen=True)
class Parameters:
    b1: np.ndarray
    b4: np.ndarray
    omega1: float
    omega4: float
    k1: tuple
    k4: tuple
    aref: float
    m1: float
    center: np.ndarray  # rows b0, bA, s0, sA, c2, c3; columns x/y
    theta_bounds: tuple = (-20., 20.)
    a_bounds: tuple = (0., 6.)
    denominator_min: float = 1e-8

    def __post_init__(self):
        for name in ("k1", "k4"):
            value = tuple(float(x) for x in getattr(self, name))
            if len(value) != 3 or not np.isfinite(value).all():
                raise ValueError(f"{name} requires three finite coefficients")
            object.__setattr__(self, name, value)
        for name in ("theta_bounds", "a_bounds"):
            value = tuple(float(x) for x in getattr(self, name))
            if len(value) != 2 or not np.isfinite(value).all() or value[0] >= value[1]:
                raise ValueError(f"invalid {name}")
            object.__setattr__(self, name, value)
        if not np.isfinite([self.omega1, self.omega4, self.aref, self.m1, self.denominator_min]).all():
            raise ValueError("nonfinite global parameter")
        if self.denominator_min <= 0:
            raise ValueError("denominator margin must be positive")
        for name, shape in (("b1", (3, 2)), ("b4", (3, 2)), ("center", (6, 2))):
            value = np.array(getattr(self, name), dtype=np.float64, copy=True)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"{name} requires finite {shape} values")
            value.setflags(write=False)
            object.__setattr__(self, name, value)


def local_angles(theta_visual, omega1, omega4):
    theta = np.asarray(theta_visual, dtype=np.float64)
    return theta - omega1, theta - omega4


def keystone(xi, baseline, coefficients, denominator_min=1e-8):
    xi = np.asarray(xi, dtype=np.float64)
    alpha, beta, gamma = coefficients
    sx, sy = 1 + alpha * xi**2, 1 + beta * xi**2
    denominator = 1 + gamma * xi[..., None] * baseline[..., :, 1]
    factors = np.stack((sx, sy), axis=-1)[..., None, :]
    valid = ((sx > 0) & (sy > 0) & np.all(denominator > denominator_min, axis=-1)
             & np.isfinite(denominator).all(axis=-1))
    with np.errstate(divide="ignore", invalid="ignore"):
        points = baseline * factors / denominator[..., None]
    valid &= np.isfinite(points).all(axis=(-2, -1))
    return np.where(valid[..., None, None], points, np.nan), valid


def p1_reference(theta_visual, params):
    points, valid = keystone(np.asarray(theta_visual) - params.omega1, params.b1,
                            params.k1, params.denominator_min)
    edges = (points[..., 1:, :] - points[..., :1, :]).reshape(points.shape[:-2] + (4,))
    return points, points.mean(axis=-2), edges, valid


def p4_reference(theta_visual, accommodation, params):
    theta, accommodation = np.broadcast_arrays(theta_visual, accommodation)
    magnification = 1 + params.m1 * (accommodation - params.aref)
    baseline = magnification[..., None, None] * params.b4
    points, valid = keystone(theta - params.omega4, baseline, params.k4, params.denominator_min)
    valid &= np.isfinite(magnification) & (magnification > 0)
    points = np.where(valid[..., None, None], points, np.nan)
    return points, points.mean(axis=-2), valid


def center_polynomial(theta_visual, accommodation, params):
    theta, accommodation = np.broadcast_arrays(theta_visual, accommodation)
    t, a = theta / 10, accommodation - params.aref
    basis = np.stack((np.ones_like(t), a, t, a * t, t**2, t**3), axis=-1)
    return basis @ params.center


def predict_relative(theta_visual, accommodation, observed_edges, edge_covariance,
                     params, held_point=None):
    theta, accommodation = np.broadcast_arrays(theta_visual, accommodation)
    _, mu1, edges, valid1 = p1_reference(theta, params)
    f4, _, valid4 = p4_reference(theta, accommodation, params)
    g, valid_scale = p1_scale(edges, observed_edges, edge_covariance)
    d = center_polynomial(theta, accommodation, params)
    prediction = g[..., None] * np.concatenate(
        (edges, (d[..., None, :] + f4 - mu1[..., None, :]).reshape(theta.shape + (6,))), axis=-1)
    valid = (valid1 & valid4 & valid_scale & np.isfinite(theta) & np.isfinite(accommodation)
             & (theta >= params.theta_bounds[0]) & (theta <= params.theta_bounds[1])
             & (accommodation >= params.a_bounds[0]) & (accommodation <= params.a_bounds[1]))
    return np.where(valid[..., None], prediction, np.nan)[..., retained_indices(held_point)], g, valid
