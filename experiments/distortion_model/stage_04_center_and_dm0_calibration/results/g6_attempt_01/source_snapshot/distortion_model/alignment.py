"""Observed fixed-axis pattern balance; physical source symmetry may be unknown."""
import numpy as np


def p1_balance(points):
    """Scale/translation invariant bilateral imbalance in stored L/M/R slots.

    This describes measured points relative to the fixed camera y axis. Without
    known mirror-source geometry it is an empirical reference diagnostic only,
    not evidence of a physical optical axis. No triangle shape is imposed.
    """
    points = np.asarray(points, dtype=np.float64)
    span = np.linalg.norm(points[..., 2, :] - points[..., 0, :], axis=-1)
    components = np.stack((points[..., 0, 0] + points[..., 2, 0] - 2*points[..., 1, 0],
                           points[..., 2, 1] - points[..., 0, 1]), axis=-1)
    valid = np.isfinite(points).all(axis=(-2, -1)) & np.isfinite(span) & (span > 1e-12)
    with np.errstate(divide='ignore', invalid='ignore'):
        components = components / span[..., None]
    components = np.where(valid[..., None], components, np.nan)
    return np.linalg.norm(components, axis=-1), components, valid


def empirical_template(points):
    """Mean of observed centered triples, with fixed native-pixel length and origin.

    No symmetrizing, affine alignment, area division, or fitted source geometry.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 3 or points.shape[1:] != (3, 2) or not np.isfinite(points).all():
        raise ValueError('template requires finite measured triples')
    if len(points) < 2:
        raise ValueError('a reference needs multiple frames')
    template = (points - points.mean(axis=1, keepdims=True)).mean(axis=0)
    radius = float(np.sqrt(np.mean(np.sum(template**2, axis=1))))
    if not np.isfinite(radius) or radius <= 1e-12:
        raise ValueError('collapsed empirical reference')
    return template, radius
