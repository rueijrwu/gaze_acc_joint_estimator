"""Linear, translation invariant relative coordinates and covariance marginals."""
import numpy as np


def relative_map():
    """Native order: P1[0:3,x/y], then corresponding P4[0:3,x/y]."""
    matrix = np.zeros((10, 12), dtype=np.float64)
    for edge, point in enumerate((1, 2)):
        for axis in range(2):
            matrix[2 * edge + axis, 2 * point + axis] = 1
            matrix[2 * edge + axis, axis] = -1
    for point in range(3):
        for axis in range(2):
            row = 4 + 2 * point + axis
            matrix[row, 6 + 2 * point + axis] = 1
            matrix[row, axis:6:2] = -1 / 3
    return matrix


def retained_indices(held_point=None):
    if held_point is None:
        return np.arange(10)
    if held_point not in (0, 1, 2):
        raise ValueError("held_point must be 0, 1 or 2")
    return np.delete(np.arange(10), [4 + 2 * held_point, 5 + 2 * held_point])


def relative_coordinates(p1, p4, held_point=None):
    """Never read the held P4 coordinate, even for initialization or validity."""
    p1, p4 = np.asarray(p1, dtype=np.float64), np.asarray(p4, dtype=np.float64)
    edges = (p1[..., 1:, :] - p1[..., :1, :]).reshape(p1.shape[:-2] + (4,))
    keep = [j for j in range(3) if j != held_point]
    retained = p4[..., keep, :] - p1.mean(axis=-2)[..., None, :]
    return np.concatenate((edges, retained.reshape(p1.shape[:-2] + (2 * len(keep),))), axis=-1)


def relative_covariance(native_covariance, held_point=None):
    matrix = relative_map()[retained_indices(held_point)]
    return matrix @ native_covariance @ matrix.T


def p1_scale(reference_edges, observed_edges, edge_covariance):
    """Positive P1-only GLS profile; invalid proposals return NaN and False."""
    a, e = np.broadcast_arrays(reference_edges, observed_edges)
    weighted_a = np.linalg.solve(edge_covariance, a[..., None])[..., 0]
    energy = np.sum(a * weighted_a, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.sum(e * weighted_a, axis=-1) / energy
    valid = np.isfinite(scale) & np.isfinite(energy) & (energy > 1e-12) & (scale > 0)
    return np.where(valid, scale, np.nan), valid
