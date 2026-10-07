"""Shared P1/P4 measurements; returns [e_n, rho4], with d = -e_n."""
import numpy as np


def measurements(arrays, separation, min_p1_separation=1e-6):
    frame = np.asarray(arrays["frame_index"])
    count = len(frame)
    p1 = np.asarray(arrays["p1_points"], dtype=float)
    p4 = np.asarray(arrays["p4_points"], dtype=float)
    if p1.shape != (count, 2, 2) or p4.shape != (count, 2, 2):
        raise ValueError("p1_points and p4_points must have shape (frames, 2, 2)")
    pair_valid = np.asarray(arrays["pair_valid"], dtype=bool)
    timestamp = np.asarray(arrays["timestamp_ms"])
    if pair_valid.shape != (count,) or timestamp.shape != (count,):
        raise ValueError("pair_valid and timestamp_ms must match frame_index")
    delta1 = p1[:, 1, 0] - p1[:, 0, 0]
    delta4 = p4[:, 1, 0] - p4[:, 0, 0]
    if separation == "absolute_x":
        delta1, delta4 = np.abs(delta1), np.abs(delta4)
    e = p1[:, :, 0].mean(axis=1) - p4[:, :, 0].mean(axis=1)
    valid = (pair_valid & np.isfinite(p1).all(axis=(1, 2))
             & np.isfinite(p4).all(axis=(1, 2)) & (np.abs(delta1) >= min_p1_separation))
    y = np.full((count, 2), np.nan)
    np.divide(np.column_stack([e, delta4]), delta1[:, None], out=y, where=valid[:, None])
    valid &= np.isfinite(y).all(axis=1)
    return y, valid


RELATIVE_SCHEMA = dict(basis=['m', 'S1', 'S4'], units='pixels',
    ordering='stored image-horizontal left/right; no persistent illuminator identity',
    crossed_basis=['P4_left-P1_right', 'P4_right-P1_left', 'S4'],
    normalized_basis=['d=m/S1', 'rho4=S4/S1'],
    covariance='full covariance in the declared independent basis')
CROSSED_TRANSFORM = np.array([[1., -.5, -.5], [1., .5, .5], [0., 0., 1.]])


def relative_measurements(arrays, min_p1_separation=1e-6):
    """Independent relative pixels on the stored ordered branch.

    Return all original rows and explicit ordering failures. Legacy measurement
    validity is retained separately; reversed pairs are never silently relabeled.
    Pupil/fixation gates remain the caller's responsibility.
    """
    legacy, legacy_valid = measurements(arrays, 'absolute_x', min_p1_separation)
    p1 = np.asarray(arrays['p1_points'], dtype=float)
    p4 = np.asarray(arrays['p4_points'], dtype=float)
    s1, s4 = p1[:, 1, 0]-p1[:, 0, 0], p4[:, 1, 0]-p4[:, 0, 0]
    ordering_valid = (s1 > 0) & (s4 > 0)
    valid = legacy_valid & ordering_valid
    m = p4[:, :, 0].mean(axis=1)-p1[:, :, 0].mean(axis=1)
    z = np.column_stack([m, s1, s4])
    crossed = np.column_stack([p4[:, 0, 0]-p1[:, 1, 0], p4[:, 1, 0]-p1[:, 0, 0], s4])
    normalized = legacy.copy(); normalized[:, 0] *= -1
    return dict(z=z, crossed=crossed, normalized=normalized, valid=valid,
                legacy_valid=legacy_valid, ordering_valid=ordering_valid,
                ordering_failure=legacy_valid & ~ordering_valid,
                schema=RELATIVE_SCHEMA.copy())
