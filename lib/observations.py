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
