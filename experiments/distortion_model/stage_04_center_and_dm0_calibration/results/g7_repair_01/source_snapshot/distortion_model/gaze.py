"""Centroid inverse bootstrap, stored separately from the final forward D law."""
import numpy as np


def fit_bootstrap(displacement_means, targets):
    displacement_means = np.asarray(displacement_means, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if displacement_means.shape != (5,) or targets.shape != (5,):
        raise ValueError("bootstrap requires exactly five reference fixation summaries")
    if not np.isfinite(displacement_means).all() or not np.isfinite(targets).all():
        raise ValueError("nonfinite reference summaries")
    origin = float(displacement_means.mean())
    scale = float(displacement_means.std())
    if scale <= np.finfo(float).eps * max(1., abs(origin)):
        raise ValueError("collapsed reference displacement")
    design = np.column_stack((np.ones(5), (displacement_means - origin) / scale))
    coefficients, _, rank, singular = np.linalg.lstsq(design, targets, rcond=None)
    if rank != 2 or abs(coefficients[1]) <= 1e-12:
        raise ValueError("bootstrap has no invertible linear gain")
    order = np.argsort(targets)
    increments = np.diff(displacement_means[order])
    ordered = bool(np.all(increments > 0) or np.all(increments < 0))
    return {"degree": 1, "basis": "[1, (displacement_px-origin_px)/scale_px]",
            "coefficients_deg": coefficients.tolist(), "origin_px": origin, "scale_px": scale,
            "gain_deg_per_px": float(coefficients[1] / scale), "rank": int(rank),
            "condition_number": float(singular[0] / singular[-1]), "ordered": ordered,
            "reference_displacement_domain_px": [float(displacement_means.min()), float(displacement_means.max())],
            "reference_mean_predictions_deg": (design @ coefficients).tolist(),
            "reference_mean_residuals_deg": (design @ coefficients - targets).tolist(),
            "interpretation": "provisional centroid inverse; reference g=1; not forward D or physiological truth"}


def apply_bootstrap(displacement, bootstrap):
    c0, c1 = bootstrap["coefficients_deg"]
    return c0 + c1 * (np.asarray(displacement) - bootstrap["origin_px"]) / bootstrap["scale_px"]


def centroid_displacement(p1, p4, held_point=None):
    """Calibration centroid or retained-only proposal; never touches omitted P4."""
    keep = [j for j in range(3) if j != held_point]
    return np.asarray(p4)[..., keep, :].mean(axis=-2) - np.asarray(p1).mean(axis=-2)
