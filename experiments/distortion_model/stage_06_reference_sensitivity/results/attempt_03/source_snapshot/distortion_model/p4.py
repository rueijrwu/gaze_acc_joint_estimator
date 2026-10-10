"""Independent P4 reference diagnostics; no P4 nuisance-scale normalization."""
from dataclasses import dataclass
import numpy as np
from .alignment import p1_balance
from .optics import local_angles


@dataclass(frozen=True)
class DualZeros:
    omega1: float
    omega4: float

    def __post_init__(self):
        if not np.isfinite([self.omega1, self.omega4]).all():
            raise ValueError('finite independent visual offsets required')

    @property
    def delta14(self):
        return self.omega4-self.omega1

    def angles(self, theta_visual):
        return local_angles(theta_visual, self.omega1, self.omega4)

    def visual_from_p4(self, xi4):
        return np.asarray(xi4, dtype=float)+self.omega4

    def p1_from_p4(self, xi4):
        return np.asarray(xi4, dtype=float)+self.delta14


def corrected_p4_pattern(points, p1_scale):
    """Translation-free shape in reference pixels, divided only by P1 g.

    Centering is a calibration display, not a spatial optical-center estimate.
    The displacement channel is kept separately by the caller. No division by
    a P4 size, area, radius, fitted M or symmetry score occurs here.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 3 or points.shape[1:] != (3, 2):
        raise ValueError('expected N corresponding P4 triples')
    scale = np.asarray(p1_scale, dtype=float)
    if scale.shape != (len(points),):
        raise ValueError('one P1 scale per frame required')
    valid = np.isfinite(points).all(axis=(1, 2)) & np.isfinite(scale) & (scale>0)
    with np.errstate(divide='ignore', invalid='ignore'):
        centered = (points-points.mean(axis=1, keepdims=True))/scale[:, None, None]
    return np.where(valid[:, None, None], centered, np.nan), valid


def p4_balance(points, p1_scale):
    """Observed fixed-axis balance in source-corresponding slots, without sorting."""
    corrected, valid = corrected_p4_pattern(points, p1_scale)
    score, components, balanced_valid = p1_balance(corrected)
    return score, components, valid & balanced_valid


def select_shared_reference(records):
    """Equal-capture mean score on the fixed five nominal bins; stable tie order.

    Records must cover the full four-by-five schedule. No fitted states or
    training residuals select the minimum. The offset is assigned separately
    from the visual snapshot in the low-demand reference capture.
    """
    targets = [-10., -5., 0., 5., 10.]
    captures = sorted({r['capture'] for r in records})
    if len(captures)!=4 or len(records)!=20:
        raise ValueError('reference selection requires all twenty exposures')
    profiles = []
    for target in targets:
        selected = [r for r in records if r['nominal_target_deg']==target]
        if len(selected)!=4 or sorted(r['capture'] for r in selected)!=captures:
            raise ValueError('missing or duplicated capture/target')
        scores = np.array([r['score']['mean'] for r in selected], dtype=float)
        if not np.isfinite(scores).all() or any(r['valid']<=0 for r in selected):
            raise ValueError('no supported reference diagnostic population')
        profiles.append({'nominal_target_deg':target, 'equal_capture_mean_score':float(scores.mean()),
                         'across_capture_sd':float(scores.std()),
                         'exposures':[r['exposure'] for r in selected]})
    chosen = min(profiles, key=lambda r:r['equal_capture_mean_score'])
    return chosen, profiles


def near_reference(theta_visual, zeros, half_width=2.5):
    """Diagnostic window only; never gates the full calibration population."""
    if not np.isfinite(half_width) or half_width<=0:
        raise ValueError('positive finite near-reference window required')
    _, xi4 = zeros.angles(theta_visual)
    return np.isfinite(xi4) & (np.abs(xi4)<=half_width)
