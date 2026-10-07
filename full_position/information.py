"""Leakage-safe retained-y linear information masks on frozen responses."""
from __future__ import annotations
import numpy as np
from .geometry import context
from .invert import STARTS, invert, predict_holdout
from .noise import marginal, reference_covariance

MASKS = ("x", "x_y_common", "x_y_difference", "x_y_common_difference")


def retained_transform(held, mask):
    if held not in range(3) or mask not in (*MASKS, "xy"):
        raise ValueError("Unknown excluded point or retained mask")
    kept = np.array([i for i in range(6) if i//2 != held])
    # Retained order: xa, ya, xb, yb. Marginalize before transforming.
    rows = [[1., 0., 0., 0.], [0., 0., 1., 0.]]
    if mask in ("x_y_common", "x_y_common_difference"):
        rows.append([0., .5, 0., .5])
    if mask in ("x_y_difference", "x_y_common_difference"):
        rows.append([0., 1., 0., -1.])
    return kept, np.eye(4) if mask == "xy" else np.asarray(rows)


class TransformedResponse:
    """The same response with H applied to values, Jacobians and Hessians."""
    def __init__(self, model, kept, transform):
        self.position, self.kept, self.transform = model, np.asarray(kept), np.asarray(transform)
        self.channels = len(transform)

    def predict(self, x, r, derivatives=False):
        if derivatives:
            value, jac = self.position.predict(x, r, True)
            return (np.einsum("ij,...j->...i", self.transform, value[..., self.kept]),
                    np.einsum("ij,...ja->...ia", self.transform, jac[..., self.kept, :]))
        return np.einsum("ij,...j->...i", self.transform, self.position.predict(x, r)[..., self.kept])

    def state_hessian(self, x, r):
        return np.einsum("ij,...jab->...iab", self.transform,
                         self.position.state_hessian(x, r)[..., self.kept, :, :])


def predict_transformed(model, ctx, held, retained, full_cov, mask, starts=STARTS):
    """retained contains only observations used by this mask, never excluded y."""
    kept, H = retained_transform(held, mask)
    if mask == "x":
        return predict_holdout(model, ctx, held, retained, full_cov, starts, channels="x")
    if mask == "xy":
        return predict_holdout(model, ctx, held, retained, full_cov, starts, channels="xy")
    if np.shape(retained) != (4,):
        raise ValueError("Y masks require exactly the four retained coordinates")
    response = TransformedResponse(model, kept, H)
    R = marginal(full_cov, kept)
    result = invert(response, ctx.r, H@retained, H@R@H.T, starts=starts)
    result.update(held_point=int(held), retained_image_channels=mask, retained_indices=kept.tolist(),
                  measurement_transform=H.tolist(), statistical_covariance=(H@R@H.T).tolist())
    if not result["available"]:
        return result
    omitted = np.array([2*held, 2*held+1])
    result["predictions"] = []
    for branch in result["plausible_branches"]:
        v = model.predict(np.asarray(branch["state"]), ctx.r)[omitted]
        result["predictions"].append(dict(state=branch["state"], normalized=v.tolist(),
                                            pixel=(ctx.c+ctx.ell*v).tolist()))
    result["testable"] = bool(result["rank"] == 2 and not result["ambiguous"])
    if result["testable"] and not result["at_bound"]:
        _, J = model.predict(np.asarray(result["state"]), ctx.r, True)
        K = np.linalg.solve(H@R@H.T, H@J[kept])
        B = np.linalg.solve(J[kept].T@H.T@K, K.T)@H
        cross = full_cov[np.ix_(omitted, kept)]
        V = full_cov[np.ix_(omitted, omitted)]+J[omitted]@B@R@B.T@J[omitted].T
        V -= cross@B.T@J[omitted].T+J[omitted]@B@cross.T
        V = (V+V.T)/2
        if np.linalg.eigvalsh(V).min() > 0:
            result["predictive_covariance_normalized"] = V.tolist()
    return result


def predict_raw_holdout(model, p, q, point_valid, held, pilot, reference, sigma,
                        mask="x", starts=STARTS):
    """Raw-array prediction boundary. No labels or all-three states are accepted.

    Validity is an explicit fixed mask; only retained coordinates are normalized.
    Noise depends exclusively on P1 and the frozen training pilot/reference.
    """
    kept, _ = retained_transform(held, mask)
    ctx = context(p)
    if not ctx.valid:
        return dict(available=False, held_point=int(held), reason="invalid_P1_geometry", branches=[])
    if not np.asarray(point_valid, bool)[np.arange(3) != held].all():
        return dict(available=False, held_point=int(held), reason="insufficient_retained_P4", branches=[])
    if mask == "x":
        kept = kept[::2]
    raw = np.asarray(q)[kept//2, kept % 2]
    retained = (raw-ctx.c[kept % 2])/ctx.ell
    cov = reference_covariance(np.asarray(p)[None], pilot, np.asarray(reference), np.asarray(sigma))[0]
    result = predict_transformed(model, ctx, held, retained, cov, mask, starts)
    result.update(retained_image_channels=mask, retained_indices=kept.tolist())
    return result
