"""Explicit new-artifact compatibility checks and JSON persistence."""
from __future__ import annotations
import json
import hashlib
from pathlib import Path
import numpy as np
from .model import PositionModel, SummaryModel

SCHEMA = "exp5_full_position_v1"


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # JSON stays standards-compliant; non-finite diagnostics become explicit null.
    def clean(v):
        if isinstance(v, np.ndarray):
            return clean(v.tolist())
        if isinstance(v, np.generic):
            return clean(v.item())
        if isinstance(v, float) and not np.isfinite(v):
            return None
        if isinstance(v, dict):
            return {k: clean(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [clean(x) for x in v]
        return v
    path.write_text(json.dumps(clean(value), indent=2, default=json_default, allow_nan=False)+"\n")


def artifact(model, pilot, reference, sigma, provenance, diagnostics):
    return dict(schema=SCHEMA, model=model.name, coefficients=model.beta,
        theta_scale_deg=10., state_scale=[10., 4.], targets_deg=[-10, -5, 0, 5, 10],
        bounds_physical=[[-20, 0], [20, 6]],
        information_reference_increments={"theta_deg": 1., "A_diopters": 1.},
        normalization="sqrt(abs(P1_triangle_area)); P1_centroid; no_rotation",
        coordinate_order=["v1x", "v1y", "v2x", "v2y", "v3x", "v3y"],
        coefficient_order="Dx[1,A,t,tL,t2,t2L,t3]; Dy,T11,T12,T21,T22[1,t,L,tL,(t2,t2L)]"
            if model.channels == 6 else "d[1,A,t,tL,t2,t2L,t3]; rho[1,t,t2,L,tL,t2L]",
        reference_state=reference, pilot_coefficients=pilot.beta,
        pilot_model=pilot.name, coordinate_covariance=sigma, provenance=provenance,
        calibration=diagnostics, source_grid_geometry=None,
        physiological_validation=False)


def load_model(path, allow_failed=False):
    obj = json.loads(Path(path).read_text())
    if obj.get("schema") != SCHEMA or obj.get("theta_scale_deg") != 10.:
        raise ValueError("Incompatible or missing full-position schema/gaze scale")
    if obj.get("state_scale") != [10., 4.] or obj.get("targets_deg") != [-10, -5, 0, 5, 10]:
        raise ValueError("Incompatible state encoding or target grid")
    if not obj["calibration"]["converged"] and not allow_failed:
        raise ValueError("Nonconverged calibration checkpoint cannot be applied as a model")
    name = obj["model"]
    model = SummaryModel(obj["coefficients"]) if name == "two_channel13" else PositionModel(int(name.removeprefix("conditional")), obj["coefficients"])
    if model.beta.shape != (model.size,) or not np.isfinite(model.beta).all():
        raise ValueError("Invalid coefficient array")
    return model, obj


def source_hashes(root):
    root = Path(root)
    paths = list((root/"full_position").glob("*.py")) + [root/"Theory.md", root/"ESTIMATOR_PLAN.md"]
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
