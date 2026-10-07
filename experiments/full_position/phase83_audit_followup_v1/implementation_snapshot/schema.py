"""Explicit new-artifact compatibility checks and JSON persistence."""
from __future__ import annotations
import json
import hashlib
from pathlib import Path
import numpy as np
from .model import PositionModel, SummaryModel

SCHEMA = "exp5_full_position_v1"
NORMALIZATION = "sqrt(abs(P1_triangle_area)); P1_centroid; no_rotation"
COORDINATE_ORDER = ["v1x", "v1y", "v2x", "v2y", "v3x", "v3y"]
POSITION_ORDER = "Dx[1,A,t,tL,t2,t2L,t3]; Dy,T11,T12,T21,T22[1,t,L,tL,(t2,t2L)]"
SUMMARY_ORDER = "d[1,A,t,tL,t2,t2L,t3]; rho[1,t,t2,L,tL,t2L]"
COVARIANCE_SEMANTICS = "mapped_native_pixel_P1_then_P4; fixed_training_effective_coordinate_noise"


def protocol_metadata():
    """User-supplied protocol annotation; no new per-frame truth measurement."""
    return dict(state_variables=["theta_x_deg", "A_D"], calibration_theta_y_nominal_deg=0.,
        calibration_theta_y_role="protocol_constraint_not_framewise_ground_truth",
        image_axis_convention="stored pixel x/y axes; P1 normalization without rotation; image-y is not vertical gaze",
        protocol_provenance="user clarification recorded in PHASE_8_3_AUDIT.md, 2026-10-07")


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def clean_json(v):
    """Nonfinite diagnostics become explicit null in JSON and JSONL."""
    if isinstance(v, np.ndarray):
        return clean_json(v.tolist())
    if isinstance(v, np.generic):
        return clean_json(v.item())
    if isinstance(v, float) and not np.isfinite(v):
        return None
    if isinstance(v, dict):
        return {k: clean_json(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [clean_json(x) for x in v]
    return v


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(value), indent=2, default=json_default, allow_nan=False)+"\n")


def artifact(model, pilot, reference, sigma, provenance, diagnostics):
    return dict(schema=SCHEMA, model=model.name, coefficients=model.beta, **protocol_metadata(),
        theta_scale_deg=10., state_scale=[10., 4.], targets_deg=[-10, -5, 0, 5, 10],
        bounds_physical=[[-20, 0], [20, 6]],
        information_reference_increments={"theta_deg": 1., "A_diopters": 1.},
        normalization=NORMALIZATION, coordinate_order=COORDINATE_ORDER,
        coefficient_order=POSITION_ORDER if model.channels == 6 else SUMMARY_ORDER,
        covariance_semantics=COVARIANCE_SEMANTICS,
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
    for key in ("state_variables", "calibration_theta_y_nominal_deg", "calibration_theta_y_role"):
        if key in obj and obj[key] != protocol_metadata()[key]:
            raise ValueError(f"Incompatible calibration protocol: {key}")
    if not obj["calibration"]["converged"] and not allow_failed:
        raise ValueError("Nonconverged calibration checkpoint cannot be applied as a model")
    name = obj["model"]
    if name not in ("two_channel13", "conditional27", "conditional37"):
        raise ValueError("Unknown model family")
    expected = dict(normalization=NORMALIZATION, coordinate_order=COORDINATE_ORDER,
        coefficient_order=SUMMARY_ORDER if name == "two_channel13" else POSITION_ORDER,
        bounds_physical=[[-20,0],[20,6]], pilot_model="conditional27")
    for key, value in expected.items():
        if obj.get(key) != value:
            raise ValueError(f"Incompatible {key}")
    # Original v1 artifacts implicitly use this covariance convention.
    if obj.get("covariance_semantics", COVARIANCE_SEMANTICS) != COVARIANCE_SEMANTICS:
        raise ValueError("Incompatible covariance semantics")
    if obj.get("information_reference_increments", {"theta_deg": 1., "A_diopters": 1.}) != {"theta_deg": 1., "A_diopters": 1.}:
        raise ValueError("Incompatible information increments")
    pilot = np.asarray(obj.get("pilot_coefficients"), float)
    reference = np.asarray(obj.get("reference_state"), float)
    sigma = np.asarray(obj.get("coordinate_covariance"), float)
    if pilot.shape != (27,) or not np.isfinite(pilot).all():
        raise ValueError("Invalid pilot coefficients")
    if reference.shape != (2,) or not np.isfinite(reference).all() or np.any(reference < [-20,0]) or np.any(reference > [20,6]):
        raise ValueError("Invalid reference state")
    if sigma.shape != (12,12) or not np.isfinite(sigma).all() or not np.allclose(sigma,sigma.T,atol=1e-12,rtol=1e-10):
        raise ValueError("Invalid coordinate covariance")
    if np.linalg.eigvalsh(sigma).min() <= 0:
        raise ValueError("Coordinate covariance must be positive definite")
    provenance = obj.get('provenance', {})
    if 'anchor_range' in provenance:
        anchor = np.asarray(provenance['anchor_range'],float)
        if anchor.shape != (2,2) or not np.isfinite(anchor).all() or np.any(anchor[0]>anchor[1]):
            raise ValueError('Invalid anchor support')
    support = provenance.get('conditional_context_support')
    if support is not None:
        low,high = np.asarray(support.get('r_min'),float), np.asarray(support.get('r_max'),float)
        if low.shape!=(3,2) or high.shape!=(3,2) or not np.isfinite(low).all() or not np.isfinite(high).all() or np.any(low>high):
            raise ValueError('Invalid P1 context support')
        if not set(support.get('signed_P1_area_branches',[])).issubset({-1.,1.}):
            raise ValueError('Invalid P1 area parity')
    model = SummaryModel(obj["coefficients"]) if name == "two_channel13" else PositionModel(int(name.removeprefix("conditional")), obj["coefficients"])
    if model.beta.shape != (model.size,) or not np.isfinite(model.beta).all():
        raise ValueError("Invalid coefficient array")
    return model, obj


def source_hashes(root):
    root = Path(root)
    paths = list((root/"full_position").glob("*.py")) + [root/"Theory.md", root/"ESTIMATOR_PLAN.md"]
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
