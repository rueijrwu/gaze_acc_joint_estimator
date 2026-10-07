"""Separate, strict artifacts for the four declared accommodation laws."""
import json
from pathlib import Path
import numpy as np
from .accommodation import PowerResponseModel, COEFFICIENT_ORDER
from .model import LOWER, UPPER
from .schema import (artifact as legacy_artifact, protocol_metadata,
                     NORMALIZATION, COORDINATE_ORDER, COVARIANCE_SEMANTICS)

SCHEMA = "conditional_power_response_v1"
POLICY = dict(theta_scale_deg=10.,state_scale=[10.,4.],targets_deg=[-10,-5,0,5,10],
    bounds_physical=[[-20,0],[20,6]],normalization=NORMALIZATION,
    coordinate_order=COORDINATE_ORDER,coefficient_order=COEFFICIENT_ORDER,
    covariance_semantics=COVARIANCE_SEMANTICS,pilot_model="conditional27",
    pilot_response_exponent=0.,retained_image_channels="xy",
    frame_state_policy="independent_two_variable_state_per_valid_frame",
    temporal_regularization_strength=0.,absolute_accuracy_thresholds=None,
    nominal_error_is_acceptance_gate=False,information_reference_increments={"theta_deg":1.,"A_diopters":1.})


def artifact(model,pilot,reference,sigma,provenance,diagnostics,coverage_policy):
    if not isinstance(model,PowerResponseModel) or pilot.name != "conditional27":
        raise ValueError("Power artifact requires a declared response and log27 pilot")
    if not coverage_policy or not provenance:
        raise ValueError("Declare coverage and training provenance before saving")
    obj = legacy_artifact(model,pilot,reference,sigma,provenance,diagnostics)
    obj.update(POLICY)
    obj.update(schema=SCHEMA,accommodation_response=model.metadata(),coverage_policy=coverage_policy,
        model=model.name,coefficient_count=27,metric_version="equal_exposure_three_way_crosscheck_v1")
    return obj


def from_object(obj,allow_failed=False):
    if obj.get("schema") != SCHEMA:
        raise ValueError("Power artifacts need an explicit schema; legacy is not inferred")
    for key,value in {**POLICY,**protocol_metadata()}.items():
        if key not in obj or obj[key] != value:
            raise ValueError(f"Power response policy mismatch: {key}")
    definition = obj.get("accommodation_response",{})
    if "coefficients" not in obj or obj["coefficients"] is None:
        raise ValueError("Power artifact requires explicit coefficients")
    model = PowerResponseModel(definition.get("exponent"),obj.get("coefficients"))
    if definition != model.metadata() or obj.get("model") != model.name or obj.get("coefficient_count") != 27:
        raise ValueError("Power family, exponent, capacity or coefficient order mismatch")
    if obj.get("calibration",{}).get("converged") is not True and not allow_failed:
        raise ValueError("Failed calibration is a checkpoint, not an inference model")
    if not obj.get("coverage_policy") or not obj.get("provenance"):
        raise ValueError("Power artifact lacks population/coverage provenance")
    pilot = np.asarray(obj.get("pilot_coefficients"),float)
    reference = np.asarray(obj.get("reference_state"),float)
    sigma = np.asarray(obj.get("coordinate_covariance"),float)
    if pilot.shape != (27,) or not np.isfinite(pilot).all():
        raise ValueError("Pilot must contain 27 finite log coefficients")
    if reference.shape != (2,) or not np.isfinite(reference).all() or np.any(reference<LOWER) or np.any(reference>UPPER):
        raise ValueError("Invalid physical covariance reference")
    if sigma.shape != (12,12) or not np.isfinite(sigma).all() or not np.allclose(sigma,sigma.T,rtol=1e-10,atol=1e-12):
        raise ValueError("Native P1/P4 covariance must be finite symmetric 12x12")
    np.linalg.cholesky(sigma)
    return model,obj


def load_model(path,allow_failed=False):
    return from_object(json.loads(Path(path).read_text()),allow_failed)
