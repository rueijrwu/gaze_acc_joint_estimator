"""Reproduce the supplemental failed-fit stopping check; never accept a model."""
from pathlib import Path
import gzip
import hashlib
import json

import numpy as np
from scipy.optimize import least_squares

from full_position.calibrate import ProfiledProblem, projected_gradient
from full_position.model import PositionModel, STATE_SCALE, LOWER, UPPER
from full_position.noise import reference_covariance
from full_position.schema import clean_json, source_hashes
from full_position.sensitivity import checked_training, settings


def main():
    output = Path(__file__).resolve().parent
    root = output.parents[2]
    config = json.loads((output / "config.json").read_text())
    actual = source_hashes(root)
    for path, expected in config["implementation_hashes"].items():
        if path.startswith("full_position/"):
            if actual.get(path) != expected:
                raise ValueError(f"Study implementation changed: {path}")
        else:
            saved = output / "design_snapshot" / path
            if hashlib.sha256(saved.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Study design snapshot changed: {path}")
    source = root / "experiments/full_position/audit_polished_v1"
    _, _, data, anchors, frozen = checked_training(root, source, "capture_1")
    _, meta, _ = frozen["conditional27"]
    change = settings(meta, anchors, "prior_weak")
    covariance = reference_covariance(data["p"], PositionModel(27, meta["pilot_coefficients"]),
        np.asarray(change["reference_state"]), np.asarray(meta["coordinate_covariance"]))
    problem = ProfiledProblem(PositionModel(27), data["v"], data["r"], covariance,
        data["groups"], anchors, change["prior_strength"], change["anchor_scales"])
    folder = output / "variants/prior_weak/capture_1/conditional27"
    states = np.asarray(json.loads((folder / "training_states.json").read_text())["states"])
    scale = np.tile(STATE_SCALE, len(states))
    lower, upper = np.tile(LOWER, len(states)), np.tile(UPPER, len(states))
    def measure(z):
        problem.update(z)
        gradient = problem.vjp(problem.residual) / scale
        return dict(cost=float(problem.residual @ problem.residual / 2),
            projected_stationarity_physical=projected_gradient(problem.x.ravel(), gradient, lower, upper))
    initial = (states / STATE_SCALE).ravel()
    before = measure(initial)
    result = least_squares(problem.fun, initial, jac=problem.jac,
        bounds=(lower / scale, upper / scale), method="trf", tr_solver="lsmr", x_scale=1.,
        ftol=None, xtol=1e-12, gtol=None, max_nfev=100,
        tr_options={"atol": 1e-9, "btol": 1e-9, "maxiter": 200})
    after = measure(result.x)
    record = json.loads((output / "supplemental_stop_check.json").read_text())
    record.update(before=before, after=after, additional_nfev=int(result.nfev),
                  solver_status=int(result.status), implementation_hashes=config["implementation_hashes"],
                  initial_training_states_sha256=hashlib.sha256((folder / "training_states.json").read_bytes()).hexdigest())
    (output / "supplemental_stop_check.json").write_text(json.dumps(clean_json(record), indent=2) + "\n")
    with gzip.open(output / "supplemental_stop_checkpoint.json.gz", "wt") as stream:
        json.dump(clean_json(dict(accepted=False, diagnostics=record,
            coefficients=problem.beta, states=problem.x)), stream, separators=(",", ":"), allow_nan=False)
    print(json.dumps(after))


if __name__ == "__main__":
    main()
