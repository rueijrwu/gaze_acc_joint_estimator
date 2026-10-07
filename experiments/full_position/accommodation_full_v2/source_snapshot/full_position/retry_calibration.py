"""Strict continuation of the preserved failed Phase8.2 training checkpoint."""
import argparse
import gzip
import json
from pathlib import Path
import numpy as np
from .calibrate import fit
from .model import PositionModel
from .noise import reference_covariance
from .schema import artifact, clean_json, source_hashes, write_json
from .sensitivity import checked_training, digest, read_json, settings


def run(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = root/"experiments/full_position/audit_polished_v1"
    failed = root/"experiments/full_position/joint_sensitivity_v1/variants/prior_weak/capture_1/conditional27"
    _, _, data, anchors, frozen = checked_training(root, source, "capture_1")
    _, meta, _ = frozen["conditional27"]
    change = settings(meta, anchors, "prior_weak")
    pilot = PositionModel(27, meta["pilot_coefficients"])
    sigma = np.asarray(meta["coordinate_covariance"])
    ref = np.asarray(change["reference_state"])
    covariance = reference_covariance(data["p"], pilot, ref, sigma)
    initial = np.asarray(read_json(failed/"training_states.json")["states"])
    with gzip.open(output/"calibration_candidates.jsonl.gz", "wt") as stream:
        def checkpoint(diag, beta, states):
            stream.write(json.dumps(clean_json(dict(diagnostics=diag, coefficients=beta, states=states)), allow_nan=False)+"\n")
            stream.flush()
        model, states, diag = fit(PositionModel(27), data["v"], data["r"], covariance, data["groups"],
            anchors, change["prior_strength"], starts=1, max_nfev=300,
            additional_initial_states=[initial], anchor_scales=change["anchor_scales"],
            checkpoint=checkpoint, continuation_stages=3)
    provenance = dict(meta["provenance"], source_sha256=source_hashes(root),
        retry_initial_states_sha256=digest(failed/"training_states.json"),
        phase="8.2_audit_strict_continuation", evaluation_usage="training-only solver diagnostic; no evaluation")
    write_json(output/("model.json" if diag["converged"] else "failed_checkpoint.json"),
               artifact(model, pilot, ref, sigma, provenance, diag))
    write_json(output/"training_states.json", dict(states=states, rows=data["rows"], groups=data["groups"], nominal_anchors=anchors))
    write_json(output/"completion.json", dict(complete=True, converged=diag["converged"], calibration=diag,
        original_failed_fit_preserved=True, evaluation_performed=False, implementation_hashes=source_hashes(root)))
    print(json.dumps(dict(converged=diag["converged"], selected_start=diag["selected_start"], alternatives=diag["alternatives"])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.root, args.output)
