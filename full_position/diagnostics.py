"""Add covariance and context-support diagnostics to existing experiment outputs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from .data import load_reviewed
from .geometry import context, area_gradient
from .model import PositionModel
from .noise import reference_covariance, marginal, whitening
from .schema import load_model, write_json


def enrich(run):
    run = Path(run)
    captures, _, _ = load_reviewed(Path(__file__).resolve().parents[1])
    for path in sorted(run.glob("*/*/model.json")):
        frames_path = path.parent/"frames.json"
        if not frames_path.exists():
            continue
        model, meta = load_model(path)
        pilot = PositionModel(27, meta["pilot_coefficients"])
        sigma = np.array(meta["coordinate_covariance"])
        reference = np.array(meta["reference_state"])
        trained = json.loads((path.parent/"training_states.json").read_text())
        states = np.array(trained["states"])
        low, high = states.min(0), states.max(0)
        rlow = np.array(meta["provenance"]["conditional_context_support"]["r_min"])
        rhigh = np.array(meta["provenance"]["conditional_context_support"]["r_max"])
        records = []
        for row in json.loads(frames_path.read_text()):
            cap = captures[row["capture"]]
            ctx = context(cap.p[row["row"]])
            record = dict(row=row["row"], capture=row["capture"], fixation=row["fixation"],
                          p1_edge_condition=float(ctx.edge_condition), residual_covariance=None,
                          state_covariance_physical=None, context_outside_training_extrema=None,
                          state_outside_empirical_training_range=None)
            if ctx.valid:
                R = reference_covariance(cap.p[row["row"]:row["row"]+1], pilot, reference, sigma,
                                          model.channels == 2)[0]
                record["residual_covariance"] = R.tolist()
                grad = area_gradient(ctx.p).ravel()
                record["p1_area_signal_to_noise"] = float(abs(ctx.signed_area)/np.sqrt(grad@sigma[:6, :6]@grad))
                record["context_outside_training_extrema"] = bool(np.any((ctx.r < rlow) | (ctx.r > rhigh)))
                if row["estimated"]:
                    x = np.array([row["theta"], row["A"]])
                    record["state_outside_empirical_training_range"] = bool(np.any((x < low) | (x > high)))
                    indices = np.repeat(cap.point_valid[row["row"]], 2).nonzero()[0] if model.channels == 6 else np.arange(2)
                    _, J = model.predict(x, ctx.r, True)
                    WJ = whitening(marginal(R, indices))@J[indices]
                    if row["rank"] == 2 and not row["at_bound"] and not row["ambiguous"]:
                        record["state_covariance_physical"] = np.linalg.inv(WJ.T@WJ).tolist()
            records.append(record)
        write_json(path.parent/"frame_uncertainty.json", dict(records=records,
            empirical_training_state_range=[low.tolist(), high.tolist()],
            context_support_definition="componentwise training extrema diagnostic; not a validated support envelope",
            covariance_definition="local fixed-calibration conditional covariance; excludes coefficient uncertainty and model discrepancy"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    enrich(args.run)
