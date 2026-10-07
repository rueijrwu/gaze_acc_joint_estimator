"""Apply a converged new model, preserving every row and optional P4 holdouts."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from .data import load_capture
from .geometry import context, summaries, signed_area
from .model import PositionModel
from .noise import reference_covariance, marginal
from .invert import invert, predict_holdout, score_holdout
from .schema import load_model, write_json, json_default, clean_json
from .diagnostics import frame_diagnostics


def apply(model_path, detection_path, output, every=1, holdouts=False):
    model, meta = load_model(model_path)
    if model.channels != 6:
        raise ValueError("Use the baseline scripts for summary-only application")
    pilot = PositionModel(int(meta["pilot_model"].removeprefix("conditional")), meta["pilot_coefficients"])
    sigma = np.asarray(meta["coordinate_covariance"])
    cap = load_capture(detection_path)
    expected = meta["provenance"]["correspondence"]
    if cap.name in expected and cap.metadata["pair_index"] != expected[cap.name]:
        raise ValueError("Capture correspondence differs from model provenance")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if every < 1:
        raise ValueError("every must be positive")
    selected = np.arange(0, len(cap.frame), every)
    good = selected[cap.ctx.valid[selected]]
    covs = reference_covariance(cap.p[good], pilot, np.array(meta["reference_state"]), sigma)
    covariance = dict(zip(good, covs))
    empirical_range = None
    training_path = Path(model_path).parent/'training_states.json'
    if training_path.exists():
        states = np.asarray(json.loads(training_path.read_text())['states'])
        empirical_range = [states.min(0), states.max(0)]
    selected = set(selected.tolist())
    estimated = 0
    with (output/"frames.jsonl").open("w") as f:
        for i in range(len(cap.frame)):
            record = dict(row=i, frame=int(cap.frame[i]), timestamp_ms=float(cap.timestamp[i]),
                          baseline_valid=bool(cap.baseline_valid[i]), p4_valid=cap.point_valid[i].tolist(),
                          selected=i in selected, estimated=False, state=None, predicted_q=None,
                          reason="not_sampled" if i not in selected else "invalid_input")
            inverse = dict(available=False)
            if i in selected and cap.ctx.valid[i]:
                ctx = context(cap.p[i])
                ix = np.repeat(cap.point_valid[i], 2).nonzero()[0]
                result = invert(model, ctx.r, cap.v[i].ravel()[ix], marginal(covariance[i], ix), ix)
                record["inverse"] = result
                inverse = result
                record["reason"] = result["reason"]
                if result["available"]:
                    x = np.array(result["state"])
                    v = model.predict(x, ctx.r).reshape(3, 2)
                    D, T = model.components(x)
                    record.update(estimated=True, state=result["state"],
                                  estimate_kind="all_three" if cap.baseline_valid[i] else "partial_two_P4",
                                  predicted_q=(ctx.c+ctx.ell*v).tolist(), predicted_d=D.tolist(),
                                  predicted_rho=float(abs(np.linalg.det(T))), predicted_T=T.tolist(),
                                  signed_predicted_det=float(np.linalg.det(T)),
                                  observed_signed_det=float(signed_area(cap.q[i])/ctx.signed_area) if cap.baseline_valid[i] else None)
                    record["observed_summaries"] = summaries(ctx, cap.q[i]).tolist() if cap.baseline_valid[i] else None
                    estimated += 1
                if holdouts:
                    record["holdouts"] = []
                    for j in range(3):
                        ix = [k for k in range(6) if k//2 != j]
                        if not cap.point_valid[i][np.arange(3)!=j].all():
                            prediction = dict(available=False, held_point=j, reason="insufficient_retained_P4")
                        else:
                            prediction = predict_holdout(model, ctx, j, cap.v[i].ravel()[ix], covariance[i])
                        q = cap.q[i, j] if cap.point_valid[i, j] else np.full(2, np.nan)
                        record["holdouts"].append(score_holdout(prediction, q, ctx.ell))
            record['diagnostics'] = frame_diagnostics(model, meta, cap.p[i], cap.point_valid[i],
                inverse, empirical_range, covariance.get(i))
            # Clean nonfinite invalid-geometry diagnostics while preserving each row.
            f.write(json.dumps(clean_json(record), default=json_default, allow_nan=False)+"\n")
    write_json(output/"summary.json", dict(original_rows=len(cap.frame), sampled_rows=len(selected),
        estimated_rows=estimated, every=every, holdouts=holdouts, source_sha256=cap.sha256,
        model_path=str(model_path), model_schema=meta["schema"], physiological_validation=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("detections", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--every", type=int, default=1)
    parser.add_argument("--holdouts", action="store_true")
    args = parser.parse_args()
    apply(args.model, args.detections, args.output, args.every, args.holdouts)
