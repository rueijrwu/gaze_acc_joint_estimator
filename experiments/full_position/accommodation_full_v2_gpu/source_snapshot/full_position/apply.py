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
from .invert import invert, score_holdout
from .information import predict_raw_holdout
from .schema import load_model, write_json, json_default, clean_json
from .diagnostics import frame_diagnostics, enrich_subset_records


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
    population, source_frames, held_records = [], [], []
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
                        prediction = predict_raw_holdout(model, cap.p[i], cap.q[i], cap.point_valid[i], j,
                            pilot, np.asarray(meta["reference_state"]), sigma, "xy")
                        q = cap.q[i, j] if cap.point_valid[i, j] else np.full(2, np.nan)
                        score = score_holdout(prediction, q, ctx.ell)
                        score.update(capture=cap.name, fixation=0, row=i)
                        record["holdouts"].append(score)
                        held_records.append(score)
            if holdouts and i in selected:
                population.append(dict(capture=cap.name, fixation=0, row=i, frame=int(cap.frame[i]),
                    selected_for_evaluation=True, p1_valid_geometry=bool(cap.ctx.valid[i]),
                    **{f"p4_{j+1}_valid": bool(cap.point_valid[i, j]) for j in range(3)}))
                source_frames.append(dict(capture=cap.name, fixation=0, row=i, frame=int(cap.frame[i]),
                    estimated=record["estimated"], estimate_kind=record.get("estimate_kind"),
                    theta=record["state"][0] if record["state"] is not None else None,
                    A=record["state"][1] if record["state"] is not None else None, reason=record["reason"]))
            record['diagnostics'] = frame_diagnostics(model, meta, cap.p[i], cap.point_valid[i],
                inverse, empirical_range, covariance.get(i))
            # Clean nonfinite invalid-geometry diagnostics while preserving each row.
            f.write(json.dumps(clean_json(record), default=json_default, allow_nan=False)+"\n")
    summary = dict(original_rows=len(cap.frame), sampled_rows=len(selected),
        estimated_rows=estimated, every=every, holdouts=holdouts, source_sha256=cap.sha256,
        model_path=str(model_path), model_schema=meta["schema"], physiological_validation=False)
    if holdouts:
        from .crosscheck import join_records, _hash
        from .scorecard import build
        joined = join_records("application", "application", model.name, population, source_frames, held_records)
        enrich_subset_records(joined, {cap.name: cap}, meta,
            np.asarray(states) if training_path.exists() else None, _hash(model_path))
        write_json(output/"crosscheck_frames.json", joined)
        summary["crosscheck_scorecard"] = build(joined, purpose="application")
        summary["application_exposure_policy"] = "whole sampled capture; fixation labels unavailable"
    write_json(output/"summary.json", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("detections", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--every", type=int, default=1)
    parser.add_argument("--holdouts", action="store_true")
    args = parser.parse_args()
    apply(args.model, args.detections, args.output, args.every, args.holdouts)
