"""Phase 8.3: frozen-response retained-x versus retained-x/y P4 holdouts."""
from __future__ import annotations
import argparse
import gzip
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from .audit_followup import transitions
from .crosscheck import _read_population, _summarize, join_records
from .data import load_reviewed
from .diagnostics import enrich_subset_records
from .geometry import context
from .invert import predict_holdout, score_holdout
from .model import PositionModel
from .noise import reference_covariance
from .schema import clean_json, load_model, source_hashes, write_json
from .sensitivity import compact, digest, read_holdouts, read_json

RESPONSES = {"baseline27": ("baseline", "conditional27"),
             "strong_anchor37": ("anchor_strong", "conditional37")}


def task(root, source, output, fold, response):
    variant, name = RESPONSES[response]
    frozen = Path(source)/"variants"/variant/fold/name
    dest = Path(output)/response/fold
    dest.mkdir(parents=True, exist_ok=True)
    if (dest/"completion.json").exists():
        return
    started = time.monotonic()
    captures, groups, interval_hash = load_reviewed(root)
    model, meta = load_model(frozen/"model.json")
    provenance = meta["provenance"]
    if interval_hash != provenance["interval_sha256"] or any(
            captures[n].sha256 != h for n, h in provenance["capture_sha256"].items()):
        raise ValueError("Raw detections/labels differ from frozen calibration")
    pilot = PositionModel(27, meta["pilot_coefficients"])
    population = _read_population(Path(source)/"populations"/(fold+".csv.gz"))
    xy = read_holdouts(frozen/"holdouts.jsonl")
    frames = read_json(frozen/"frames.json")
    trained = np.asarray(read_json(frozen/"training_states.json")["states"])
    holds = []
    with gzip.open(dest/"inverse_candidates_x.jsonl.gz", "wt") as archive:
        for row in population:
            if row["selected_for_evaluation"] != "True":
                continue
            gi, i = int(row["fixation"]), int(row["row"])
            cap, group = captures[row["capture"]], groups[gi]
            ctx = context(cap.p[i])
            if not ctx.valid:
                continue
            cov = reference_covariance(cap.p[i:i+1], pilot, np.asarray(meta["reference_state"]),
                                       np.asarray(meta["coordinate_covariance"]))[0]
            for j in range(3):
                kept = np.array([2*k for k in range(3) if k != j])
                if not cap.point_valid[i][np.arange(3) != j].all():
                    prediction = dict(available=False, held_point=j, reason="insufficient_retained_P4", branches=[])
                else:
                    # Input contains exactly the two retained x measurements.
                    prediction = predict_holdout(model, ctx, j, cap.v[i].ravel()[kept], cov, channels="x")
                archive.write(json.dumps(clean_json(dict(fixation=gi, row=i, held_point=j,
                    candidates=prediction.pop("candidates", []))), allow_nan=False)+"\n")
                score = score_holdout(prediction, cap.q[i, j] if cap.point_valid[i, j] else np.full(2, np.nan), ctx.ell)
                score.update(fixation=gi, capture=cap.name, row=i, nominal_theta=group["target_theta_deg"],
                             demand=group["demand_diopters_label"])
                holds.append(score)
    with (dest/"holdouts_x.jsonl").open("w") as stream:
        for row in holds:
            stream.write(json.dumps(clean_json(row), allow_nan=False)+"\n")
    for method, records in (("x", holds), ("xy", xy)):
        joined = join_records(fold.split("_")[0], fold, response, population, frames, records)
        enrich_subset_records(joined, captures, meta, trained, digest(frozen/"model.json"), method)
        for frame in joined:
            for slot in frame["slots"]:
                slot["retained_image_channels"] = method
        write_json(dest/(method+"_frames.json"), joined)
    write_json(dest/"completion.json", dict(complete=True, seconds=time.monotonic()-started,
        response_model_sha256=digest(frozen/"model.json"), xy_holdouts_sha256=digest(frozen/"holdouts.jsonl"),
        xy_policy="reuse certified frozen 49-start scalar inverse; identical default policy",
        x_policy="same 49 starts and independent polish; exact two-coordinate marginal covariance",
        scheduled_frames=sum(r["selected_for_evaluation"] == "True" for r in population)))
    print(f"Phase8.3 {response}/{fold}: {time.monotonic()-started:.1f}s", flush=True)


def report(output, folds):
    summaries = {}
    with gzip.open(output/"paired_transition_membership.jsonl.gz", "wt") as membership:
        for response in RESPONSES:
            summaries[response] = {}
            for family in ("gaze", "capture"):
                records = {m: [] for m in ("x", "xy")}
                for fold in folds:
                    if fold.startswith(family+"_"):
                        for method in records:
                            records[method].extend(read_json(output/response/fold/(method+"_frames.json")))
                pair = transitions(records["x"], records["xy"])
                membership.write(json.dumps(clean_json(dict(response=response, family=family,
                    reference="x", candidate="xy", points=pair.pop("points"), frames=pair.pop("frames"))), allow_nan=False)+"\n")
                summaries[response][family] = dict(methods={m: compact(_summarize(r, len(r))) for m, r in records.items()},
                    paired_squared_changes=pair)
    write_json(output/"summary.json", dict(schema="phase83_retained_channels_v1", responses=summaries,
        definitions="E: vector RMS; G: raw physical-unit subset disagreement; paired changes xy minus x squared error",
        interpretation="frozen-response development ablation, not independent physical-state validation",
        branch_policy="same heuristic absolute residual cost gap 2; report unavailable/rank/ambiguity/bounds separately"))


def run(root, source, output, workers=12, folds=None):
    root, source, output = (Path(p).resolve() for p in (root, source, output))
    folds = read_json(source/"config.json")["folds"] if folds is None else folds
    hashes = {str(p.relative_to(source)): digest(p) for p in source.rglob("*") if p.is_file()}
    config = dict(schema="phase83_retained_channels_v1", source=str(source), frozen_source_hashes=hashes,
        implementation_hashes=source_hashes(root), folds=folds, responses=RESPONSES,
        workers=workers, threads={k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")},
        policy="predeclared frozen responses, shared excluded P4, all P1 context; no held-out driven selection")
    output.mkdir(parents=True, exist_ok=True)
    if (output/"config.json").exists() and read_json(output/"config.json") != clean_json(config):
        raise ValueError("Resume must match frozen inputs and implementation")
    write_json(output/"config.json", config)
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(task, root, source, output, fold, response) for fold in folds for response in RESPONSES]
        for future in as_completed(futures):
            future.result()
    report(output, folds)
    unchanged = all(digest(source/p) == h for p, h in hashes.items())
    if not unchanged:
        raise RuntimeError("Frozen artifacts changed during Phase8.3")
    write_json(output/"completion.json", dict(complete=True, tasks=len(folds)*len(RESPONSES),
        seconds=time.monotonic()-started, frozen_source_unchanged=unchanged))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--source", type=Path, default=Path("experiments/full_position/joint_sensitivity_v1"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--folds", nargs="+")
    args = parser.parse_args()
    run(args.root, args.source, args.output, args.workers, args.folds)
