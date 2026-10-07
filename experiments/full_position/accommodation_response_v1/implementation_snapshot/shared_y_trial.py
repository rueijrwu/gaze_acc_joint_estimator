"""One training-scaled shared-y covariance trial; no grid or promotion."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
import gzip
import json
import shutil
import time
import numpy as np
from .audit83 import BASE, source_folder, supported
from .crosscheck import _read_population, join_records
from .geometry import context
from .information import predict_raw_holdout
from .invert import score_holdout
from .latest_audit import COHORTS, paired_summary
from .model import PositionModel
from .population import manifest
from .schema import clean_json, load_model, source_hashes, write_json
from .scorecard import build
from .sensitivity import checked_training, digest, read_json


def training_tau(residual_px, groups):
    """Equal-training-fixation RMS of three-point common-y residual in pixels.

    This is a robust-weighting proxy including bias and variation, not an
    independently estimated localization/discrepancy variance. No outer data.
    """
    residual, groups = np.asarray(residual_px, float), np.asarray(groups)
    if groups.ndim != 1 or residual.ndim != 3 or residual.shape[1:] != (3, 2) or len(residual) != len(groups) or not len(groups):
        raise ValueError("Training residuals require aligned nonempty N x 3 x 2 records")
    if not np.isfinite(residual).all():
        raise ValueError("Training residuals must be finite")
    if groups.dtype.kind in "fc" and not np.isfinite(groups).all():
        raise ValueError("Training group labels must be finite")
    common = residual[..., 1].mean(1)
    return float(np.sqrt(np.mean([np.square(common[groups == gi]).mean() for gi in np.unique(groups)])))


def task(job):
    root, output, fold = job
    root, output = Path(root), Path(output)
    dest = output/fold
    dest.mkdir(parents=True, exist_ok=True)
    captures, groups, data, _, _ = checked_training(root, root/BASE/"audit_polished_v1", fold)
    frozen = source_folder(root, "baseline27", fold)
    model, meta = load_model(frozen/"model.json")
    trained = read_json(frozen/"training_states.json")
    if not np.array_equal(trained["rows"], data["rows"]) or not np.array_equal(trained["groups"], data["groups"]):
        raise ValueError("Training population mismatch")
    ctx_train = context(data["p"])
    residual = (data["v"]-model.predict(np.asarray(trained["states"]), data["r"])).reshape(-1, 3, 2)*ctx_train.ell[:, None, None]
    tau = training_tau(residual, data["original_group"])
    # Write the training declaration before opening the evaluation population.
    write_json(dest/"training_tau.json", dict(tau_px=tau, frozen_model_sha256=digest(frozen/"model.json"),
        training_group_ids=sorted(map(int, set(data["original_group"]))),
        rule="equal-fixation RMS of common-all-three-y pixel training residual", use_outer_errors=False,
        interpretation="extra shared-mode robust weighting proxy, includes bias/variation; not validated noise variance"))
    pilot = PositionModel(27, meta["pilot_coefficients"])
    reference, sigma = np.asarray(meta["reference_state"]), np.asarray(meta["coordinate_covariance"])
    population = _read_population(frozen.parent/"population.csv.gz")
    holds = []
    with gzip.open(dest/"inverse_candidates.jsonl.gz", "wt") as archive:
        for row in population:
            if row["selected_for_evaluation"] != "True":
                continue
            gi, i = int(row["fixation"]), int(row["row"])
            cap = captures[row["capture"]]
            ctx = context(cap.p[i])
            if not ctx.valid:
                continue
            for j in range(3):
                prediction = predict_raw_holdout(model, cap.p[i], cap.q[i], cap.point_valid[i], j,
                    pilot, reference, sigma, "xy", discrepancy_tau_px=tau)
                archive.write(json.dumps(clean_json(dict(fixation=gi, row=i, held_point=j,
                    candidates=prediction.pop("candidates", []))), allow_nan=False)+"\n")
                score = score_holdout(prediction, cap.q[i, j] if cap.point_valid[i, j] else np.full(2, np.nan), ctx.ell)
                score.update(capture=cap.name, fixation=gi, row=i, retained_image_channels="xy_shared_y_training_rms")
                holds.append(score)
    with (dest/"holdouts.jsonl").open("w") as stream:
        for h in holds:
            stream.write(json.dumps(clean_json(h), allow_nan=False)+"\n")
    joined = join_records(fold.split("_")[0], fold, "baseline27", population, read_json(frozen/"frames.json"), holds)
    supported(root, joined, frozen, "xy_shared_y_training_rms")
    for frame in joined:
        for slot in frame["slots"]:
            slot["shared_y_discrepancy_tau_px"] = tau
            slot["noise_policy_modifier"] = "R0+(tau_px/ell)^2 aaT; training-only tau; unchanged pilot/localization"
    schedule = manifest([dict(fold=fold, capture=r["capture"], fixation=int(r["fixation"]), row=int(r["row"]),
        source_frame_index=int(r["frame"])) for r in population if r["selected_for_evaluation"] == "True"])
    write_json(dest/"frames.json", joined)
    write_json(dest/"scorecard.json", build(joined, population=schedule))
    return fold


def run(root, output, source, workers=12):
    root, output, source = Path(root).resolve(), Path(output).resolve(), Path(source).resolve()
    output.mkdir(parents=True, exist_ok=False)
    folds = read_json(source/"config.json")["folds"]
    inputs = [p for name in ("joint_sensitivity_v1", "audit_polished_v1", "phase83_audit_followup_v1")
              for p in (root/BASE/name).rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    inputs += [source/name for name in ("config.json", "completion.json", "training_diagnosis.json", "direct_comparison.json")]
    inputs += [p for p in (source/"training_diagnosis").rglob("*") if p.is_file()]
    hashes = {str(p.relative_to(root)): digest(p) for p in inputs}
    implementation = source_hashes(root)
    write_json(output/"config.json", dict(schema="training_shared_y_covariance_trial_v1", workers=workers,
        process_start_method="fork", folds=folds, source_hashes=hashes, implementation_hashes=implementation,
        tau_policy="one training common-y RMS per fold; no grid, no outer-score tuning",
        promotion="no_promotion_decision", scope="exploratory frozen baseline27 response; historical development folds"))
    for rel in implementation:
        dest = output/("implementation_snapshot" if rel.startswith("full_position/") else "design_snapshot")/Path(rel).name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root/rel, dest)
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("fork")) as pool:
        list(pool.map(task, [(root, output, fold) for fold in folds]))
    reference = {}
    with gzip.open(root/BASE/"phase83_audit_followup_v1/scorecard_frames.jsonl.gz", "rt") as stream:
        for line in stream:
            item = json.loads(line)
            if item["candidate"].startswith("phase83/baseline27/") and item["candidate"].endswith("/xy"):
                reference[item["candidate"].split("/")[2]] = reference.get(item["candidate"].split("/")[2], [])+[item["frame"]]
    summaries = {}
    with gzip.open(output/"paired_membership.jsonl.gz", "wt") as archive:
        for family in ("gaze", "capture"):
            ref, candidate, scheduled = [], [], []
            for fold in folds:
                if fold.startswith(family+"_"):
                    ref.extend(reference[fold])
                    candidate.extend(read_json(output/fold/"frames.json"))
                    for row in _read_population(source_folder(root, "baseline27", fold).parent/"population.csv.gz"):
                        if row["selected_for_evaluation"] == "True":
                            scheduled.append(dict(fold=fold, capture=row["capture"], fixation=int(row["fixation"]),
                                row=int(row["row"]), source_frame_index=int(row["frame"])))
            population = manifest(scheduled)
            cohorts = {}
            for cohort in COHORTS:
                report, members = paired_summary(ref, candidate, population, cohort)
                report["direction"] = "training_scaled_shared_y_minus_xy"
                cohorts[cohort] = report
                archive.write(json.dumps(clean_json(dict(comparison=family, **members)), allow_nan=False)+"\n")
            summaries[family] = dict(cohorts=cohorts, reference=build(ref, population=population),
                candidate=build(candidate, population=population))
    write_json(output/"summary.json", dict(families=summaries, promotion="no_promotion_decision",
        limitation="tau is a training misfit scale proxy; does not distinguish systematic capture bias from unpredictable variation"))
    if any(digest(root/rel) != h for rel, h in hashes.items()):
        raise RuntimeError("Trial sources changed")
    write_json(output/"completion.json", dict(complete=True, tasks=len(folds), seconds=time.monotonic()-started,
        source_files_unchanged=len(hashes)))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path.cwd())
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=12)
    a = p.parse_args()
    run(a.root, a.output, a.source, a.workers)
