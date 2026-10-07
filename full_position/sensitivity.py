"""Grouped joint latent-state/coefficient refits for the Phase 8.2 audit.

The variants are declared before evaluation. Saved training states supply starts,
never penalties or evaluation-frame initializers. Failed calibrations retain their
checkpoints and scheduled denominators, and are never applied as fitted models.
"""
from __future__ import annotations

import argparse
import copy
import csv
import gzip
import hashlib
import json
import os
import platform
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

import numpy as np
import scipy

from .calibrate import fit
from .crosscheck import (COORDINATE_MODELS, IDENTITY, _frame_csv_row,
                        _frame_scalar_bundle, _paired, _point_metric_bundle,
                        _read_population, _summarize, join_records, stats)
from .data import load_reviewed, training_data
from .diagnostics import COVARIANCE_DEFINITION, frame_diagnostics
from .geometry import context
from .model import PositionModel
from .noise import reference_covariance
from .schema import artifact, clean_json, load_model, source_hashes, write_json
from .validate import evaluate


SCHEMA = "exp5_joint_sensitivity_v1"
VARIANTS = {
    "baseline": dict(anchor_factor=1., prior_factor=1., covariance_factor=1., reference="median"),
    "anchor_weak": dict(anchor_factor=2., prior_factor=1., covariance_factor=1., reference="median"),
    "anchor_strong": dict(anchor_factor=.5, prior_factor=1., covariance_factor=1., reference="median"),
    "prior_weak": dict(anchor_factor=1., prior_factor=.1, covariance_factor=1., reference="median"),
    "prior_strong": dict(anchor_factor=1., prior_factor=10., covariance_factor=1., reference="median"),
    "covariance_4x": dict(anchor_factor=1., prior_factor=1., covariance_factor=4., reference="median"),
    "reference_q25": dict(anchor_factor=1., prior_factor=1., covariance_factor=1., reference="q25"),
    "reference_q75": dict(anchor_factor=1., prior_factor=1., covariance_factor=1., reference="q75"),
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@lru_cache(maxsize=1)
def reviewed(root):
    return load_reviewed(root)


def read_json(path):
    return json.loads(Path(path).read_text())


def read_holdouts(path):
    path = Path(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def compact(value):
    """Metric membership is retained once in cohort masks and frame/point files."""
    if isinstance(value, dict):
        return {k: compact(v) for k, v in value.items() if k != "membership"}
    if isinstance(value, list):
        return [compact(v) for v in value]
    return value


def settings(meta, anchors, variant):
    rule = VARIANTS[variant]
    base_scales = np.asarray(meta["calibration"]["anchor_scales"], float)
    ref = np.asarray(meta["reference_state"], float)
    if rule["reference"] != "median":
        ref = np.quantile(anchors, .25 if rule["reference"] == "q25" else .75, axis=0)
    return dict(anchor_scales=(base_scales*rule["anchor_factor"]).tolist(),
                anchor_precision_factor=1/rule["anchor_factor"]**2,
                prior_strength=meta["calibration"]["prior_strength"]*rule["prior_factor"],
                covariance_factor=rule["covariance_factor"], reference_state=ref.tolist(),
                reference_policy=rule["reference"], temporal_strength=0.)


def checked_training(root, source, fold):
    captures, groups, interval_hash = reviewed(str(root))
    models = {}
    for name in COORDINATE_MODELS:
        model, meta = load_model(source/fold/name/"model.json")
        p = meta["provenance"]
        if p["interval_sha256"] != interval_hash:
            raise ValueError("Fixation labels differ from frozen fold")
        if any(captures[n].sha256 != h for n, h in p["capture_sha256"].items()):
            raise ValueError("Detection source differs from frozen fold")
        data, anchors = training_data(captures, groups, p["training_group_ids"], 48)
        saved = read_json(source/fold/name/"training_states.json")
        checks = ((data["rows"], p["sampled_training_rows"]),
                  (data["original_group"], p["sampled_training_group"]),
                  (data["rows"], saved["rows"]), (data["groups"], saved["groups"]),
                  (anchors, saved["nominal_anchors"]))
        if not all(np.array_equal(a, b) for a, b in checks):
            raise ValueError("Training population/order differs from frozen trajectory")
        x = np.asarray(saved["states"], float)
        if x.shape != (len(data["p"]), 2) or not np.isfinite(x).all():
            raise ValueError("Invalid frozen training trajectory")
        train, test = set(p["training_group_ids"]), set(p["evaluation_group_ids"])
        if train & test or train | test != set(range(len(groups))):
            raise ValueError("Invalid outer grouped split")
        models[name] = (model, meta, x)
    a, b = (models[n][1] for n in COORDINATE_MODELS)
    for field in ("pilot_coefficients", "coordinate_covariance", "reference_state"):
        if not np.array_equal(a[field], b[field]):
            raise ValueError("Paired capacities must share the frozen pilot/covariance")
    for field in ("training_group_ids", "evaluation_group_ids", "sampled_training_rows", "sampled_training_group"):
        if a["provenance"][field] != b["provenance"][field]:
            raise ValueError("Paired model populations differ")
    return captures, groups, data, anchors, models


def training_diagnostics(model, states, data, anchors, frozen_model, frozen_states):
    residual = data["v"]-model.predict(states, data["r"])
    means = []
    for local in range(len(anchors)):
        ix = data["groups"] == local
        mean = states[ix].mean(0)
        means.append(dict(original_group=int(data["original_group"][ix][0]), count=int(ix.sum()),
                          nominal_anchor=anchors[local].tolist(), mean_state=mean.tolist(),
                          mean_minus_anchor=(mean-anchors[local]).tolist(),
                          signed_residual_mean_normalized=residual[ix].mean(0).tolist()))
    delta = states-frozen_states
    pred_delta = model.predict(frozen_states, data["r"])-frozen_model.predict(frozen_states, data["r"])
    scale = context(data["p"]).ell[:, None, None]
    pixel = (pred_delta.reshape(-1, 3, 2)*scale).reshape(-1, 2)
    pixel_squared = np.sum((residual.reshape(-1, 3, 2)*scale)**2, axis=-1)
    anchor_difference = np.asarray([m["mean_minus_anchor"] for m in means])
    return dict(group_means=means, training_state_shift_theta_deg=stats(delta[:, 0]),
                training_state_shift_A_D=stats(delta[:, 1]),
                training_fixation_mean_gaze_anchor_discrepancy_deg=stats(anchor_difference[:, 0]),
                training_fixation_mean_accommodation_demand_discrepancy_D=stats(anchor_difference[:, 1]),
                training_equal_fixation_point_vector_rms_px=float(np.sqrt(np.mean([
                    pixel_squared[data["groups"] == k].mean() for k in range(len(anchors))]))),
                response_shift_at_fixed_frozen_training_states_px=stats(np.linalg.norm(pixel, axis=1)),
                response_shift_at_fixed_states_normalized=stats(np.linalg.norm(pred_delta.reshape(-1, 2), axis=1)),
                training_bound_state_count=int(np.sum(np.any((states <= [-20+1e-5, 1e-5]) |
                                                            (states >= [20-1e-5, 6-1e-5]), axis=1))),
                raw_training_residual_rms_normalized=float(np.sqrt(np.mean(residual**2))),
                state_parametrization="theta degrees; accommodation diopters; optimizer encoding [theta/10,A/4]",
                uncertainty_definition=COVARIANCE_DEFINITION)


def task(root, source, output, fold, variant, starts, max_nfev, seed):
    root, source, output = map(Path, (root, source, output))
    dest_fold = output/"variants"/variant/fold
    dest_fold.mkdir(parents=True, exist_ok=True)
    population = dest_fold/"population.csv.gz"
    if not population.exists():
        population.symlink_to(Path("../../../populations")/(fold+".csv.gz"))
    result = dict(fold=fold, variant=variant, models={})
    for name in COORDINATE_MODELS:
        dest = dest_fold/name
        dest.mkdir(parents=True, exist_ok=True)
        done = dest/"completion.json"
        if done.exists():
            result["models"][name] = read_json(done)
            continue
        begun = time.monotonic()
        fit_converged = False
        diag = None
        try:
            captures, groups, data, anchors, frozen = checked_training(root, source, fold)
            old_model, old_meta, old_states = frozen[name]
            change = settings(old_meta, anchors, variant)
            pilot = PositionModel(27, old_meta["pilot_coefficients"])
            sigma = np.asarray(old_meta["coordinate_covariance"])*change["covariance_factor"]
            reference = np.asarray(change["reference_state"])
            cov = reference_covariance(data["p"], pilot, reference, sigma)
            extra = [old_states]
            if name == "conditional37":
                extra.append(frozen["conditional27"][2])
            with gzip.open(dest/"calibration_candidates.jsonl.gz", "wt") as archive:
                def checkpoint(record, beta, states):
                    archive.write(json.dumps(clean_json(dict(diagnostics=record, coefficients=beta,
                                                             states=states)), separators=(",", ":"))+"\n")
                    archive.flush()
                model, states, diag = fit(PositionModel(old_model.size), data["v"], data["r"], cov,
                    data["groups"], anchors, change["prior_strength"], starts, max_nfev, seed,
                    additional_initial_states=extra, anchor_scales=change["anchor_scales"], checkpoint=checkpoint)
            model.training_range = [anchors.min(0), anchors.max(0)]
            fit_converged = bool(diag["converged"])
            provenance = copy.deepcopy(old_meta["provenance"])
            provenance.update(source_sha256=source_hashes(root), phase="8.2_joint_training_sensitivity",
                frozen_model_sha256=digest(source/fold/name/"model.json"), variant=variant,
                initialization_policy="same seeded nominal starts plus frozen same-fold training trajectories only",
                sensitivity_settings=change, seed=seed,
                evaluation_usage="development sensitivity; not model selection or untouched validation")
            meta = artifact(model, pilot, reference, sigma, provenance, diag)
            write_json(dest/("model.json" if diag["converged"] else "failed_checkpoint.json"), meta)
            write_json(dest/"training_states.json", dict(states=states, nominal_anchors=anchors,
                                                       groups=data["groups"], rows=data["rows"]))
            diagnostics = training_diagnostics(model, states, data, anchors, old_model, old_states)
            write_json(dest/"training_diagnostics.json", diagnostics)
            if diag["converged"]:
                evaluate(model, captures, groups, provenance["evaluation_group_ids"], pilot,
                         reference, sigma, dest, count=8, progress=lambda *a, **k: None)
                records = []
                empirical = [states.min(0), states.max(0)]
                for row in read_json(dest/"frames.json"):
                    cap = captures[row["capture"]]
                    inverse = dict(available=row["estimated"])
                    if row["estimated"]:
                        inverse.update(state=[row["theta"], row["A"]], rank=row["rank"],
                                       at_bound=row["at_bound"], ambiguous=row["ambiguous"])
                    record = frame_diagnostics(model, meta, cap.p[row["row"]],
                                               cap.point_valid[row["row"]], inverse, empirical)
                    record.update(row=row["row"], capture=row["capture"], fixation=row["fixation"])
                    records.append(record)
                write_json(dest/"frame_uncertainty.json", dict(records=records,
                    empirical_training_state_range=empirical, covariance_definition=COVARIANCE_DEFINITION))
            completion = dict(complete=True, calibration_converged=diag["converged"], calibration=diag,
                              settings=change, seconds=time.monotonic()-begun,
                              evaluation_complete=bool(diag["converged"]), training_diagnostics=diagnostics)
        except Exception as error:
            completion = dict(complete=True, calibration_converged=fit_converged,
                              calibration=diag, evaluation_complete=False, failed_exception=True,
                              exception_type=type(error).__name__, message=str(error),
                              seconds=time.monotonic()-begun)
            (dest/"failure.log").write_text(traceback.format_exc())
        write_json(done, completion)
        result["models"][name] = completion
        print(f"{variant}/{fold}/{name}: calibration_converged={completion['calibration_converged']}", flush=True)
    return result


def joined(root, source, fold, model, variant=None):
    """Use the same score eligibility and subset metrics as Phase 8.1."""
    folder = source/fold/model
    population = _read_population(source/fold/"population.csv.gz")
    frames = read_json(folder/"frames.json") if (folder/"frames.json").exists() else []
    holds = read_holdouts(folder/"holdouts.jsonl")
    records = join_records(fold.split("_")[0], fold, model, population, frames, holds)
    meta_path = folder/"model.json"
    meta = read_json(meta_path) if meta_path.exists() else None
    completion_path = folder/"completion.json"
    evaluation_failed = (completion_path.exists() and
                         not read_json(completion_path).get("evaluation_complete", True))
    if evaluation_failed:
        frames, holds = [], []
        records = join_records(fold.split("_")[0], fold, model, population, frames, holds)
    trained_path = folder/"training_states.json"
    empirical = None
    if trained_path.exists():
        states = np.asarray(read_json(trained_path)["states"], float)
        empirical = [states.min(0), states.max(0)]
    uncertain = read_json(folder/"frame_uncertainty.json") if (folder/"frame_uncertainty.json").exists() else {}
    by_id = {(r["capture"], r["fixation"], r["row"]): r for r in uncertain.get("records", [])}
    _, groups, _ = reviewed(str(root))
    for frame in records:
        group = groups[frame["fixation"]]
        frame.update(variant=variant, nominal_theta=group["target_theta_deg"], demand=group["demand_diopters_label"])
        unavailable = "evaluation_failed" if meta and evaluation_failed else "calibration_not_certified"
        if not meta or evaluation_failed:
            frame["reason"] = unavailable
        common = by_id.get((frame["capture"], frame["fixation"], frame["row"]), {})
        for point in frame["slots"]:
            point.update(variant=variant, nominal_theta=frame["nominal_theta"], demand=frame["demand"])
            if (not meta or evaluation_failed) and point["input_valid"]:
                point["failure_reason"] = unavailable
            support = point["support"]
            for key in ("context_outside_training_extrema", "p1_parity_seen_in_training"):
                support[key] = common.get(key)
            if meta and point["state"] is not None:
                x = np.asarray(point["state"])
                anchor = meta["provenance"]["anchor_range"]
                support.update(theta_anchor=bool(anchor[0][0] <= x[0] <= anchor[1][0]),
                               A_anchor=bool(anchor[0][1] <= x[1] <= anchor[1][1]),
                               theta_empirical=bool(empirical[0][0] <= x[0] <= empirical[1][0]) if empirical is not None else None,
                               A_empirical=bool(empirical[0][1] <= x[1] <= empirical[1][1]) if empirical is not None else None)
    return records


def identity(record, point=False):
    return tuple(record[k] for k in IDENTITY)+(tuple([record["held_point"]]) if point else ())


def paired(reference, candidate):
    old = {identity(f): f for f in reference}
    new = {identity(f): f for f in candidate}
    if old.keys() != new.keys():
        raise ValueError("Sensitivity comparison must have identical scheduled frame IDs")
    old_points = {identity(p, True): p for f in reference for p in f["slots"]}
    new_points = {identity(p, True): p for f in candidate for p in f["slots"]}
    point_ids = sorted(k for k in old_points if old_points[k]["scored"] and new_points[k]["scored"])
    frame_ids = sorted(k for k in old if old[k]["complete_triple"] and new[k]["complete_triple"])
    interior_points = [k for k in point_ids if old_points[k]["interior"] and new_points[k]["interior"]]
    interior_frames = [k for k in frame_ids if old[k]["complete_interior"] and new[k]["complete_interior"]]
    result = dict(scheduled_frame_count=len(old), shared_scored_point_ids=point_ids,
                  shared_complete_frame_ids=frame_ids, shared_interior_point_ids=interior_points,
                  shared_complete_interior_frame_ids=interior_frames)
    for label, ids in (("all_testable", frame_ids), ("interior", interior_frames)):
        result[label] = {side: {field: compact(_frame_scalar_bundle([table[k] for k in ids], field))
            for field in ("E_px", "G_theta_deg", "G_A_D", "worst_point_px", "E_normalized")}
            for side, table in (("reference", old), ("candidate", new))}
    for label, ids in (("points", point_ids), ("interior_points", interior_points)):
        result[label] = {side: compact(_point_metric_bundle([table[k] for k in ids]))
                        for side, table in (("reference", old_points), ("candidate", new_points))}
    delta = np.asarray([np.asarray(new_points[k]["state"])-old_points[k]["state"] for k in point_ids])
    result["subset_state_shift_theta_deg"] = stats(delta[:, 0]) if len(delta) else stats([])
    result["subset_state_shift_A_D"] = stats(delta[:, 1]) if len(delta) else stats([])
    return result


def coefficient_refit_dispersion(root, source, output, folds, variants):
    """Descriptive grouped response variation, never a total uncertainty covariance."""
    _, _, data, _, _ = checked_training(root, source, folds[0])
    probe_context = context(data["p"][0])
    records = {}
    for family in ("gaze", "capture"):
        members = [f for f in folds if f.startswith(family+"_")]
        if not members:
            continue
        bounds = [read_json(source/f/"conditional27"/"model.json")["provenance"]["anchor_range"] for f in members]
        low = np.max(np.asarray(bounds)[:, 0], axis=0)
        high = np.min(np.asarray(bounds)[:, 1], axis=0)
        states = np.array([[t, a] for t in np.linspace(low[0], high[0], 3)
                           for a in np.linspace(low[1], high[1], 3)])
        for variant in variants:
            for name in COORDINATE_MODELS:
                paths = [output/"variants"/variant/f/name/"model.json" for f in members]
                paths = [p for p in paths if p.exists()]
                prediction, labels, support = [], [], []
                for path in paths:
                    model, meta = load_model(path)
                    prediction.append(model.predict(states, probe_context.r).reshape(-1, 3, 2))
                    labels.append(path.parent.parent.name)
                    box = meta["provenance"]["conditional_context_support"]
                    support.append(not np.any((probe_context.r < box["r_min"]) | (probe_context.r > box["r_max"])))
                spread = None
                if len(prediction) >= 2:
                    pred = np.asarray(prediction)
                    spread = stats(np.linalg.norm((pred-pred.mean(0))*probe_context.ell, axis=-1).ravel())
                records[f"{family}/{variant}/{name}"] = dict(fold_ids=labels, scheduled_fold_count=len(members),
                    accepted_fold_count=len(paths), states_physical=states.tolist(), predictions_normalized=prediction,
                    common_context_inside_fold_component_extrema=support,
                    grouped_response_dispersion_px=spread)
    return dict(records=records, probe_P1_points=probe_context.p, probe_normalized_P1=probe_context.r,
                probe_scale_px=probe_context.ell,
                definition="descriptive response dispersion across accepted grouped refits at fixed physical states/context; correlated folds; not a predictive probability or total uncertainty covariance",
                probe_policy="first fold's first frozen training P1; physical states inside intersection of family nominal anchor boxes; reporting only")


def report(root, source, output, folds, variants):
    root, source, output = map(Path, (root, source, output))
    frozen = {family: {name: [] for name in COORDINATE_MODELS} for family in ("gaze", "capture")}
    study = {v: {family: {name: [] for name in COORDINATE_MODELS}
                 for family in frozen} for v in variants}
    completions = {}
    frame_rows, point_rows = [], []
    for fold in folds:
        family = fold.split("_")[0]
        for name in COORDINATE_MODELS:
            frozen[family][name].extend(joined(root, source, fold, name, "frozen_original"))
        for variant in variants:
            for name in COORDINATE_MODELS:
                base = output/"variants"/variant
                path = base/fold/name/"completion.json"
                completions[f"{variant}/{fold}/{name}"] = read_json(path) if path.exists() else dict(complete=False)
                frames = joined(root, base, fold, name, variant)
                study[variant][family][name].extend(frames)
                for frame in frames:
                    frame_rows.append(dict(variant=variant, **_frame_csv_row(frame)))
                    for slot in frame["slots"]:
                        point = {k: slot.get(k) for k in (*IDENTITY, "source_frame_index", "held_point", "input_valid",
                            "held_point_measurement_valid", "available", "certified", "identifiable", "unambiguous",
                            "scored", "failure_reason", "state", "error_px", "error_normalized", "bound", "interior",
                            "rank", "singular_values_physical", "condition_number", "support", "retained_subset_cost")}
                        point.update(variant=variant, holdout_artifact=str((base/fold/name/"holdouts.jsonl").relative_to(output)))
                        point_rows.append(point)
    for filename, rows in (("crosscheck_frames.csv.gz", frame_rows), ("crosscheck_points.csv.gz", point_rows)):
        with gzip.open(output/filename, "wt", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            for row in rows:
                writer.writerow({k: json.dumps(clean_json(v), separators=(",", ":")) if isinstance(v, (list, dict)) else v
                                 for k, v in row.items()})
    families = {}
    comparisons = {}
    for family in frozen:
        if not any(frozen[family].values()):
            continue
        families[family] = {}
        for variant in variants:
            models = study[variant][family]
            families[family][variant] = {"models": {name: compact(_summarize(frames, len(frames)))
                                         for name, frames in models.items()}, "paired_27_37": compact(_paired(models))}
            for name in COORDINATE_MODELS:
                comparison = {"versus_frozen": paired(frozen[family][name], models[name])}
                if "baseline" in study:
                    comparison["versus_joint_baseline"] = paired(study["baseline"][family][name], models[name])
                comparisons[f"{family}/{variant}/{name}"] = comparison
    write_json(output/"summary.json", dict(schema=SCHEMA, split_families=families,
        calibrations=completions, settings=VARIANTS, selection_policy="predeclared one-factor sensitivity; no holdout-based model/threshold selection",
        metric_conventions="Phase 8.1 E/G physical-unit metrics; pooled and equal-exposure RMS; exact shared IDs",
        uncertainty_boundary=COVARIANCE_DEFINITION))
    with gzip.open(output/"paired_comparisons.json.gz", "wt") as stream:
        json.dump(clean_json(comparisons), stream, separators=(",", ":"), allow_nan=False)
    write_json(output/"grouped_refit_dispersion.json", coefficient_refit_dispersion(root, source, output, folds, variants))


def run(root, source, output, workers=6, variants=None, folds=None, starts=2, max_nfev=300, seed=17):
    root, source, output = map(lambda p: Path(p).resolve(), (root, source, output))
    if output == source or output.is_relative_to(source):
        raise ValueError("Sensitivity output must be separate from frozen source results")
    variants = list(VARIANTS) if variants is None else list(variants)
    available = sorted(p.name for p in source.iterdir() if p.is_dir() and (p/"population.csv.gz").exists())
    folds = available if folds is None else list(folds)
    if not variants or not folds or set(variants)-VARIANTS.keys() or set(folds)-set(available):
        raise ValueError("Unknown/empty sensitivity variants or folds")
    if len(set(variants)) != len(variants) or len(set(folds)) != len(folds):
        raise ValueError("Duplicate variants/folds")
    if workers < 1 or starts < 1 or max_nfev < 1:
        raise ValueError("Workers, starts, and budget must be positive")
    hashes = {str(p.relative_to(source)): digest(p) for p in sorted(source.rglob("*")) if p.is_file()}
    config = dict(schema=SCHEMA, source=str(source), source_hashes=hashes,
        implementation_hashes=source_hashes(root), variants=variants, variant_settings={v: VARIANTS[v] for v in variants},
        folds=folds, starts=starts, max_nfev=max_nfev, seed=seed, train_count=48, eval_count=8,
        temporal_strength=0., targets_deg=[-10, -5, 0, 5, 10],
        warm_start_policy="frozen same-fold training states; same-capacity plus conditional27 for conditional37",
        purpose="grouped development joint calibration sensitivity; no selection using held-out measurements",
        coefficient_prior_policy="recompute nominal column scaling/prior center under each covariance policy, following the original calibration definition",
        covariance_control="covariance_4x and anchor_strong joint objectives differ by an overall factor four with this normalized prior; fixed-model no-prior inverse minimizers are invariant to uniform covariance scaling",
        runtime=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
        thread_environment={k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")})
    if output.exists() and any(output.iterdir()) and not (output/"config.json").exists():
        raise FileExistsError("Existing output is not a resumable sensitivity study")
    output.mkdir(parents=True, exist_ok=True)
    config_path = output/"config.json"
    if config_path.exists() and read_json(config_path) != config:
        raise ValueError("Resume settings/source/implementation must match the existing study")
    write_json(config_path, config)
    populations = output/"populations"
    populations.mkdir(exist_ok=True)
    for fold in folds:
        target = populations/(fold+".csv.gz")
        raw = (source/fold/"population.csv.gz").read_bytes()
        if target.exists() and target.read_bytes() != raw:
            raise ValueError("Study population changed")
        target.write_bytes(raw)
    begun = time.monotonic()
    jobs = [(fold, variant) for fold in folds for variant in variants]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(task, root, source, output, fold, variant, starts, max_nfev, seed): (fold, variant)
                   for fold, variant in jobs}
        for future in as_completed(futures):
            future.result()
    report(root, source, output, folds, variants)
    changed = [path for path, value in hashes.items() if digest(source/path) != value]
    if changed:
        raise RuntimeError(f"Frozen source artifacts changed: {changed}")
    write_json(output/"completion.json", dict(complete=True, seconds=time.monotonic()-begun,
        task_count=len(jobs)*len(COORDINATE_MODELS), original_results_unchanged=True,
        note="complete means every task outcome is recorded, not that every calibration converged"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--source", type=Path, default=Path("experiments/full_position/audit_polished_v1"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS))
    parser.add_argument("--folds", nargs="+")
    parser.add_argument("--starts", type=int, default=2)
    parser.add_argument("--max-nfev", type=int, default=300)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    run(args.root, args.source, args.output, args.workers, args.variants,
        args.folds, args.starts, args.max_nfev, args.seed)


if __name__ == "__main__":
    main()
