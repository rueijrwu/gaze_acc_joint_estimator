"""Phase8.3 audit response: shared scorecards, support and retained-y information."""
from __future__ import annotations
import argparse
import gzip
import json
import os
import shutil
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from .crosscheck import _read_population, join_records, stats
from .diagnostics import enrich_subset_records
from .geometry import context
from .information import MASKS, TransformedResponse, predict_raw_holdout, retained_transform
from .invert import score_holdout
from .model import PositionModel
from .noise import reference_covariance, marginal
from .phase83 import RESPONSES
from .profile import profile_inverse
from .schema import clean_json, load_model, protocol_metadata, source_hashes, write_json
from .scorecard import build, comparison
from .sensitivity import checked_training, digest, read_holdouts, read_json, reviewed

BASE = Path("experiments/full_position")
SOURCES = ("audit_polished_v1", "joint_sensitivity_v1", "axis_anchor_sensitivity_v1", "phase83_retained_channels_v1")
Y_MASKS = MASKS[1:]


def source_folder(root, response, fold):
    variant, name = RESPONSES[response]
    return Path(root)/BASE/"joint_sensitivity_v1"/"variants"/variant/fold/name


def supported(root, frames, frozen, mask):
    captures, groups, _ = reviewed(str(root))
    model_path = frozen/"model.json"
    meta_path = model_path if model_path.exists() else frozen/"failed_checkpoint.json"
    meta = read_json(meta_path) if meta_path.exists() else {}
    states = np.asarray(read_json(frozen/"training_states.json")["states"]) if model_path.exists() else None
    enrich_subset_records(frames, captures, meta, states, digest(meta_path) if meta_path.exists() else None, mask)
    for frame in frames:
        group = groups[frame["fixation"]]
        frame.update(nominal_theta=group["target_theta_deg"], demand=group["demand_diopters_label"])
        for slot in frame["slots"]:
            slot.update(nominal_theta=frame["nominal_theta"], demand=frame["demand"], retained_image_channels=mask)
    return frames


def backfill(root, output, folds):
    """All available prior phase predictions; never change their scores or files."""
    cards, family_records = {}, defaultdict(list)
    with gzip.open(output/"scorecard_frames.jsonl.gz", "wt") as stream:
        for phase, source, variants in (
                ("phase81", root/BASE/"audit_polished_v1", [None]),
                ("phase82", root/BASE/"joint_sensitivity_v1", read_json(root/BASE/"joint_sensitivity_v1/config.json")["variants"]),
                ("axis_anchors", root/BASE/"axis_anchor_sensitivity_v1", ["anchor_gaze_strong", "anchor_accommodation_strong"])):
            for variant in variants:
                base = source if variant is None else source/"variants"/variant
                for fold in folds:
                    pop = _read_population(base/fold/"population.csv.gz")
                    for name in ("conditional27", "conditional37"):
                        dest = base/fold/name
                        frames = read_json(dest/"frames.json") if (dest/"frames.json").exists() else []
                        holds = read_holdouts(dest/"holdouts.jsonl")
                        joined = join_records(fold.split("_")[0], fold, name, pop, frames, holds)
                        supported(root, joined, dest, "xy")
                        key = "/".join((phase, variant or "frozen", fold, name))
                        cards[key] = build(joined)
                        family_records[(phase, variant or "frozen", fold.split("_")[0], name)].extend(joined)
                        for frame in joined:
                            stream.write(json.dumps(clean_json(dict(candidate=key, frame=frame)), allow_nan=False)+"\n")
                    if variant is None:
                        # The summary control has no individual-P4 decoder.
                        expected = {(fold, r["capture"], int(r["fixation"])) for r in pop
                                    if r["selected_for_evaluation"] == "True"}
                        cards[f"phase81/frozen/{fold}/two_channel13"] = build([], expected_exposures=expected,
                            individual_decoder=False)
        for response in RESPONSES:
            for fold in folds:
                frozen = source_folder(root, response, fold)
                for mask in ("x", "xy"):
                    path = root/BASE/"phase83_retained_channels_v1"/response/fold/(mask+"_frames.json")
                    joined = supported(root, read_json(path), frozen, mask)
                    key = f"phase83/{response}/{fold}/{mask}"
                    cards[key] = build(joined)
                    family_records[("phase83", response, fold.split("_")[0], mask)].extend(joined)
                    for frame in joined:
                        stream.write(json.dumps(clean_json(dict(candidate=key, frame=frame)), allow_nan=False)+"\n")
    families = {"/".join(k): build(rows) for k, rows in family_records.items()}
    paired = {}
    for response in RESPONSES:
        for family in ("gaze", "capture"):
            pair = comparison(family_records[("phase83", response, family, "x")],
                              family_records[("phase83", response, family, "xy")])
            paired[f"{response}/{family}"] = pair["paired"]
    write_json(output/"backfill_scorecards.json", dict(schema="phase83_audit_backfill_v1", folds=cards,
        families=families, paired_phase83=paired, protocol=protocol_metadata(),
        historical_scores_changed=False, retrospective_validation_status="development; not nested retroactively"))


def information_task(root, output, fold, response, mask):
    root, output = Path(root), Path(output)
    dest = output/"information"/response/fold/mask
    if (dest/"completion.json").exists():
        return
    dest.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    captures, groups, _ = reviewed(str(root))
    frozen = source_folder(root, response, fold)
    model, meta = load_model(frozen/"model.json")
    pilot = PositionModel(27, meta["pilot_coefficients"])
    reference, sigma = np.asarray(meta["reference_state"]), np.asarray(meta["coordinate_covariance"])
    population = _read_population(frozen.parent/"population.csv.gz")
    frames = read_json(frozen/"frames.json")
    holds, profile_count = [], 0
    with gzip.open(dest/"inverse_candidates.jsonl.gz", "wt") as archive, gzip.open(dest/"profile_audits.jsonl.gz", "wt") as profiles:
        for row in population:
            if row["selected_for_evaluation"] != "True":
                continue
            gi, i = int(row["fixation"]), int(row["row"])
            cap, group = captures[row["capture"]], groups[gi]
            ctx = context(cap.p[i])
            if not ctx.valid:
                continue
            for j in range(3):
                prediction = predict_raw_holdout(model, cap.p[i], cap.q[i], cap.point_valid[i], j,
                    pilot, reference, sigma, mask)
                archive.write(json.dumps(clean_json(dict(fixation=gi, row=i, held_point=j,
                    candidates=prediction.pop("candidates", []))), allow_nan=False)+"\n")
                # Independent polynomial profile is a diagnostic, not a new branch
                # policy or an excluded-error-based replacement of the primary solve.
                if prediction.get("available") and (prediction.get("rank", 0) < 2 or prediction.get("ambiguous")):
                    kept, H = retained_transform(j, mask)
                    R = reference_covariance(cap.p[i:i+1], pilot, reference, sigma)[0]
                    y = (cap.q[i].ravel()[kept]-ctx.c[kept % 2])/ctx.ell
                    transformed = TransformedResponse(model, kept, H)
                    diagnostic = profile_inverse(transformed, ctx.r, H@y, H@marginal(R, kept)@H.T, grid=65)
                    profiles.write(json.dumps(clean_json(dict(fixation=gi, row=i, held_point=j,
                        primary_cost=prediction["cost"], diagnostic=diagnostic)), allow_nan=False)+"\n")
                    profile_count += 1
                score = score_holdout(prediction, cap.q[i, j] if cap.point_valid[i, j] else np.full(2, np.nan), ctx.ell)
                score.update(fixation=gi, capture=cap.name, row=i, nominal_theta=group["target_theta_deg"],
                             demand=group["demand_diopters_label"], retained_image_channels=mask)
                holds.append(score)
    with (dest/"holdouts.jsonl").open("w") as stream:
        for row in holds:
            stream.write(json.dumps(clean_json(row), allow_nan=False)+"\n")
    joined = join_records(fold.split("_")[0], fold, response, population, frames, holds)
    supported(root, joined, frozen, mask)
    write_json(dest/"frames.json", joined)
    write_json(dest/"scorecard.json", build(joined))
    write_json(dest/"completion.json", dict(complete=True, seconds=time.monotonic()-started,
        frozen_model_sha256=digest(frozen/"model.json"), profile_diagnostic_cases=profile_count,
        primary_policy="unchanged 49 scalar starts and polishing; profiles diagnostic only"))
    print(f"{response}/{fold}/{mask}: {time.monotonic()-started:.1f}s", flush=True)


def information_report(root, output, folds):
    cards, comparisons, equivalence = {}, {}, []
    with gzip.open(output/"information_paired_membership.jsonl.gz", "wt") as stream:
        for response in RESPONSES:
            for family in ("gaze", "capture"):
                records = {mask: [] for mask in (*MASKS, "xy")}
                for fold in folds:
                    if not fold.startswith(family+"_"):
                        continue
                    frozen = source_folder(root, response, fold)
                    for mask in ("x", "xy"):
                        path = root/BASE/"phase83_retained_channels_v1"/response/fold/(mask+"_frames.json")
                        records[mask].extend(supported(root, read_json(path), frozen, mask))
                    for mask in Y_MASKS:
                        records[mask].extend(read_json(output/"information"/response/fold/mask/"frames.json"))
                cards[f"{response}/{family}"] = {m: build(rows) for m, rows in records.items()}
                for mask in Y_MASKS:
                    key = f"{response}/{family}/{mask}_minus_x"
                    pair = comparison(records["x"], records[mask])["paired"]
                    stream.write(json.dumps(clean_json(dict(comparison=key, points=pair.pop("points"),
                        frames=pair.pop("frames"))), allow_nan=False)+"\n")
                    comparisons[key] = pair
                # Compare discovered branch sets without choosing by excluded error.
                xy = {(f["capture"], f["fixation"], f["row"], s["held_point"]): s
                      for f in records["xy"] for s in f["slots"]}
                for f in records["x_y_common_difference"]:
                    for s in f["slots"]:
                        key = (f["capture"], f["fixation"], f["row"], s["held_point"])
                        old = xy[key]
                        branches = s.get("branches", [])
                        reference_branches = old.get("branches", [])
                        matches = [any(np.all(np.abs(np.asarray(b["state"])-o["state"]) < [.01, .01]) for o in reference_branches)
                                   for b in branches]
                        reverse = [any(np.all(np.abs(np.asarray(b["state"])-o["state"]) < [.01, .01]) for b in branches)
                                   for o in reference_branches]
                        equivalent = (s["available"] == old["available"] and s.get("rank") == old.get("rank") and
                            s["unambiguous"] == old["unambiguous"] and all(matches) and all(reverse))
                        equivalence.append(dict(response=response, family=family, identity=key,
                            branch_clusters_equivalent=bool(equivalent), xy_branch_count=len(reference_branches),
                            transformed_branch_count=len(branches), xy_rank=old.get("rank"), transformed_rank=s.get("rank"),
                            selected_state_max_abs_difference=(np.abs(np.asarray(s["state"])-old["state"]).tolist()
                                if s.get("state") is not None and old.get("state") is not None else None),
                            cost_abs_difference=(abs(s["retained_subset_cost"]-old["retained_subset_cost"])
                                if s.get("retained_subset_cost") is not None and old.get("retained_subset_cost") is not None else None)))
    write_json(output/"information_summary.json", dict(schema="retained_y_information_v1", scorecards=cards,
        paired_comparisons=comparisons, masks=MASKS, covariance_policy="H @ marginal(R,retained) @ H.T",
        promotion="no_promotion_decision; exploratory frozen-response masks", interpretation="image-y components; no vertical gaze state"))
    write_json(output/"invertible_equivalence.json", dict(cases=equivalence,
        branch_cluster_tolerance=[.01, .01], all_branch_clusters_equivalent=all(r["branch_clusters_equivalent"] for r in equivalence),
        meaning="finite multistart discovered clusters, not global branch completeness; exact objective invariance independently tested"))


def residual_diagnostics(root, output, folds):
    captures, groups, _ = reviewed(str(root))
    training = []
    for fold in folds:
        _, _, data, anchors, _ = checked_training(root, root/BASE/"audit_polished_v1", fold)
        for response in RESPONSES:
            frozen = source_folder(root, response, fold)
            model, meta = load_model(frozen/"model.json")
            saved = read_json(frozen/"training_states.json")
            if not np.array_equal(saved["rows"], data["rows"]) or not np.array_equal(saved["groups"], data["groups"]):
                raise ValueError("Training residual population mismatch")
            states = np.asarray(saved["states"])
            residual = (data["v"]-model.predict(states, data["r"])).reshape(-1, 3, 2)*context(data["p"]).ell[:, None, None]
            for i, gi in enumerate(data["original_group"]):
                group = groups[gi]
                for j in range(3):
                    training.append(dict(response=response, fold=fold, capture=group["capture"], fixation=int(gi),
                        row=int(data["rows"][i]), held_point=j, nominal_theta=group["target_theta_deg"],
                        error_px=residual[i, j].tolist(), scope="in_sample_training_residual_not_crosscheck"))
    with gzip.open(output/"training_residuals.jsonl.gz", "wt") as stream:
        for row in training:
            stream.write(json.dumps(row)+"\n")
    summaries = {}
    for response in RESPONSES:
        rows = [r for r in training if r["response"] == response]
        for field in ("capture", "nominal_theta", "held_point"):
            summaries[f"{response}/{field}"] = {}
            for value in sorted(set(r[field] for r in rows)):
                members = [r for r in rows if r[field] == value]
                summaries[f"{response}/{field}"][str(value)] = dict(count=len(members),
                    signed_axis_px={axis: stats(r["error_px"][i] for r in members) for i, axis in enumerate(("x", "y"))})
    write_json(output/"training_residual_summary.json", dict(scope="secondary in-sample diagnostic; no model selection",
        summaries=summaries, note="overlapping fold training populations are correlated; no independent uncertainty claim"))
    held_summaries = {}
    for response in RESPONSES:
        for family in ("gaze", "capture"):
            for mask in ("x", "xy"):
                frames = []
                for fold in folds:
                    if fold.startswith(family+"_"):
                        path = root/BASE/"phase83_retained_channels_v1"/response/fold/(mask+"_frames.json")
                        frames.extend(supported(root, read_json(path), source_folder(root, response, fold), mask))
                for field in ("capture", "fixation", "nominal_theta"):
                    held_summaries[f"{response}/{family}/{mask}/{field}"] = {
                        str(value): build([f for f in frames if f[field] == value])
                        for value in sorted(set(f[field] for f in frames))}
    write_json(output/"held_residual_summary.json", dict(scope="group-held-out development cross-check; no tuning",
        summaries=held_summaries, paired_axis_changes="backfill_scorecards.json paired_phase83; identical cohorts",
        interpretation="signed image axes, per-point errors, worst point and tails; nominal gaze is only a stratum label"))


def run(root, output, workers=12, mode="all"):
    root, output = Path(root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    folds = read_json(root/BASE/"joint_sensitivity_v1/config.json")["folds"]
    frozen_hashes = {str(p.relative_to(root)): digest(p) for name in SOURCES
                     for p in (root/BASE/name).rglob("*") if p.is_file()}
    implementation = source_hashes(root)
    config = dict(schema="phase83_audit_followup_v1", mode=mode, workers=workers, folds=folds,
        frozen_source_hashes=frozen_hashes, implementation_hashes=implementation, protocol=protocol_metadata(),
        masks=MASKS, profile_policy="65-grid independent diagnostic for primary rank/ambiguity cases; never replaces scores",
        thread_environment={k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")},
        interpretation="development follow-up; no hyperparameter promotion; captures5/6 untouched")
    if (output/"config.json").exists() and read_json(output/"config.json") != clean_json(config):
        raise ValueError("Resume requires identical sources, implementation and settings")
    write_json(output/"config.json", config)
    for rel in implementation:
        folder = "implementation_snapshot" if rel.startswith("full_position/") else "design_snapshot"
        dest = output/folder/Path(rel).name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root/rel, dest)
    started = time.monotonic()
    if mode in ("all", "backfill"):
        backfill(root, output, folds)
        residual_diagnostics(root, output, folds)
    if mode in ("all", "information"):
        jobs = [(fold, response, mask) for fold in folds for response in RESPONSES for mask in Y_MASKS]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(information_task, root, output, *job) for job in jobs]
            for future in as_completed(futures):
                future.result()
        information_report(root, output, folds)
    unchanged = all(digest(root/p) == h for p, h in frozen_hashes.items())
    if not unchanged:
        raise RuntimeError("Historical artifacts changed during audit follow-up")
    write_json(output/"completion.json", dict(complete=True, seconds=time.monotonic()-started,
        historical_sources_unchanged=unchanged, information_tasks=54 if mode in ("all", "information") else 0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--mode", choices=("all", "backfill", "information"), default="all")
    args = parser.parse_args()
    run(args.root, args.output, args.workers, args.mode)
