"""Direct saved-result comparison and training-only common-y diagnosis."""
from __future__ import annotations
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
from pathlib import Path
import shutil
import time
import numpy as np
from .audit83 import BASE, source_folder
from .audit_followup import transitions
from .crosscheck import stats
from .geometry import context
from .phase83 import RESPONSES
from .population import frame_id, manifest
from .schema import clean_json, load_model, source_hashes, write_json
from .scorecard import build, equal_exposure, exposure
from .sensitivity import checked_training, digest, read_json

SOURCE = "phase83_audit_followup_v1"
COHORTS = ("full_common", "joint_interior", "joint_empirical_state", "joint_P1", "joint_state_and_P1")
TARGETS = ("common_all_y", "pair_common_held0", "pair_common_held1", "pair_common_held2",
           "diff_y01", "diff_y02", "diff_y12", "point_y0", "point_y1", "point_y2", "point_x0", "point_x1", "point_x2")


def supported_slot(slot, cohort):
    if cohort == "full_common":
        return True
    if cohort == "joint_interior":
        return slot.get("interior") is True
    s = slot.get("support", {})
    state = s.get("theta_empirical") is True and s.get("A_empirical") is True
    p1 = (s.get("P1_context_valid") is True and s.get("context_outside_training_extrema") is False
          and s.get("p1_parity_seen_in_training") is True)
    return state if cohort == "joint_empirical_state" else p1 if cohort == "joint_P1" else state and p1


def squared_delta(rows, reference, candidate, expected):
    return equal_exposure(rows, np.square(candidate)-np.square(reference), expected)


def paired_summary(reference, candidate, population, cohort):
    population.validate(reference)
    population.validate(candidate)
    ref = {frame_id(f): f for f in reference}
    cand = {frame_id(f): f for f in candidate}
    frames, points = [], []
    for key in population.frame_ids:
        a, b = ref[key], cand[key]
        sa = {s["held_point"]: s for s in a["slots"]}
        sb = {s["held_point"]: s for s in b["slots"]}
        keep = [j for j in range(3) if sa[j]["scored"] and sb[j]["scored"]
                and supported_slot(sa[j], cohort) and supported_slot(sb[j], cohort)]
        points.extend((key+(j,), sa[j], sb[j]) for j in keep)
        if len(keep) == 3 and a["complete_triple"] and b["complete_triple"]:
            frames.append((key, a, b))
    expected = population.exposures
    frame_changes = {}
    for field in ("E_px", "G_theta_deg", "G_A_D", "worst_point_px"):
        rows = [a for _, a, _ in frames]
        av, bv = [a[field] for _, a, _ in frames], [b[field] for _, _, b in frames]
        frame_changes[field] = dict(squared_change=squared_delta(rows, av, bv, expected),
            reference_distribution=stats(av), candidate_distribution=stats(bv),
            p95_change=stats(bv)["p95"]-stats(av)["p95"] if frames else None)
    point_rows = [a for _, a, _ in points]
    point_changes = {}
    for axis, name in enumerate(("x", "y")):
        av = [a["error_px"][axis] for _, a, _ in points]
        bv = [b["error_px"][axis] for _, _, b in points]
        point_changes[name] = dict(squared_change=squared_delta(point_rows, av, bv, expected),
            reference_signed=stats(av), candidate_signed=stats(bv))
    point_changes["vector"] = squared_delta(point_rows,
        [np.linalg.norm(a["error_px"]) for _, a, _ in points],
        [np.linalg.norm(b["error_px"]) for _, _, b in points], expected)
    per_point = {}
    for j in range(3):
        members = [(a, b) for _, a, b in points if a["held_point"] == j]
        per_point[str(j)] = {name: squared_delta([a for a, _ in members],
            [a["error_px"][axis] for a, _ in members], [b["error_px"][axis] for _, b in members], expected)
            for axis, name in enumerate(("x", "y"))}
    arm_cards = {name: build([item[i] for item in frames], expected_exposures=expected)
                 for name, i in (("reference", 1), ("candidate", 2))}
    report = dict(direction="differential_y_minus_xy", cohort=cohort,
        scheduled_frames=len(population.frame_ids), scheduled_slots=3*len(population.frame_ids),
        shared_complete_frames=len(frames), shared_scored_points=len(points),
        shared_frame_fraction=len(frames)/len(population.frame_ids),
        frame_changes=frame_changes, point_changes=point_changes, per_point_axis_changes=per_point,
        same_frame_cohort_scorecards=arm_cards,
        scientific_scope="paired frozen-response development; no promotion thresholds defined")
    membership = dict(cohort=cohort, frame_ids=[k for k, _, _ in frames], point_ids=[k for k, _, _ in points])
    return report, membership


def direct_comparison(root, output, folds):
    source = root/BASE/SOURCE
    refs = defaultdict(list)
    with gzip.open(source/"scorecard_frames.jsonl.gz", "rt") as stream:
        for line in stream:
            item = json.loads(line)
            parts = item["candidate"].split("/")
            if parts[0] == "phase83" and parts[-1] == "xy":
                refs[(parts[1], parts[2])].append(item["frame"])
    reports = {}
    with gzip.open(output/"direct_paired_membership.jsonl.gz", "wt") as archive:
        for response in RESPONSES:
            for family in ("gaze", "capture"):
                reference, candidate, schedule = [], [], []
                for fold in folds:
                    if not fold.startswith(family+"_"):
                        continue
                    reference.extend(refs[(response, fold)])
                    candidate.extend(read_json(source/"information"/response/fold/"x_y_difference/frames.json"))
                    # Immutable, independently saved evaluation sampling predates all mask predictions.
                    from .crosscheck import _read_population
                    for row in _read_population(source_folder(root, response, fold).parent/"population.csv.gz"):
                        if row["selected_for_evaluation"] == "True":
                            schedule.append(dict(fold=fold, capture=row["capture"], fixation=int(row["fixation"]),
                                row=int(row["row"]), source_frame_index=int(row["frame"])))
                population = manifest(schedule)
                population.validate(reference)
                population.validate(candidate)
                key = f"{response}/{family}"
                cells = {}
                for cohort in COHORTS:
                    cells[cohort], members = paired_summary(reference, candidate, population, cohort)
                    archive.write(json.dumps(clean_json(dict(comparison=key, **members)), allow_nan=False)+"\n")
                pair = transitions(reference, candidate)
                archive.write(json.dumps(clean_json(dict(comparison=key, cohort="all_transitions",
                    points=pair.pop("points"), frames=pair.pop("frames"))), allow_nan=False)+"\n")
                reports[key] = dict(cohorts=cells, strata=pair["strata"],
                    full_reference=build(reference, population=population), full_candidate=build(candidate, population=population))
    write_json(output/"direct_comparison.json", dict(schema="direct_differential_y_vs_xy_v1", comparisons=reports,
        direction="candidate_minus_reference", reference="xy", candidate="x_y_difference", promotion="no_promotion_decision"))


def ridge_predict(train_x, train_y, x):
    """Fixed ridge=1 diagnostic; standardization/centering fit on supplied train only."""
    mean = train_x.mean(0)
    scale = np.maximum(train_x.std(0), 1e-8)
    B, C = (train_x-mean)/scale, (x-mean)/scale
    center = train_y.mean(0)
    beta = np.linalg.solve(B.T@B+np.eye(B.shape[1]), B.T@(train_y-center))
    return C@beta+center


def training_task(job):
    root, output, fold, response = job
    root, output = Path(root), Path(output)
    _, groups, data, _, _ = checked_training(root, root/BASE/"audit_polished_v1", fold)
    frozen = source_folder(root, response, fold)
    model, meta = load_model(frozen/"model.json")
    saved = read_json(frozen/"training_states.json")
    if not np.array_equal(saved["rows"], data["rows"]) or not np.array_equal(saved["groups"], data["groups"]):
        raise ValueError("Training diagnosis population mismatch")
    states = np.asarray(saved["states"])
    ctx = context(data["p"])
    residual = (data["v"]-model.predict(states, data["r"])).reshape(-1, 3, 2)*ctx.ell[:, None, None]
    y = residual[..., 1]
    targets = np.column_stack((y.mean(1), (y[:, 1]+y[:, 2])/2, (y[:, 0]+y[:, 2])/2,
        (y[:, 0]+y[:, 1])/2, y[:, 0]-y[:, 1], y[:, 0]-y[:, 2], y[:, 1]-y[:, 2], y, residual[..., 0]))
    original = np.asarray(data["original_group"])
    captures = np.asarray([groups[i]["capture"] for i in original])
    nominal = np.asarray([groups[i]["target_theta_deg"] for i in original])
    t, A = states[:, 0]/10., states[:, 1]/4.
    Xstate = np.column_stack((t, A, t*t, t*A))
    Xcontext = np.column_stack((Xstate, data["r"].reshape(len(t), -1), ctx.c, ctx.ell))
    dummies = np.column_stack([captures == name for name in sorted(set(captures))]).astype(float)
    feature_sets = dict(state=Xstate, state_P1=Xcontext,
        state_P1_gaze=np.column_stack((Xcontext, nominal/10., (nominal/10.)**2)),
        state_P1_capture=np.column_stack((Xcontext, dummies, dummies*t[:, None])))
    if set(original) & set(meta["provenance"]["evaluation_group_ids"]):
        raise ValueError("Outer evaluation groups entered training diagnosis")
    metrics = {}
    predictions = {}
    for scheme, labels in (("fixation", original), ("capture", captures), ("horizontal_gaze", nominal)):
        baseline = np.zeros_like(targets)
        by_features = {name: np.zeros_like(targets) for name in feature_sets}
        for level in np.unique(labels):
            test = labels == level
            train = ~test
            if not train.any():
                raise ValueError("Blocked residual regression has no training groups")
            baseline[test] = targets[train].mean(0)
            for name, features in feature_sets.items():
                by_features[name][test] = ridge_predict(features[train], targets[train], features[test])
        def eq_mse(error):
            return np.mean([np.square(error[original == gi]).mean(0) for gi in np.unique(original)], axis=0)
        mse0 = eq_mse(targets-baseline)
        metrics[scheme] = {name: dict(zip(TARGETS, [dict(mse_px2=float(m), baseline_mse_px2=float(b),
            skill_vs_train_mean=float(1-m/b) if b > 0 else None) for m, b in zip(eq_mse(targets-pred), mse0)]))
            for name, pred in by_features.items()}
        predictions[scheme] = by_features["state_P1"]
    fitted = ridge_predict(Xcontext, targets, Xcontext)
    capture_summary = {}
    for name in sorted(set(captures)):
        ix = captures == name
        capture_summary[name] = dict(count=int(ix.sum()), common_y_raw=stats(targets[ix, 0]),
            common_y_after_state_P1=stats((targets-fitted)[ix, 0]),
            per_point_y_mean=residual[ix, :, 1].mean(0).tolist())
    dest = output/"training_diagnosis"/response/fold
    dest.mkdir(parents=True, exist_ok=True)
    with gzip.open(dest/"training_components.jsonl.gz", "wt") as stream:
        for i in range(len(t)):
            row = dict(fixation=int(original[i]), capture=str(captures[i]), row=int(data["rows"][i]),
                state=states[i].tolist(), nominal_theta=float(nominal[i]), P1_context=data["r"][i].tolist(),
                P1_center=ctx.c[i].tolist(), P1_scale=float(ctx.ell[i]), targets_px=dict(zip(TARGETS, targets[i].tolist())),
                state_P1_conditioned_residual_px=dict(zip(TARGETS, (targets-fitted)[i].tolist())))
            stream.write(json.dumps(row, allow_nan=False)+"\n")
    report = dict(response=response, fold=fold, frozen_model_sha256=digest(frozen/"model.json"),
        training_group_ids=sorted(map(int, set(original))), outer_group_ids=meta["provenance"]["evaluation_group_ids"],
        target_names=TARGETS, ridge_strength=1., blocked_diagnostic_metrics=metrics, capture_summary=capture_summary,
        scope="training-only residual diagnosis of frozen jointly fitted response/states; blocked regression is not nested calibration validation",
        caveats="capture and accommodation demand are confounded; capture regressors diagnostic only, never free held-capture correction",
        tau_selection="none; predictable context/capture bias and unpredictable shared variation require separate evidence")
    write_json(dest/"summary.json", report)
    return report


def run(root, output, workers=12):
    root, output = Path(root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output/"completion.json").exists():
        raise ValueError("Use a new output directory for a completed study")
    sources = ("audit_polished_v1", "joint_sensitivity_v1", "axis_anchor_sensitivity_v1",
               "phase83_retained_channels_v1", SOURCE)
    hashes = {str(p.relative_to(root)): digest(p) for name in sources for p in (root/BASE/name).rglob("*")
              if p.is_file() and "__pycache__" not in p.parts}
    implementation = source_hashes(root)
    folds = read_json(root/BASE/"joint_sensitivity_v1/config.json")["folds"]
    config = dict(schema="latest_results_audit_followup_v1", workers=workers, folds=folds,
        source_hashes=hashes, implementation_hashes=implementation,
        diagnosis_policy="fixed ridge=1; scaling and regressor fit within training blocks; no outer labels/errors used",
        evaluation_scope="saved development masks; no new calibration or final transfer", guards="not supplied; no promotion",
        covariance_candidate="shared_y_covariance implemented; not automatically selected or launched")
    if (output/"config.json").exists() and read_json(output/"config.json") != clean_json(config):
        raise ValueError("Resume requires identical sources, implementation and policy")
    write_json(output/"config.json", config)
    for rel in implementation:
        folder = "implementation_snapshot" if rel.startswith("full_position/") else "design_snapshot"
        dest = output/folder/Path(rel).name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root/rel, dest)
    started = time.monotonic()
    direct_comparison(root, output, folds)
    jobs = [(root, output, fold, response) for fold in folds for response in RESPONSES]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        reports = list(pool.map(training_task, jobs))
    write_json(output/"training_diagnosis.json", dict(reports=reports,
        purpose="training-only diagnostic; no automatic covariance/response correction",
        interpretation="blocked regressor diagnostics do not remove prior joint-calibration reuse of these training groups"))
    if any(digest(root/rel) != h for rel, h in hashes.items()):
        raise RuntimeError("Historical sources changed")
    write_json(output/"completion.json", dict(complete=True, seconds=time.monotonic()-started,
        source_files_unchanged=len(hashes), direct_comparison_cells=4, training_diagnosis_tasks=len(jobs), workers=workers))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    run(args.root, args.output, args.workers)
