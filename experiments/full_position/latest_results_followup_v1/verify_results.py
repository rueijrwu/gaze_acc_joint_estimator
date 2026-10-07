#!/usr/bin/env python3
"""Independent verification for the saved latest-results follow-up."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASE = ROOT / "experiments/full_position"
sys.path.insert(0, str(ROOT))
RESPONSES = {"baseline27": ("baseline", "conditional27"),
             "strong_anchor37": ("anchor_strong", "conditional37")}
COHORTS = ("full_common", "joint_interior", "joint_empirical_state", "joint_P1",
           "joint_state_and_P1")
TARGETS = ("common_all_y", "pair_common_held0", "pair_common_held1", "pair_common_held2",
           "diff_y01", "diff_y02", "diff_y12", "point_y0", "point_y1", "point_y2",
           "point_x0", "point_x1", "point_x2")
TOL = 2e-8


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def same(a, b, tol=TOL):
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, (int, float, np.number)) and isinstance(b, (int, float, np.number)):
        return bool(np.isclose(float(a), float(b), atol=tol, rtol=tol, equal_nan=False))
    return a == b


def compare_numbers(got, expected, label):
    require(same(got, expected), f"{label}: saved={got!r}; recomputed={expected!r}")


def describe(values):
    v = np.asarray(list(values), dtype=float)
    v = v[np.isfinite(v)]
    if not len(v):
        return {"n": 0, "mean": None, "median": None, "rms": None, "p90": None, "p95": None}
    return {"n": int(len(v)), "mean": float(v.mean()), "median": float(np.median(v)),
            "rms": float(np.sqrt(np.mean(v*v))), "p90": float(np.percentile(v, 90)),
            "p95": float(np.percentile(v, 95))}


def exposure_mean(rows, values, expected):
    grouped = defaultdict(list)
    for row, value in zip(rows, values):
        if value is not None and np.isfinite(float(value)):
            grouped[(row["fold"], row["capture"], int(row["fixation"]))].append(float(value))
    absent = sorted(set(expected)-set(grouped))
    means = [float(np.mean(v)) for v in grouped.values()]
    return {"mean": float(np.mean(means)) if means else None,
            "contributing_exposure_count": len(grouped), "scheduled_exposure_count": len(expected),
            "absent_exposure_ids": [list(k) for k in absent],
            "status": "complete_exposure_support" if not absent else "partial_exposure_support"}


def support_ok(slot, cohort):
    if cohort == "full_common":
        return True
    if cohort == "joint_interior":
        return slot.get("interior") is True
    s = slot.get("support", {})
    state = s.get("theta_empirical") is True and s.get("A_empirical") is True
    p1 = (s.get("P1_context_valid") is True and
          s.get("context_outside_training_extrema") is False and
          s.get("p1_parity_seen_in_training") is True)
    if cohort == "joint_empirical_state":
        return state
    if cohort == "joint_P1":
        return p1
    return state and p1


def frame_key(frame):
    return (frame["fold"], frame["capture"], int(frame["fixation"]), int(frame["row"]),
            int(frame["source_frame_index"]))


def independent_schedule(response, folds):
    variant, _ = RESPONSES[response]
    keys = []
    for fold in folds:
        pop_path = BASE / "joint_sensitivity_v1" / "variants" / variant / fold / "population.csv.gz"
        with gzip.open(pop_path, "rt", newline="") as stream:
            for row in csv.DictReader(stream):
                if row["selected_for_evaluation"] != "True":
                    continue
                keys.append((fold, row["capture"], int(row["fixation"]), int(row["row"]), int(row["frame"])))
    require(len(keys) == len(set(keys)), f"duplicate population identities for {response}")
    return tuple(sorted(keys))


def check_card(card, frames, expected_exposures, label):
    slots = [s for f in frames for s in f["slots"]]
    complete = [f for f in frames if f["complete_triple"]]
    counts = card["coverage"]
    for key, value in (("scheduled_frames", len(frames)), ("scheduled_slots", len(slots)),
                       ("scored", sum(bool(s["scored"]) for s in slots)),
                       ("complete_triples", len(complete))):
        compare_numbers(counts[key], value, f"{label} coverage.{key}")
    for sup in ("theta_anchor", "A_anchor", "theta_empirical", "A_empirical", "P1_context_valid",
                "context_outside_training_extrema", "p1_parity_seen_in_training"):
        counter = Counter("unknown" if s.get("support", {}).get(sup) is None else
                          "true" if s["support"][sup] else "false" for s in slots)
        for category in ("true", "false", "unknown"):
            compare_numbers(card["support_counts"][sup].get(category, 0), counter.get(category, 0),
                            f"{label} support {sup}/{category}")
    for axis, index in (("x", 0), ("y", 1)):
        rows = [s for s in slots if s["scored"]]
        values = [s["error_px"][index] for s in rows]
        agg = exposure_mean(rows, [v*v for v in values], expected_exposures)
        rms = float(np.sqrt(agg["mean"])) if agg["mean"] is not None else None
        compare_numbers(card["axis_errors"][axis]["equal_exposure_rms"], rms,
                        f"{label} axis {axis} RMS")
    for key, field in (("E_cross_px", "E_px"), ("G_theta_cross_deg", "G_theta_deg"),
                       ("G_A_cross_D", "G_A_D"), ("worst_point_px", "worst_point_px")):
        rows = [f for f in complete if f.get(field) is not None]
        agg = exposure_mean(rows, [f[field]**2 for f in rows], expected_exposures)
        rms = float(np.sqrt(agg["mean"])) if agg["mean"] is not None else None
        compare_numbers(card["outcomes"][key]["equal_exposure_rms"], rms, f"{label} {key} RMS")


def recompute_cohort(reference, candidate, schedule, cohort):
    rmap = {frame_key(f): f for f in reference}
    cmap = {frame_key(f): f for f in candidate}
    point_rows, frame_rows = [], []
    for key in schedule:
        a, b = rmap[key], cmap[key]
        sa = {int(s["held_point"]): s for s in a["slots"]}
        sb = {int(s["held_point"]): s for s in b["slots"]}
        keep = [j for j in range(3) if sa[j]["scored"] and sb[j]["scored"] and
                support_ok(sa[j], cohort) and support_ok(sb[j], cohort)]
        point_rows.extend((key+(j,), sa[j], sb[j]) for j in keep)
        if len(keep) == 3 and a["complete_triple"] and b["complete_triple"]:
            frame_rows.append((key, a, b))
    return point_rows, frame_rows


def check_delta_report(report, points, frames, expected, label):
    compare_numbers(report["shared_complete_frames"], len(frames), f"{label} complete-frame count")
    compare_numbers(report["shared_scored_points"], len(points), f"{label} scored-point count")
    compare_numbers(report["shared_frame_fraction"], len(frames)/report["scheduled_frames"],
                    f"{label} complete-frame coverage")
    for axis, idx in (("x", 0), ("y", 1)):
        a = [old["error_px"][idx] for _, old, _ in points]
        b = [new["error_px"][idx] for _, _, new in points]
        agg = exposure_mean([old for _, old, _ in points], [bv*bv-av*av for av, bv in zip(a, b)], expected)
        compare_numbers(report["point_changes"][axis]["squared_change"]["mean"], agg["mean"],
                        f"{label} point Δ{axis}²")
        for side, values in (("reference_signed", a), ("candidate_signed", b)):
            stats_saved = report["point_changes"][axis][side]
            stats_recomputed = describe(values)
            for metric in stats_recomputed:
                compare_numbers(stats_saved[metric], stats_recomputed[metric], f"{label} {axis} {side}/{metric}")
    vec_a = [float(np.linalg.norm(old["error_px"])) for _, old, _ in points]
    vec_b = [float(np.linalg.norm(new["error_px"])) for _, _, new in points]
    agg = exposure_mean([old for _, old, _ in points], [bv*bv-av*av for av, bv in zip(vec_a, vec_b)], expected)
    compare_numbers(report["point_changes"]["vector"]["mean"], agg["mean"], f"{label} point Δvector²")
    for j in range(3):
        members = [(old, new) for _, old, new in points if int(old["held_point"]) == j]
        for axis, idx in (("x", 0), ("y", 1)):
            agg = exposure_mean([old for old, _ in members],
                [new["error_px"][idx]**2-old["error_px"][idx]**2 for old, new in members], expected)
            compare_numbers(report["per_point_axis_changes"][str(j)][axis]["mean"], agg["mean"],
                            f"{label} point{j} Δ{axis}²")
    for field in ("E_px", "G_theta_deg", "G_A_D", "worst_point_px"):
        a = [old[field] for _, old, _ in frames]
        b = [new[field] for _, _, new in frames]
        agg = exposure_mean([old for _, old, _ in frames], [bv*bv-av*av for av, bv in zip(a, b)], expected)
        compare_numbers(report["frame_changes"][field]["squared_change"]["mean"], agg["mean"],
                        f"{label} {field} squared delta")
        da, db = describe(a), describe(b)
        for metric in da:
            compare_numbers(report["frame_changes"][field]["reference_distribution"][metric], da[metric],
                            f"{label} {field} reference {metric}")
            compare_numbers(report["frame_changes"][field]["candidate_distribution"][metric], db[metric],
                            f"{label} {field} candidate {metric}")
        compare_numbers(report["frame_changes"][field]["p95_change"], db["p95"]-da["p95"],
                        f"{label} {field} p95 change")


def check_transition_strata(summary, point_records, frame_records, label):
    def get_stratum(kind, row):
        if kind == "transition":
            return row["transition"]
        if kind == "fixation":
            return str((row["identity"][1], row["capture"], row["fixation"]))
        if kind == "signed_gaze":
            return str(row["signed_gaze"])
        return row["capture"]

    def summarize(rows, field):
        selected = [r for r in rows if field in r]
        vals = [r[field] for r in selected]
        grouped = defaultdict(list)
        for r in selected:
            grouped[(r["identity"][1], r["capture"], r["fixation"])].append(r[field])
        return {"pooled": describe(vals),
                "equal_fixation_mean": float(np.mean([np.mean(v) for v in grouped.values()])) if grouped else None,
                "exposure_count": len(grouped),
                "per_exposure_counts": {str(k): len(v) for k, v in sorted(grouped.items())}}

    for kind in ("transition", "fixation", "signed_gaze", "capture"):
        saved = summary["strata"][kind]
        labels = set(get_stratum(kind, r) for r in point_records)
        labels |= set(get_stratum(kind, r) for r in frame_records)
        if kind == "transition":
            labels |= {f"{a}_to_{b}" for a in ("interior", "bound") for b in ("interior", "bound")}
            labels.add("unavailable_pair")
        require(labels == set(saved), f"{label} {kind} stratum labels differ")
        for value in labels:
            p = [r for r in point_records if get_stratum(kind, r) == value]
            f = [r for r in frame_records if get_stratum(kind, r) == value]
            cell = saved[value]
            compare_numbers(cell["scheduled_points"], len(p), f"{label} {kind}/{value} scheduled points")
            compare_numbers(cell["shared_scored_points"], sum(r["shared_scored"] for r in p),
                            f"{label} {kind}/{value} shared points")
            for output, field in (("point_squared_error_change", "delta_squared_vector_error_px2"),):
                expected = summarize(p, field)
                for k, v in expected.items():
                    if k == "pooled":
                        for metric, number in v.items():
                            compare_numbers(cell[output][k][metric], number, f"{label} {kind}/{value} {output}/{k}/{metric}")
                    else:
                        compare_numbers(cell[output][k], v, f"{label} {kind}/{value} {output}/{k}")
            for axis in ("x", "y"):
                for output, field in (("point_axis_squared_changes", f"delta_squared_{axis}_error_px2"),
                                     ("frame_axis_squared_changes", f"delta_squared_{axis}_error_px2")):
                    rows = p if output.startswith("point") else f
                    expected = summarize(rows, field)
                    for k, v in expected.items():
                        if k == "pooled":
                            for metric, number in v.items():
                                compare_numbers(cell[output][axis][k][metric], number,
                                                f"{label} {kind}/{value} {output}/{axis}/{metric}")
                        else:
                            compare_numbers(cell[output][axis][k], v,
                                            f"{label} {kind}/{value} {output}/{axis}/{k}")


def check_training(root, diagnosis):
    from full_position.data import load_reviewed, training_data
    from full_position.geometry import context
    from full_position.schema import load_model

    captures, groups, _ = load_reviewed(root)
    seen = set()
    total_metrics = 0
    total_components = 0

    def pred(train_x, train_y, test_x):
        mean = train_x.mean(0)
        scale = np.maximum(train_x.std(0), 1e-8)
        b = (train_x-mean)/scale
        x = (test_x-mean)/scale
        center = train_y.mean(0)
        beta = np.linalg.solve(b.T@b+np.eye(b.shape[1]), b.T@(train_y-center))
        return x@beta+center

    for row in diagnosis["reports"]:
        response, fold = row["response"], row["fold"]
        key = (response, fold)
        require(key not in seen, f"duplicate training report {key}")
        seen.add(key)
        variant, model_name = RESPONSES[response]
        frozen = BASE/"joint_sensitivity_v1"/"variants"/variant/fold/model_name
        model, meta = load_model(root/frozen/"model.json")
        provenance = meta["provenance"]
        train_ids = set(map(int, provenance["training_group_ids"]))
        outer_ids = set(map(int, provenance["evaluation_group_ids"]))
        require(not train_ids & outer_ids and train_ids | outer_ids == set(range(len(groups))),
                f"invalid outer split {key}")
        require(sorted(train_ids) == row["training_group_ids"] and sorted(outer_ids) == row["outer_group_ids"],
                f"saved split metadata differs {key}")
        data, anchors = training_data(captures, groups, provenance["training_group_ids"], 48)
        state_file = load(root/frozen/"training_states.json")
        states = np.asarray(state_file["states"], float)
        require(np.array_equal(state_file["rows"], data["rows"]) and
                np.array_equal(state_file["groups"], data["groups"]) and states.shape == (len(data["rows"]), 2),
                f"frozen training rows/states do not align {key}")
        original = np.asarray(data["original_group"], int)
        require(set(original.tolist()) == train_ids and not set(original.tolist()) & outer_ids,
                f"outer group leaked into training rows {key}")
        residual = (data["v"]-model.predict(states, data["r"])).reshape(-1, 3, 2)
        scale_px = context(data["p"]).ell
        residual_px = residual*scale_px[:, None, None]
        y = residual_px[..., 1]
        targets = np.column_stack((y.mean(1), (y[:, 1]+y[:, 2])/2, (y[:, 0]+y[:, 2])/2,
            (y[:, 0]+y[:, 1])/2, y[:, 0]-y[:, 1], y[:, 0]-y[:, 2], y[:, 1]-y[:, 2],
            y, residual_px[..., 0]))
        require(tuple(row["target_names"]) == TARGETS and targets.shape[1] == len(TARGETS),
                f"training target schema mismatch {key}")
        nominal = np.asarray([groups[int(i)]["target_theta_deg"] for i in original], float)
        cap_labels = np.asarray([groups[int(i)]["capture"] for i in original])
        t, a = states[:, 0]/10., states[:, 1]/4.
        xstate = np.column_stack((t, a, t*t, t*a))
        ctx = context(data["p"])
        xcontext = np.column_stack((xstate, data["r"].reshape(len(t), -1), ctx.c, ctx.ell))
        cap_names = sorted(set(cap_labels))
        dummies = np.column_stack([cap_labels == name for name in cap_names]).astype(float)
        features = {"state": xstate, "state_P1": xcontext,
                    "state_P1_gaze": np.column_stack((xcontext, nominal/10., (nominal/10.)**2)),
                    "state_P1_capture": np.column_stack((xcontext, dummies, dummies*t[:, None]))}
        # Independently refit each held-group ridge regression and verify every target score.
        for scheme, labels in (("fixation", original), ("capture", cap_labels), ("horizontal_gaze", nominal)):
            labels = np.asarray(labels)
            baseline = np.zeros_like(targets)
            fitted = {name: np.zeros_like(targets) for name in features}
            for level in np.unique(labels):
                test = labels == level
                train = ~test
                require(train.any(), f"empty training side in {key}/{scheme}/{level}")
                baseline[test] = targets[train].mean(0)
                for name, X in features.items():
                    fitted[name][test] = pred(X[train], targets[train], X[test])
            baseline_mse = np.mean([np.square((targets-baseline)[original == gi]).mean(0)
                                    for gi in np.unique(original)], axis=0)
            for name, values in fitted.items():
                mse = np.mean([np.square((targets-values)[original == gi]).mean(0)
                               for gi in np.unique(original)], axis=0)
                for j, target in enumerate(TARGETS):
                    saved = row["blocked_diagnostic_metrics"][scheme][name][target]
                    compare_numbers(saved["mse_px2"], mse[j], f"{key}/{scheme}/{name}/{target} MSE")
                    compare_numbers(saved["baseline_mse_px2"], baseline_mse[j], f"{key}/{scheme}/{name}/{target} baseline")
                    score = 1-mse[j]/baseline_mse[j] if baseline_mse[j] > 0 else None
                    compare_numbers(saved["skill_vs_train_mean"], score, f"{key}/{scheme}/{name}/{target} skill")
                    total_metrics += 1
        # Recompute the all-training state+P1 fit and check every saved raw component row.
        fullfit = pred(xcontext, targets, xcontext)
        component_path = OUT/"training_diagnosis"/response/fold/"training_components.jsonl.gz"
        with gzip.open(component_path, "rt") as stream:
            components = [json.loads(line) for line in stream if line.strip()]
        require(len(components) == len(targets), f"component row count mismatch {key}")
        total_components += len(components)
        for i, component in enumerate(components):
            compare_numbers(component["fixation"], int(original[i]), f"{key} component fixation row {i}")
            compare_numbers(component["row"], int(data["rows"][i]), f"{key} component row {i}")
            for field, expected in (("state", states[i]), ("P1_context", data["r"][i]),
                                    ("P1_center", ctx.c[i])):
                require(np.allclose(component[field], expected, atol=TOL, rtol=TOL),
                        f"{key} component {field} mismatch at row {i}")
            compare_numbers(component["P1_scale"], ctx.ell[i], f"{key} component P1 scale row {i}")
            for j, target in enumerate(TARGETS):
                compare_numbers(component["targets_px"][target], targets[i, j],
                                f"{key} component {target} row {i}")
                compare_numbers(component["state_P1_conditioned_residual_px"][target],
                                targets[i, j]-fullfit[i, j], f"{key} conditioned {target} row {i}")
        # Capture-conditional signed common-y means are recomputed from pixels.
        for name in sorted(set(cap_labels)):
            ix = cap_labels == name
            selected = row["capture_summary"][name]
            for label, values in (("common_y_raw", targets[ix, 0]),
                                  ("common_y_after_state_P1", (targets-fullfit)[ix, 0])):
                stats = describe(values)
                for metric in stats:
                    compare_numbers(selected[label][metric], stats[metric], f"{key}/{name}/{label}/{metric}")
    require(seen == {(r, f) for r in RESPONSES for f in load(OUT/"config.json")["folds"]},
            "training diagnosis is not exactly two responses × nine folds")
    return {"report_count": len(seen), "blocked_metric_count": total_metrics,
            "component_rows_checked": total_components}


def check_covariance_trial(root):
    from full_position.audit83 import source_folder
    from full_position.data import load_reviewed, training_data
    from full_position.geometry import context
    from full_position.model import PositionModel, STATE_SCALE, LOWER, UPPER
    from full_position.noise import reference_covariance
    from full_position.schema import load_model

    trial = OUT/"covariance_trial"
    cfg = load(trial/"config.json")
    done = load(trial/"completion.json")
    require(done.get("complete") is True and done.get("tasks") == 9 and cfg.get("workers") == 12,
            "covariance trial did not finish the declared nine-fold/12-worker study")
    bad_sources = [name for name, expected in cfg["source_hashes"].items() if sha(root/name) != expected]
    require(not bad_sources, f"covariance-trial source changed: {bad_sources[:3]}")
    bad_snapshots = []
    for rel, expected in cfg["implementation_hashes"].items():
        folder = "design_snapshot" if rel in ("ESTIMATOR_PLAN.md", "Theory.md") else "implementation_snapshot"
        if not (trial/folder/Path(rel).name).exists() or sha(trial/folder/Path(rel).name) != expected:
            bad_snapshots.append(rel)
    require(not bad_snapshots, f"covariance-trial implementation snapshot mismatch: {bad_snapshots[:3]}")

    captures, groups, _ = load_reviewed(root)
    tau_by_fold = {}
    objective_count = 0
    prediction_count = 0
    certificate_checks = []
    all_trial_frames = {}
    all_trial_tau = {}
    for fold in cfg["folds"]:
        folder = trial/fold
        frozen = source_folder(root, "baseline27", fold)
        model, meta = load_model(frozen/"model.json")
        training = load(frozen/"training_states.json")
        data, _ = training_data(captures, groups, meta["provenance"]["training_group_ids"], 48)
        require(np.array_equal(training["rows"], data["rows"]) and
                np.array_equal(training["groups"], data["groups"]), f"trial training row order mismatch {fold}")
        states = np.asarray(training["states"], float)
        residual = (data["v"]-model.predict(states, data["r"])).reshape(-1, 3, 2)
        scale = context(data["p"]).ell
        residual_px = residual*scale[:, None, None]
        common_y = residual_px[..., 1].mean(1)
        original = np.asarray(data["original_group"])
        tau = float(np.sqrt(np.mean([np.square(common_y[original == group]).mean()
                                     for group in np.unique(original)])))
        tau_record = load(folder/"training_tau.json")
        compare_numbers(tau_record["tau_px"], tau, f"{fold} independently recomputed training tau")
        require(set(tau_record["training_group_ids"]) == set(map(int, np.unique(original))) and
                tau_record["use_outer_errors"] is False, f"trial tau declaration leaks outer evidence {fold}")
        compare_numbers(tau_record["frozen_model_sha256"], sha(frozen/"model.json"),
                        f"{fold} tau source-model digest")
        tau_by_fold[fold] = tau
        all_trial_tau[fold] = tau

        pilot = PositionModel(27, meta["pilot_coefficients"])
        reference_state = np.asarray(meta["reference_state"], float)
        sigma = np.asarray(meta["coordinate_covariance"], float)
        kept_records = []
        with open(folder/"holdouts.jsonl") as stream:
            kept_records = [json.loads(line) for line in stream if line.strip()]
        sampled_certificate_points = set()
        unavailable_holdouts = 0
        for h in kept_records:
            if not h.get("available"):
                require(h.get("state") is None and h.get("cost") is None,
                        f"unavailable holdout carries a selected solution {fold}/{h.get('row')}/{h.get('held_point')}")
                unavailable_holdouts += 1
                continue
            cap = captures[h["capture"]]
            i, held = int(h["row"]), int(h["held_point"])
            ctx = context(cap.p[i])
            require(bool(ctx.valid), f"invalid P1 entered covariance trial holdout {fold}/{i}/{held}")
            kept = np.asarray([j for j in range(6) if j//2 != held], dtype=int)
            require(np.array_equal(h["retained_indices"], kept.tolist()),
                    f"retained indices mismatch {fold}/{i}/{held}")
            require(bool(cap.point_valid[i, np.arange(3) != held].all()),
                    f"invalid retained P4 entered covariance trial {fold}/{i}/{held}")
            r0 = reference_covariance(cap.p[i:i+1], pilot, reference_state, sigma)[0]
            loading = np.array([0., 1., 0., 1., 0., 1.])
            adjusted = r0+(tau/ctx.ell)**2*np.outer(loading, loading)
            marginal = adjusted[np.ix_(kept, kept)]
            compare_numbers(h["shared_y_discrepancy_tau_px"], tau,
                            f"{fold}/{i}/{held} holdout tau")
            require(h["retained_image_channels"] == "xy_shared_y_training_rms",
                    f"unexpected trial retained-channel policy {fold}/{i}/{held}")
            observed = (cap.q[i].reshape(-1)[kept]-ctx.c[kept % 2])/ctx.ell
            for branch in h.get("branches", []):
                state = np.asarray(branch["state"], float)
                error = model.predict(state, ctx.r)[kept]-observed
                objective = float(error@np.linalg.solve(marginal, error))
                compare_numbers(branch["cost"], objective, f"{fold}/{i}/{held} branch retained objective")
                objective_count += 1
            if h.get("available") and h.get("state") is not None:
                state = np.asarray(h["state"], float)
                error = model.predict(state, ctx.r)[kept]-observed
                objective = float(error@np.linalg.solve(marginal, error))
                compare_numbers(h["cost"], objective, f"{fold}/{i}/{held} selected retained objective")
                objective_count += 1
                omitted = np.asarray([2*held, 2*held+1])
                for predicted in h.get("predictions", []):
                    branch_state = np.asarray(predicted["state"], float)
                    normalized = model.predict(branch_state, ctx.r)[omitted]
                    pixel = ctx.c[omitted % 2]+ctx.ell*normalized
                    require(np.allclose(predicted["normalized"], normalized, atol=TOL, rtol=TOL) and
                            np.allclose(predicted["pixel"], pixel, atol=TOL, rtol=TOL),
                            f"{fold}/{i}/{held} omitted-point prediction mismatch")
                    prediction_count += 1
                sample_key = (fold, held)
                branch = next((b for b in h.get("branches", [])
                               if np.allclose(b["state"], state, atol=1e-12, rtol=0)), None)
                if branch is not None and not h.get("at_bound") and sample_key not in sampled_certificate_points:
                    cert = branch["certificate"]
                    # Independently recompute the encoded-state gradient, exact Hessian,
                    # active face, free curvature, Newton correction and stable-cost probe.
                    encoded = state/STATE_SCALE
                    value, jac = model.predict(state, ctx.r, True)
                    e = value[kept]-observed
                    q = np.linalg.inv(marginal)
                    jz = jac[kept]*STATE_SCALE
                    hphys = model.state_hessian(state, ctx.r)[kept]
                    hz = hphys*STATE_SCALE[:, None]*STATE_SCALE[None, :]
                    gradient = jz.T@q@e
                    hessian = jz.T@q@jz+np.einsum("i,iab->ab", q@e, hz)
                    hessian = (hessian+hessian.T)/2
                    lower, upper = LOWER/STATE_SCALE, UPPER/STATE_SCALE
                    active = ((encoded-lower < 1e-9) & (gradient > 1e-7)) | \
                             ((upper-encoded < 1e-9) & (gradient < -1e-7))
                    free = np.flatnonzero(~active)
                    eigen = np.linalg.eigvalsh(hessian[np.ix_(free, free)]) if len(free) else np.array([])
                    stationarity = float(np.max(np.abs(encoded-np.clip(encoded-gradient, lower, upper))))
                    compare_numbers(cert["stationarity_encoded"], stationarity,
                                    f"{fold}/{i}/{held} certificate stationarity")
                    require(cert["certified"] and cert["stable_cost"] and cert["local_minimum"],
                            f"{fold}/{i}/{held} selected branch lacks accepted certificate")
                    require(np.allclose(cert["free_curvature_eigenvalues"], eigen, atol=2e-7, rtol=2e-7),
                            f"{fold}/{i}/{held} certificate curvature mismatch")
                    if len(free):
                        curvature_scale = max(1., float(np.max(np.abs(eigen))) if len(eigen) else 1.)
                        shift = max(0., 1e-12*curvature_scale-(eigen[0] if len(eigen) else 0.))
                        step = np.zeros(2)
                        step[free] = -np.linalg.solve(hessian[np.ix_(free, free)]+shift*np.eye(len(free)),
                                                       gradient[free])
                    else:
                        step = np.zeros(2)
                    projected = np.clip(encoded+step, lower, upper)-encoded
                    correction = float(np.max(np.abs(projected*STATE_SCALE)))
                    compare_numbers(cert["physical_correction"], correction,
                                    f"{fold}/{i}/{held} certificate correction")
                    probe = encoded+projected
                    pv = model.predict(probe*STATE_SCALE, ctx.r)[kept]-observed
                    probe_cost = float(pv@q@pv/2)
                    half_cost = objective/2
                    require(abs(probe_cost-half_cost) <= 1e-10*(1+half_cost),
                            f"{fold}/{i}/{held} selected certificate stable-cost probe failed")
                    certificate_checks.append(dict(fold=fold, held_point=held, row=i,
                        stationarity=stationarity, correction=correction,
                        minimum_free_curvature=float(eigen[0]) if len(eigen) else None))
                    sampled_certificate_points.add(sample_key)
        require(len(sampled_certificate_points) == 3,
                f"could not independently sample one interior certificate for every held point in {fold}")

        # The frame file must preserve every independently scheduled denominator.
        selected_schedule = independent_schedule("baseline27", [fold])
        frames = load(folder/"frames.json")
        require(set(frame_key(f) for f in frames) == set(selected_schedule) and len(frames) == len(selected_schedule),
                f"covariance trial frame schedule mismatch {fold}")
        exposures = sorted({k[:3] for k in selected_schedule})
        check_card(load(folder/"scorecard.json"), frames, exposures, f"covariance trial {fold}")
        all_trial_frames[fold] = frames

    summary = load(trial/"summary.json")
    paired = {}
    with gzip.open(trial/"paired_membership.jsonl.gz", "rt") as stream:
        for line in stream:
            item = json.loads(line)
            key = (item["comparison"], item["cohort"])
            require(key not in paired, f"duplicate covariance trial paired record {key}")
            paired[key] = item
    pair_checks = 0
    for family in ("gaze", "capture"):
        selected_folds = [f for f in cfg["folds"] if f.startswith(family+"_")]
        schedule = independent_schedule("baseline27", selected_folds)
        ref = []
        with gzip.open(BASE/"phase83_audit_followup_v1"/"scorecard_frames.jsonl.gz", "rt") as stream:
            for line in stream:
                item = json.loads(line)
                parts = item["candidate"].split("/")
                if parts[0] == "phase83" and parts[1] == "baseline27" and parts[2] in selected_folds and parts[3] == "xy":
                    ref.append(item["frame"])
        candidate = [f for fold in selected_folds for f in all_trial_frames[fold]]
        rmap, cmap = ({frame_key(f): f for f in rows} for rows in (ref, candidate))
        require(len(rmap) == len(ref) == len(schedule) and set(rmap) == set(schedule),
                f"covariance-trial reference schedule mismatch {family}")
        require(len(cmap) == len(candidate) == len(schedule) and set(cmap) == set(schedule),
                f"covariance-trial candidate schedule mismatch {family}")
        exposures = sorted({k[:3] for k in schedule})
        card = summary["families"][family]
        check_card(card["reference"], [rmap[k] for k in schedule], exposures, f"trial {family} reference")
        check_card(card["candidate"], [cmap[k] for k in schedule], exposures, f"trial {family} candidate")
        for cohort in COHORTS:
            points, frames = recompute_cohort([rmap[k] for k in schedule],
                [cmap[k] for k in schedule], schedule, cohort)
            members = paired[(family, cohort)]
            require(members["frame_ids"] == [list(k) for k, _, _ in frames] and
                    members["point_ids"] == [list(k) for k, _, _ in points],
                    f"covariance-trial paired membership mismatch {family}/{cohort}")
            report = card["cohorts"][cohort]
            check_delta_report(report, points, frames, exposures, f"trial/{family}/{cohort}")
            pair_checks += 1
    return {"status": "pass", "source_hash_count": len(cfg["source_hashes"]),
            "snapshot_count": len(cfg["implementation_hashes"]), "fold_tau_px": tau_by_fold,
            "tau_min_px": float(min(tau_by_fold.values())), "tau_max_px": float(max(tau_by_fold.values())),
            "saved_branch_objectives_recomputed": objective_count,
            "omitted_point_predictions_recomputed": prediction_count,
            "unavailable_holdouts_preserved": unavailable_holdouts,
            "selected_interior_certificates_recomputed": certificate_checks,
            "exact_paired_cohort_checks": pair_checks,
            "scope": "training-only tau; no candidate fitting; held-out P4 used only for final scoring"}


def scalar_profile_check():
    # Reproduce the named algebra test with its fixed seed, without importing test code.
    rng = np.random.default_rng(20261010)
    root = rng.normal(size=(4, 4))
    covariance = root@root.T+np.eye(4)*.8
    residual = rng.normal(size=4)
    shared = np.array([0., 1., 0., 1.])
    precision = np.linalg.inv(covariance)
    offset = (shared@precision@residual)/(shared@precision@shared)
    profiled = float((residual-offset*shared)@precision@(residual-offset*shared))
    H = np.array([[1., 0., 0., 0.], [0., 0., 1., 0.], [0., 1., 0., -1.]])
    transformed = H@residual
    transformed_covariance = H@covariance@H.T
    differential = float(transformed@np.linalg.solve(transformed_covariance, transformed))
    compare_numbers(profiled, differential, "free scalar shared-y offset vs x+differential-y profile")
    return {"test": "test_shared_y_marginal_precision_matches_free_scalar_offset_and_large_tau_limit",
            "profiled_quadratic": profiled, "differential_quadratic": differential,
            "absolute_difference": abs(profiled-differential), "tolerance": TOL}


def main():
    config = load(OUT/"config.json")
    completion = load(OUT/"completion.json")
    require(completion.get("complete") is True, "runner completion absent")
    checks = {}
    # Verify every raw frozen input digest and every copied implementation snapshot.
    bad_sources = [name for name, expected in config["source_hashes"].items()
                   if sha(ROOT/name) != expected]
    require(not bad_sources, f"changed source inputs: {bad_sources[:3]}")
    bad_snapshots = []
    for rel, expected in config["implementation_hashes"].items():
        folder = "design_snapshot" if rel in ("ESTIMATOR_PLAN.md", "Theory.md") else "implementation_snapshot"
        snapshot = OUT/folder/Path(rel).name
        if not snapshot.exists() or sha(snapshot) != expected:
            bad_snapshots.append(rel)
    require(not bad_snapshots, f"implementation snapshot mismatch: {bad_snapshots}")
    checks["source_hashes"] = {"status": "pass", "count": len(config["source_hashes"])}
    checks["implementation_snapshots"] = {"status": "pass", "count": len(config["implementation_hashes"])}

    initial_test_text = (OUT/"tests.txt").read_text()
    final_test_text = (OUT/"tests_final.txt").read_text()
    require("87 passed" in initial_test_text and "90 passed" in final_test_text,
            "saved initial/final test evidence is missing expected pass counts")
    checks["tests"] = {"status": "pass", "initial_evidence": initial_test_text.strip(),
                       "final_evidence": final_test_text.strip(), "final_pass_count": 90}

    direct = load(OUT/"direct_comparison.json")
    require(direct["schema"] == "direct_differential_y_vs_xy_v1", "wrong direct comparison schema")
    membership = {}
    with gzip.open(OUT/"direct_paired_membership.jsonl.gz", "rt") as stream:
        for line in stream:
            item = json.loads(line)
            key = (item["comparison"], item["cohort"])
            require(key not in membership, f"duplicate paired membership record {key}")
            membership[key] = item
    require(len(direct["comparisons"]) == 4, "expected four saved-response/family comparisons")
    require(len(membership) == 4*(len(COHORTS)+1), "unexpected direct membership record count")
    folds = config["folds"]
    pair_checks = 0
    for response in RESPONSES:
        for family in ("gaze", "capture"):
            label = f"{response}/{family}"
            compare = direct["comparisons"][label]
            selected_folds = [f for f in folds if f.startswith(family+"_")]
            schedule = independent_schedule(response, selected_folds)
            ref = []
            candidate = []
            with gzip.open(BASE/"phase83_audit_followup_v1"/"scorecard_frames.jsonl.gz", "rt") as stream:
                for line in stream:
                    item = json.loads(line)
                    parts = item["candidate"].split("/")
                    if parts[0] == "phase83" and parts[1] == response and parts[2] in selected_folds and parts[3] == "xy":
                        ref.append(item["frame"])
            for fold in selected_folds:
                candidate.extend(load(BASE/"phase83_audit_followup_v1"/"information"/response/fold/
                                      "x_y_difference"/"frames.json"))
            refmap = {frame_key(f): f for f in ref}
            cmap = {frame_key(f): f for f in candidate}
            require(len(refmap) == len(ref) == len(schedule) and set(refmap) == set(schedule),
                    f"reference does not match independent scheduled frame population {label}")
            require(len(cmap) == len(candidate) == len(schedule) and set(cmap) == set(schedule),
                    f"candidate does not match independent scheduled frame population {label}")
            for key in schedule:
                for table, frame in ((refmap, "reference"), (cmap, "candidate")):
                    slots = table[key]["slots"]
                    require(len(slots) == 3 and {int(s["held_point"]) for s in slots} == {0, 1, 2},
                            f"{label} {frame} does not have held points 0/1/2 at {key}")
                    require(all(frame_key({**s, "source_frame_index": s.get("source_frame_index", key[-1])}) == key
                                for s in slots), f"{label} {frame} slot/frame identity mismatch {key}")
            exposures = sorted({k[:3] for k in schedule})
            check_card(compare["full_reference"], [refmap[k] for k in schedule], exposures, label+" xy")
            check_card(compare["full_candidate"], [cmap[k] for k in schedule], exposures, label+" differential-y")
            for cohort in COHORTS:
                points, frames = recompute_cohort([refmap[k] for k in schedule],
                    [cmap[k] for k in schedule], schedule, cohort)
                saved_members = membership[(label, cohort)]
                expected_frame_ids = [list(k) for k, _, _ in frames]
                expected_point_ids = [list(k) for k, _, _ in points]
                require(saved_members["frame_ids"] == expected_frame_ids,
                        f"{label}/{cohort} saved frame membership differs from independent reconstruction")
                require(saved_members["point_ids"] == expected_point_ids,
                        f"{label}/{cohort} saved point membership differs from independent reconstruction")
                check_delta_report(compare["cohorts"][cohort], points, frames, exposures, label+"/"+cohort)
                pair_checks += 1
            trans = membership[(label, "all_transitions")]
            check_transition_strata(compare, trans["points"], trans["frames"], label)
    checks["direct_pairing"] = {"status": "pass", "comparisons": len(direct["comparisons"]),
                                 "cohort_checks": pair_checks,
                                 "transition_strata": ["transition", "fixation", "signed_gaze", "capture"],
                                 "population_source": "selected rows in frozen per-fold population.csv.gz files"}

    diagnosis = load(OUT/"training_diagnosis.json")
    training = check_training(ROOT, diagnosis)
    checks["training_diagnosis"] = {"status": "pass", **training,
        "outer_evaluation_groups_used": 0,
        "note": "all fold-held rows and blocked ridge predictions independently reconstructed from frozen training rows"}
    checks["shared_y_profile_identity"] = {"status": "pass", **scalar_profile_check(),
        "test_evidence": "tests/test_phase83_information_contracts.py"}
    checks["covariance_trial"] = check_covariance_trial(ROOT)

    # Compact, deterministic fold-equal summaries for handoff; no inferential intervals.
    skill = defaultdict(list)
    capture_means = defaultdict(list)
    for report in diagnosis["reports"]:
        for scheme in ("fixation", "capture", "horizontal_gaze"):
            for feature in ("state", "state_P1"):
                skill[(report["response"], scheme, feature)].append(
                    report["blocked_diagnostic_metrics"][scheme][feature]["common_all_y"]["skill_vs_train_mean"])
        for cap, values in report["capture_summary"].items():
            capture_means[(report["response"], cap, "raw")].append(values["common_y_raw"]["mean"])
            capture_means[(report["response"], cap, "after_state_P1")].append(values["common_y_after_state_P1"]["mean"])
    skill_summary = {"/".join(k): {"fold_count": len(v), "equal_fold_mean": float(np.mean(v)),
                                    "min": float(np.min(v)), "max": float(np.max(v))}
                     for k, v in sorted(skill.items())}
    capture_summary = {"/".join(k): {"fold_count": len(v), "equal_fold_mean_px": float(np.mean(v)),
                                     "min_px": float(np.min(v)), "max_px": float(np.max(v))}
                       for k, v in sorted(capture_means.items())}
    checks["training_summary"] = {"skill": skill_summary, "capture_common_y_means": capture_summary,
        "interpretation_scope": "descriptive; fold-equal means/ranges, no uncertainty claim"}
    result = {"schema": "latest_results_followup_verification_v1", "status": "pass", "checks": checks}
    (OUT/"verification.json").write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"status": "pass", "source_hashes": checks["source_hashes"]["count"],
        "snapshots": checks["implementation_snapshots"]["count"], "cohort_checks": pair_checks,
        "training_reports": training["report_count"], "blocked_metrics": training["blocked_metric_count"],
        "scalar_profile_abs_difference": checks["shared_y_profile_identity"]["absolute_difference"],
        "covariance_trial_objectives": checks["covariance_trial"]["saved_branch_objectives_recomputed"],
        "covariance_trial_certificates": len(checks["covariance_trial"]["selected_interior_certificates_recomputed"]),
        "covariance_trial_cohorts": checks["covariance_trial"]["exact_paired_cohort_checks"]}, indent=2))


if __name__ == "__main__":
    main()
