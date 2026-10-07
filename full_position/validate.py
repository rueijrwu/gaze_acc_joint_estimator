"""Grouped experiments and diagnostic summaries (no physiological claims)."""
from __future__ import annotations
import csv
from contextlib import ExitStack
import gzip
import json
import platform
import time
from pathlib import Path
import numpy as np
import scipy
from .data import load_reviewed, core_rows, fixed_sample, training_data, noise_blocks
from .geometry import context, summaries, observed_map
from .model import PositionModel, SummaryModel
from .noise import coordinate_covariance, reference_covariance
from .calibrate import fit
from .invert import invert, predict_holdout, score_holdout
from .schema import artifact, write_json, source_hashes, json_default


def pilot_fit(data, anchors):
    """Training-only common 27-coefficient pilot for BOTH response capacities."""
    pilot = PositionModel(27)
    x = anchors[data["groups"]]
    counts = np.bincount(data["groups"])
    weight = 1/np.sqrt(len(anchors)*counts[data["groups"]])
    B = (pilot.design(x, data["r"])*weight[:, None, None]).reshape(-1, 27)
    b = (data["v"]*weight[:, None]).ravel()
    scale = np.maximum(np.linalg.norm(B, axis=0), 1e-12)
    Bn = B/scale
    ridge = np.ones(27)*1e-3
    ridge[pilot.intercepts] = 0
    pilot.beta = np.linalg.lstsq(np.vstack((Bn, np.diag(ridge))),
                                 np.r_[b, np.zeros(27)], rcond=1e-10)[0]/scale
    return pilot


def fold_specs(groups, requested):
    folds = []
    if "gaze" in requested:
        for target in [-10., -5., 0., 5., 10.]:
            folds.append((f"gaze_{target:+g}", [i for i, g in enumerate(groups) if g["target_theta_deg"] == target]))
    if "capture" in requested:
        for cap in range(1, 5):
            folds.append((f"capture_{cap}", [i for i, g in enumerate(groups) if g["capture"] == f"capture_{cap}_detections.pkl"]))
    if "full" in requested:
        folds.append(("full_development", []))
    return folds


def stats(values):
    a = np.asarray(values, float)
    a = a[np.isfinite(a)]
    if not len(a):
        return dict(n=0)
    return dict(n=len(a), mean=float(a.mean()), median=float(np.median(a)),
                p90=float(np.percentile(a, 90)), p95=float(np.percentile(a, 95)),
                rms=float(np.sqrt(np.mean(a*a))))


def summarize(rows, holdouts):
    valid = [r for r in rows if r.get("estimated")]
    scored = [r for r in holdouts if r.get("score_available")]
    return dict(evaluation_rows=len(rows), baseline_valid_rows=sum(r["baseline_valid"] for r in rows),
        estimated_rows=len(valid), inverse_failure_rows=sum(r["baseline_valid"] and not r.get("estimated") for r in rows),
        ambiguous_rows=sum(r.get("ambiguous", False) for r in valid),
        bound_rows=sum(r.get("at_bound", False) for r in valid),
        weak_rank_rows=sum(r.get("rank", 0)<2 for r in valid),
        gaze_mean_anchor_discrepancy_deg=stats([r["theta"]-r["nominal_theta"] for r in valid]),
        accommodation_mean_demand_discrepancy_D=stats([r["A"]-r["demand"] for r in valid]),
        geometry_cost=stats([r["cost"] for r in valid]),
        holdout_tests=len(holdouts), testable_holdouts=len(scored),
        inconclusive_holdouts=len(holdouts)-len(scored),
        heldout_error_px=stats([np.linalg.norm(r["error_px"]) for r in scored]),
        heldout_axis_rms_px=[stats([r["error_px"][axis] for r in scored]).get("rms") for axis in range(2)],
        heldout_mahalanobis=stats([r.get("mahalanobis", np.nan) for r in scored]),
        holdout_by_point={str(j): stats([np.linalg.norm(r["error_px"]) for r in scored if r["held_point"]==j]) for j in range(3)})


def evaluate(model, captures, groups, test_ids, pilot, reference, sigma, output, count=8, progress=print):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows, holdouts = [], []
    # Record full fixed evaluation population before looking at model residuals.
    with ExitStack() as stack:
        # Population membership is model-independent: keep one compressed copy per fold.
        population_path = output.parent/"population.csv.gz"
        writer = None
        if not population_path.exists():
            population = stack.enter_context(gzip.open(population_path, "wt", newline=""))
            writer = csv.writer(population)
            writer.writerow(["fixation", "capture", "row", "frame", "timestamp_ms", "selected_for_evaluation",
                             "p1_valid_geometry", "p4_1_valid", "p4_2_valid", "p4_3_valid", "baseline_valid"])
        hf = stack.enter_context((output/"holdouts.jsonl").open("w")) if model.channels == 6 else None
        for gi in test_ids:
            g, cap = groups[gi], captures[groups[gi]["capture"]]
            core = core_rows(g)
            selected = fixed_sample(core, count)
            selected_set = set(selected.tolist())
            if writer is not None:
                for i in core:
                    writer.writerow([gi, cap.name, i, cap.frame[i], cap.timestamp[i], i in selected_set,
                        bool(cap.ctx.valid[i]), *cap.point_valid[i].tolist(), bool(cap.baseline_valid[i])])
            progress(f"  evaluating {model.name} fixation={gi} selected={len(selected)}", flush=True)
            good = selected[cap.ctx.valid[selected]]
            covs = reference_covariance(cap.p[good], pilot, reference, sigma, model.channels == 2) if len(good) else []
            cov_by_row = dict(zip(good, covs))
            for i in selected:
                row = dict(fixation=gi, capture=cap.name, row=int(i), frame=int(cap.frame[i]),
                           timestamp_ms=float(cap.timestamp[i]), nominal_theta=g["target_theta_deg"],
                           demand=g["demand_diopters_label"], baseline_valid=bool(cap.baseline_valid[i]),
                           point_valid=cap.point_valid[i].tolist(), estimated=False,
                           reason="invalid_input", estimate_kind="none")
                ctx = context(cap.p[i])
                if not ctx.valid or (model.channels == 2 and not cap.baseline_valid[i]):
                    rows.append(row)
                    continue
                if model.channels == 6:
                    ix = np.repeat(cap.point_valid[i], 2).nonzero()[0]
                    if len(ix) < 4:
                        row["reason"] = "insufficient_P4_support"
                        rows.append(row)
                        continue
                    y = cap.v[i].ravel()[ix]
                    from .noise import marginal
                    result = invert(model, ctx.r, y, marginal(cov_by_row[i], ix), ix)
                else:
                    y = summaries(ctx, cap.q[i])[[0, 2]]
                    result = invert(model, ctx.r, y, cov_by_row[i])
                row["reason"] = result["reason"]
                if result["available"]:
                    row.update(estimated=True, theta=result["state"][0], A=result["state"][1],
                               estimate_kind="all_three" if cap.baseline_valid[i] else "partial_two_P4")
                    for key in ["cost", "ambiguous", "at_bound", "rank", "stationarity_encoded", "singular_values_physical",
                                "numerical_ties", "failed_starts", "branches"]:
                        row[key] = result[key]
                    x = np.asarray(result["state"])
                    row["gaze_outside_training_anchors"] = bool(x[0] < model.training_range[0][0] or x[0] > model.training_range[1][0])
                    row["A_outside_training_anchors"] = bool(x[1] < model.training_range[0][1] or x[1] > model.training_range[1][1])
                    if model.channels == 6:
                        D, T = model.components(x)
                        pred = model.predict(x, ctx.r).reshape(3, 2)
                        qhat = ctx.c+ctx.ell*pred
                        row.update(predicted_q=qhat.tolist(), predicted_d=D.tolist(),
                                   predicted_rho=float(abs(np.linalg.det(T))), predicted_T=T.tolist(),
                                   signed_predicted_det=float(np.linalg.det(T)))
                        if cap.baseline_valid[i]:
                            row["observed_summaries"] = summaries(ctx, cap.q[i]).tolist()
                            row["observed_T"] = observed_map(ctx, cap.q[i]).tolist()
                            row["all_three_error_px"] = (cap.q[i]-qhat).tolist()
                            row["observed_signed_det"] = float(np.linalg.det(np.asarray(row["observed_T"])))
                    else:
                        row["predicted_d_rho"] = model.predict(x, ctx.r).tolist()
                rows.append(row)
                if model.channels != 6:
                    continue
                for j in range(3):
                    ix = [k for k in range(6) if k//2 != j]
                    if not cap.point_valid[i][np.arange(3)!=j].all():
                        prediction = dict(available=False, held_point=j, reason="insufficient_retained_P4", branches=[])
                    else:
                        prediction = predict_holdout(model, ctx, j, cap.v[i].ravel()[ix], cov_by_row[i])
                    score = score_holdout(prediction, cap.q[i, j] if cap.point_valid[i, j] else np.full(2, np.nan), ctx.ell)
                    score.update(fixation=gi, capture=cap.name, row=int(i), nominal_theta=g["target_theta_deg"],
                                 demand=g["demand_diopters_label"])
                    if score.get("available") and result.get("available"):
                        score["state_difference_from_all"] = (np.asarray(score["state"])-result["state"]).tolist()
                    holdouts.append(score)
                    hf.write(json.dumps(score, default=json_default, allow_nan=False)+"\n")
    write_json(output/"frames.json", rows)
    summary = summarize(rows, holdouts)
    summary["by_fixation"] = {str(gi): summarize([r for r in rows if r["fixation"]==gi],
                                               [r for r in holdouts if r["fixation"]==gi]) for gi in test_ids}
    summary["evaluation_population_policy"] = "central80_original_rows_evenly_sampled_before_validity; no_residual_rejection"
    write_json(output/"summary.json", summary)
    return summary


def run(root, output, train_count=48, eval_count=8, folds=("gaze", "capture"),
        starts=2, max_nfev=300, prior=.001, seed=17, only_fold=None):
    root, output = Path(root), Path(output)
    if train_count < 0 or eval_count < 0 or starts < 1 or max_nfev < 1 or not np.isfinite(prior) or prior < 0:
        raise ValueError("Counts must be nonnegative, starts/budget positive, and prior finite/nonnegative")
    output.mkdir(parents=True, exist_ok=False)
    begun = time.monotonic()
    captures, groups, interval_hash = load_reviewed(root)
    specs = fold_specs(groups, folds)
    if only_fold:
        specs = [s for s in specs if s[0] == only_fold]
        if not specs:
            raise ValueError("Unknown requested fold")
    config = dict(train_per_fixation=train_count, eval_per_fixation=eval_count, folds=list(folds),
                  starts=starts, max_nfev=max_nfev, prior=prior, seed=seed,
                  purpose="exploratory_paired_capacity_test_not_nested_model_selection",
                  candidates=["conditional27", "conditional37", "two_channel13"],
                  temporal_strength=0., jump_filter=False, detector_rerun=False,
                  runtime=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__))
    write_json(output/"config.json", config)
    source = source_hashes(root)
    all_summaries = {}
    for fold_name, test_ids in specs:
        print(f"FOLD {fold_name}", flush=True)
        train_ids = [i for i in range(len(groups)) if i not in test_ids]
        data, anchors = training_data(captures, groups, train_ids, train_count)
        sigma, noise_meta = coordinate_covariance(noise_blocks(captures, groups, train_ids))
        pilot = pilot_fit(data, anchors)
        reference = np.median(anchors, axis=0)
        provenance = dict(interval_sha256=interval_hash, source_sha256=source,
            capture_sha256={n: c.sha256 for n, c in captures.items()},
            training_group_ids=train_ids, evaluation_group_ids=test_ids,
            training_groups=[groups[i] for i in train_ids], sampled_training_rows=data["rows"].tolist(),
            sampled_training_group=data["original_group"].tolist(),
            sampling_policy="evenly_spaced_original_core_rows_before_validity", noise=noise_meta,
            correspondence={n: c.metadata["pair_index"] for n, c in captures.items()},
            detector_configuration={n: c.metadata.get("config") for n, c in captures.items()},
            detector_location_definition="stored selected native coordinates; exact fitted-center implementation unavailable",
            detector_implementation_version=None, source_grid_geometry=None,
            seed=seed, runtime=config["runtime"], anchor_range=[anchors.min(0).tolist(), anchors.max(0).tolist()],
            conditional_context_support=dict(r_min=data["r"].min(0).tolist(), r_max=data["r"].max(0).tolist(),
                signed_P1_area_branches=np.unique(np.sign(context(data["p"]).signed_area)).tolist()))
        common_cov = reference_covariance(data["p"], pilot, reference, sigma)
        control_cov = reference_covariance(data["p"], pilot, reference, sigma, True)
        reduced_states = None
        for model in [PositionModel(27), PositionModel(37), SummaryModel()]:
            print(f"  calibrating {model.name} observations={len(data['p'])}", flush=True)
            cov = common_cov if model.channels == 6 else control_cov
            y = data["v"] if model.channels == 6 else data["y2"]
            model, states, diag = fit(model, y, data["r"], cov, data["groups"], anchors,
                prior, starts, max_nfev, seed, progress=lambda r: print("    "+json.dumps(r), flush=True),
                additional_initial_states=[reduced_states] if model.name == "conditional37" and reduced_states is not None else None)
            if model.name == "conditional27" and diag["converged"]:
                reduced_states = states.copy()
            model.training_range = [anchors.min(0), anchors.max(0)]
            dest = output/fold_name/model.name
            model_path = dest/("model.json" if diag["converged"] else "failed_checkpoint.json")
            write_json(model_path, artifact(model, pilot, reference, sigma, provenance, diag))
            write_json(dest/"training_states.json", dict(states=states,
                nominal_anchors=anchors, groups=data["groups"], rows=data["rows"]))
            if model.channels == 6:
                probe = np.array([[t, a] for t in np.linspace(-20, 20, 41) for a in np.linspace(0, 6, 13)])
                _, T = model.components(probe)
                determinants = np.linalg.det(T)
                supported = ((probe >= anchors.min(0)) & (probe <= anchors.max(0))).all(-1)
                write_json(dest/"determinant_audit.json", dict(states=probe, determinants=determinants,
                    nominal_support=supported, min_absolute_det_supported=float(np.min(np.abs(determinants[supported]))),
                    sign_branches_supported=np.unique(np.sign(determinants[supported])).tolist(),
                    sign_branches_computational=np.unique(np.sign(determinants)).tolist()))
            if not diag["converged"]:
                all_summaries[f"{fold_name}/{model.name}"] = dict(calibration_converged=False, calibration=diag)
                write_json(output/"summary.json", all_summaries)
                continue
            if test_ids:
                summary = evaluate(model, captures, groups, test_ids, pilot, reference, sigma, dest, eval_count)
            else:
                summary = dict(development_training_only=True)
            summary["calibration_converged"] = True
            summary["calibration"] = diag
            all_summaries[f"{fold_name}/{model.name}"] = summary
            write_json(output/"summary.json", all_summaries)
    write_json(output/"completion.json", dict(seconds=time.monotonic()-begun, complete=True))
    return all_summaries
