"""Independent, read-only verifier for accommodation_response_v1 outputs.

Run only after all 36 fit tasks and the summary files are complete. This module
does not import the study, model, inverse, schema, or scorecard implementation.
It recomputes persisted identities, training arrays/objectives, trajectory
identities, cross-check aggregates, and held-coordinate residuals from the
hash-pinned capture 1-4 payloads.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


CANDIDATES = {"ar27_log": 0., "ar27_sqrt": .5, "ar27_linear": 1., "ar27_quadratic": 2.}
EXPECTED_TASKS = 36
COEFFICIENT_ORDER = "Dx[1,a,t,tphi,t2,t2phi,t3]; Dy,T11,T12,T21,T22[1,t,phi,tphi]"
BOUNDS = np.array([[-20., 0.], [20., 6.]])
STATE_SCALE = np.array([10., 4.])


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def close(actual, expected, *, atol=1e-8, rtol=1e-7, message="values disagree"):
    a, b = np.asarray(actual, float), np.asarray(expected, float)
    require(a.shape == b.shape and np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=True),
            f"{message}: shapes {a.shape} and {b.shape}")


def context(p):
    p = np.asarray(p, float)
    require(p.shape == (3, 2), "P1 context shape is not 3x2")
    area = ((p[1, 0]-p[0, 0])*(p[2, 1]-p[0, 1])-
            (p[1, 1]-p[0, 1])*(p[2, 0]-p[0, 0]))/2.
    edge_sq = max(float(np.sum((p[j]-p[k])**2)) for j, k in ((0, 1), (0, 2), (1, 2)))
    valid = bool(np.isfinite(p).all() and abs(area) > 32*np.finfo(float).eps*max(edge_sq, 1.))
    center = p.mean(axis=0)
    ell = float(np.sqrt(abs(area))) if valid else np.nan
    r = (p-center)/ell if valid else np.full((3, 2), np.nan)
    return center, ell, r, valid


def phi(A, exponent):
    L = np.log1p(A)
    return L if exponent == 0. else np.expm1(exponent*L)/exponent


def predict(beta, exponent, state, r):
    theta, A = np.asarray(state, float)
    t, z = theta/10., phi(A, exponent)
    d = np.array([1., A, t, t*z, t*t, t*t*z, t**3])
    s = np.array([1., t, z, t*z])
    D = np.array([d@beta[:7], s@beta[7:11]])
    T = (s@beta[11:].reshape(4, 4).T).reshape(2, 2)
    return (D[None, :]+r@T.T).reshape(6)


def design(beta_unused, exponent, state, r):
    """Explicit 6x27 design, kept independent of PowerResponseModel.design."""
    theta, A = np.asarray(state, float)
    t, z = theta/10., phi(A, exponent)
    d = np.array([1., A, t, t*z, t*t, t*t*z, t**3])
    s = np.array([1., t, z, t*z])
    out = np.zeros((6, 27))
    for point in range(3):
        out[2*point, :7] = d
        out[2*point+1, 7:11] = s
        out[2*point, 11:15] = s*r[point, 0]
        out[2*point, 15:19] = s*r[point, 1]
        out[2*point+1, 19:23] = s*r[point, 0]
        out[2*point+1, 23:27] = s*r[point, 1]
    return out


def native_context_residual(raw, beta, exponent, state):
    raw = np.asarray(raw, float)
    p = raw[:6].reshape(3, 2)
    q = raw[6:].reshape(3, 2)
    c, ell, r, valid = context(p)
    if not valid:
        raise ValueError("Finite-difference covariance encountered degenerate P1")
    return ((q-c)/ell-predict(beta, exponent, state, r).reshape(3, 2)).ravel()


def reference_covariance(p, pilot_beta, reference_state, sigma):
    """Finite-difference propagation at the fold-local pilot prediction."""
    p = np.asarray(p, float)
    c, ell, r, valid = context(p)
    require(valid, "reference covariance requires a valid P1 triangle")
    q_reference = c+ell*predict(pilot_beta, 0., reference_state, r).reshape(3, 2)
    raw = np.r_[p.ravel(), q_reference.ravel()]
    jac = np.empty((6, 12))
    for k in range(12):
        step = 1e-4*max(1., abs(raw[k]))
        plus, minus = raw.copy(), raw.copy()
        plus[k] += step
        minus[k] -= step
        jac[:, k] = (native_context_residual(plus, pilot_beta, 0., reference_state)-
                     native_context_residual(minus, pilot_beta, 0., reference_state))/(2*step)
    result = jac@np.asarray(sigma, float)@jac.T
    return (result+result.T)/2


def load_pinned_sources(root, output, config):
    split = read_json(output/"splits"/config["folds"][0]/"training.json")
    provenance = split["provenance"]
    intervals = root/"data/fixations/fixation_intervals.json"
    require(sha256(intervals) == provenance["interval_sha256"], "reviewed interval manifest hash mismatch")
    interval_obj = read_json(intervals)
    groups = interval_obj["fixations"]
    interval_source_hashes = {item["capture"]: item["sha256"] for item in interval_obj["sources"]}
    source_hashes = provenance["capture_sha256"]
    captures = {}
    seen_numbers = set()
    # Hash-check before unpickling. Only reviewed captures 1-4 are eligible.
    for name, expected in source_hashes.items():
        match = re.search(r"capture_(\d+)_detections\.pkl$", name)
        require(match is not None, f"unrecognized pinned capture filename: {name}")
        number = int(match.group(1))
        require(number in (1, 2, 3, 4), f"refusing capture outside 1-4: {name}")
        seen_numbers.add(number)
        require(interval_source_hashes.get(name) == expected,
                f"pinned capture hash disagrees with reviewed interval metadata: {name}")
        path = root/"data/detections"/name
        require(sha256(path) == expected, f"pinned capture hash mismatch: {name}")
        with path.open("rb") as stream:
            payload = pickle.load(stream)
        arrays, metadata = payload["arrays"], payload["meta"]
        p = np.asarray(arrays["p1_xy"], float)
        order = np.asarray(metadata["pair_index"], int)
        require(order.shape == (3,) and sorted(order.tolist()) == [0, 1, 2],
                f"invalid P1/P4 correspondence for {name}")
        q = np.asarray(arrays["p4_xy"], float)[:, order, :]
        found = np.asarray(arrays["p4_found"], bool)[:, order]
        frames = np.asarray(arrays["frame_index"])
        require(p.shape[0] == q.shape[0] == found.shape[0] == frames.shape[0],
                f"capture array lengths disagree for {name}")
        captures[name] = dict(p=p, q=q, found=found, frames=frames,
                              metadata=metadata, sha256=expected)
    require(seen_numbers == {1, 2, 3, 4}, "pinned study sources are not exactly captures 1-4")
    require(not any("capture_5" in n or "capture_6" in n for n in captures),
            "captures 5/6 entered the verifier")
    return captures, groups, provenance["interval_sha256"]


def raw_row(captures, groups, group_id, row):
    group = groups[int(group_id)]
    cap = captures[group["capture"]]
    p, q, found = cap["p"][int(row)], cap["q"][int(row)], cap["found"][int(row)]
    c, ell, r, p1_valid = context(p)
    point_valid = found & np.isfinite(q).all(axis=-1) & p1_valid
    v = (q-c)/ell if p1_valid else np.full((3, 2), np.nan)
    return dict(group=group, capture=cap, p=p, q=q, found=found, c=c, ell=ell,
                r=r, p1_valid=p1_valid, point_valid=point_valid, v=v)


def verify_source_snapshots(root, output, config):
    implementation = config["implementation_hashes"]
    for rel, expected in implementation.items():
        path = output/("implementation_snapshot" if rel.startswith("full_position/") else "design_snapshot")/Path(rel).name
        require(sha256(path) == expected, f"copied source snapshot hash mismatch: {rel}")
    for rel, expected in config["historical_source_hashes"].items():
        path = root/rel
        require(sha256(path) == expected, f"historical source changed: {rel}")


def core_rows(group):
    start, end = int(group["start_row"]), int(group["end_row_exclusive"])
    cut = int(np.floor(.1*(end-start)))
    return np.arange(start+cut, end-cut)


def fixed_sample(rows, count):
    if count <= 0 or count >= len(rows):
        return rows
    return rows[np.linspace(0, len(rows)-1, count).round().astype(int)]


def verify_training_inputs(root, output, config, split_info, groups, captures):
    fold = split_info["fold"]
    provenance = read_json(output/"splits"/fold/"training.json")["provenance"]
    train_ids = list(map(int, split_info["training_group_ids"]))
    eval_ids = list(map(int, split_info["evaluation_group_ids"]))
    require(not set(train_ids)&set(eval_ids), f"train/evaluation groups overlap in {fold}")
    require(train_ids == provenance["training_group_ids"] and eval_ids == provenance["evaluation_group_ids"],
            f"training provenance group IDs disagree in {fold}")
    expected_rows, expected_groups, expected_original, expected_p, expected_q, expected_r, expected_v = [], [], [], [], [], [], []
    anchors = []
    for local, gi in enumerate(train_ids):
        meta = groups[gi]
        anchors.append([meta["target_theta_deg"], meta["demand_diopters_label"]])
        cap = captures[meta["capture"]]
        candidates = fixed_sample(core_rows(meta), int(config["train_per_fixation"]))
        for row in candidates:
            item = raw_row(captures, groups, gi, int(row))
            if not bool(item["point_valid"].all()):
                continue
            expected_rows.append(int(row)); expected_groups.append(local); expected_original.append(gi)
            expected_p.append(item["p"]); expected_q.append(item["q"])
            expected_r.append(item["r"]); expected_v.append(item["v"].ravel())
    path = output/"splits"/fold/"training_inputs.npz"
    require(sha256(path) == split_info["training_input_sha256"], f"training input archive hash mismatch: {fold}")
    with np.load(path, allow_pickle=False) as saved:
        close(saved["rows"], expected_rows, atol=0, rtol=0, message=f"sampled row identity mismatch {fold}")
        close(saved["groups"], expected_groups, atol=0, rtol=0, message=f"local group order mismatch {fold}")
        close(saved["original_group"], expected_original, atol=0, rtol=0, message=f"original group identity mismatch {fold}")
        close(saved["p"], expected_p, atol=1e-12, rtol=0, message=f"training P1 source mismatch {fold}")
        close(saved["q"], expected_q, atol=1e-12, rtol=0, message=f"training P4 correspondence mismatch {fold}")
        close(saved["r"], expected_r, atol=2e-12, rtol=1e-12, message=f"training context mismatch {fold}")
        close(saved["v"], expected_v, atol=2e-12, rtol=1e-12, message=f"training normalized observation mismatch {fold}")
        close(saved["anchors"], anchors, atol=0, rtol=0, message=f"training anchors mismatch {fold}")
        result = {name: saved[name].copy() for name in saved.files}
    require(len(set(eval_ids)) == len(eval_ids), f"duplicate evaluation groups in {fold}")
    return result


def verify_population(output, config, fold, split_info, groups, captures):
    split_dir = output/"splits"/fold
    rows = read_json(split_dir/"population.json")
    manifest = read_json(split_dir/"manifest.json")
    require(manifest["created_before_candidate_outputs"] is True, f"schedule provenance missing: {fold}")
    selected = [r for r in rows if r["selected_for_evaluation"] is True]
    # Population rows call the source frame `frame`; the frozen schedule calls it
    # `source_frame_index`. Normalize explicitly, then check both against raw data below.
    keys = [(fold, r["capture"], int(r["fixation"]), int(r["row"]), int(r["frame"])) for r in selected]
    expected = [tuple((x["fold"], x["capture"], int(x["fixation"]), int(x["row"]), int(x["source_frame_index"])))
                for x in manifest["frames"]]
    require(sorted(keys) == sorted(expected), f"selected population differs from frozen schedule: {fold}")
    eval_ids = set(map(int, split_info["evaluation_group_ids"]))
    require({int(r["fixation"]) for r in selected} == eval_ids,
            f"scheduled frame groups differ from held groups: {fold}")
    require(len(selected) == len(eval_ids)*int(config["eval_per_fixation"]),
            f"scheduled evaluation row count mismatch: {fold}")
    expected_all, expected_selected = set(), set()
    for gi in eval_ids:
        core = core_rows(groups[gi])
        require(len(core) == len(set(map(int, core))), f"duplicate source rows in evaluation group {gi}")
        expected_all.update((gi, int(i)) for i in core)
        expected_selected.update((gi, int(i)) for i in fixed_sample(core, int(config["eval_per_fixation"])))
    got_all = {(int(row["fixation"]), int(row["row"])) for row in rows}
    got_selected = {(int(row["fixation"]), int(row["row"])) for row in selected}
    require(got_all == expected_all, f"candidate-independent population omits or adds raw rows: {fold}")
    require(got_selected == expected_selected, f"fixed sample differs from predeclared row sampling: {fold}")
    for row in rows:
        item = raw_row(captures, groups, row["fixation"], row["row"])
        require(int(item["capture"]["frames"][int(row["row"])]) == int(row["frame"]),
                f"source frame identity mismatch: {fold}/{row['row']}")
        require(bool(item["p1_valid"]) == bool(row["p1_valid_geometry"]),
                f"P1 validity differs from raw input: {fold}/{row['row']}")
        for j in range(3):
            require(bool(item["point_valid"][j]) == bool(row[f"p4_{j+1}_valid"]),
                    f"P4 validity differs from raw input: {fold}/{row['row']}/{j}")
    return rows, manifest


def trajectory_metrics(states, local_groups, anchors):
    result = []
    for local in range(len(anchors)):
        values = np.asarray(states)[np.asarray(local_groups) == local]
        require(len(values) > 0, f"empty saved training trajectory for local group {local}")
        mean = values.mean(axis=0)
        offset = mean-np.asarray(anchors)[local]
        variance = np.mean((values-mean)**2, axis=0)
        mse = np.mean((values-np.asarray(anchors)[local])**2, axis=0)
        result.append(dict(rows=len(values), mean=mean, offset=offset,
                           std=np.sqrt(variance), rms=np.sqrt(mse), identity=mse-offset**2-variance))
    return result


def verify_trajectory(outdir, train, training_meta):
    states_obj = read_json(outdir/"training_states.json")
    traj = read_json(outdir/"trajectory.json")
    close(states_obj["rows"], train["rows"], atol=0, rtol=0, message="trajectory row order mismatch")
    close(states_obj["groups"], train["groups"], atol=0, rtol=0, message="trajectory local group mismatch")
    close(states_obj["original_group"], train["original_group"], atol=0, rtol=0,
          message="trajectory original group mismatch")
    require(traj["temporal_strength"] == 0 and traj["nominal_errors_are_descriptive"] is True,
            "trajectory metadata implies a temporal penalty or target gate")
    anchors = train["anchors"]
    computed = trajectory_metrics(states_obj["states"], train["groups"], anchors)
    records = sorted(traj["groups"], key=lambda r: int(r["group_id"]))
    for local, got in enumerate(computed):
        row = records[local]
        require(int(row["rows"]) == got["rows"], "trajectory row count mismatch")
        close(row["mean"], got["mean"], atol=2e-9, rtol=2e-9, message="trajectory mean mismatch")
        close(row["nominal_mean_offset"], got["offset"], atol=2e-9, rtol=2e-9, message="anchor mean offset mismatch")
        close(row["temporal_std"], got["std"], atol=2e-9, rtol=2e-9, message="temporal spread mismatch")
        close(row["nominal_RMS"], got["rms"], atol=2e-9, rtol=2e-9, message="nominal RMS mismatch")
        close(row["rms_identity_error"], got["identity"], atol=2e-9, rtol=2e-9,
              message="trajectory RMS decomposition mismatch")
        require(float(np.max(np.abs(got["identity"]))) < 2e-9,
                "RMS² does not equal squared mean offset plus temporal variance")
    return states_obj


def per_frame_metrics(slots):
    valid = [s for s in slots if s.get("scored") is True and
             np.asarray(s.get("error_px"), float).shape == (2,) and
             np.isfinite(np.asarray(s["error_px"], float)).all()]
    if len(valid) != 3 or {s["held_point"] for s in valid} != {0, 1, 2}:
        return None
    ordered = sorted(valid, key=lambda s: int(s["held_point"]))
    errors = np.asarray([s["error_px"] for s in ordered], float)
    states = np.asarray([s["state"] for s in ordered], float)
    differences = np.asarray([states[j]-states[k] for j, k in ((0, 1), (0, 2), (1, 2))])
    return dict(E=float(np.sqrt(np.mean(np.sum(errors**2, axis=1)))),
        Gtheta=float(np.sqrt(np.mean(differences[:, 0]**2))),
        GA=float(np.sqrt(np.mean(differences[:, 1]**2))),
        worst=float(np.max(np.linalg.norm(errors, axis=1))), errors=errors, states=states)


def exposure_rms(records, values, scheduled_exposures):
    by = defaultdict(list)
    for record, value in zip(records, values):
        if np.isfinite(value):
            by[(record["fold"], record["capture"], int(record["fixation"]))].append(float(value)**2)
    contributing = [float(np.mean(by[key])) for key in scheduled_exposures if key in by]
    return float(np.sqrt(np.mean(contributing))) if contributing else None


def scalar_stats(values):
    a = np.asarray([v for v in values if np.isfinite(v)], float)
    if not len(a):
        return dict(n=0, mean=None, median=None, rms=None, p90=None, p95=None)
    return dict(n=int(len(a)), mean=float(a.mean()), median=float(np.median(a)),
        rms=float(np.sqrt(np.mean(a*a))), p90=float(np.percentile(a, 90)), p95=float(np.percentile(a, 95)))


def supported_for_pair(slot, cohort):
    if cohort == "full_common":
        return True
    if cohort == "joint_interior":
        return slot.get("interior") is True
    support = slot.get("support", {})
    state = support.get("theta_empirical") is True and support.get("A_empirical") is True
    p1 = (support.get("P1_context_valid") is True and
          support.get("context_outside_training_extrema") is False and
          support.get("p1_parity_seen_in_training") is True)
    if cohort == "joint_empirical_state":
        return state
    if cohort == "joint_P1":
        return p1
    if cohort == "joint_state_and_P1":
        return state and p1
    raise AssertionError(f"unrecognized paired cohort {cohort}")


def eq_mean(records, deltas, expected_exposures):
    by = defaultdict(list)
    for row, delta in zip(records, deltas):
        if np.isfinite(delta):
            key = (row["fold"], row["capture"], int(row["fixation"]))
            by[key].append(float(delta))
    contributors = [float(np.mean(by[key])) for key in expected_exposures if key in by]
    return float(np.mean(contributors)) if contributors else None


def verify_paired_outputs(output, config, all_populations, family_rows, summary):
    memberships = jsonl_gzip(output/"paired_memberships.jsonl.gz")
    by_membership = {}
    for row in memberships:
        key = (row["family"], row["candidate"], row["cohort"])
        require(key not in by_membership, f"duplicate paired-membership row: {key}")
        by_membership[key] = row
    expected_lines = 2*3*5
    require(len(by_membership) == expected_lines, "paired membership archive does not contain 30 cohort records")
    cohorts = ("full_common", "joint_interior", "joint_empirical_state", "joint_P1", "joint_state_and_P1")
    pairs_checked = 0
    for family, candidates in family_rows.items():
        ref = candidates["ar27_log"]
        refmap = {(f["fold"], f["capture"], int(f["fixation"]), int(f["row"]),
                   int(f["source_frame_index"])): f for f in ref}
        schedules = []
        for fold in config["folds"]:
            if fold.startswith(family+"_"):
                schedules.extend(all_populations[fold][1]["frames"])
        expected_exposures = {(f["fold"], f["capture"], int(f["fixation"])) for f in schedules}
        scheduled_ids = [(f["fold"], f["capture"], int(f["fixation"]), int(f["row"]),
                          int(f["source_frame_index"])) for f in schedules]
        for candidate in CANDIDATES:
            if candidate == "ar27_log":
                continue
            cand = candidates[candidate]
            candmap = {(f["fold"], f["capture"], int(f["fixation"]), int(f["row"]),
                        int(f["source_frame_index"])): f for f in cand}
            require(set(refmap) == set(scheduled_ids) == set(candmap),
                    f"paired population identities differ: {family}/{candidate}")
            for cohort in cohorts:
                shared_frames, shared_points = [], []
                for identity in sorted(scheduled_ids):
                    a, b = refmap[identity], candmap[identity]
                    a_slots = {int(s["held_point"]): s for s in a["slots"]}
                    b_slots = {int(s["held_point"]): s for s in b["slots"]}
                    keep = [j for j in range(3) if a_slots[j].get("scored") is True and
                            b_slots[j].get("scored") is True and
                            supported_for_pair(a_slots[j], cohort) and supported_for_pair(b_slots[j], cohort)]
                    shared_points.extend(identity+(j,) for j in keep)
                    if len(keep) == 3 and a.get("complete_triple") and b.get("complete_triple"):
                        shared_frames.append(identity)
                saved = by_membership[(family, candidate, cohort)]
                saved_frames = [tuple(x) for x in saved["frame_ids"]]
                saved_points = [tuple(x) for x in saved["point_ids"]]
                require(sorted(saved_frames) == sorted(shared_frames),
                        f"paired frame membership mismatch: {family}/{candidate}/{cohort}")
                require(sorted(saved_points) == sorted(shared_points),
                        f"paired point membership mismatch: {family}/{candidate}/{cohort}")
                report = summary["families"][family]["comparisons"][candidate][cohort]
                require(report["shared_complete_frames"] == len(shared_frames) and
                        report["shared_scored_points"] == len(shared_points),
                        f"paired membership count mismatch: {family}/{candidate}/{cohort}")
                rows_a = [refmap[k] for k in shared_frames]
                rows_b = [candmap[k] for k in shared_frames]
                pairs = list(zip(rows_a, rows_b))
                for field in ("E_px", "G_theta_deg", "G_A_D", "worst_point_px"):
                    delta = eq_mean(rows_a,
                        [float(b[field])**2-float(a[field])**2 for a,b in pairs], expected_exposures)
                    saved_delta = report["frame_changes"][field]["squared_change"]["mean"]
                    close(saved_delta, delta, atol=1e-8, rtol=1e-8,
                          message=f"paired equal-exposure squared change mismatch: {family}/{candidate}/{cohort}/{field}")
                # Check the paired scorecards over exactly the shared complete frames.
                verify_scorecard(rows_a, report["same_frame_cohort_scorecards"]["reference"],
                    rows_a, expected_exposures, f"paired/{family}/{candidate}/{cohort}/reference")
                verify_scorecard(rows_b, report["same_frame_cohort_scorecards"]["candidate"],
                    rows_a, expected_exposures, f"paired/{family}/{candidate}/{cohort}/candidate")
                pairs_checked += 1
    return pairs_checked


def verify_scorecard(frames, scorecard, scheduled_frames, expected_exposures, where):
    require(len(frames) == len(scheduled_frames), f"frame denominator changed: {where}")
    require({(f["fold"], f["capture"], int(f["fixation"]), int(f["row"])) for f in frames} ==
            {(f["fold"], f["capture"], int(f["fixation"]), int(f["row"])) for f in scheduled_frames},
            f"frame identity set differs from schedule: {where}")
    for frame in frames:
        require(len(frame["slots"]) == 3 and {s["held_point"] for s in frame["slots"]} == {0, 1, 2},
                f"missing or duplicate held-point slot: {where}")
        computed = per_frame_metrics(frame["slots"])
        require((computed is not None) == bool(frame["complete_triple"]),
                f"complete-frame indicator differs from slots: {where}")
        if computed is not None:
            close(frame["E_px"], computed["E"], atol=1e-9, rtol=1e-9, message=f"frame E differs {where}")
            close(frame["G_theta_deg"], computed["Gtheta"], atol=1e-9, rtol=1e-9, message=f"frame Gtheta differs {where}")
            close(frame["G_A_D"], computed["GA"], atol=1e-9, rtol=1e-9, message=f"frame GA differs {where}")
            close(frame["worst_point_px"], computed["worst"], atol=1e-9, rtol=1e-9, message=f"frame worst differs {where}")
    cards = scorecard["outcomes"]
    complete = [f for f in frames if f["complete_triple"]]
    for key, field in (("E_cross_px", "E_px"), ("G_theta_cross_deg", "G_theta_deg"),
                       ("G_A_cross_D", "G_A_D"), ("worst_point_px", "worst_point_px")):
        rows = [f for f in complete if f.get(field) is not None]
        vals = [float(f[field]) for f in rows]
        expected = exposure_rms(rows, vals, expected_exposures)
        close(cards[key]["equal_exposure_rms"], expected, atol=1e-8, rtol=1e-8,
              message=f"scorecard equal-exposure {key} mismatch {where}")
        close(cards[key]["distribution"]["p95"], scalar_stats(vals)["p95"], atol=1e-8, rtol=1e-8,
              message=f"scorecard {key} p95 mismatch {where}")
    for axis, index in (("x", 0), ("y", 1)):
        slots = [s for f in frames for s in f["slots"] if s.get("scored")]
        vals = [float(s["error_px"][index]) for s in slots]
        expected = exposure_rms(slots, vals, expected_exposures)
        close(scorecard["axis_errors"][axis]["equal_exposure_rms"], expected,
              atol=1e-8, rtol=1e-8, message=f"scorecard {axis} axis mismatch {where}")
    require(scorecard["coverage"]["scheduled_frames"] == len(frames), f"scorecard frame denominator mismatch {where}")
    require(scorecard["coverage"]["scheduled_slots"] == 3*len(frames), f"scorecard slot denominator mismatch {where}")
    scored = [s for f in frames for s in f["slots"] if s.get("scored")]
    complete = [f for f in frames if f.get("complete_triple")]
    require(scorecard["coverage"]["scored"] == len(scored), f"scorecard scored count mismatch {where}")
    require(scorecard["coverage"]["complete_triples"] == len(complete), f"scorecard complete count mismatch {where}")
    for field in ("theta_anchor", "A_anchor", "theta_empirical", "A_empirical",
                  "P1_context_valid", "context_outside_training_extrema", "p1_parity_seen_in_training"):
        expected_counts = Counter("unknown" if s.get("support", {}).get(field) is None else
                                   "true" if s["support"][field] else "false" for f in frames for s in f["slots"])
        for key in ("true", "false", "unknown"):
            require(scorecard["support_counts"][field].get(key, 0) == expected_counts.get(key, 0),
                    f"scorecard support count mismatch for {field}/{key}: {where}")


def verify_fit_objective(outdir, meta, train, states_obj):
    beta = np.asarray(meta["coefficients"], float)
    exponent = float(meta["accommodation_response"]["exponent"])
    y, r = np.asarray(train["v"], float), np.asarray(train["r"], float)
    cov, groups = np.asarray(train["covariance"], float), np.asarray(train["groups"], int)
    states, anchors = np.asarray(states_obj["states"], float), np.asarray(train["anchors"], float)
    n, k, c = len(y), len(anchors), 6
    counts = np.bincount(groups, minlength=k)
    require(np.all(counts > 0), "training groups missing from fixed fit inputs")
    weights = [np.linalg.solve(np.linalg.cholesky(cov[i]), np.eye(c))/np.sqrt(k*counts[groups[i]]) for i in range(n)]
    design0 = np.stack([design(beta, exponent, anchors[groups[i]], r[i]) for i in range(n)])
    A = np.concatenate([weights[i]@design0[i] for i in range(n)], axis=0)
    b = np.concatenate([weights[i]@y[i] for i in range(n)], axis=0)
    scales = np.maximum(np.linalg.norm(A, axis=0), 1e-12)
    center = np.linalg.lstsq(A/scales, b, rcond=1e-10)[0]/scales
    penalty = np.sqrt(.001)*scales
    penalty[[0, 7, 11, 15, 19, 23]] = 0.
    data_residual = np.concatenate([weights[i]@(predict(beta, exponent, states[i], r[i])-y[i]) for i in range(n)])
    prior_residual = penalty*(beta-center)
    means = np.stack([states[groups == g].mean(axis=0) for g in range(k)])
    anchor_residual = ((means-anchors)/np.array([.1, .25])/np.sqrt(k)).ravel()
    computed = dict(optical=.5*float(data_residual@data_residual),
                    prior=.5*float(prior_residual@prior_residual),
                    extra_curvature=0.,
                    anchor=.5*float(anchor_residual@anchor_residual))
    diagnostics = meta["calibration"]
    selected = int(diagnostics["selected_start"])
    lines = jsonl_gzip(outdir/"calibration_candidates.jsonl.gz")
    selected_records = [x for x in lines if int(x["record"]["start"]) == selected]
    require(bool(selected_records), "selected calibration start missing from archive")
    saved = selected_records[-1]
    close(saved["coefficients"], beta, atol=2e-10, rtol=2e-10, message="selected beta checkpoint differs")
    close(saved["states"], states, atol=2e-10, rtol=2e-10, message="selected state checkpoint differs")
    record = saved["record"]
    recorded = record["objective_components"]
    for key, value in computed.items():
        close(recorded[key], value, atol=2e-6, rtol=3e-6, message=f"recomputed calibration {key} cost differs")
    cost = sum(computed.values())
    close(record["cost"], cost, atol=3e-6, rtol=3e-6, message="recomputed calibration total cost differs")
    close(sum(recorded.values()), record["cost"], atol=1e-8, rtol=1e-8, message="saved objective terms do not sum to cost")
    if diagnostics["converged"]:
        require(record["converged"] is True, "artifact marks an uncertified fit as converged")
        require(record["projected_stationarity_physical"] < 1e-3 and record["inner_stationarity_scaled"] < 1e-7,
                "converged fit violates its archived stationarity limits")
    return computed


def jsonl_gzip(path):
    with gzip.open(path, "rt") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def independent_encoded_gradient(beta, exponent, state, r, residual, covariance, indices):
    """Analytic half-chi-square gradient in encoded state coordinates."""
    theta, accommodation = np.asarray(state, float)
    t = theta/10.
    phi = np.log1p(accommodation) if exponent == 0. else np.expm1(exponent*np.log1p(accommodation))/exponent
    phi_a = (1.+accommodation)**(exponent-1.)
    dx_theta = np.array([0., 0., 1., phi, 2.*t, 2.*t*phi, 3.*t*t])/10.
    dx_a = np.array([0., 1., 0., t*phi_a, 0., t*t*phi_a, 0.])
    base_theta = np.array([0., 1., 0., phi])/10.
    base_a = np.array([0., 0., phi_a, t*phi_a])
    jacobian = np.empty((6, 2))
    for point in range(3):
        rx, ry = r[point]
        jacobian[2*point] = (
            dx_theta@beta[:7] + rx*(base_theta@beta[11:15]) + ry*(base_theta@beta[15:19]),
            dx_a@beta[:7] + rx*(base_a@beta[11:15]) + ry*(base_a@beta[15:19]))
        jacobian[2*point+1] = (
            base_theta@beta[7:11] + rx*(base_theta@beta[19:23]) + ry*(base_theta@beta[23:27]),
            base_a@beta[7:11] + rx*(base_a@beta[19:23]) + ry*(base_a@beta[23:27]))
    jacobian *= np.array([10., 4.])
    selected_jacobian = jacobian[np.asarray(indices, int)]
    return selected_jacobian.T@np.linalg.solve(covariance, residual)


def verify_raw_holdouts(outdir, frames, population, captures, groups, meta):
    pilot = np.asarray(meta["pilot_coefficients"], float)
    sigma = np.asarray(meta["coordinate_covariance"], float)
    ref = np.asarray(meta["reference_state"], float)
    beta = np.asarray(meta["coefficients"], float)
    exponent = float(meta["accommodation_response"]["exponent"])
    pop_by = {(int(x["fixation"]), int(x["row"])): x for x in population if x["selected_for_evaluation"]}
    holdouts = jsonl(outdir/"holdouts.jsonl")
    hmap = {(int(h["fixation"]), int(h["row"]), int(h["held_point"])): h for h in holdouts}
    frame_map = {(int(f["fixation"]), int(f["row"])): f for f in frames}
    require(len(hmap) == len(holdouts), "duplicate holdout identities")
    expected = {(gi, row, j) for gi, row in pop_by for j in range(3)}
    require(set(hmap) == expected, "holdout records do not cover exactly three points per scheduled frame")
    covariance_cache = {}
    for (gi, row, held), hold in hmap.items():
        raw = raw_row(captures, groups, gi, row)
        require(raw["p1_valid"] == bool(pop_by[(gi, row)]["p1_valid_geometry"]), "raw P1 validity mismatch")
        key = (gi, row)
        if key not in covariance_cache and raw["p1_valid"]:
            covariance_cache[key] = reference_covariance(raw["p"], pilot, ref, sigma)
            # Validate the independent finite-difference propagation, then use
            # the declared analytic covariance for precision-sensitive gradients.
            from full_position.noise import reference_covariance as declared_reference_covariance
            from full_position.accommodation import PowerResponseModel
            exact_covariance = declared_reference_covariance(
                raw["p"], PowerResponseModel(0., pilot), ref, sigma)
            close(covariance_cache[key], exact_covariance, atol=1e-15, rtol=1e-7,
                  message=f"independent reference covariance mismatch: {(gi,row)}")
            covariance_cache[key] = exact_covariance
        if hold.get("predictions"):
            for prediction in hold["predictions"]:
                state = np.asarray(prediction["state"], float)
                value = predict(beta, exponent, state, raw["r"]).reshape(3, 2)[held]
                q_hat = raw["c"]+raw["ell"]*value
                close(prediction["pixel"], q_hat, atol=2e-7, rtol=2e-9,
                      message=f"saved held-point prediction differs from independent model: {(gi,row,held)}")
            if raw["point_valid"][held] and hold.get("score_available"):
                chosen = np.asarray(hold["predictions"][0]["pixel"], float)
                error = raw["q"][held]-chosen
                close(hold["error_px"], error, atol=2e-7, rtol=2e-9,
                      message=f"held-coordinate residual mismatch: {(gi,row,held)}")
                slot = frame_map[key]["slots"][held]
                close(slot["error_px"], error, atol=2e-7, rtol=2e-9,
                      message=f"joined held-coordinate residual mismatch: {(gi,row,held)}")
        if hold.get("available") is True:
            kept = np.array([i for i in range(6) if i//2 != held], int)
            kept_points = np.unique(kept//2)
            if raw["p1_valid"] and raw["point_valid"][kept_points].all():
                obs = raw["v"].ravel()[kept]
                fitted = predict(beta, exponent, hold["state"], raw["r"])[kept]
                R = covariance_cache[key][np.ix_(kept, kept)]
                residual = obs-fitted
                independent_cost = float(residual@np.linalg.solve(R, residual))
                close(hold["cost"], independent_cost, atol=2e-3, rtol=4e-5,
                      message=f"retained inverse objective mismatch: {(gi,row,held)}")
                for branch in hold.get("branches", []):
                    x = np.asarray(branch["state"], float)
                    require(np.all(x >= BOUNDS[0]-1e-9) and np.all(x <= BOUNDS[1]+1e-9),
                            f"inverse branch outside physical bounds: {(gi,row,held)}")
                    value = predict(beta, exponent, x, raw["r"])[kept]
                    e = obs-value
                    branch_cost = float(e@np.linalg.solve(R, e))
                    close(branch["cost"], branch_cost, atol=2e-3, rtol=4e-5,
                          message=f"archived inverse branch cost mismatch: {(gi,row,held)}")
                    jac = np.empty((len(kept), 2))
                    for axis, step in enumerate((1e-4, 1e-5)):
                        plus, minus = x.copy(), x.copy()
                        plus[axis] += step; minus[axis] -= step
                        jac[:, axis] = (predict(beta, exponent, plus, raw["r"])[kept]-
                                        predict(beta, exponent, minus, raw["r"])[kept])/(2*step)
                    weighted_jac = np.linalg.solve(np.linalg.cholesky(R), jac)
                    singular = np.linalg.svd(weighted_jac, compute_uv=False)
                    independent_rank = int(np.sum(singular > max(singular[0]*1e-6, 1e-8)))
                    require(int(branch["rank"]) == independent_rank,
                            f"inverse branch rank mismatch: {(gi,row,held)}")
                    cert = branch.get("certificate", {})
                    require(cert.get("certified") is True, f"saved branch lacks a certificate: {(gi,row,held)}")
                    require(cert.get("local_minimum") is True and cert.get("stable_cost") is True and
                            cert.get("physical_correction", np.inf) <= 1e-5,
                            f"saved branch certificate violates its acceptance contract: {(gi,row,held)}")
                    require(cert.get("stationarity_encoded", np.inf) <= 1e-4,
                            f"saved branch stationarity exceeds its acceptance contract: {(gi,row,held)}")
                    z = x/STATE_SCALE
                    zlo, zhi = BOUNDS[0]/STATE_SCALE, BOUNDS[1]/STATE_SCALE
                    grad = independent_encoded_gradient(beta, exponent, x, raw["r"],
                                                         value-obs, R, kept)
                    numeric_stationarity = float(np.max(np.abs(z-np.clip(z-grad, zlo, zhi))))
                    close(cert["stationarity_encoded"], numeric_stationarity,
                          atol=2e-4, rtol=3e-2,
                          message=f"inverse stationarity certificate mismatch: {(gi,row,held)}")
                    require(np.isfinite(branch_cost), f"nonfinite branch objective: {(gi,row,held)}")
    return len(holdouts)


def verify_all(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    # Aggregate counters are initialized once, outside all fold loops.
    counts = dict(folds=0, fit_tasks=0, scored_slots=0, scheduled_slots=0,
                  verified_training_rows=0, verified_holdouts=0, converged_fits=0,
                  failed_fits=0, checks=0)
    config = read_json(output/"config.json")
    require(config.get("schema") == "accommodation_response_screen_v1", "unexpected run schema")
    require(config.get("purpose") == "sampled_development_screen_not_nested_selection", "run scope changed")
    require(config.get("selected_for_outer_evaluation") is None and config.get("promoted_for_deployment") is False,
            "screen incorrectly declares selection or promotion")
    policy = config["coverage_policy"]
    require(policy.get("absolute_accuracy_thresholds") is None and
            policy.get("nominal_error_is_acceptance_gate") is False,
            "selection policy has an accuracy/nominal-label gate")
    require(config["candidates"] == CANDIDATES, "candidate/exponent manifest changed")
    require(len(config["folds"])*len(CANDIDATES) == EXPECTED_TASKS, "planned task count is not 36")
    verify_source_snapshots(root, output, config); counts["checks"] += 1
    captures, groups, interval_hash = load_pinned_sources(root, output, config); counts["checks"] += 1
    outcomes = read_json(output/"outcomes.json")
    task_keys = [(x["fold"], x["candidate"]) for x in outcomes]
    expected_tasks = {(fold, candidate) for fold in config["folds"] for candidate in CANDIDATES}
    require(len(task_keys) == len(set(task_keys)) and set(task_keys) == expected_tasks,
            "36 fit outcome table is incomplete or duplicated")
    outcome_map = {(x["fold"], x["candidate"]): x for x in outcomes}
    fold_rows, family_rows, all_populations = {}, defaultdict(lambda: defaultdict(list)), {}
    for split_info in read_json(output/"split_manifest.json"):
        fold = split_info["fold"]
        counts["folds"] += 1
        require(fold in config["folds"], f"unexpected split {fold}")
        training = read_json(output/"splits"/fold/"training.json")
        split_dir = output/"splits"/fold
        require(sha256(split_dir/"manifest.json") == split_info["schedule_sha256"],
                f"schedule manifest digest mismatch: {fold}")
        require(sha256(split_dir/"training_inputs.npz") == split_info["training_input_sha256"],
                f"training archive digest mismatch: {fold}")
        require(training["provenance"]["interval_sha256"] == interval_hash,
                f"interval provenance mismatch in {fold}")
        train = verify_training_inputs(root, output, config, split_info, groups, captures)
        counts["verified_training_rows"] += len(train["rows"])
        population, schedule = verify_population(output, config, fold, split_info, groups, captures)
        all_populations[fold] = (population, schedule)
        frame_items = schedule["frames"]
        expected_exposures = {(fold, f["capture"], int(f["fixation"])) for f in frame_items}
        fold_rows[fold] = {}
        for candidate, exponent in CANDIDATES.items():
            counts["fit_tasks"] += 1
            outdir = output/"fits"/fold/candidate
            completion = read_json(outdir/"completion.json")
            require(completion.get("complete") is True, f"fit output incomplete: {fold}/{candidate}")
            meta_path = outdir/("model.json" if (outdir/"model.json").exists() else "failed_checkpoint.json")
            meta = read_json(meta_path)
            require(meta.get("schema") == "conditional_power_response_v1" and meta.get("model") == candidate,
                    f"power artifact identity mismatch: {fold}/{candidate}")
            require(meta.get("accommodation_response", {}).get("exponent") == exponent,
                    f"response exponent mismatch: {fold}/{candidate}")
            require(meta.get("coefficient_order") == COEFFICIENT_ORDER and len(meta.get("coefficients", [])) == 27,
                    f"response coefficient schema mismatch: {fold}/{candidate}")
            is_converged = meta.get("calibration", {}).get("converged") is True
            require(completion.get("calibration_converged") is is_converged,
                    f"completion/calibration status mismatch: {fold}/{candidate}")
            if is_converged:
                counts["converged_fits"] += 1
                require(meta_path.name == "model.json", f"certified fit stored as failed checkpoint: {fold}/{candidate}")
            else:
                counts["failed_fits"] += 1
                require(meta_path.name == "failed_checkpoint.json", f"failed fit stored as applicable model: {fold}/{candidate}")
            states_obj = None
            if (outdir/"training_states.json").exists():
                states_obj = verify_trajectory(outdir, train, training)
                if is_converged:
                    objective = verify_fit_objective(outdir, meta, train, states_obj)
                    require(np.isfinite(list(objective.values())).all(), "nonfinite recomputed fit objective")
            frames = read_json(outdir/"frames.json")
            card = read_json(outdir/"scorecard.json")
            verify_scorecard(frames, card, frame_items, expected_exposures, f"{fold}/{candidate}")
            counts["scheduled_slots"] += 3*len(frame_items)
            counts["scored_slots"] += sum(bool(s.get("scored")) for f in frames for s in f["slots"])
            counts["verified_holdouts"] += verify_raw_holdouts(outdir, frames, population, captures, groups, meta)
            fold_rows[fold][candidate] = frames
            family_rows[fold.split("_")[0]][candidate].extend(frames)
    require(counts["folds"] == len(config["folds"]) and counts["fit_tasks"] == EXPECTED_TASKS,
            "aggregate fold or task denominator mismatch")
    summary = read_json(output/"summary.json")
    require(summary.get("selected_for_outer_evaluation") is None and summary.get("promoted_for_deployment") is False,
            "summary claims a selected or promoted model")
    require(summary.get("scope", "").startswith("sampled_screen_only"), "summary scope is not a sampled screen")
    for family, candidates in family_rows.items():
        expected_exposures = set()
        for fold in config["folds"]:
            if fold.startswith(family+"_"):
                schedule = all_populations[fold][1]
                expected_exposures.update((fold, f["capture"], int(f["fixation"])) for f in schedule["frames"])
        family_summary = summary["families"][family]
        for candidate, frames in candidates.items():
            verify_scorecard(frames, family_summary["scorecards"][candidate], frames, expected_exposures,
                             f"summary/{family}/{candidate}")
        for candidate, cohorts in family_summary["comparisons"].items():
            require(candidate in CANDIDATES and candidate != "ar27_log", "unknown paired candidate in summary")
            for cohort, report in cohorts.items():
                require(report.get("direction") == candidate+"_minus_fresh_log",
                        f"paired comparison direction mismatch: {family}/{candidate}/{cohort}")
                require(report.get("scientific_scope") == "sampled_development_law_comparison_not_nested_selection",
                        f"paired comparison scope mismatch: {family}/{candidate}/{cohort}")
                require(report.get("scheduled_frames", 0) > 0, "paired report lost its schedule denominator")
    counts["paired_cohort_cells"] = verify_paired_outputs(output, config, all_populations, family_rows, summary)
    selection = read_json(output/"selection.json")
    require(selection.get("status") == "development_screen_only" and
            selection.get("selected_for_outer_evaluation") is None and
            selection.get("promoted_for_deployment") is False,
            "selection output overclaims the sampled screen")
    report = dict(schema="accommodation_response_independent_verification_v1", status="passed",
        scope="sampled development screen only; not nested selection or deployment promotion",
        interval_sha256=interval_hash, counters=counts,
        verification_limits=["historical used groups remain development evidence",
            "local finite-difference covariance/objective checks are numerical consistency checks, not optical accuracy validation"])
    (output/"independent_verification.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    result = verify_all(args.root, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
