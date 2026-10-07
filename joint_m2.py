"""Fit and estimate gaze/accommodation from exp5 three-point P1/P4 triangles.

Observables are d=(mean(P4x)-mean(P1x))/sqrt(area(P1)) and
rho=area(P4)/area(P1). Fixation intervals and nominal labels are read from
the reviewed exp5 fixation_intervals.json; stored detections are never rerun.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE
EXP2 = HERE / "lib"
sys.path.insert(0, str(EXP2))
from calibrate_profiled import ProfiledProblem, noise_covariance  # noqa: E402
from calibrate_continuation import array_hash, atomic_json, continue_fit, make_manifest  # noqa: E402
import m2_model  # noqa: E402

FOLDS = {"full": None, "holdout3": 3, "holdout2": 4}
OBSERVABLE_SCHEMA = "exp5_triangle_d_area_ratio_v1"
OBSERVABLE_DESCRIPTION = (
    "d=(mean(P4x)-mean(P1x))/sqrt(area(P1)); "
    "rho=area(P4)/area(P1); triangle area=abs(2D cross product)/2; pixels"
)


def triangle_area(points: np.ndarray) -> np.ndarray:
    """Absolute areas for (..., 3, 2) triangles, in square pixels."""
    p = np.asarray(points, dtype=float)
    cross = ((p[..., 1, 0] - p[..., 0, 0]) * (p[..., 2, 1] - p[..., 0, 1])
             - (p[..., 1, 1] - p[..., 0, 1]) * (p[..., 2, 0] - p[..., 0, 0]))
    return np.abs(cross) * 0.5


def measurements(arrays: dict) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Return [d,rho], strict exp5 validity, and inspectable raw components."""
    p1 = np.asarray(arrays["p1_xy"], dtype=float)
    p4 = np.asarray(arrays["p4_xy"], dtype=float)
    n = len(np.asarray(arrays["frame_index"]))
    if p1.shape != (n, 3, 2) or p4.shape != (n, 3, 2):
        raise ValueError("exp5 P1/P4 coordinates must have shape (n,3,2)")
    finite = np.isfinite(p1).all(axis=(1, 2)) & np.isfinite(p4).all(axis=(1, 2))
    if "p4_found" not in arrays:
        raise ValueError("exp5 detections lack p4_found validity flags")
    found = np.asarray(arrays["p4_found"], dtype=bool)
    if found.shape != (n, 3):
        raise ValueError("p4_found must have shape (n,3)")
    a1, a4 = triangle_area(p1), triangle_area(p4)
    # Exclude only numerically degenerate P1 triangles; threshold scales with
    # the triangle's own edge lengths and does not impose a pixel-size gate.
    edge_sq = np.maximum.reduce([
        np.sum((p1[:, 1] - p1[:, 0])**2, axis=1),
        np.sum((p1[:, 2] - p1[:, 0])**2, axis=1),
        np.sum((p1[:, 2] - p1[:, 1])**2, axis=1),
    ])
    threshold = 32 * np.finfo(float).eps * np.maximum(edge_sq, 1.)
    valid = finite & found.all(axis=1) & (a1 > threshold)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.sqrt(a1)
        shift = p4[:, :, 0].mean(axis=1) - p1[:, :, 0].mean(axis=1)
        rho = a4 / a1
        d = shift / scale
    y = np.column_stack([d, rho])
    valid &= np.isfinite(y).all(axis=1)
    return y, valid, {"area_p1_px2": a1, "area_p4_px2": a4,
                      "sqrt_area_p1_px": scale, "centroid_dx_px": shift}


def _isolated_jump_mask(y: np.ndarray, groups: np.ndarray, radius: int = 12,
                        threshold: float = 8.0) -> tuple[np.ndarray, list[dict]]:
    """Flag single-frame excursions that immediately recover within each fixation."""
    y = np.asarray(y, float)
    groups = np.asarray(groups, int)
    reject = np.zeros(len(y), dtype=bool)
    details = []
    for group in np.unique(groups[groups >= 0]):
        ix = np.flatnonzero(groups == group)
        if len(ix) < 3:
            continue
        values = y[ix]
        # Second differences remove local linear motion; their robust scale
        # estimates the observation noise without treating a ramp as a jump.
        second = np.diff(values, n=2, axis=0)
        sigma = 1.4826 * np.median(np.abs(second - np.median(second, axis=0)), axis=0) / np.sqrt(6.)
        scale_floor = np.finfo(float).eps * np.maximum(1., np.max(np.abs(values), axis=0)) * 64
        sigma = np.maximum(sigma, scale_floor)
        candidates = []
        for pos in range(1, len(ix)-1):
            left, center, right = values[pos-1:pos+2]
            baseline = (left + right) * 0.5
            signed_residual = center - baseline
            residual = np.abs(signed_residual)
            # A one-frame excursion creates opposite signed increments into
            # and out of the sample. A monotone ramp/step does not.
            before, after = center-left, right-center
            lo, hi = max(0, pos-radius), min(len(values), pos+radius+1)
            local_second = np.diff(values[lo:hi], n=2, axis=0)
            if len(local_second):
                local_sigma = 1.4826 * np.median(np.abs(local_second - np.median(local_second, axis=0)), axis=0) / np.sqrt(6.)
                sigma_here = np.maximum(local_sigma, scale_floor)
            else:
                sigma_here = sigma
            recovery = (before * after < 0) & (np.minimum(np.abs(before), np.abs(after)) > threshold*sigma_here)
            if np.any((residual > threshold*sigma_here) & recovery):
                candidates.append((pos, residual / sigma_here, signed_residual, sigma_here, before, after))
        # A single spike can make the adjacent recovery sample look like a
        # second candidate; keep only the strongest point in each adjacent run.
        candidate_by_pos = {c[0]: c for c in candidates}
        positions = sorted(candidate_by_pos)
        runs = []
        for pos in positions:
            if not runs or pos > runs[-1][-1] + 1:
                runs.append([pos])
            else:
                runs[-1].append(pos)
        for run in runs:
            pos = max(run, key=lambda p: float(np.max(candidate_by_pos[p][1])))
            _, score, signed_residual, sigma_here, before, after = candidate_by_pos[pos]
            reject[ix[pos]] = True
            details.append(dict(group=int(group), index=int(ix[pos]),
                                    d_residual=float(signed_residual[0]), rho_residual=float(signed_residual[1]),
                                    d_sigma=float(sigma_here[0]), rho_sigma=float(sigma_here[1]),
                                    d_recovery_gap=float(abs(before[0] + after[0])),
                                    rho_recovery_gap=float(abs(before[1] + after[1])),
                                    reason="isolated_observation_jump"))
    return reject, details


def _jump_return_bursts(y: np.ndarray, frames: np.ndarray, groups: np.ndarray,
                        window_frames: int = 201, max_span_frames: int = 150,
                        threshold: float = 8.0, min_d_jump: float = 0.02,
                        min_rho_jump: float = 0.005,
                        refractory_frames: int = 50) -> tuple[np.ndarray, list[dict]]:
    """Find abrupt excursions that return to the local pre-event level.

    Spans use original frame numbers, so invalid-detection gaps never compress
    a long event into a short burst. Only samples inside the excursion are
    marked; recovery samples and fixation transitions remain in the fit.
    """
    y, frames, groups = np.asarray(y, float), np.asarray(frames, int), np.asarray(groups, int)
    reject = np.zeros(len(y), dtype=bool)
    details = []
    for group in np.unique(groups[groups >= 0]):
        ix = np.flatnonzero(groups == group)
        if len(ix) < 4:
            continue
        values, frame = y[ix], frames[ix]
        gaps = np.diff(frame)
        local_diffs = np.diff(values, axis=0)[gaps <= 2]
        if len(local_diffs) < 3:
            continue
        sigma = 1.4826 * np.median(np.abs(local_diffs - np.median(local_diffs, axis=0)), axis=0) / np.sqrt(2.)
        sigma = np.maximum(sigma, np.finfo(float).eps * np.maximum(1., np.max(np.abs(values), axis=0))*64)
        pos = 1
        min_jump = np.array([min_d_jump, min_rho_jump], dtype=float)
        while pos < len(ix)-1:
            onset = values[pos] - values[pos-1]
            if (frame[pos]-frame[pos-1] > max_span_frames or
                    not np.all(np.abs(onset) > threshold*sigma) or
                    not np.all(np.abs(onset) >= min_jump)):
                pos += 1
                continue
            pre = np.flatnonzero((frame[:pos] >= frame[pos]-window_frames) &
                                 (frame[:pos] < frame[pos]))
            if not len(pre):
                pos += 1
                continue
            baseline = np.median(values[pre], axis=0)
            stop = np.searchsorted(frame, frame[pos]+max_span_frames, side="right")
            recovery = None
            for end in range(pos+1, min(stop, len(ix))):
                    post = np.flatnonzero((frame[end:] <= frame[end]+window_frames)) + end
                    if not len(post):
                        continue
                    post_level = np.median(values[post], axis=0)
                    return_jump = values[end]-values[end-1]
                    pulse_amplitude = np.max(np.abs(values[pos:end]-baseline), axis=0)
                    net_drift = np.abs(post_level-baseline)
                    near_baseline = np.all(net_drift <= 0.2*pulse_amplitude + threshold*sigma)
                    returns_abruptly = (np.all(np.abs(return_jump) > threshold*sigma) and
                                        np.all(np.abs(return_jump) >= min_jump))
                    if (near_baseline and returns_abruptly and
                            np.all(np.abs(values[end]-baseline) <= 0.2*pulse_amplitude + threshold*sigma)):
                        recovery = end
                        break
            if recovery is not None:
                event = ix[pos:recovery]
                # Require at least one displaced observation and avoid
                # labelling a gradual trend as a detector excursion.
                displaced = np.all(np.abs(values[pos:recovery]-baseline) > threshold*sigma, axis=1)
                if displaced.any():
                    reject[event] = True
                    residuals = np.abs(values[pos:recovery]-baseline)
                    pulse_amplitude = np.max(residuals, axis=0)
                    net_drift = np.abs(np.median(values[np.flatnonzero(
                        (frame[recovery:] <= frame[recovery]+window_frames))+recovery], axis=0)-baseline)
                    pulse_consistent = np.all(net_drift <= 0.2*pulse_amplitude + threshold*sigma)
                    if not pulse_consistent:
                        reject[event] = False
                        pos += 1
                        continue
                    d = dict(group=int(group), start_index=int(ix[pos]), end_index=int(ix[recovery]),
                             start_frame=int(frame[pos]), end_frame=int(frame[recovery]),
                             span_frames=int(frame[recovery]-frame[pos]),
                             valid_sample_count=int(len(event)), baseline_d=float(baseline[0]),
                             baseline_rho=float(baseline[1]),
                             max_abs_d_residual=float(residuals[:, 0].max()),
                             max_abs_rho_residual=float(residuals[:, 1].max()),
                             d_sigma=float(sigma[0]), rho_sigma=float(sigma[1]),
                             min_d_jump=float(min_d_jump), min_rho_jump=float(min_rho_jump),
                             refractory_frames=int(refractory_frames),
                             reason="isolated_jump_return_burst")
                    details.extend(dict(d, index=int(k), d=float(y[k, 0]), rho=float(y[k, 1]))
                                     for k in event)
                    pos = int(np.searchsorted(frame, frame[recovery]+refractory_frames, side="left"))
                    continue
            pos += 1
    return reject, details


def _report_and_data(experiment_dir: Path, interval_path: Path, reject_isolated_jumps: bool = False,
                     jump_window: int = 12, jump_threshold: float = 8.0,
                     edge_fraction: float = 0.1, jump_burst_window: int = 201,
                     jump_max_burst_frames: int = 150, jump_min_d: float = 0.02,
                     jump_min_rho: float = 0.005, jump_refractory_frames: int = 50):
    if not 0 <= edge_fraction < 0.5:
        raise ValueError("edge_fraction must be in [0, 0.5)")
    raw = interval_path.read_bytes()
    report = json.loads(raw)
    rows = report.get("fixations")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Fixation interval report has no fixations")
    if report.get("capture5_used") is not False:
        raise ValueError("Expected the reviewed exp5 capture1-4 fixation set")
    expected_captures = [f"capture_{i}_detections.pkl" for i in range(1, 5)]
    if len(rows) != 20 or sorted({r.get("capture") for r in rows}) != expected_captures:
        raise ValueError("Expected five stored fixations in each of captures 1-4")
    if report.get("native_detection_rerun") is not False:
        raise ValueError("Expected frozen intervals produced without rerunning detection")
    hashes = {source["capture"]: source["sha256"] for source in report.get("sources", [])}
    capture_names = sorted({r["capture"] for r in rows}, key=lambda x: int(x.split("_")[1]))
    records, meta, ylist, flist, glist = [], [], [], [], []
    observed_sources = {}
    for name in capture_names:
        path = experiment_dir / name
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if hashes.get(name) != digest:
            raise ValueError(f"Detection source hash mismatch for {name}")
        data = pickle.loads(payload)
        arrays = data["arrays"]
        y, valid, components = measurements(arrays)
        frame = np.asarray(arrays["frame_index"], dtype=np.int64)
        if len(frame) != len(y) or np.any(np.diff(frame) <= 0):
            raise ValueError(f"Invalid frame index sequence in {name}")
        full = np.zeros(len(y), dtype=bool)
        core = np.zeros(len(y), dtype=bool)
        groups = np.full(len(y), -1, dtype=np.int64)
        selected_rows = [r for r in rows if r["capture"] == name]
        for r in selected_rows:
            j = len(meta)
            start, end = int(r["start_row"]), int(r["end_row_exclusive"])
            if not 0 <= start < end <= len(y):
                raise ValueError(f"Invalid fixation row bounds in {name}")
            if r.get("first_frame") != int(frame[start]) or r.get("last_frame_inclusive") != int(frame[end-1]):
                raise ValueError(f"Fixation frame provenance mismatch in {name}")
            if r.get("full_valid_count") != int(valid[start:end].sum()):
                raise ValueError(f"Fixation validity count mismatch in {name}")
            if full[start:end].any():
                raise ValueError(f"Overlapping fixation intervals in {name}")
            if not np.isfinite(r.get("target_theta_deg", np.nan)) or not np.isfinite(r.get("demand_diopters_label", np.nan)):
                raise ValueError("Every fitted interval must provide finite nominal target and demand labels")
            full[start:end] = True
            cut = int(np.floor(edge_fraction * (end-start)))
            c0, c1 = start+cut, end-cut
            core[c0:c1] = True
            full_use = np.flatnonzero(valid[start:end]) + start
            use = np.flatnonzero(valid[c0:c1]) + c0
            if not len(full_use) or not len(use):
                raise ValueError(f"No valid observations in full interval or central core of fixation {j}")
            groups[use] = j
            meta.append(dict(r, fixation_index=j, valid_count=int(len(full_use)),
                             core_start_row=c0, core_end_row_exclusive=c1,
                             core_valid_count=int(len(use))))
            ylist.append(y[use]); flist.append(frame[use]); glist.append(np.full(len(use), j, dtype=np.int64))
        records.append(dict(capture=name, arrays=arrays, y=y, valid=valid, full=full,
                            core=core, groups=groups, components=components))
        observed_sources[name] = digest
    raw_y, raw_frames, raw_groups = np.vstack(ylist), np.concatenate(flist), np.concatenate(glist)
    jump_reject = np.zeros(len(raw_y), dtype=bool)
    jump_details = []
    if reject_isolated_jumps:
        single_mask, single_details = _isolated_jump_mask(raw_y, raw_groups, jump_window, jump_threshold)
        burst_mask, burst_details = _jump_return_bursts(
            raw_y, raw_frames, raw_groups, jump_burst_window,
            jump_max_burst_frames, jump_threshold, jump_min_d,
            jump_min_rho, jump_refractory_frames)
        jump_reject = single_mask | burst_mask
        jump_details = burst_details + [d for d in single_details if not burst_mask[d["index"]]]
    y, frames, groups = raw_y[~jump_reject], raw_frames[~jump_reject], raw_groups[~jump_reject]
    # Record each flagged sample by original capture/frame so the rejection
    # list remains auditable independently of compacted fit arrays.
    for d in jump_details:
        gi, original_index = d["group"], d["index"]
        d["capture"] = meta[gi]["capture"]
        d["frame_index"] = int(raw_frames[original_index])
        d["d"] = float(raw_y[original_index, 0]); d["rho"] = float(raw_y[original_index, 1])
        if "baseline_d" not in d:
            d["baseline_d"] = float(raw_y[original_index, 0] - d["d_residual"])
            d["baseline_rho"] = float(raw_y[original_index, 1] - d["rho_residual"])
        d["d_residual"] = float(raw_y[original_index, 0] - d["baseline_d"])
        d["rho_residual"] = float(raw_y[original_index, 1] - d["baseline_rho"])
    reject_counts = {str(j): int(np.sum(jump_reject & (raw_groups == j))) for j in range(len(meta))}
    for d in jump_details:
        rec = next(r for r in records if r["capture"] == d["capture"])
        f = np.asarray(rec["arrays"]["frame_index"])
        k = int(np.searchsorted(f, d["frame_index"]))
        d["timestamp_ms"] = float(rec["arrays"]["timestamp_ms"][k])
    for j, row in enumerate(meta):
        row["fit_valid_count"] = row["core_valid_count"] - reject_counts[str(j)]
    rejected_keys = {(d["capture"], d["frame_index"]) for d in jump_details}
    for rec in records:
        rec["fit_valid"] = rec["valid"] & rec["core"]
        if reject_isolated_jumps:
            rec["fit_valid"] &= np.array([(rec["capture"], int(f)) not in rejected_keys
                                           for f in rec["arrays"]["frame_index"]])
    policy = dict(enabled=bool(reject_isolated_jumps), method="isolated_jump_and_return_v2",
                   window_radius_frames=int(jump_window), threshold_sigma=float(jump_threshold),
                   burst_baseline_window_frames=int(jump_burst_window),
                   max_burst_span_frames=int(jump_max_burst_frames),
                   minimum_d_boundary_jump=float(jump_min_d),
                   minimum_rho_boundary_jump=float(jump_min_rho),
                   burst_refractory_frames=int(jump_refractory_frames),
                   recovery_rule="isolated spike recovers immediately; burst needs both d/rho boundaries above sigma and absolute floors, pre/post level agreement, same fixation, and refractory gap",
                   counts_by_fixation=reject_counts, total_rejected=int(jump_reject.sum()))
    provenance = {
        "experiment": "exp5_joint_triangle_m2_v1", "fold": None,
        "selected_interval_path": str(interval_path),
        "selected_interval_sha256": hashlib.sha256(raw).hexdigest(),
        "source_sha256": observed_sources,
        "validity_policy": "stored exp5 validity: finite P1/P4, p4_found all true, numerically nondegenerate P1 triangle; no pupil gate",
        "fixation_support": "central core of each stored interval, with exp2 floor(edge_fraction*interval_rows) margin per side",
        "edge_trimming": bool(edge_fraction > 0),
        "edge_fraction": float(edge_fraction),
        "support_counts_by_fixation": {str(j): dict(full_valid=meta[j]["valid_count"],
            core_valid=meta[j]["core_valid_count"], retained=meta[j]["fit_valid_count"])
            for j in range(len(meta))},
        "outlier_rejection": bool(reject_isolated_jumps),
        "observation_filter": policy,
        "require_valid_pupil": False,
        "native_detection_rerun": False,
        "target_and_demand_labels": "read from stored fixation interval metadata; used only as nominal mean-state anchors and initialization",
        "observable_schema": OBSERVABLE_SCHEMA,
        "observable_description": OBSERVABLE_DESCRIPTION,
        "sqrt_area_p1_degeneracy_threshold": "32*machine_epsilon*max(max_squared_edge_length,1 px^2)",
    }
    return y, frames, groups, meta, records, provenance, jump_details


def _training_data(y, frames, groups, meta, fold_name):
    held_capture = FOLDS[fold_name]
    held = [j for j, row in enumerate(meta)
            if held_capture is not None and int(row["capture"].split("_")[1]) == held_capture]
    selected = [j for j in range(len(meta)) if j not in held]
    mask = np.isin(groups, selected)
    # Remap global fixation ids to dense local optimizer ids.
    local_group = np.searchsorted(selected, groups[mask])
    ty, tf = y[mask].copy(), frames[mask].copy()
    targets = np.array([meta[j]["target_theta_deg"] for j in selected], dtype=float)
    demands = np.array([meta[j]["demand_diopters_label"] for j in selected], dtype=float)
    coef = m2_model.initial_coefficients(ty, local_group, targets, demands, len(selected))
    theta = m2_model.initial_theta(coef, ty[:, 0], demands[local_group])
    covariance = noise_covariance(ty, tf, local_group)
    ranges = np.ptp(ty, axis=0)
    if np.any(ranges <= 0):
        raise ValueError("Degenerate exp5 training observable range")
    problem = ProfiledProblem(
        ty, tf, local_group, targets, demands, None, coef,
        np.linalg.inv(covariance), np.diag(1 / ranges**2), model="m2",
        theta_anchor_scale_deg=0.1)
    initial = problem.encode(theta, demands[local_group])
    detail = dict(initial_coefficients=coef.tolist(),
                  initial_fit="training-frame observations at interval nominal labels; equal fixation weight",
                  training_mean_observations=[ty[local_group == j].mean(axis=0).tolist() for j in range(len(selected))],
                  noise_covariance=covariance.tolist(), prior_rule="diag(1/ptp(training frame observations,axis=0)^2), strength 0.1",
                  nominal_gaze_anchor_scale_deg=0.1, nominal_accommodation_anchor_scale_diopters=0.25,
                  temporal_smoothing_strength=0.1, temporal_smoothing_scales=[1.0, 0.25])
    return problem, initial, selected, held, local_group, covariance, detail


def _apply_cached_inverse_initial(problem, initial, y, frames, groups, meta, csv_path: Path,
                                  expected_provenance: dict):
    """Replace only the latent-state start with a verified cached independent inverse."""
    csv_path = Path(csv_path)
    rows = list(csv.DictReader(csv_path.open(newline="")))
    if len(rows) != problem.n:
        raise ValueError(f"Cached inverse row count {len(rows)} does not match selected core observations {problem.n}")
    by_key = {}
    for row in rows:
        key = (row["capture"], int(row["frame_index"]))
        if key in by_key:
            raise ValueError(f"Duplicate cached inverse frame {key}")
        by_key[key] = row
    theta, accommodation, ambiguous = np.empty(problem.n), np.empty(problem.n), np.empty(problem.n, dtype=bool)
    bounds = np.zeros(problem.n, dtype=bool)
    for i, (frame, group) in enumerate(zip(frames, groups)):
        key = (meta[int(group)]["capture"], int(frame))
        row = by_key.get(key)
        if row is None:
            raise ValueError(f"Cached inverse lacks selected observation {key}")
        if not (np.isclose(float(row["d"]), y[i, 0], rtol=1e-10, atol=1e-12)
                and np.isclose(float(row["rho"]), y[i, 1], rtol=1e-10, atol=1e-12)):
            raise ValueError(f"Cached inverse observations differ at {key}")
        theta[i], accommodation[i] = float(row["theta_deg"]), float(row["A_diopters"])
        ambiguous[i] = int(row["equivalent_minima_count"]) > 1
        bounds[i] = str(row["theta_bound"]).lower() == "true" or str(row["A_bound"]).lower() == "true"
    if not np.isfinite(theta).all() or not np.isfinite(accommodation).all():
        raise ValueError("Cached inverse states contain non-finite values")
    if np.any((theta < -20) | (theta > 20) | (accommodation < 0) | (accommodation > 6)):
        raise ValueError("Cached inverse states violate physical bounds")
    model_path = csv_path.parent.parent / "model.json"
    if not model_path.is_file():
        raise ValueError("Cached inverse needs the adjacent initializer model.json")
    model = json.loads(model_path.read_text())
    provenance = model["diagnostics"]["provenance"]
    if not model["diagnostics"].get("baseline_only", provenance.get("baseline_only")):
        raise ValueError("Cached inverse model is not marked as an initializer baseline")
    if not np.allclose(model["coefficients"], problem.initial_coef, rtol=1e-10, atol=1e-12):
        raise ValueError("Cached inverse coefficients do not match the current nominal-state initialization")
    if not np.allclose(model["precision"], problem.W, rtol=1e-10, atol=1e-12):
        raise ValueError("Cached inverse precision does not match the current training covariance")
    for key in ("selected_interval_sha256", "source_sha256", "observable_schema",
                "edge_fraction", "observation_filter"):
        if provenance.get(key) != expected_provenance.get(key):
            raise ValueError(f"Cached inverse {key} differs from current training data policy")
    initial = problem.encode(theta, accommodation)
    detail = dict(mode="cached independent per-frame inverse states; labels are not inverse inputs",
                  source_csv=str(csv_path), source_csv_sha256=hashlib.sha256(csv_path.read_bytes()).hexdigest(),
                  source_model=str(model_path), source_model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
                  source_provenance=provenance,
                  ambiguous_count=int(ambiguous.sum()), ambiguous_fraction=float(ambiguous.mean()),
                  bound_count=int(bounds.sum()), frame_count=int(problem.n))
    return initial, detail


def _csv(path: Path, rows: list[dict]):
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def _new_dir(path: Path):
    if path.exists():
        raise ValueError(f"Output already exists; choose a new directory: {path}")
    path.mkdir(parents=True)


def _forward_m2_only(theta, accommodation, coef):
    """M2 prediction without allocating the Jacobian for trial states."""
    theta, accommodation = np.broadcast_arrays(
        np.asarray(theta, dtype=float), np.asarray(accommodation, dtype=float))
    c = np.asarray(coef, dtype=float)
    t = theta / 15
    log_a = np.log1p(accommodation)
    slope = c[2] + c[3] * log_a
    curvature = c[4] + c[5] * log_a
    ratio = c[7] + c[8] * t + c[9] * t * t
    ratio_a = c[10] + c[11] * t + c[12] * t * t
    return np.stack([
        c[0] + c[1] * accommodation + slope * t + curvature * t * t + c[6] * t**3,
        ratio + log_a * ratio_a,
    ], axis=-1)


def _invert_batch_m2_fast(y, coef, W, maxiter):
    """Same multistart GN and diagnostics as exp2, with Jacobian-free trial predictions."""
    L = np.linalg.cholesky(W).T
    starts = np.array([(th, a) for a in m2_model.A_STARTS for th in m2_model.THETA_STARTS])
    z = np.broadcast_to(starts[None], (len(y), len(starts), 2)).copy()
    lower = np.array([m2_model.THETA_BOUNDS[0], m2_model.A_BOUNDS[0]])
    upper = np.array([m2_model.THETA_BOUNDS[1], m2_model.A_BOUNDS[1]])
    damping = np.full(z.shape[:2], 1e-4)
    for _ in range(maxiter):
        pred, J = m2_model.forward_jac_m2(z[..., 0], z[..., 1], coef)
        error = (pred-y[:, None]) @ L.T
        JW = np.einsum('ab,nkbc->nkac', L, J)
        g = np.einsum('nkac,nka->nkc', JW, error)
        h00 = np.sum(JW[..., 0]**2, axis=-1)
        h11 = np.sum(JW[..., 1]**2, axis=-1)
        h01 = np.sum(JW[..., 0]*JW[..., 1], axis=-1)
        d0, d1 = damping*np.maximum(h00, 1.), damping*np.maximum(h11, 1.)
        determinant = np.maximum((h00+d0)*(h11+d1)-h01*h01, 1e-30)
        step = np.stack([((h11+d1)*g[..., 0]-h01*g[..., 1])/determinant,
                         ((h00+d0)*g[..., 1]-h01*g[..., 0])/determinant], axis=-1)
        trial = np.clip(z-step, lower, upper)
        before = np.sum(error**2, axis=-1)
        trial_pred = _forward_m2_only(trial[..., 0], trial[..., 1], coef)
        after = np.sum(((trial_pred-y[:, None]) @ L.T)**2, axis=-1)
        accepted = after <= before
        z[accepted] = trial[accepted]
        damping = np.where(accepted, np.maximum(damping/3, 1e-12), np.minimum(damping*10, 1e12))
    pred, J = m2_model.forward_jac_m2(z[..., 0], z[..., 1], coef)
    error = (pred-y[:, None]) @ L.T
    JW = np.einsum('ab,nkbc->nkac', L, J)
    g = np.einsum('nkac,nka->nkc', JW, error)
    stationarity = np.max(np.abs(z-np.clip(z-g, lower, upper)), axis=-1)
    costs = np.sum(error**2, axis=-1)
    best = costs.min(axis=1)
    near = costs <= best[:, None]+1e-7
    n, k_count = costs.shape
    row = np.arange(n)
    order = np.lexsort((z[..., 0], z[..., 1]), axis=1)
    ordered = z[row[:, None], order]
    ordered_near = near[row[:, None], order]
    unique = np.zeros((n, k_count), dtype=bool)
    scale = np.array([1., .25])
    for rank in range(k_count):
        candidate = ordered[:, rank]
        close = np.linalg.norm((candidate[:, None, :] - z) / scale, axis=2) < 1e-4
        duplicate = np.any(close & unique, axis=1)
        unique[row, order[:, rank]] = ordered_near[:, rank] & ~duplicate
    count = unique.sum(axis=1)
    chosen_rank = np.argmax(unique[row[:, None], order], axis=1)
    chosen = order[row, chosen_rank]
    ordinal = np.cumsum(unique[row[:, None], order], axis=1)
    second_rank = np.argmax(ordinal == 2, axis=1)
    has_second = count > 1
    first_state = ordered[row, chosen_rank]
    second_state = ordered[row, second_rank]
    delta = second_state - first_state
    distance = np.where(has_second, np.linalg.norm(delta / scale, axis=1), np.nan)
    second_theta_delta = np.where(has_second, delta[:, 0], np.nan)
    second_A_delta = np.where(has_second, delta[:, 1], np.nan)
    theta_range = np.max(np.where(unique[row[:, None], order], ordered[..., 0], -np.inf), axis=1) - np.min(
        np.where(unique[row[:, None], order], ordered[..., 0], np.inf), axis=1)
    A_range = np.max(np.where(unique[row[:, None], order], ordered[..., 1], -np.inf), axis=1) - np.min(
        np.where(unique[row[:, None], order], ordered[..., 1], np.inf), axis=1)
    return z[row, chosen], dict(weighted_cost=costs[row, chosen], minimum_candidate_cost=best,
        equivalent_minima_count=count, second_branch_distance=distance,
        second_branch_theta_delta_deg=second_theta_delta, second_branch_A_delta_D=second_A_delta,
        equivalent_theta_range_deg=theta_range, equivalent_A_range_D=A_range,
        segment_projected_stationarity=stationarity[row, chosen], candidate_cost_max=costs.max(axis=1))


def train(args):
    y, frames, groups, meta, records, provenance, rejected = _report_and_data(
        args.experiment_dir, args.intervals, args.reject_isolated_jumps,
        args.jump_window, args.jump_threshold, args.edge_fraction,
        args.jump_burst_window, args.jump_max_burst_frames,
        args.jump_min_d, args.jump_min_rho, args.jump_refractory_frames)
    problem, initial, selected, held, local_groups, covariance, detail = _training_data(y, frames, groups, meta, args.fold)
    if args.initial_state_inverse:
        if args.fold != "full":
            raise ValueError("Cached full-data inverse initialization is allowed only for --fold full")
        initial, inverse_detail = _apply_cached_inverse_initial(
            problem, initial, y, frames, groups, meta,
            args.initial_state_inverse_csv, provenance)
        detail["state_initialization"] = inverse_detail
    elif args.initial_state_inverse_csv is not None:
        raise ValueError("--initial-state-inverse-csv requires --initial-state-inverse")
    provenance.update(fold=args.fold, training_fixations=selected, heldout_fixations=held,
                      heldout_fixation=None, training_frame_count=problem.n,
                      heldout_frame_count=int(np.isin(groups, held).sum()), initialization=detail)
    output = args.output_dir or HERE / "experiments" / args.fold / "training"
    _new_dir(output)
    atomic_json(output / "provenance.json", provenance)
    (output / "selected_intervals.json").write_bytes(args.intervals.read_bytes())
    if rejected:
        _csv(output / "rejected_observations.csv", rejected)
    x, final_coef, final_history = initial, None, None
    stage_specs = [("quadratic", None), ("robust", args.kappa)]
    final_converged = False
    for stage, kappa in stage_specs:
        stage_dir = output / stage
        stage_dir.mkdir()
        manifest = make_manifest(problem, provenance, local_groups, kappa)
        manifest.update(experiment=provenance["experiment"], fold=args.fold,
                        observable_schema=OBSERVABLE_SCHEMA,
                        observable_description=OBSERVABLE_DESCRIPTION,
                        heldout_fixations=held, training_fixations=selected,
                        target_demand_labels=[{k: meta[j][k] for k in ("capture", "target_theta_deg", "demand_diopters_label")}
                                              for j in selected])
        x, final_coef, history, converged, status = continue_fit(
            problem, x, stage_dir, manifest, kappa=kappa,
            max_nfev=args.max_nfev, wall_seconds=args.wall_seconds,
            lsmr_maxiter=args.lsmr_maxiter, lsmr_atol=args.lsmr_atol,
            lsmr_btol=args.lsmr_btol, robust_outer=args.robust_outer,
            robust_max_nfev=args.robust_max_nfev, physical_gtol=args.physical_gtol,
            physical_step_tol=args.physical_step_tol,
            relative_cost_tol=args.relative_cost_tol)
        final_history = history
        final_converged = converged
        atomic_json(stage_dir / "status.json", status)
        # Keep each loss stage independently usable as a model artifact. The
        # top-level model below remains the final (robust) stage by default.
        stage_rmse = np.sqrt(np.mean(((problem.forward_matrix(x) @ final_coef)-problem.y)**2, axis=0))
        stage_model = dict(schema=m2_model.SCHEMA, model_type="m2",
                 coefficient_schema=m2_model.SCHEMA,
                 coefficient_order=m2_model.COEFFICIENT_ORDER,
                 coefficient_names=m2_model.COEFFICIENT_NAMES,
                 coefficients=final_coef.tolist(), initial_coefficients=problem.initial_coef.tolist(),
                 precision=problem.W.tolist(), covariance=covariance.tolist(),
                 theta_bounds=[-20, 20], A_bounds=[0, 6], p=None,
                 observable_schema=OBSERVABLE_SCHEMA,
                 observable_description=OBSERVABLE_DESCRIPTION,
                 basis_description=m2_model.BASIS_DESCRIPTION,
                 state_encoding="theta/15, A/4 (A in diopters)",
                 diagnostics=dict(converged=bool(converged), termination=status.get("termination"),
                    status=status, rmse=stage_rmse.tolist(), state_count=problem.n,
                    covariance=covariance.tolist(),
                    measurement_support="central-core rows retained by the selected observation filter within stored training fixation intervals",
                    validation="In-sample optical residuals and nominal mean anchors; not measured physiological accuracy",
                    provenance=provenance, history=history))
        atomic_json(stage_dir / "model.json", stage_model)
        atomic_json(stage_dir / "fit_summary.json", dict(fold=args.fold,
                    training_fixations=selected, heldout_fixations=held,
                    stage_status={stage: status}, model_schema=m2_model.SCHEMA,
                    observable_schema=OBSERVABLE_SCHEMA,
                    labels=[{k: meta[j][k] for k in ("capture", "target_theta_deg", "demand_diopters_label")}
                            for j in selected]))
        print(json.dumps(dict(stage=stage, converged=converged, status=status)), flush=True)
    model = dict(schema=m2_model.SCHEMA, model_type="m2",
                 coefficient_schema=m2_model.SCHEMA,
                 coefficient_order=m2_model.COEFFICIENT_ORDER,
                 coefficient_names=m2_model.COEFFICIENT_NAMES,
                 coefficients=final_coef.tolist(), initial_coefficients=problem.initial_coef.tolist(),
                 precision=problem.W.tolist(), covariance=covariance.tolist(),
                 theta_bounds=[-20, 20], A_bounds=[0, 6], p=None,
                 observable_schema=OBSERVABLE_SCHEMA,
                 observable_description=OBSERVABLE_DESCRIPTION,
                 basis_description=m2_model.BASIS_DESCRIPTION,
                 state_encoding="theta/15, A/4 (A in diopters)",
                 diagnostics=dict(converged=bool(final_converged),
                    termination=status.get("termination"), status=status,
                    rmse=np.sqrt(np.mean(((problem.forward_matrix(x) @ final_coef)-problem.y)**2, axis=0)).tolist(),
                    state_count=problem.n, covariance=covariance.tolist(),
                    measurement_support="central-core rows retained by the selected observation filter within stored training fixation intervals",
                    validation="In-sample optical residuals and nominal mean anchors; not measured physiological accuracy",
                    provenance=provenance, history=final_history))
    atomic_json(output / "model.json", model)
    th, _, A, _ = problem.decode(x)
    pred = problem.forward_matrix(x) @ final_coef
    train_rows = []
    record_by_capture = {r["capture"]: r for r in records}
    frame_row_by_capture = {r["capture"]: {int(f): i for i, f in enumerate(r["arrays"]["frame_index"])}
                            for r in records}
    for i in range(problem.n):
        j = selected[int(problem.groups[i])]
        capture = meta[j]["capture"]
        comp = record_by_capture[capture]["components"]
        rowidx = frame_row_by_capture[capture][int(problem.frames[i])]
        train_rows.append(dict(fixation_index=j, capture=meta[j]["capture"], frame_index=int(problem.frames[i]),
            d=float(problem.y[i, 0]), rho=float(problem.y[i, 1]), theta_deg=float(th[i]),
            A_diopters=float(A[i]), predicted_d=float(pred[i, 0]), predicted_rho=float(pred[i, 1]),
            area_p1_px2=float(comp["area_p1_px2"][rowidx]), area_p4_px2=float(comp["area_p4_px2"][rowidx]),
            sqrt_area_p1_px=float(comp["sqrt_area_p1_px"][rowidx]),
            centroid_dx_px=float(comp["centroid_dx_px"][rowidx])))
    _csv(output / "training_states.csv", train_rows)
    atomic_json(output / "fit_summary.json", dict(fold=args.fold, training_fixations=selected,
        heldout_fixations=held, stage_status={s: json.loads((output/s/"status.json").read_text()) for s, _ in stage_specs},
        model_schema=m2_model.SCHEMA, observable_schema=OBSERVABLE_SCHEMA,
        labels=[{k: meta[j][k] for k in ("capture", "target_theta_deg", "demand_diopters_label")} for j in range(len(meta))]))


def estimate(args):
    model = json.loads((args.model_dir / "model.json").read_text())
    if model.get("schema") != m2_model.SCHEMA or model.get("observable_schema") != OBSERVABLE_SCHEMA:
        raise ValueError("Model schema does not match exp5 triangle M2 observations")
    provenance = model["diagnostics"]["provenance"]
    if provenance.get("fold") != args.fold:
        raise ValueError("Requested fold differs from trained model")
    y, frames, groups, meta, records, current, rejected = _report_and_data(
        args.experiment_dir, args.intervals, args.reject_isolated_jumps,
        args.jump_window, args.jump_threshold, args.edge_fraction,
        args.jump_burst_window, args.jump_max_burst_frames,
        args.jump_min_d, args.jump_min_rho, args.jump_refractory_frames)
    for key in ("source_sha256", "selected_interval_sha256", "observable_schema"):
        if current.get(key) != provenance.get(key):
            raise ValueError(f"Estimation {key} differs from training provenance")
    if current.get("observation_filter") != provenance.get("observation_filter"):
        raise ValueError("Estimation observation filter differs from training provenance")
    if current.get("edge_fraction") != provenance.get("edge_fraction"):
        raise ValueError("Estimation edge fraction differs from training provenance")
    coef, W = np.asarray(model["coefficients"], float), np.asarray(model["precision"], float)
    if (coef.shape != (m2_model.N_COEF,) or W.shape != (2, 2)
            or not np.isfinite(coef).all() or not np.isfinite(W).all()
            or not np.allclose(W, W.T)):
        raise ValueError("Model coefficients or observable precision have invalid dimensions/values")
    np.linalg.cholesky(W)
    held = set(provenance["heldout_fixations"])
    selected_frames = np.isin(groups, list(held if args.fold != "full" else provenance["training_fixations"]))
    if args.fold == "full":
        selected_frames = groups >= 0
    output = args.output_dir or HERE / "experiments" / args.fold / "estimates"
    _new_dir(output)
    if rejected:
        _csv(output / "rejected_observations.csv", rejected)
    training_converged = bool(model["diagnostics"].get("converged", False))
    training_termination = model["diagnostics"].get("termination", "unknown")
    print(json.dumps(dict(training_converged=training_converged,
                          training_termination=training_termination)), flush=True)
    prediction_rows = []
    iy, iframe, igroup = y[selected_frames], frames[selected_frames], groups[selected_frames]
    record_by_capture = {r["capture"]: r for r in records}
    frame_row_by_capture = {r["capture"]: {int(f): i for i, f in enumerate(r["arrays"]["frame_index"])}
                            for r in records}
    # Independent per-frame inversion; labels and neighboring frames are not inputs.
    for start in range(0, len(iy), args.batch_size):
        stop = min(start + args.batch_size, len(iy))
        states, diagnostics = _invert_batch_m2_fast(iy[start:stop], coef, W, args.inverse_iterations)
        predicted, physical = m2_model.state_diagnostics_m2(states, iy[start:stop], coef, W)
        for k, state in enumerate(states):
            i = start + k; global_fix = int(igroup[i]); row_meta = meta[global_fix]
            rec = record_by_capture[row_meta["capture"]]
            rowidx = frame_row_by_capture[row_meta["capture"]][int(iframe[i])]
            comp = rec["components"]
            one = dict(fixation_index=global_fix, capture=row_meta["capture"], frame_index=int(iframe[i]),
                theta_deg=float(state[0]), A_diopters=float(state[1]), d=float(iy[i, 0]), rho=float(iy[i, 1]),
                predicted_d=float(predicted[k, 0]), predicted_rho=float(predicted[k, 1]),
                residual_d=float(predicted[k, 0]-iy[i, 0]), residual_rho=float(predicted[k, 1]-iy[i, 1]),
                area_p1_px2=float(comp["area_p1_px2"][rowidx]), area_p4_px2=float(comp["area_p4_px2"][rowidx]),
                sqrt_area_p1_px=float(comp["sqrt_area_p1_px"][rowidx]),
                centroid_dx_px=float(comp["centroid_dx_px"][rowidx]),
                target_theta_deg=float(row_meta["target_theta_deg"]),
                demand_diopters_label=float(row_meta["demand_diopters_label"]))
            for source in (diagnostics, physical):
                one.update({key: (bool(val[k]) if val.dtype.kind == "b" else int(val[k]) if val.dtype.kind in "iu" else float(val[k]))
                            for key, val in source.items()})
            prediction_rows.append(one)
        print(json.dumps(dict(completed_frames=stop, total=len(iy))), flush=True)
    _csv(output / "frame_estimates.csv", prediction_rows)
    # Per-recording table covers every original row and makes invalid/freeview
    # support visible; outside the selected fixation groups, no inversion is run.
    by_key = {(r["fixation_index"], r["frame_index"]): r for r in prediction_rows}
    for rec in records:
        rows = []
        frame = rec["arrays"]["frame_index"]
        time = rec["arrays"]["timestamp_ms"]
        rejected_frames = {d["frame_index"] for d in rejected if d["capture"] == rec["capture"]}
        for i, f in enumerate(frame):
            g = int(rec["groups"][i]); match = by_key.get((g, int(f))) if g >= 0 else None
            row = dict(frame_index=int(f), timestamp_ms=float(time[i]), fixation_index=g,
                       detection_valid=bool(rec["valid"][i]), fit_valid=bool(rec["fit_valid"][i]),
                       status="invalid_detection" if not rec["valid"][i] else
                       "excluded_fixation_edge" if rec["full"][i] and not rec["core"][i] else
                       "isolated_observation_jump" if int(f) in rejected_frames else
                       "fixation_estimated" if match else "freeview_or_unselected")
            if match:
                row.update(match)
            elif rec["valid"][i]:
                row.update(d=float(rec["y"][i, 0]), rho=float(rec["y"][i, 1]),
                    area_p1_px2=float(rec["components"]["area_p1_px2"][i]),
                    area_p4_px2=float(rec["components"]["area_p4_px2"][i]),
                    sqrt_area_p1_px=float(rec["components"]["sqrt_area_p1_px"][i]),
                    centroid_dx_px=float(rec["components"]["centroid_dx_px"][i]))
            rows.append(row)
        _csv(output / (Path(rec["capture"]).stem + "_states.csv"), rows)
    atomic_json(output / "prediction_manifest.json", dict(experiment=provenance["experiment"],
        fold=args.fold, model_schema=model["schema"], observable_schema=OBSERVABLE_SCHEMA,
        model_sha256=hashlib.sha256((args.model_dir / "model.json").read_bytes()).hexdigest(),
        interval_sha256=current["selected_interval_sha256"], source_sha256=current["source_sha256"],
        edge_fraction=current["edge_fraction"], observation_filter=current["observation_filter"],
        frame_count=len(prediction_rows), batch_size=args.batch_size,
        inverse_iterations=args.inverse_iterations,
        training_converged=training_converged,
        training_termination=training_termination,
        inverse_implementation="exp5 Jacobian-free trial prediction and vectorized branch diagnostics; same starts, iterations, convergence path, bounds, and tie rule as exp2 inverter",
        inference="independent bounded per-frame inversion; no nominal state or temporal inputs",
        branch_rule="M2 multistart bounded inverse; smallest accommodation then gaze among equivalent minima"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    tr = sub.add_parser("train", help="fit M2 coefficients and joint fixation states")
    es = sub.add_parser("estimate", help="independently invert observations using a trained model")
    for cmd in (tr, es):
        cmd.add_argument("--fold", choices=list(FOLDS), default="full")
        cmd.add_argument("--experiment-dir", type=Path, default=HERE / "data/detections")
        cmd.add_argument("--intervals", type=Path, default=HERE / "data/fixations/fixation_intervals.json")
        cmd.add_argument("--output-dir", type=Path)
    tr.add_argument("--max-nfev", type=int, default=300)
    tr.add_argument("--wall-seconds", type=float, default=600.)
    tr.add_argument("--lsmr-maxiter", type=int, default=100)
    tr.add_argument("--lsmr-atol", type=float, default=1e-7)
    tr.add_argument("--lsmr-btol", type=float, default=1e-7)
    tr.add_argument("--robust-outer", type=int, default=12)
    tr.add_argument("--robust-max-nfev", type=int, default=25)
    tr.add_argument("--physical-gtol", type=float, default=1e-5)
    tr.add_argument("--physical-step-tol", type=float, default=1e-5)
    tr.add_argument("--relative-cost-tol", type=float, default=1e-10)
    tr.add_argument("--kappa", type=float, default=2.)
    tr.add_argument("--initial-state-inverse", action="store_true",
                    help="initialize latent states from a verified cached independent inverse CSV")
    tr.add_argument("--initial-state-inverse-csv", type=Path,
                    help="cached frame_estimates.csv from the matching initializer baseline")
    es.add_argument("--model-dir", type=Path, required=True)
    es.add_argument("--batch-size", type=int, default=512)
    es.add_argument("--inverse-iterations", type=int, default=100)
    for cmd in (tr, es):
        cmd.add_argument("--reject-isolated-jumps", action="store_true",
                         help="reject one-frame d/rho excursions that immediately recover within a fixation")
        cmd.add_argument("--jump-window", type=int, default=12, help="local robust-scale window radius in frames")
        cmd.add_argument("--jump-threshold", type=float, default=8.0, help="jump residual threshold in robust noise scales")
        cmd.add_argument("--edge-fraction", type=float, default=0.1,
                         help="fraction of each interval's rows to exclude at each edge (exp2 default 0.1)")
        cmd.add_argument("--jump-burst-window", type=int, default=201,
                         help="robust baseline radius in original frames for jump-return burst detection")
        cmd.add_argument("--jump-max-burst-frames", type=int, default=150,
                         help="maximum original-frame span for a recoverable observation burst")
        cmd.add_argument("--jump-min-d", type=float, default=0.02,
                         help="minimum absolute d jump at both burst boundaries")
        cmd.add_argument("--jump-min-rho", type=float, default=0.005,
                         help="minimum absolute rho jump at both burst boundaries")
        cmd.add_argument("--jump-refractory-frames", type=int, default=50,
                         help="skip this many frames after a recovered burst to avoid chaining events")
    args = p.parse_args()
    for field in ("max_nfev", "wall_seconds", "lsmr_maxiter", "robust_outer", "robust_max_nfev", "kappa", "batch_size", "inverse_iterations"):
        if hasattr(args, field) and getattr(args, field) <= 0:
            p.error(f"{field.replace('_', '-')} must be positive")
    if (args.jump_window < 2 or args.jump_threshold <= 0 or not 0 <= args.edge_fraction < 0.5
            or args.jump_burst_window < 2 or args.jump_max_burst_frames < 1
            or args.jump_min_d <= 0 or args.jump_min_rho <= 0 or args.jump_refractory_frames < 0):
        p.error("invalid jump windows/threshold or edge-fraction; require edge-fraction in [0, 0.5)")
    if args.command == "train" and args.initial_state_inverse and args.initial_state_inverse_csv is None:
        p.error("--initial-state-inverse requires --initial-state-inverse-csv")
    globals()[args.command](args)


if __name__ == "__main__":
    main()
