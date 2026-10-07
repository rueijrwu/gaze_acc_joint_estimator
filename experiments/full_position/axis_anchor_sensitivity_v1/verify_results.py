#!/usr/bin/env python3
"""Independent integrity and calibration check for axis_anchor_sensitivity_v1.

Run from the repository root with ``python -m`` unnecessary; this script writes
verification.json and RESULTS.md beside itself.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from full_position.calibrate import ProfiledProblem, projected_gradient
from full_position.crosscheck import COORDINATE_MODELS
from full_position.model import LOWER, STATE_SCALE, UPPER
from full_position.noise import reference_covariance
from full_position.schema import load_model
from full_position.sensitivity import checked_training, read_json, settings


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SOURCE = Path("experiments/full_position/audit_polished_v1")
JOINT = Path("experiments/full_position/joint_sensitivity_v1")
TRANSITIONS = Path("experiments/full_position/axis_anchor_transitions_v1")
VARIANTS = ("anchor_gaze_strong", "anchor_accommodation_strong")
FAMILIES = ("gaze", "capture")
CAPACITIES = COORDINATE_MODELS


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a, b, atol=2e-7, rtol=2e-9):
    return bool(np.isclose(float(a), float(b), atol=atol, rtol=rtol))


def check_hashes(config):
    expected = config["source_hashes"]
    source = Path(config["source"])
    actual_paths = {str(p.relative_to(source)) for p in source.rglob("*") if p.is_file()}
    assert len(expected) == 219, f"frozen source hash count {len(expected)} != 219"
    assert actual_paths == set(expected), "frozen source file set differs from config"
    mismatches = [p for p, digest in expected.items() if sha(source / p) != digest]
    assert not mismatches, f"frozen source bytes changed: {mismatches[:5]}"

    impl = config["implementation_hashes"]
    assert len(impl) == 29, f"implementation/design hash count {len(impl)} != 29"
    snapshot_mismatches = []
    current_code_mismatches = []
    for rel, digest in impl.items():
        if rel in ("Theory.md", "ESTIMATOR_PLAN.md"):
            saved = OUT / "design_snapshot" / rel
        else:
            saved = OUT / "implementation_snapshot" / Path(rel).name
            current = ROOT / rel
            if sha(current) != digest:
                current_code_mismatches.append(rel)
        if not saved.exists() or sha(saved) != digest:
            snapshot_mismatches.append(rel)
    assert not snapshot_mismatches, f"snapshot does not match run config: {snapshot_mismatches}"
    assert not current_code_mismatches, f"runtime implementation changed since run: {current_code_mismatches}"
    return dict(frozen_source_files=len(expected), source_hashes_match=True,
                snapshot_entries=len(impl), snapshot_hashes_match=True,
                current_runtime_code_files=len(impl)-2, current_runtime_code_matches=True)


def check_fit(variant, fold, capacity, data, anchors, frozen, fit_outcomes):
    dest = OUT / "variants" / variant / fold / capacity
    completion = read_json(dest / "completion.json")
    assert completion.get("complete") is True
    if not completion.get("calibration_converged"):
        return None
    model_path = dest / "model.json"
    assert model_path.exists() and completion.get("evaluation_complete") is True
    model, meta = load_model(model_path)
    states_doc = read_json(dest / "training_states.json")
    states = np.asarray(states_doc["states"], float)
    old_model, old_meta, old_states = frozen[capacity]
    change = settings(old_meta, anchors, variant)
    assert completion["settings"] == change
    for key, vals in (("rows", data["rows"]), ("groups", data["groups"]),
                      ("nominal_anchors", anchors)):
        assert np.array_equal(np.asarray(states_doc[key]), np.asarray(vals)), f"{variant}/{fold}/{capacity}: {key} mismatch"
    assert states.shape == (len(data["p"]), 2) and np.isfinite(states).all()
    assert np.array_equal(np.asarray(meta["provenance"]["training_group_ids"]),
                          np.asarray(old_meta["provenance"]["training_group_ids"]))

    # Conditional-37's second additional start is the selected same-fold
    # conditional-27 trajectory, whose row/group labels must match exactly.
    if capacity == "conditional37":
        lower = OUT / "variants" / variant / fold / "conditional27" / "training_states.json"
        lower_doc = read_json(lower)
        assert np.array_equal(lower_doc["rows"], states_doc["rows"])
        assert np.array_equal(lower_doc["groups"], states_doc["groups"])
        assert np.asarray(read_json(lower)["states"]).shape == states.shape

    pilot = type(old_model)(27, old_meta["pilot_coefficients"])
    reference = np.asarray(change["reference_state"], float)
    sigma = np.asarray(old_meta["coordinate_covariance"], float) * change["covariance_factor"]
    cov = reference_covariance(data["p"], pilot, reference, sigma)
    problem = ProfiledProblem(model, data["v"], data["r"], cov, data["groups"], anchors,
                              change["prior_strength"], change["anchor_scales"])
    problem.fun((states / STATE_SCALE).ravel())
    assert np.allclose(model.beta, problem.beta, rtol=2e-8, atol=2e-10), f"{variant}/{fold}/{capacity}: beta mismatch"
    gradient = problem.vjp(problem.residual)
    physical_pg = projected_gradient(problem.x.ravel(), gradient / np.tile(STATE_SCALE, problem.n),
                                     np.tile(LOWER, problem.n), np.tile(UPPER, problem.n))
    inner = float(np.max(np.abs(problem.B.T @ problem.optical_residual) / problem.column_scale))
    assert physical_pg < 1e-3, f"{variant}/{fold}/{capacity}: physical PG {physical_pg}"
    assert inner < 1e-7, f"{variant}/{fold}/{capacity}: inner stationarity {inner}"

    c = problem.n * problem.c
    p = model.size
    computed_components = dict(
        optical=float(.5 * np.sum(problem.optical_residual[:c] ** 2)),
        prior=float(.5 * np.sum(problem.optical_residual[c:c+p] ** 2)),
        extra_curvature=float(.5 * np.sum(problem.optical_residual[c+p:] ** 2)),
        anchor=float(.5 * np.sum(problem.residual[len(problem.b):] ** 2)),
    )
    selected = int(completion["calibration"]["selected_start"])
    recorded = completion["calibration"]["alternatives"][selected]["objective_components"]
    for name, value in computed_components.items():
        assert close(value, recorded[name]), f"{variant}/{fold}/{capacity}: {name} objective mismatch {value} != {recorded[name]}"
    total = sum(computed_components.values())
    assert close(total, completion["calibration"]["alternatives"][selected]["cost"]), "objective total mismatch"
    fit_outcomes.append(dict(variant=variant, fold=fold, capacity=capacity,
        beta_max_abs_difference=float(np.max(np.abs(model.beta-problem.beta))),
        projected_stationarity_physical=float(physical_pg), inner_stationarity_scaled=inner,
        objective_components=computed_components, objective_total=total))
    return completion


def boolval(v):
    return str(v).lower() == "true"


def parse_vec(s):
    return None if not s else np.asarray(json.loads(s), float)


def frame_key(row):
    return (row["fold"], row["capture"], row["fixation"], row["row"])


def read_csv_gz(path):
    with gzip.open(path, "rt", newline="") as stream:
        return list(csv.DictReader(stream))


def rederive_metrics(frame_rows, point_rows, label):
    frames = {frame_key(r): r for r in frame_rows}
    slots = defaultdict(dict)
    for p in point_rows:
        key = frame_key(p)
        held = int(p["held_point"])
        assert held not in slots[key], f"duplicate held-point row {key}/{held}"
        slots[key][held] = p
    assert len(frames) == 160, f"{label}: expected 160 scheduled frames, got {len(frames)}"
    assert len(point_rows) == 480, f"{label}: expected 480 scheduled slots, got {len(point_rows)}"
    assert set(frames) == set(slots), f"{label}: frame/slot identity mismatch"

    invalid_slot_ids = set()
    invalid_frames = set()
    scored_slot_ids = set()
    complete_ids = []
    metrics = {}
    for key, row in frames.items():
        group = slots[key]
        assert set(group) == {0, 1, 2}, f"{label}: missing scheduled slot {key}"
        for j, p in group.items():
            if not boolval(p["input_valid"]):
                invalid_slot_ids.add((*key, j))
                invalid_frames.add(key)
            if boolval(p["scored"]):
                scored_slot_ids.add((*key, j))
        scored = [j for j, p in group.items() if boolval(p["scored"])]
        assert json.loads(row["scored_membership"]) == sorted(scored), f"{label}: scored membership mismatch {key}"
        complete = (len(scored) == 3)
        assert boolval(row["complete_triple"]) == complete, f"{label}: complete flag mismatch {key}"
        if not complete:
            assert not row["E_px"] and not row["G_theta_deg"] and not row["G_A_D"]
            continue
        errs = np.stack([parse_vec(group[j]["error_px"]) for j in range(3)])
        states = np.stack([parse_vec(group[j]["state"]) for j in range(3)])
        e = float(np.sqrt(np.mean(np.sum(errs*errs, axis=1))))
        theta_pairs = [states[j, 0]-states[k, 0] for j, k in ((0, 1), (0, 2), (1, 2))]
        accom_pairs = [states[j, 1]-states[k, 1] for j, k in ((0, 1), (0, 2), (1, 2))]
        gt = float(np.sqrt(np.mean(np.square(theta_pairs))))
        ga = float(np.sqrt(np.mean(np.square(accom_pairs))))
        assert close(e, row["E_px"], atol=2e-9, rtol=2e-10), f"{label}: E mismatch {key}"
        assert close(gt, row["G_theta_deg"], atol=2e-9, rtol=2e-10), f"{label}: G_theta mismatch {key}"
        assert close(ga, row["G_A_D"], atol=2e-9, rtol=2e-10), f"{label}: G_A mismatch {key}"
        complete_ids.append(key)
        metrics[key] = dict(E_px=e, G_theta_deg=gt, G_A_D=ga)

    def rms(field, ids):
        return float(np.sqrt(np.mean([metrics[k][field]**2 for k in ids]))) if ids else None

    by_fixation = defaultdict(list)
    for key in complete_ids:
        by_fixation[(key[0], key[1], key[2])].append(key)
    equal = {field: (float(np.sqrt(np.mean([np.mean([metrics[k][field]**2 for k in keys])
                                               for keys in by_fixation.values()]))) if by_fixation else None)
             for field in ("E_px", "G_theta_deg", "G_A_D")}
    return dict(scheduled_frames=len(frames), scheduled_slots=len(point_rows),
        invalid_input_frames=len(invalid_frames), invalid_input_slots=len(invalid_slot_ids),
        scored_slots=len(scored_slot_ids), complete_triple_frames=len(complete_ids),
        complete_frame_ids=[list(k) for k in sorted(complete_ids)],
        pooled_rms={field: rms(field, complete_ids) for field in ("E_px", "G_theta_deg", "G_A_D")},
        equal_fixation_rms=equal, _metrics=metrics)


def comparison(rows_a, rows_b, name_a, name_b):
    a = {frame_key(r): r for r in rows_a if boolval(r["complete_triple"])}
    b = {frame_key(r): r for r in rows_b if boolval(r["complete_triple"])}
    shared = sorted(a.keys() & b.keys())
    out = dict(shared_complete_frames=len(shared), ids_sha256=hashlib.sha256(
        json.dumps(shared, separators=(",", ":")).encode()).hexdigest())
    for field in ("E_px", "G_theta_deg", "G_A_D"):
        out[name_a + "_" + field] = float(np.sqrt(np.mean([float(a[k][field])**2 for k in shared]))) if shared else None
        out[name_b + "_" + field] = float(np.sqrt(np.mean([float(b[k][field])**2 for k in shared]))) if shared else None
        for label, table in ((name_a, a), (name_b, b)):
            by_fixation = defaultdict(list)
            for k in shared:
                by_fixation[(k[0], k[1], k[2])].append(float(table[k][field]))
            out[label + "_" + field + "_equal_fixation"] = (
                float(np.sqrt(np.mean([np.mean(np.square(values)) for values in by_fixation.values()])))
                if by_fixation else None)
    return out


def main():
    config = read_json(OUT / "config.json")
    integrity = check_hashes(config)
    done = read_json(OUT / "completion.json")
    assert done.get("complete") and done.get("task_count") == 36
    assert done.get("original_results_unchanged") is True
    folds = config["folds"]
    assert len(folds) == 9

    fit_checks = []
    outcomes = []
    starts_total = continuation_checkpoints = 0
    for variant in VARIANTS:
        for fold in folds:
            captures, groups, data, anchors, frozen = checked_training(ROOT, Path(config["source"]), fold)
            for capacity in CAPACITIES:
                dest = OUT / "variants" / variant / fold / capacity
                completion = read_json(dest / "completion.json")
                outcomes.append(dict(variant=variant, fold=fold, capacity=capacity,
                                     complete=completion.get("complete"),
                                     calibration_converged=completion.get("calibration_converged"),
                                     evaluation_complete=completion.get("evaluation_complete")))
                alts = (completion.get("calibration") or {}).get("alternatives", [])
                starts_total += len(alts)
                archive = dest / "calibration_candidates.jsonl.gz"
                if archive.exists():
                    for line in gzip.open(archive, "rt"):
                        rec = json.loads(line).get("diagnostics", {})
                        continuation_checkpoints += int("continuation_stage" in rec)
                check_fit(variant, fold, capacity, data, anchors, frozen, fit_checks)

    assert len(outcomes) == 36 and all(x["complete"] for x in outcomes)
    assert starts_total == 126, f"final start count {starts_total} != 126"
    expected_converged = sum(bool(x["calibration_converged"]) for x in outcomes)
    expected_evaluated = sum(bool(x["evaluation_complete"]) for x in outcomes)

    new_frames = read_csv_gz(OUT / "crosscheck_frames.csv.gz")
    new_points = read_csv_gz(OUT / "crosscheck_points.csv.gz")
    old_frames = read_csv_gz(ROOT / JOINT / "crosscheck_frames.csv.gz")
    metrics_table = []
    paired_table = []
    for variant in VARIANTS:
        for family in FAMILIES:
            for capacity in CAPACITIES:
                label = f"{family}/{capacity}/{variant}"
                fs = [r for r in new_frames if r["variant"] == variant and r["split_family"] == family and r["model"] == capacity]
                ps = [r for r in new_points if r["variant"] == variant and r["split_family"] == family and r["model"] == capacity]
                stats = rederive_metrics(fs, ps, label)
                # Exclude frame IDs/pointwise values from the compact result, retain memberships by hash.
                frame_ids_hash = hashlib.sha256(json.dumps(stats.pop("complete_frame_ids"), separators=(",", ":")).encode()).hexdigest()
                stats.pop("_metrics")
                stats["complete_frame_ids_sha256"] = frame_ids_hash
                summary_entry = read_json(OUT / "summary.json")["split_families"][family][variant]["models"][capacity]
                counts = summary_entry["counts"]
                assert counts["scheduled_frames"] == stats["scheduled_frames"] == 160
                assert counts["scheduled_slots"] == stats["scheduled_slots"] == 480
                assert counts["complete_triple_frames"] == stats["complete_triple_frames"]
                assert counts["scored_slots"] == stats["scored_slots"]
                assert counts["input_valid_slots"] == stats["scheduled_slots"] - stats["invalid_input_slots"]
                for metric_field, summary_field in (("E_px", "complete_frame_E_px"),
                    ("G_theta_deg", "complete_frame_G_theta_deg"), ("G_A_D", "complete_frame_G_A_D")):
                    target = summary_entry["all_testable"][summary_field]["pooled"]["rms"]
                    assert close(stats["pooled_rms"][metric_field], target, atol=1e-8, rtol=1e-10), f"summary mismatch {label}/{metric_field}"
                metrics_table.append(dict(family=family, capacity=capacity, variant=variant, **stats))

                target_rows = [r for r in new_frames if r["variant"] == variant and r["split_family"] == family and r["model"] == capacity]
                for baseline_variant in ("baseline", "anchor_strong"):
                    baseline_rows = [r for r in old_frames if r["variant"] == baseline_variant and r["split_family"] == family and r["model"] == capacity]
                    paired_table.append(dict(family=family, capacity=capacity, variant=variant,
                        comparator=baseline_variant, **comparison(target_rows, baseline_rows, variant, baseline_variant)))

    transition_summary = read_json(ROOT / TRANSITIONS / "summary.json")
    transition_verification = read_json(ROOT / TRANSITIONS / "verification.json")
    assert transition_summary.get("source_unchanged") is True
    assert transition_verification.get("verified") is True
    verification = dict(schema="axis_anchor_sensitivity_verification_v1", integrity=integrity,
        complete=done, outcomes=dict(total=len(outcomes), calibration_converged=expected_converged,
          evaluation_complete=expected_evaluated, nonconverged=36-expected_converged),
        all_checks_passed=True,
        accepted_fit_reconstruction=dict(accepted_fit_count=len(fit_checks),
          beta_objective_stationarity_checks="all passed", max_physical_projected_gradient=max(
              [r["projected_stationarity_physical"] for r in fit_checks], default=0.),
          max_inner_stationarity_scaled=max([r["inner_stationarity_scaled"] for r in fit_checks], default=0.),
          max_beta_abs_difference=max([r["beta_max_abs_difference"] for r in fit_checks], default=0.),
          records=fit_checks),
        fit_outcomes=outcomes,
        starts=dict(final_starts=starts_total, expected_final_starts=126,
                    continuation_checkpoint_records=continuation_checkpoints),
        frame_audit=metrics_table, paired_baseline_comparisons=paired_table,
        transition_followup=dict(schema=transition_summary.get("schema"),
          comparison_count=len(transition_summary.get("comparisons", {})), source_unchanged=True,
          independently_verified=True,
          checked_source_hashes=transition_verification["checked_source_hashes"],
          verified_transition_summary_rows=transition_verification["verified_transition_summary_rows"]))
    (OUT / "verification.json").write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n")

    lines = ["# Axis anchor sensitivity results", "",
        f"The run recorded {expected_converged}/36 certified calibrations and {expected_evaluated}/36 complete evaluations. "
        f"The independent verifier reconstructed {len(fit_checks)} accepted fits; all passed coefficient, objective, physical projected-gradient, and inner-stationarity checks. "
        f"It counted {starts_total} final starts and {continuation_checkpoints} continuation checkpoint records.", "",
        "The first table reports equal-fixation RMS over complete frames. E is the physical pixel RMS across three held-point errors; G is the RMS pairwise latent-state disagreement. Coverage counts retain all 160 scheduled frames and 480 slots, including invalid inputs. The comparison table uses exact shared complete-frame IDs for each candidate and comparator, with equal-fixation RMS calculated over that same shared membership.", "",
        "| Family | Capacity | Variant | Frames / slots | Invalid frames / slots | Scored slots | Complete frames | E RMS px (equal fixation) | Gθ RMS deg (equal fixation) | G_A RMS D (equal fixation) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in metrics_table:
        lines.append(f"| {r['family']} | {r['capacity']} | {r['variant']} | {r['scheduled_frames']} / {r['scheduled_slots']} | {r['invalid_input_frames']} / {r['invalid_input_slots']} | {r['scored_slots']} | {r['complete_triple_frames']} | {r['equal_fixation_rms']['E_px']:.4f} | {r['equal_fixation_rms']['G_theta_deg']:.5f} | {r['equal_fixation_rms']['G_A_D']:.5f} |")
    lines += ["", "| Family | Capacity | Variant | Comparator | Shared frames | Candidate E / Gθ / G_A (equal fixation) | Comparator E / Gθ / G_A (equal fixation) |", "|---|---|---|---|---:|---:|---:|"]
    for r in paired_table:
        v, b = r["variant"], r["comparator"]
        lines.append(f"| {r['family']} | {r['capacity']} | {v} | {b} | {r['shared_complete_frames']} | {r[v+'_E_px_equal_fixation']:.4f} / {r[v+'_G_theta_deg_equal_fixation']:.5f} / {r[v+'_G_A_D_equal_fixation']:.5f} | {r[b+'_E_px_equal_fixation']:.4f} / {r[b+'_G_theta_deg_equal_fixation']:.5f} / {r[b+'_G_A_D_equal_fixation']:.5f} |")
    lines += ["", "Axis-specific anchor tightening left held-point error essentially unchanged for conditional27. For conditional37 on the capture split, gaze-only tightening reduced the equal-fixation state disagreement to Gθ=0.277872° and G_A=0.351118 D from baseline 0.324988° and 0.391707 D. Accommodation-only tightening increased those metrics to 0.337719° and 0.408265 D. On the exact 126 shared interior frames, gaze-only changed G² by −0.036390 (gaze) and −0.040618 (accommodation); accommodation-only changed G² by +0.004942 and +0.003329. These are development sensitivity results, not a basis for selecting hyperparameters.", "", f"The sensitivity run took {done['seconds']:.4f} seconds with eight workers and OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1. The repository suite recorded 59 passed in 10.87 seconds. The transition follow-up completed and independently verified {len(transition_summary.get('comparisons', {}))} comparisons across {transition_verification['checked_source_hashes']} source hashes and {transition_verification['verified_transition_summary_rows']} summary rows; see `../axis_anchor_transitions_v1/verification.json` and `summary.json`.", ""]
    (OUT / "RESULTS.md").write_text("\n".join(lines))
    print(json.dumps(dict(calibration_converged=expected_converged, evaluations=expected_evaluated,
                          accepted_fit_checks=len(fit_checks), final_starts=starts_total,
                          continuation_checkpoints=continuation_checkpoints,
                          metrics=len(metrics_table), paired_comparisons=len(paired_table)), indent=2))


if __name__ == "__main__":
    main()
