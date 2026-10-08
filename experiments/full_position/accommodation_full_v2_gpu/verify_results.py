#!/usr/bin/env python3
"""One-pass mechanical verification of completed full-period calibration outputs.

Run from the repository root after the run's top-level completion.json exists.
This verifier refuses incomplete/open outputs and writes result_verification.json
only inside this experiment directory. It does not mutate inputs or summaries.
"""
from __future__ import annotations

from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
NAMES = ("ar27_log", "ar27_sqrt", "ar27_linear", "ar27_quadratic")
REFERENCE = "ar27_log"
METRICS = (("E_cross_px", "E_px"), ("G_theta_cross_deg", "G_theta_deg"),
           ("G_A_cross_D", "G_A_D"), ("worst_point_px", "worst_point_px"))
FRAME_ID = ("fold", "capture", "fixation", "row")
EXP_ID = ("fold", "capture", "fixation")


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest_json(rows):
    raw = json.dumps(rows, separators=(",", ":"), ensure_ascii=False,
                     allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def require(condition, errors, label):
    if not condition:
        errors.append(label)


def close(a, b, *, atol=2e-9, rtol=2e-8):
    if a is None or b is None:
        return a is None and b is None
    try:
        return math.isclose(float(a), float(b), rel_tol=rtol, abs_tol=atol)
    except (TypeError, ValueError, OverflowError):
        return False


def iter_json_array(path, chunk_size=1024 * 1024):
    """Stream a JSON array without retaining the expanded frame table."""
    decoder = json.JSONDecoder()
    with Path(path).open("r", encoding="utf-8") as stream:
        buf, pos, eof, opened = "", 0, False, False

        def refill():
            nonlocal buf, pos, eof
            if pos:
                buf, pos = buf[pos:], 0
            part = stream.read(chunk_size)
            if part:
                buf += part
            else:
                eof = True

        while True:
            while True:
                while pos < len(buf) and buf[pos].isspace():
                    pos += 1
                if pos < len(buf):
                    break
                if eof:
                    raise ValueError(f"unexpected EOF in JSON array: {path}")
                refill()
            if not opened:
                if buf[pos] != "[":
                    raise ValueError(f"expected JSON array: {path}")
                opened, pos = True, pos + 1
                continue
            if buf[pos] == "]":
                return
            if buf[pos] == ",":
                pos += 1
                continue
            try:
                item, end = decoder.raw_decode(buf, pos)
            except json.JSONDecodeError:
                if eof:
                    raise
                refill()
                continue
            pos = end
            yield item


def tuple_id(record, fields=FRAME_ID):
    values = []
    for key in fields:
        value = record[key]
        values.append(int(value) if key in ("fixation", "row") else value)
    return tuple(values)


def canonical_ids(ids):
    return [list(value) for value in sorted(ids)]


def check_equal_exposure(card, metric_name, field, rows, expected_exposures, errors, label):
    """Recreate scorecard equal-exposure means from saved frame metrics."""
    grouped = defaultdict(list)
    for identity, values in rows.items():
        value = values.get(field)
        if value is not None and math.isfinite(float(value)):
            grouped[identity[:3]].append(float(value) ** 2)
    present = set(grouped)
    missing = sorted(set(expected_exposures) - present)
    exposure_means = {exp: sum(vals) / len(vals) for exp, vals in grouped.items()}
    mean = (sum(exposure_means.values()) / len(exposure_means)) if exposure_means else None
    outcome = card.get("outcomes", {}).get(metric_name, {})
    agg = outcome.get("squared_error_aggregation", {})
    require(close(agg.get("mean"), mean), errors, f"{label}: {metric_name} squared mean mismatch")
    require(agg.get("contributing_exposure_count") == len(present), errors,
            f"{label}: {metric_name} contributing exposure count mismatch")
    require(agg.get("scheduled_exposure_count") == len(expected_exposures), errors,
            f"{label}: {metric_name} scheduled exposure count mismatch")
    stored_missing = {tuple(x) for x in agg.get("absent_exposure_ids", [])}
    require(stored_missing == set(missing), errors, f"{label}: {metric_name} absent exposure membership mismatch")
    stored_per = agg.get("per_exposure", {})
    expected_keys = {json.dumps(exp) for exp in present}
    require(set(stored_per) == expected_keys, errors, f"{label}: {metric_name} per-exposure membership mismatch")
    for exp, value in exposure_means.items():
        key = json.dumps(exp)
        require(stored_per.get(key, {}).get("count") == len(grouped[exp]), errors,
                f"{label}: {metric_name} per-exposure count mismatch for {exp}")
        require(close(stored_per.get(key, {}).get("mean"), value), errors,
                f"{label}: {metric_name} per-exposure mean mismatch for {exp}")
    require(close(outcome.get("equal_exposure_rms"), math.sqrt(mean) if mean is not None else None),
            errors, f"{label}: {metric_name} equal-exposure RMS mismatch")
    return mean, exposure_means


def main():
    completion_path = OUT / "completion.json"
    completion = read_json(completion_path) if completion_path.exists() else None
    if not completion or completion.get("complete") is not True:
        print("Refusing verification: top-level completion.json is absent or incomplete.", file=sys.stderr)
        return 2

    errors, checks = [], {}
    config = read_json(OUT / "config.json")
    summary = read_json(OUT / "summary.json")
    outcomes = read_json(OUT / "outcomes.json")
    require(config.get("training_window") == "fixation_period", errors, "config training window is not full fixation_period")
    require(config.get("agreement_window") == "fixation_period", errors, "config agreement window is not full fixation_period")
    require(config.get("calibration_rows") == 89175, errors, "config calibration row count differs from 89,175")
    require(config.get("scheduled_agreement_frames") == 100090, errors, "config scheduled frame count differs from 100,090")
    require(config.get("fresh_calibration_per_law") is True, errors, "fresh calibration guard missing")
    require(config.get("reused_previous_fitted_models") is False, errors, "previous model reuse guard failed")
    require(completion.get("planned_fits") == 4 and completion.get("completed_fit_tasks") == 4,
            errors, "top-level completion does not report all four terminal fit tasks")
    require(completion.get("calibration_rows") == 89175, errors, "top-level completion training count mismatch")
    require(completion.get("scheduled_agreement_frames_per_law") == 100090, errors,
            "top-level completion schedule count mismatch")
    require(summary.get("run_counts", {}).get("completed_fits") == 4, errors,
            "summary does not report four completed fits")
    require(len(outcomes) == 4, errors, "outcomes.json does not contain four terminal outcomes")
    outcome_by_name = {row.get("candidate"): row for row in outcomes}
    require(set(outcome_by_name) == set(NAMES), errors, "outcomes.json law set mismatch")
    fit_exceptions = []
    terminal_laws = []

    # Recheck launch-time input protection without touching source/output artifacts.
    def verify_hash_map(mapping, label):
        result = {}
        for rel, expected in mapping.items():
            path = ROOT / rel
            actual = sha256(path) if path.is_file() else None
            result[rel] = {"expected": expected, "actual": actual, "matches": actual == expected}
            if actual != expected:
                errors.append(f"{label} hash mismatch or missing file: {rel}")
        return result

    source_checks = verify_hash_map(config.get("source_hashes", {}), "source")
    protected_checks = verify_hash_map(config.get("protected_hashes", {}), "protected")
    require(bool(source_checks) and all(x["matches"] for x in source_checks.values()), errors,
            "frozen source hash guard failed")
    require(bool(protected_checks) and all(x["matches"] for x in protected_checks.values()), errors,
            "protected input hash guard failed")

    split = OUT / "splits" / "full_calibration"
    population = read_json(split / "population.json")
    manifest = read_json(split / "manifest.json")
    selected = [row for row in population if row.get("selected_for_evaluation") is True]
    expected_frames = {("full_calibration", row["capture"], int(row["fixation"]), int(row["row"])):
                       int(row["frame"]) for row in selected}
    manifest_frames = {tuple_id(row): int(row["source_frame_index"]) for row in manifest["frames"]}
    expected_ids = set(expected_frames)
    require(len(population) == 100090 and len(selected) == 100090, errors,
            "frozen population does not contain 100,090 selected full-period frames")
    require(len(expected_frames) == 100090, errors, "frozen population has duplicate selected frame identities")
    require(manifest.get("slots_per_frame") == 3, errors, "population manifest slots_per_frame is not three")
    require(len(manifest.get("expected_exposures", [])) == 20, errors,
            "frozen schedule does not contain twenty expected exposures")
    require(manifest_frames == expected_frames, errors,
            "population and immutable manifest frame/source-index membership differ")
    expected_exposures = {tuple(x) for x in manifest.get("expected_exposures", [])}
    require(expected_exposures == {identity[:3] for identity in expected_ids}, errors,
            "frozen schedule exposure membership differs from selected frame identities")

    # Training and per-law terminal artifacts distinguish completed failures from pending work.
    training = read_json(split / "training.json")
    provenance = training.get("provenance", {})
    train_rows = provenance.get("sampled_training_rows", [])
    train_groups = provenance.get("sampled_training_group", [])
    require(len(train_rows) == 89175 and len(train_groups) == 89175, errors,
            "saved training provenance does not enumerate 89,175 full-period rows")
    require(provenance.get("training_window") == "fixation_period", errors,
            "training provenance window mismatch")
    require(provenance.get("agreement_window") == "fixation_period", errors,
            "agreement provenance window mismatch")
    require(provenance.get("interval_sha256") == config.get("interval_sha256"), errors,
            "training interval hash differs from run config")
    training_inputs = split / "training_inputs.npz"
    require(provenance.get("shared_training_input_sha256") == sha256(training_inputs), errors,
            "saved training-input digest differs from training provenance")
    checks["schedule"] = {
        "scheduled_frames": len(expected_ids), "scheduled_slots_per_law": 3 * len(expected_ids),
        "selected_invalid_P4_frames": sum(any(not row[f"p4_{j}_valid"] for j in (1, 2, 3)) for row in selected),
        "selected_invalid_baseline_frames": sum(not row["baseline_valid"] for row in selected),
        "selected_invalid_geometry_frames": sum(not row["p1_valid_geometry"] for row in selected),
        "expected_exposures": [list(x) for x in sorted(expected_exposures)],
        "schedule_membership_sha256": digest_json(canonical_ids(expected_ids)),
        "manifest_sha256": sha256(split / "manifest.json"),
        "population_sha256": sha256(split / "population.json"),
    }

    frame_ids, complete_ids, scored_points = {}, {}, {}
    frame_values = {}
    per_law_checks = {}
    for name in NAMES:
        folder = OUT / "fits" / "full_calibration" / name
        law_completion_path = folder / "completion.json"
        law_completion = read_json(law_completion_path) if law_completion_path.exists() else None
        model_path, failed_path = folder / "model.json", folder / "failed_checkpoint.json"
        model = read_json(model_path) if model_path.exists() else None
        failed = read_json(failed_path) if failed_path.exists() else None
        accepted = model is not None
        law_terminal = (law_completion is not None and law_completion.get("complete") is True
                        and bool(model) != bool(failed))
        if law_terminal:
            terminal_laws.append(name)
        require(law_completion is not None and law_completion.get("complete") is True, errors,
                f"{name}: terminal completion missing")
        require(bool(model) != bool(failed), errors, f"{name}: expected exactly one model or failed checkpoint")
        require(law_completion is not None and law_completion.get("calibration_converged") is accepted,
                errors, f"{name}: completion/model certification mismatch")
        require(law_completion is not None and law_completion.get("scheduled_frames") == 100090,
                errors, f"{name}: completion scheduled frame count mismatch")
        require(law_completion is not None and law_completion.get("scheduled_slots") == 300270,
                errors, f"{name}: completion scheduled slot count mismatch")
        artifact = model if model is not None else failed
        calibration = (artifact or {}).get("calibration", {})
        require(calibration.get("converged") is accepted, errors,
                f"{name}: terminal calibration status mismatch")
        if model is not None:
            require(model.get("model") == name and len(model.get("coefficients", [])) == 27, errors,
                    f"{name}: model law or 27-coefficient schema mismatch")
        require(outcome_by_name.get(name, {}).get("converged") is accepted, errors,
                f"{name}: outcomes.json confuses certified and unfinished/failed")
        require(summary.get("candidates", {}).get(name, {}).get("certified") is accepted, errors,
                f"{name}: summary confuses certified and uncertified outcome")
        has_exception = (outcome_by_name.get(name, {}).get("exception") is not None
                         or (folder / "exception.txt").exists())
        if has_exception:
            fit_exceptions.append(name)
        artifact_provenance = (artifact or {}).get("provenance", {})
        require(artifact_provenance.get("interval_sha256") == config.get("interval_sha256"), errors,
                f"{name}: artifact interval hash mismatch")
        require(artifact_provenance.get("source_sha256") == config.get("source_hashes"), errors,
                f"{name}: artifact source provenance mismatch")
        require(artifact_provenance.get("training_window") == "fixation_period", errors,
                f"{name}: artifact training window mismatch")

        # Read candidate archives linewise. One record is written for each of three
        # held points for every selected frame, including invalid input frames.
        archive_path = folder / "inverse_candidates.jsonl.gz"
        line_count = 0
        with gzip.open(archive_path, "rt", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                line_count += 1
        # The archive is an audit trail of one task per selected frame/held point.
        # Count and CRC-check it without parsing its large multistart candidate payloads;
        # exact frame/slot identities are checked from the saved joined frames below.
        require(line_count == 300270, errors,
                f"{name}: inverse archive line count differs from 300,270 scheduled slots")

        frames_path = folder / "frames.json"
        law_frame_ids, law_complete, law_scored = set(), set(), set()
        law_metrics = {}
        exposure_sums = {metric: defaultdict(list) for metric, _ in METRICS}
        slot_counts = defaultdict(int)
        frame_count = 0
        for frame in iter_json_array(frames_path):
            frame_count += 1
            fid = tuple_id(frame)
            if fid in law_frame_ids:
                errors.append(f"{name}: duplicate saved frame {fid}")
            law_frame_ids.add(fid)
            require(fid in expected_frames, errors, f"{name}: unexpected saved frame {fid}")
            if fid in expected_frames:
                require(int(frame.get("source_frame_index", -1)) == expected_frames[fid], errors,
                        f"{name}: source frame index mismatch at {fid}")
            slots = frame.get("slots", [])
            require(len(slots) == 3 and {int(slot.get("held_point", -1)) for slot in slots} == {0, 1, 2},
                    errors, f"{name}: frame lacks exactly three held-point slots at {fid}")
            slot_by_point = {int(slot["held_point"]): slot for slot in slots}
            for held, slot in slot_by_point.items():
                require(tuple_id(slot) == fid, errors, f"{name}: slot/frame identity mismatch at {fid}/{held}")
                require(int(slot.get("source_frame_index", frame.get("source_frame_index", -1)))
                        == int(frame.get("source_frame_index", -2)), errors,
                        f"{name}: slot source-frame index mismatch at {fid}/{held}")
                slot_counts["scheduled"] += 1
                for key in ("input_valid", "available", "certified", "identifiable", "scored"):
                    slot_counts[key] += bool(slot.get(key))
                slot_counts["unambiguous"] += bool(slot.get("unambiguous") and slot.get("available")
                    and slot.get("certified") and slot.get("identifiable") and slot.get("testable"))
                slot_counts["bound"] += slot.get("bound") is True
                if slot.get("scored"):
                    law_scored.add(fid + (held,))
            scored = [slot_by_point[j] for j in range(3) if slot_by_point[j].get("scored")]
            complete = len(scored) == 3
            require(frame.get("complete_triple") is complete, errors,
                    f"{name}: complete_triple disagrees with scored slots at {fid}")
            if not complete:
                require(frame.get("E_px") is None and frame.get("G_theta_deg") is None
                        and frame.get("G_A_D") is None, errors,
                        f"{name}: incomplete frame has aggregate metrics at {fid}")
                continue

            law_complete.add(fid)
            errs = [slot_by_point[j].get("error_px") for j in range(3)]
            states = [slot_by_point[j].get("state") for j in range(3)]
            finite = (all(isinstance(x, list) and len(x) == 2 and all(math.isfinite(float(v)) for v in x)
                          for x in errs + states))
            require(finite, errors, f"{name}: complete frame has nonfinite state/error {fid}")
            if not finite:
                continue
            e2 = sum(float(v) ** 2 for err in errs for v in err) / 3.0
            pairs = ((0, 1), (0, 2), (1, 2))
            gt2 = sum((float(states[i][0]) - float(states[j][0])) ** 2 for i, j in pairs) / 3.0
            ga2 = sum((float(states[i][1]) - float(states[j][1])) ** 2 for i, j in pairs) / 3.0
            require(close(float(frame.get("E_px")) ** 2, e2), errors, f"{name}: E_px^2 arithmetic mismatch at {fid}")
            require(close(float(frame.get("G_theta_deg")) ** 2, gt2), errors,
                    f"{name}: G_theta_deg^2 arithmetic mismatch at {fid}")
            require(close(float(frame.get("G_A_D")) ** 2, ga2), errors,
                    f"{name}: G_A_D^2 arithmetic mismatch at {fid}")
            values = {"E_px": frame["E_px"], "G_theta_deg": frame["G_theta_deg"],
                      "G_A_D": frame["G_A_D"], "worst_point_px": frame.get("worst_point_px")}
            law_metrics[fid] = values
            for metric, value in values.items():
                if value is not None and math.isfinite(float(value)):
                    exposure_sums[metric][fid[:3]].append(float(value) ** 2)

        require(frame_count == 100090 and law_frame_ids == expected_ids, errors,
                f"{name}: frames.json does not match exact 100,090-frame schedule membership")
        require(len(law_complete) == len(law_metrics), errors, f"{name}: incomplete metric recomputation state")
        card = read_json(folder / "scorecard.json")
        expected_coverage = {
            "scheduled_frames": frame_count, "scheduled_slots": slot_counts["scheduled"],
            "input_eligible": slot_counts["input_valid"], "scored": slot_counts["scored"],
            "certified": slot_counts["certified"], "identifiable": slot_counts["identifiable"],
            "unambiguous": slot_counts["unambiguous"], "bound_slots": slot_counts["bound"],
            "complete_triples": len(law_complete),
        }
        for key, value in expected_coverage.items():
            require(card.get("coverage", {}).get(key) == value, errors,
                    f"{name}: scorecard coverage count mismatch for {key}")
        scorecard_metric_map = {"E_px": "E_cross_px", "G_theta_deg": "G_theta_cross_deg",
                                "G_A_D": "G_A_cross_D", "worst_point_px": "worst_point_px"}
        for field, metric_name in scorecard_metric_map.items():
            check_equal_exposure(card, metric_name, field, law_metrics, expected_exposures,
                                 errors, name)
        summary_card = summary.get("candidates", {}).get(name, {}).get("cross_agreement", {})
        for key in ("coverage", "expected_exposure_ids", "absent_scheduled_exposure_ids"):
            require(summary_card.get(key) == card.get(key), errors,
                    f"{name}: summary/per-law scorecard {key} mismatch")
        require(summary_card.get("outcomes") == card.get("outcomes"), errors,
                f"{name}: summary/per-law reported outcome metrics differ")

        frame_ids[name], complete_ids[name], scored_points[name] = law_frame_ids, law_complete, law_scored
        frame_values[name] = law_metrics
        per_law_checks[name] = {
            "certified": accepted,
            "terminal_status": "certified_model" if accepted else "completed_uncertified_checkpoint",
            "scheduled_frames": frame_count, "scheduled_slots": slot_counts["scheduled"],
            "complete_triple_frames": len(law_complete), "scored_slots": slot_counts["scored"],
            "inverse_candidate_records": line_count,
            "inverse_records_per_scheduled_frame": line_count / frame_count if frame_count else None,
            "scheduled_frame_membership_sha256": digest_json(canonical_ids(law_frame_ids)),
            "complete_frame_membership_sha256": digest_json(canonical_ids(law_complete)),
            "scored_slot_membership_sha256": digest_json(canonical_ids(law_scored)),
        }

    # Validate saved paired membership identities and exact common-cohort ordering.
    paired_path = OUT / "paired_memberships.jsonl.gz"
    paired_records = {}
    with gzip.open(paired_path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            candidate = row["candidate"]
            require(candidate not in paired_records, errors, f"duplicate paired membership for {candidate}")
            paired_records[candidate] = row
    expected_paired_names = set(NAMES) - {REFERENCE}
    require(set(paired_records) == expected_paired_names, errors,
            "paired_memberships.jsonl.gz candidate set mismatch")
    paired_membership_hashes = {}
    for candidate in expected_paired_names:
        shared_frames = frame_ids[REFERENCE] & frame_ids[candidate]
        shared_slots = scored_points[REFERENCE] & scored_points[candidate]
        shared_complete = complete_ids[REFERENCE] & complete_ids[candidate]
        expected_points = sorted(shared_slots)
        stored = paired_records.get(candidate, {})
        require(stored.get("cohort") == "full_common", errors, f"{candidate}: paired cohort label mismatch")
        require(stored.get("frame_ids") == [list(x) for x in sorted(shared_complete)], errors,
                f"{candidate}: paired complete-frame membership mismatch")
        require(stored.get("point_ids") == [list(x) for x in expected_points], errors,
                f"{candidate}: paired scored-point membership mismatch")
        paired_membership_hashes[candidate] = {
            "complete_frames_sha256": digest_json(canonical_ids(shared_complete)),
            "scored_points_sha256": digest_json(canonical_ids(shared_slots)),
            "complete_frame_count": len(shared_complete), "scored_point_count": len(shared_slots),
        }

    certified = [name for name in NAMES if summary.get("candidates", {}).get(name, {}).get("certified") is True]
    require(summary.get("run_counts", {}).get("certified_fits") == len(certified), errors,
            "summary certified fit count does not match model artifacts")
    common = summary.get("common_model_comparison", {})
    common_ids = set.intersection(*(complete_ids[name] for name in certified)) if certified else set()
    common_missing = sorted(expected_exposures - {identity[:3] for identity in common_ids})
    enough = REFERENCE in certified and len(certified) >= 2 and bool(common_ids)
    should_compare = enough and not common_missing
    expected_success = len(certified) == 4 and should_compare
    expected_status = "successful" if expected_success else "completed_incomplete"
    require(summary.get("experiment_success") is expected_success, errors,
            "summary experiment_success does not match terminal certification/common cohort")
    require(summary.get("experiment_status") == expected_status, errors,
            "summary experiment_status does not distinguish completed incomplete from success")
    require(completion.get("experiment_success") is expected_success, errors,
            "top-level completion experiment_success disagrees with summary")
    require(completion.get("experiment_status") == expected_status, errors,
            "top-level completion experiment_status disagrees with summary")
    require(completion.get("certified") == len(certified), errors,
            "top-level completion certified fit count mismatch")
    require(common.get("status") == ("internal_cross_agreement_comparison" if should_compare
                                      else "incomplete_comparison"), errors,
            "common comparison completion status disagrees with certified common coverage")
    common_check = {"certified_laws": certified, "shared_complete_frames": len(common_ids),
                    "missing_exposures": [list(x) for x in common_missing],
                    "shared_complete_membership_sha256": digest_json(canonical_ids(common_ids))}
    if should_compare:
        require(common.get("shared_frame_ids") == [list(x) for x in sorted(common_ids)], errors,
                "summary common cohort membership differs from recomputed intersection")
        require(common.get("shared_complete_frames") == len(common_ids), errors,
                "summary common complete-frame count mismatch")
        require({tuple(x) for x in common.get("expected_exposures", [])} == expected_exposures, errors,
                "summary common exposure membership differs from schedule")
        expected_values = {}
        for name in certified:
            expected_values[name] = {}
            card = common.get("common_cohort_scorecards", {}).get(name, {})
            common_rows = {fid: frame_values[name][fid] for fid in common_ids}
            for metric_name, field in METRICS:
                mean, _ = check_equal_exposure(card, metric_name, field, common_rows,
                                               expected_exposures, errors, f"common/{name}")
                expected_values[name][metric_name] = mean
                stored_mean = common.get("equal_exposure_squared_metrics", {}).get(name, {}).get(metric_name)
                require(close(stored_mean, mean), errors,
                        f"common/{name}: equal-exposure squared value mismatch for {metric_name}")
        expected_order = {
            metric: sorted(certified, key=lambda name: (expected_values[name][metric] is None,
                            expected_values[name][metric], name != REFERENCE, name))
            for metric, _ in METRICS
        }
        require(common.get("metric_order") == expected_order, errors,
                "common-cohort metric order does not match recomputed equal-exposure squared losses")
        require(common.get("lowest_cross_prediction_model") == expected_order["E_cross_px"][0], errors,
                "lowest cross-prediction model does not match recomputed common-cohort ordering")
        expected_flags = {name: [metric for metric, _ in METRICS[1:]
                                 if expected_values[name][metric] > expected_values[REFERENCE][metric]]
                          for name in certified}
        require(common.get("tradeoff_flags_vs_log") == expected_flags, errors,
                "common comparison tradeoff flags differ from recomputed squared losses")
        common_check["equal_exposure_squared_metrics"] = expected_values
        common_check["metric_order"] = expected_order

    # Recompute paired E_px^2 changes, including whether all expected exposures contribute.
    paired_delta_checks = {}
    for candidate in expected_paired_names:
        ids = complete_ids[REFERENCE] & complete_ids[candidate]
        grouped = defaultdict(list)
        for fid in ids:
            ref = frame_values[REFERENCE][fid]["E_px"]
            cand = frame_values[candidate][fid]["E_px"]
            grouped[fid[:3]].append(float(cand) ** 2 - float(ref) ** 2)
        means = {exp: sum(vals) / len(vals) for exp, vals in grouped.items()}
        delta = sum(means.values()) / len(means) if means else None
        absent = expected_exposures - set(means)
        report = summary.get("paired_internal_agreement", {}).get(candidate, {})
        loss = report.get("frame_changes", {}).get("E_px", {}).get("squared_change", {})
        require(close(loss.get("mean"), delta), errors,
                f"{candidate}: paired E_px squared-loss change mismatch")
        require({tuple(x) for x in loss.get("absent_exposure_ids", [])} == absent, errors,
                f"{candidate}: paired E_px absent exposure membership mismatch")
        require(report.get("comparison_complete") is (not absent), errors,
                f"{candidate}: paired comparison completeness flag mismatch")
        require(close(report.get("primary_delta_L_cross"), delta if not absent else None), errors,
                f"{candidate}: primary delta L_cross mismatch")
        paired_delta_checks[candidate] = {"complete_common_frames": len(ids),
                                          "equal_exposure_squared_delta": delta,
                                          "complete_exposure_support": not absent,
                                          "membership_sha256": digest_json(canonical_ids(ids))}

    result = {
        "schema": "full_calibration_results_verification_v1",
        "status": "passed" if not errors else "failed",
        "scope": "mechanical verification of completed internal calibration/cross-agreement outputs",
        "limitations": [
            "Internal full-population agreement is not independent validation or nominal-label accuracy.",
            "Invalid scheduled frames remain in the frozen population; scored coverage is reported separately.",
            "No deployment or absolute-accuracy decision is made by this verifier.",
        ],
        "run_completion": completion,
        "run_classification": {
            "orchestration_state": "all_four_tasks_terminal" if len(terminal_laws) == 4 else "orchestration_error",
            "terminal_laws": terminal_laws,
            "fit_exception_laws": fit_exceptions,
            "scientific_status": summary.get("experiment_status"),
            "uncertified_laws": [name for name in NAMES if name not in certified],
        },
        "source_hash_checks": source_checks,
        "protected_hash_checks": protected_checks,
        "schedule": checks["schedule"],
        "per_law": per_law_checks,
        "paired_membership": paired_membership_hashes,
        "paired_delta_checks": paired_delta_checks,
        "common_comparison": common_check,
        "errors": errors,
    }
    (OUT / "result_verification.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "errors": errors,
                      "per_law": per_law_checks, "common_comparison": common_check}, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
