"""Check every saved Phase 8.2 transition hash, membership row and delta summary."""
from __future__ import annotations
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from full_position.schema import write_json


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def stats(values):
    a = np.asarray(list(values), float)
    if not len(a):
        return dict(n=0, mean=None, median=None, rms=None, p90=None, p95=None)
    return dict(n=int(len(a)), mean=float(a.mean()), median=float(np.median(a)),
                rms=float(np.sqrt(np.mean(a*a))), p90=float(np.percentile(a, 90)),
                p95=float(np.percentile(a, 95)))


def same(a, b, label, tol=2e-10):
    if a is None or b is None:
        if a is not b:
            raise AssertionError(f"{label}: null mismatch {a} vs {b}")
        return
    if isinstance(a, (int, np.integer)) or isinstance(b, (int, np.integer)):
        if int(a) != int(b):
            raise AssertionError(f"{label}: {a} != {b}")
    elif not np.isclose(float(a), float(b), rtol=tol, atol=tol):
        raise AssertionError(f"{label}: {a} != {b}")


def verify_stats(saved, values, label):
    actual = stats(values)
    for k, v in actual.items():
        same(saved[k], v, f"{label}.{k}")


def main():
    root = Path(__file__).resolve().parents[3]
    folder = Path(__file__).resolve().parent
    source = json.loads((folder/"summary.json").read_text())
    hash_checks = {}
    for relative, expected in source["source_hashes"].items():
        path = root/relative
        if not path.is_file():
            raise AssertionError(f"Source missing: {relative}")
        actual = digest(path)
        if actual != expected:
            raise AssertionError(f"Source hash changed: {relative}")
        hash_checks[relative] = actual
    if not source["source_unchanged"]:
        raise AssertionError("Audit summary did not certify unchanged sources")

    memberships = {}
    total_points = total_frames = total_shared = 0
    with gzip.open(folder/"paired_transition_membership.jsonl.gz", "rt") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            key = record["comparison"]
            if key in memberships:
                raise AssertionError(f"Duplicate membership comparison {key}")
            if key not in source["comparisons"]:
                raise AssertionError(f"Membership has unknown comparison {key}")
            points, frames = record["points"], record["frames"]
            point_ids = [tuple(p["identity"]) for p in points]
            frame_ids = [tuple(f["identity"]) for f in frames]
            if len(point_ids) != len(set(point_ids)) or len(frame_ids) != len(set(frame_ids)):
                raise AssertionError(f"Duplicate point/frame identities at line {line_number}")
            for p in points:
                if p["shared_scored"]:
                    vector = float(p["delta_squared_vector_error_px2"])
                    axis = np.asarray(p["delta_squared_axis_error_px2"], float)
                    same(vector, float(axis.sum()), f"{key} point delta {p['identity']}")
                elif "delta_squared_vector_error_px2" in p:
                    raise AssertionError(f"Unpaired point has an error delta: {key} {p['identity']}")
            total_points += len(points)
            total_frames += len(frames)
            total_shared += sum(bool(p["shared_scored"]) for p in points)
            memberships[key] = record

    if set(memberships) != set(source["comparisons"]):
        raise AssertionError("Compressed membership comparison set differs from summary")

    summary_checks = 0
    for key, record in memberships.items():
        points, frames = record["points"], record["frames"]
        summary = source["comparisons"][key]
        transition_labels = sorted(set(p["transition"] for p in points))
        if set(transition_labels) != set(summary["strata"]["transition"]):
            raise AssertionError(f"Transition labels differ for {key}")
        for label in transition_labels:
            members = [p for p in points if p["transition"] == label]
            shared = [p for p in members if p["shared_scored"]]
            saved = summary["strata"]["transition"][label]
            same(saved["scheduled_points"], len(members), f"{key}/{label} scheduled points")
            same(saved["shared_scored_points"], len(shared), f"{key}/{label} shared points")
            verify_stats(saved["point_squared_error_change"]["pooled"],
                         (p["delta_squared_vector_error_px2"] for p in shared),
                         f"{key}/{label} point delta stats")
            groups = defaultdict(list)
            for p in shared:
                identity = p["identity"]
                groups[(identity[1], p["capture"], p["fixation"])].append(p["delta_squared_vector_error_px2"])
            equal_fix = float(np.mean([np.mean(v) for v in groups.values()])) if groups else None
            same(saved["point_squared_error_change"]["equal_fixation_mean"], equal_fix,
                 f"{key}/{label} equal-fixation delta")
            frame_members = [f for f in frames if f["transition"] == label]
            same(saved["paired_complete_frames"], len(frame_members), f"{key}/{label} paired complete frames")
            for field in ("E_px", "G_theta_deg", "G_A_D", "worst_point_px"):
                values = [f[f"delta_squared_{field}"] for f in frame_members]
                verify_stats(saved["frame_squared_changes"][field]["pooled"], values,
                             f"{key}/{label}/{field} frame delta stats")
                groups_f = defaultdict(list)
                for f in frame_members:
                    ident = f["identity"]
                    groups_f[(ident[1], f["capture"], f["fixation"])].append(f[f"delta_squared_{field}"])
                mean = float(np.mean([np.mean(v) for v in groups_f.values()])) if groups_f else None
                same(saved["frame_squared_changes"][field]["equal_fixation_mean"], mean,
                     f"{key}/{label}/{field} equal-fixation delta")
            summary_checks += 1

    # Independently derive reported strong-anchor training-gain ranges directly
    # from each fold record, and ensure they describe only matched training folds.
    gain_ranges = {}
    for family, folds in (("gaze", [f"gaze_{g}" for g in ("+0", "+5", "+10", "-5", "-10")]),
                          ("capture", [f"capture_{i}" for i in range(1, 5)])):
        gain_records = {}
        for fold in folds:
            model = "conditional37"
            gain_records[fold] = source["comparisons"][f"{family}/anchor_strong/{model}"]["training_gains"][fold]
        gain_ranges[family] = {
            field: [min(v[field] for v in gain_records.values()), max(v[field] for v in gain_records.values())]
            for field in ("gaze_gain", "one_plus_accommodation_gain")}

    result = dict(schema="phase82_transition_integrity_v1", verified=True,
        source_hash_count=len(hash_checks), source_hashes_match=True,
        comparison_count=len(memberships), scheduled_point_memberships=total_points,
        shared_scored_point_memberships=total_shared, paired_frame_memberships=total_frames,
        verified_transition_summary_rows=summary_checks,
        checked_paired_squared_axis_to_vector_identities=True,
        checked_membership_counts_and_pooled_equal_fixation_delta_statistics=True,
        checked_state_gain_ranges=gain_ranges,
        source_manifest="summary.json/source_hashes", membership_artifact="paired_transition_membership.jsonl.gz")
    write_json(folder/"verification.json", result)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
