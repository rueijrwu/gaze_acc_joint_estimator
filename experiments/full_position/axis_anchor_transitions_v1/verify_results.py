"""Independently verify the eight axis-anchor transition comparisons."""
from __future__ import annotations

import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def same(a, b, label, tol=2e-10):
    if a is None or b is None:
        assert a is b, f"{label}: null mismatch"
    elif isinstance(a, (int, np.integer)) or isinstance(b, (int, np.integer)):
        assert int(a) == int(b), f"{label}: {a} != {b}"
    else:
        assert np.isclose(float(a), float(b), rtol=tol, atol=tol), f"{label}: {a} != {b}"


def stats(values):
    a = np.asarray(list(values), float)
    if not len(a):
        return dict(n=0, mean=None, median=None, rms=None, p90=None, p95=None)
    return dict(n=int(len(a)), mean=float(a.mean()), median=float(np.median(a)),
                rms=float(np.sqrt(np.mean(a*a))), p90=float(np.percentile(a, 90)),
                p95=float(np.percentile(a, 95)))


def verify_stats(saved, values, label):
    for field, actual in stats(values).items():
        same(saved[field], actual, f"{label}.{field}")


def main():
    root = Path(__file__).resolve().parents[3]
    folder = Path(__file__).resolve().parent
    summary = json.loads((folder / "summary.json").read_text())
    checks = 0
    for relative, expected in summary["source_hashes"].items():
        path = root / relative
        assert path.is_file(), f"Missing source: {relative}"
        assert digest(path) == expected, f"Source hash changed: {relative}"
        checks += 1
    for relative, expected in summary["training_source_hashes"].items():
        path = root / relative
        assert path.is_file(), f"Missing training source: {relative}"
        assert digest(path) == expected, f"Training source hash changed: {relative}"
        checks += 1
    assert summary["source_unchanged"] is True

    memberships = {}
    point_count = frame_count = shared_points = summary_rows = 0
    with gzip.open(folder / "paired_transition_membership.jsonl.gz", "rt") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            key = record["comparison"]
            assert key not in memberships, f"Duplicate comparison {key}"
            assert key in summary["comparisons"], f"Unknown comparison {key}"
            points, frames = record["points"], record["frames"]
            pids = [tuple(p["identity"]) for p in points]
            fids = [tuple(f["identity"]) for f in frames]
            assert len(pids) == len(set(pids)), f"Duplicate point membership at line {line_no}"
            assert len(fids) == len(set(fids)), f"Duplicate frame membership at line {line_no}"
            for point in points:
                if point["shared_scored"]:
                    vector = float(point["delta_squared_vector_error_px2"])
                    axes = np.asarray(point["delta_squared_axis_error_px2"], float)
                    same(vector, float(axes.sum()), f"{key} point delta {point['identity']}")
                else:
                    assert "delta_squared_vector_error_px2" not in point
            point_count += len(points)
            frame_count += len(frames)
            shared_points += sum(bool(p["shared_scored"]) for p in points)
            memberships[key] = record
    assert set(memberships) == set(summary["comparisons"])
    assert len(memberships) == 8

    for key, record in memberships.items():
        points, frames = record["points"], record["frames"]
        current = summary["comparisons"][key]
        labels = sorted({p["transition"] for p in points})
        assert set(labels) == set(current["strata"]["transition"])
        for label in labels:
            members = [p for p in points if p["transition"] == label]
            shared = [p for p in members if p["shared_scored"]]
            saved = current["strata"]["transition"][label]
            same(saved["scheduled_points"], len(members), f"{key}/{label} scheduled points")
            same(saved["shared_scored_points"], len(shared), f"{key}/{label} shared points")
            verify_stats(saved["point_squared_error_change"]["pooled"],
                         (p["delta_squared_vector_error_px2"] for p in shared), f"{key}/{label} point change")
            exposure = defaultdict(list)
            for p in shared:
                ident = p["identity"]
                exposure[(ident[1], p["capture"], p["fixation"])].append(p["delta_squared_vector_error_px2"])
            mean = float(np.mean([np.mean(v) for v in exposure.values()])) if exposure else None
            same(saved["point_squared_error_change"]["equal_fixation_mean"], mean,
                 f"{key}/{label} equal-fixation point change")
            frame_members = [f for f in frames if f["transition"] == label]
            same(saved["paired_complete_frames"], len(frame_members), f"{key}/{label} paired frames")
            for field in ("E_px", "G_theta_deg", "G_A_D", "worst_point_px"):
                values = [f[f"delta_squared_{field}"] for f in frame_members]
                verify_stats(saved["frame_squared_changes"][field]["pooled"], values,
                             f"{key}/{label}/{field}")
                exposure_f = defaultdict(list)
                for frame in frame_members:
                    ident = frame["identity"]
                    exposure_f[(ident[1], frame["capture"], frame["fixation"])].append(frame[f"delta_squared_{field}"])
                equal_mean = float(np.mean([np.mean(v) for v in exposure_f.values()])) if exposure_f else None
                same(saved["frame_squared_changes"][field]["equal_fixation_mean"], equal_mean,
                     f"{key}/{label}/{field} equal-fixation change")
            summary_rows += 1

    result = dict(schema="axis_anchor_transition_integrity_v1", verified=True,
        source_hash_count=len(summary["source_hashes"]), training_source_hash_count=len(summary["training_source_hashes"]),
        checked_source_hashes=checks, source_hashes_match=True,
        comparison_count=len(memberships), scheduled_point_memberships=point_count,
        shared_scored_point_memberships=shared_points, paired_frame_memberships=frame_count,
        verified_transition_summary_rows=summary_rows,
        checked_squared_axis_to_vector_identities=True,
        checked_membership_counts_and_pooled_equal_fixation_delta_statistics=True)
    (folder / "verification.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
