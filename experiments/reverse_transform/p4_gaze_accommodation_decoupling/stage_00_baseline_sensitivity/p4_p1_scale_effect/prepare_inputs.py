#!/usr/bin/env python3
"""Freeze the exact filtered row roster used by the Stage 00 P4 comparison.

Run once from the existing audited results before archiving their historical
data-loader dependencies. The resulting compact bundle is the input to future
baseline refits; source hashes from the original preparation are retained in
the adjacent JSON manifest.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
ARCHIVE = "experiments/archive/reverse_transform/superseded_baseline/"
ARCHIVED_EXPERIMENTS = (
    "capture1_p4_inverse_keystone",
    "per_capture_p4_inverse_keystone",
    "fixed_keystone_barrel_zero_gaze",
    "joint_keystone_barrel_learned_reference",
    "p4_fixation_geometry",
    "p1_scale_analysis",
    "p1_joint_keystone_magnification",
    "p1_joint_bounded_magnification",
    "p1_zero_origin_comparison",
)
ARCHIVED_SOURCE_ROOTS = {
    **{
        name: f"experiments/reverse_transform/{name}"
        for name in (
            "capture1_p4_inverse_keystone",
            "per_capture_p4_inverse_keystone",
            "fixed_keystone_barrel_zero_gaze",
            "joint_keystone_barrel_learned_reference",
            "p4_fixation_geometry",
        )
    },
    **{
        name: f"experiments/reverse_transform/p4_gaze_accommodation_decoupling/stage_00_baseline_sensitivity/{name}"
        for name in (
            "p1_scale_analysis",
            "p1_joint_keystone_magnification",
            "p1_joint_bounded_magnification",
            "p1_zero_origin_comparison",
        )
    },
}


def archive_path(path: str) -> str:
    """Map historical source paths to the preserved archive snapshots."""
    for name in ARCHIVED_EXPERIMENTS:
        marker = f"experiments/reverse_transform/{name}/"
        stage_marker = f"experiments/reverse_transform/p4_gaze_accommodation_decoupling/stage_00_baseline_sensitivity/{name}/"
        for prefix in (marker, stage_marker):
            if path.startswith(prefix):
                return ARCHIVE + name + "/" + path[len(prefix):]
    return path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def main() -> None:
    source_npz = OUT / "arrays.npz"
    source_summary = OUT / "summary.json"
    summary = json.loads(source_summary.read_text())
    required = (
        "capture", "source_row", "fixation_id", "nominal_theta_deg",
        "saved_gaze_theta_deg", "nominal_demand_diopters", "P4_observed",
        "p1_m", "reference_B4_initial_px", "frozen_R04_px",
    )
    with np.load(source_npz, allow_pickle=False) as saved:
        bundle = {key: np.asarray(saved[key]).copy() for key in required}
    n = len(bundle["capture"])
    if n != 88978 or any(len(bundle[key]) != n for key in required if bundle[key].ndim > 0 and key != "reference_B4_initial_px"):
        raise ValueError("The frozen bundle must contain the complete 88,978-row P4 cohort")
    ids = np.stack((bundle["capture"], bundle["source_row"]), axis=1)
    if len(np.unique(ids, axis=0)) != n:
        raise ValueError("The frozen P4 source-row identities are not unique")
    if not np.all(np.isfinite(bundle["P4_observed"])) or not np.all(np.isfinite(bundle["saved_gaze_theta_deg"])):
        raise ValueError("Nonfinite P4 observations or saved gaze in source results")
    if not np.all(np.isfinite(bundle["p1_m"]) & (bundle["p1_m"] > 0)):
        raise ValueError("Invalid frozen P1 scale values")
    roster = np.asarray([-10., -5., 0., 5., 10.])
    for cid in range(1, 5):
        for fid, target in enumerate(roster):
            ix = (bundle["capture"] == cid) & (bundle["fixation_id"] == fid)
            if not ix.any() or not np.all(bundle["nominal_theta_deg"][ix] == target):
                raise ValueError(f"Unexpected or missing capture {cid}, fixation {fid} target")
    out_npz = HERE / "baseline_inputs.npz"
    np.savez_compressed(out_npz, **bundle)
    manifest = {
        "schema": "p4-scale-effect-inputs-v1",
        "row_count": n,
        "row_identity": "unique exact (capture, source_row) from the reviewed P4-complete filtered roster",
        "fixations": roster.tolist(),
        "captures": [1, 2, 3, 4],
        "nominal_demand_diopters_by_capture": {
            str(cid): float(summary["nominal_A_labels_only"][str(cid)]["nominal_demand_diopters"])
            for cid in range(1, 5)
        },
        "input_rule": summary["cohort"]["inclusion_rule"],
        "saved_gaze_rule": "frozen P4 polynomial gaze, checked against P1 on exact (capture, source_row) identities",
        "p1_scale_source": {
            **summary["p1_scale_source"],
            **{
                key: archive_path(summary["p1_scale_source"][key])
                for key in ("artifact", "fit_summary_artifact")
                if key in summary["p1_scale_source"]
            },
        },
        "p4_reference": summary["reference"],
        "source_results": {
            "arrays_path": str(source_npz.relative_to(HERE.parents[4])),
            "arrays_sha256": file_sha256(source_npz),
            "summary_path": str(source_summary.relative_to(HERE.parents[4])),
            "summary_sha256": file_sha256(source_summary),
        },
        "source_input_hashes": summary["input_hashes"],
        "archived_source_path_map": {
            old: ARCHIVE + name for name, old in ARCHIVED_SOURCE_ROOTS.items()
        },
        "bundle_file": "baseline_inputs.npz",
        "bundle_sha256": file_sha256(out_npz),
        "bundle_array_hashes": {key: array_sha256(value) for key, value in bundle.items()},
    }
    (HERE / "baseline_inputs.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"bundle": str(out_npz), "rows": n, "bytes": out_npz.stat().st_size,
                      "sha256": manifest["bundle_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
