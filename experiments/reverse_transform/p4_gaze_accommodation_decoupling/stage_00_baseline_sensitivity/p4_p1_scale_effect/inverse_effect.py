#!/usr/bin/env python3
"""Apply the established raw P4 inverse to both matched scale fits.

The inverse uses the same frozen frame scale as its forward fit. Its output is
compared with that fit's native, uncentered reference triangle in pixels. The
primary summaries pool eligible frames; capture/fixation summaries expose the
sampling distribution without introducing fit weights into this diagnostic.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
sys.path.insert(0, str(ROOT))

from distortion_model.p4_scale_comparison import ScaleComparisonProblem


def summarize(error: np.ndarray, valid: np.ndarray) -> dict:
    values = error[valid]
    return {
        "frames": int(len(error)),
        "valid_frames": int(np.sum(valid)),
        "invalid_frames": int(np.sum(~valid)),
        "pooled_vertex_rms_px": float(np.sqrt(np.mean(values**2))) if len(values) else None,
        "median_frame_vertex_rms_px": float(np.median(values)) if len(values) else None,
        "p95_frame_vertex_rms_px": float(np.quantile(values, .95)) if len(values) else None,
    }


def main() -> None:
    summary = json.loads((OUT / "summary.json").read_text())
    with np.load(OUT / "arrays.npz", allow_pickle=False) as saved:
        a = {key: saved[key] for key in saved.files}
    X = a["P4_observed"]
    capture = a["capture"]
    fixation = a["fixation_id"]
    theta = a["saved_gaze_theta_deg"]
    reference = a["reference_B4_initial_px"]
    R0 = float(a["frozen_R04_px"])
    branches = {
        "baseline_g1": ("baseline_g1", np.ones(len(X))),
        "baseline_frozen_p1m": ("baseline_g1", a["p1_m"]),
        "p1m": ("p1m", a["p1_m"]),
    }
    outputs: dict[str, np.ndarray] = {
        "capture": capture,
        "source_row": a["source_row"],
        "fixation_id": fixation,
        "saved_gaze_theta_deg": theta,
    }
    result: dict = {
        "protocol": "centered measured P4 / fixed branch frame scale; restore fitted raw optical mean; reverse stretch, projective keystone, then capture barrel; compare native recovered vertices with branch fitted B4",
        "metric": "sqrt(mean over three vertices of squared 2D distance to branch fitted native B4); pooled RMS squares this frame metric before averaging frames",
        "fit_objective_note": "Inverse metrics are row-pooled diagnostics, separate from the equal-capture/equal-fixation forward fitting objective.",
        "branches": {},
        "capture": {},
    }
    group_rows: list[dict] = []
    for branch, (fit_source, g) in branches.items():
        problem = ScaleComparisonProblem(X, theta, capture, fixation, reference, g=g, R0=R0)
        params = a[f"{fit_source}_params"]
        if not summary["fit"][fit_source]["stationary"]:
            raise ValueError(f"{fit_source}: inverse requires the selected stationary fit")
        if not np.array_equal(params, np.asarray(summary["fit"][fit_source]["params"])):
            raise ValueError(f"{fit_source}: saved array and fit summary parameters differ")
        B, _, _, _ = problem.unpack(params)
        recovered, valid, diagnostics = problem.inverse(params)
        diff = recovered - B[None]
        frame_error = np.sqrt(np.mean(np.sum(diff**2, axis=2), axis=1))
        outputs[f"{branch}_recovered_px"] = recovered
        outputs[f"{branch}_valid"] = valid.astype(np.uint8)
        outputs[f"{branch}_frame_vertex_rms_px"] = frame_error
        outputs[f"{branch}_B_fit_px"] = B
        branch_summary = summarize(frame_error, valid)
        template_radius = float(np.sqrt(np.mean(np.sum(B**2, axis=1))))
        branch_summary["fitted_template_rms_radius_px"] = template_radius
        branch_summary["pooled_inverse_error_percent_template_radius"] = (
            100*branch_summary["pooled_vertex_rms_px"]/template_radius)
        branch_summary["inverse_diagnostics"] = diagnostics
        branch_summary["scale_range"] = [float(np.min(g)), float(np.max(g))]
        branch_summary["fitted_theta04_deg"] = float(params[3])
        branch_summary["fit_source"] = fit_source
        result["branches"][branch] = branch_summary
        for cid in range(1, 5):
            mask_capture = capture == cid
            result["capture"].setdefault(str(cid), {})[branch] = summarize(
                frame_error[mask_capture], valid[mask_capture])
            for fid in range(5):
                mask = mask_capture & (fixation == fid)
                row = {
                    "branch": branch,
                    "capture": cid,
                    "fixation": fid,
                    "target_gaze_deg": float(np.unique(a["nominal_theta_deg"][mask])[0]),
                    "mean_saved_gaze_deg": float(np.mean(theta[mask])),
                }
                row.update(summarize(frame_error[mask], valid[mask]))
                group_rows.append(row)
    b = result["branches"]["baseline_g1"]["pooled_vertex_rms_px"]
    m = result["branches"]["p1m"]["pooled_vertex_rms_px"]
    result["comparison"] = {
        "delta_p1m_minus_g1_px": m-b,
        "percent_p1m_vs_g1": 100*(m/b-1),
        "delta_frozen_scale_only_minus_g1_px": result["branches"]["baseline_frozen_p1m"]["pooled_vertex_rms_px"]-b,
        "percent_frozen_scale_only_vs_g1": 100*(result["branches"]["baseline_frozen_p1m"]["pooled_vertex_rms_px"]/b-1),
        "paired_frames": int(len(X)),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "inverse_arrays.npz", **outputs)
    with (OUT / "inverse_fixation_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(group_rows[0]))
        writer.writeheader()
        writer.writerows(group_rows)
    (OUT / "inverse_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"baseline_g1": b, "p1m": m, **result["comparison"]}, sort_keys=True))


if __name__ == "__main__":
    main()
