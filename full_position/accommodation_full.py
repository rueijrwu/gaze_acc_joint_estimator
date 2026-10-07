"""Full-data log/power calibration and internal three-pair cross-agreement.

All reviewed conditions participate in calibration. No condition is withheld.
Each valid frame has its own horizontal gaze/accommodation state. Frozen
coefficients are checked by rotating the excluded P4 point within each frame.
The resulting errors measure internal agreement, not independent accuracy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .accommodation import CANDIDATES
from .accommodation_study import _fit_task, schedule
from .data import training_data, noise_blocks
from .latest_audit import paired_summary, COHORTS
from .audit83 import reviewed
from .geometry import context
from .noise import coordinate_covariance, reference_covariance
from .population import manifest
from .schema import source_hashes, write_json
from .scorecard import build
from .validate import pilot_fit

FOLD = "full_calibration"
NAMES = ("ar27_log", "ar27_sqrt", "ar27_linear", "ar27_quadratic")
SCHEMA = "full_calibration_internal_agreement_v1"


def load(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_conditions(groups):
    expected = {f"capture_{i}_detections.pkl" for i in range(1, 5)}
    actual = {g["capture"] for g in groups}
    if actual != expected:
        raise ValueError("Expected captures 1-4; captures 5/6 remain untouched")
    for capture in sorted(expected):
        angles = sorted(float(g["target_theta_deg"]) for g in groups if g["capture"] == capture)
        if angles != [-10., -5., 0., 5., 10.]:
            raise ValueError(f"Incomplete calibration grid for {capture}")
    return list(range(len(groups)))


def paired_agreement(reference, candidate):
    def indexed(frames):
        out = {}
        for f in frames:
            key = (f["fold"], f["capture"], int(f["fixation"]), int(f["row"]))
            if key in out:
                raise ValueError("Duplicate frame identity")
            out[key] = f
        return out
    r, c = indexed(reference), indexed(candidate)
    if r.keys() != c.keys():
        raise ValueError("Candidate changed scheduled frame population")
    exposures = sorted({key[:3] for key in r})
    shared = [key for key in r if r[key]["complete_triple"] and c[key]["complete_triple"]]
    fields = ("E_px", "G_theta_deg", "G_A_D", "worst_point_px")
    by_exposure = []
    for exposure in exposures:
        ids = [key for key in shared if key[:3] == exposure]
        by_exposure.append({
            "exposure": exposure,
            "scheduled": sum(key[:3] == exposure for key in r),
            "shared_complete": len(ids),
            "delta_squared": {
                name: float(np.mean([c[key][name] ** 2 - r[key][name] ** 2 for key in ids]))
                if ids else None for name in fields
            },
        })
    missing = [row["exposure"] for row in by_exposure if not row["shared_complete"]]
    return {
        "direction": "candidate_minus_log",
        "scope": "internal_calibration_cross_agreement",
        "scheduled_frames": len(r),
        "shared_complete_frames": len(shared),
        "missing_exposures": missing,
        "equal_exposure_delta_squared": {
            name: float(np.mean([row["delta_squared"][name] for row in by_exposure]))
            if not missing else None for name in fields
        },
        "per_exposure": by_exposure,
    }


def summarize(output):
    pop = manifest(load(output / "splits" / FOLD / "manifest.json")["frames"])
    rows, results = {}, {}
    for name in NAMES:
        folder = output / "fits" / FOLD / name
        frames = load(folder / "frames.json")
        pop.validate(frames)
        rows[name] = frames
        card = build(frames, population=pop, purpose="internal_full_calibration")
        results[name] = {
            "certified": (folder / "model.json").exists(),
            "cross_agreement": card,
        }
    comparisons = {name: paired_agreement(rows["ar27_log"], rows[name])
                   for name in NAMES if name != "ar27_log"}
    summary = {
        "schema": SCHEMA,
        "full_calibration": True,
        "held_out_conditions": [],
        "independent_validation": False,
        "selected_model": None,
        "candidates": results,
        "paired_internal_agreement": comparisons,
    }
    write_json(output / "summary.json", summary)
    lines = [
        "# Full calibration: accommodation laws and cross-pair agreement", "",
        "All 20 reviewed calibration conditions participate in each fit.",
        "Gaze and accommodation vary freely per frame; fixation labels are soft mean anchors.",
        "Each cross-check excludes one P4 only during state inversion, not calibration.",
        "These are internal consistency scores, not independent validation or accuracy limits.", "",
        "| Law | Certified | Gaze agreement (deg) | Accommodation agreement (D) | P4 cross-prediction (px) | Complete frames | Scored slots |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for name, rec in results.items():
        card = rec["cross_agreement"]
        def value(metric):
            v = card["outcomes"][metric]["equal_exposure_rms"]
            return "n/a" if v is None else f"{v:.6g}"
        coverage = card["coverage"]
        lines.append(f"| {name} | {rec['certified']} | {value('G_theta_cross_deg')} | "
                     f"{value('G_A_cross_D')} | {value('E_cross_px')} | "
                     f"{coverage['complete_triples']}/{coverage['scheduled_frames']} | "
                     f"{coverage['scored']}/{coverage['scheduled_slots']} |")
    lines += ["", "Compare the exact shared populations in summary.json. "
              "RMS summarizes disagreement; no absolute RMS or fixation-flatness gate is imposed.", ""]
    (output / "RESULTS.md").write_text("\n".join(lines))
    return summary


def run(root, output, workers=1, max_nfev=300, seed=17, agreement_per_fixation=0,
        source_commit=None):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Choose a new output directory")
    if not 1 <= workers <= min(4, os.cpu_count() or 1):
        raise ValueError("Use 1..4 workers")
    if max_nfev < 1 or agreement_per_fixation < 0:
        raise ValueError("Invalid iteration budget or diagnostic sampling")
    if os.environ.get("OPENBLAS_NUM_THREADS") != "1" or os.environ.get("OMP_NUM_THREADS") != "1":
        raise ValueError("Set OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1")
    if source_commit is None:
        source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if len(source_commit) != 40:
        raise ValueError("Provide full source commit SHA")
    if tuple(CANDIDATES) != NAMES:
        raise ValueError("Response candidate registry changed")
    captures, groups, interval_hash = reviewed(str(root))
    ids = verify_conditions(groups)
    data, anchors = training_data(captures, groups, ids, count=0)
    if set(map(int, data["original_group"])) != set(ids):
        raise ValueError("A calibration condition has no valid rows")
    population, pop = schedule(captures, groups, ids, FOLD, agreement_per_fixation)
    source_hash = source_hashes(root)
    protected = {
        str(p.relative_to(root)): digest(p)
        for directory in (root / "data", root / "models", root / "experiments")
        if directory.exists()
        for p in directory.rglob("*") if p.is_file() and "__pycache__" not in p.parts
    }
    output.mkdir(parents=True)
    policy = {
        "absolute_accuracy_thresholds": None,
        "nominal_error_is_acceptance_gate": False,
        "temporal_strength": 0,
        "condition_holdouts": False,
        "independent_validation": False,
    }
    config = {
        "schema": SCHEMA, "source_commit": source_commit,
        "source_hashes": source_hash, "protected_hashes": protected,
        "interval_sha256": interval_hash, "candidates": CANDIDATES,
        "training_group_ids": ids, "evaluation_group_ids": [],
        "calibration_rows": len(data["rows"]), "scheduled_agreement_frames": len(pop.frame_ids),
        "calibration_sampling": "all_valid_central80_rows",
        "agreement_per_fixation": agreement_per_fixation,
        "anchor_scales": [0.1, 0.25], "prior_strength": 0.001,
        "workers": workers, "max_nfev": max_nfev, "seed": seed,
        "policy": policy, "runtime": {"python": platform.python_version()},
    }
    write_json(output / "config.json", config)
    started = time.monotonic()
    pilot = pilot_fit(data, anchors)
    sigma, noise_meta = coordinate_covariance(noise_blocks(captures, groups, ids))
    reference = np.median(anchors, axis=0)
    covariance = reference_covariance(data["p"], pilot, reference, sigma)
    split = output / "splits" / FOLD
    split.mkdir(parents=True)
    np.savez_compressed(split / "training_inputs.npz", **data, anchors=anchors, covariance=covariance)
    write_json(split / "population.json", population)
    write_json(split / "manifest.json", {
        "frames": [dict(zip(("fold", "capture", "fixation", "row", "source_frame_index"), key))
                   for key in pop.frame_ids],
        "expected_exposures": pop.exposures, "slots_per_frame": 3,
    })
    provenance = {
        "interval_sha256": interval_hash,
        "source_sha256": source_hash,
        "capture_sha256": {name: cap.sha256 for name, cap in captures.items()},
        "training_group_ids": ids, "evaluation_group_ids": [],
        "training_groups": groups,
        "sampled_training_rows": data["rows"].tolist(),
        "sampled_training_group": data["original_group"].tolist(),
        "sampling_policy": "all_valid_core_rows",
        "noise": noise_meta,
        "correspondence": {name: cap.metadata["pair_index"] for name, cap in captures.items()},
        "anchor_range": [anchors.min(0).tolist(), anchors.max(0).tolist()],
        "conditional_context_support": {
            "r_min": data["r"].min(0).tolist(), "r_max": data["r"].max(0).tolist(),
            "signed_P1_area_branches": np.unique(np.sign(context(data["p"]).signed_area)).tolist(),
        },
    }
    write_json(split / "training.json", {
        "provenance": provenance,
        "pilot_coefficients": pilot.beta,
        "reference_state": reference,
        "coordinate_covariance": sigma,
    })
    del data, covariance
    jobs = [(root, output, FOLD, name,
             {"max_nfev": max_nfev, "seed": seed, "coverage_policy": policy})
            for name in NAMES]
    if workers == 1:
        outcomes = [_fit_task(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            outcomes = list(executor.map(_fit_task, jobs))
    write_json(output / "outcomes.json", outcomes)
    summary = summarize(output)
    changed = [path for path, h in {**protected, **source_hash}.items()
               if digest(root / path) != h]
    if changed:
        raise RuntimeError(f"Input files changed during run: {changed}")
    write_json(output / "completion.json", {
        "complete": True, "fit_tasks": len(jobs),
        "certified": sum(bool(x["converged"]) for x in outcomes),
        "seconds": time.monotonic() - started,
        "scope": "full_calibration_internal_agreement",
    })
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-nfev", type=int, default=300)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--agreement-per-fixation", type=int, default=0)
    parser.add_argument("--source-commit")
    args = parser.parse_args()
    run(args.root, args.output, args.workers, args.max_nfev,
        args.seed, args.agreement_per_fixation, args.source_commit)


if __name__ == "__main__":
    main()
