"""Recalibrate each accommodation law on all data with the existing algorithm.

Reuse accommodation_study._fit_task (calibrate.fit and the existing cross-check).
Do NOT reuse previously fitted law coefficients or latent trajectories. Each
law gets a fresh full calibration on the same 20 conditions and all valid core
rows, followed by frozen-coefficient same-frame three-pair cross-agreement.
No calibration condition is withheld; RMS is not an accuracy acceptance gate.
"""
from __future__ import annotations

import argparse
import gzip
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
from .accommodation_schema import load_model
from .accommodation_study import _fit_task, schedule
from .data import training_data, noise_blocks
from .latest_audit import paired_summary
from .audit83 import reviewed
from .geometry import context
from .noise import coordinate_covariance, reference_covariance
from .population import frame_id, manifest
from .schema import clean_json, source_hashes, write_json
from .scorecard import build
from .validate import pilot_fit

FOLD = "full_calibration"
EXPECTED_CANDIDATES = {"ar27_log": 0., "ar27_sqrt": .5,
                       "ar27_linear": 1., "ar27_quadratic": 2.}
NAMES = tuple(EXPECTED_CANDIDATES)
REFERENCE = "ar27_log"
SCHEMA = "full_calibration_internal_agreement_v2"
METRICS = ("E_cross_px", "G_theta_cross_deg", "G_A_cross_D", "worst_point_px")


def load(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_conditions(groups):
    expected = {f"capture_{i}_detections.pkl" for i in range(1, 5)}
    if len(groups) != 20 or {g["capture"] for g in groups} != expected:
        raise ValueError("Expected all 20 reviewed conditions from captures 1-4")
    for capture in sorted(expected):
        angles = sorted(float(g["target_theta_deg"]) for g in groups if g["capture"] == capture)
        if angles != [-10., -5., 0., 5., 10.]:
            raise ValueError(f"Incomplete calibration grid for {capture}")
    return list(range(len(groups)))


def common_comparison(rows, certified, population):
    """Compare freshly calibrated laws on one identical complete-frame cohort.

    Reuse the existing scorecard. Do not mix physical units into one score or
    choose flatter trajectories. Missing exposures prevent a complete ranking.
    """
    for records in rows.values():
        population.validate(records)
    names = [name for name in NAMES if certified[name]]
    result = dict(reference=REFERENCE, compared_candidates=names,
                  unavailable_candidates=[name for name in NAMES if not certified[name]],
                  selected_model=None, promoted_for_deployment=False,
                  absolute_accuracy_thresholds=None)
    if REFERENCE not in names or len(names) < 2:
        return dict(result, status="incomplete_comparison",
                    reason="Need a certified fresh log reference and at least one alternative")
    tables = {name: {frame_id(f): f for f in rows[name] if f["complete_triple"]}
              for name in names}
    shared = sorted(set.intersection(*(set(t) for t in tables.values())))
    expected = set(population.exposures)
    missing = sorted(expected - {key[:3] for key in shared})
    result.update(scheduled_frames=len(population.frame_ids), shared_complete_frames=len(shared),
                  shared_frame_ids=shared, expected_exposures=sorted(expected),
                  missing_exposures=missing)
    if missing or not shared:
        return dict(result, status="incomplete_comparison",
                    reason="No complete common cohort across every scheduled exposure")
    for table in tables.values():
        for key in shared:
            if any(table[key].get(field) is None or not np.isfinite(table[key][field])
                   or table[key][field] < 0
                   for field in ("E_px", "G_theta_deg", "G_A_D", "worst_point_px")):
                raise ValueError("Complete-frame cross-agreement metrics must be finite and nonnegative")
    cards = {name: build([tables[name][key] for key in shared],
                        expected_exposures=population.exposures,
                        purpose="internal_full_calibration") for name in names}
    values = {name: {metric: cards[name]["outcomes"][metric]["squared_error_aggregation"]["mean"]
                     for metric in METRICS} for name in names}
    if any(value is None or not np.isfinite(value) for v in values.values() for value in v.values()):
        raise ValueError("Cross-agreement comparison needs finite scores")
    order = {metric: sorted(names, key=lambda name: (values[name][metric], name != REFERENCE, name))
             for metric in METRICS}
    flags = {name: [metric for metric in METRICS[1:]
                    if values[name][metric] > values[REFERENCE][metric]] for name in names}
    return dict(result, status="internal_cross_agreement_comparison",
                equal_exposure_squared_metrics=values, metric_order=order,
                lowest_cross_prediction_model=order["E_cross_px"][0],
                tradeoff_flags_vs_log=flags, common_cohort_scorecards=cards,
                interpretation="Lower cross-prediction loss is comparative evidence; report state/tail/bound tradeoffs, not automatic promotion")


def summarize(output):
    output = Path(output)
    pop = manifest(load(output / "splits" / FOLD / "manifest.json")["frames"])
    rows, results, certified = {}, {}, {}
    for name in NAMES:
        folder = output / "fits" / FOLD / name
        frames = load(folder / "frames.json")
        pop.validate(frames)
        accepted = (folder / "model.json").exists()
        if accepted:
            model, _ = load_model(folder / "model.json")
            if model.metadata()["exponent"] != EXPECTED_CANDIDATES[name]:
                raise ValueError(f"Wrong response law stored for {name}")
        rows[name], certified[name] = frames, accepted
        results[name] = dict(certified=accepted, cross_agreement=build(
            frames, population=pop, purpose="internal_full_calibration"))
    comparisons = {}
    with gzip.open(output / "paired_memberships.jsonl.gz", "wt", encoding="utf-8") as stream:
        for name in NAMES:
            if name == REFERENCE:
                continue
            report, membership = paired_summary(rows[REFERENCE], rows[name], pop, "full_common")
            report.update(direction=name + "_minus_fresh_log",
                          scientific_scope="internal_full_calibration_cross_agreement")
            loss = report["frame_changes"]["E_px"]["squared_change"]
            report["comparison_complete"] = not loss["absent_exposure_ids"]
            report["primary_delta_L_cross"] = loss["mean"] if report["comparison_complete"] else None
            comparisons[name] = report
            stream.write(json.dumps(clean_json(dict(candidate=name, **membership)), allow_nan=False) + "\n")
    common = common_comparison(rows, certified, pop)
    summary = dict(schema=SCHEMA, full_calibration=True, fresh_calibration_per_law=True,
                   reused_previous_fitted_models=False, held_out_conditions=[],
                   independent_validation=False, selected_model=None,
                   candidates=results, paired_internal_agreement=comparisons,
                   common_model_comparison=common)
    write_json(output / "summary.json", summary)
    lines = ["# Fresh full calibrations: log versus power accommodation laws", "",
             "Same fitting algorithm and calibration data; a new coefficient set and new framewise states for each law.",
             "All 20 conditions and all valid central-80% rows participate. No condition is withheld.",
             "Cross-checks use each law's newly fitted coefficients, frozen only after that full calibration.",
             "RMS summarizes internal agreement, not nominal-label accuracy or an acceptance threshold.", "",
             "## Individual full-population scorecards", "",
             "| Law | Certified | P4 cross-prediction (px) | Gaze agreement (deg) | Accommodation agreement (D) | Complete frames | Scored slots |",
             "|---|---|---:|---:|---:|---:|---:|"]

    def value(card, metric):
        v = card["outcomes"][metric]["equal_exposure_rms"]
        return "n/a" if v is None else f"{v:.6g}"

    for name, rec in results.items():
        card = rec["cross_agreement"]
        coverage = card["coverage"]
        lines.append(f"| {name} | {rec['certified']} | {value(card, 'E_cross_px')} | "
                     f"{value(card, 'G_theta_cross_deg')} | {value(card, 'G_A_cross_D')} | "
                     f"{coverage['complete_triples']}/{coverage['scheduled_frames']} | "
                     f"{coverage['scored']}/{coverage['scheduled_slots']} |")
    lines += ["", "Individual rows may have different usable membership. Model ordering below uses exactly the same complete frames.",
              "", "## Common-frame cross-agreement comparison", ""]
    if common["status"] == "internal_cross_agreement_comparison":
        lines += [f"Shared complete frames: {common['shared_complete_frames']}/{common['scheduled_frames']}; every scheduled exposure contributes.",
                  "", "| Law (ordered by P4 cross-prediction) | P4 cross-prediction (px) | Gaze agreement (deg) | Accommodation agreement (D) | Worse companion metrics than log |",
                  "|---|---:|---:|---:|---|"]
        for name in common["metric_order"]["E_cross_px"]:
            card = common["common_cohort_scorecards"][name]
            flags = ", ".join(common["tradeoff_flags_vs_log"][name]) or "none in state/worst-point RMS"
            lines.append(f"| {name} | {value(card, 'E_cross_px')} | {value(card, 'G_theta_cross_deg')} | "
                         f"{value(card, 'G_A_cross_D')} | {flags} |")
        lines += ["", f"Lowest internal P4 cross-prediction loss: **{common['lowest_cross_prediction_model']}**.",
                  "Separate gaze/accommodation orderings, signed axes, tails, coverage, bounds and support are in summary.json. This is not an automatic overall winner or deployment decision."]
    else:
        lines.append("No complete comparative ranking: " + common["reason"] + ".")
    lines += ["", "## Paired differences from the freshly calibrated log reference", "",
              "| Candidate | Complete exposure comparison | Delta L_cross (px squared) |", "|---|---|---:|"]
    for name, report in comparisons.items():
        delta = report["primary_delta_L_cross"]
        lines.append(f"| {name} | {report['comparison_complete']} | {'n/a' if delta is None else f'{delta:.6g}'} |")
    lines += ["", "Negative paired differences mean lower cross-prediction loss; each pair's exact cohort and exposure contributions are preserved.",
              "Gaze and accommodation may change during fixation. Nominal RMS and temporal spread never rank a law.",
              "The coefficients were calibrated on these observations: this measures internal consistency, not independent physiological accuracy.", ""]
    (output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def run(root, output, workers=1, max_nfev=300, seed=17, agreement_per_fixation=0,
        source_commit=None):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Choose a new output directory; every run recalibrates all laws")
    if not 1 <= workers <= min(4, os.cpu_count() or 1):
        raise ValueError("Use 1..4 workers")
    if max_nfev < 1 or agreement_per_fixation < 0:
        raise ValueError("Invalid iteration budget or diagnostic sampling")
    if os.environ.get("OPENBLAS_NUM_THREADS") != "1" or os.environ.get("OMP_NUM_THREADS") != "1":
        raise ValueError("Set OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1")
    if source_commit is None:
        source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if len(source_commit) != 40 or any(c not in "0123456789abcdefABCDEF" for c in source_commit):
        raise ValueError("Provide full source commit SHA")
    if dict(CANDIDATES) != EXPECTED_CANDIDATES:
        raise ValueError("Response candidate IDs or exponents changed")
    captures, groups, interval_hash = reviewed(str(root))
    ids = verify_conditions(groups)
    # count=0 is the existing loader's all-valid-core-rows mode, not a new fit algorithm.
    data, anchors = training_data(captures, groups, ids, count=0)
    if set(map(int, data["original_group"])) != set(ids):
        raise ValueError("A calibration condition has no valid rows")
    population, pop = schedule(captures, groups, ids, FOLD, agreement_per_fixation)
    source_hash = source_hashes(root)
    protected = {str(p.relative_to(root)): digest(p)
                 for directory in (root / "data", root / "models", root / "experiments")
                 if directory.exists() for p in directory.rglob("*")
                 if p.is_file() and "__pycache__" not in p.parts}
    output.mkdir(parents=True)
    policy = dict(scope="internal_full_calibration_cross_agreement",
                  absolute_accuracy_thresholds=None, nominal_error_is_acceptance_gate=False,
                  temporal_strength=0, condition_holdouts=False, independent_validation=False)
    config = dict(schema=SCHEMA, source_commit=source_commit, source_hashes=source_hash,
                  protected_hashes=protected, interval_sha256=interval_hash,
                  candidates=EXPECTED_CANDIDATES, training_group_ids=ids, evaluation_group_ids=[],
                  internal_agreement_group_ids=ids, fresh_calibration_per_law=True,
                  reused_previous_fitted_models=False, fitting_worker="accommodation_study._fit_task",
                  fitting_algorithm="calibrate.fit (unchanged)", crosscheck_algorithm="existing predict_raw_holdout/score_holdout",
                  calibration_rows=len(data["rows"]), scheduled_agreement_frames=len(pop.frame_ids),
                  calibration_rows_per_group={str(gi): int(np.sum(data["original_group"] == gi)) for gi in ids},
                  calibration_sampling="all_valid_central80_rows", agreement_per_fixation=agreement_per_fixation,
                  anchor_scales=[0.1, 0.25], prior_strength=0.001, calibration_starts=2,
                  continuation_stages=2, workers=workers, max_nfev=max_nfev, seed=seed,
                  policy=policy, runtime={"python": platform.python_version()})
    write_json(output / "config.json", config)
    started = time.monotonic()
    # Shared weighting pilot only. It is NOT a previously fitted candidate and
    # is NOT substituted for the fresh ar27_log calibration performed below.
    pilot = pilot_fit(data, anchors)
    sigma, noise_meta = coordinate_covariance(noise_blocks(captures, groups, ids))
    reference = np.median(anchors, axis=0)
    covariance = reference_covariance(data["p"], pilot, reference, sigma)
    split = output / "splits" / FOLD
    split.mkdir(parents=True)
    np.savez_compressed(split / "training_inputs.npz", **data, anchors=anchors, covariance=covariance)
    write_json(split / "population.json", population)
    write_json(split / "manifest.json", dict(
        frames=[dict(zip(("fold", "capture", "fixation", "row", "source_frame_index"), key)) for key in pop.frame_ids],
        expected_exposures=pop.exposures, slots_per_frame=3, purpose="internal_calibration_agreement"))
    provenance = dict(interval_sha256=interval_hash, source_sha256=source_hash,
        capture_sha256={name: cap.sha256 for name, cap in captures.items()},
        training_group_ids=ids, evaluation_group_ids=[], internal_agreement_group_ids=ids,
        fresh_calibration_per_law=True, reused_previous_fitted_models=False,
        training_groups=groups, sampled_training_rows=data["rows"].tolist(),
        sampled_training_group=data["original_group"].tolist(), sampling_policy="all_valid_core_rows",
        shared_training_input_sha256=digest(split / "training_inputs.npz"), noise=noise_meta,
        correspondence={name: cap.metadata["pair_index"] for name, cap in captures.items()},
        anchor_range=[anchors.min(0).tolist(), anchors.max(0).tolist()],
        conditional_context_support=dict(r_min=data["r"].min(0).tolist(), r_max=data["r"].max(0).tolist(),
            signed_P1_area_branches=np.unique(np.sign(context(data["p"]).signed_area)).tolist()))
    write_json(split / "training.json", dict(provenance=provenance, pilot_coefficients=pilot.beta,
                                            reference_state=reference, coordinate_covariance=sigma))
    del data, covariance, captures
    # The existing worker constructs PowerResponseModel(CANDIDATES[name]) with
    # no saved coefficients, then calls the existing joint fit on all rows.
    # It freezes THAT NEW fit for cross-checking, never a historical fold model.
    jobs = [(root, output, FOLD, name,
             {"max_nfev": max_nfev, "seed": seed, "coverage_policy": policy}) for name in NAMES]
    if workers == 1:
        outcomes = [_fit_task(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            outcomes = list(executor.map(_fit_task, jobs))
    write_json(output / "outcomes.json", outcomes)
    summary = summarize(output)
    changed = [path for path, h in {**protected, **source_hash}.items() if digest(root / path) != h]
    if changed:
        raise RuntimeError(f"Input files changed during run: {changed}")
    write_json(output / "completion.json", dict(complete=True, fit_tasks=len(jobs),
        certified=sum(bool(x["converged"]) for x in outcomes), seconds=time.monotonic() - started,
        scope="full_calibration_internal_agreement", fresh_calibration_per_law=True))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True, help="New run directory; no reuse of saved candidate fits")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent full fits (1..4); does not change the algorithm")
    parser.add_argument("--max-nfev", type=int, default=300)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--agreement-per-fixation", type=int, default=0,
                        help="0 checks all core rows; a positive count samples only checks, never calibration")
    parser.add_argument("--source-commit")
    args = parser.parse_args()
    run(args.root, args.output, args.workers, args.max_nfev,
        args.seed, args.agreement_per_fixation, args.source_commit)


if __name__ == "__main__":
    main()
