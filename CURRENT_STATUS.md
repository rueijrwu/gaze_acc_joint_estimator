# Current status — 2026-10-07

The full-position Python prototype is implemented and its first grouped evaluation
is complete. Synthetic recovery and mathematical checks pass. The real-data
results do **not yet establish an improvement over the two-channel estimator**.
Keep the frozen baseline while addressing inverse-search and model/noise issues.

## What has been implemented

The separate [full_position/](full_position/README.md) package follows the revised
[Theory.md](Theory.md) and [ESTIMATOR_PLAN.md](ESTIMATOR_PLAN.md):

- Correspondence-safe loading using payload metadata, with P4 coordinates and
  flags reordered together. The recorded mapping is `[2,1,0]`.
- P1-centroid and square-root P1 triangle-area normalization, preserving all six
  P4 coordinate responses and state-dependent deformation.
- `conditional27` and `conditional37`: one horizontal gaze/accommodation state
  predicts relative translation and all four effective triangle-map entries.
  New models use `theta_deg/10`; the frozen baseline retains `theta_deg/15`.
- Complete shared-P1 residual covariance propagation and fixed, training-only
  reference weights, including correct covariance marginals for subsets.
- Bounded variable-projection calibration with exact matrix-free derivatives,
  soft fixation-mean anchors, multiple starts, and convergence diagnostics.
- Independent bounded inversion with 49 starts, rank/branch/bound diagnostics,
  and leakage-safe predictions of each excluded P4 point.
- A freshly trained 13-coefficient two-channel control inside each fold.
- Schema/provenance validation, per-frame outputs, uncertainty/support diagnostics,
  application commands, reports, and plots.
- Optional batched NumPy/CuPy inversion and CPU/GPU comparison tools. The primary
  evaluation uses the scalar SciPy reference.

The baseline scripts, `models/quadratic_model.json`, stored detections, and prior
baseline outputs are unchanged. The optional nine-component joint P1/P4 model,
detector changes, and physiological validation have not been implemented.

## Verification

**14 acceptance tests pass.** They cover geometry identities and feature rank;
physical and profiled derivatives; fixed optimizer linearizations; covariance
propagation against finite differences and Monte Carlo; subset marginals;
excluded-point noninterference for every point/axis; synthetic latent calibration
and recovery; weak rank and multiple branches; bounds and invalid-row persistence;
distortion and state-like perturbations; baseline scale conversion equivalence;
and batched numerical formulas.

```bash
rtk proxy env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m unittest discover -s tests -v
```

## Initial real-data experiment

Primary results: [experiments/full_position/grouped_v2/RESULTS.md](experiments/full_position/grouped_v2/RESULTS.md).

- Five folds each exclude one nominal gaze across all four captures:
  `[-10,-5,0,5,10]` degrees.
- Four additional folds each exclude a whole capture/accommodation demand.
- Training selects up to 48 evenly spaced original rows per fixation from its
  central 80%; evaluation selects eight rows per fixation before inspecting
  validity. Each split family has 160 selected rows, including 17 invalid rows.
- Both coordinate capacities and the two-channel control use zero temporal
  regularization, soft mean anchors, and coefficient prior strength 0.001.
- All **27 fold/model calibrations** produced an accepted fit. Failed calibration
  starts and invalid evaluation rows remain recorded.

This is a sampled, exploratory comparison with fixed settings, not nested model
selection or a full-data performance claim. The gaze and capture split families
reuse observations and are not independent samples. Endpoint gaze holdouts are
extrapolation relative to the remaining training anchors.

### Withheld-point results

Errors below compare exactly matching testable frame/point IDs, with 426 point
tests in each split family:

| Split family | Model | Median error, px | RMS error, px |
|---|---|---:|---:|
| Gaze holdout | conditional27 | 2.665 | 3.423 |
| Gaze holdout | conditional37 | 3.828 | 6.197 |
| Capture/demand holdout | conditional27 | 3.053 | 4.067 |
| Capture/demand holdout | conditional37 | 3.105 | 4.354 |

The larger model is not consistently better. Its median endpoint errors are
9.33 px at -10 degrees and 6.72 px at +10 degrees, versus 2.88 and 3.44 px for
`conditional27`. Inverse-search limitations prevent attributing all of this
degradation to model capacity alone.

Coverage and bounds on the fixed sampled population:

| Split family | Model | Estimates / 143 valid full rows | Testable holdouts / 429 attempts | Primary bound hits |
|---|---|---:|---:|---:|
| Gaze | conditional27 | 143 | 429 | 10 |
| Gaze | conditional37 | 139 | 426 | 31 |
| Capture/demand | conditional27 | 143 | 429 | 1 |
| Capture/demand | conditional37 | 142 | 426 | 11 |

### State and uncertainty findings

The new models do not improve fixation-mean agreement with nominal labels in
this study. For gaze-condition holdouts, accommodation-mean discrepancy RMS is
1.135 D for `conditional27`, 1.450 D for `conditional37`, and 0.536 D for the
fold-local two-channel control. Demand is a soft anchor, **not measured actual
accommodation**; these numbers are not physiological accuracy estimates.

Withheld errors greatly exceed the provisional short-timescale noise prediction.
For `conditional27`, median available squared Mahalanobis scores are approximately
1,531 in gaze folds and 2,521 in capture folds. Model discrepancy, calibration
uncertainty, and effective-noise assumptions need further assessment. No artifact
alarm threshold or false-positive rate has been validated.

Matched development runs using 24 rows per fixation were also attempted. Prior
strength 0.001 produced accepted fits for all three models; without the prior,
all starts reached their 300-evaluation budgets. The zero-prior outputs remain
failed checkpoints. This is numerical sensitivity, not proof of identification.

## CPU/GPU findings

- The calibration-kernel benchmark was slower with two or four BLAS threads than
  with one. Independent folds ran concurrently in six processes, each using
  `OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1`.
- CuPy is installed and both GPUs are accessible. The optional GPU path preserves
  the NumPy formulas and analytic derivatives; PyTorch was not needed.
- On the RTX 5070 Ti, 1,024 batched frame inversions took **5.77 seconds with
  CuPy versus 85.54 seconds with batched NumPy**, about **14.8× faster**. Small
  batches were slower on GPU. Timing inputs repeat representative rows and do
  not add statistical validation evidence.
- GPU forward values and derivatives matched CPU at approximately `1e-16`.
  All 32 initial `conditional27` full/subset checks matched the scalar reference.
- The `conditional37` audit found three solutions where the scalar reference
  reported no converged inverse, and a subset solution with cost **24,441.70
  versus 26,295.89**. CPU refinement from the GPU candidates confirmed convergence
  and those costs. The subset seed used only retained coordinates.

The finite-grid scalar solver can therefore miss a better branch. The primary
tables retain the original scalar results; GPU audit results were not substituted
silently. Optional acceleration is not yet a universally equivalent replacement.

## Next work, in order

1. Strengthen branch search using independent CPU/GPU candidates, CPU refinement,
   and denser/adaptive development starts. Preserve plausible branches and rerun
   leakage tests. Reevaluate coverage and endpoint predictions first.
2. Separate response-model discrepancy from noise, anchor, and prior effects
   through grouped training-only sensitivity runs. Inspect point/axis bias,
   signed gaze curvature, gaze/accommodation cross-talk, and P1-context dependence.
3. Audit capture-dependent geometry and stored detector selection. Introduce
   nuisance/context terms only with supported rank and calibration evidence.
   Follow with nested selection and denser evaluation.

Captures 5/6 remain unused by this study and available for later untouched
transfer checks. Neither the new estimator nor geometric agreement establishes
independent physiological accuracy. No new model has replaced the baseline.

## Where to resume

- [Implementation and run commands](full_position/README.md).
- [Detailed results and next experiment](experiments/full_position/grouped_v2/RESULTS.md).
- [Metrics](experiments/full_position/grouped_v2/metrics.csv),
  [point errors](experiments/full_position/grouped_v2/heldout_errors.csv), and
  [fixation means](experiments/full_position/grouped_v2/fixation_means.csv).
- [Prediction plot](experiments/full_position/grouped_v2/heldout_prediction.png) and
  [accommodation plot](experiments/full_position/grouped_v2/accommodation_anchors.png).
- [Experiment index](experiments/full_position/README.md), distinguishing primary,
  sensitivity, and GPU-audit evidence retained after cleanup.
- [GPU branch refinement audit](experiments/full_position/gpu_check37/cpu_refinement_audit.json).

The implementation and results supersede statements that the full-position
prototype is unimplemented in the earlier handoff/design documents. Those
documents remain sources for the baseline history and original design; use this
file and the linked results for current execution status.

## Code and result cleanup (2026-10-07)

The full-position result tree was reduced from **53.6 MiB to 12.2 MiB** by
removing superseded debug/interrupted runs, duplicate worker outputs, and empty
sensitivity plots. All 27 primary fitted models, frame/holdout results, reports,
prior-sensitivity checkpoints, and GPU/CPU branch audits remain. The frozen
baseline and original detection files were preserved.

Identical per-model population CSVs now share one losslessly compressed
`population.csv.gz` per fold; original hashes are in the
[cleanup manifest](experiments/full_position/cleanup_manifest.json). Unique
worker configuration/completion moved to fold `execution.json` files.
Future parallel runs collect one authoritative summary and delete successful
scratch outputs while retaining failed-worker logs. Calibration Jacobian
products share the existing frozen implementation; reporting skips empty
holdout plots. Scripts and acceptance tests remain available for reproduction.

Verification: all 27 primary models load, all nine compressed populations match
their original hashes, and regenerated metrics, point errors, fixation means,
and matched-support summaries are byte-for-byte unchanged. All 14 tests pass,
including shared-population serialization and parallel output collection.
