# Current status — 2026-10-07

The full-position Python prototype and its frozen-model audit are complete.
Inverse certification is more reliable, but real-data results do **not establish
an improvement over the two-channel estimator**. Keep the frozen baseline while
investigating response mismatch and state-estimation sensitivity.

## Audit implementation and reevaluation (latest)

The fixes from [AUDIT_REPORT.md](AUDIT_REPORT.md) are implemented. All 27 saved
models were reevaluated with their original sampled populations, coefficients,
and covariances; no models were retrained. The original `grouped_v2` files were
verified unchanged. Revised outputs and the complete comparison are in
[audit_polished_v1/RESULTS.md](experiments/full_position/audit_polished_v1/RESULTS.md).

Every scalar start is retained in compressed diagnostics and receives bounded,
descent-safeguarded exact-Hessian polishing before acceptance/clustering. The
original `1e-4` gradient threshold is preserved; physical correction, stable cost,
and local curvature are also checked. This addresses candidates that reached a
good basin but were discarded by certification, rather than only adding starts.

`conditional37` coverage improves from **139 to 143** valid gaze frames and
**142 to 143** valid capture frames. Both coordinate models now score all **429**
P4 point tests per split family. The two-channel control and `conditional27`
remain unchanged. On the same 426 previously scored gaze tests, `conditional37`
RMS rises from **6.197 to 6.632 px**; on paired capture support it remains
**4.354 px**. On the expanded common 429-point support, revised RMS is
**3.423/6.620 px** for gaze and **4.062/4.353 px** for capture
(`conditional27`/`conditional37`). The audit's row-1380 error reduction
**5.959 to 2.695 px** is reproduced. Of eight lower-cost subset branches
recovered, seven have worse withheld predictions. Better inverse certification
does not establish better accuracy.

Reporting now distinguishes per-frame discrepancies from equal-weight
fixation-mean discrepancies, includes counts/row membership and common state
support, and reports selected-start cost separately from rejected-start cost.
Calibration records separate optical/anchor/prior costs and distinguish stable
step/cost certificates from ordinary solver stops with sufficient stationarity.
Compatibility checks validate model conventions, pilot/reference/covariance,
and declared support. Application and grouped evaluation share support, parity,
and local uncertainty diagnostics, including explicit unavailable fields.

[Audit checks](experiments/full_position/audit_checks_v1/) reproduce the eight
targeted full/subset inverses and independently profile their polynomial gaze
branches. All eight newly selected lower-cost subset transitions were also
independently profile-checked (maximum cost difference 4.1e-10). Controlled
measurement ablations use the same frozen response.
Training-only signed-residual analyses and conditional coefficient refits are
saved; latent states/anchors were not refitted. Those fixed-state refits are
conditional coefficient sensitivity analyses, not tests of anchor or latent-state
sensitivity. Local covariance still excludes coefficient uncertainty and response
discrepancy. No physiological accuracy has been established. Captures 5/6 remain
untouched.

Verification confirms all **199** source hashes match, all **27** model copies
are byte-identical and loadable, and all **9** compressed populations round-trip
exactly. The 21-test suite passes. Eight transition profiles agree within
**4.1e-10** cost difference. Training objective reconstruction matches within
**3.2e-12**, and median absolute signed training bias is **0.497 px**. These
training diagnostics reuse observations across folds. Full latent-state and
anchor sensitivity, response discrepancy, coefficient uncertainty, and
physiological accuracy remain unresolved.

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

**21 acceptance tests pass.** They cover geometry identities and feature rank;
physical and profiled derivatives; fixed optimizer linearizations; covariance
propagation against finite differences and Monte Carlo; subset marginals;
excluded-point noninterference for every point/axis; synthetic latent calibration
and recovery; weak rank and multiple branches; bounds and invalid-row persistence;
distortion and state-like perturbations; baseline scale conversion equivalence;
and batched numerical formulas. Audit regressions add exact objective/model
Hessians, polynomial coordinates, two recorded high-residual inverses, bounded
and indefinite-Hessian polishing, candidate retention, compatibility rejection,
raw-application holdout noninterference, and reporting/common-support aggregation.

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

The following is the **historical original `grouped_v2` evaluation**, before the
polishing audit. It is retained for provenance; the latest paired and expanded
support results appear above and in
[audit results](experiments/full_position/audit_polished_v1/RESULTS.md). Errors
compare exactly matching testable frame/point IDs, with 426 point tests in each
split family:

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

Original coverage and bounds on the fixed sampled population:

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

1. **Training sensitivity:** jointly refit latent states and coefficients while
   varying fixation anchors, coefficient priors, and reference covariance.
   Diagnose endpoint point/axis bias, capture/demand confounding, and P1-context
   dependence before changing the response basis.
2. **Discrepancy and context:** extend independent branch/profile checks and
   controlled measurement ablations across development populations. Audit
   capture-dependent geometry and detector selection; add targeted basis or
   context terms only when residual patterns and supported rank justify them.
3. **Evaluation:** use nested selection and denser grouped evaluation, then reserve
   captures 5/6 for untouched transfer checks.

Captures 5/6 remain unused by this study and available for later untouched
transfer checks. Neither the new estimator nor geometric agreement establishes
independent physiological accuracy. No new model has replaced the baseline.

## Where to resume

- [Implementation and run commands](full_position/README.md).
- [Audit implementation results](experiments/full_position/audit_polished_v1/RESULTS.md),
  [original/revised comparison](experiments/full_position/audit_polished_v1/comparison.json),
  [verification](experiments/full_position/audit_polished_v1/verification.json), and
  [audit checks](experiments/full_position/audit_checks_v1/).
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

Historical cleanup verification showed all 27 primary models load, all nine
compressed populations match their original hashes, and regenerated metrics,
point errors, fixation means, and matched-support summaries are byte-for-byte
unchanged. The cleanup run reported 14 passing tests; the current audit suite
reports 21, including shared-population serialization and parallel output
collection.

The final audit cleanup removed a duplicate 185,021-byte CSV of training
residual groups after verifying that all 1,680 rows match the corresponding JSON
fields; the JSON also retains the `r_mean` context. It also removed an unused
whitening import and the unreferenced `geometry.pixel_prediction` helper. No
further redundant outputs were found. Required scripts, all 27 original models
and result sets, complete candidate archives, sensitivity failures, and GPU
evidence remain. The cleanup equivalence and updated source hashes are recorded in
[audit verification](experiments/full_position/audit_polished_v1/verification.json);
all 21 tests pass after cleanup.
