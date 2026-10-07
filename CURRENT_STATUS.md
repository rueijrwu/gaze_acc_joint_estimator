# Current status — 2026-10-07

The conditional full-position prototype and its frozen-model audit are complete,
and the saved-model three-way cross-check has now been reported. The primary
comparison evaluates excluded-P4 coordinate prediction and subset-state
agreement separately. The two-channel control has no individual-P4 decoder, so
its nominal-anchor consistency is a secondary calibration diagnostic and cannot
rank it on the primary cross-check task. None of these geometric results
establishes physiological accuracy.

## Three-way cross-check reporting (Phase 8.1)

[The `crosscheck_v1` report](experiments/full_position/crosscheck_v1/RESULTS.md)
post-processes the frozen `audit_polished_v1` frame and holdout records. It does
not recalibrate coefficients, refit latent states, or rerun predictions. Each
frame has three excluded-P4 coordinate checks and, when the triple is complete,
three pairwise comparisons among the subset-inferred gaze/accommodation states.
These are correlated checks because they share P1 measurements and overlap in
their retained P4 inputs.

Each split family contains 160 scheduled frames and 480 P4-check slots. The
existing saved population has 143 eligible frames and 429 valid point tests;
the 17 invalid frames and 51 corresponding slots remain visible as unscored.
Both coordinate models have 143 complete scored triples per split family. The
table reports equal-fixation RMS across 20 fixation/capture exposures. `E_frame`
is the RMS of the three 2D point-error norms; `G_theta` and `G_A` are RMS values
over the three unordered subset-state differences in degrees and diopters.
The interior mask requires all three subset states to be strictly inside the
computational bounds (`theta` in [-20, 20] degrees and `A` in [0, 6] D); it does
not imply that the states or P1 context lie within empirical training support.

| Split family | Model | `E_frame` (px) | `G_theta` (deg) | `G_A` (D) |
|---|---|---:|---:|---:|
| Gaze holdout | conditional27 | **3.421** | **0.269** | **0.311** |
| Gaze holdout | conditional37 | 6.700 | 0.509 | 0.929 |
| Capture/demand holdout | conditional27 | **4.014** | **0.281** | **0.319** |
| Capture/demand holdout | conditional37 | 4.307 | 0.325 | 0.392 |

On the exact shared interior-only complete-frame support, there are 110 gaze
frames and 125 capture/demand frames. Equal-fixation RMS is:

| Split family | Model | `E_frame` (px) | `G_theta` (deg) | `G_A` (D) |
|---|---|---:|---:|---:|
| Gaze holdout | conditional27 | **3.198** | **0.241** | **0.258** |
| Gaze holdout | conditional37 | 4.699 | 0.408 | 0.377 |
| Capture/demand holdout | conditional27 | **3.924** | **0.276** | **0.317** |
| Capture/demand holdout | conditional37 | 4.097 | 0.339 | 0.413 |

The separate matched interior point-level intersections contain 347 gaze and
388 capture/demand point tests; these are not the complete-frame cohorts above.
The earlier pooled point-vector RMS results remain **3.423/6.620 px** for gaze
and **4.062/4.353 px** for capture/demand (`conditional27`/`conditional37`).
Those pooled point scores weight each available point test equally and should
not be substituted for the equal-fixation complete-frame values.

Boundary counts remain material: conditional27/conditional37 have 23/79 gaze
and 8/38 capture/demand boundary slots. Shared accommodation clipping appears
in 3/17 gaze and 1/10 capture/demand frames. Some clipped frames still have
substantial cross-prediction error. In the gaze `-10` fold for capture 2,
conditional37 has five frames with `G_A` from 4.22 to 4.47 D and `E_frame` from
18.71 to 22.47 px. All seven valid frames at this fixation contain a boundary
subset estimate. The other two have shared accommodation clipping and
`G_A=0`, while `E_frame` is 13.69 and 16.04 px. Conditional27
also has three clipped frames with zero or nearly zero `G_A` and `E_frame`
from 4.02 to 4.44 px. Clipping can create apparent state agreement and is not
evidence that the estimates are physically correct.

Conditional27 has lower equal-fixation RMS for both coordinate prediction and
raw subset-state disagreement in both split families, including the matched
interior complete-frame comparisons. This is evidence for this sampled
predictive comparison, not a universal ranking: model medians and tails do not
all order the same way, and model-dependent boundary coverage differs. The
two-channel control has no individual-P4 prediction score. Nominal-anchor
consistency remains a secondary diagnostic; independent physiological
references are still required for accuracy claims. No pass threshold or
physiological result is established.

The full `pytest` suite now passes **42 tests**, including the function-based
cross-check and contract regressions. The prior `audit_polished_v1` source tree
was unchanged by this post-processing run: 219 files and 22,490,019 bytes with
the same SHA-256 tree digest before and after. The
[verification record](experiments/full_position/crosscheck_v1/verification.json) documents the hash
algorithm and confirms all nine compressed population files match the existing
cleanup manifest.

## Prior implementation audit and reevaluation

The implementation fixes recorded in the prior [audit](AUDIT_REPORT.md) are in
place. This historical implementation phase reevaluated all 27 saved models
with their original sampled populations, coefficients, and covariances; no
models were retrained. The original `grouped_v2` files were verified unchanged.
Those fixes do not mean that every metric and scientific question in the revised
audit has been completed. Its cross-check reporting phase is documented below.
The original revised outputs and comparison are in
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
exactly. The 21-test unittest-discovered suite was the count reported at that
earlier audit stage; it predates the function-based cross-check tests. The
current full pytest count is recorded in Phase 8.1. Eight transition profiles
agree within **4.1e-10** cost difference. Training objective reconstruction
matches within **3.2e-12**, and median absolute signed training bias is
**0.497 px**. These training diagnostics reuse observations across folds. Full
latent-state and anchor sensitivity, response discrepancy, coefficient
uncertainty, and physiological accuracy remain unresolved.

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

## Initial prototype verification (historical 21-test suite)

The initial prototype's **21 acceptance tests passed** at that audit stage. They
cover geometry identities and feature rank;
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

This historical `unittest` command does not collect the function-based
cross-check tests. Use the full pytest command in
[full_position/README.md](full_position/README.md) for the current suite; its
42 passing tests are recorded in Phase 8.1.

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

1. **Phase 8.2, joint training sensitivity:** jointly refit latent states and
   coefficients while varying fixation-anchor constraints, coefficient priors,
   and reference covariance. Diagnose endpoint point/axis bias,
   capture/demand confounding, and P1-context dependence before changing the
   response basis. This phase has not been run.
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
- [Three-way cross-check results](experiments/full_position/crosscheck_v1/RESULTS.md),
  [summary and matched masks](experiments/full_position/crosscheck_v1/crosscheck_summary.json),
  [frame metrics](experiments/full_position/crosscheck_v1/crosscheck_frames.csv),
  [point records](experiments/full_position/crosscheck_v1/crosscheck_points.jsonl.gz),
  and [verification](experiments/full_position/crosscheck_v1/verification.json).
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
unchanged. The cleanup run reported 14 passing tests; the audit suite at that
snapshot reported 21, including shared-population serialization and parallel
output collection.

The final audit cleanup removed a duplicate 185,021-byte CSV of training
residual groups after verifying that all 1,680 rows match the corresponding JSON
fields; the JSON also retains the `r_mean` context. It also removed an unused
whitening import and the unreferenced `geometry.pixel_prediction` helper. No
further redundant outputs were found in that cleanup snapshot. Required scripts, all 27 original models
and result sets, complete candidate archives, sensitivity failures, and GPU
evidence remain. The cleanup equivalence and updated source hashes are recorded in
[audit verification](experiments/full_position/audit_polished_v1/verification.json);
all 21 tests pass after cleanup.

On 2026-10-07, a repository-wide static reference scan found two additional
unreferenced research helpers: `relative_measurements` and its relative-schema
constants in `lib/observations.py`, plus `grid_audit_m2` in `lib/m2_model.py`.
They had no tracked code, test, documentation, or result-artifact references and
were removed. The unused `array_hash` import in `joint_m2.py` and unused test
imports were removed as well. The six-capture detections, frozen baseline model,
baseline scripts/results, grouped and polished evaluations, audit checks,
cross-check records, sensitivity evidence, GPU/CPU branch audits, reports, and
tests remain. Four workspace caches (`.pytest_cache` and three `__pycache__`
directories) occupied 511,080 bytes before the initial removal. Caches
regenerated by verification were cleared again before handoff. The full suite
passed (42 tests), and `joint_m2.py --help` succeeded. These caches can be
regenerated locally.
