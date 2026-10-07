# Current status — 2026-10-07

The new [accommodation response screen](experiments/full_position/accommodation_response_v1/RESULTS.md)
is complete. All **36 fits certified** in **246 seconds with 12 CPU workers**.
The four laws have the same 27 coefficients and free states for each frame.
No law improves every split family and shared cohort. The fresh log model
remains the reference. This is a sampled development screen. A denser nested
selection study has not been run. See the accommodation section below and
[implementation guide](full_position/ACCOMMODATION_RESPONSE.md).

The conditional full-position prototype, Phase 8.2 audit follow-up, Phase 8.3
retained-channel comparison, and Phase 8.3 audit follow-up are complete. The primary
comparison evaluates excluded-P4 coordinate prediction and subset-state
agreement separately. The two-channel control has no individual-P4 decoder, so
its nominal-anchor consistency is a secondary calibration diagnostic and cannot
rank it on the primary cross-check task. None of these geometric results
establishes physiological accuracy.

## Latest-results audit implementation (latest)

The [latest-results audit](LATEST_RESULTS_AUDIT.md) is implemented in
[the follow-up study](experiments/full_position/latest_results_followup_v1/RESULTS.md).
Selection now requires an immutable, independently declared frame schedule with
three held-point slots per frame. Missing rows or slots cannot shrink coverage
denominators. Shared complete-frame comparisons preserve the expected exposure
roster and enforce predeclared overall and per-exposure shared-coverage guards;
an exposure lost from the intersection prevents promotion. Inner partitions can
explicitly hold out whole captures or horizontal-gaze conditions across captures.
The nested coordinator requires a separate schedule callback and still returns
no promotion decision when external guards have not been declared.

Direct differential-y versus xy comparisons use exact shared point and frame
memberships, with joint interior, empirical-state and P1-support cohorts kept
separate. On all 143 common complete baseline27 frames per family,
differential-y minus xy has equal-exposure changes in E²/Gθ²/GA² of
**−0.641/+0.0649/+0.0403** for gaze and
**−3.258/−0.0013/−0.0109** for capture, in px²/deg²/D².
The gaze prediction gain therefore retains a state-agreement tradeoff. On
jointly interior frames, ΔE² remains negative: **−0.845** for gaze and
**−2.878** for capture. Candidate-specific supported cohorts cannot establish
superiority. Baseline27 differential-y remains a development challenger;
baseline27 xy remains the reference.

Training-only residual diagnosis uses saved training states, measured P1,
nominal gaze and capture structure, with fixed-ridge regressions and separate
fixation/capture/horizontal-gaze blocked predictions. For baseline27, the
state-plus-P1 common-y predictor has mean blocked skill versus the training-mean
baseline of **−1.962/−3.500/−7.761**, respectively, across the nine correlated
outer training sets. Capture 2/3 common-y means remain approximately
**−0.664/+0.398 px** after conditioning on state and P1. Capture and demand are
confounded; these results justify neither a free held-capture correction nor
added response capacity. These blocked regression checks are diagnostic:
the frozen response calibration already used their groups, so they are not a
fresh nested calibration validation.

Retained-pair common y includes a state-dependent deformation term and is not
pure translation. An additional nine-fold exploratory baseline27 xy trial uses
`Rτ = R0 + (τ_px/ell)² aaᵀ`, with each τ fixed from the equal-training-fixation
RMS of three-point common-y residuals before evaluation. This single rule is a
robust-weighting proxy containing bias and variation, not an independently
validated noise variance. There is no outer-score tuning or promotion. The
zero-τ reference is exact; profiling an unrestricted shared-y offset gives the
x-plus-differential-y retained objective.

The shared-y covariance trial completed nine folds in **34.42 seconds**.
Its training-only τ values ranged from **1.267 to 1.712 px**. Independent
verification recomputed 1,827 retained objectives, checked 27 analytic
interior certificates and 10 exact paired cohorts. Across all nine folds, 54
held-point slots on 18 frames were unavailable because retained P4 input was
insufficient; this is an input-availability outcome, not solver failure. The
slots are reflected in per-family schedule coverage: 143/160 complete frames
and 429/480 scored points for gaze, and the same counts for capture. The final
repository suite passed **90 tests**. These results remain a sensitivity
analysis; the development challenger and reference are unchanged.

The primary paired reporting and 18 training-diagnosis tasks finished in
**3.78 seconds with 12 CPU workers and BLAS/OMP=1**. Independent reconstruction
checked 2,689 historical source hashes, 35 implementation/design snapshots,
all 20 paired cohort memberships, 12,110 training component rows and 2,808
blocked regression target metrics. Historical files, model coefficients,
detectors and captures 5/6 remain unchanged. Full nested recalibration and
untouched transfer remain future work: predeclare physical-state, axis, tail,
coverage, support and bound guards, grouping and uncertainty policy, then freeze
a denser population before running that study.

## Phase 8.3 audit follow-up

[Results](experiments/full_position/phase83_audit_followup_v1/RESULTS.md) implement
the [Phase 8.3 audit](PHASE_8_3_AUDIT.md). The common scorecard now reports
excluded-P4 equal-exposure squared loss/RMS, physical subset-state disagreement,
coverage, signed axes, worst point, tails, exact paired cohorts and absent
exposures. Each subset's own state is checked against frozen nominal anchors
and empirical training extrema; measured P1 context/parity and explicit unknown
reasons accompany it. Historical scores and files remain unchanged. Backfill
contains 243 fold scorecards (234 decoder candidates and nine explicitly
inapplicable summary controls), 52 family scorecards and 8,320 scheduled frames.
The rejected weak-prior calibration still has zero scored slots and five missing
exposures. Bounds and empirical support remain distinct.

The two estimated variables remain horizontal gaze and accommodation. Every
calibration fixation has nominal vertical gaze zero, from the user's protocol
clarification; artifacts identify this as a protocol constraint, not framewise
truth. Measured image-y and its state derivatives remain intact.

The new study freezes baseline27 and strong-anchor37 responses across nine
folds and compares x plus common y, differential y, or both, with x/xy references.
It transforms retained observations, model values/Jacobians/Hessians and the
marginal covariance together. Both excluded coordinates are predicted. The
54 new tasks completed in **172.53 seconds**, using **12 CPU workers**, each
with one BLAS/OMP thread. New transformed masks have 1,920 scheduled frames,
5,760 scheduled slots, 5,148 scored slots and 1,716 complete triples. Sampling
remains the original eight rows per fixation, before validity filtering.

For baseline27, equal-exposure complete-frame results are:

| Family | Retained information | E (px) | G_theta (deg) | G_A (D) | Worst point (px) |
|---|---|---:|---:|---:|---:|
| Gaze | x | 4.203 | 0.930 | 1.010 | 5.417 |
| Gaze | x + common y | 5.087 | 0.986 | 1.096 | 6.820 |
| Gaze | x + differential y | 3.326 | 0.371 | 0.370 | 3.954 |
| Gaze | xy / x + both y components | 3.421 | 0.269 | 0.311 | 4.385 |
| Capture | x | 4.149 | 0.721 | 0.906 | 5.130 |
| Capture | x + common y | 7.593 | 1.436 | 1.438 | 9.668 |
| Capture | x + differential y | 3.586 | 0.278 | 0.302 | 4.146 |
| Capture | xy / x + both y components | 4.014 | 0.281 | 0.319 | 5.222 |

Each baseline row has 143 complete triples and 429 scored slots out of 160/480
scheduled. Common y alone worsens prediction in all four response/family cells.
Differential y improves E, state disagreement and worst-point squared changes
against x on exact shared interior cohorts in all four cells. Its lower E versus
xy is a tradeoff: gaze-family state disagreement is worse, and baseline27 capture
has **99** triples inside empirical state support versus **111** for xy. These
candidate-specific support cohorts have different memberships and cannot be
ranked as paired comparisons. The baseline27 capture xy y-axis/worst regressions
remain explicit: y RMS **3.411 → 3.565 px**, worst **5.130 → 5.222 px** versus x.

Combined common/differential y is an invertible control: all **1,920** slot
comparisons match xy availability, rank, ambiguity and discovered branch clusters.
Maximum selected-state differences are 4.39e-8 degrees and 5.30e-8 D; maximum
statistical-cost difference is 2.76e-10. No new y-mask case triggered the
rank/ambiguity profile diagnostic. A separate retained-only 65-grid profile of
**17** prior x-only cases (three rank-weak, 15 ambiguous, with overlap) found no
new branches or lower costs in 2.30 seconds. Original scores were not replaced.
Finite multistart/profiling is not a global completeness proof; cost gap two
remains heuristic. The regression suite passes **73 tests**.

[Independent verification](experiments/full_position/phase83_audit_followup_v1/verification.json)
passes with zero errors: 5,148 transformed-covariance checks, 5,750 retained
objectives, 5,733 branch certificates, 1,716 new frame E/G checks, 30,720 subset
support records and 12 exact paired-membership checks. All 2,309 historical
source hashes match, and all 8,320 backfilled primary frame/state records are
preserved. Three representative inversions rerun the full 49 starts; the other
numerical checks recompute retained objectives/certificates at saved states.

Signed in-sample training residual diagnostics retain capture structure:
baseline27 mean y residuals for captures 2/3/4 are approximately -0.848/+0.529/
+0.324 px; strong-anchor37 gives -0.860/+0.586/+0.351 px. These overlapping
training folds are correlated, and this secondary diagnostic does not select
weights or prove the cause. Common-y sensitivity motivates checking capture/
context discrepancy before adding capacity or changing the covariance.

The nested grouped-selection coordinator is implemented and tested: fitting gets
only its training groups; inner cross-check selection precedes the outer refit
and one sealed outer evaluation. Fitting callbacks must construct pilot/noise/
prior/anchor policies solely from those training groups. Externally predeclared
coverage, physical-state, axis, tail, support and bound guards are required.
Missing guards give **no promotion decision** and retain the reference. A real
nested calibration study has not been run, and historical development results
cannot become independent validation retroactively.

Next predeclare those guards and the grouping/uncertainty policy, freeze a denser
evaluation population, then execute train-only nested calibration comparisons.
Test the common-y discrepancy using training-only residual evidence before any
justified covariance/context or conditional37 curvature candidate; never tune
from outer scores. **Baseline27 with xy remains the development reference.**
Captures 5/6, detector outputs, historical studies and model coefficients remain
untouched. Necessary scripts, branch archives, scorecards, snapshots and verifier
evidence are retained under `phase83_audit_followup_v1/`.

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

## Phase 8.2 joint-training sensitivity (complete, 2026-10-07)

The grouped sensitivity run refit both conditional capacities across eight
predeclared calibration/reference settings and nine frozen outer folds. It
recorded **144 task outcomes: 143 accepted and one uncertified**
(`prior_weak/capture_1/conditional27`), with no task exceptions. The failed fit
contributes no scores; the aggregate capture/27 prior-weak result retains its
312/429 valid-slot coverage and reports the 117 otherwise-valid unavailable
slots. The run also retains all 504 calibration-start records and 13,611 scored
holdout records. No setting was selected for deployment.

The strongest positive development finding is `anchor_strong/conditional37`:
all three equal-fixation metrics improve against baseline in both split families
on full-population support, with the same 429 scores and fewer bound slots. Its
capture/37 result also improves on exact shared-interior support. Other settings
show meaningful tradeoffs; for example, `prior_weak/conditional37` reduces
capture point error while increasing state disagreement, and performs worse on
all three gaze/37 matched-interior metrics. The matched-support plot and tables
keep these measures separate. These are development results, not physiological
accuracy or a final setting selection.

The completed [results](experiments/full_position/joint_sensitivity_v1/RESULTS.md),
[verification](experiments/full_position/joint_sensitivity_v1/verification.json),
and per-task [calibration diagnostics](experiments/full_position/joint_sensitivity_v1/calibration_diagnostics.csv)
retain failures, support denominators, and weighted objective components. The
[audit report](AUDIT_REPORT.md) records the provenance and supplemental stop
check. The 144-task run took 53 minutes and used six parallel CPU workers with
BLAS and OMP set to one, matching the measured calibration benchmark
([benchmark](experiments/full_position/blas_benchmark.json)). The user's
parallel-CPU-or-verified-GPU preference was met with parallel CPU execution;
the saved small-batch inverse benchmark favored CPU, while large GPU batches
remain a separate acceleration option. All 52 repository tests passed in 9.72
seconds; 219 frozen-source hashes matched and the original result artifacts
remained unchanged. Exact pre-run copies of [Theory.md](experiments/full_position/joint_sensitivity_v1/design_snapshot/Theory.md)
and [ESTIMATOR_PLAN.md](experiments/full_position/joint_sensitivity_v1/design_snapshot/ESTIMATOR_PLAN.md)
are retained with hashes matching the run config.

## Phase 8.2 audit implementation and Phase 8.3 (complete, 2026-10-07)

The [Phase 8.2 audit](PHASE_8_2_AUDIT.md) recommendations have been implemented.
[Theory.md](Theory.md) now describes the exact optical state-scale symmetry;
both capacities have prediction/derivative regression tests. Descriptive positive
gains and residual trajectory differences use matched training rows only.
Evaluation states are never rescaled. Numerical inverse objective scaling now
keeps statistical branch costs unchanged; a real two-minimum regression shows
that changing covariance can instead change ambiguity.

The [saved-result decomposition](experiments/full_position/phase82_audit_transitions_v1/RESULTS.md)
retains all four bound/interior transitions, exact point/frame memberships, and
paired squared changes by fixation, signed gaze, and capture. Strong-anchor37
gaze gains concentrate in bound-related strata. Its 107 shared-interior gaze
frames have equal-fixation changes in E²/Gθ²/GA² of -0.8544 px²/-0.00245 deg²/
+0.00009 D². Capture bound-to-interior transitions worsen pixel error, while the
126 shared-interior capture frames improve E²/Gθ²/GA² by -0.1400/-0.03720/-0.03359
in their respective squared units. These are descriptive strata, not a causal
decomposition or independent validation. Training-only strong-anchor37 gains
range from 0.9788 to 1.0258 for gaze and 0.8871 to 1.2170 for `1+A`.

The [strict training-only retry](experiments/full_position/phase82_strict_retry_v1/RESULTS.md)
uses an explicit preconditioned continuation with tighter linear solves and the
unchanged acceptance gates. From the preserved failed trajectory, 12 continuation
evaluations reduce physical projected stationarity from 0.00318374 to 0.00099793
(limit 0.001); inner stationarity is 2.40e-12 (limit 1e-7). Independent reconstruction
matches coefficients, objective components, and stationarity. The original
143/144 study and its missing evaluation coverage remain unchanged; this retry
does not add evaluation scores to that historical study.

The [two missing anchor conditions](experiments/full_position/axis_anchor_sensitivity_v1/RESULTS.md)
jointly refit both capacities on all nine frozen folds: **36/36 certified fits**,
126 archived starts, and the same 429 scored slots per family/condition/capacity.
Eight parallel CPU workers with BLAS/OMP=1 completed the run in **634 seconds**.
Only the indicated soft fixation-mean penalty scale changes; the training-only
pilot, covariance, populations, seed, and prior policy remain fixed.

| Family | Capacity | Anchor condition | E (px) | Gθ (deg) | GA (D) |
|---|---|---|---:|---:|---:|
| Gaze | 27 | gaze-only strong | 3.421 | 0.287 | 0.331 |
| Gaze | 27 | accommodation-only strong | 3.426 | 0.267 | 0.315 |
| Gaze | 37 | gaze-only strong | 4.700 | 0.398 | 0.436 |
| Gaze | 37 | accommodation-only strong | 4.760 | 0.420 | 0.450 |
| Capture | 27 | gaze-only strong | 4.170 | 0.292 | 0.324 |
| Capture | 27 | accommodation-only strong | 4.172 | 0.278 | 0.315 |
| Capture | 37 | gaze-only strong | 4.323 | 0.278 | 0.351 |
| Capture | 37 | accommodation-only strong | 4.312 | 0.338 | 0.408 |

All table metrics use equal-fixation complete-frame RMS. The gaze-only scales are
(0.05 deg, 0.25 D); accommodation-only scales are (0.10 deg, 0.125 D). These are
penalty scales, not physiological uncertainty estimates. In conditional37 capture
folds, gaze-only strengthening improves state agreement, while accommodation-only
strengthening slightly worsens it relative to baseline37. On 126 matched interior
capture frames, gaze-only ΔGθ²/ΔGA² is -0.03639/-0.04062; accommodation-only is
+0.00494/+0.00333. Thus stabilization is mainly associated with gaze anchoring in
this population, with no uniformly better setting or replacement for baseline27.

[Phase 8.3](experiments/full_position/phase83_retained_channels_v1/RESULTS.md)
compares retained x with retained x/y under the same frozen baseline27 or
strong-anchor37 response, all three P1 references, and exactly the same excluded
P4 point. The inverse API receives only the two retained x coordinates or four
retained x/y coordinates and their covariance marginal. All 18 response/fold
tasks completed in **57 seconds with 10 CPU workers**, reusing certified frozen
x/y holdouts and computing the x inverses with the same 49-start scalar policy.
The 84,084 x start candidates are archived; 1,716 selected branches pass
independent stationarity, correction, curvature, and stable-cost checks.

| Family | Frozen response | Retained | E (px) | Gθ (deg) | GA (D) | Scores / 480 | Complete / 160 | Bound slots |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Gaze | baseline27 | x | 4.203 | 0.930 | 1.010 | 429 | 143 | 39 |
| Gaze | baseline27 | x/y | 3.421 | 0.269 | 0.311 | 429 | 143 | 23 |
| Capture | baseline27 | x | 4.149 | 0.721 | 0.906 | 429 | 143 | 40 |
| Capture | baseline27 | x/y | 4.014 | 0.281 | 0.319 | 429 | 143 | 8 |
| Gaze | strong-anchor37 | x | 5.550 | 1.141 | 1.379 | 423 | 137 | 82 |
| Gaze | strong-anchor37 | x/y | 4.764 | 0.440 | 0.530 | 429 | 143 | 29 |
| Capture | strong-anchor37 | x | 4.692 | 0.745 | 1.135 | 418 | 132 | 41 |
| Capture | strong-anchor37 | x/y | 4.293 | 0.276 | 0.366 | 429 | 143 | 30 |

Each cell retains 51 invalid slots. X-only strong-anchor37 additionally loses
6 gaze and 11 capture scores to rank/ambiguity; these remain visible. The raw
RMS rows have different support for this response, so comparisons use exact
paired memberships. On jointly interior complete frames, x/y minus x
equal-fixation ΔE²/ΔGθ²/ΔGA² is negative in all four response/family cells
(121/112 baseline27 gaze/capture frames; 84/110 strong-anchor37 frames).
This supports additional retained-y information beyond boundary effects. It
does not prove physiological accuracy or superiority of conditional37. Worst
point and axis error, rank, branches, and coverage remain separate diagnostics;
baseline27 capture worst-point RMS slightly increases despite lower average E.

All **59 regression tests pass**. New fits and Phase 8.3 preserve source hashes
and implementation/design snapshots; saved-result diagnostics also have
standalone verification scripts. Earlier
result files and captures 5/6 remain untouched. Parallel CPU execution meets the
acceleration preference; no GPU calibration or inverse policy was introduced.

## Next work, in order

1. **Predeclare nested selection:** fix the capture or horizontal-gaze transfer
   question, complete frame schedule, shared-coverage guards and physical-state,
   axis, tail, support and bound limits before fresh calibration comparisons.
2. **Selective curvature test (audit P3, conditional):** the tested
   `fit(..., curvature_strength=...)` option shrinks only the ten added
   conditional37 t²/t²L coefficients toward zero while jointly fitting shared
   coefficients/states. It is not enabled in these runs. Run a predeclared,
   training-selected nested comparison only if residual evidence justifies
   retaining the richer capacity; current evidence keeps baseline27 as reference.
3. **Nested evaluation and transfer (P4):** use denser grouped evaluation and
   reserve captures 5/6 for untouched transfer checks. Independent detector
   review and physiological references remain separate requirements for
   accuracy claims.

Captures 5/6 remain unused by this study and available for later untouched
transfer checks. Neither the new estimator nor geometric agreement establishes
independent physiological accuracy. No new model has replaced the baseline.

## Where to resume

- [Latest-results audit follow-up](experiments/full_position/latest_results_followup_v1/RESULTS.md),
  [independent verification](experiments/full_position/latest_results_followup_v1/verification.json),
  [paired comparisons](experiments/full_position/latest_results_followup_v1/direct_comparison.json),
  [training diagnostics](experiments/full_position/latest_results_followup_v1/training_diagnosis.json),
  [signed point tails](experiments/full_position/latest_results_followup_v1/point_signed_tails.csv),
  [final test evidence](experiments/full_position/latest_results_followup_v1/tests_final.txt),
  and [single shared-y covariance trial](experiments/full_position/latest_results_followup_v1/covariance_trial/summary.json)
  with its [independent verification](experiments/full_position/latest_results_followup_v1/verification.json).
- [Phase 8.3 results](experiments/full_position/phase83_retained_channels_v1/RESULTS.md),
  [verification](experiments/full_position/phase83_retained_channels_v1/verification.json),
  and [plot](experiments/full_position/phase83_retained_channels_v1/phase83_metrics.png).
- [Anchor-axis results](experiments/full_position/axis_anchor_sensitivity_v1/RESULTS.md),
  [verification](experiments/full_position/axis_anchor_sensitivity_v1/verification.json),
  and [matched transitions](experiments/full_position/axis_anchor_transitions_v1/summary.json).
- [Saved boundary/gauge diagnostics](experiments/full_position/phase82_audit_transitions_v1/RESULTS.md)
  and [strict retry](experiments/full_position/phase82_strict_retry_v1/RESULTS.md).

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
- [Phase 8.2 sensitivity results](experiments/full_position/joint_sensitivity_v1/RESULTS.md),
  [verification](experiments/full_position/joint_sensitivity_v1/verification.json),
  [paired-interior plot](experiments/full_position/joint_sensitivity_v1/paired_interior_vs_baseline.png),
  [calibration diagnostics](experiments/full_position/joint_sensitivity_v1/calibration_diagnostics.csv),
  and [run config](experiments/full_position/joint_sensitivity_v1/config.json).
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


## Accommodation response-law screen (2026-10-07)

The proposed log-versus-shifted-power comparison completed as a sampled
development screen in
[accommodation_response_v1](experiments/full_position/accommodation_response_v1/RESULTS.md).
All 36 fits across nine gaze/capture folds certified in 246.13 seconds using 12
CPU workers with one BLAS/OMP thread per worker. Each candidate retained 143/160
complete frames and 429/480 scored P4 slots, with all scheduled exposures
accounted for. No exponent was selected or promoted; fresh log27 xy remains the
reference. Results vary by split family and support cohort, and capture
quadratic improves full-cohort prediction while substantially increasing
same-frame state disagreement and bound hits. This screen is development
evidence, not a nested evaluation or physiological exponent claim. Captures 5/6
remain untouched.

An outside-sandbox hardware probe found two visible GPUs and a working minimal
CuPy operation. The accommodation family has not passed GPU value/derivative/
branch/certificate parity, so the screen used CPU. Per-exposure paired deltas,
signed axes, tails, coverage, bounds, support and training-only trajectory
diagnostics are saved with the run. Independent verification passed all nine
folds and 36 fits, including 3,840 holdouts and 6,055 training rows; all fits
converged. The persisted counters and completed continuation log are linked
from the accommodation results page. The screen remains development evidence,
not a nested evaluation or physiological exponent claim.

## Full accommodation calibration (running; results pending)

The [full-calibration runner](full_position/accommodation_full.py) and
[experiment plan](ACCOMMODATION_FULL_CALIBRATION_PLAN.md) specify four fresh
fits, one per fixed response law, using the existing joint optimizer and
three-way frozen-coefficient P4 agreement check. Each law uses the same 20
reviewed conditions. The prior sampled screen above is unchanged and supplies
no coefficients or latent states to these fits. The real-data run is active;
no calibration outcomes or agreement results are available yet.
The required suite passed (54 tests) before launch, and exactly one final review
found no substantive issues. The run's launch check is recorded in
[launch_verification.json](experiments/full_position/accommodation_full_v2/launch_verification.json).

The `core` training-window and agreement-window defaults remain unchanged:
71,784 central-80% training rows and 80,072 core agreement frames per law.
The active run explicitly sets both windows to `fixation_period`, using 89,175
valid training rows and scheduling 100,090 raw frames per law (300,270 P4 slots
per law; 1,201,080 slots across four laws). Invalid or unavailable measurements
remain tracked. These counts are persisted in the shared training input and
schedule manifest.

The run started at 2026-10-07 21:37:51 UTC with four CPU fit workers, each
using one BLAS, OpenMP and MKL thread. Runner PID: 9986; managed execution
session: 7482. Output: [accommodation_full_v2](experiments/full_position/accommodation_full_v2/).
Stdout/stderr log: `/tmp/accommodation_full_v2.log`. GPU value, derivative,
branch and certificate parity remains unverified; the configured inverse backend
is the existing scalar CPU path. This is an internal-consistency comparison,
not independent validation or a physiological accuracy test, and it defines no
absolute RMS gate or deployment choice. Run completion and certification have
not yet been established.

At the latest check (~4 minutes after start), all four calibration workers were
active at about 99% CPU each, with roughly 1.5–2.1 GiB RSS per worker. The four
`calibration_candidates.jsonl.gz` files had not emitted their first flushed
start/stage checkpoint yet; no candidate certification is available. The output
tree was 77 MiB and workspace/tmp each had 1.3 TiB free. The tracked PID file is
`/tmp/accommodation_full_v2.pid` (9986); managed session ID is 7482. Monitor with:

```bash
rtk proxy ps -p 9986,10058,10061,10062,10063 -o pid,stat,etime,pcpu,rss
rtk ls -lh experiments/full_position/accommodation_full_v2/fits/full_calibration/*/calibration_candidates.jsonl.gz
```

Expected final report paths are
[summary.json](experiments/full_position/accommodation_full_v2/summary.json),
[RESULTS.md](experiments/full_position/accommodation_full_v2/RESULTS.md), and
[completion.json](experiments/full_position/accommodation_full_v2/completion.json).
