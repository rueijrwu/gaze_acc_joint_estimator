# Audit implementation and frozen-model reevaluation — 2026-10-07

Inverse certification is more reliable, while the larger model's prediction
mismatch remains. All 27 saved models were reevaluated without retraining or
changing their weights. The original `grouped_v2` results remain byte-for-byte
unchanged. Captures 5/6 were not used.

## Changes implemented

- Retain every scalar start's initial/final state, cost, termination, gradient,
  polishing outcome, and acceptance, including rejected candidates.
- Apply exact-Hessian Newton polishing with active bounds, a descent safeguard,
  and handling of indefinite curvature before rejection and branch clustering.
  Keep the original encoded projected-gradient threshold `1e-4`. Also require
  physical correction at most `1e-5`, stable cost, and local minimum curvature.
  Small roundoff-level cost changes can pass only when a small correction
  improves stationarity; gradient tolerances are unchanged.
- Independently enumerate stationary gaze roots of the fixed-accommodation
  polynomial objective and refine accommodation near candidate basins. This
  finite-grid audit is not a global completeness proof.
- Distinguish per-frame and equal-weight fixation-mean discrepancy statistics;
  save membership/counts and common state support across all three candidates.
  Report selected calibration-start cost and best rejected-start cost separately.
- Validate normalization, coordinate/coefficient order, bounds, pilot/reference,
  covariance semantics/positive definiteness, and declared support metadata.
  Original v1 artifacts retain their documented implicit covariance convention.
- Share application/evaluation support, parity, and local covariance diagnostics.
  Calibration records now separate optical, anchor, and prior costs and state
  which convergence certificate was used.

## Population and preservation

The same nine grouped folds, 8 selected rows per fixation, and original invalid
rows were retained. Each split family has 160 selected rows and 143 complete
valid rows; the families reuse observations. No residual filtering, new labels,
detector selection changes, or held-out branch selection were introduced.
SHA-256 checks cover every original result file, and model copies are identical
to the source. The reevaluation completed in about 164 seconds using six CPU
processes with one BLAS/OMP thread per process.

| Model | Gaze frames before → after | Capture frames before → after | Scored point tests per family before → after |
|---|---:|---:|---:|
| conditional27 | 143 → 143 | 143 → 143 | 429 → 429 |
| conditional37 | 139 → 143 | 142 → 143 | 426 → 429 |
| two_channel13 | 143 → 143 | 143 → 143 | No coordinate prediction |

Previously accepted full-frame states do not change by more than 0.01 in either
physical coordinate. Primary bound counts are unchanged (gaze: 10/31,
capture: 1/11 for conditional27/37). Both coordinate models now score all 429
point tests per family.

## Prediction results

The following before/after comparison uses the **same previously scored points
within each model**, so recovered observations cannot explain the change.

| Split | Model | Paired points | Original RMS, px | Revised RMS, px |
|---|---|---:|---:|---:|
| Gaze | conditional27 | 429 | 3.423 | 3.423 |
| Gaze | conditional37 | 426 | 6.197 | 6.632 |
| Capture | conditional27 | 429 | 4.062 | 4.062 |
| Capture | conditional37 | 426 | 4.354 | 4.354 |

On the expanded common 429-point support, revised conditional27/37 RMS is
**3.423/6.620 px** for gaze and **4.062/4.353 px** for capture. The initial report
compared models on 426 matching points, which explains its slightly different
conditional27 capture RMS of 4.067 px.

Eight gaze-subset inverses choose a lower retained-coordinate cost after
polishing. One improves withheld error and seven worsen it. Six of the worsened
predictions occur in the capture-2, -10-degree fixation and now have errors around
23.9–25.3 px. A lower retained-coordinate cost therefore exposes response-model
inconsistency rather than guaranteeing better withheld prediction. The excluded
measurement was used only for scoring; these worse results were retained.

### Recorded audit cases

For capture 1, row 1380, excluding P4 index 1, the selected cost drops from
**26,295.893 to 24,441.697**, with state approximately **(-11.000176 degrees,
0.961266 D)**. Withheld error improves from **5.959 to 2.695 px**. The original
gradient threshold passes after polishing.

Row 1928's full-frame solution is also certified at cost **31,026.288**.
All eight targeted full/subset solves are available. Independent polynomial
profiles agree with their best costs to numerical precision. The eight newly
selected lower-cost subset transitions independently match profile minima to at
most **4.1e-10** cost difference. Some other targeted withheld errors still reach
10.2 px. These are targeted numerical checks, not new validation observations.

## State comparisons on common support

All three revised candidates estimate the same 143 valid rows per split family.
The following statistics are RMSE of fixation means, with equal weight per
fixation, against nominal labels. Accommodation demand is not actual measured
accommodation.

| Split | Model | Gaze mean RMSE, degrees | Accommodation mean RMSE, D |
|---|---|---:|---:|
| Gaze | conditional27 | 0.761 | 1.135 |
| Gaze | conditional37 | 0.748 | 1.407 |
| Gaze | two_channel13 | 0.564 | 0.536 |
| Capture | conditional27 | 0.500 | 0.849 |
| Capture | conditional37 | 0.711 | 1.221 |
| Capture | two_channel13 | 0.366 | 0.731 |

These comparisons establish neither superior physiological accuracy for the
control nor improved overall state estimation from the coordinate models.

## Discrepancy diagnostics and controlled measurements

Training-only objective reconstruction agrees with the saved selected-start
costs within **3.2e-12**, separating optical, anchor, and prior terms. Across 1,680
training fixation/point/axis summaries (with observations reused across folds),
the median absolute signed bias is **0.497 px** and the 95th percentile is
**1.845 px**. Signed residual plots cover gaze, demand/capture, and P1 context.
These are descriptive in-sample diagnostics, not independent observations.

Conditional coefficient refits hold the saved latent states fixed. Multiplying
the prior strength by ten changes training predictions by **0.053–0.142 px RMS**
for conditional27 and **0.083–0.174 px** for conditional37. Changing the
training-only reference to anchor quartiles changes predictions by at most
0.0153/0.0077 px, respectively. Uniform covariance scaling by four leaves this
conditional coefficient solution unchanged because the normalized coefficient
penalty scales with the optical weights. This does not establish invariance of
latent-state calibration: the anchor tradeoff would change. Full latent-state
and anchor sensitivity remain a subsequent experiment.

The same frozen conditional37 response was inverted using its derived centroid
and area on the two audit rows. Both summary solutions reached the 6 D upper
bound, while six-coordinate solutions were near 1.048 and 0.252 D. This isolates
measurement compression from a change of coefficients, but is not a withheld-P4
baseline or proof of physical accuracy. Retained-x versus retained-x/y inverses
both exclude the tested P4 point: x/y improves five of six targeted predictions
and worsens one. The sample is too small for a population conclusion.

The noise covariance was not inflated. Revised median held-out squared
Mahalanobis scores remain about **1,531/2,315** (gaze, conditional27/37) and
**2,521/2,743** (capture); boundary cases omit local Gaussian scores. Effective
short-timescale coordinate noise still does not account for response discrepancy
or coefficient uncertainty.

## Verification and next experiment

**21 tests pass**, including both recorded high-residual cases, exact derivatives
and Hessians, bounded/indefinite-Hessian polishing, retention of rejected starts,
raw-input holdout noninterference, compatibility rejection, selected-start cost,
frame/fixation aggregation, and common-support state comparisons. The initial
`grouped_v2/RESULTS.md` remains an archived record of its twelve-test study;
`CURRENT_STATUS.md` describes the current suite.

Next, perform grouped training-only latent-state/coefficient refits varying
anchors, priors, and reference covariance. Inspect the large capture-2 endpoint
bias before adding specific mixed response terms or context terms. Extend
controlled measurement/branch checks to a denser development population.
Nested selection and detector/context audits must precede untouched capture-5/6
transfer. No baseline replacement or physiological accuracy claim is supported.

## Artifacts and reproduction

- [Before/after comparison and transitions](comparison.json), [metrics](metrics.csv),
  [point errors](heldout_errors.csv), [fixation means](fixation_means.csv).
- Each model has `inverse_candidates.jsonl.gz` retaining every scalar-start
  outcome. Invalid rows and inconclusive predictions remain in frame/holdout files.
- [Targeted polynomial checks](../audit_checks_v1/targeted_branch_checks.json),
  [transition profile checks](../audit_checks_v1/transition_profile_checks.json),
  [measurement ablations](../audit_checks_v1/measurement_ablations.json),
  [training objective components](../audit_checks_v1/training_objective_components.json),
  [conditional coefficient sensitivity](../audit_checks_v1/conditional_coefficient_sensitivity.json).
- [Signed training residual plot](../audit_checks_v1/training_signed_residuals.png)
  and [demand/context plot](../audit_checks_v1/training_demand_context_residuals.png).
- [Run commands and numerical contracts](../../../full_position/README.md).
