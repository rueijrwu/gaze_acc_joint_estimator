# Current status and handoff

Updated: 2026-10-10. Repository: `/home/aplab/ACC`.

## Decision

**Stage 4 / S7 / G7 is implemented but COMPLETE_UNCERTIFIED / PAUSE. Do not advance to Stage 5 / G8 yet.**

The current result is **G7 attempt 02**, a full joint DM0 fit with the user-accepted one-degree accommodation-to-apparent-gaze bound. The bound and independent reconstruction of the objective pass. Global stationarity and constraint-curvature certification do not pass. Preserve this fit as numerical/scientific evidence, not as a certified calibration.

Stages 1–3 (G0–G5) and Stage 4's G6 have provisional GO_WITH_LIMIT decisions. **G6 attempt 02 is the last usable parent checkpoint.** G7 attempt 01 is the preserved, historical unconstrained fit; attempt 02 is the current bounded result.

## Goal and model

Estimate the distortion transformations of P1 and P4 and infer dynamic horizontal gaze `theta` and accommodation `A` from their measured geometry. The user's main criterion is **agreement between predicted and measured relative point positions**. Nominal fixation and accommodation targets are not per-frame ground truth. Absolute image position is unnecessary for this evaluation.

The model has two states per valid frame and shared optical/center parameters. `g` is a fresh positive P1-only GLS scale, with its derivatives included in fitting. P1 and P4 use separate local coordinates `theta - omega1` and `theta - omega4`. Their operational reference angles remain fixed separately; their physical optical-zero interpretation remains unresolved.

DM0 uses `M(A) = 1 + m1 * (A - Aref)`, applied before the P4 projective transformation, including its denominator. The degree-2 center law is `D(theta,A)`. The fit uses the full ten-coordinate covariance, equal exposure weighting, finite full-exposure mean anchors (0.10 degrees for gaze, 0.25 D for accommodation), and zero temporal penalty. It does not clamp frame states to target labels. State domains are theta [-20,20] degrees and A [0,6] D.

Keep the existing scientific population and conventions: all twenty reviewed full intervals from captures 1–4; native coordinates; the P4 `[2,1,0]` permutation exactly once; no residual-selected trimming or subsampling. There are **100,090 scheduled rows, 89,175 complete valid rows, and 10,915 unavailable rows**. The shared model has 19 free global coordinates. No extra gaze dimension, independent frame-center trajectory, or P4 nuisance scale has been introduced.

## Relative points and distortion centers

Let `c1 = mean(measured P1)` and let `F1_i`, `F4_i` be the transformed template points before scale/translation. Define `mu1 = mean(F1)` and `mu4 = mean(F4)`. Against the common P1-centroid reference, predicted positions are:

```text
P1_i - c1:  g * (F1_i - mu1)
P4_i - c1:  g * (D(theta,A) + F4_i - mu1)
```

Using a common reference retains both pattern distortion and the strong P4-minus-P1 gaze signal. Centering each pattern on its own centroid would discard their separation.

Once the transformation, states, and scale are specified, centroid-based estimates of the model origins are:

```text
C1_est = c1 - g * mu1
C4_est = c4 - g * mu4
C4_est - C1_est = (c4 - c1) - g * (mu4 - mu1)
model-predicted origin separation = g * D(theta,A)
```

These are model-derived origins under the fitted convention, not independently established physical distortion centers. Measurement residuals mean the centroid-based separation need not equal `g*D` exactly. Absolute origins can be reconstructed if raw positions are retained, but are not needed for the current relative-position score.

## Why the one-degree bound was added

The user observed that P4-minus-P1 is a strong gaze signal and that accommodation should induce only a small centroid shift. The previous unconstrained fit allowed large accommodation/gaze exchanges and slope reversal. The user explicitly accepted **1 degree over A=0 to 4 D** without requiring the missing optical simulation export. Do not request that export as a prerequisite again.

For the normalized horizontal centroid law `Hx = D_x + mu4_x - mu1_x` at `g=1`, attempt 02 enforces:

```text
dHx/dtheta >= s_min
abs(dHx/dA) <= s_min / 4
s_min = 10.0270022259 reference pixels/degree
```

The slope floor is half the G6 outward-interval lower gaze slope. Outward-rounded interval inequalities cover theta [-20,20] degrees and A [0,6] D: 1,920 cells and 5,760 inequalities. They imply at most one degree of inverse-equivalent horizontal change over 0–4 D where the monotone inverse remains within the declared gaze domain.

This is a **user-accepted model assumption**, not a measured optical calibration, an individual gaze-accuracy guarantee, or a limit on fixation-mean deviations. It adds a hard constraint, not another loss term.

## Results and metric definitions

| Measure | G7 attempt 01, unconstrained | G7 attempt 02, bounded |
|---|---:|---:|
| Selected full objective J | 1516.074461 | 1925.601339 |
| Native relative-coordinate RMS | 4.985584 px | 4.678851 px |
| Fixation-mean gaze deviation from labels | 2.31665 degrees RMS | 0.92741 degrees RMS |
| Compact withheld-P4 coordinate RMS | 28.3941 px | 29.3673 px |
| Full fit certified | No | No |

G6's native relative-coordinate RMS was 4.560659 px. Attempt 02's objective components are point loss 1880.990476, gaze anchor 43.004635, accommodation anchor 1.581828, regularization 0.024400, and temporal penalty zero. A constrained model can have a higher minimum J than its unconstrained predecessor; that alone is not a defect.

The compact scores above have different usable coverage: attempt 01 scored 267 slots with three unresolved, while attempt 02 scored 270. They are not a paired comparison on identical usable slots.

**Current fitting RMS:** each frame has ten scalar relative coordinates: x/y of `P1[1]-P1[0]` and `P1[2]-P1[0]`, plus x/y of all three `P4_i-c1`. For measured-minus-predicted residual `r[k,i,c]`, exposure k with N_k valid rows:

```text
RMS = sqrt((1/20) * sum_k [(1/(10*N_k)) * sum_i sum_c r[k,i,c]^2])
```

This is native-pixel, unwhitened coordinate RMS with equal exposure weight. The fitting objective separately uses covariance weighting. Expressing all P1 points against their centroid contains the same relative information, but would change an unweighted RMS definition.

**Compact prediction RMS:** withhold one P4 point, infer the frame states from the other eight scalar coordinates with the frozen model, and then compare the predicted withheld position with its measurement. The score is `sqrt(sum(error_x^2 + error_y^2)/(2*S))` over usable slots. Attempt 02 has S=270 of 300 predetermined slots; 30 remain unavailable, zero unresolved, and 48 inferred solutions are at state bounds. This is a progress diagnostic, not a full G8 evaluation or a certificate of the shared fit. For these same slots, RMS of 2D error distances would be sqrt(2) times coordinate RMS.

**Gaze-mean RMS is only label agreement:** it is the RMS of twenty fitted exposure-mean theta deviations from nominal labels. Its improvement does not establish physiological gaze accuracy.

The user prefers measurement-prediction evaluation over target agreement. Future reporting should include median and upper-percentile 2D point errors, signed x/y residuals by condition, and prediction coverage, separately for P1 and P4. Those additional summaries have not yet replaced the saved metrics. Preserve the declared fitting objective when improving reporting.

## Numerical certificate and audit

Both starts completed the declared 32 outer cycles; selected result: perturbed. Total runtime was 782.62 seconds on CuPy 14.2.0 / Tesla P100. No automatic budget extension was run.

Selected-fit evidence:

- Physical bound feasible: minimum normalized slack 3.71e-14; three active inequalities. The continuous inverse-equivalent bound is approximately 1 degree.
- State projected gradient: 1.20e-9, passing the 1e-6 threshold.
- **Global scaled KKT residual: 2.8122e-4, failing the 1e-6 threshold.** Complementarity is 3.12e-12 and passes.
- **Active-constraint Hessian finite-difference step disagreement: 0.4508, failing the 0.01 threshold.** Full constrained profile curvature was not evaluated; do not infer global rank deficiency from this.
- All 89,175 state data Jacobians have rank two, and optical domains are valid. Independent observed-state curvature census found zero negative eigenvalues among 169,935 free eigenvalues. This does not certify the full coupled model.
- Independent NumPy reconstruction matches J within 4.55e-13, predictions within 3.98e-13 px, and g within 8.88e-16. Population accounting and parent/source hashes pass.

The common start also fails certification. Histories contain thousands of nonlinear-bound proposal rejections and heavily damped joint steps. These are numerical diagnostics; the exact cause is not yet established. The certificate's generic false state-curvature flag can also arise from the later constraint-curvature exception; use the independent census rather than interpreting that flag as proof of negative local curvature.

Keep `fit_complete`, `fit_certified`, `crosscheck_complete`, and `comparison_complete` false. No automated test suite was added or run for this campaign; the evidence above comes from experiment preflight and saved-result audits.

## Next action

1. Diagnose why the active interval constraints produce unstable Hessian estimates near the boundary and why shared updates require severe damping. Distinguish numerical behavior from structural/model mismatch.
2. Repair the constrained solver/certificate while retaining the same relative-position objective, physical bound, full population, references, and priors. Investigate a smooth equivalent constraint/certificate; do not relax the certificate to obtain a pass or add optical capacity to hide a numerical failure.
3. Preserve both existing attempts. Any justified new experiment needs a fresh output directory and explicit source/configuration provenance; do not blindly increase the iteration budget.
4. Advance to G8 only after at least one complete joint fit is numerically certified. Then evaluate the full declared withheld-point schedule with a frozen model, retaining all missing/unresolved outcomes.

Open scientific limits remain: empirical omega1/omega4 references are not independently established optical zeros; omega1 continuous uncertainty is unquantified; G3 discrete alternatives and physical accommodation calibration remain unresolved. Large compact prediction errors also remain unexplained.

## Files and execution handoff

- Scientific definitions: [Theory](docs/Theory.md), [estimator plan](docs/ESTIMATOR_PLAN.md), [stage gates](docs/STAGE_GATES.md), [Stage 4 plan](docs/stages/04_CENTER_AND_DM0_CALIBRATION.md).
- Current evidence: [G7 attempt 02 report](experiments/distortion_model/stage_04_center_and_dm0_calibration/results/g7_attempt_02/STAGE_REPORT.md), [summary](experiments/distortion_model/stage_04_center_and_dm0_calibration/results/g7_attempt_02/summary.json), [checkpoint](experiments/distortion_model/stage_04_center_and_dm0_calibration/results/g7_attempt_02/checkpoint.json), [arrays](experiments/distortion_model/stage_04_center_and_dm0_calibration/results/g7_attempt_02/fitted.npz), [independent audit](experiments/distortion_model/stage_04_center_and_dm0_calibration/results/g7_attempt_02/mechanical_audit.json).
- Parent: [G6 attempt 02 checkpoint](experiments/distortion_model/stage_04_center_and_dm0_calibration/results/g6_attempt_02/checkpoint.json). Its ancestry/provenance identifies G5; G7 provenance also records G3 and prior-attempt hashes.
- Historical result: [G7 attempt 01 report](experiments/distortion_model/stage_04_center_and_dm0_calibration/results/g7_attempt_01/STAGE_REPORT.md).
- Stage navigation: [README](experiments/distortion_model/stage_04_center_and_dm0_calibration/README.md), [results](experiments/distortion_model/stage_04_center_and_dm0_calibration/docs/RESULTS.md), [progress](experiments/distortion_model/stage_04_center_and_dm0_calibration/docs/PROGRESS.md).
- Implementation: [joint model/solver](distortion_model/joint.py), [bounded solver](distortion_model/centroid_bound.py), [bounded runner](experiments/distortion_model/stage_04_center_and_dm0_calibration/scripts/run_joint_bounded.py), [audit script](experiments/distortion_model/stage_04_center_and_dm0_calibration/scripts/audit_g7_bounded.py), [dependencies](requirements-stage7.txt).

The attempt's `source_snapshot/` and `provenance.json` are authoritative. After the campaign, the two live bounded fitting files were restored byte-for-byte to the versions actually used. Preserve archived source bytes when diagnosing or implementing a repair.

From the repository root, the existing result can be independently reconstructed with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python experiments/distortion_model/stage_04_center_and_dm0_calibration/scripts/audit_g7_bounded.py --attempt experiments/distortion_model/stage_04_center_and_dm0_calibration/results/g7_attempt_02
```

The bounded runner accepts `--output`; use a fresh attempt directory for any justified future campaign. It rejects an existing output directory. Full fitting is not the immediate next action until the numerical diagnosis is resolved.

Standing user preferences: root agent performs implementation and scientific audit; delegate mechanical work to **gpt-6-luna, low reasoning**. Execute shell commands outside the sandbox using the available escalation mechanism. Prefer vectorized/chunked GPU computation and appropriate parallelism. Keep experiment scripts, results, and reports organized by stage. Preserve unrelated working-tree changes; no commit or push has been requested.
