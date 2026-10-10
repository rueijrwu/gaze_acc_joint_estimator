# Current status and handoff

Updated: 2026-10-10. Repository: `/home/aplab/ACC`.

## Decision

**Stage 4 / S7 / G7 is COMPLETE_CERTIFIED_WITH_LIMIT / GO_WITH_LIMIT in attempt04.** G8 has not run and awaits separate authorization.

The current result is **G7 attempt04**, one reviewed constrained saved-point correction to attempt03. It meets the unchanged full-population 1e-6 stationarity and complementarity thresholds: global scaled KKT residual 4.24e-10 and state projected gradient 1.32e-13. All 89,175 valid state Jacobians have rank two; constrained profile curvature is positive with rank16; P1/P4 domains and the one-degree physical bound are valid. Independent public-optics reconstruction matches objective components within 2.28e-13, predictions within 3.41e-13 px, and g within 7.77e-16. The correction changes J by −5.9962e-12 and maximum predicted position by 6.19e-6 px; no material empirical improvement is claimed. `crosscheck_complete=false` and `comparison_complete=false`.

Stages 1–3 (G0–G5) and Stage 4's G6 have provisional GO_WITH_LIMIT decisions. G7 attempts01–03 and repair records remain preserved historical results. Attempt04 is the current certified numerical checkpoint, with the same G6 operational-reference lineage and attempt03 as its immediate parent. Compact results are historical attempt03 data and were not rerun for attempt04.

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

| Measure | G7 attempt 01, unbounded | G7 attempt 02, bounded (old FD) | G7 attempt 03, continued | G7 attempt04, one correction |
|---|---:|---:|---:|---:|
| Selected full objective J | 1516.074461 | 1925.601339 | 1925.601339 | 1925.601339 |
| Native relative-coordinate RMS | 4.985584 px | 4.678851 px | 4.678857 px | 4.678857 px |
| Fixation-mean gaze deviation from labels | 2.31665 degrees RMS | 0.92741 degrees RMS | 0.9274 degrees RMS | not recalculated |
| Compact withheld-P4 coordinate RMS | 28.3941 px | 29.3673 px | 29.3673 px | attempt03 historical: 29.3673 px |
| Global scaled KKT residual | not certified | 2.8122e-4 | 2.5103e-5 | 4.2363e-10 |
| Full fit certified | No | No | No | Yes, GO_WITH_LIMIT |

Attempt02's displayed global residual is its historical finite-difference certificate. Repair01's same-state selected-branch derivative reassessment measured 0.0035570; those derivatives are not directly comparable. Attempt04's current certificate uses the reviewed repaired derivative implementation.

G6's native relative-coordinate RMS was 4.560659 px. Attempt 02's objective components are point loss 1880.990476, gaze anchor 43.004635, accommodation anchor 1.581828, regularization 0.024400, and temporal penalty zero. A constrained model can have a higher minimum J than its unconstrained predecessor; that alone is not a defect.

The compact scores above have different usable coverage: attempt 01 scored 267 slots with three unresolved, while attempts 02 and 03 scored the same 270 slots with zero unresolved and 30 unavailable. Attempts 02/03 use the same saved schedule and eligible slot IDs; see the matched comparison in the compact aggregation report.

**Current fitting RMS:** each frame has ten scalar relative coordinates: x/y of `P1[1]-P1[0]` and `P1[2]-P1[0]`, plus x/y of all three `P4_i-c1`. For measured-minus-predicted residual `r[k,i,c]`, exposure k with N_k valid rows:

```text
RMS = sqrt((1/20) * sum_k [(1/(10*N_k)) * sum_i sum_c r[k,i,c]^2])
```

This is native-pixel, unwhitened coordinate RMS with equal exposure weight. The fitting objective separately uses covariance weighting. Expressing all P1 points against their centroid contains the same relative information, but would change an unweighted RMS definition.

**Compact prediction RMS:** withhold one P4 point, infer the frame states from the other eight scalar coordinates with the frozen model, and then compare the predicted withheld position with its measurement. The score is `sqrt(sum(error_x^2 + error_y^2)/(2*S))` over usable slots. Attempts 02/03 have S=270 of 300 predetermined slots; 30 remain unavailable, zero unresolved, and 48 inferred solutions are at state bounds. This is a progress diagnostic, not a full G8 evaluation or a certificate of the shared fit. For these same slots, RMS of 2D error distances would be sqrt(2) times coordinate RMS.

**Gaze-mean RMS is only label agreement:** it is the RMS of twenty fitted exposure-mean theta deviations from nominal labels. Its improvement does not establish physiological gaze accuracy.

The user prefers measurement-prediction evaluation over target agreement. The saved compact P4 reports now include median and upper-percentile 2D errors, signed x/y residuals by condition, coverage, endpoints and bound strata, alongside the existing RMS. Full-population evaluation and separate P1/P4 reporting remain future work. Preserve the declared fitting objective when improving reporting.

## Numerical certificate and audit

Both starts in attempt02 completed the declared 32 outer cycles; selected result: perturbed. Total runtime was 782.62 seconds on CuPy 14.2.0 / Tesla P100. No automatic budget extension was run.

Historical attempt02 evidence, evaluated with the old finite-difference constraint derivatives:

- Physical bound feasible: minimum normalized slack 3.71e-14; three active inequalities. The continuous inverse-equivalent bound is approximately 1 degree.
- State projected gradient: 1.20e-9, passing the 1e-6 threshold.
- **Global scaled KKT residual: 2.8122e-4, failing the 1e-6 threshold.** Complementarity is 3.12e-12 and passes.
- **Active-constraint Hessian finite-difference step disagreement: 0.4508, failing the 0.01 threshold.** Full constrained profile curvature was not evaluated; do not infer global rank deficiency from this.
- All 89,175 state data Jacobians have rank two, and optical domains are valid. Independent observed-state curvature census found zero negative eigenvalues among 169,935 free eigenvalues. This does not certify the full coupled model.
- Independent NumPy reconstruction matches J within 4.55e-13, predictions within 3.98e-13 px, and g within 8.88e-16. Population accounting and parent/source hashes pass.

### Attempt 03 continuation update

Repair01 reassessed attempt02 at the exact same states and globals, without optimization. It reproduced the objective exactly, passed the deterministic derivative regression, found a smooth active branch and positive constrained profile curvature (rank 16), and measured global KKT residual 0.003557. That result justified one compatible continuation; no new start or fitting-budget extension was added. Attempt03's initial states and globals are bitwise equal to attempt02's selected solution, and all 144 attempt02 hashes captured by repair01 match.

Attempt03 completed 8 outer updates at J=1925.601338676, down 6.69e-8 from the warm start. Native equal-exposure relative-coordinate RMS is 4.678857434 px. Its physical bound is feasible; state projected gradient is 1.23e-11, complementarity 1.89e-13, active derivatives are smooth, profile curvature is positive with rank 16, and zero of 169,935 free local eigenvalues are negative. The local minimum 124.707 is per-frame Hessian divided by positive exposure row weight; magnitude is not directly comparable with attempt02's weighted census. Global KKT residual 2.5103e-5 still fails the 1e-6 threshold.

History records 62 accepted and 35 rejected updates, including 27 accepted joint outer steps. Joint-proposal rejection counts are 810 same-objective line-search rejections and 5 nonpositive shared-proposal-curvature events. Only one of the maximum six polish proposals was attempted; it was unaccepted and recorded 36 line-search plus 5 nonpositive-curvature rejections. It ended with global residual 2.5103e-5. Attempt03 runtime was 156.16 seconds. Compact progress remains 270/300 scored, 30 unavailable, zero unresolved; its coordinate RMS is 29.3673 px. This remains historical and is not the current certificate.

Attempt02's common start also failed certification. Its histories contain thousands of nonlinear-bound proposal rejections and heavily damped joint steps. Its old certificate conflated an unavailable profile calculation with failed local curvature. The repaired certificate records local and profile status independently, uses `NOT_EVALUATED` for unavailable profile curvature, and blocks smooth certification at active endpoint ties. Constraint derivatives now use selected-branch forward automatic differentiation; the outward feasibility bounds remain unchanged.

### Attempt04 current certified correction

Attempt04 applies one reviewed constrained Newton correction to attempt03's selected saved state. It uses the unchanged objective, population, covariance, anchors, priors, references, state domains and user-authorized one-degree physical bound. It is a single local correction, not a new fit loop or subset inference.

| Certificate item | Attempt04 |
|---|---:|
| Global scaled KKT residual | 4.2363e-10 (threshold 1e-6) |
| State projected gradient | 1.3156e-13 |
| Complementarity | 3.1461e-14 (threshold 1e-6) |
| Active physical constraints | 3; feasible, minimum normalized slack 0 |
| State data rank | 89,175 / 89,175 rank 2 |
| Free-state observed curvature | 0 negative of 169,935; minimum 124.707 per positive row-weight normalization |
| Constrained observed profile | Positive, rank 16; minimum normalized eigenvalue 4.9430e-5 |
| Optical domains | P1 and P4 valid |
| Full objective J | 1925.6013386759976 |
| Native relative-coordinate RMS | 4.678857454 px |

Direct objective change is −5.9962e-12 and maximum predicted movement is 6.19e-6 px; maximum state changes are 1.70e-7 degrees and 6.28e-7 D. RMS changes by about 2e-8 px, so no material empirical improvement is claimed. Independent public-optics reconstruction on all valid rows matches components within 2.28e-13, predictions within 3.41e-13 px, g within 7.77e-16, and exposure means within 1.96e-14. All attempt03 parent files remain hash-identical.

Attempt04 flags are `fit_complete=true`, `fit_certified=true`, `crosscheck_complete=false`, and `comparison_complete=false`. Compact evidence remains the attempt03 diagnostic: 300 scheduled, 270 scored, 30 unavailable; 90 complete triples across 20 exposures; E=41.8138 px, Gtheta=1.50113 degrees, GA=1.31867 D. The two largest rows contribute 94.3568% of E² and 91.8257% of Gtheta²; no cause is asserted and no row was trimmed. G8 awaits separate authorization.

Attempt04 has `fit_complete=true` and `fit_certified=true`; `crosscheck_complete=false` and `comparison_complete=false`. This is a limited numerical certificate under the declared model and user-authorized bound. No automated test suite was added or run for attempt04; verification consists of the full saved-point certificate and independent NumPy/public-optics reconstruction.

## Next action

1. Preserve attempt04 as the current numerically certified checkpoint and retain its GO_WITH_LIMIT restrictions.
2. Keep G8 paused until its separate authorization. If authorized, evaluate the declared full withheld-point schedule with the frozen attempt04 model, retaining every missing/unresolved outcome; do not treat the historical attempt03 compact scores as an attempt04 evaluation.

Open scientific limits remain: empirical omega1/omega4 references are not independently established optical zeros; omega1 continuous uncertainty is unquantified; G3 discrete alternatives and physical accommodation calibration remain unresolved. Large compact prediction errors also remain unexplained.

## Files and execution handoff

- Scientific definitions: [Theory](docs/Theory.md), [estimator plan](docs/ESTIMATOR_PLAN.md), [stage gates](docs/STAGE_GATES.md), [Stage 4 plan](docs/stages/04_CENTER_AND_DM0_CALIBRATION.md).
- Current evidence: [G7 attempt04 report](experiments/distortion_model/stage_04_center_and_dm0_calibration/results/g7_attempt_04/STAGE_REPORT.md), [summary/certificate](experiments/distortion_model/stage_04_center_and_dm0_calibration/results/g7_attempt_04/summary.json), [checkpoint](experiments/distortion_model/stage_04_center_and_dm0_calibration/results/g7_attempt_04/checkpoint.json), [arrays](experiments/distortion_model/stage_04_center_and_dm0_calibration/results/g7_attempt_04/fitted.npz), [adoption verification](experiments/distortion_model/stage_04_center_and_dm0_calibration/results/g7_attempt_04/verification.json), [independent public-optics audit](experiments/distortion_model/stage_04_center_and_dm0_calibration/results/g7_attempt_04/independent_public_optics_audit.json), [report-only source history](experiments/distortion_model/stage_04_center_and_dm0_calibration/results/g7_attempt_04/reporting_source_history.json).
- Historical evidence: [attempt03 report](experiments/distortion_model/stage_04_center_and_dm0_calibration/results/g7_attempt_03/STAGE_REPORT.md), [audit](experiments/distortion_model/stage_04_center_and_dm0_calibration/results/g7_attempt_03/mechanical_audit.json), [compact aggregation](experiments/distortion_model/stage_04_center_and_dm0_calibration/results/g7_attempt_03/compact_diagnostics/REPORT.md), [saved-point probe and N1–N4/E1 report](experiments/distortion_model/stage_04_center_and_dm0_calibration/results/g7_polish_repair_01/STAGE_REPORT.md).
- Same-state repair: [repair01 report](experiments/distortion_model/stage_04_center_and_dm0_calibration/results/g7_repair_01/STAGE_REPORT.md), [summary](experiments/distortion_model/stage_04_center_and_dm0_calibration/results/g7_repair_01/summary.json).
- Parent: [G6 attempt 02 checkpoint](experiments/distortion_model/stage_04_center_and_dm0_calibration/results/g6_attempt_02/checkpoint.json). Its ancestry/provenance identifies G5; G7 provenance also records G3 and prior-attempt hashes.
- Historical results: [G7 attempt 01](experiments/distortion_model/stage_04_center_and_dm0_calibration/results/g7_attempt_01/STAGE_REPORT.md) and [attempt 02](experiments/distortion_model/stage_04_center_and_dm0_calibration/results/g7_attempt_02/STAGE_REPORT.md).
- Stage navigation: [README](experiments/distortion_model/stage_04_center_and_dm0_calibration/README.md), [results](experiments/distortion_model/stage_04_center_and_dm0_calibration/docs/RESULTS.md), [progress](experiments/distortion_model/stage_04_center_and_dm0_calibration/docs/PROGRESS.md).
- Implementation: [joint model/solver](distortion_model/joint.py), [bounded solver](distortion_model/centroid_bound.py), [bounded runner](experiments/distortion_model/stage_04_center_and_dm0_calibration/scripts/run_joint_bounded.py), [audit script](experiments/distortion_model/stage_04_center_and_dm0_calibration/scripts/audit_g7_bounded.py), [dependencies](requirements-stage7.txt).
- Repair implementation: [interval automatic derivatives](distortion_model/interval_ad.py), [constrained polishing](distortion_model/polishing.py), [regression and same-state reassessment](experiments/distortion_model/stage_04_center_and_dm0_calibration/scripts/repair_g7.py), [compatible continuation runner](experiments/distortion_model/stage_04_center_and_dm0_calibration/scripts/continue_g7_repaired.py), [saved-point probe](experiments/distortion_model/stage_04_center_and_dm0_calibration/scripts/probe_g7_polishing.py), [one-correction adoption](experiments/distortion_model/stage_04_center_and_dm0_calibration/scripts/accept_g7_polish.py), [independent audit](experiments/distortion_model/stage_04_center_and_dm0_calibration/scripts/audit_g7_polish.py), [saved compact aggregation](experiments/distortion_model/stage_04_center_and_dm0_calibration/scripts/summarize_g7_compact.py).

Each attempt's `source_snapshot/` and `provenance.json` are authoritative. Attempt04 records the reviewed saved-point correction sources and the hash-identical attempt03 optimization parent. Its provenance records one reporting-only summarizer source change after the probe snapshot, with both hashes; no fitting, proposal, or certificate source mismatch was accepted. Preserve the original pending-review probe and historical attempt bytes.

From the repository root, the existing result can be independently reconstructed with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python experiments/distortion_model/stage_04_center_and_dm0_calibration/scripts/audit_g7_polish.py --attempt experiments/distortion_model/stage_04_center_and_dm0_calibration/results/g7_attempt_04
```

The one-correction adoption can be reconstructed from the preserved probe into a fresh output directory with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python experiments/distortion_model/stage_04_center_and_dm0_calibration/scripts/accept_g7_polish.py --output experiments/distortion_model/stage_04_center_and_dm0_calibration/results/g7_attempt_04_replay
```

G8 has not run and awaits separate authorization. The crosscheck/comparison flags remain false; do not describe the historical attempt03 compact scores as an attempt04 evaluation.

Standing user preferences: root agent performs implementation and scientific audit; delegate mechanical work to **gpt-6-luna, low reasoning**. Execute shell commands outside the sandbox using the available escalation mechanism. Prefer vectorized/chunked GPU computation and appropriate parallelism. Keep experiment scripts, results, and reports organized by stage. Preserve unrelated working-tree changes; no commit or push has been requested.
