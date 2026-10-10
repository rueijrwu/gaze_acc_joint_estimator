# G7 certified-results audit — numerical closure and accommodation agreement

**Inspected commit:** `18d8049bda9ae1e417f3f1312f39fff1f404d6c2`  
**Branch:** `exp5_distortion_model`  
**Scope:** Review of saved G7 attempt04, independent numerical verification, and historical attempt03 compact subset diagnostics. No new fitting or G8 inference was performed.

## Executive decision

**Accept G7 as `COMPLETE_CERTIFIED_WITH_LIMIT / GO_WITH_LIMIT`, stop numerical polishing, and request separate authorization for G8 on the frozen attempt04 model.** Numerical certification is not validation of the physical accommodation law, optical reference zeros, or physiological gaze accuracy.

## Numerical certification

Attempt04 is one constraint-consistent Newton correction from attempt03, not a new full-data fit. All 89,175 valid frames were included and 10,915 scheduled rows remain unavailable. The unchanged declared objective, covariance, anchors, priors, and user-authorized one-degree centroid-response bound were preserved.

| Quantity | Attempt04 |
|---|---:|
| Full objective J | 1925.6013386759976 |
| Global scaled KKT residual | 4.24e-10 (threshold 1e-6) |
| State projected gradient | 1.32e-13 |
| Constraint complementarity | 3.15e-14 |
| Physical constraint | Feasible; 3 active rows |
| State data Jacobian | Rank 2 on 89,175 valid rows |
| Free-state curvature | Positive |
| Constrained profile curvature | Positive, rank 16 |
| Equal-exposure native relative-coordinate RMS | 4.678857454 px |

The repair uses active-constraint Lagrangian curvature and independently checks null-space versus bordered-system steps. The saved independent public-optics reconstruction agrees with predictions to 3.41e-13 px and objective components to 2.28e-13. The correction changes the objective by only -5.9962e-12, maximum predicted pixels by 6.19e-6 px, frame gaze by at most 1.70e-7 deg, and frame accommodation by at most 6.28e-7 D. **It establishes numerical closure, not material empirical improvement.** A full automated repository test suite was not run.

## The compact cross-agreement remains the main scientific concern

**All compact metrics below are from attempt03, not a new attempt04 inference.** Ninety eligible frame triples across 20 exposures were scored; 270 of 300 omission slots were scored and 30 were unavailable.

| Metric | Equal-exposure RMS |
|---|---:|
| Held-P4 2D vector reconstruction E | 41.8138 px |
| Same-frame gaze disagreement Gtheta | 1.50113 deg |
| Same-frame accommodation disagreement GA | 1.31867 D |

The adjacent capture2 frames at rows 6799 and 6800 contribute **94.3568% of E² and 91.8257% of Gtheta²**, but only **31.0541% of GA²**. Thus the two large outliers do **not** explain most accommodation disagreement. Do not trim them or assume they are transitions without inspecting original correspondence and neighboring measurements.

Descriptive compact partitions, with no change to the primary denominator:

| Partition | Frames | E RMS (px) | Gtheta RMS (deg) | GA RMS (D) |
|---|---:|---:|---:|---:|
| Interior schedule positions | 54 | 10.5433 | 0.431867 | 1.14573 |
| No subset state at bound | 61 | 5.6313 | 0.171593 | 0.726907 |
| Any subset at bound | 29 | 72.7074 | 2.61642 | 2.02696 |

The interior and bound-free partitions overlap; they are descriptive, not independent experiments. Accommodation disagreement remains notable even without bound clipping.

Point-specific evidence: omitted P4 index 1 has vector-error p95 approximately **39.79 px**, versus 11.35 and 11.69 px for indices 0 and 2. Its interior x-coordinate RMS is roughly 13.53–17.09 px versus y-coordinate RMS 3.68–3.93 px. This may reflect retained-pair conditioning, a reference/rotation error, or point-specific measurement behavior; it does not by itself justify radial distortion or a new accommodation exponent.

## Metric and interpretation safeguards

- Full-fit RMS (4.678857454 px) is a ten-relative-coordinate full-calibration residual, **not** a held-P4 cross-agreement score.
- Historical compact pooled scalar-coordinate RMS (29.3673 px) is **not** the same definition as the 41.8138 px equal-exposure 2D vector-error RMS.
- The compact result was not rerun for attempt04; tiny forward-prediction changes do not guarantee identical retained-pair inverse branches or bound classifications.
- Rank-two local Jacobians establish numerical local rank, not useful physical accommodation precision.
- The one-degree physical bound limits accommodation-induced apparent horizontal gaze in the centroid law, not actual gaze error or subset Gtheta.
- Both operational optical symmetry references and their relation to physical symmetry zeros remain unresolved. Numerical certification does not validate the optical parity assumptions.

## Next authorized decision

**Do not run another G7 optimization or expand DM0.** Propose one frozen-attempt04 G8 evaluation, subject to separate authorization, preserving the full 100,090-row schedule and three P4 omissions per row (300,270 scheduled omission slots). Keep missing/unavailable/unresolved counts and all original endpoints.

G8 should report equal-exposure E, Gtheta and GA together with medians/quantiles, signed point/axis errors, boundary/bound strata, top contributors, and retained-pair conditioning. On a predeclared subset, compare the same timestamps using (a) anchored calibration states, (b) label-free all-three inversion, and (c) the three two-P4 inversions, all with the same frozen global model. This distinguishes anchor dependence from measurement-subset inconsistency.

Use evidence to choose **one** later optical follow-up: correspondence repair; operational-reference/rotation sensitivity; identifiable radial response (DM1); a single accommodation response-law comparison with fresh matched calibration; or an explicit weak-information conclusion. RMS is a diagnostic, not a hard acceptance threshold.

**Gate flags:** `fit_certified=true`; `crosscheck_complete=false`; `comparison_complete=false`.

## Primary evidence

- [Attempt04 report](../../experiments/distortion_model/stage_04_center_and_dm0_calibration/results/g7_attempt_04/STAGE_REPORT.md)
- [Attempt04 verification](../../experiments/distortion_model/stage_04_center_and_dm0_calibration/results/g7_attempt_04/verification.json)
- [Independent public-optics audit](../../experiments/distortion_model/stage_04_center_and_dm0_calibration/results/g7_attempt_04/independent_public_optics_audit.json)
- [Saved compact empirical diagnostics](../../experiments/distortion_model/stage_04_center_and_dm0_calibration/results/g7_polish_repair_01/empirical_diagnostics/REPORT.md)
- [Stage 4 resume pointer](../../experiments/distortion_model/stage_04_center_and_dm0_calibration/docs/PROGRESS.md)

**Review limitation:** This is a source and saved-result audit, not an independent full-array rerun or new empirical calibration.
