# Stage 5 results

**Status: COMPLETE_WITH_LIMIT / GO_WITH_LIMIT.** The frozen-model full-schedule cross-check completed and passed independent numerical and held-input masking audits. The separate model comparison remains incomplete: `comparison_complete=false`.

## Coverage and primary scorecard

The schedule has 100,090 reviewed rows and 300,270 held-point slots. All slots and their statuses are retained. Across the three omission routes, 267,565 slots had valid retained inputs and certified inversions; 267,525 had a measured omitted point and were scored. There were 32,705 retained-input-unavailable slots, 41,016 bound-active inversions overall (40,981 with scored held measurements), and no ambiguous or unresolved inversions. The 40 difference between certified inversions and scored slots consists of held measurements that were unavailable for scoring.

There are 89,175 eligible complete triples covering all 20 exposures. Equal-exposure scorecard results are:

| Measure | Result | Meaning |
|---|---:|---|
| E | 12.40853 px | RMS of three held-point vector errors, within-exposure squared mean then equal exposure weights |
| Gθ | 0.477866° | RMS of within-frame gaze disagreement among the three omissions, with the same exposure weighting |
| GA | 1.101718 D | RMS of within-frame accommodation disagreement among the three omissions, with the same exposure weighting |

The 267,525 scored held-point errors have vector RMS 12.53097 px, coordinate RMS 8.86073 px, signed x/y bias [-2.19782, 0.523997] px, and axis RMS [10.4513, 6.91347] px. Error summaries by held point and all exposure/block/boundary strata are in [`metrics.json`](../results/attempt_01/metrics.json).

| Omitted P4 | Retained valid / certified | Held valid | Scored | Bound scored | Vector RMS (px) | Median / p95 / max (px) | x / y bias (px) | x / y RMS (px) |
|---:|---:|---:|---:|---:|---:|---|---|---|
| 0 | 89,175 | 89,215 | 89,175 | 11,224 | 9.28256 | 5.43959 / 11.2517 / 274.285 | -1.44382 / 1.38260 | 6.96109 / 6.14078 |
| 1 | 89,215 | 89,175 | 89,175 | 8,948 | 17.7694 | 5.97717 / 37.2896 / 341.662 | -2.34625 / -0.09909 | 15.7093 / 8.30462 |
| 2 | 89,175 | 89,215 | 89,175 | 20,809 | 8.31622 | 4.42239 / 11.1800 / 272.441 | -2.80339 / 0.28848 | 5.69626 / 6.05905 |

## Three-way state comparison

The predeclared diagnostic schedule contains 110 rows, including 100 rows with certified, unambiguous all-three inversions. Selected neighboring capture-2 rows are deliberately overrepresented in this diagnostic; it is not a representative full-period sample. On the 100 matched rows, all-three label-free inversion and anchored calibration states differ by 0.0402032° equal-exposure gaze RMS and 0.0713189 D accommodation RMS.

| Two-point route relative to all-three route | Matched rows | Equal-exposure gaze RMS difference (°) | Equal-exposure accommodation RMS difference (D) |
|---|---:|---:|---:|
| Omit P4_0 | 100 | 0.0369406 | 0.706610 |
| Omit P4_1 | 100 | 2.92557 | 1.04813 |
| Omit P4_2 | 100 | 0.179739 | 1.23879 |

These are same-frame application comparisons, not target-label accuracy estimates. All-three point cost was lower than anchored point cost on all 100 matched rows, with median difference -1.10176 in the declared covariance-weighted objective. The diagnostic subset is not used as the whole-period primary scorecard.

On the 88 matched rows outside the predeclared capture-2 neighbor case, all-three minus calibration RMS is 0.0399256 degrees / 0.0712533 D; omission-minus-all-three RMS differences are 0.018119 degrees / 0.375236 D (omit P4_0), 0.549161 degrees / 0.520937 D (omit P4_1), and 0.107763 degrees / 1.058980 D (omit P4_2). In the neighbor-enriched 100-row set, omission-1 gaze disagreement is larger (2.92557 degrees); that subset must not be used as a full-period representative.

## Conditional information and measurement diagnostics

Pooled per-frame medians / p95 / maxima are E 5.795 / 23.293 / 275.106 px, Gtheta 0.1676 / 1.0517 / 10.0484 degrees, and GA 0.6719 / 2.3172 / 4.8990 D. The largest complete-triple frame contributes 0.628% of E² and the second largest 0.622%; the leading two Gtheta² contributions are 0.565% and 0.562%. The top two GA² contributions are 0.0322% and 0.0287%. These are descriptive rankings only and do not assign a cause.

At the same all-three state, median conditional accommodation information retained by the two-point routes was 44.9% (omit 0), 49.8% (omit 1), and 7.92% (omit 2) of the all-three metric. Median absolute gaze/accommodation column cosine was 0.266, 0.360, and 0.816 respectively. These quantities describe the declared covariance weighting and local model geometry; they are not calibrated physical precision or accuracy.

The full population has zero measured P4 signed-area orientation conflicts among 89,175 complete rows. The 12 predeclared capture-2 neighborhood rows (source rows 6794–6805) all have three available P4 points and no orientation conflict; most subset inversions are at state bounds. Exact measurements, states, errors, and statuses are retained in [`neighbor_case.json`](../results/attempt_01/neighbor_case.json). No row or endpoint was trimmed.

The full-period primary GA remains 1.101718 D, and the exposure-averaged interior GA is 1.101624 D. The pooled no-bound frame RMS (0.752604 D) uses a different weighting and population; exposure 4 is absent, so this is not a substitute for the equal-exposure 20-condition result. The per-exposure and interval-block held-point tables in [`point_exposure_diagnostics.json`](../results/attempt_01/point_exposure_diagnostics.json) are reporting-only pooled groups; the original [`metrics.json`](../results/attempt_01/metrics.json) is unchanged and defines the primary scorecard. The report-only [`slot_context.npz`](../results/attempt_01/slot_context.npz) records frozen-model calibration status and calibration-frame availability per slot, separately from inference status. Pooled full-population per-frame distributions are in [`native_frame_distributions.json`](../results/attempt_01/native_frame_distributions.json); their quantiles are pooled, not equal-exposure results.

## Reproducibility and limits

The [independent audit](../results/attempt_01/independent_audit.json) passed for all 300,270 slots, with unchanged parent and source hashes. Its predicted errors matched exactly, and recomputed costs differed by at most 1.67e-10. The preflight verified no omitted-point leakage for huge finite and unavailable-NaN perturbations through the full inference loop. The reviewed [scientific decision](../results/attempt_01/scientific_review.json) records COMPLETE_WITH_LIMIT / GO_WITH_LIMIT, `diagnostic_three_way_complete=true`, and `comparison_complete=false`; the readable [review](../results/attempt_01/SCIENTIFIC_REVIEW.md) gives its rationale and proposed follow-up. Mechanical hashes and group reconciliation are in [completion verification](../results/attempt_01/completion_verification.json). Full run, preflight, audit, and report commands and console evidence are linked from [`PROGRESS.md`](PROGRESS.md).

This is internal agreement on recordings also used for calibration. It does not establish physiological accuracy, optical-zero accuracy, or a universal winner. Operational reference and rotation sensitivity remain unresolved. The decision is GO_WITH_LIMIT for this empirical cross-check. One later follow-up is proposed: a matched operational-reference sensitivity comparison using the adjacent P4 reference `omega4=-5.0148989655383795°` against the current `omega4=-10.016974132845107°`, with affected G4–G8 stages refit consistently from matched baselines. Do not change separate rotation parameters at the same time. This proposal was not executed and is not authorized by this report. No candidate-model comparison was run; `comparison_complete=false`.
