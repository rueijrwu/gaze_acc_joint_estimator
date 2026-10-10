# Stage 5 report: frozen-model cross-check and model decision

**Status: COMPLETE_WITH_LIMIT / GO_WITH_LIMIT.** The authorized G8 evaluation is complete on the full reviewed schedule. The scientific disposition is alignment/identifiability unresolved. `comparison_complete=false`; no candidate-model comparison or refit was performed.

## Parent, schedule, and command

G8 used the frozen certified [Stage 4 attempt04 checkpoint](../../stage_04_center_and_dm0_calibration/results/g7_attempt_04/checkpoint.json). The runner verified checkpoint/model/ancestry hashes and confirmed that frame states equal the parent checkpoint. The independent audit verified that the parent files remained unchanged. There was no optimization of shared parameters.

From repository root, the run command was:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python experiments/distortion_model/stage_05_crosscheck_and_model_decision/scripts/run.py --output experiments/distortion_model/stage_05_crosscheck_and_model_decision/results/attempt_01 --chunk-size 8192
```

It completed in 191.64 seconds using CuPy float64 and one P100 GPU process. Before launch, actual-record benchmarks compared 4,096 and 8,192 row chunks. The 8,192 chunk took 5.037 seconds for 8,192 scored rows and left 11.34 GiB free; 4,096 rows took 5.241 seconds. The 8,192 chunk was selected. The independent audit and saved-result report commands were:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python experiments/distortion_model/stage_05_crosscheck_and_model_decision/scripts/audit.py --attempt experiments/distortion_model/stage_05_crosscheck_and_model_decision/results/attempt_01
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python experiments/distortion_model/stage_05_crosscheck_and_model_decision/scripts/report.py --attempt experiments/distortion_model/stage_05_crosscheck_and_model_decision/results/attempt_01
```

The first preflight tested actual-record legacy/compacted and CPU/GPU agreement plus huge-finite and unavailable-NaN held-input perturbations. The saved audit and report perform no new inference. No automated repository-wide suite was run.

## Coverage and primary results

The run scheduled 100,090 rows and retained all 300,270 omission slots. Across the three routes, 267,565 valid retained-input slots yielded certified inversions; 267,525 had an available held measurement and were scored. There were 32,705 retained-input-unavailable slots, 41,016 bound-active inversions overall (40,981 scored), and zero ambiguous or unresolved inversions. The 40 certified-but-unscored cases had unavailable held observations. Missing input, bound-active inversion, and unscored held output remain distinct slot outcomes.

There are 89,175 complete eligible triples across all 20 exposures. Primary equal-exposure metrics use within-exposure squared means followed by equal weighting of all 20 exposures:

| Measure | Value |
|---|---:|
| E, held-point vector error | 12.40853 px |
| Gtheta, within-frame gaze disagreement | 0.477866 degrees |
| GA, within-frame accommodation disagreement | 1.101718 D |

Across all 267,525 scored errors, vector RMS is 12.53097 px and coordinate RMS is 8.86073 px. Signed x/y bias is [-2.19782, 0.523997] px and x/y RMS is [10.4513, 6.91347] px. By held point, vector RMS values are 9.28256 px (omit P4_0), 17.7694 px (omit P4_1), and 8.31622 px (omit P4_2). Omit P4_1 has x RMS 15.7093 px and p95 error norm 37.2896 px.

The per-exposure and interval-block held-point tables in [`point_exposure_diagnostics.json`](../results/attempt_01/point_exposure_diagnostics.json) are reporting-only pooled groups. The original full-run aggregate [`metrics.json`](../results/attempt_01/metrics.json) remains unchanged and defines the primary equal-exposure scorecard. The report-only [`slot_context.npz`](../results/attempt_01/slot_context.npz) explicitly records calibration status and calibration-frame availability per slot; it does not replace inference status. Pooled per-frame distributions are in [`native_frame_distributions.json`](../results/attempt_01/native_frame_distributions.json) and are not equal-exposure statistics. Equal-exposure GA is 1.101718 D; its interior value is 1.101624 D. The pooled no-bound frame RMS is 0.752604 D, but exposure 4 is absent and this population/weighting differs. Do not substitute it for the full 20-exposure score.

## Three-way matched state comparison

The predeclared state diagnostic has 110 scheduled rows, including the 100 rows with certified, unambiguous all-three inversion. It intentionally overrepresents neighboring capture-2 rows; it is not representative of the full period. On the 88 matched rows outside that neighbor case, all-three minus calibration RMS is 0.0399256 degrees / 0.0712533 D, and omission-minus-all-three RMS differences are 0.018119 degrees / 0.375236 D (omit P4_0), 0.549161 degrees / 0.520937 D (omit P4_1), and 0.107763 degrees / 1.058980 D (omit P4_2). On those same 100 timestamps, all-three inversion differs from anchored calibration by equal-exposure RMS 0.0402032 degrees in theta and 0.0713189 D in accommodation.

| Two-point inversion relative to all-three | Matched rows | Theta RMS difference | Accommodation RMS difference |
|---|---:|---:|---:|
| Omit P4_0 | 100 | 0.0369406 degrees | 0.706610 D |
| Omit P4_1 | 100 | 2.92557 degrees | 1.04813 D |
| Omit P4_2 | 100 | 0.179739 degrees | 1.23879 D |

All-three point cost is below anchored point cost on each of the 100 matched rows; median difference is -1.10176 in the declared covariance-weighted objective. Calibration includes full-exposure mean anchors; application inversions use frozen globals, measured points only, and no labels or state warm starts. This is a predeclared diagnostic, not a full-period representativeness or accuracy claim.

At the same all-three state, median conditional accommodation information retained was 44.9% for omit P4_0, 49.8% for omit P4_1, and 7.92% for omit P4_2. Median absolute theta/accommodation column cosine was 0.266, 0.360, and 0.816 respectively. These derive from the declared covariance metric and model Jacobians; they are not physically calibrated precision estimates.

## Geometry and retained evidence

Measured P4 triangle orientation matched the frozen positive-domain model on all 89,175 complete rows. The predeclared capture-2 neighborhood, source rows 6794–6805, has all three P4 observations available and no orientation conflicts. Most omission solutions in that neighborhood are at state bounds. The complete per-row measurements, states, errors, statuses and orientation flags are in [`neighbor_case.json`](../results/attempt_01/neighbor_case.json).

All 300,270 slot outcomes, identifiers, states, status flags, predictions, and errors are retained in [`crosscheck.npz`](../results/attempt_01/crosscheck.npz). The predeclared three-way schedule is recorded in [`diagnostic_schedule.npz`](../results/attempt_01/diagnostic_schedule.npz); its states and conditioning diagnostics are separate saved artifacts. The report generator archives its own script and provenance separately; it does not modify core fit sources or primary metrics.

## Independent verification

The [preflight](../results/preflight_01/summary.json) passed actual-record compact-vs-legacy, CPU/GPU and held-input noninterference checks. The full-run [independent audit](../results/attempt_01/independent_audit.json) passed on all 300,270 slots, verified unchanged parent and archived sources, and reproduced all three held-point errors exactly (maximum absolute error difference 0 px). Recomputed costs differed by at most 1.67e-10. The equal-exposure E/Gtheta/GA values reproduce exactly from the saved arrays. The reviewed [scientific decision](../results/attempt_01/scientific_review.json) records COMPLETE_WITH_LIMIT / GO_WITH_LIMIT, with the three-way diagnostic complete and model comparison incomplete. Full-run console output, audit output, report output, source snapshot, provenance, and input hashes are retained in attempt01.

## Decision and limits

The result is GO_WITH_LIMIT: the full frozen-model empirical cross-check is complete, with substantial route-dependent state differences and weak conditional accommodation information when P4_2 is omitted. The 100-row three-way diagnostic is intentionally neighbor-enriched, and held-point errors are internal residuals on recordings used in calibration. The results do not establish physiological accuracy, absolute optical zeros, or that any one route is a universal winner. Fixed operational references, rotation/alignment sensitivity, and accommodation identifiability remain unresolved.

One follow-up is proposed: an operational-reference sensitivity comparison using the adjacent P4 reference `omega4=-5.0148989655383795°` against the current `omega4=-10.016974132845107°`. Reinitialize and refit affected G4–G8 stages consistently from matched baselines; do not change separate rotation parameters at the same time. The detailed proposal is in [`SCIENTIFIC_REVIEW.md`](../results/attempt_01/SCIENTIFIC_REVIEW.md). It was not executed or authorized by the G8 decision. `comparison_complete=false`; G8 does not authorize a candidate-model comparison, model expansion, or a refit.
