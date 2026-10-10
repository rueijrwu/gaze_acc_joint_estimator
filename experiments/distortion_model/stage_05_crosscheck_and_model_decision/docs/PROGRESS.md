# Stage 5 progress

**Current status: COMPLETE_WITH_LIMIT / GO_WITH_LIMIT.** G8 completed on the full reviewed population. The frozen-model cross-check is complete; `comparison_complete=false` because no separate candidate-model comparison was run.

## Frozen parent

The run used the certified Stage 4 checkpoint at [`g7_attempt_04`](../../stage_04_center_and_dm0_calibration/results/g7_attempt_04/checkpoint.json). The run records parent file and checkpoint hashes; the saved-result audit confirms the parent remained unchanged. No calibration refit or G8 model change was performed.

## Run and verification

Command from repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python experiments/distortion_model/stage_05_crosscheck_and_model_decision/scripts/run.py --output experiments/distortion_model/stage_05_crosscheck_and_model_decision/results/attempt_01 --chunk-size 8192
```

The attempt completed in 191.64 seconds. Before the run, actual-record benchmarks selected chunk size 8192: 8,192 rows took 5.037 seconds, compared with 5.241 seconds for 4,096, with about 11.34 GiB GPU memory free after the larger batch.

The independent saved-result audit passed: 100,090 scheduled rows and 300,270 slots; parent unchanged; archived sources match; slot identity and availability match. It reconstructed all three held-point predictions and equal-exposure metrics without new inference. The report generator then wrote [`REPORT.md`](../results/attempt_01/REPORT.md), [`metrics.json`](../results/attempt_01/metrics.json) for immutable full-run aggregates, and [`point_exposure_diagnostics.json`](../results/attempt_01/point_exposure_diagnostics.json) for separately generated per-condition/block views.

Preflight also passed legacy-versus-compacted CPU/GPU comparisons and omitted-input masking with both large finite values and unavailable NaNs. Its 1,024-row timing was 5.307 seconds. The full run produced zero unresolved or ambiguous inversions.

## Current result and next action

The full-schedule metrics and limitations are summarized in [`RESULTS.md`](RESULTS.md), with detailed evidence in [`STAGE_REPORT.md`](STAGE_REPORT.md) and the [scientific review](../results/attempt_01/SCIENTIFIC_REVIEW.md). G8 does not resolve alignment and identifiability limits. One operational-reference sensitivity comparison is proposed using the adjacent P4 reference values recorded in the review, with affected G4–G8 stages refit consistently from matched baselines; it does not vary rotation parameters simultaneously. That follow-up is not authorized by this stage report and was not run under G8.
