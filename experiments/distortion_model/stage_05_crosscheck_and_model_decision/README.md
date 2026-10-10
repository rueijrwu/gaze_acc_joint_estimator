# Stage 5: Cross-check and model decision

**Status: COMPLETE_WITH_LIMIT / GO_WITH_LIMIT.** G8 evaluated the frozen, certified Stage 4 attempt04 model over the full reviewed schedule. The calculations are complete; the separate cross-model comparison is not: `comparison_complete=false`.

## Authority and evidence

- Stage criteria: [`docs/stages/05_CROSSCHECK_AND_MODEL_DECISION.md`](../../../docs/stages/05_CROSSCHECK_AND_MODEL_DECISION.md)
- G7 audit scope: [`docs/audits/G7_CERTIFIED_RESULTS_AUDIT.md`](../../../docs/audits/G7_CERTIFIED_RESULTS_AUDIT.md)
- Frozen parent checkpoint: [`Stage 4 attempt04 checkpoint`](../stage_04_center_and_dm0_calibration/results/g7_attempt_04/checkpoint.json)
- Run summary and independent reconstruction: [`attempt01`](results/attempt_01/summary.json), [`scientific review`](results/attempt_01/scientific_review.json), [`independent audit`](results/attempt_01/independent_audit.json)
- Readable run report: [`REPORT.md`](results/attempt_01/REPORT.md)
- Environment pins: [`requirements-stage8.txt`](../../../requirements-stage8.txt)

The schedule retained 100,090 rows and 300,270 held-point slots. There were 89,175 complete eligible triples across all 20 exposures. Equal-exposure metrics were E=12.40853 px, gaze disagreement=0.477866 degrees, and accommodation disagreement=1.101718 D. The route comparison and limitations are in [`docs/RESULTS.md`](docs/RESULTS.md).

## Reproduction

From the repository root, reproduce into a fresh output directory with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python experiments/distortion_model/stage_05_crosscheck_and_model_decision/scripts/run.py --output experiments/distortion_model/stage_05_crosscheck_and_model_decision/results/attempt_01_replay --chunk-size 8192
```

Independent saved-result verification and report generation are recorded in [`docs/PROGRESS.md`](docs/PROGRESS.md). No refit was performed, and the Stage 4 attempt04 checkpoint remains unchanged.

## Files

- `scripts/`: full-schedule runner, saved-result audit, and report generation.
- `docs/PROGRESS.md`: current gate, result pointer, and one follow-up.
- `docs/RESULTS.md`: reviewed summary and limits.
- `docs/STAGE_REPORT.md`: detailed coverage, reconstruction, and interpretation.
- `results/attempt_01/`: full outputs, source snapshot, console logs, audit and report artifacts.
