# Full-position experiment index

The latest audit follow-up is in
[phase83_audit_followup_v1/RESULTS.md](phase83_audit_followup_v1/RESULTS.md), with
support-enriched cross-check scorecards and frozen common/differential-y tests.
The original saved-model cross-check is in
[crosscheck_v1/RESULTS.md](crosscheck_v1/RESULTS.md). Its source is the frozen
polished evaluation in [audit_polished_v1/RESULTS.md](audit_polished_v1/RESULTS.md).
The archived initial results are in [grouped_v2/RESULTS.md](grouped_v2/RESULTS.md).
The frozen baseline, detection files, and prior experiment directories were not
modified.

| Run | Use |
|---|---|
| `grouped_v2/` | Completed five gaze folds and four capture folds, both full-position capacities and a fresh two-channel control |
| `audit_polished_v1/` | All 27 original models reevaluated after inverse polishing; original results preserved; comparison and candidate archives |
| `crosscheck_v1/` | Post-processing of frozen audit frames/holdouts into three excluded-P4 scores per frame, subset-state disagreement, full scheduled-slot coverage, and exact matched interior cohorts; no refits or new predictions |
| `audit_checks_v1/` | Recorded regressions, independent polynomial profiles, fixed-response measurement ablations, training residuals and conditional coefficient sensitivity |
| `joint_sensitivity_v1/` | Completed Phase 8.2 grouped joint-calibration sensitivity: 144 task outcomes, 143 accepted and one retained uncertified calibration; see [`RESULTS.md`](joint_sensitivity_v1/RESULTS.md), [`verification.json`](joint_sensitivity_v1/verification.json), and the [root audit](../../AUDIT_REPORT.md) |
| `phase82_audit_transitions_v1/`, `axis_anchor_transitions_v1/` | Exact matched point/frame boundary transitions, training-scale diagnostics and exposure memberships |
| `phase82_strict_retry_v1/` | Training-only continuation of the original rejected trajectory; historical failure and missing evaluation preserved |
| `axis_anchor_sensitivity_v1/` | Separate gaze/accommodation anchor strengthening, 36 accepted fits |
| `phase83_retained_channels_v1/` | Frozen baseline27/strong-anchor37 retained-x versus xy comparison; original eight-frame population and branch archives |
| `phase83_audit_followup_v1/` | Shared scorecard/support backfill, 54 common/differential-y tasks, exact xy equivalence control, signed residual and support cohorts, supplemental x profiles, scripts and verification; no model promotion |
| `prior_reference_v2/` | Full-development, 24 original rows per fixation, prior 0.001; converged models |
| `zero_prior_v2/` | Matched development sampling without a coefficient prior; all fits reached their evaluation budgets and remain failed checkpoints |
| `gpu_check27/` | Optional CuPy solver versus scalar reference, 32 real-data full/subset checks; agreement on these cases |
| `gpu_check37/` | Optional acceleration disagreements and CPU refinement audit demonstrating a missed lower-cost inverse branch |
| `blas_benchmark.json` | Calibration-kernel thread-count benchmark |
| `gpu_batch_benchmark.json` | Batch-size timing on representative repeated rows; not new validation observations |

Superseded debug/interrupted runs and duplicate worker outputs were removed.
All 27 primary fitted models, frame/point results, sensitivity checkpoints,
GPU audits, and reproduction scripts remain. Identical evaluation-population
CSVs were consolidated into one losslessly compressed `population.csv.gz` per
fold. Worker configuration/completion is retained in fold `execution.json`
files where available. [Cleanup manifest](cleanup_manifest.json) records the
removed runs and the original population SHA-256 hashes.

The initial study is sampled and exploratory. It does not select a final
deployment model or establish physiological accuracy. Captures 5/6 have not been
used for this study.

The standalone two-channel estimator has no individual-P4 decoder, so it has
no score on the excluded-P4 cross-prediction metric. Nominal-anchor consistency
remains a separate secondary diagnostic. Read the versioned cross-check report
for metric conventions, matched support, boundaries, and missing-slot counts.

## Phase 8.2: joint-training sensitivity (run complete)

The `joint_sensitivity_v1/` study refit the conditional 27- and 37-coefficient
models over the frozen grouped folds from `audit_polished_v1/` under eight
predeclared calibration/reference settings. It retains the uncertified fit,
compares exact matched support against the frozen models and a fresh joint
baseline, and leaves model selection open. See the [results](joint_sensitivity_v1/RESULTS.md),
[verification record](joint_sensitivity_v1/verification.json), root
[audit report](../../AUDIT_REPORT.md), and root [current status](../../CURRENT_STATUS.md), plus
[reproduction and output-schema notes](../../full_position/README.md#phase-82-joint-training-sensitivity).
