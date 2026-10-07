# Three-way excluded-P4 cross-check reporting — 2026-10-07

The saved conditional coordinate models were evaluated on three P4 holdouts per
frame, with subset-state disagreement reported as a separate companion metric.
This `crosscheck_v1` analysis post-processes the frozen
[`audit_polished_v1`](../audit_polished_v1/RESULTS.md) frame and holdout records.
It performs no model prediction, recalibration, latent-state refit, or branch
selection. The source records remain unchanged.

## Evaluation contract and coverage

For each held-out P4 point, the other two P4 measurements and all three P1
measurements define a subset inversion. The held point is scored only after its
subset state and branches have been saved. The three subset estimates overlap in
P4 inputs and share P1 context, so their differences are correlated and are not
independent physiological measurements.

`E_frame` is the square root of the mean of the three squared 2D excluded-point
error norms. `G_theta` and `G_A` are RMS values across the three unordered
pairwise differences in subset-inferred gaze (degrees) and accommodation
(diopters). Equal-fixation aggregation first averages squared frame-level
metrics within each `(fold, capture, fixation)` exposure, gives the 20 exposures
equal weight, and then takes the square root. The report also retains pooled
point-vector RMS; that point-level score is distinct from equal-fixation
complete-frame `E_frame`.

Each split family has 160 scheduled frames and 480 scheduled P4-check slots.
The saved population contains 143 eligible frames and 429 eligible point tests.
The other 17 frames (51 slots) have invalid input and remain visible as null,
unscored records. Both coordinate models have 429 certified, identifiable,
unambiguous scored slots and 143 complete triples in each family.

| Split family | Model | Scheduled frames / slots | Eligible frames / tests | Complete triples | Boundary slots | Shared accommodation clipping frames |
|---|---|---:|---:|---:|---:|---:|
| Gaze holdout | conditional27 | 160 / 480 | 143 / 429 | 143 | 23 | 3 |
| Gaze holdout | conditional37 | 160 / 480 | 143 / 429 | 143 | 79 | 17 |
| Capture/demand holdout | conditional27 | 160 / 480 | 143 / 429 | 143 | 8 | 1 |
| Capture/demand holdout | conditional37 | 160 / 480 | 143 / 429 | 143 | 38 | 10 |

The all-testable set includes eligible boundary solutions, with their bound flags
retained. The interior-only mask requires all three subset estimates for a frame
to be strictly inside the computational bounds (`theta` in `[-20,20]` degrees;
`A` in `[0,6]` D). This mask does not assert empirical training support or
physiological validity. Paired model comparisons use the exact intersection of
both models' IDs, rather than each model's separate interior population.

## Equal-fixation complete-frame results

| Split family | Model | `E_frame` (px) | `G_theta` (deg) | `G_A` (D) |
|---|---|---:|---:|---:|
| Gaze holdout | conditional27 | **3.421** | **0.269** | **0.311** |
| Gaze holdout | conditional37 | 6.700 | 0.509 | 0.929 |
| Capture/demand holdout | conditional27 | **4.014** | **0.281** | **0.319** |
| Capture/demand holdout | conditional37 | 4.307 | 0.325 | 0.392 |

These values average over 20 fixation/capture exposures per split family. They
are coordinate-prediction and raw-state-disagreement summaries, not calibrated
uncertainty scores or pass/fail thresholds.

On the exact common interior-only complete-frame support, the gaze family has
110 frames and the capture/demand family has 125. The associated equal-fixation
summary has 16 gaze and 19 capture/demand exposures:

| Split family | Model | `E_frame` (px) | `G_theta` (deg) | `G_A` (D) |
|---|---|---:|---:|---:|
| Gaze holdout | conditional27 | **3.198** | **0.241** | **0.258** |
| Gaze holdout | conditional37 | 4.699 | 0.408 | 0.377 |
| Capture/demand holdout | conditional27 | **3.924** | **0.276** | **0.317** |
| Capture/demand holdout | conditional37 | 4.097 | 0.339 | 0.413 |

The separate matched interior point-level intersections contain 347 gaze and
388 capture/demand point tests. Those point counts do not describe the complete
frame cohorts above. On the full 429-point common support, the pooled
point-vector RMS is 3.423/6.620 px for gaze and 4.062/4.353 px for
capture/demand (`conditional27`/`conditional37`). These reproduce the prior
polished audit's pooled point errors and use a different weighting convention
from `E_frame`.

## Boundary-only diagnostics

A boundary slot is a scored subset estimate at a computational bound. A
boundary-only complete frame has all three checks scored and at least one subset
at a bound. Its point-level slot counts and complete-frame counts are:

| Split family | Model | Boundary slots | Boundary-only complete frames | Boundary-only equal-fixation `E_frame` (px) | `G_theta` (deg) | `G_A` (D) |
|---|---|---:|---:|---:|---:|---:|
| Gaze holdout | conditional27 | 23 | 12 | 3.572 | 0.234 | 0.248 |
| Gaze holdout | conditional37 | 79 | 33 | 10.955 | 0.734 | 1.740 |
| Capture/demand holdout | conditional27 | 8 | 6 | 6.528 | 0.160 | 0.138 |
| Capture/demand holdout | conditional37 | 38 | 17 | 6.686 | 0.267 | 0.284 |

Shared accommodation clipping is a stricter descriptive flag: all three subset
states clip at the same accommodation edge within the declared `1e-5` physical
bound tolerance. It occurs in 3/17 gaze and 1/10 capture/demand frames for
conditional27/conditional37. Apparent state agreement from clipping does not
show that the state explains the geometry.

In the gaze `-10` fold for capture 2, conditional37 has five frames with
`G_A` from 4.22 to 4.47 D and `E_frame` from 18.71 to 22.47 px. All seven valid
frames at this fixation contain a boundary subset estimate. The other two have
shared accommodation clipping and `G_A=0`, while `E_frame` is 13.69 and 16.04
px. Conditional27 has three clipped frames with zero or nearly zero
`G_A` and `E_frame` from 4.02 to 4.44 px. These cases show why state agreement
must be considered with excluded-point prediction and bound coverage.

## Interpretation and limits

Conditional27 has lower equal-fixation RMS for `E_frame`, `G_theta`, and `G_A`
in both split families and on the common interior complete-frame cohorts.
This describes the current sampled comparison. Model medians and tails do not
all order the same way, and conditional37 has substantially more boundary
slots. The result does not prove universal superiority of conditional27.

The standalone two-channel control has no function that predicts an individual
withheld P4 point. Its primary cross-prediction score is unavailable; its
nominal-anchor consistency remains a secondary calibration diagnostic and cannot
rank it on this metric. Agreement among subset estimates can retain shared bias.
Claims of absolute physiological accuracy require independent physiological
references. None are supplied by this experiment.

The next experiment is Phase 8.2: jointly refit latent states and coefficients
while varying fixation-anchor constraints, coefficient priors, and reference
covariance. This would test training sensitivity that fixed-state coefficient
refits cannot measure. It has not been performed in this cross-check phase.

## Saved artifacts and verification

- [`crosscheck_summary.json`](crosscheck_summary.json) contains coverage, metric
  summaries, and exact matched ID masks.
- [`crosscheck_frames.csv`](crosscheck_frames.csv) has one record per selected
  frame, including complete-triple and pairwise-state metrics.
- [`crosscheck_points.jsonl.gz`](crosscheck_points.jsonl.gz) has three explicit
  scheduled slots per frame. It retains validity, certification, rank,
  conditioning, bounds, source reason, retained subset cost, and branch details;
  unavailable scores remain null.
- [`crosscheck_agreement.png`](crosscheck_agreement.png) plots complete-frame
  coordinate error against raw subset-state disagreement, marking boundary and
  interior frames separately.
- [`verification.json`](verification.json) records the exact verification
  procedure and results: the 219-file, 22,490,019-byte polished source tree had
  the same SHA-256 tree digest before and after processing; the nine compressed
  population members match the existing cleanup manifest; and the full pytest
  run passed 42 tests.

The full pytest command is recorded in [verification.json](verification.json).
The reusable command for another fresh output path is in the
[full-position package README](../../../full_position/README.md). The previous
[polished audit report](../audit_polished_v1/RESULTS.md) and the original
[`grouped_v2` results](../grouped_v2/RESULTS.md) remain as historical records.
