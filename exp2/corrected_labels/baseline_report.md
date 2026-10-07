# Joint Algorithm Baseline Performance

## Holdout-3 Fold (5 heldout fixations)

**Held-out 3 D cohort metrics (accommodation demand 3.0 diopters):**
- Gaze bias RMS: 0.161 deg (expected 0.161)
- Pooled within-fixation SD: 0.183 deg (expected 0.183)
- All-frame RMSE: 0.248 deg (expected 0.248)

## Full Fold (all 20 fixations)

**In-sample 3 D cohort metrics (accommodation demand 3.0 diopters):**
- Gaze bias RMS: 0.207 deg (expected 0.207)
- Pooled within-fixation SD: 0.180 deg (expected 0.180)
- All-frame RMSE: 0.280 deg (expected 0.280)

## Convergence Status

- Holdout3 training: converged in 7 iterations (13.6 seconds)
- Full training: converged in 8 iterations (23.3 seconds)
- Both used robust regression with corrected target overrides

## Per-Fixation Signed Gaze Bias (Holdout-3, Held-Out 3 D Cohort)
- Fixation 10: +0.138563 deg
- Fixation 11: -0.005135 deg
- Fixation 12: -0.147631 deg
- Fixation 13: +0.142186 deg
- Fixation 14: +0.261821 deg

## Important Caveat

In-sample gaze predictions are soft-anchored to corrected targets in the cost function, so only held-out predictions represent fair model assessment. Reported biases on in-sample data reflect target consistency, not absolute accuracy.
