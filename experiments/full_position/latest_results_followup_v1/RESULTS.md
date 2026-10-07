# Latest frozen-result follow-up

This is a paired development comparison of saved baseline27/strong_anchor37 predictions using xy versus x plus differential-y, followed by one training-scaled shared-y covariance sensitivity. It does not recalibrate either response or promote a new model.

The schedule was rebuilt from the selected rows in each frozen `population.csv.gz`. Every comparison uses exact shared scheduled frame and held-point identities; paired errors are candidate minus reference and equal-weight exposures. Joint support means both arms meet the cohort rule at the same point/frame IDs.

## Direct differential-y versus xy

The signed-tail file reports x and y residual mean, signed quantiles, absolute 95th percentile and RMS by arm, capture, signed gaze and held point, using exact shared full-cohort scored points.

[Detailed signed point tails by capture, gaze and held point](point_signed_tails.csv) (208 rows).

| Response | Family | Cohort | Shared complete frames / points | ΔE² | ΔGθ² | ΔG_A² | Δworst² | Δx² | Δy² |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| baseline27 | gaze | Full shared | 143 / 429 | -0.641 | 0.0649 | 0.0403 | -3.594 | -0.008 | -0.633 |
| baseline27 | gaze | Joint interior | 129 / 399 | -0.845 | 0.0702 | 0.0349 | -3.974 | 0.005 | -0.599 |
| baseline27 | gaze | Joint state + P1 support | 63 / 189 | -1.793 | -0.0189 | -0.0411 | -6.074 | -0.462 | -1.331 |
| baseline27 | capture | Full shared | 143 / 429 | -3.258 | -0.0013 | -0.0109 | -10.079 | -0.710 | -2.548 |
| baseline27 | capture | Joint interior | 125 / 394 | -2.878 | -0.0001 | -0.0090 | -9.488 | -0.538 | -1.769 |
| baseline27 | capture | Joint state + P1 support | 73 / 231 | -1.733 | -0.0516 | -0.0738 | -7.849 | -0.666 | -2.150 |
| strong_anchor37 | gaze | Full shared | 143 / 429 | -0.019 | 0.1466 | 0.2721 | -4.593 | -0.564 | 0.545 |
| strong_anchor37 | gaze | Joint interior | 116 / 386 | 0.044 | 0.1552 | 0.3863 | -3.008 | -2.856 | 0.629 |
| strong_anchor37 | gaze | Joint state + P1 support | 63 / 189 | -2.318 | -0.0017 | -0.0205 | -6.094 | -0.500 | -1.818 |
| strong_anchor37 | capture | Full shared | 143 / 429 | -4.194 | -0.0432 | -0.0788 | -12.875 | -1.382 | -2.811 |
| strong_anchor37 | capture | Joint interior | 130 / 392 | -4.672 | -0.0466 | -0.0839 | -14.343 | -1.371 | -2.940 |
| strong_anchor37 | capture | Joint state + P1 support | 81 / 249 | -1.153 | -0.0519 | -0.0707 | -7.459 | -2.027 | 0.507 |

Negative squared changes favor differential-y. The four comparisons share the same 160-frame/480-slot schedule per response-family cell. Full scoring coverage is 429/480 points and 143/160 complete triples in every arm; the following table shows the model-specific bound slots and empirically supported complete-frame counts.

| Response | Family | Arm | Scored / scheduled points | Complete / scheduled frames | Bound slots | Interior complete | Empirical-state complete | Measured-P1 complete | Joint state+P1 complete |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| baseline27 | gaze | xy | 429/480 | 143/160 | 23 | 131 | 87 | 109 | 63 |
| baseline27 | gaze | differential-y | 429/480 | 143/160 | 25 | 130 | 86 | 109 | 63 |
| baseline27 | capture | xy | 429/480 | 143/160 | 8 | 137 | 111 | 111 | 85 |
| baseline27 | capture | differential-y | 429/480 | 143/160 | 30 | 130 | 99 | 111 | 73 |
| strong_anchor37 | gaze | xy | 429/480 | 143/160 | 29 | 124 | 85 | 109 | 63 |
| strong_anchor37 | gaze | differential-y | 429/480 | 143/160 | 40 | 116 | 84 | 109 | 63 |
| strong_anchor37 | capture | xy | 429/480 | 143/160 | 30 | 132 | 96 | 111 | 81 |
| strong_anchor37 | capture | differential-y | 429/480 | 143/160 | 36 | 130 | 99 | 111 | 86 |

The detailed tail table preserves signed x/y behavior by capture, gaze and held point. Complete-frame support counts above are per arm; the paired cohort table uses their exact intersection, so paired counts can be smaller than either arm's independent count.

## Training-only common-y diagnosis

The fixed ridge diagnostic predicts the common-all-three-y residual from frozen state features or state plus measured P1 context. Standardization and ridge coefficients are refit inside each group-blocked training split. Values are fold-equal mean skill relative to the training-block mean predictor; negative values indicate worse blocked prediction. These are development diagnostics, not untouched test estimates.

| Response | Held-out training block | State-only skill | State + P1 skill |
|---|---|---:|---:|
| baseline27 | Fixation | -0.433 | -1.962 |
| baseline27 | Capture | -1.282 | -3.500 |
| baseline27 | Signed gaze | -0.701 | -7.761 |
| strong_anchor37 | Fixation | -0.396 | -2.013 |
| strong_anchor37 | Capture | -1.401 | -3.321 |
| strong_anchor37 | Signed gaze | -0.254 | -7.473 |

Capture-specific signed common-y residual means remain after conditioning on state and P1. Each value below is the equal-fold mean over the eight outer folds where that capture remains in training.

| Response | Capture | Raw common-y mean (px) | After state + P1 (px) |
|---|---|---:|---:|
| baseline27 | 1 | 0.010 | 0.024 |
| baseline27 | 2 | -0.864 | -0.664 |
| baseline27 | 3 | 0.536 | 0.398 |
| baseline27 | 4 | 0.333 | 0.246 |
| strong_anchor37 | 1 | 0.008 | 0.031 |
| strong_anchor37 | 2 | -0.877 | -0.656 |
| strong_anchor37 | 3 | 0.595 | 0.359 |
| strong_anchor37 | 4 | 0.361 | 0.264 |

## Training-scaled shared-y covariance sensitivity

One xy trial used one training-only τ per fold, set to the equal-fixation RMS of the common-y training residual. It adds a shared y-mode term to the frozen localization covariance while leaving the response, pilot and calibration unchanged. This RMS includes systematic bias and variation; it is not a validated noise variance and may double-count the existing covariance.

| Family | Reference E RMS | Trial E RMS | Reference worst RMS | Trial worst RMS | Full ΔE² | Full ΔGθ² | Full ΔG_A² | Full Δworst² | τ range (px) | Runtime |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Gaze | 3.421 | 3.326 | 4.385 | 3.954 | -0.641 | 0.0648 | 0.0402 | -3.594 | 1.267–1.712 | 34.4s |
| Capture | 4.014 | 3.586 | 5.222 | 4.146 | -3.256 | -0.0014 | -0.0110 | -10.077 | 1.267–1.712 | 34.4s |

The saved covariance trial closely reproduces the differential-y aggregate prediction results, consistent with downweighting a shared-y residual direction. This is not independent confirmation and does not validate the covariance model. The differential-y response remains the challenger; xy remains the reference. No automatic promotion was made.

## Verification and provenance

- Primary audit: 12 workers, 3.78s, 4 direct cells, 18 training reports.
- Covariance sensitivity: 12 workers, 34.42s, 9 fold-specific τ values; 1827 retained objectives, 27 analytic certificate samples, and 10 exact paired cohorts checked.
- Input availability: 54 held-point slots across 18 frames were unavailable because retained P4 input was insufficient; these are not solver failures. This is already reflected in the scheduled coverage: gaze 143/160 complete frames and 429/480 scored points; capture 143/160 frames and 429/480 points.
- Independent verifier: [verification.json](verification.json); reproducible command: `python verify_results.py`.
- Final test evidence: [tests_final.txt](tests_final.txt) (90 passed); initial 87-pass record remains in [tests.txt](tests.txt).
- Execution metadata: [execution.json](execution.json). The initial sandbox forkserver bind failure is recorded in `run_forkserver_failure.log`; the successful run used an explicit fork context.
- All results use historical development folds. Independent nested recalibration needs externally declared guards and grouping; this report makes no final-transfer claim.
