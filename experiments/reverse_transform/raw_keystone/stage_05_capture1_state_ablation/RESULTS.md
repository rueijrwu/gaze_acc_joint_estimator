# Stage 05 results

**Status: COMPLETE; saved CPU audit PASS.** The analysis uses all 24,280 complete Capture 1 frames. For each model, the P1/P4 forward and inverse summaries below pool 72,840 vertex distances (three vertices per frame). RMS is the root mean square of those vertexwise Euclidean distances in native centered pixels; median and P95 are over the same pooled distances. The inverse metrics use the common-valid population, which here equals the full population for all models.

## Pooled coordinate errors

Each cell is median / P95 / RMS in px.

| Model | P1 forward | P4 forward | P1 inverse | P4 inverse |
|---|---:|---:|---:|---:|
| A0 | 0.675 / 2.009 / 1.017 | 0.950 / 1.908 / 1.162 | 0.674 / 2.021 / 1.022 | 0.942 / 1.921 / 1.163 |
| G | 0.549 / 1.377 / 0.804 | 1.429 / 2.707 / 1.646 | 0.547 / 1.383 / 0.807 | 1.438 / 2.700 / 1.644 |
| GA | 0.550 / 1.433 / 0.834 | 0.928 / 1.937 / 1.170 | 0.548 / 1.440 / 0.837 | 0.934 / 1.958 / 1.173 |

All three fitted outputs had 24,280/24,280 valid P1 and P4 inversions; inverse failures: 0. A0 preserves Stage 04 states. G holds A at the 0.360360 D reference while fitting horizontal gaze; GA fits both horizontal gaze and A. G lowers P1 RMS but not P4 RMS; GA improves P4 relative to G, while its P4 RMS is slightly higher than A0 (1.170 vs 1.162 px). These are same-recording fit metrics, not held-out accuracy.

## Fixation means and within-fixation spread

Values are mean ± within-fixation SD, not uncertainty of the mean. Nominal gaze labels are −10°, −5°, 0°, +5°, +10°. The G accommodation state is fixed at 0.360360 D.

| Nominal gaze | A0 gaze (°) | A0 A (D) | G gaze (°) | GA gaze (°) | GA A (D) |
|---:|---:|---:|---:|---:|---:|
| −10° | −9.993 ± 0.512 | 0.5663 ± 0.0728 | −9.999 ± 5.015 | −9.999 ± 5.285 | 0.5670 ± 0.0723 |
| −5° | −5.004 ± 0.252 | 0.5871 ± 0.0994 | −5.004 ± 0.921 | −5.002 ± 1.165 | 0.5872 ± 0.1004 |
| 0° | 0.000 ± 0.175 | 0.3621 ± 0.0568 | 0.000 ± 2.225 | 0.000 ± 2.127 | 0.3622 ± 0.0544 |
| +5° | +5.049 ± 0.570 | 0.2115 ± 0.1599 | +5.018 ± 12.808 | +5.027 ± 11.522 | 0.2143 ± 0.1554 |
| +10° | +9.981 ± 0.272 | 0.2431 ± 0.0793 | +9.972 ± 1.507 | +9.971 ± 1.612 | 0.2431 ± 0.0762 |

| Model | Gaze lower / upper bound frames | A lower / upper bound frames |
|---|---:|---:|
| A0 | 0 / 0 | 902 / 0 |
| G | 1,301 / 2 | fixed at reference |
| GA | 1,155 / 2 | 878 / 0 |

GA A means differ from A0 by at most 0.002857 D, while GA gaze SD is large at several fixations. This indicates that freeing gaze did not remove the observed A pattern; it does not establish either set of frame states as physiological truth.

## Starts, objective, and conditioning

The three G starts (initial gaze offsets 0°, −1°, +1°) produced objectives 3.402393, **3.392117** (selected), and 3.405194 px² per equal-fixation mean. The three GA starts produced 2.507293, **2.490495** (selected), and 2.503660. Optimizer iteration limits were reached, after which block-Newton polishing yielded small projected gradients. Selected G/GA projected-gradient infinity norms were 5.49e−8 / 7.12e−9; minimum free-frame curvatures were 9.27e−5 / 8.78e−5. The saved independent audit reproduces their local stationarity, positive free-frame curvature, and derivative checks. These are local certificates, not global-optimum certificates.

For GA, the local two-column gaze/A Jacobian condition median/P95 was 52.53 / 228.75 for P4 and 33.62 / 85.22 for joint P1+P4. Median acute column angles were 84.48° and 86.32°. Raw condition numbers use degree and diopter coordinates; the angles are unit-invariant. Local columns are not near-collinear, but bounds, within-fixation spread, and distinct stationary solutions limit interpretation.

## Audit and files

The audit independently replayed raw forward predictions (maximum discrepancy below 2.3e−13 px), finite-difference Jacobians/gradients/Hessians, projected stationarity, free-frame curvature, frozen inputs, and synthetic inverse closure (maximum 1.42e−13 px). All 24,280 frames remain present and valid in the saved inverse results. This verifies numerical implementation under the model; no framewise ground truth or held-out recording was used.

See the [saved summary](results/run/summary.json), [audit](results/run/audit.json), [protocol](results/run/protocol.json), [provenance](results/run/provenance.json), and [reproduction instructions](README.md). The six plots are linked in the README. Further interpretation and model limitations are in the [raw-chain scientific review](../SCIENTIFIC_REVIEW.md).
