# Attempt 01 — capture1 P1 shape fit

## Scope and audit

This is an in-sample fit on the five reviewed intervals from `capture_1_detections.pkl` only. No previous model, state, template, or covariance was loaded (`previous_model_used=false`). Of 24,700 scheduled rows, 24,280 have complete P1/P4 points and 420 are unavailable; all 24,280 passed the inverse domain check. The independent audit passed and confirmed capture1-only provenance, zero previous-model use, the polynomial coefficient conversion, shape derivatives, scale bookkeeping, profile normal equations, and rank-four local curvature. The audit explicitly limits its scope to an in-sample P1 fit with fixed polynomial gaze; it certifies neither coefficient precision nor gaze truth. The audit reports five files checked, profile Gauss-Newton rank 4 and condition number 86.39. The fit took 6.84 s with Python 3.13.15, NumPy 2.5.3, SciPy 1.18.1 and CPU SciPy L-BFGS-B (no GPU); plots used Matplotlib 3.11.2.

## Model and coefficients

Let `B_i=(B_ix,B_iy)` be the centered empirical zero-fixation reference triangle and `R_B = sqrt(mean_i ||B_i||^2)`. Angles are in degrees. The implemented two-dimensional keystone is

```
s_x = exp(k_ax * theta_x^2 - k_ay * theta_y^2)
s_y = 1 / s_x
D_i = 1 + k_px * theta_x * B_iy + k_py * theta_y * B_ix
W_i = (s_x * B_ix / D_i, s_y * B_iy / D_i)
K_i(theta; B) = C(W)_i * R_B / sqrt(mean_j ||C(W)_j||^2)
P1_centered,i = M * K_i(theta; B)
```

`C` subtracts the triangle centroid. Thus `K` is centered and normalized back to the empirical reference RMS radius. The only fitted framewise scale is the positive scalar `M`; it absorbs every overall size change, including any size change induced by the keystone. The four shape coefficients are shared across frames.

| Native coefficient | Estimate | Units |
|---|---:|---|
| `k_ax` | 1.8427109700e-5 | deg^-2 |
| `k_ay` | 1.5333939976e-2 | deg^-2 |
| `k_px` | -1.8548734898e-6 | (deg px)^-1 |
| `k_py` | 4.2113126090e-5 | (deg px)^-1 |

The native `k_ay` is 832.14 times `k_ax`; `|k_py|` is 22.71 times `|k_px|`. The estimated framewise vertical gaze spans -0.582418 to +0.581294°, and the five interval means are -0.221304, -0.073497, 0, +0.157107, +0.352648°. At the largest |theta_y|, `k_ay * theta_y^2` contributes 0.005201 to log stretch (about 0.52%); the largest absolute vertical denominator contribution `k_py * theta_y * B_ix` is 0.005642. Corresponding largest horizontal contributions are 0.002246 to log stretch and 0.005032 to the denominator. The much smaller assumed vertical span means the coefficient ratio alone is not an effect-size comparison. Sensitivity to the shared vertical slope and reference is needed; these estimates do not establish a vertical optical mechanism.

The horizontal gaze polynomial, in native `delta_x = mean(P4)-mean(P1)` pixels, is

```
theta_x = -2.5187431672 + 0.04828352683 * delta_x
          + 1.020712493e-6 * delta_x^2  degrees
```

Vertical gaze is assumed to be `theta_y = 0.04838990142 * (delta_y - 44.90292539)`, where `delta_y` is the same mean P4-minus-P1 centroid difference in pixels. Its slope is the shared first-order reference slope. The five nominal x target labels calibrate the x map; there are no measured vertical target labels. The zero-fixation mean estimated gaze is `(0, 0)` by construction. This calibration uses the same capture1 observations as the shape fit.

The empirical reference triangle (left, middle, right; centered x/y pixels) is

```
[[-230.03015087,  124.16337008],
 [  17.39823254, -245.70354104],
 [ 212.63191833,  121.54017096]]
```

Its RMS radius is 250.98983194 px.

## Interval results

`M` is the profiled per-frame magnification. Inverse error is the distance between the recovered reference-coordinate point and the same empirical reference point; it is a conditional, in-sample consistency diagnostic, not error against independent truth. Values below are point-distance median and P95 in pixels, not RMS.

| Nominal x fixation | Complete frames | M mean | M SD | M P05 | M P95 | Inverse error median | Inverse error P95 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| -10° | 4,433 | 1.003706 | 0.001218 | 1.001098 | 1.004469 | 0.6732 | 1.1359 |
| -5° | 4,800 | 1.002526 | 0.000243 | 1.002134 | 1.002869 | 0.4072 | 0.5531 |
| 0° | 4,900 | 1.000000 | 0.000541 | 0.998917 | 1.000706 | 0.3670 | 1.2442 |
| +5° | 4,993 | 0.992276 | 0.001880 | 0.990132 | 0.996869 | 1.7039 | 2.2162 |
| +10° | 5,154 | 1.000238 | 0.001154 | 0.998737 | 1.001644 | 1.0321 | 1.5869 |

Across all complete frames, M has mean 0.999638, SD 0.004156, and range 0.984125–1.004664. Pooled raw, scale-only, inverse, and forward point-distance medians/P95 are respectively 0.9565/3.8171, 0.6868/2.3239, 0.6742/2.0214, and 0.6745/2.0088 px.

The equal-fixation mean point SSE is 1.013175578956354 px² for the full shape fit, versus 1.228690801764458 px² for scale-only. The full model reduces this objective by **17.54%**. This is an in-sample objective comparison under the declared fixed gaze mapping; it is not an accuracy or independent validation result. The improvement is not uniform across all fixation summaries: at +10°, scale-only inverse error has median/P95 0.8456/1.9044 px, while the full inverse has 1.0321/1.5869 px. Thus the full fit worsens the median there while improving its P95; the pooled objective does not imply pointwise improvement at every fixation.

## Numerical checks and interpretation limits

All six deterministic starts converged to the same objective within 1.78e-15 px². The best analytic gradient infinity norm is 7.48e-8; the finite-difference gradient discrepancy is 2.11e-9. The numerical Hessian eigenvalues are positive (673.91, 4,567.89, 15,637.61, 58,219.50), no coefficient is at a bound, and the independent profile Gauss-Newton audit has rank four with condition number 86.39. Synthetic inverse closure and normalization checks pass at floating-point precision. These are local numerical checks, not uncertainty bounds.

The centroid-to-gaze map is calibrated from this same data; vertical gaze follows an assumption rather than measured vertical targets. Point recovery is evaluated against the fixed empirical reference computed from the same recording, so the distributions are not independent recovery accuracy. The large native vertical coefficients need sensitivity analysis. There is no accommodation estimate, framewise gaze truth, radial term, or absolute optical origin claim. No other capture was loaded or evaluated.

Figures generated from the saved arrays by `scripts/plot_results.py` are [fit distributions](fit_distributions.png), [mean triangles](mean_triangles.png), and [magnification and gaze by source row](magnification_gaze_by_row.png). See the stage [README](../../README.md) and [scientific review](../../SCIENTIFIC_REVIEW.md) for protocol and broader interpretation.
