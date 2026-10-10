# Fresh capture 1 P1 fit: implementation and scientific review

This replaces the discarded experiment that reused an earlier fitted optical
model. Inputs are capture 1 raw detections and its reviewed fixation intervals.
No previous model, fitted states, templates or covariance are loaded.

## Implemented sequence

1. Compute the vector `delta = mean(P4)-mean(P1)` for every complete frame.
2. Fit a horizontal quadratic from five fixation means to their nominal gaze
   labels, giving each fixation equal weight. Use the polynomial's frame-mean
   moments so the zero fixation's mean estimated horizontal gaze is exactly
   zero. No estimated framewise gaze is clipped.
3. Following the user's assumption, use the horizontal reference slope
   **0.0483899014 deg/px** for vertical centroid displacement as well. Subtract
   the zero fixation's mean vertical separation so its mean vertical gaze is
   also zero. No vertical quadratic or vertical target calibration is invented.
4. Center each P1 triangle on its measured centroid and average the triangles
   in the zero fixation. Freeze this measured, non-equilateral reference.
5. Keep the initial 2D gaze fixed. Fit four shared keystone shape coefficients
   and one positive total magnification `M_i` per frame. Profile `M_i`
   analytically in original centered camera coordinates.
6. Invert the fitted map with the predicted keystone centroid restored before
   nonlinear inversion. Compare corresponding recovered vertices to the
   fixed reference; also report signed residuals, tails and source-row blocks.

## One magnification, including keystone's uniform scale

The unnormalized shape map is

```text
a = k_ax * theta_x^2 - k_ay * theta_y^2
d_j = 1 + k_px * theta_x * B_jy + k_py * theta_y * B_jx
F_j = (exp(a) * B_jx, exp(-a) * B_jy) / d_j
C_j = F_j - mean(F)
H_j = C_j * RMS_radius(B) / RMS_radius(C)
predicted_P1_centered_j = M_i * H_j
```

Thus `H` has exactly the reference RMS radius, and **all overall scale is
assigned to the one fitted magnification `M_i`**. Model-size normalization is
a deterministic shape convention, not another fitted scale; observed data
are not area-normalized. The horizontal perspective denominator retains the
previously agreed dependence on `B_y`, with vertical gaze acting on `B_x`.
This is an explicitly declared empirical 2D model, not a derived physical
optical law.

The objective is squared point error in the original centered camera pixels,
averaged over the three corresponding vertices and then equally over the
five fixation intervals. This avoids using a candidate inverse's contraction
to select the model. Inverse-space point distances remain diagnostics.

## Numerical findings

All 24,700 scheduled frames are accounted for: 24,280 are complete and
420 unavailable. All complete frames admit the inverse. Six deterministic
starts reach the same objective to within **1.8e-15 px²/point**. The best
gradient infinity norm is **7.5e-8**, the local Hessian is positive definite,
and no fitted coefficient is at a declared bound. This is evidence of a
consistent local solution; it is not a proof of global optimality.

The fitted coefficients in native units are:

| Coefficient | Value | Units |
|---|---:|---|
| `k_ax` | 1.84271097e-5 | deg^-2 |
| `k_ay` | 1.53339400e-2 | deg^-2 |
| `k_px` | -1.85487349e-6 | (deg px)^-1 |
| `k_py` | 4.21131261e-5 | (deg px)^-1 |

The total magnification has mean **0.999638**, standard deviation **0.004156**,
and observed range **0.984125–1.004664**. The average in the zero fixation is
approximately one because the fixed reference is its mean pattern; the fit
does not add another scale or enforce each reference frame's magnification
to be one.

Median recovered point distance is **0.674 px**, p95 **2.021 px**. A separately
fitted scale-only comparison gives median **0.687 px**, p95 **2.324 px**.
The original-coordinate equal-fixation objective falls from **1.228691** to
**1.013176 px²/point**, approximately **17.54%**. This demonstrates additional
in-sample shape explanation beyond total magnification. It does not establish
independent predictive improvement.

The +5 degree interval remains the largest systematic mismatch: recovered
point median **1.704 px**, p95 **2.216 px**. This interval also has lower
magnification, mean **0.992276**, compared with means near one elsewhere.
The plots and interval/block residuals should guide the next bounded test.

Improvement is not uniform across fixations. At +10 degrees, the recovered
median increases from **0.846 px** for scale-only correction to **1.032 px**
for the fitted keystone correction, while p95 decreases from **1.904** to
**1.587 px**. A single aggregate objective therefore does not capture all
reference-matching tradeoffs.

## Interpretation and limits

The horizontal fixation-mean calibration errors are about +0.0074, -0.0039,
0, +0.0486 and -0.0192 degrees. These quantify agreement with the nominal
calibration labels. They are not measurements of framewise gaze accuracy.

Mean estimated vertical gaze ranges from approximately -0.221 to +0.353
degrees across the horizontally labeled fixations. Its entire conversion
uses the assumed common first-order slope. There are no measured vertical
target angles. The large difference between horizontal and vertical native
anisotropic coefficients warrants a reference/slope sensitivity study before
interpreting either as an optical parameter. The small vertical range and
sequential fixation order also limit separation of gaze effects from drift.

The reference, initial gaze and shape fit reuse these same frames. Conditional
algebra and stationarity checks do not quantify uncertainty in the reference
or polynomial calibration. No accommodation in diopters, physical optical
center, radial distortion or physiological accuracy is inferred.

The next test should keep this pipeline explicit, examine the +5 degree
pointwise and block residuals, and test stability to independent reference
blocks and the assumed vertical conversion. No next-stage fit is run here.
