# Scientific review: capture1 P4 fixed-scale keystone

## Review disposition

The corrected capture1 fit is a conditional, in-sample P4 geometry analysis. It reuses the saved capture1 P1 centroid-derived gaze and one framewise P1 magnification, and compares a fixed-M baseline with a four-coefficient 2D keystone. The independent audit passes. The result supports an empirical shape correction under those fixed inputs; it does not establish physical optical parameters or independent recovery accuracy.

## Model and evidence

The reference is the mean centered P4 triangle in the nominal zero-gaze interval. For the reference capture the radial coefficient is fixed at zero, defining a relative reference gauge. The model applies anisotropic gaze-dependent stretch and a gaze-dependent denominator, centers the transformed triangle, and normalizes its model RMS radius to the fixed reference radius. The prediction is then multiplied by the P1 magnification for that same frame. Measured P4 triangles are not normalized in the forward objective, and no extra P4 scale or accommodation state is fitted.

The equal-fixation forward objective falls from 6.85869191 to 2.93474448 px²/point, a 57.21% reduction. The shared keystone fit is stationary and numerically well determined within the declared four-parameter empirical model. This does not guarantee global optimality or validate the model's physical interpretation.

The fixed reference RMS radius is 191.254 px. The separate post-inverse diagnostic centers each inverse triangle `Q_i`, computes `R_i = sqrt(mean_j ||C(Q_i)_j||^2)`, then uses only for scoring `s_i = 191.254 px/R_i`. This removes uniform size after inversion; `s_i` is not a fitted or applied P4 magnification. Its pooled shape median/P95 is 0.6481/1.6516 px after keystone versus 1.8385/3.9370 px before. The zero-gaze interval worsens from 0.2342/0.6236 to 0.5307/1.6523 px. The model's aggregate improvement therefore is not uniform across the five fixations.

## Interpretation limits

The fixed P4 reference and gaze map use the same capture. The vertical gaze conversion assumes the same first-order slope as horizontal gaze and has no independent vertical-target calibration. P1 magnification is frozen, so any P4 size changes not represented by the empirical shape may remain in residuals. The reference demand label is a protocol label, not a per-frame accommodation measurement; with one capture and one nominal demand condition, no demand response can be inferred.

The fitted keystone is not an absolute barrel calibration. The reference capture's relative radial coefficient is fixed at zero, so the fit does not test a radial response. It provides no accommodation estimate, physical optical center, framewise gaze truth, or physiological accuracy estimate. Further capture-level analyses must use independently fitted capture geometry and keep demand association descriptive when demand is confounded with capture.
