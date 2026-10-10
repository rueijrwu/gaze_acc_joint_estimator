# Capture 1 P4 reflection: run results

## Scope

The run uses only the five reviewed intervals from `capture_1_detections.pkl`. It includes 24,700 scheduled rows, 24,280 complete P1/P4 rows and 420 unavailable rows. The fixed reference is capture1's mean centered P4 triangle in the nominal zero-gaze interval; its label is 0.360360 D. The capture1 framewise P1 magnifications and centroid-derived 2D gaze are reused exactly from the audited P1 fit. No P4-specific magnification, accommodation state, fitted radial coefficient, or other capture is used. The relative radial coefficient is fixed at zero for the reference capture. Demand labels are not fitting inputs.

## Fitted keystone and objective

The four fitted native keystone coefficients are:

| Coefficient | Estimate | Units |
|---|---:|---|
| `k_ax` | -1.49472263e-5 | deg^-2 |
| `k_ay` | 5.35324158e-3 | deg^-2 |
| `k_px` | -3.34284030e-6 | (deg px)^-1 |
| `k_py` | -5.05839639e-4 | (deg px)^-1 |

For the centered reference triangle `B`, use `t=theta/U`, `a=k_ax*theta_x^2-k_ay*theta_y^2`, and

```
F_i = (exp(a) B_ix, exp(-a) B_iy)
      / (1 + k_px theta_x B_iy + k_py theta_y B_ix)
K_i = C(F)_i * R_B / sqrt(mean_j ||C(F)_j||^2)
P4_predicted_centered,i = M_P1(frame) * K_i
```

`C` centers the three vertices. `R_B` is the fixed empirical reference RMS radius, and the normalization applies to the model triangle only. The same P1 magnification `M_P1` is held fixed in the P4 forward model. No extra P4 scale or accommodation parameter is fitted; the relative radial coefficient is fixed to zero in this reference capture.

The equal-fixation mean original-coordinate point SSE is **6.85869191 px²/point** for the fixed-M baseline and **2.93474448 px²/point** with keystone, a **57.21%** reduction. Six deterministic starts agreed within 8.9e-16 px²/point. The projected gradient infinity norm was 8.77e-14; the free Hessian was positive definite; no shape coefficient was at a bound. The inverse passed for all 24,280 complete frames.

## Post-inverse uniform-radius shape diagnostic

The fixed reference RMS radius is `R_ref = 191.254 px`. For each inverse triangle `Q_i`, compute `R_i = sqrt(mean_j ||C(Q_i)_j||^2)`, where `C` centers its vertices, then apply `s_i = R_ref/R_i` and compare `s_i C(Q_i)` with the reference. This removes uniform size after inversion; `s_i` is diagnostic only, not fitted or applied as an extra P4 magnification, and does not affect the forward fit. Pooled median/P95 shape distance improved from **1.8385/3.9370 px** before keystone to **0.6481/1.6516 px** after keystone.

| Nominal horizontal fixation | Complete frames | Before median / P95 (px) | After median / P95 (px) |
|---:|---:|---:|---:|
| -10° | 4,433 | 2.1450 / 2.8851 | 0.6568 / 2.0303 |
| -5° | 4,800 | 1.2863 / 1.7303 | 0.7421 / 1.1788 |
| 0° | 4,900 | 0.2342 / 0.6236 | 0.5307 / 1.6523 |
| +5° | 4,993 | 1.9344 / 2.2690 | 0.7163 / 1.7613 |
| +10° | 5,154 | 3.8427 / 4.2033 | 0.6038 / 1.4837 |

The zero-gaze interval is worse after the keystone inversion despite the strong aggregate improvement. The aggregate score does not imply improvement at every fixation. These are comparisons to a fixed empirical reference computed from the same recording, not independent accuracy measurements.

## Audit and limits

The independent audit passed. It checked finite-difference gradient agreement (maximum 2.44e-9), synthetic forward/inverse closure (2.56e-13 px), and stationarity. Its independent CPU certificates give projected-gradient infinity norms 7.48e-8 (P1) and 1.18e-13 (P4), minimum free-Hessian eigenvalues 673.91 and 391.41, and no active coefficient bounds; there are no numerical limits. The objective remained within the declared numerical limits. The fit ran in 1.83 s with CuPy 14.2.0 on a Tesla P100 using float64 device-resident cost and analytic gradient with a SciPy L-BFGS-B controller.

Gaze and the fixed reference are same-recording quantities. Vertical gaze uses the shared first-order x/y centroid slope assumption; there are no measured vertical targets or framewise gaze truth. Capture1's one nominal demand label is not sufficient to identify a demand response. The fitted keystone is empirical and capture-relative. No physical barrel, accommodation, absolute optical origin, or physiological accuracy claim follows.
