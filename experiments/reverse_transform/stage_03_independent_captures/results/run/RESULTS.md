# Independent P1/P4 fits for captures 1–4

This run fits the four captures independently while using capture 1’s nominal zero-gaze means at 0.360360 D as the shared P1 and P4 empirical references. Capture 1’s audited P1 gaze and framewise magnification are reused exactly; captures 2–4 receive new P1 keystone fits to the same reference. For each capture, P4 uses that capture’s P1 magnification frame by frame and fits a separate keystone plus one constant relative radial coefficient. No P4 magnification or framewise accommodation state is fitted.

Capture 1’s relative radial increment κ is fixed at zero by reference convention. The radial center is the empirical reference centroid, which may itself contain distortion. Demand labels are attached only after the image fits to describe association; each label occurs in one capture, so demand and capture effects are confounded.

## Coverage and P1 fits

| Capture | Demand label (D) | Scheduled / complete / unavailable | P1 magnification mean ± SD | P1 inverse point error median / P95 (px) |
|---:|---:|---:|---:|---:|
| 1 | 0.36036 | 24,700 / 24,280 / 420 | 0.999638 ± 0.004156 | 0.674 / 2.021 |
| 2 | 4 | 24,890 / 22,026 / 2,864 | 1.003195 ± 0.003687 | 0.867 / 2.528 |
| 3 | 3 | 25,100 / 19,795 / 5,305 | 1.003052 ± 0.004177 | 0.781 / 2.296 |
| 4 | 2 | 25,400 / 23,074 / 2,326 | 1.000868 ± 0.002343 | 0.494 / 1.406 |

P1 inverse errors are same-recording fit residuals, not independent accuracy estimates. Gaze is calibrated from triangle-centroid displacement; vertical gaze uses an assumed shared first-order x/y slope and has no independent vertical target measurement.

## P4 fits and shape diagnostics

All 89,175 complete frames contribute to forward fitting; no forward rows are trimmed. Inverse comparisons use the common 89,132-frame valid set for all three reconstructions. Capture 2 has 43 failed radial inverses (31 at −10°, 12 at −5°); these rows remain in the forward fit and are excluded from inverse metrics. The tables and plots report common-valid counts.

Point-error quantiles pool the three vertex distances over frames. Shape-only error first centers each inverse triangle, then scales its RMS radius to the reference RMS radius; no rotation alignment is applied. This is a diagnostic and is not used to fit or apply another magnification.

| Capture | P4 objective: P1 M only → fitted (px²/point) | Relative κ (px⁻²) | Shape-only median / P95 (px): P1 M only → keystone → keystone + barrel |
|---:|---:|---:|---:|
| 1 | 6.859 → 2.935 | 0 (reference) | 1.839 / 3.937 → 0.648 / 1.652 → 0.648 / 1.652 |
| 2 | 197.259 → 24.217 | −1.856868e−6 | 2.181 / 3.328 → 1.768 / 2.760 → 1.791 / 2.893 |
| 3 | 114.877 → 5.464 | −1.472843e−6 | 1.982 / 3.196 → 1.509 / 3.147 → 1.490 / 3.198 |
| 4 | 48.928 → 9.533 | −8.477171e−7 | 1.785 / 3.064 → 1.927 / 4.067 → 1.896 / 4.317 |

The keystone shape diagnostic improves captures 1–3 and worsens capture 4. The full radial correction has a larger size effect and reduces point error, but capture 4’s shape-only median/P95 remain worse than its P1-scale-only values. The pooled objective decrease therefore does not imply uniform keystone shape improvement across captures. These results do not identify physical barrel distortion: uniform size changes, including accommodation-related changes, can contribute to the fitted relative radial offsets.

## Native keystone coefficients

| Capture | `k_ax` (deg⁻²) | `k_ay` (deg⁻²) | `k_px` (deg⁻¹ px⁻¹) | `k_py` (deg⁻¹ px⁻¹) |
|---:|---:|---:|---:|---:|
| 1 | −1.494723e−5 | 5.353242e−3 | −3.342840e−6 | −5.058396e−4 |
| 2 | 6.444609e−5 | 1.847704e−4 | 5.370451e−7 | −4.160383e−5 |
| 3 | 3.164038e−5 | −4.394979e−3 | −1.615965e−6 | −6.238853e−4 |
| 4 | 1.243203e−5 | −1.254627e−2 | −1.370967e−6 | 2.569031e−4 |

## Numerical checks and limits

The saved-result audit passed for all four captures. It independently checks source and parent hashes, population counts, exact capture 1 P1 gaze/magnification reuse, recalibrated gaze, finite-difference gradients, projected gradients, bounds, free Hessian curvature, synthetic inverse closure, and reported inverse metrics. Maximum finite-difference gradient discrepancy was 6.22e−9; maximum synthetic forward/inverse closure was 2.84e−13 px. Independent projected-gradient and minimum free-Hessian eigenvalue checks are:

| Capture | P1 projected gradient | P1 min free-Hessian eigenvalue | P1 active bounds | P4 projected gradient | P4 min free-Hessian eigenvalue | P4 active bounds |
|---:|---:|---:|---|---:|---:|---|
| 1 | 7.48e−8 | 673.91 | none | 1.18e−13 | 391.41 | none |
| 2 | 2.79e−9 | 76.125 | none | 1.11e−11 | 45.989 | none |
| 3 | 5.07e−14 | 804.63 | none | 1.67e−11 | 419.41 | none |
| 4 | 4.89e−13 | 32.244 | none | 2.10e−12 | 13.935 | `k_ay` |

The P1 fits are stationary with all coefficients interior. The P4 fits are stationary with positive free-subspace curvature and valid optical domains. Capture 4’s P4 `k_ay` coefficient (coefficient index 1) is at its bound; report it as a constrained stationary solution with a numerical parameter-domain limit, not as an unrestricted optimum. The other listed P4 coefficients are interior.

Six optimizer starts per fit reached nearly identical objective values: maximum start-to-start objective range was 1.78e−15 for P1 and 3.55e−14 for P4.

CuPy 14.2.0 ran float64 cost and analytic gradients on a Tesla P100 GPU. GPU/CPU cost differences were at most 3.55e−15 and gradient differences at most 1.28e−12. The run took 7.92 s. Fit status is `COMPLETE`; this is numerical model consistency, not physiological validation. No accommodation truth or absolute physical barrel coefficient is established.

The descriptive, post-fit demand association is an anchored slope of −5.255210e−7 px⁻² per demand-label D. It was computed from the four independent capture coefficients after fitting; it was not used as a fitting law. Four demand labels from four captures cannot separate demand effects from other capture differences.

## Figures

- [Inverse point error by fixation](point_error_by_fixation.png)
- [Radius-normalized shape error by fixation](shape_error_by_fixation.png)
- [Reference and fixation-mean triangles](triangle_overlays.png)
- [Radius-normalized vertex-error clouds](normalized_vertex_clouds.png)
- [Recovered radius distributions](radius_distributions.png)
- [Independent coefficients and post-fit demand association](postfit_barrel_vs_demand.png)

The machine-readable [summary](summary.json), [post-fit association](postfit.json), and [audit](audit.json) include detailed per-fixation results and provenance. Run logs are in the stage directory: [runner](../../console_run.log), [audit](../../audit_run.log), and [plotting](../../plot_run.log).
