# Current plan: fixed-keystone, shared-reference P4 barrel step

**Status: Completed and accepted for now as a conditional shared-reference zero-gaze stage.** This geometric result does not establish a physical barrel response.  
**Current starting point:** the separately fitted seven-parameter raw P4 keystone models in `experiments/archive/reverse_transform/superseded_baseline/per_capture_p4_inverse_keystone/results/`.  
**Next experiment:** `experiments/archive/reverse_transform/superseded_baseline/fixed_keystone_barrel_zero_gaze/`.

## Scope

Add a radial/barrel coefficient for each capture while holding the preceding keystone mapping fixed. Use only the zero-nominal-gaze fixation in each capture for this first step. Capture 1 defines the empirical common reference and is fixed at `kappa_1 = 0` by convention; fit one coefficient for each of captures 2–4. These coefficients are conditional changes relative to that reference. This stage does not estimate accommodation, refit gaze or keystone, add a per-frame scale, or normalize the output.

## Fixed keystone and common reference for this stage

The preceding inverse-keystone fits each used an independently fitted symmetric template `B_c`. For this stage, the user accepts the shared triangle `B_ref` as a declared common-reference convention. The resulting coefficients are conditional geometric comparisons under that convention; they are not proof of a physical barrel response.

For each capture `c`, preserve its saved seven-parameter fit
`(w_c,h_c,alpha_c,b0_c,cx_c,cy_c,origin_c)` and its fitted scale gauge
`R_c = sqrt(mean_j ||B_cj||^2)`. The raw keystone applied to an input point `z` is

```text
phi = (theta_x - origin_c)/10
a = b0_c * phi^2
K_c(z,theta_x) = (exp(a) z_x, exp(-a) z_y) /
                  (1 + phi * (cx_c z_x + cy_c z_y)/R_c)
```

All saved keystone coefficients and origins stay fixed. In particular, do not refit the original template `B_c` or the capture-specific keystone coefficients during the barrel fit.

The shared reference `B_ref` is the arithmetic mean of Capture 1's saved **un-centered, frame-first inverse-keystone vertices** in the nominal zero-gaze fixation, using the existing filtered fit and its saved trial-consistent predicted `mu`. The saved mean is `[[159.148913, -95.416337], [-5.332421, 200.673709], [-153.737871, -105.171940]] px` in vertex order, with centroid `(0.02620675, 0.02847731) px`. Preserve this small nonzero centroid. Do not recenter or rotate `B_ref`; it carries the declared origin convention of the existing Capture 1 inverse. The current filtered zero-target support is C1–C4: 4900, 4552, 4811, and 4747 rows, respectively; recheck the saved masks and report actual support at run time.

## Barrel model and fit

Use the empirical incremental radial map

```text
Q_kappa(B) = (1 + kappa * ||B||^2) B
```

with native coefficient units `px^-2`. Apply it before the frozen raw keystone. At `kappa=0`, Capture 1 is exactly the common empirical reference. For every capture, the forward centered vertices are

```text
F_cj = K_c(Q_kappa(B_ref,j), theta_x)
mu_c(theta_x) = mean_j F_cj
Xhat_cij = F_cj - mu_c(theta_x_i)
```

`mu` must be recomputed from the trial `kappa` and trial gaze at every objective evaluation. Fit one `kappa_c` independently for C2–C4 to the existing centered measured P4 vertices, with equal fixation weight; set C1's `kappa` exactly to zero as the radial identity relative to the reference. This convention does not imply a zero C1 observation loss. Use only complete rows passing the existing per-fixation P4 geometry filter; the already-frozen horizontal gaze input is used as-is. The primary fit fixes the shared nuisance magnification `g=1`; it does not estimate `g` from P1. Do not assume P1 has a direct accommodation response. The fitted `kappa_c` is conditional geometry and cannot yet be read as physical accommodation response.

Also fit one secondary uniform intrinsic size per capture by using `s*B_ref` before the same frozen K and recomputing trial `mu`. This diagnostic compares isotropic size change with radial deformation and is not the primary model or an accommodation estimate. Report whether the predictions are distinguishable over the three near-equal reference radii. As a separate sensitivity control, evaluate the unfiltered complete zero-target rows with the selected filtered-fit parameters held fixed; do not refit the raw control.

Optimize a dimensionless parameter `beta = kappa*M`, where `M=max_j ||B_ref,j||²`, and report `kappa=beta/M` in `px^-2`; this avoids using an optimizer tolerance that is too coarse for coefficients near 1e-6. Keep the radial map on its monotone, nonfolding branch for every reference radius by requiring both `1+kappa*r^2 > delta` and `1+3*kappa*r^2 > delta` on `B_ref`. The negative bound follows from the tightest derivative condition (`beta > -(1-delta)/3`); use a declared finite positive upper bound and then check the frozen keystone denominator against the chosen positive margin over all fitted points and observed horizontal gaze inputs. Save bounds, margins, minimum denominators, convergence status, and active-bound status.

## Inverse and reporting

For each measured centered triangle, add the trial-consistent `mu_c(theta_x)` from the fitted model, invert the frozen keystone, then invert the scalar radial map around the fixed reference origin. The scalar relation is `r_d = r + kappa*r^3`; for negative `kappa`, bracket only on the increasing branch `0 <= r <= sqrt(-1/(3*kappa))` and reject radii beyond the reachable maximum. At `kappa=0`, return the input exactly. Use a safeguarded monotone scalar solve; record failures rather than clipping or silently recentering. The primary recovered points remain uncentered for vertex-error reporting. Center only display copies of triangles where a shape overlay requires it.

Report the `kappa=0` common-reference baseline and the fitted one-parameter result per capture. Include forward residuals, inverse vertex errors, mean-triangle errors separately from pooled frame errors, per-capture side/angle geometry, and before/after overlays on common native-pixel axes. Report `kappa_c` in `px^-2` and the predicted radial fractional effect `kappa_c*r^2` at each `B_ref` vertex radius. Keep all four capture results separate; capture identity and nominal accommodation demand are confounded.

## Reproducibility and boundaries

Create a standalone runner and results under `experiments/archive/reverse_transform/superseded_baseline/fixed_keystone_barrel_zero_gaze/`. Save source provenance, frozen input parameter hashes, selected masks/row counts, the shared reference vertices including centroid, per-frame gaze and masks, fit summaries, raw numeric arrays, plots, and a concise `RESULTS.md`. Do not overwrite the existing Capture 1 or per-capture keystone outputs. Do not change the archived estimators, fit accommodation, estimate a new P1-derived shared nuisance magnification, jointly fit captures, or add per-frame radial parameters in this first stage.

## Later steps, not run

After this zero-fixation stage, use a soft anchor on each capture's zero-fixation mean radial coefficient, extend those coefficients to all fixations using the current estimated gaze, and consider estimating the shared nuisance magnification from P1. Then perform a conditional joint K/radial refinement with the current keystone parameters as soft anchors and the full vertex loss. Keep the reference gauge fixed. Evaluate radial and tangential residual patterns and test identifiability; do not assume that adding more fixations or a joint refinement makes the radial response identifiable. These later stages are only a sequence proposal and are not part of the current experiment.
