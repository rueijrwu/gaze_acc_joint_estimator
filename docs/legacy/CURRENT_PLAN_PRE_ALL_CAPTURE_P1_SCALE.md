# P1 joint keystone/magnification calibration, then P4 gaze/accommodation decoupling

**Status (2026-10-11):** Stage 00 is complete for P1 scale diagnostics/sensitivities and the matched P4 scale-effect comparison. The matched P4 refit found the frozen bounded P1 `m` raises equal-capture/equal-fixation objective from 1.20434 to 1.71720 px² (+42.58%; weighted RMS 1.09742 to 1.31042 px); capture RMS worsens in C1–C3 and is nearly unchanged in C4. Both selected fits are stationary and all 88,978 rows pass the model domain checks. This shows that P1 `m` does not improve P4 geometric fit under this model; it does not establish physical scale. Keep `g=1` as the working primary input for the next `k(A)`/A calibration. The P1 `m` branch remains a conditional sensitivity only until its scale convention and gaze/offset dependence are disentangled and validated. The free P1 `θ01` and P4 `θ04` offsets are distinct nuisance parameters in saved-gaze coordinates and have no absolute physical-angle meaning. The fixed-zero P1 fit remains a sensitivity, not the primary protocol.

## Evidence and scope

The P1-only [size and retained affine shape diagnostic](experiments/archive/reverse_transform/superseded_baseline/p1_scale_analysis/results/RESULTS.md) covers 88,978 rows from the existing filtered P4-eligible cohort. Its reference is the mean centered P1 triangle from 4,900 C1 nominal-zero rows. Across C1–C4, descriptive area-scale gaze response ranges are 1.1949%, 1.0121%, 1.3074%, and 0.6011%; the maximum fixed-rotation similarity difference among fixation means is 0.0534%. These determinant-derived sizes include isotropic distortion and do not identify physical magnification. Synthetic and integrity checks passed. The report's three-point affine identity is not validation; block intervals are conditional on a fixed reference and fragmented segments limit certainty. Gaze/masks are inherited from P4 eligibility and the saved full-roster gaze calibration, so results are conditional rather than independent validation.

The completed [Stage 00 P4 scale-effect comparison](experiments/reverse_transform/p4_gaze_accommodation_decoupling/stage_00_baseline_sensitivity/p4_p1_scale_effect/results/RESULTS.md) refits shared P4 keystone/template, free `θ04`, and capture-specific radial coefficients for both `g=1` and frozen bounded learned-P1 `m`. The P1-m model worsens the equal-weight objective by 42.58%, largely from C1–C3; C4 is effectively unchanged. The frozen `m` mean is 0.98786, 0.78% of rows hit its 0.98 floor, and in C1 the +5° fixation mean is 0.981875 with about 11% at the floor versus 0.992378 at +10°. Gaze/fixation order is confounded, so this pattern cannot be assigned a cause. The fitted `θ04` moves from −44.89° (`g=1`) to −12.50° (P1-m), a clear nuisance/scale/template tradeoff. The in-sample inverse diagnostic gives pooled RMS 1.2112 px (`g=1`), 3.0124 px for P1-m substituted into the fixed `g=1` inverse, and 1.4859 px after full P4 refit under P1-m, with zero invalid frames. The scale-only substitution is a fixed-geometry mismatch diagnostic, not the refit comparison; see the report for capture-level results. This conditional comparison finds no improvement from P1-m in this P4 geometry model and neither identifies physical scale nor infers accommodation. The selected fits are stationary (7/8 and 8/8 near-best starts respectively), and both evaluate validly across the full cohort.

Earlier P4 evidence remains in the [independent per-capture keystone fits](experiments/archive/reverse_transform/superseded_baseline/per_capture_p4_inverse_keystone/results/RESULTS.md), [conditional zero-gaze common-reference stage](experiments/archive/reverse_transform/superseded_baseline/fixed_keystone_barrel_zero_gaze/results/RESULTS.md), and [joint learned-reference comparison](experiments/archive/reverse_transform/superseded_baseline/joint_keystone_barrel_learned_reference/results/RESULTS.md). The latest joint P4 fit used 88,978 frames: all-data measured P4 MSE was 3.764975 px² for historical initialization, 1.351373 fixed-reference radial, 1.245008 learned-reference radial, and 1.256081 uniform-size control. The learned radial measured RMS was 1.1158 px. Gains were concentrated in C1/C2; C4 worsened from 1.021754 to 1.112314 px². At both leave-one-target endpoints the learned radial fit was worse than fixed radial (3.84615/4.58719 versus 2.58783/2.62639 px²), using historical-anchor conditional fits with one start. These results do not establish physical separation. Preserve all existing reports and artifacts unchanged. The completed historical plan is preserved byte-for-byte at [legacy plan](docs/legacy/CURRENT_PLAN_COMPLETED_JOINT_P4_LEARNED_REFERENCE.md).

## Completed joint P1 fit and audit

The completed implementation and report are in [p1_joint_keystone_magnification](experiments/archive/reverse_transform/superseded_baseline/p1_joint_keystone_magnification/results/RESULTS.md). The scope remains P1-only: shared C1 keystone/template plus positive per-frame `m_i`, fixed-template control, temporal train/test conditional holdout, and frozen C1 transfer to C2–C4. The wide-`theta01`-envelope candidate improves native geometry but yields strongly envelope-sensitive native and zero-relative `m`; the fitted offset is a nuisance parameter relative to saved gaze, and full rank does not establish physical magnification. Conditional transfer geometry is reported without a P4 success claim. See the report for source-row/hash checks, profiles with stationarity audit, synthetic equation checks, and the model/domain specification.

Use saved per-frame horizontal gaze from the polynomial `meanP4 − meanP1` estimate. The vertical diagnostic is the native first-order x-polynomial slope; the current model is horizontal only. Do not share P1 and P4 templates, origins, or coefficients. P1 fitting uses P4-derived eligibility masks and full-roster gaze, so label the initial result conditional. Strong validation should recalibrate the gaze polynomial and references/anchors using training conditions only, with the saved-gaze result shown separately.

## Completed bounded P1 scale sensitivity

The sibling [bounded P1 scale sensitivity report](experiments/archive/reverse_transform/superseded_baseline/p1_joint_bounded_magnification/results/RESULTS.md) applies the explicitly bounded profiled scalar `m∈[0.98,1.02]` as primary, with `[0.99,1.01]` and `[0.95,1.05]` controls. Learned and fixed templates were fit at each width, with primary-width train-only controls and frozen all-capture scoring. The training-only C1 nominal-zero reference and `R0` use 3,823 saved training rows; the P1-eligible cohort and saved gaze remain full-roster. For the primary learned C1 fit, equal-fixation RMS is 0.8328 px; fitted P1 nuisance offset `θ01=42.40°`; `m` spans 0.9800–0.99495, with 2.83% of rows at the lower bound and none at the upper bound. Frozen learned/fixed C1→C1/C2/C3/C4 equal-fixation RMS is 0.8328/1.249/1.154/0.597 px and 0.9687/1.251/1.175/0.7005 px, respectively; the corresponding scale ranges and active-bound fractions are tabulated in the report. The fitted P1 nuisance offset changes from 28.85° at ±1% to 54.06° at ±5%; the fixed-template fit's offset is near −5.53°. These are fitted parameters relative to the saved gaze coordinate, not physical angles. The fixed-template scale values span about 0.9843–1.0048 across widths (the ±1% bound truncates its lower range to 0.99). All eight selected fits were valid and stationary; the primary learned fit has rank 7/7, no invalid rows, maximum scalar KKT residual 2.11e−10, independently recomputed objective agreement within 2.22e−16, and verified 44 source/artifact hash entries. The ±1/2/5% intervals are imposed model constraints, not estimated physical bounds. The primary learned values are mostly interior but below one, so the near-one assumption does not force them to center at one; widening the allowed interval lowers loss and changes the fitted offset. Treat this as bounded model-relative scale and conditional P1 geometry, not physical identification. The matched full P4 refit is now complete and is summarized above; P1 `m` worsened the tested P4 geometry, so it remains sensitivity-only rather than the working primary.

## Completed P1 zero-offset sensitivity

The [zero-offset sensitivity report](experiments/archive/reverse_transform/superseded_baseline/p1_zero_origin_comparison/results/RESULTS.md) constrains `θ01=0°` under the same `m∈[0.98,1.02]` convention and compares with the active free-`θ01` bounded all-C1 fits. This arbitrary fixed-offset fit is a sensitivity only; the active P1 protocol freely fits one `θ01` across its frames. The completed Stage 00 P4 fit used its own free `θ04` across P4 captures, with no equality constraint to `θ01`; neither parameter represents a physical angle. On C1, equal-fixation RMS increases from 0.83277 to 0.94457 px (+0.11180 px, +13.43%) for learned template and from 0.96874 to 1.02328 px (+0.05454 px, +5.63%) for fixed template. Learned `m` ranges 0.98445–1.00492 with no contacts at `θ01=0°`, versus 0.98000–0.99495 with 2.83% lower-bound contacts at free `θ01`. For matched C1 rows, learned `m`/gaze correlation changes +0.519 to −0.587 and `Δm = m_zero − m_free` has RMS 0.01427; fixed-template scales correlate 0.999987 across the constraint comparison and their `Δm` RMS is 0.0000995. Frozen C1 transfer effects vary by capture and template, including lower fixed-template RMS in C3 under the sensitivity constraint. This measures fit sensitivity to fixing a nuisance offset; it does not estimate an angle reference or physical magnification. This P1-only sensitivity contains no P4 fit; the separate matched P4 fit is complete and reported above, though physical-scale transfer remains conditional.

### Offset-coordinate and implementation audit

The offsets are fitted nuisance parameters at fixed saved gaze, and their numerical values depend on the chosen gaze coordinate. Under a common rebase of saved gaze, `theta_i → theta_i + c`, `theta01 → theta01 + c`, and `theta04 → theta04 + c`, the relative arguments `phi1` and `phi4` are unchanged. Thus the numerical values of `theta01` and `theta04` have no absolute physical-angle meaning. Changing an offset alone while holding saved `theta_i` fixed changes the corresponding K model and profiled `m`: the P1 `theta01=0°` sensitivity raises C1 equal-fixation RMS by 0.11180 px (learned) and 0.05454 px (fixed); matched-row `Δm` RMS is 0.01427 and 0.0000995, respectively.

Freely fitting separate `theta01` and `theta04` avoids forcing P1 and P4 into one shared offset. In the completed Stage 00 matched P4 fit, `theta04` moved by about 32.4° between the `g=1` and P1-m conventions, showing a substantial nuisance/scale/template tradeoff. Further P4 accommodation models must validate that offset with held-out gaze and D-equivalent leakage analyses. Existing `distortion_model/p4_joint.py` freezes per-capture offsets and does not implement the shared free `theta04` used in the new matched comparison; treat that estimator as legacy, not as evidence that future-stage work is complete.

For frame `i` and vertex `j`, center the observed P1 triangle and model it as:

```text
X1_i = center(observed P1_i)
phi_i = (theta_i - theta01)/10
Z1_ij = K1(B1_j, theta_i; alpha1, theta01, R01)
H1_ij = Z1_ij - mean_j(Z1_ij)
Xhat1_ij = m_i H1_ij
```

`K1` uses the raw native-pixel form also used for P4 keystone:

```text
a_i = b01 * phi_i^2
Z1_ij = (exp(a_i)*B1_xj, exp(-a_i)*B1_yj) /
        (1 + phi_i*(cx1*B1_xj + cy1*B1_yj)/R01)
```

Share `B1`, `alpha1=(b01,cx1,cy1)`, and one freely fitted nuisance offset `theta01` across C1 frames; `m_i` is positive and per-frame. `theta01` is a fitted offset in the saved gaze coordinate, with no absolute physical-angle meaning. The fixed `theta01=0` fit is a sensitivity, not the primary protocol. Initialize `B1` from the measured mean centered C1-zero triangle, while keeping that observed reference distinct from the latent calibrated template. Compare a learned template with a fixed-template control; if the learned template centroid is free, profile and report it rather than adding per-frame center nuisance terms. Keep the declared native coordinate origin fixed throughout; the triangle centroid is not a fitted physical distortion center. Set `R01` once from the initial observed C1-zero reference. Do not force K identity at nominal zero, impose symmetry, interpret `theta01` as a physical reference, apply a per-frame determinant normalization/full affine correction to K, or add a second scale. Do not add a free isotropic gaze multiplier to K1: it is exactly confounded with arbitrary frame `m_i`. Raw projective K may change overall size; retain that size, linked to shape by the selected model.

For fixed global parameters, first compute the unconstrained analytic scalar:

```text
m_i = <H1_i, W1 X1_i> / <H1_i, W1 H1_i>
```

The completed bounded sensitivity uses the user-specified primary interval `[0.98,1.02]`, with `[0.99,1.01]` and `[0.95,1.05]` controls. Its frame scale is the exact box-constrained least-squares solution `m_i=min(max(m_i_unconstrained, lower), upper)`. Save both values and explicit lower/upper/tie flags; this is the solution of the constrained scalar objective, not silent clipping. On strictly active bounds the profiled scale derivative is zero; at exact ties the scalar map is nonsmooth. Check lower-bound KKT gradient `>=0`, upper-bound gradient `<=0`, and interior gradient `=0`; report any violations and the interior scale-orthogonality check. A frame on a bound has lost its interior frame-scale degree of freedom under the imposed convention. Use unweighted coordinates (`W1=I`) as primary; any coordinate weights must be declared and estimated on training data only. Recompute centered `H1_i` at every global objective evaluation. Count weak-shape denominators and invalid rows; never suppress invalid inputs or domain failures.

Fix exactly one template/global-scale gauge. With a learned template, either set mean `log(m_i)=0` over C1-zero calibration rows or fix template energy to the native initial reference; state the choice. A fixed-template control already fixes template scale, so do not also impose the mean-log-scale constraint. `m=1` at the operational reference is a convention, not an optical assertion. A reference mean alone does not remove gaze-dependent scalar confounding. Any gauge conversion must transform template, projective coefficients, and scales consistently under the raw projective model; never normalize each K output independently.

Use multistarts and report the data-only profiled-global Jacobian rank/singular modes and profiles for nuisance offset `theta01`, `B1`, `cx1`, and `cy1`. Report parameter, scale, and prediction stability across starts, template controls, and reference/profile uncertainty. A scale-neutral K shape may be used only as a diagnostic decomposition, not as a replacement forward model.

### Holdout and transfer audit

After freezing C1 global calibration, infer `m_i` on held-out rows from their P1 observations alone. Report native forward residual and the shape residual orthogonal to the scalar direction, within-fixation `m_i` variation, gaze trends, eligible-row counts, positivity, denominator margins, and support. Held-out `m_i` consumes one degree of freedom per frame and is not independent magnification truth. A three-point affine reconstruction identity is not validation. Include reference/global-K/noise uncertainty; do not treat fragmented frozen-segment zero-width bootstrap intervals as certainty. Then apply frozen C1 calibration to C2–C4 and report transfer. Any alignment adjustment or capture-specific control fit must be P1-only and identified as such.

Deliver `m_i` arrays with source-frame IDs, global parameters CSV, masks and hashes, a summary/report, actual-gaze plots, and fixation-segment time series broken at gaps. Keep the audit reproducible and preserve all observations; do not filter rows by post-fit residual.

## P4 recalibration after the P1 audit

The matched whole-model P4 comparison is complete for the current bounded learned-P1 `m` candidate. It found worse geometry than `g=1`, so use `g_i=1` as the working primary input for the next `k(A)`/A calibration. Keep the P1-derived `m_i` as an explicit conditional sensitivity only. Any further scale comparison must refit the complete P4 K/barrel/B model under matched parameters, weights, splits, and priors; never apply `m_i` to a previously fit P4 calibration. P4 must never select `m_i` or tune P1 K/B. The comparison does not establish physical scale, and the P1 scale convention remains unresolved.

For P4, keep separate P1/P4 templates, origins, and coefficients. Use the following native-pixel forward model, with the declared radial coordinate origin fixed and `R04` set once from the initializer and held fixed as `B4` is learned:

```text
Q_j = (1 + kappa(A)*||B4_j||^2) B4_j
phi4 = (theta - theta04)/10
a4 = b04 * phi4^2
Z4_ij = (exp(a4)*Q_xj, exp(-a4)*Q_yj) /
        (1 + phi4*(cx4*Q_xj + cy4*Q_yj)/R04)
mu4_i = mean_j(Z4_ij)
Yhat4_ij = g_i * (Z4_ij - mu4_i),  with primary g_i=1
```

Recompute `mu4_i` at each objective evaluation. Do not assume K is identity at zero gaze. For the working primary, `g_i=1`; an optional sensitivity may set `g_i` to frozen P1 `m_i` and must be refit end-to-end. In inversion, divide centered measured P4 vertices by the same declared `g_i` first, then restore the model centroid, invert K, and invert the barrel map. Do not add a second P4 scale or normalize K outputs.

Keep the P4 hypotheses and conventions: `theta` starts from the saved per-frame horizontal polynomial; use native pixels and recompute each model centroid. Use `g_i=1` for the working primary and treat frozen P1-derived `m_i` as a conditional sensitivity, since the completed Stage 00 fit worsened the P4 objective. Retain `A_ref=0.36036036 D` (`kappa(A_ref)=0`), nominal demand anchors C1=0.36036, C2=4, C3=3, C4=2 D, and the `sigma_A=0.25 D` soft anchor. These labels define the D axis, not framewise truth. Capture and demand are confounded; do not jointly free-fit A means and a `kappa` slope. Fit a P4 nuisance offset `theta04` shared across its fit frames/captures, separately from P1 `theta01`, with no equality constraint. Both are offsets in the saved gaze coordinate and have no absolute physical-angle meaning. Do not impose symmetry or recentering.

## P4 stages and deliverables

### 00 — Baseline sensitivity (`stage_00_baseline_sensitivity`)

The descriptive P1 size analysis, positive-only fit, bounded P1 sensitivity, fixed-`theta01=0` sensitivity, and matched P4 scale-effect comparison are complete. The P1-m convention worsened this P4 objective, so `g=1` remains the working primary and frozen P1 `m_i` is sensitivity-only. **Do not treat fitted scale as physical or transfer it without a full P4 refit.** Next fit and validate the accommodation-dependent radial relation `k(A)` and frame/fixation accommodation `A` under the working primary; retain the P1-m branch as an explicitly conditional comparison. The fixed-zero offset comparison is a sensitivity only, not a proposed primary or a physical-angle test. The P4 coupled `K(A)` interpolation is a descriptive control, including any retained nuisance-offset `theta04(A)` dependence; distinguish it from the primary shared-K/separable hypothesis. With `H4=K4(...)-mean(K4(...))`, preserve residual and sensitivity calculations below.

### 01 — Separable calibration (`stage_01_separable_calibration`)

Reuse the audited Stage 00 `g=1` shared `B`/K/`theta04` and capture-specific radial fit as the static geometry baseline; do not repeat that same unconstrained fit as a new deliverable. Extend it by fitting a shared linear `k(A)=s(A−A_ref)` relation (`k(A_ref)=0`) under the working `g=1` convention, using nominal capture demand anchors and the same equal-capture/fixation weighting. Compare its loss and held-out performance with the Stage 00 free-per-capture `k` baseline; include frozen P1 `m` only as a conditional sensitivity. Because capture and nominal demand are confounded, describe this as a nominal-demand indexed radial relation, not a physiological accommodation slope. Fit and report the relation jointly with shared P4 `B`, K, and free `theta04` as needed to test whether the restricted relation remains adequate, with multistarts and domain/stationarity checks. This adds the `k(A)` dependency to the audited static baseline; it does not test arbitrary per-capture `K(A)`.

### 02 — Dependency and decoupling (`stage_02_dependency_and_decoupling`)

After validating the shared `k(A)` relation in Stage 01, test whether radial coefficients vary with gaze and whether K varies with accommodation anchors. Add small `K(A)` terms or barrel-by-gaze interaction only as controlled model families, one dependency at a time, with matched splits and anchor budgets. Optional capture-alignment deviations must not absorb the signal without bound. Retain a uniform intrinsic-size control with matched regularization and splits. Reference invariance supports stability but does not prove physical separability.

### 03 — Frame accommodation (`stage_03_frame_accommodation`)

Freeze shared P4 transformation, declared input scale (`g=1` primary; frozen P1 `m_i` sensitivity only), and `kappa(A)`. Estimate framewise `A_i` at saved `theta_i`, comparing geometry-only estimates with fixation-mean soft anchors of width 0.25 D. Estimate fixation means on training data and freeze for held-out scoring; show their gaze dependence before anchoring. Include a nominal capture-mean anchored control. Do not recenter A by fixation to remove gaze bias. Separate within-fixation variation from systematic fixation-mean changes and disclose the loss/prior balance.

### 04 — Validation and report (`stage_04_validation`)

Use temporally blocked held-out rows with gaps and leave-one-gaze-fixation-out analyses with multiple starts. In strong holdouts, recalibrate the gaze polynomial, P1 scale reference/calibration, and all penalties, reference, K, and barrel inputs from retained training conditions; infer held-out `m_i` from held-out P1 only. Show full-roster frozen-gaze as a conditional control. No held-out target may enter calibration, anchors, or centering. For leave-capture/demand-out, distinguish interpolation from endpoint extrapolation without recalibrating against held-out labels. Use repeated blocked splits or bootstrap; frames are temporally correlated. Profile P4 nuisance offset `theta04`, B, and `kappa` coupling, including weak/no-anchor cases; penalized profiles alone are not posterior confidence.

## Leakage metrics and interpretation

Stack the six centered P4 vertex coordinates consistently; let `r=Y−Yhat`. Use `g_i=1` for the working primary and report the frozen P1 `m_i(theta)` derivative as a separate sensitivity. For a declared input scale `g_i(theta)`, report both total and conditional gaze sensitivities:

```text
J_A = dYhat/dA  (includes kappa'(A), allowed K'(A), and derivative of model centroid)
J_theta,total = (dg_i/dtheta) H4 + g_i dH4/dtheta
J_theta,conditional = g_i dH4/dtheta
J_g = dYhat/dg = H4
```

With documented weights W, report:

```text
deltaA_equiv = (J_A^T W r)/(J_A^T W J_A)                 [D]
dAhat/dtheta = -(J_A^T W J_theta)/(J_A^T W J_A)         [D/degree]
rho = (J_A^T W J_theta) /
      sqrt((J_A^T W J_A)*(J_theta^T W J_theta))
dAhat/dg = -(J_A^T W J_g)/(J_A^T W J_A)                [D per unit g]
g*dAhat/dg                                               [D per unit relative scale error]
```

For the P1-m sensitivity, report the fixed-`m` conditional derivative and P1 global-calibration uncertainty. A 1% relative scale error contributes 0.01 times `g*dAhat/dg`. Keep raw area/similarity fixed-observation derivatives as controls. Low `J_A` norm means ill-conditioning, not finite confidence. Check the local approximation against nonlinear fits; separate observed gaze-associated changes from sensitivity to gaze error. Without independent ground truth, use “accommodation-equivalent model mismatch” and “gaze-associated bias,” not true framewise accommodation.

Primary endpoint is reduced gaze-associated A bias and held-out accommodation-equivalent residual under the working `g=1` primary, without merely strengthening anchors. Pixel RMS and geometry fit are supporting metrics. Report capture-by-gaze means/spreads of P1 `m_i` as sensitivity, P1 shape residuals, separate scale-to-A versus direct P4 K-to-A leakage, mean A changes against same-capture zero and nominal anchors, residual D spread, `rho`, within-fixation tails, `kappa(A)` by gaze, K by A, and uncertainty. Do not set an arbitrary D threshold absent an application requirement.

## Numerical checks and plots

Preserve complete-fixation eligibility and saved outlier masks. Keep captures separate in means and legends; plot eligible rows using actual estimated gaze and nominal target labels. Check equations/roundtrips, positive `m_i`, positive radial scale/derivative, positive projective denominators, invalid counts and margins. Never clip or drop invalid points to improve fit. Document actual measurement/prior scales and penalties; compare anchor factors 0.5, 1, and 2 using validation, with geometry-only/weak-anchor sensitivity. Keep native radius normalization, gauge, and source rows consistent. Vectorize/batch framewise profiling in NumPy or CuPy float64; use CPU for small global solves when useful and avoid thread oversubscription. If using a GPU, check CPU/GPU equation agreement.

Plots include P1 size and shape versus actual gaze, P1 fixation-segment `m_i` time series broken at gaps, P4 A versus gaze, frame A versus time without lines bridging gaps, capture-by-gaze D-residual heatmap, `kappa(A)` by gaze, K by A, reference/native-triangle overlays, and forward-fit metrics.
