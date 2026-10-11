# Joint P4 keystone and barrel fit with a learned shared triangle

**Status:** Completed as a conditional geometric model comparison. The planned protocol below was run and reported at [`joint_keystone_barrel_learned_reference/results/RESULTS.md`](experiments/archive/reverse_transform/superseded_baseline/joint_keystone_barrel_learned_reference/results/RESULTS.md). This is not a physical lens or accommodation calibration. The plan is retained as the provenance record for the executed comparison.

**Experiment:** `experiments/archive/reverse_transform/superseded_baseline/joint_keystone_barrel_learned_reference/`

**Current evidence:** independent per-capture keystone candidates are reported in [`per_capture_p4_inverse_keystone`](experiments/archive/reverse_transform/superseded_baseline/per_capture_p4_inverse_keystone/results/RESULTS.md), and the conditional zero-gaze common-reference stage is reported in [`fixed_keystone_barrel_zero_gaze`](experiments/archive/reverse_transform/superseded_baseline/fixed_keystone_barrel_zero_gaze/results/RESULTS.md). The latter fixes the Capture 1 inverse-mean triangle across captures; that convention is accepted for that stage, inherits the preceding keystone assumptions, and does not establish a physical barrel response. The all-fixation learned-reference comparison is reported in [`joint_keystone_barrel_learned_reference`](experiments/archive/reverse_transform/superseded_baseline/joint_keystone_barrel_learned_reference/results/RESULTS.md).

## Completed zero-gaze benchmark

The conditional zero-gaze stage fixed each capture's keystone and shared nuisance magnification `g=1`, used the common C1 reference, and fitted one radial coefficient per capture with `kappa_1=0` by convention. It did not fit per-frame `kappa`, `g`, or accommodation `A`.

| Capture | `kappa` (px⁻²) | Pooled inverse frame RMS, baseline → barrel (px) |
|---:|---:|---:|
| C1 | 0 | 0.3744 → 0.3744 |
| C2 | −1.88307e−6 | 13.5654 → 3.3433 |
| C3 | −1.49124e−6 | 10.7629 → 2.4116 |
| C4 | −9.38964e−7 | 6.7675 → 1.8763 |

The same report gives frame-first mean-triangle inverse RMS, which drops from 13.5598 to 3.3089 px in C2, 10.7255 to 2.1904 px in C3, and 6.7374 to 1.7403 px in C4; C1 is 0.0000318 px before and after at reported precision. This is an inverse diagnostic under the fixed common reference. The completed all-fixation comparison below evaluates a freely learned common triangle and a uniform-size alternative against the fixed-reference model rather than treating the zero-gaze reference as established ground truth.

## Joint-stage findings

The all-row equal-capture/fixation measured P4 MSE was 3.764975 px² for the historical initialization, 1.351373 px² for fixed common `B` with capture-specific radial terms, 1.245008 px² for the learned common `B` radial model, and 1.256081 px² for the learned common `B` uniform-size control. Learning `B` lowered the fixed-radial MSE by 7.87%; the uniform-size control was 0.89% above the learned-radial MSE. These are in-sample comparisons because all rows contributed to each all-row fit.

Leave-one-target fits used one start each. Learned-radial held-out MSE was worse than fixed-radial at both endpoints: 3.84615 vs 2.58783 px² at −10°, and 4.58719 vs 2.62639 px² at +10°. The temporal conditional and training-only-anchor fits each used three starts; their held-out row scores still condition on the frozen full-roster gaze estimates. The all-row primary and anchor-sensitivity fits used five starts each.

The learned-radial inverse frame RMS versus its fitted `B` was 1.2704 px and its frame-first mean-triangle RMS was 1.0551 px, compared with 1.3279 and 1.1191 px for fixed `B` radial. The uniform-size control had lower inverse diagnostics (1.1962 and 1.0020 px), but inverse compression is not the measured-fit objective. The fitted triangle and inverse mean shape metrics, residual quantiles, domain margins, and all per-capture parameters are in the standalone report. Data-only Jacobian ranks were 15/15 for fixed `B` and 21/21 for both learned-template controls. No active bounds or invalid observed inverse frames occurred in the three primary models. The full source-hash, CPU/GPU, synthetic roundtrip, and row-split checks are recorded in `results/independent_audit.json`.

Capture and nominal demand remain confounded. The learned radial fit improves all-row forward loss but not the endpoint leave-one-target fits; the result does not identify a physical barrel mechanism or accommodation response. All temporal test scores are conditional on gaze initialized using the full roster. No per-frame magnification, radial coefficient, or accommodation state was fitted.

## Executed protocol and model family

The experiment compared a common, freely learned P4 triangle with capture-specific gaze-dependent keystone and one radial coefficient per capture against (a) the fixed common-reference model and (b) an intrinsic uniform-size alternative. It used the current per-frame horizontal gaze estimates as fixed inputs. Capture and nominal accommodation demand remain confounded; coefficient-versus-demand association is not a causal accommodation effect.

For capture `c`, fixation `f`, row `i`, and vertex `j`, let `Y_cfij` be the existing measured P4 triangle after the existing per-frame centering convention. Let `B_j` be the shared learned reference triangle, `Q_kappa_c(B_j)=(1+kappa_c*||B_j||²)B_j`, and `K_c` the existing raw keystone family. Initially allow updates only to `b0_c,cx_c,cy_c`; initialize from the saved per-capture fits. Freeze each capture's gaze origin, fitted `R_c` scale convention, and declared distortion-center convention. Do not change the gaze inputs during this experiment. The frozen per-capture forward keystone is

```text
phi = (theta_x - origin_c)/10
a = b0_c * phi^2
K_c(z,theta_x) = (exp(a)*z_x, exp(-a)*z_y) /
                 (1 + phi*(cx_c*z_x + cy_c*z_y)/R_c)
```

where `R_c` is fixed from that capture's saved fit. For scaled optimization use `beta_c=kappa_c*M`, with `M=max_j||B_ref,j||²` computed once from the saved C1 reference and held fixed. Thus `kappa_c=beta_c/M` in native `px^-2` units even while the trial `B` changes.

For each trial parameter set and observed `theta_x_ci`, recompute the predicted image triangle and its centroid:

```text
Z_cij(theta_x_ci) = K_c(Q_kappa_c(B_j), theta_x_ci)
mu_ci = mean_j Z_cij(theta_x_ci)
Yhat_cij = Z_cij(theta_x_ci) - mu_ci
```

The primary data loss is the measured centered-vertex loss between `Y_cfij` and `Yhat_cij`, with equal total weight per capture and per fixation, and equal row/vertex weight within each fixation. Recompute `mu` for every objective evaluation from the trial `B`, `kappa`, and `K` parameters. Keep the forward measured-vertex objective primary; inverse residual spread alone can shrink because a transform removes observed geometric variation.

The shared `B` is initialized from the existing C1 zero-gaze inverse-mean triangle, including its small nonzero centroid, but is learned jointly and is not hard-fixed or constrained to be symmetric. Preserve the prescribed measured and predicted per-frame centroid subtraction in the forward loss. Do not add any other recentering of `B` or inverse outputs, or any per-frame normalization. The radial map remains about the declared coordinate origin. A learned triangle centroid may expose a translation or other weak mode; assess this with the data-only scaled Jacobian and profiles. Add no translation, size, RMS, or symmetry constraint unless the data-only analysis justifies it and the constraint is documented. `kappa_1=0` defines the relative radial convention for C1; it does not assert an undistorted physical C1 image.

The radial inverse can be evaluated for any trial `B`; it does not mathematically require a prescribed common reference triangle. Here, centering the prediction makes `mu` depend on the latent `B`, `K`, and `kappa`, so the frame-first inverse mean from the earlier fixed-reference stage is only an initializer. Repeatedly inverting observations and averaging them is not guaranteed to minimize the forward measured-vertex objective. Preserve individual measured rows throughout fitting rather than replacing them with fixation means; real within-fixation frame variation remains in the data loss and diagnostics. An inverse transform can compress measurement noise or residuals without improving the measured forward fit; fixed global size alone does not prevent anisotropic compression. Therefore use measured forward vertex loss as the primary objective and held-out score, with inverse spread only as a separate diagnostic.

Keep the present fixed per-capture `R_c` values (or exactly equivalent native projective coefficients) as the keystone scale convention while `B` is learned. Use the fixed numerical parameter scales defined above, not scales recomputed from the evolving `B`. Check for parameter gauges and rank deficiencies before adding constraints. Keep `g=1` throughout this step: no per-frame or capturewise magnification, no per-frame radial coefficient, and no accommodation `A`.

## Models compared

1. **Fixed-reference benchmark:** reproduce the saved common-reference model over the same all-fixation roster where possible, with the existing `B_ref`, initialized/frozen gaze, and the proposed `K` updates treated consistently. Also retain the saved zero-gaze-only result as its historical benchmark; distinguish it from a newly evaluated all-fixation comparison.
2. **Learned common triangle with radial terms:** jointly learn all six coordinates of `B` and one `kappa_c` per capture, with `kappa_1=0` by convention. Update only `b0,cx,cy` per capture initially. Apply soft anchors to previous `K` and `kappa` estimates; previous fits are initialization and prior information, not ground truth.
3. **Uniform intrinsic-size control:** use `s_c*B` before the same `K_c`, with `s_1=1` by convention, and no jointly fitted `kappa`. Fit this as a separate branch with the same data, gaze, `K` update family, weighting, and anchor-sensitivity procedure. Do not jointly fit `s_c` and `kappa_c` in the primary comparisons. The three reference vertex radii are close, so the available data may weakly distinguish uniform size from radial deformation.

Report training and held-out losses separately, and keep prior penalties separate from data loss. For the learned-reference branch, initialize from the old C1 triangle but report how the learned `B` differs from that initialization. Compare it with the fixed-reference branch; do not assume the learned triangle is a physical undistorted shape.

## Anchors, domains, and numerical checks

Use soft anchors in scaled parameter coordinates for `b0,cx,cy` and `kappa`. Treat saved historical estimates as multistart initialization only unless a run explicitly includes them as penalty or constraint information; report that distinction. Do not invent numerical prior widths: select and document a defensible range, then show at least narrower and wider anchor sensitivity cases. For each case report the unpenalized data loss, anchor penalty, fitted parameters, and their movement. Include a data-only local Jacobian rank, scaled singular values/conditioning, and profiles for the important `kappa`/template tradeoffs. An optimizer success flag alone is not evidence of identifiability.

Enforce and report the positive radial scale and derivative conditions over every reference or recovered point used by the forward model, and positive frozen/updated keystone denominators over the full observed gaze support. Do not silently clip invalid points. Radial inverse points outside the reachable monotone branch or points failing the denominator domain remain explicit failures; do not drop them from the objective to improve the score. Report invalid counts by cause, minimum margins, bound contacts, and multistart spread.

Preserve the existing per-fixation masks and geometry filter, and use only the existing complete rows. Do not alter row eligibility after seeing residuals. Save per-fixation and per-frame support, and keep frame-first inverse means distinct from inverse-of-mean calculations. Include CPU forward/inverse equations and roundtrip checks on synthetic points; use vectorized frame arrays and batched GPU computation only where it is beneficial. Avoid excessive CPU thread oversubscription.

## Validation and diagnostics

Before fitting, partition rows into temporally contiguous training and held-out blocks **within each fixation**, with gaps between blocks to reduce adjacent-frame leakage. Historical `K`/`kappa` anchors and frozen origins were estimated using the full roster, not training rows alone. A split that uses those full-roster values for penalties or fixed inputs is a conditional historical-calibration sensitivity analysis, not independent held-out validation. Separate historical parameter values used only to initialize multistarts from values used as penalty/constraint information. For a stronger predictive check, recompute training-only `K`/`kappa` anchor inputs and fit the latent/reference parameters using training rows only, then evaluate held-out measured P4 vertices without refitting. Add leave-one-fixation/gaze-out analysis as a separate diagnostic. State which conditions interpolate within the measured gaze range and which extrapolate beyond its endpoints. Even with training-only parameter anchors, the supplied gaze estimates were initialized using the full existing roster, so P4 held-out comparisons remain conditional on those gaze inputs and are not fully independent gaze validation. These are roster-based predictive checks, not physiological validation.

Report:

- measured forward vertex error by capture, fixation, and temporal holdout;
- inverse vertex RMS and distribution, separately for pooled frames and frame-first fixation means;
- within-fixation frame variation separately from between-fixation mean-shape variation;
- radial and tangential inverse residuals versus gaze, plus side lengths, angles, signed area, and symmetry diagnostics;
- learned `B` coordinates, centroid, scale, and shape against its initialization and the fixed-reference benchmark;
- `K` and `kappa` estimates, anchor movement, multistart stability, bound/domain status, and data-only rank/profile results;
- the uniform-size control and whether it is distinguishable from radial deformation over the available vertex radii.

Do not infer a physical optical axis, accommodation response, or undistorted reference from the smallest residual or most symmetric triangle alone. Retain residual size and report parameter uncertainty/tradeoffs supported by the profiles rather than declaring a unique physical decomposition when gauges or weak modes remain.

## Completed milestones and outputs

1. **Setup and gauge audit:** completed; frozen inputs/masks/gaze, fixation-block splits, trial-consistent centroid, domain, equation, roundtrip, and rank checks are recorded.
2. **Fixed-reference and uniform-size controls:** completed using the same all-fixation roster.
3. **Learned-triangle branch:** completed with per-capture `b0,cx,cy`, one radial coefficient per capture, five starts, and anchor sensitivity.
4. **Held-out and stability analysis:** completed; temporal test and leave-one-target results are conditional on full-roster gaze, and endpoint LOO fits each used one start.
5. **Standalone report:** completed under `experiments/archive/reverse_transform/superseded_baseline/joint_keystone_barrel_learned_reference/`, including runner, frozen input hashes, parameters, masks/splits, arrays, plots, `RESULTS.md`, and independent audit.

The comparison reports all-row measured P4 fit, conditional temporal holdouts, leave-one-target diagnostics, inverse residual distributions, model-domain margins, and parameter sensitivity. The learned-reference radial branch improved the all-row objective over the fixed-reference radial branch; the uniform-size control was close, while one-start learned LOO fits were worse at both gaze endpoints. No arbitrary improvement threshold or claim of physical separation is used.

## Deferred work

This plan does not estimate per-frame magnification or accommodation, does not jointly fit `g` or `A`, and does not use P1 to define a response. If later work adds P1-derived shared nuisance magnification or accommodation states, it should be a separately scoped model comparison with explicit identifiability and validation evidence.
