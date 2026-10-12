# P1 scale calibration, capture-specific P4 inversion, and joint gaze/accommodation estimation

> **Archived historical plan.** This plan described the superseded P4 pipeline. Its experiment files and outputs are preserved under `experiments/archive/reverse_transform/superseded_p1_scale_capturewise_p4/`. The active corrected barrel-first protocol is in the root [CURRENT_PLAN.md](../../CURRENT_PLAN.md); do not use this plan to run current experiments.

**Protocol status:** Original five-stage benchmark, learned P4-reference sensitivity, and section 11 per-gaze barrel-anchor/barrel-first comparison are implemented and audited. Sections 1–10 document the preserved earlier benchmark. Completed results and limitations are summarized in [CURRENT_STATUS.md](../../CURRENT_STATUS.md). The prior version is preserved in [docs/legacy/CURRENT_PLAN_PRE_ALL_CAPTURE_P1_SCALE.md](CURRENT_PLAN_PRE_ALL_CAPTURE_P1_SCALE.md).

Ray tracing establishes one shared physical magnification `g` for P1 and P4; the fully recalibrated `g=1` branch is a diagnostic and does not change that constraint. The goal is to estimate how distortion depends on gaze and accommodation while measuring how geometry error leaks into accommodation. Use all four captures to calibrate shared P1 keystone and bounded frame magnification; apply that same P1-derived magnification to P4; initialize independent P4 keystone models and one barrel coefficient per capture; then jointly estimate frame gaze and accommodation with frozen soft-anchor targets. Native vertex agreement, held-out geometry, and accommodation-equivalent residuals are the principal evidence. A flatter accommodation trace alone does not establish improvement.

The new primary input scale is the P1-derived frame value `g[c,t]`. The earlier `g=1` result is a comparison control. Existing runs used different calibration scopes and do not count as completion of this plan. In particular, the previous bounded P1 fit calibrated global parameters on Capture 1 and transferred them to the other captures; this plan calibrates P1 globals jointly on captures 1–4. The prior matched P4 refit worsened its objective by 42.58% with that older P1 scale candidate; preserve this evidence and test the new candidate rather than assuming scale transfer succeeds. See [the previous P4 comparison](../../experiments/reverse_transform/p4_gaze_accommodation_decoupling/stage_00_baseline_sensitivity/p4_p1_scale_effect/results/RESULTS.md) and [experiment history](../../experiments/RECORD.md).

## 1. Data, notation, and fixed conventions

- `c=1,...,4` identifies capture, `f` identifies fixation, `t` identifies the original frame, and `j=1,2,3` identifies the corresponding vertex.
- Preserve original P1/P4 detections, vertex labels, source-frame IDs, fixation intervals, and reviewed eligibility/outlier masks. Use the same matched frame roster in comparisons. The existing filtered cohort contains 88,978 rows; verify and report the roster actually used by this new run. Do not remove rows based on post-fit residuals.
- `P1[c,t,j]` and `P4[c,t,j]` are observed native-pixel coordinates. `Xr=P r-mean_j(P r)` denotes a centered observed triangle for pattern `r`. Centering defines the observed shape; it does not measure the physical distortion center.
- Keep captures separate when forming means, plotting, and reporting. Means across gaze fixations within one capture are allowed where specified. Accommodation conditions must not be averaged together to create calibration observations.
- Each capture contains nominal horizontal gaze fixations at `[-10,-5,0,5,10]` degrees. Nominal demand labels in capture order are `[0.36036036036036034,4,3,2]` D.
- Set `A_ref=0.36036036036036034` D. A zero relative barrel coefficient at this reference does not assert that the reference pattern has no intrinsic optical distortion.
- P1 and P4 have distinct templates, keystone parameters, and nuisance offsets. Fit one shared P1 offset `theta01` and one shared P4 offset `theta04`; do not equate them or interpret them as physical angles. The P4 coefficients may be capture-specific while its offset uses one common coordinate convention.
- Freeze every model's native radius `R` from its declared initializer and preserve it during optimization. This radius sets coefficient units; it is not a normalization of corrected output. No per-frame RMS, determinant, area, rotation, or full affine normalization is added to keystone.
- Weight captures equally, fixations equally within capture, and frames equally within fixation. Average squared Euclidean vertex residuals over the three vertices. Report this weighted RMS separately from pooled frame/vertex RMS and RMS of mean triangles.

## 2. Transform model and inversion

The primary model initially follows the existing horizontal keystone family, with vertical gaze retained as a diagnostic from Stage 01. The existing bounded P1 implementation is horizontal only. Do not claim 2D correction or jointly refined vertical gaze from this family. Any 2D extension requires an explicit equation, derivatives, identifiability audit under the small observed vertical motion, and a separately reported comparison.

For a native template point `p=(x,y)`, define raw keystone by:

```text
phi = (theta_x - theta0r)/10
K_r(p,theta_x;alpha_r,R_r) =
    (exp(b_r*phi^2)*x, exp(-b_r*phi^2)*y) /
    (1 + phi*(cx_r*x + cy_r*y)/R_r)
```

Here `theta0r` is the pattern's nuisance offset. Keystone's native size change is retained. The inverse-barrel calculation uses the inverse of this forward radial map:

```text
D_kappa(p) = (1 + kappa*||p||^2)*p
```

P4's corresponding centered forward model is:

```text
Q_j = D_kappa(B4_j)
Z_j = K4_c(Q_j,theta_x)
mu4(theta_x,kappa) = mean_j(Z_j)
Xhat4_j = g*(Z_j - mu4)
```

Its frame inverse is:

```text
Bhat4_j = D_kappa_inverse(
    K4_c_inverse(X4_j/g + mu4(theta_x,kappa),theta_x)
)
```

Therefore the inverse order is: divide the centered observation by P1 scale, restore the model-predicted raw centroid, invert keystone, then invert barrel. The raw centroid is recomputed for every trial gaze, barrel coefficient, template, or keystone parameter. Restoring it is part of the inverse equation, not an extra fitted translation or normalization.

Frozen de-keystoned triangles may initialize the barrel fit, but cannot replace this trial-consistent inverse in the calibrated result. Even with frozen keystone parameters, changing barrel changes the raw model centroid. Use the increasing radial branch, and record every forward/inverse domain failure.

## 3. Stage 01 — Per-capture polynomial gaze initialization

For each capture independently:

1. Compute `delta[c,t]=mean_j(P4[c,t,j])-mean_j(P1[c,t,j])` on the retained matched frames.
2. Compute the arithmetic mean displacement in each fixation. Let `(x0,y0)` be the nominal-zero fixation mean and `h=(max_f(mean_delta_x)-min_f(mean_delta_x))/2`.
3. Set `u_f=(mean_delta_x_f-x0)/h` and fit `a1*u_f+a2*u_f^2` to the five nominal horizontal gaze labels using equal fixation weights. The polynomial intercept is zero at the declared nominal-zero displacement.
4. For each frame, set `theta_x_initial=a1*u+a2*u^2`, with `u=(delta_x-x0)/h`, and `theta_y_initial=(a1/h)*(delta_y-y0)`. Vertical gaze uses the horizontal polynomial's native first-order slope only, without a separate vertical polynomial.
5. Check finite inputs, nondegenerate `h`, and polynomial monotonicity over the observed frame input range. Distinguish polynomial values at mean inputs from the means of framewise polynomial outputs.

These estimates become frozen gaze-anchor targets for the subsequent joint fit. Per-capture calibration compensates capture-specific displacement offsets empirically; it does not establish independent gaze accuracy. Nominal gaze and fixation order remain confounded.

**Deliverables:** polynomial coefficients and units per capture, fixation means/counts, original frame IDs, frame gaze arrays, masks/hashes, calibration residuals, and actual-gaze fixation/time plots. For validation, refit the calibration using training conditions only; hold out fixation labels from calibration.

## 4. Stage 02 — Shared all-capture P1 keystone and bounded scale

Use the Stage 01 frame gaze as fixed input. Jointly fit one shared P1 keystone vector, one shared P1 template, one P1 nuisance offset, and a scale for each retained frame across captures 1–4.

Initialize `B1` from the mean centered P1 triangle in Capture 1's nominal-zero fixation. Freeze its native radius `R01`. For the learned-template primary, preserve the existing single template-scale gauge: template energy equals the initializer's energy, and the template centroid is zero in the declared coordinate convention. Do not also impose mean `g=1` or mean `log(g)=0`.

```text
Z1[c,t,j] = K1(B1_j,theta_x_initial[c,t])
H1[c,t,j] = Z1[c,t,j] - mean_j(Z1[c,t,j])
Xhat1[c,t,j] = g[c,t]*H1[c,t,j]
g_unconstrained[c,t] = <H1[c,t],X1[c,t]>/<H1[c,t],H1[c,t]>
g[c,t] = argmin over 0.98 <= g <= 1.02 of ||X1[c,t]-g*H1[c,t]||^2
```

Use the existing exact bounded scalar least-squares profile and native forward P1 objective. Preserve the unconstrained scale, lower/upper contact flags, tie flags, and scalar KKT checks. P1 alone supplies the scale; P4 may not optimize it or select the P1 calibration. Recompute centered P1 predictions at every global parameter trial.

The primary interval is the user-selected `[0.98,1.02]`. Include a fixed-template control under the same interval. Wider/narrower scale bands are sensitivity checks when contact rates or parameter profiles warrant them; they must not silently replace the primary bound. Keep multistart, nuisance-offset profiles, data-only rank/singular modes, and prediction/scale stability audits.

**Deliverables:** shared calibration parameters, native template/radius/gauge, frame `g` joined by exact `(capture,source_frame)` identity, P1 forward/inverse/shape residuals, scale distribution and bound fractions by capture/fixation, `g` versus actual gaze, and gap-aware fixation time plots. Shared-K failure or systematic capture mismatch is reported rather than absorbed into a new free scale. Do not interpret bounded `g` as independently measured physical magnification.

## 5. Stage 03 — Capture-specific P4 keystone initialization using P1 scale

Use Stage 01 gaze and exactly the Stage 02 P1 frame scales. Fit an independent P4 keystone coefficient vector for each capture. These initial maps may contain accommodation-related effects.

Each capture starts with its own mean scale-corrected centered P4 pattern from its nominal-zero fixation. This stage does not impose the common Capture 1 P4 reference. Preserve each initializer's native size and frozen radius, and use a fixed-template primary with a learned-shape control whose native template energy stays fixed to that capture's initializer. Avoid imposing symmetry or adding a P4 scale. Retain one shared P4 nuisance offset as the common gaze-coordinate convention; fit the independent coefficient blocks in one batch if necessary to enforce it.

Fit the inverse-keystone recovery against the declared capture template using native vertex RMS. For every trial parameter vector, restore its model centroid before inversion. Report the matching measured-pixel forward objective and a forward-fit control, because a changing inverse can amplify or contract observation noise. Expose parameter, inverse, and forward changes across starts and template controls.

The output is each capture's scale/keystone-corrected native triangle per frame, with residuals and domain flags. It is an initializer rather than proof that keystone is accommodation independent. Captured radial effects may have been absorbed into the template or K; retain this concern in the Stage 04/05 interpretation. Compare predicted maps over a common gaze grid as well as coefficients, since fitted offsets, templates, and frozen radii affect coefficient comparability.

**Deliverables:** four P4 K vectors, shared P4 offset, native templates/radii, frame inverse triangles and centroids, per-fixation inverse/forward/radial/tangential errors, parameter tables, and capture overlays.

## 6. Stage 04 — One barrel coefficient per capture and a Capture 1 reference

Construct the common reference once from Stage 03 Capture 1 results:

```text
B4_ref,j = mean over C1 fixations f of(
    mean over retained frames t in fixation f of Bhat4_C1,t,j
)
```

Give the five fixation means equal weight. Correct frames before averaging. Save the contributing rows, fixation means, final native vertices, and reference centroid. Preserve the recovered coordinate origin; do not recenter or normalize the reference without an explicit consistent transformation of the radial model and K.

Freeze this common reference during the stage. It is an empirical convention that inherits Stage 03 distortion and accommodation assumptions; it is not an undistorted physical triangle. For held-out evaluation, construct it from training rows/fixations only.

With Stage 03 K, Stage 01 gaze, and Stage 02 `g` frozen, fit one capture-constant `kappa_c` using all its eligible fixation frames. Fix `kappa_1=0` to define the relative barrel origin. For C2–C4 minimize equally weighted fixation inverse vertex mean-square error against `B4_ref`, using the full trial-consistent inverse in section 2. Retain the original K radii/units when feeding the common reference into the corresponding forward models.

Report the matching native forward residual for every inverse-fit candidate and a forward-fit coefficient control. Separate frame residuals, fixation-mean triangle residuals, and the mean across fixations. A reference formed from C1 data makes its mean agreement partly conventional; it does not demonstrate successful removal of C1's gaze distortion.

Compare `kappa=0` with fitted barrel. A uniform radial-size control may diagnose whether the three vertices provide information beyond overall size; it is a separate control and never supplies a second scale to the primary model. Report radial/tangential residual components and coefficient support by gaze, including nonlinearity and any boundary contact.

**Deliverables:** one coefficient per capture, objective profiles and domain margins, reference vertices/provenance, corrected frames, before/after forward and inverse errors by capture/fixation, and actual-gaze overlays. Show coefficient association with nominal demand descriptively; capture and demand remain confounded.

## 7. Stage 05 — Joint frame gaze/accommodation with frozen anchors

### 7.1 Establish the radial-to-accommodation relation

First fit the shared initialization `kappa(A)=s0*(A-A_ref)` to Stage 04 capture coefficients indexed by nominal demand. Use equal capture weights and the anchored origin, retaining the slope sign, residuals, and domain support. Require a finite informative slope; do not manufacture accommodation from an uninformative radial relation.

Construct one initial accommodation target per capture as `Abar0_c=A_ref+kappa_c/s0`, then repeat it for that capture's five fixation-mean targets. Freeze these targets before joint refinement. They are P4-derived regularizers from the same recording. Report them alongside nominal labels, and include nominal capture demands as a separately labeled anchor-target control. Invalid or out-of-domain targets fail initialization; do not silently clip or substitute them.

### 7.2 Fit frame states with fixed optical calibration

Keep P1 global calibration, the four P4 K calibrations, the empirical P4 reference, and `s0` fixed. Jointly estimate frame horizontal gaze and accommodation across all captures with the same fixed Stage 02 P1 `g` applied to both P1 and P4 predictions. Vertical gaze remains the Stage 01 diagnostic in the horizontal primary model.

Use the measured native-pixel P1/P4 forward residual as the joint data objective, with equal pattern and capture/fixation weights. Recompute every predicted centroid. Save inverse triangle RMS as a corresponding geometric diagnostic. The P1 model constrains gaze through shape and scale consistency; P4 adds gaze and accommodation information through its K and radial map. For the primary joint objective, divide native data squared errors by a fixed residual scale of `(1 px)^2` before adding the dimensionless prior terms. Report raw geometry metrics in pixels separately. Any estimated measurement weighting must be trained and frozen explicitly.

Use soft gaze anchors at the frozen polynomial estimates with width `sigma_theta=0.5 degrees`. Use a soft penalty of width `sigma_A=0.25 D` on each fixation's dynamically computed mean A against its frozen target. This is not a 0.25 D per-frame hard bound or per-frame A penalty. Operational state limits initially follow the earlier joint implementation: horizontal gaze within +/-5 degrees of its initializer and A in `[0,6]` D. Record bound contacts and constrained gradients; these limits are numerical/model policies, not accuracy claims. Add no temporal smoothing or fixation recentering of A.

```text
L = weighted native P1/P4 vertex data error
    + weighted ((theta_x-theta_x_initial)/sigma_theta)^2
    + mean_fixations ((mean_fixation(A)-Abar0_fixation)/sigma_A)^2
    + declared optical-anchor penalty when optical parameters are refined
```

### 7.3 Jointly refine P4 calibration and frame states

After the fixed-optics state fit passes its audit, jointly refine the declared P4 global parameters and frame states using the same all-frame objective. P4 K remains shared within each capture, its nuisance offset remains shared across captures, and the radial slope is shared across captures. Do not add independently adjustable per-frame K or barrel coefficients: frame variation comes through gaze and `kappa(A)`.

Use frozen initialization results as soft optical anchors with declared widths. Prefer anchors on predicted K patterns over a saved gaze grid when coefficient/radius/offset conventions differ; report any raw-coefficient penalty and its coordinate basis explicitly. Freeze anchor targets and assess their strength rather than moving them after each update.

Compare this capture-specific-K model with a shared-K model; bounded capture deviations can be an intermediate controlled hypothesis. The independent-K branch supplies a conditional estimator, but cannot by itself establish accommodation-independent keystone. Report whether shared K or deviations improve held-out geometry and reduce gaze-to-A leakage. Add explicit K(A) or barrel-by-gaze dependencies only as subsequent controlled extensions, one at a time.

P1 calibration remains frozen and cannot be tuned by P4 residuals. Keep the Stage 02 `g` fixed in the first joint branch as requested. Because it was estimated conditional on initial gaze, report its P1 consistency after gaze refinement. If this becomes material, run a separately labeled branch that reprofiles bounded `g` from P1 alone at updated gaze, using frozen P1 globals; use that identical value for P4 and include its gaze derivative. P4 never receives an independent scale parameter.

Refining `s` together with A exposes the scale ambiguity `A-A_ref -> q*(A-A_ref), s -> s/q`. The anchors select a D scale; data-only agreement does not independently identify it. Inspect parameter profiles/rank separately with and without priors and report data, gaze, A-mean, and optical penalty components. Compare A-mean widths 0.125/0.25/0.5 D with otherwise matched settings; the 0.125 D anchor is stronger and the 0.5 D anchor weaker than the 0.25 D primary, and each changes the estimator.

**Deliverables:** frame states and fixed/updated scale policy, fitted optical dependencies, parameter tables, initial-versus-final fixation means and spreads, all loss components, multistart/stationarity/domain audits, bound flags, gaze-associated A contrasts, held-out results, and a clear progression decision.

## 8. Metrics, validation, and numerical gates

Always report native forward vertex RMS, inverse vertex RMS, radial/tangential components, and fixation-mean triangle residuals with their exact weighting. In addition, express residuals and input sensitivity in accommodation units. With centered P4 coordinate residual `r=X4-Xhat4`, measurement weights `W`, and derivatives evaluated under the declared scale policy:

```text
J_A = dXhat4/dA
J_theta = dXhat4/dtheta_x
J_g = dXhat4/dg
DeltaA_equiv = (J_A^T W r)/(J_A^T W J_A)
dA/dtheta_error = -(J_A^T W J_theta)/(J_A^T W J_A)
dA/dg_error = -(J_A^T W J_g)/(J_A^T W J_A)
```

Include centroid derivatives. For the P1-reprofiled branch, distinguish fixed-g sensitivity from the total gaze derivative including `dg/dtheta_x`. These expressions are local conditional model sensitivities; joint nuisance parameters and calibration uncertainty require profiling/resampling. Flag weak `J_A` support, compare local sensitivities with nonlinear perturbation fits, and report P1-scale uncertainty separately from direct P4 gaze leakage. A small pixel RMS can still imply substantial accommodation-equivalent error.

Use temporally blocked holdouts with gaps and leave-one-gaze-fixation-out checks. Train all gaze polynomials, templates, references, optical parameters, and prior targets using retained conditions only. Infer held-out P1 scales from held-out P1 measurements with frozen training calibration; exclude withheld P4 vertices from state inference when scoring prediction. For leave-capture/demand-out comparisons, distinguish interpolation from extrapolation and do not recalibrate against held-out demand labels. Use block resampling to assess fixation/reference/calibration uncertainty.

If a holdout removes Capture 1 or the nominal-zero fixation needed for an initializer, define an alternative reference using retained training conditions before fitting, preserve its coordinate/scale convention consistently, and label the run as a reference-change comparison. Do not obtain a missing reference or radial anchor from withheld observations or labels. Compare in the declared native model space and report reference changes separately from ordinary within-reference holdouts.

Retain a fully recalibrated `g=1` control under matched models, rows, splits, weights, and anchor settings. Simply substituting a scale into an older P4 inverse is a mismatch diagnostic, not the full comparison. Keep the fixed-template P1 control and expose cases where higher P1 fit flexibility worsens P4 or accommodation sensitivity.

Numerical gates require: valid equations and forward/inverse roundtrips; positive P1 scale; positive radial factor and radial derivative on the chosen branch; positive projective denominators; noncollinear templates; declared parameter/state bounds; complete invalid/bound-contact reporting; multiple-start agreement or disclosed ambiguity; and stationarity measured in declared parameter units. Do not promote an uncertified checkpoint as a completed fit or hide inverse failures by dropping rows.

Plot only reviewed fixation periods using the saved display/outlier policy. Keep all four capture legends, actual estimated gaze, nominal labels, and original source order. Break time traces at gaps. Display masking must not silently change the estimator cohort or validation denominator. Produce P1 scale/time/gaze plots, P4 recovered-triangle overlays, accommodation/gaze time traces, capture-by-gaze accommodation contrasts, radial/keystone parameter dependency plots, residual-D summaries, and bound/domain panels.

Progress when the numerical audit passes and the next model gives informative held-out geometry and accommodation sensitivity without merely tightening anchors. No arbitrary physiological accuracy threshold is claimed without independent ground truth. Remaining accommodation/gaze effects can reflect model error, genuine state variation, or fixation order; report the observed association rather than imposing a flat A trace as truth.

## 9. Implementation, result organization, and documentation handoff

Use a new family under `experiments/archive/reverse_transform/superseded_p1_scale_capturewise_p4/`:

```text
stage_01_gaze_initialization/
stage_02_shared_p1_scale/
stage_03_capturewise_p4_keystone/
stage_04_capturewise_barrel/
stage_05_joint_states/
```

Each stage contains its runner and a `results/` directory with `RESULTS.md`, machine-readable protocol/summary, numeric arrays, parameter/fixation tables, hashes/masks/frame IDs, and only the figures needed to evaluate its gate. Stage 05 keeps validation and sensitivity runs labeled within that stage. New runs use new output locations; historical results remain reproducible evidence.

Reuse the shared `distortion_model/` implementation, especially the bounded P1 scalar profile and audited centroid-consistent inverse. Extend common libraries rather than duplicating transformation equations across experiment scripts. Mechanical file/data/report tasks use GPT-6-Luna low as requested; the primary agent plans and audits the scientific model and implementation.

Vectorize/batch frame observations and independent state updates across all captures. Use CuPy float64 for large GPU workloads when available and useful, with CPU/GPU equation agreement checks. Keep small global solves on CPU when appropriate; control BLAS/process thread counts to avoid oversubscription. Record backend, precision, seed, runtime, chunking, and optimizer settings. The user requested execution outside the sandbox.

## 10. Stage 05 learned P4 reference refit (completed and audited)

The separate learned-reference sensitivity preserves the completed fixed-reference Stage 05 outputs. The physical P1/P4 magnification remains shared and fixed by the existing P1 calibration. The established Stage 05 primary remains the default behavior; new results are saved under the clearly labeled `learned_p4_reference` branch for both the production family and the training-only temporal-validation family.

Represent one shared P4 template across all captures as `B4 = c0 + T(q_template)`, where `c0` is the existing nonzero native initializer centroid and `T` is its centered component expressed in the P1 learned-template coordinate chart. Preserve `c0` and fix only the centered component energy to the initializer centered energy as the scale gauge; do not force the native template centroid to zero. Do not normalize model outputs or add any other template or output normalization. Freeze the existing native `R4` values. Bound the three chart coordinates to ±0.75 as a numerical policy, not as a physical prior. With the previous 14 global optical coordinates, the learned-reference fit therefore has 17 global coordinates.

Initialize from the existing Stage 05 optimum with zero chart coordinates. Freeze P1 calibration and its shared per-frame `g`. Jointly optimize the shared P4 template chart, capture-specific keystone coefficients, shared P4 offset and radial slope, and all frame gaze and accommodation states. Preserve the existing 0.5° frame-gaze anchors, 0.25 D fixation-mean accommodation anchors, and the existing optical prior exactly: the mean of the 14 standardized coefficient residual squares. Do not penalize the learned template chart. Keep the existing reference behavior unchanged whenever the new mode is not selected.

For both production and training-only temporal families, use at least three starts and report stationarity, start agreement, domain checks, and parameter contacts. Audit analytic derivatives, agreement with CPU finite differences and GPU evaluation, and the data-only profiled optical rank after eliminating frame states with the learned template included. For temporal validation, learn the shared template using training rows only, freeze it for the held-out rows, and run the matched omitted-vertex temporal scoring with the learned template fixed during held-vertex inference.

Compare fixed and learned references using matched P1 scale, rows, weights, anchors, and score definitions. Report forward and inverse geometry, capture/fixation accommodation means and ranges, radial slopes by capture, template-reference displacement, parameter bounds, and held-vertex scores. The production and training-only refits, implementation and numerical audits, and matched omitted-vertex temporal scoring are complete. Interpret the learned reference as a model sensitivity, not as a measured distortion-free template. Results, limitations, and the next decision are recorded in [CURRENT_STATUS.md](../../CURRENT_STATUS.md) and [experiments/RECORD.md](../../experiments/RECORD.md).

**Documentation handoff:** The completed evidence, exact conventions, validation results, limitations, and next decision are in [CURRENT_STATUS.md](../../CURRENT_STATUS.md). The previous root status is preserved in [docs/legacy/CURRENT_STATUS_PRE_P1_SCALE_CAPTUREWISE.md](CURRENT_STATUS_PRE_P1_SCALE_CAPTUREWISE.md), and experiment history is updated in [experiments/RECORD.md](../../experiments/RECORD.md). `docs/CURRENT_STATUS.md` was intentionally removed and the obsolete `CURRENT_STATU.md` synchronization instruction is retired.

## 11. Per-gaze barrel anchors and barrel-first P4 inversion (completed)

Keep the completed P1 calibration and exactly its shared frame magnification. At each of the five fixation gazes, independently compare the four captures' scale-corrected P4 triangles. Fit a relative barrel coefficient for each capture at that gaze and a latent triangle softly anchored to Capture 1's mean triangle at the same gaze. Set Capture 1's coefficient to zero separately at each gaze to declare the relative origin. Do not average different gazes or accommodation conditions into calibration observations, and do not constrain these twenty initial coefficients to be gaze independent. Record the reference-prior width, native units, domain support, stationarity, and all coefficient estimates. These initial estimates can still contain capture-dependent keystone effects; they are empirical soft anchors, not pure physical barrel measurements.

For this new branch, change the complete P4 composition consistently:

```text
Y_j = K4_c(B4_j, theta)
Z_j = D_kappa(Y_j)
Xhat4_j = g * (Z_j - mean_j(Z_j))
Bhat4_j = K4_c_inverse(D_kappa_inverse(X4_j/g + mean_j(Z_j)), theta)
```

Thus invert barrel first and keystone last, with the trial-consistent raw centroid restored before either inverse. This is a different model hypothesis from section 2; reordering correction of an old fitted model is not equivalent. Add no extra output normalization or P4 scale.

Use the per-gaze barrel estimates to initialize P4 keystone, then jointly fit frame gaze/accommodation, capture-specific P4 keystone, the shared P4 nuisance offset and radial accommodation slope, and a shared learned P4 template. Preserve P1-only scale, template-scale convention, reviewed frame roster, and original frame identities. Keep `kappa=s*(A-A_ref)` in the final model; frame barrel changes through A rather than an additional independent frame coefficient. Freeze the twenty empirical coefficient targets and softly penalize differences from each fixation's mean predicted barrel. Declare the coefficient width explicitly; initially use the old radial slope times 0.25 D as a reproducible width, distinct from the existing 0.25 D accommodation-mean prior. Report their separate loss contributions because both provide accommodation regularization.

Run the new composition with and without this new coefficient penalty, alongside the preserved learned-reference baseline. This distinguishes composition changes from added anchor strength. Use at least three starts, derivative/roundtrip/domain checks, state/global stationarity, and matched temporal omitted-vertex validation with references and coefficient targets learned from training rows only. Report the full twenty-coefficient table, gaze-associated accommodation contrasts, native forward/inverse RMS, accommodation-equivalent residuals, and parameter/bound support. A flatter A trace caused by extra regularization alone does not demonstrate better decoupling.

The implementation and audits are complete. Stage 04 fits five independent gaze-specific relative-barrel calibrations with the C1-zero gauge and a 1 px latent-reference prior; the resulting twenty coefficients are frozen as training-only targets in temporal validation. Stage 05 uses the barrel-first inverse and compares otherwise matched anchor-off and anchor-on branches against the learned-reference baseline. Anchor-on uses the declared `sigma_kappa = |s0| * 0.25 D` target width, in addition to the existing accommodation-mean prior. Three starts certify each production and training fit; CPU/GPU, derivative, inverse and source-isolation checks pass. Both temporal branches pass all six held-state certificates with zero invalid scores. Neither barrel-first candidate is promoted: anchor-off is slightly worse in production and temporal held scoring, while anchor-on is worse in both. The 1 px target prior, twenty-coefficient table, fit metrics and validation are in [the comparison report](../../experiments/archive/reverse_transform/superseded_p1_scale_capturewise_p4/stage_05_joint_states/results/controls/keystone_first_anchor_on/comparison_report/REPORT.md), [production audit](../../experiments/archive/reverse_transform/superseded_p1_scale_capturewise_p4/stage_05_joint_states/results/controls/keystone_first_anchor_on/IMPLEMENTATION_AUDIT.json), and [temporal held-out results](../../experiments/archive/reverse_transform/superseded_p1_scale_capturewise_p4/stage_05_joint_states/results/validation/temporal/keystone_first_anchor_on/RESULTS.md). Results are also recorded in [CURRENT_STATUS.md](../../CURRENT_STATUS.md) and [experiments/RECORD.md](../../experiments/RECORD.md).
