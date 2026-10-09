# Implementation plan: dual optical alignment and staged distortion calibration

**Branch:** `exp5_distortion_model`.  
**Date:** 2026-10-09.  
**Status:** Proposed implementation/execution contract, not a completed run.  
**Inspected base:** `2fbd742f7cb2ca26eb84ea0a6029dd9de1e10b60`.  
**Authoritative mathematics:** [Theory.md](Theory.md). The two documents must be implemented together.

## 1. Deliverable and branch boundary

Build an estimator that first uses **mean P4 minus mean P1** to bootstrap visual gaze, learns P1's distortion and common magnification about P1's own symmetry zero, learns P4 accommodation response about P4's separately determined symmetry zero, then learns P4 keystone. Reconstruct model-informed centers and repeatedly refine the center-separation gaze polynomial and both distortion models.

The initial centroid polynomial is not the final center polynomial. The selected P4 optical zero is not necessarily the nominal zero-degree fixation. The P1 and P4 optical zeros are not required to match. These are mathematical changes to the model and calibration order, not cosmetic variable renames.

At the inspected starting line this is a data/documentation branch. Historical status documents and paths may refer to code/results removed during cleanup. Inspect the live tree before implementation. Do not claim those old experiments ran here, restore obsolete output trees, or modify `exp5_full` while implementing this branch. These two specifications supersede inherited single-offset, area-normalized and held-condition selection instructions.

Reuse reviewed bounded least-squares, multistart, variable-projection and GPU batching ideas. A pinned donor such as `54deab5871b4713f67e2ab6ab86f1f950dc95eee` may supply generic helpers, but its old 27-column response, area normalization and monkey-patch GPU profile are not the new model. Record provenance/licenses for any actual code port. Do not reuse historical fitted coefficients or latent trajectories as completed calibrations of a new response.

## 2. Locked conventions

| Item | Required implementation |
|---|---|
| Sign | P4 minus P1 for centroid and optical-center separation |
| Canonical state | One `(theta_visual_deg,A_D)` per frame |
| Visual labels | -10,-5,0,5,10 degrees; never shifted when optical references change |
| Optical alignments | Two shared parameters `omega1_deg`, `omega4_deg` |
| Local angles | `xi1=theta-omega1`; `xi4=theta-omega4` |
| Conversion | `delta14=omega4-omega1`; `xi1=xi4+delta14`; output `theta=xi4+omega4` |
| Optical baselines | P1 at its own symmetry zero; P4 at its own symmetry zero and declared Aref |
| Scale | One positive P1-edge-derived `g`; identical scalar used in both reflections |
| Active observations | Ten linear same-frame relative coordinates, no triangle denominator |
| Final gaze response | Center separation D(theta,A), mostly linear with up-to-cubic gaze terms |
| Mean response | Derived `h_cent=D+mean(F4)-mean(F1)`, not another free law |
| Accommodation | Effective M(A), then identifiable radial increment, then justified extensions |
| Calibration scope | All valid frames in all twenty full reviewed conditions, captures 1-4 |
| Constraints | Finite mean anchors and numerical domains, not hard physical RMS or flatness |
| Primary evaluation | Same-frame rotating P4 omission; optional separately defined P1 omission |
| Backend | PyTorch/CuPy available according to user; verify local versions/devices |

No explicit Z state, separate P4 nuisance scale, differential-scale ratio, per-frame optical offset or free per-frame distortion center. P1 and P4 have different distortion coefficients; only their nuisance magnification is shared.

## 3. Minimal implementation layout and schema

Keep a small `distortion_model/` package:

```text
data.py         # trusted payloads, intervals, immutable schedule and masks
geometry.py     # relative map, covariance, P1 edge scale
alignment.py    # independent symmetry scores, offsets, reference conversions
optics.py       # radial baselines, explicit keystone, native-angle maps
gaze.py         # initial inverse and final forward center polynomial
objective.py    # one raw-relative cost and full fixation-mean terms
calibrate.py    # staged initialization and outer block refinement
invert.py       # frozen-calibration, batched two-state inference
crosscheck.py   # independent masked inputs, scores and populations
io.py           # schemas, hashes, checkpoints and failure records
report.py       # center/mean/offset interpretation and model comparisons
```

Add tests and pinned runtime dependencies during implementation. A dedicated accelerated module is optional, not a second mathematical implementation with different conventions.

Use a new model schema, for example `distortion_dual_alignment_v1`, including:

- `sign_convention=P4_minus_P1`; `state_variables=[theta_visual_deg,A_D]`.
- `omega1_deg`, `omega4_deg`, derived `delta14_deg`, fixed/free flags, initialization fixation IDs and symmetry-score records.
- `gaze_polynomial_target=center_separation`; `gaze_basis=theta_visual_deg/10`; degree, coefficient order, A basis and priors. Store the bootstrap inverse separately.
- `baseline_mode=empirical_incremental` or `paraxial_absolute`; native angular reference of each template, spatial local origins, reference length, image axes and correspondence.
- `scale_policy=p1_profile_v1`; covariance policy; offset and state search domains; code/data/runtime hashes; all response-law functions and their units.

Reject old single-`theta_opt` artifacts unless a migration explicitly assigns and justifies both offsets. Do not silently set them equal. Separate `centroid_bootstrap` and `center_separation` coefficient artifacts; they are not interchangeable even at the same polynomial order.

API contracts:

```text
local_angles(theta_visual, omega1, omega4) -> xi1, xi4
p1_reference(theta_visual, params) -> F1, mean_F1, reference_edges
p4_reference(theta_visual, A, params) -> F4, mean_F4
p1_scale(theta_visual, p1_edges, cov11, params) -> g, validity
center_polynomial(theta_visual, A, params) -> D
predict_relative(theta_visual, A, observations, params, mask) -> yhat
fit_full(config) -> model/checkpoints + complete outcomes
infer_retained(model, masked_frame) -> states, branch/status, predictions
```

Public prediction APIs accept **visual theta** and subtract native offsets internally exactly once. Low-level keystone functions accept local xi. Names/tests must prevent double-subtracting offsets. Use a common immutable reference snapshot per objective evaluation.

## 4. Stage S0 — Input census, domains and forward-model tests

Read fixation metadata and trusted capture 1-4 pickles without changing source files. Validate flags, shapes, stored correspondence, original frame/row IDs and source hashes. Preserve timestamps but detect backward jumps before using elapsed time or temporal differences. Use `[start_row,end_row_exclusive)` full intervals. Do not silently restore central-80% trimming or a fixed row sample.

Recompute all counts; historical totals are not current acceptance constants. Declare the entire agreement schedule and three P4 slots per row before predictions. Preserve invalid rows as unavailable outcomes. Do not reject records because of target discrepancy, missing pupil alone, unusual triangle shape or large fitted residual. Any extra validity rule has its own reason and reported coverage. Captures 5/6 remain reserved and must not influence references, noise or parameter selection.

Record actual demand labels. The lowest retained label is approximately 0.36036 D. Name it Aref rather than asserting physiological zero; an optional relative accommodation coordinate must be explicit. All state anchors remain in the recorded visual/demand convention.

First numerical exploration can use theta in [-20,20] degrees and A in [0,6] D, subject to the actual optical domain. These are not physiological limits or validated coverage. Search optical offsets over the declared calibration/support range without an arbitrary near-zero constraint. Evaluate xi1/xi4 domains after shifting; check keystone denominators at the actual state/reference combinations.

Implement and test one forward adapter before large fitting runs: separate native offsets, common P1 scale, baseline modes, center polynomial, relative prediction and derivatives. Mathematical tests precede costly calibration, but do not turn this task into a replacement optimizer project.

## 5. Stage S1 — Fixation-mean centroid gaze bootstrap

Follow the requested order even though scale and distortion are initially imperfect:

1. In the reference-accommodation capture, take spatial means over the three P1 and P4 points in each valid frame.
2. Form `delta_cent_x=c4_x-c1_x` for each frame; then take its temporal mean in each of the five fixations.
3. Fit a displacement-to-visual-gaze initializer of degree 1, allowing degree 2 and up to 3 with stronger shrinkage of curvature. Five fixation summaries do not justify an arbitrary high-order exact interpolant.
4. Apply that initializer to each frame's own displacement. Do not replace the frame by its fixation label.
5. Use provisional predictions for other accommodation recordings only as starting states; accommodation may bias their displacement response.

Before P1 scale exists, use the explicitly provisional reference-scale assumption. After S2, refit using the mean of each frame's `(c4_x-c1_x)/g_i`. In general this is not equal to `(mean(c4_x)-mean(c1_x))/mean(g)`. With a nonlinear inverse, `P(mean(delta))` is not the same as `mean(P(delta))`; final anchors constrain actual mean frame states.

Save the bootstrap basis, sign, coefficient order and domain. It is an inverse initialization artifact, not the final polynomial and not independent truth. No demand equality or constant gaze is imposed on every frame.

## 6. Stage S2 — P1 symmetry zero, gaze distortion and scale across all demands

Find P1's own most symmetric gaze using the reference capture and provisional frame gazes, then pool/compare all accommodation conditions. Specify the symmetry metric from the real illuminator arrangement and stable point correspondence. A known mirrored source pairing and fixed camera/optical axis can define a useful balance score; do not invent an equilateral template or allow per-frame affine alignment that erases the effect being measured.

Select a fixation-level minimum using many frames, record competing minima/ties, and initialize `omega1_deg` at its visual gaze. A discrete selection is an initialization for the shared continuous offset, not a demand that each frame in that fixation has xi1=0. Where the minimum is broad or outside sampled support, record weak identification rather than silently choosing visual zero.

Build the P1 real template in its native symmetry reference. Preserve its baseline distortion. Fit the P1 gaze-dependent keystone using `xi1=theta-omega1` and derive one scale per frame from its edges. Extend the same P1 map across all accommodation conditions, with no direct A term. Keep soft mean gaze anchors during refinement so a weak P1 response does not force a biased bootstrap to be exact.

Fix a reference-length and isotropic gaze-scale convention. Without independent axial geometry, P1 gaze scaling and framewise magnification can compensate; call the fitted scale effective reference scale where appropriate. Do not allow that gauge to drift separately from P4 and D.

Output shared P1 parameters, offset evidence, framewise P1-derived scale/status and residual patterns versus gaze/demand. The scalar scale does not require an area determinant. A collinear but noncollapsed configuration may support scale while remaining inadequate for other optical parameters.

## 7. Stage S3 — Independent P4 symmetry zero

Use the P1-derived scale to compare P4 pattern symmetry at **all five gaze conditions**, across accommodation recordings. Do not automatically choose visual zero, inherit omega1 or force the offsets to be close. P4's accommodation-dependent scale must remain in inference; any scale-invariant symmetry display is only a reference-selection diagnostic.

Choose P4's most symmetric supported condition as its optical reference and initialize `omega4_deg` independently. Use one shared offset across accommodation in the first model. Save symmetry curves per demand so reproducible A-dependent changes are visible; they are not silently absorbed by separate capture zeros.

Compute `delta14=omega4-omega1`. Whenever evaluating P1 in P4-local coordinates, require `xi1=xi4+delta14`. At P4 zero the P1 reference is generally deformed. Keep the P1 native template, or re-reference its template and homography together according to Theory Section 5.5. Never relabel the P1 native template as symmetric at P4 zero.

The selected angular reference does not measure a spatial distortion center. Initialize local spatial origins from separate reference patterns and retain the declared origin convention. Large angular offsets and small/unknown spatial-center offsets are separate parameters and need separate bounds/identifiability checks.

## 8. Stage S4 — P4 accommodation baseline at its symmetry reference

Use frames near theta=omega4 across accommodation conditions to initialize M(A) and the identifiable barrel increment. Apply each frame's P1 nuisance scale first. The reference fixation is only approximately at xi4=0; preserve its framewise gaze estimates for later correction.

Default route:

```text
baseline_mode = empirical_incremental
M(A_ref) = 1
Delta_kappa(A_ref) = 0
M(A) = 1 + m1*a
Delta_kappa(A) = kA*a   # DM1 only, if identifiable
```

Here a=(A-Aref)/(1 D). A real P4 template already includes its reference distortion; it is not a paraxial grid. Absolute radial coefficients require matched additional geometry, not a relabeled incremental artifact.

Check the scale/radial response rank about the declared center after allowing centroid displacement. Exactly equal source radii produce a scale-radial degeneracy. If unresolved, run DM0 effective scale and label the radial component unidentifiable. Do not release arbitrary centers or additional per-frame coefficients to force DM1 to fit.

Use soft accommodation means across all chosen reference-angle data; A_i is free per frame. Initializing at demand is allowed, fixing every A_i to demand is not. Optical parameter estimates may initially be approximate because the final keystone is not yet fitted.

## 9. Stage S5 — P4 keystone after accommodation response

Fit P4 gaze deformation across the five gaze conditions using the accommodation baseline and `xi4=theta-omega4`. Preserve K4(0,A)=identity in its own symmetry reference. Start with A-independent keystone coefficients, but retain the A dependence already created by the radial baseline inside the denominator.

Revisit S4 data with each frame's estimated xi4: subtracting a fixation-mean angle or setting a whole fixation to zero is not the final model. Refine the shared omega4 only with the corresponding template/operator convention and complete objective.

Do not blindly inverse-keystone a pattern about its measured centroid as if that centroid were the optical distortion center. Prefer the forward map with explicit native baseline, offset, local origin and model mean. A homography re-reference must move the baseline as well. If measured symmetry does not support even/odd coefficient constraints, document that limitation and test an explicit extension rather than pretending an arbitrary sampled reference is exact physical symmetry.

At the end of this stage, both P1 and P4 transformations are available for every candidate visual theta/A within the declared model domain. No separate P4 axial scale has been fitted.

## 10. Stage S6 — Full frame inference and distortion-corrected gaze centers

Estimate `(theta_i,A_i)` for every valid calibration frame using the current complete model, recomputing g from P1 at each trial theta. Define model means mu1 and mu4 from their separate-offset transformations. Calculate

```text
C1_hat = c1 - g*mu1
C4_hat = c4 - g*mu4
D_obs  = (c4-c1)/g - mu4 + mu1
```

These are model-informed local origins; do not expose them as independently measured absolute eye positions or optimize them freely per frame.

Refit the forward polynomial to **center separation**, using visual t=theta/10:

```text
D(theta,A) = (b0+bA*a) + (s0+sA*a)*t + c2*t*t + c3*t*t*t
```

Use a dominant linear term, a small quadratic departure and optional cubic. Both image axes may be modeled; do not add a vertical-gaze state. Keep curvature independent of A initially. Fit vector polynomial coefficients at fixed states/optics by weighted least squares against the complete relative residual; preserve covariance cross-terms.

The resulting observed centroid law is **derived** as `D+mu4-mu1`. Do not retain a second free centroid polynomial in the final likelihood. The provisional center estimates are aids for block updates and reports, not independent observations to add to the original coordinate residuals.

A cheap gaze proposal subtracts current distortion-mean offsets from `(c4_x-c1_x)/g`, then uses D's linear inverse. Refine with the full cubic-capable model. Since g and mean offsets also depend on trial theta, solving a cubic once with them frozen is only a proposal, not a complete inverse.

## 11. Stage S7 — Repeat the full calibration cycle under one objective

The stages above bootstrap the physical separation. Once available, alternate the complete sequence rather than freezing early approximations:

```text
initialize raw-centroid gaze bootstrap at A_ref
fit P1 symmetry offset, native distortion and P1-only scales
select independent P4 symmetry offset
initialize P4 accommodation baseline, then P4 keystone
initialize center-separation polynomial
repeat:
    update every frame's A at current visual theta
    update every frame's visual theta; recompute both local angles and P1 scale
    refine shared P1 distortion and permitted omega1 parameters
    refine P4 accommodation baseline using current gaze
    refine P4 keystone and permitted omega4 parameters using current A
    reconstruct model-informed centers; solve the D coefficient block
    reevaluate the SAME full-coordinate cost and full fixation means
    damp/reject nondecreasing proposals or use joint state refinement
    record parameter/offset/center changes and numerical certification
    at declared completed outer checkpoints, freeze parameters and cross-check
```

Maintain fixed gauges and one immutable parameter snapshot per trial. If offsets refine, do not shift visual target labels or independently reset frame states. Pure reference re-expression and an actual physical model update are different operations; preserve predictions during the former and evaluate J for the latter.

Use Theory Section 10 exactly: equal-exposure raw relative-pixel weighting, finite **full fixation-mean** anchors, shared parameter regularization and no temporal penalty. Initial scales 0.10 degree/0.25 D are mean weights, not individual tolerances. Parameter regularization acts on declared dimensionless scales; optional curvature shrinks toward simpler models. Do not penalize offset deviation from zero without supporting alignment evidence.

`p1_profile_v1` derives scale only from P1. P4 or A cannot change it directly at fixed visual theta and P1 parameters. Include derivatives through g, mu1, mu4, both offsets, baselines and keystone. Do not insert a `g_joint` profile or a detached scale as a silent optimization shortcut. The fixed weighting metric is not automatically the covariance of the plug-in residual; do not attach calibrated chi-square interpretations.

Use at least a nominal initialized and reproducible perturbed-state/global start. Record competing offset/branch solutions. Numerical certification requires finite domain/objective, scaled projected stationarity of the full state/global problem, and appropriate local curvature checks. Budget exhaustion is an uncertified checkpoint, not proof of model failure. Small updates after saturated constraints alone do not prove convergence.

Cross-agreement need not improve monotonically with training cost. Require numerical stabilization and report cross-score stability separately, including disagreements. Do not force zero cross-error, flatter states or a hard RMS target by adding arbitrary penalties.

## 12. GPU optimization with PyTorch/CuPy

Use PyTorch float64 for the first differentiable composed model, with a NumPy/SciPy float64 reference on small deterministic cases. `torch.func` supports batched differentiation; use vmap with jacfwd/jacrev and matrix-free JVP/VJP as appropriate [T1]. Verify analytic P1-scale and keystone derivatives against autodiff/finite differences. If an operation lacks forward-mode coverage, select a tested alternative rather than silently dropping its derivative.

CuPy float64 can implement validated array kernels and batched linear algebra; it is not necessary to maintain both GPU backends before the first full fit. Inspect installed Torch/CuPy/CUDA versions, GPUs, driver and memory. Do not assume hardware capacity or runtime from earlier sessions. Record precision and disable mixed-precision changes unless separately tested.

The only independent framewise states are theta/A. Both angular offsets are global; local xi values and g are derived. Batch local 10x2 Jacobians and small damped state systems. Do not allocate a dense `(10N) x (2N+P)` Jacobian or a `(2N)^2` Hessian. Offset/global derivatives are low-dimensional shared columns; accumulate their systems across frames or use matrix-free products.

Chunking is an execution strategy, not row subsampling. Compute full group counts and sums before mean-anchor gradients; a component's derivative from the stated anchor is `(mean-target)/(K*N_k*s^2)`. Chunk optical reductions with the original exposure weights, not chunk-local counts. Joint trial states require newly evaluated full means. Small per-frame approximate Gauss-Newton proposals can be accepted against the exact coupled objective; do not omit group coupling from final certification.

At fixed states and optical parameters, solve the center-polynomial block with QR/SVD or suitably checked small linear algebra. If differentiating a profiled residual, include its profile derivative; an objective envelope gradient is not a residual Jacobian. Offset/global updates and scale derivatives must remain in the autodiff graph.

Keep arrays on device where feasible. Avoid per-point Python loops, item calls and host copies. Use one process/device initially; distribute independent model candidates or inference batches when multiple GPUs are available. A sharded single calibration needs common global reductions and mean anchors, not independent models per shard.

Batch P4 masks, frames and the declared multistart schedule. A 7x7 theta/A grid is a reproducible initial inverse policy; add only retained-derived proposals and retain distinct competitive branches. The same numerical schedule applies to competing scientific models.

Benchmark with warm-up and synchronized CUDA events or `cupyx.profiler.benchmark`, distinguishing transfers, setup, fitting, checking and reporting [C1]. Numerical parity tests compare predictions, offset/scale/state derivatives, objective components, accepted steps and branches. Do not tune parity tolerances to favor a response law. A CPU reference and supported bounded least-squares control remain useful independent checks [S1].

## 13. Stage S8 — Internal P4 cross-agreement and optional P1 checks

Freeze all global parameters, both offsets, references and noise policy at a checkpoint. For each scheduled frame omit P4_j, retain all P1 and the other two P4, and rerun the entire estimator without calibration labels or temporal priors.

The omitted coordinate must not enter the full measured centroid, reference selection, P1 scale, initial state, residual gates, covariance, branch choice or prediction. A mean over predicted model points is permitted. The retained P4 mean has model `D+mean_I(F4)-mu1`, not the full centroid response. Its corrected center uses `mean_I(q)-g*mean_I(F4)`.

Score the omitted P4 in **original relative camera pixels** as `(q_j-c1)-g*(D+F4_j-mu1)`. Report full-fit residuals separately. Store one outcome for every frame/slot including `held_point=0/1/2`, capture, fixation, row, source frame, validity and failure reason. A failed calibration must generate not-run records rather than crash aggregation or disappear from denominators.

An optional P1-omission study uses the two retained P1 points for the reference mean and one surviving edge for scale. Rebuild the observation map and covariance before whitening, and replace mu1 with the predicted retained-P1 mean in every P4 reference equation. Predict the omitted P1 from that retained reference. Check rank and branches; do not mix its counts or losses with P4 omissions as though all outcomes had identical information. No whole-pair omission is implied.

Report E, G_theta and G_A for complete eligible P4 triples, with partial-slot summaries and denominators separately. States are reported in **visual gaze** before comparing subsets. Compute equal-exposure squared aggregates, not pooled RMS differences from mismatched populations. Keep axes, point-specific errors, tails, bounds, rank, ambiguity and common-frame membership.

Calibration used these data, so these are internal checks. A raw-input noninterference test freezes calibration then changes only the omitted coordinate. It does not claim that retraining after changing the data would leave the model unchanged.

## 14. Scientific comparisons: accommodation response, not a coefficient contest

Start with DM0 effective magnification and DM1 magnification plus identifiable radial increment. Both use separate native optical offsets, common P1 scale and the same center-polynomial family/degree. Allow DM2 only for a reproducible mechanism not explained by the simpler maps. Each candidate is freshly calibrated on all full intervals; no whole-gaze/capture holdout or nested selector is required for this task.

Hold angular-reference policy, bounds, covariance, anchors and polynomial regularization fixed across accommodation candidates. Evaluate degree-1/2/3 gaze ablations separately. Do not alter the polynomial degree and accommodation family together and attribute improvement to one mechanism.

Required result views:

- Independent P1/P4 symmetry score versus visual gaze, chosen offsets, their uncertainty/profile, local-angle support and separation omega4-omega1.
- Original centroid displacement, distortion-induced mean correction, inferred center separation and polynomial contributions in the same visual convention.
- Optical response curves and derivatives at each reflection's native zero, plus predictions evaluated at common **visual** gazes for comparison.
- Individual all-population scorecards and exact paired/common-cohort cross-agreement, including every exposure and unavailable outcome.
- Scale/radial identifiability, gaze/accommodation cross-talk, bounds, branch ambiguity and sensitivity to reference-center and anchor conventions.

The centered-pattern optical zeros may differ substantially without implying inconsistent gaze. Compare two maps at the same theta by converting both xi values. Do not report an apparent disagreement merely because local angles differ by the expected offset.

A small G_A can result from state compression or clipping. A lower coordinate loss can coexist with weaker physical accommodation identification. Report tradeoffs rather than use an arbitrary mixed-unit scalar. No deployment or physiological-accuracy claim follows from internal agreement alone.

## 15. Required tests

| Contract | Required check |
|---|---|
| Sign | P4-P1 for bootstrap, corrected centers, D, prediction and reporting |
| Distinct zeros | Nonzero unequal omega1/omega4; visual zero leaves two nonzero local inputs |
| Conversion | xi1=xi4+omega4-omega1 and theta recovery, including a large-offset case |
| Native identities | Each K is identity only at its own native xi=0 |
| P1 at P4 zero | Generally nonidentity; never use unshifted P1 argument |
| Homography reference | H(xi_b)H(xi_a)^-1 plus transformed baseline preserves prediction |
| Polynomial basis | Binomial coefficient conversion preserves a cubic under visual/local angle shift |
| Mean versus center | h_cent=D+mu4-mu1; no second independently fitted h |
| Center sign | D_obs=(c4-c1)/g-mu4+mu1 recovers synthetic C4-C1 |
| Relative geometry | Ten-coordinate rank and arbitrary common-translation invariance |
| Common scale | P1 GLS recovers scale and the same value multiplies P4 |
| Dependency | At fixed theta/P1, A or P4 changes do not directly change g |
| Derivatives | Global offsets, local conversion, g, means, baselines and cubic state/global derivatives |
| Noncommutation | Scale outside keystone; centering is not a substitute for optical origin |
| Baseline | Increment identity at Aref; no repeated reference barrel correction |
| Identifiability | Equal/near-equal radius degeneracy, template-scale gauge and weak-offset examples |
| State freedom | Nonconstant trajectories allowed; zero-mean motion preserves mean anchors |
| Group reduction | Exact whole-data loss/gradient agrees with compute-chunk versions |
| P4 omission | Entire initialization/state/center/scale pipeline ignores omitted measurement |
| P1 omission | Rebuilt retained reference/scale ignores omitted P1, when this mode is enabled |
| Covariance | Correct marginal before whitening; plug-in constraints do not invent independent errors |
| Failed fits | Every scheduled slot retains identity and reason even when a model cannot calibrate |
| Backend | CPU/PyTorch/any CuPy path agree within declared numerical tolerances |

These tests validate implementation contracts, not optical truth. Report actual pass/fail/skipped counts and the backend used. A synthetic exercise is not a real-data run or a complete repository test suite.

## 16. Artifacts, recovery and proposed commands

Use a new directory for every declared full experiment. Save `config.json`, source/data/runtime hashes, full population/slot manifests, reference templates and native-zero definitions, symmetry scores, bootstrap coefficients, global fixed/free parameters, numerical starts/checkpoints, accepted/failed models, full frame states, g, optical mean offsets, center-separation diagnostics, raw cross-check records and paired reports.

Every artifact records `distortion_dual_alignment_v1`, its sign and polynomial target, both offsets, the native baseline angle, covariance/scale policy and visual output convention. Checkpoints include the stage, offset/template snapshot and source hashes. Resume only verified compatible states; changing references or input scope requires an explicit migration or a new run. Never overwrite historical data/results or silently treat an incomplete run as certified.

Suggested interface, to be implemented rather than claimed available now:

```bash
python -m distortion_model calibrate \
  --config configs/dual_alignment_dm1.yaml \
  --output experiments/distortion_model/dual_alignment_dm1_v1

python -m distortion_model crosscheck \
  --model experiments/distortion_model/dual_alignment_dm1_v1/model.json \
  --manifest experiments/distortion_model/dual_alignment_dm1_v1/population.json \
  --output experiments/distortion_model/dual_alignment_dm1_v1/crosscheck
```

Report model usability as supported effective response, supported distortion increment, tradeoff, alignment/identifiability unresolved, or numerically unavailable. A complete deliverable is an auditable all-condition calibration plus relative point/state cross-agreement, including unresolved limitations. It does not require zero residual, flat fixation or an independently validated absolute optical center.

## References

- [Mathematics and optical sources](Theory.md).
- [G1: Fixation metadata](../data/fixations/fixation_intervals.json).
- [T1: PyTorch function transforms](https://docs.pytorch.org/docs/stable/func.html), including jacfwd/jacrev and vmap.
- [C1: CuPy performance and GPU timing](https://docs.cupy.dev/en/stable/user_guide/performance.html).
- [S1: SciPy bounded nonlinear least squares](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html).
