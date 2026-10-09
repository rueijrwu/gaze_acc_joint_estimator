# Implementation plan: full-calibration distortion model and cross-agreement

**Branch:** `rueijrwu/gaze_acc_joint_estimator / exp5_distortion_model`.  
**Revision:** 2026-10-09; reconciled against `38029d7eb52cb95a8e62f0f812bf513d42e520aa`.  
**Status:** Proposed implementation and execution contract. No new fit, code implementation, or test-suite result is claimed.  
**Mathematics:** [Theory.md](Theory.md). Implement the two documents together.

## 0. Current branch, priority, and completion target

The inspected Git tree contains README, documentation, and reviewed data. It does not contain `distortion_model/`, `full_position/`, `lib/`, `models/`, `tests/`, or experiment outputs. Inherited `CURRENT_STATUS.md` and older plans describe historical empirical experiments and include paths absent from this cleaned branch. They must not be treated as proof that a new distortion-model implementation or runnable command already exists.

The next deliverable is **one complete DM0 calibration and its same-frame cross-agreement report**, followed by a controlled DM1 comparison when its radial increment is identifiable. Do not delay this with another general exponent sweep, nested condition-holdout study, broad framework rewrite, or a requirement to reduce RMS below an arbitrary number.

Preserve the agreed sequence: centroid-based gaze bootstrap; P1 symmetry/distortion/scale; independent P4 symmetry; accommodation baseline; P4 keystone; model-corrected center-separation polynomial; full iterative refinement. DM0 and DM1 both implement this sequence. Reuse established bounded least-squares, variable projection, multistart and GPU batching methods where applicable; the new optical forward adapter is not the old empirical 27-column model.

A pinned donor such as `54deab5871b4713f67e2ab6ab86f1f950dc95eee` can supply selected generic algorithms after source inspection. Port only needed helpers, with provenance/licenses and fresh parity tests. Do not restore deleted experiment trees or silently depend on missing paths. Historical coefficients, trajectories and cached inverses are not fresh calibrations of the new model. Do not change `exp5_full` or `distortion_tracking` while implementing this branch.

## 1. Locked scientific contracts

| Contract | Required behavior |
|---|---|
| Sign | P4 minus P1 for initial centroid and final center separation |
| States | One visual horizontal theta and one accommodation A per frame |
| Fixation | Dynamic states; finite fixation-mean anchors; no temporal flattening |
| Targets | Visual -10, -5, 0, 5, 10 degrees; nominal vertical zero, measured y retained |
| Optical references | Separate shared omega1 and omega4; xi1=theta-omega1, xi4=theta-omega4 |
| Reference conversion | Delta14=omega4-omega1; xi1=xi4+Delta14 |
| Observation | Ten linear same-frame relative coordinates |
| Normalization | P1-reference edge scale; no triangle-area denominator or mandatory area ratio |
| Nuisance scale | One positive P1-only g, shared by P1 and P4; no free Z or P4 scale |
| Final gaze relation | Forward D(theta,A) for corrected center separation, up to cubic in visual gaze |
| Centroid relation | Derived h_cent=D+mu4-mu1, not another independent polynomial |
| P4 accommodation | Effective magnification, then identifiable incremental radial deformation |
| Calibration population | All valid frames in all twenty full reviewed conditions, captures 1-4 |
| Cross-check | Freeze calibration; rerun complete inference under each P4 omission |
| Scientific decision | Cross-reconstruction plus state agreement/identifiability, not a hard RMS gate |

Unknown optical source geometry, center location, or symmetry is not permission to invent it. Record unknown/fixed/constrained quantities. Equal nuisance scaling does not imply identical P1/P4 distortion coefficients. A common-scale mismatch is a diagnostic finding, not a silent extra per-frame parameter.

## 2. Minimal candidate contract

Use `a=(A-Aref)/(1 D)`. Begin with the following explicitly nested accommodation mechanisms:

| Candidate | Accommodation baseline | Required result |
|---|---|---|
| `DM0-M1` | `B4=M(A)*b4ref`, `M=1+m1*a` | First end-to-end full calibration and masked report |
| `DM1-M1K1` | `B4=M(A)*(1+kA*a*M(A)^2*||b4ref||^2)*b4ref` | Fresh matched calibration; report whether radial change is separable |
| `DM2-<mechanism>` | Exactly one declared curvature/coupling/spatial addition | A specific residual/identifiability reason and matched comparison |

The default baseline convention is `empirical_incremental`, not `paraxial_absolute`. Both candidates preserve the real reference distortion and satisfy `M(Aref)=1`; DM1 additionally has `Delta_kappa(Aref)=0`. DM0 still contains K1, K4, both angular zeros and the corrected center polynomial. Do not call it a distortion-free or centroid-only estimator.

Declare the gaze degree once per comparison. A degree-2 center polynomial is a reasonable first configuration for the proposed dominant-linear plus quadratic relation; degree-1 and degree-3 are separate paired ablations. All accommodation candidates in a comparison use the same declared degree, A terms, priors and alignment policy. Do not simultaneously change accommodation family and gaze degree and attribute the gain to one change.

Record the exact free/fixed parameter list and actual count, not a legacy capacity label. Initially fix the reference templates, their length convention, camera-axis alignment and local origins to the declared initialization; refine only parameters demonstrably supported by the current fit. Keep omega1 and omega4 independent even if one or both are fixed at an uncertain operational reference. Do not force them equal.

Global parameters and every frame state are re-estimated for each scientific candidate. A previous candidate may provide a separately recorded warm-start proposal, but it is not the result or an additional state prior; keep a common nominal and perturbed start for comparison and certify every new full fit.

## 3. Minimal code and artifact interfaces

Create only the package needed for the above experiment:

```text
distortion_model/
    data.py          # reviewed payloads, fixed schedule, validity, masks
    geometry.py      # linear relative map, covariance, P1 scale
    alignment.py     # symmetry evidence, dual offsets, reference conversions
    optics.py        # baselines, radial increment, keystone and derivatives
    gaze.py          # separate bootstrap inverse and forward center polynomial
    objective.py     # common full loss, full-group mean anchors, parameter blocks
    calibrate.py     # stages and certified iterative full calibration
    invert.py        # frozen-model, retained-only batched inference
    crosscheck.py    # exactly three P4 slots, state and point agreement
    io.py            # model schema, hashes, checkpoints, failure statuses
    report.py        # native-frame metrics, common-cohort and optical diagnostics
```

A separate accelerated module is optional. Do not maintain two unverified mathematical implementations merely to use both Torch and CuPy. Add permanent tests and pinned dependencies during implementation; neither currently exists in this branch.

Public contracts:

```text
local_angles(theta_visual, omega1, omega4) -> xi1, xi4
p1_reference(theta_visual, params) -> F1, mu1, edge_reference
p4_reference(theta_visual, A, params) -> F4, mu4
p1_scale(theta_visual, p1_edges, R11, params) -> g, validity, derivatives
center_polynomial(theta_visual, A, params) -> D
predict_relative(theta_visual, A, relative_observations, params, mask) -> yhat
calibrate_full(config) -> certified model or checkpoints + complete outcomes
infer_retained(model, masked_frame) -> state/branch/status + omitted prediction
```

Public optical functions accept visual theta; lower-level K functions accept xi. Subtract offsets exactly once. A single immutable parameter/reference snapshot must be used throughout an objective evaluation. Updating omega1 changes the P1 map, scale and all corresponding predictions; updating omega4 changes the P4 map but does not directly change g at fixed visual theta.

Use a new schema such as `distortion_dual_alignment_v1`, with an explicit theory/config revision. It records candidate mechanism, basis formulas/units, sign, full parameter roster, both offsets and Delta14, native-angle baselines, template/origin/length gauges, fixed/free masks, symmetry records, `gaze_polynomial_target=center_separation`, visual degree and coefficient order, `scale_policy=p1_profile_v1`, covariance policy, domains, data/source/runtime hashes and population IDs. Store the bootstrap inverse separately. Reject single-zero, area-normalized or historical empirical artifacts without an explicit migration; do not silently reinterpret coefficients.

## 4. Stage S0: data census, geometry, and forward adapter

### S0a. Census before fitting

Read trusted capture 1-4 payloads and `data/fixations/fixation_intervals.json`. Preserve source bytes, permutation and flags. Use `[start_row,end_row_exclusive)` for the entire reviewed interval. Do not restore central-80% trimming, subsampling, pupil requirements, shape rejection, or residual-based filtering silently.

Recompute actual rows and validity by capture/fixation; historical totals are not test constants. The lowest demand is approximately 0.36036 D. Aref is its declared reference convention, not a measured zero-accommodation frame. Preserve all demand labels. Preserve frame IDs and timestamps; report backward timestamp jumps before using time differences. Captures 5/6 do not influence calibration, noise, templates, symmetry or selection.

Freeze a candidate-independent agreement manifest of every scheduled row and three P4 slot IDs before evaluating any model. Invalid rows remain scheduled with reasoned unavailable outcomes. For initial full fitting, use frames with finite P1 and all three valid P4 as in the declared complete-pair policy; incomplete-frame fitting would be a separate extension. Numerical/model-domain invalidity is reported separately from detector validity.

### S0b. Domain and noise policy

Declare visual theta and A search bounds and check both shifted local-angle ranges. The inherited starting box [-20,20] degrees and [0,6] D is a computational proposal, not a physiological requirement. Do not add an unsupported near-zero omega prior. Positivity, denominator and radial-domain checks use actual source/reference geometry.

Use a frozen native-coordinate covariance policy from calibration data, preserving shared point errors when mapped by L. Contiguous temporal differences must not bridge original frame gaps; real eye motion may contribute. Record approximation and source. Do not estimate different residual covariances for each law just to shrink its normalized error. R11 is the P1-edge covariance marginal from the same policy.

### S0c. Forward adapter

Implement the equations from Theory Sections 4-8, including both native offsets, P1 scale, baseline mode, optical means, D and all derivatives. Test synthetic translation/scale/offset behavior before a large fit. A bounded rehearsal across all conditions is an integration test only; its fit is not the full-data deliverable. Run the full-fit path afterwards without silently reducing its data.

**Exit evidence:** immutable census/manifest, valid fixed conventions and covariance, a tested CPU reference forward adapter. No new optical calibration result is claimed at S0.

## 5. Stage S1: centroid-based visual-gaze bootstrap

In the reference-accommodation capture, spatially average the three points of each reflection per frame. Form `c4_x-c1_x`, then take its temporal mean in each of the five fixations. Fit an initial displacement-to-visual-gaze relation: primarily linear, with degree 2 or up to 3 only as a declared bootstrap configuration. Five summaries do not justify an arbitrary exact interpolant.

Apply the initializer to every frame's own displacement; do not replace a trajectory by five target values. Before P1 scale is known, use the provisional reference-scale assumption and label it accordingly. Transferring this initializer to other demands gives approximate starts; accommodation may alter its offset and gain.

After S2, update it using the mean of each frame's `(c4_x-c1_x)/g_i`. A ratio of separate means and a nonlinear inverse of a mean are not interchangeable with the corresponding framewise operations. The final anchor acts on actual frame-state means. This bootstrap is not the final center polynomial or independent truth.

## 6. Stage S2: P1 native zero, distortion, and reference scale

Specify the symmetry metric from known source correspondences and fixed optical/camera axes. Preserve the score and support across all five gaze conditions, initially in the reference capture and then across demand recordings. Do not assume an equilateral arrangement or fit away asymmetry with an arbitrary affine transform. When source symmetry cannot be specified or the minimum is broad, preserve reference alternatives or declare a fixed operational choice with uncertain alignment.

Initialize omega1 from the supported minimum and build the empirical P1 template in its own native reference. Many frames inform that template; each still has an individual estimated theta. Retain the template's original barrel distortion. Fit K1 using `xi1=theta-omega1` and compute one positive P1-only scale for every frame across all demands.

Fix length/origin and isotropic gaze-scale conventions before releasing global parameters. P1 is only weakly gaze-dependent; a flexible template and arbitrary gaze scaling can exchange with g. P1 accommodation independence is not a license to treat accommodation-biased bootstrap gaze as exact. Keep the finite full fixation-mean anchors during refinement.

Return P1 global parameters, fixed/free reference flags, omega1 evidence, frame scale/validity, and residuals versus visual gaze and demand. Scale uses two P1 edges, no triangle determinant; collapsed reference energy is invalid, while collinearity has separate consequences for other parameters.

## 7. Stage S3: independent P4 optical zero

After P1 scale correction, examine P4 symmetry at all five gaze conditions across the demand recordings. Initialize omega4 independently; do not inherit omega1 or visual zero and do not require the offsets to be close. Any scale-invariant symmetry display is for reference selection only, not removal of accommodation magnification from inference.

Use one shared omega4 initially. Save per-demand symmetry curves so a reproducible accommodation shift is visible rather than hidden by separate capture offsets. Record endpoint selections, ties, profiles, and uncertainty.

Require `Delta14=omega4-omega1`. P1 at P4 zero uses `xi1=Delta14` and generally is not symmetric. Preserve native templates or transform the template and homography together as in Theory Section 4.5. Visual labels remain unchanged. A spatial optical origin and an angular symmetry zero are distinct, separately constrained concepts.

## 8. Stage S4: accommodation baseline, initially DM0

Use near-P4-zero frames across demands with their P1 scales to initialize `M(A)` and the empirical P4 reference template. Start with DM0-M1: `M(A)=1+m1*a`, positive on the declared domain. Framewise A starts from demand but is then optimized; it is not fixed per fixation. Near-reference theta values are approximate and will be corrected after K4 is available.

Do not insert a measured distorted template into an absolute radial formula. Hold the baseline/reference origin convention fixed. Fit an effective-scale response first; this provides the complete DM0 model once K4 and D are learned.

Only DM1 adds `Delta_kappa(A)=kA*a`. Check its distinguishability from M and D about the declared origin, including near-equal radii. Report unresolved radial identification instead of freeing arbitrary spatial centers. A numerical kA returned by a solver is not proof of a physically identifiable barrel coefficient.

**Initial full-run target remains DM0.** A radial-degeneracy finding must not block reporting a usable effective accommodation response. Do not introduce a new accommodation exponent grid before this result exists.

## 9. Stage S5: P4 gaze deformation

With the accommodation baseline initialized, fit K4 across the five visual gaze conditions using `xi4=theta-omega4`. Preserve native `K4(0,A)=I`. Start with A-independent alpha4/beta4/gamma4 but include A-dependent B4 in the denominator and its derivatives.

Revisit every S4 frame with its own xi4. A chosen reference fixation is not permanently zero local gaze. Prefer the forward composed map; inverse-keystone about the measured centroid is not equivalent to a transformation about the optical origin. Refining omega4 changes a physical model unless the full baseline/operator is merely re-expressed consistently.

At this point K1 and K4 can predict their respective local patterns across trial theta/A in the declared domain. The same P1-derived g is still used for both; no P4-specific nuisance fit has been introduced.

## 10. Stage S6: corrected centers and final gaze polynomial

With individual states and optical means, compute

```text
C1_hat = c1 - g*mu1
C4_hat = c4 - g*mu4
D_obs  = (c4-c1)/g - mu4 + mu1
```

These are model-informed origins and a relative difference, not additional measured state variables. Fit the forward visual-angle polynomial

```text
t = theta_visual_deg / 10
a = (A_D - A_ref_D) / 1
D = (b0+bA*a) + (s0+sA*a)*t + c2*t*t + c3*t*t*t
```

Disable c2/c3 according to the declared degree. Both image axes are modeled, but there is no vertical gaze state. Higher gaze terms have no A coupling initially. At fixed states/optics, solve the D block as weighted linear least squares using the complete relative residual and all covariance cross-terms. Center estimates aid initialization and reporting; do not add a second independent Dobs loss.

The final measured centroid law is derived as `D+mu4-mu1`. Do not retain an independent h_cent polynomial, double-count the mean corrections, or treat the bootstrap inverse as the exact inverse of the forward cubic. At trial theta, g and mean offsets change too; a cubic root computed with them frozen is only a proposal.

## 11. Stage S7: iterative full calibration under one objective

Use the exact Theory Section 8 objective, full-population means, and immutable references per evaluation:

```text
initialize S1-S6 for the candidate
repeat outer calibration cycle:
    for current shared parameters, update every frame's accommodation
    update every frame's visual gaze; recompute xi1, xi4, P1 g, optical means
    refine permitted P1 coefficients and omega1 with gauges fixed
    refine P4 accommodation baseline under the current frame states
    refine P4 keystone and permitted omega4
    reconstruct center-separation proposals and solve the linear D block
    evaluate the SAME full relative objective and full fixation means
    damp/reject worsening steps; jointly refine theta/A when needed
    record objective components, global/frame changes, domain and stationarity
    at declared completed checkpoints, freeze and run cross-agreement
```

Each update includes relevant chain derivatives. Never detach g, substitute a P4-influenced joint scale, or independently reset frame/local zeros. The conditional state updates include the contribution to the full fixation-mean anchors, not a framewise target penalty. Application has no nominal-frame or temporal anchor.

Use the inherited finite 0.10 degree / 0.25 D mean scales as common initial controls, with explicit shared-parameter regularization in scaled units. No hard nominal RMS, G_A/G_theta, or motion threshold. If investigating sensitivity, change the same control for each candidate in a separately named comparison.

Run a common nominal and reproducible perturbed start, preserving competing offset/state minima. Certify finite domain, full scaled projected stationarity and appropriate local curvature. Budget exhaustion is an uncertified checkpoint, not an accommodation-law failure. Small updates at saturated bounds alone are insufficient. Cross-check scores need not improve monotonically with J; report numerical convergence and cross-score behavior separately.

A new candidate receives a fresh optimization of all global coefficients and frame states. Saving a fit is not the same as completing its cross-check, and neither proves physiological accuracy.

## 12. Compute strategy: one mathematical model on CPU/GPU

Use float64 throughout the first controlled comparison. Inspect actual installed packages, devices, drivers, free memory and backend capabilities; do not infer them from earlier sessions. A supported differentiable Torch implementation and a small NumPy/SciPy reference are a suitable initial route. CuPy may implement verified kernels, but two full GPU backends are not a prerequisite to a first complete fit.

Reuse established constrained least-squares/multistart methods. Batch per-frame two-state derivatives and systems, accumulate low-dimensional shared-parameter blocks, or use matrix-free products. Do not allocate dense `(10N) x (2N+P)` Jacobians or `(2N)^2` Hessians. Profile only linear blocks. Differentiating the profile objective and differentiating its residual Jacobian are different operations; preserve the needed profile terms.

Chunking is not subsampling. All optical weights use original Nk and K. Full fixation sums/counts must be computed for mean-anchor gradients; for either scalar state the anchor derivative is `(group_mean-target)/(K*Nk*s^2)`. Recompute full means for a joint trial. The final loss, gradient and certification must agree with an unchunked small reference, including group coupling.

Use one process/device initially; parallelize independent candidates or check batches only after measuring memory. A sharded calibration shares global parameters and group reductions, not independent coefficients per device. Keep arrays on device where possible; no per-point host loops or hidden mixed-precision switch. Benchmark with warm-up and synchronized timings, separating setup, transfers, fitting, inversion and reporting. Record measured timing, not a promise based on an older 48-row screen.

## 13. Stage S8: whole-loop internal P4 cross-agreement

For every scheduled frame, freeze the candidate's global parameters and solve three retained-P4 subsets. Each solve uses all P1 and two P4, fresh derived g at its trial theta, and no nominal labels. Use the same declared multistart policy across models; a 7x7 theta/A grid plus retained-only proposals is a reproducible starting contract, not a proof of complete branch enumeration.

Do not initialize from an all-three centroid/state or an unmasked warm start. The selected covariance marginal precedes whitening. A retained-P4 mean predicts `D+mean_I(F4)-mu1`, not the all-three centroid. An all-three mean of model-predicted points is safe; a mean of the omitted observation is not.

Score each omitted point in native relative pixels:

```text
error_j = (q_j-c1) - g_subset*(D + F4_j - mu1)
```

Every slot must contain held_point=0/1/2, capture, fixation, row, source frame, raw-input validity, calibration status, inference status and a reason. A calibration failure produces a complete roster of `not_run_uncertified_model` slots. Missing fields must fail validation rather than crash the whole comparison later or disappear from denominators.

Keep certified, rank-sufficient, unambiguous and bound statuses distinct. Compute triple E/G metrics only on three eligible slots; partial-slot summaries remain separately labeled. States are compared in visual theta and diopters. Do not compare different local xi values as physical disagreement.

The optional P1-omission extension is not on the critical path: rebuild retained P1 centroid, scale edge, prediction reference, covariance and rank checks. Do not pool its outcomes with P4 omissions. No whole-pair omission is implied.

These are internal checks because calibration used the same observations. A noninterference test fixes calibration and perturbs only the omitted point; the subset solve and prediction must remain unchanged. This does not claim that recalibrating modified data would preserve the coefficients.

## 14. Scientific comparison and decision record

A result must answer which accommodation mechanism gives useful shared-state information, not merely list the lowest training cost. Keep three population views:

1. Candidate-specific all-scheduled scorecards with all coverage/failure counts.
2. Candidate-versus-DM0 comparisons on exact common frame and point identities.
3. A single common complete-frame cohort across the certified candidates for transparent side-by-side ordering.

Every expected exposure and its represented count stays visible. An exposure missing from a paired/common cohort makes the all-exposure comparison incomplete; do not silently reweight over survivors or report a winner. Report cohort fractions without copying an old nested-selector accuracy policy. Integrity and comparability are not hard physical RMS requirements.

Display E, G_theta and G_A separately, plus signed axes, individual points, worst point, pooled/exposure tail conventions, bounds, branches and conditional rank. Primary optical agreement is omitted-P4 reconstruction; state agreement is essential companion evidence. No arbitrary mixed-unit scalar and no rule that every candidate must improve every metric. A tiny gain concentrated in one fixation is described as such, not as a general physical-law discovery.

Show the following interpretation views:

- P1/P4 symmetry scores, chosen omega values, Delta14, native-angle support and offset alternatives.
- Raw centroid displacement, mu4-mu1 correction, model-informed center separation and polynomial contributions.
- M(A), incremental radial response and their derivatives at native optical zero; complete P1/P4 predictions at common visual theta values.
- Radial/scale rank, gaze-A cross-talk, frame scale distribution, compression/clipping and common anchor/prior/reference sensitivity.

For a mixed result, write `tradeoff`. Other outcomes include `supported_effective_response`, `supported_radial_increment`, `alignment_or_identifiability_unresolved`, `comparison_incomplete`, and `numerically_unavailable`. Numerical eligibility is not physiological accuracy, and small G_A is not evidence of correctness when the A axis is compressed.

Only add DM2 after stating the specific repeatable model deficiency and which single extra mechanism addresses it. Do not restore broad exponent searches merely because DM0 does not fit perfectly. Failure to isolate an absolute barrel coefficient does not invalidate a useful effective accommodation signal.

## 15. Permanent tests and acceptance evidence

The following tests are required during implementation; this document does not claim they have run:

| Group | Contract |
|---|---|
| Relative measurements | L has rank 10; arbitrary common translations leave inputs and predictions unchanged |
| Dual zeros | Unequal/nonzero omega values; visual zero and P4 optical zero evaluated correctly |
| Reference changes | Homography plus baseline conversion; cubic binomial conversion preserves predictions |
| P1 scale | Positive GLS scale, edge-origin covariance equivalence, P1-only dependence at fixed theta |
| Derivatives | g, both offsets, mean corrections, radial/projective composition and state/global derivatives |
| Center law | Dobs sign; h_cent derived once; no second independent center measurement |
| Baseline | Identity at Aref, no repeated reference distortion, equal/near-equal radius degeneracy |
| Scale separation | Common scale cancels as intended; P4-only scaling remains signal/mismatch |
| Dynamic states | Nonconstant trajectories allowed; zero-mean variation preserves mean anchors |
| Objective | Linear D solve lowers the same weighted cost; analytic/full and chunked gradients agree |
| Covariance | Marginal before whitening; plug-in residual rank and correlations not treated as independent data |
| Masks | Entire P4 initialization/state/g/center pipeline ignores held input; optional P1 mask rebuilt |
| Population | Missing/duplicate frame or slot rejected; failures keep all expected identities/exposures |
| Model comparison | Identical-model paired differences zero; different cohorts not silently ranked |
| Backend | CPU and chosen GPU predictions, derivatives, objective and accepted steps agree |

Use genuine numerical tolerances, documented before comparing response models. Do not tune them to help one law. Saved summaries must be independently reproducible from frame/slot records; a few algebra checks do not certify the complete solver. Report actual passed/failed/skipped counts, real-data scope and backend.

## 16. Artifacts, checkpoints, and safe recovery

Create a new output directory per declared full experiment. Preserve config, source/data/runtime hashes, parameter/reference roster, immutable population/slot manifests, symmetry/offset evidence, bootstrap and center-polynomial artifacts, global/state checkpoints, accepted or failed calibrations, frame states, g, optical means, crosscheck records and exact common-cohort reports.

Use compact arrays for long frame records and small JSON/Markdown summaries. Retain enough row-level data to reconstruct every published aggregate; do not delete sole supporting records during cleanup. Full multistart debug archives can be optional, but failure/branch metadata and source provenance must remain. Explicitly distinguish `fit_complete`, `fit_certified`, `crosscheck_complete`, and `comparison_complete`.

Resume only a source-, data-, population-, convention- and parameter-compatible checkpoint. A changed template, optical zero convention, noise policy or dataset is not a harmless resume. Preserve the previous run; migrate explicitly or start a new one. Do not silently restore old empirical fitted states as this estimator's result.

Proposed commands, to implement rather than claim currently available:

```bash
python -m distortion_model calibrate \
  --config configs/dual_alignment_dm0.yaml \
  --output experiments/distortion_model/dual_alignment_dm0_v1

python -m distortion_model crosscheck \
  --model experiments/distortion_model/dual_alignment_dm0_v1/model.json \
  --manifest experiments/distortion_model/dual_alignment_dm0_v1/population.json \
  --output experiments/distortion_model/dual_alignment_dm0_v1/crosscheck
```

Follow with a separately configured fresh DM1 fit and a common-manifest comparison only after the DM0 path works. Suggested configuration names describe planned interfaces, not existing files.

For source/document writes, read the current remote branch and both authoritative files, record the parent SHA, then update them together in one non-forced commit. If the remote moves, reread and reconcile; do not overwrite concurrent work using stale chat attachments. Do not write to `exp5_full`. Update execution status in a subsequent implementation/run change when evidence exists; inherited historical numbers must not be relabeled as new results.

## 17. Exit criteria and scope

The first successful deliverable is a reproducible DM0 full fit using all declared valid calibration rows, its complete masked P4 report, and an honest assessment of optical zeros, state identifiability and cross-agreement. The next deliverable is the same for DM1 plus an exact matched comparison or an explicit radial-identifiability limitation. DM2 is conditional, not compulsory.

No condition-held-out calibration, nested selection, optional P1 omission, second GPU backend, perfect optics, zero residual or physiological reference is a prerequisite to reporting this internal experiment. Conversely, internal agreement must not be described as independently established physiological accuracy.

This plan and Theory.md supersede older single-zero, area-normalized, freely scaled P4, and pixel-only model-selection instructions for this branch. They do not modify data, execute code or claim completion. Reuse algorithms carefully; implement the new observation/optical adapter, not another unconstrained empirical response search.

## References

- [Authoritative equations and optical sources](Theory.md).
- [Reviewed fixation metadata](../data/fixations/fixation_intervals.json).
- [Prior branch plan](https://github.com/rueijrwu/gaze_acc_joint_estimator/blob/38029d7eb52cb95a8e62f0f812bf513d42e520aa/docs/ESTIMATOR_PLAN.md): retained design intent.
- [Pinned generic-method donor](https://github.com/rueijrwu/gaze_acc_joint_estimator/tree/54deab5871b4713f67e2ab6ab86f1f950dc95eee): inspect and port selectively; not the new optical model.
