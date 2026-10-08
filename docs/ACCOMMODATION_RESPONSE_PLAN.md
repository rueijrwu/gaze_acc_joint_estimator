# Accommodation-response experiment plan: isolated log-versus-power comparison

**Status:** PROPOSED; documentation only. Implementation and real-data runs have not started under this plan.  
**Repository / branch:** `rueijrwu/gaze_acc_joint_estimator` / `exp5_full`  
**Revision snapshot inspected:** `335660aa61b2b31785e40833a073d58fce8b1851`  
**Original design commit:** `5ee31ac4fa8ac35f63927e558a085acc0b38bfea`  
**Date:** 2026-10-07  
**Theory:** [ACCOMMODATION_RESPONSE_THEORY.md](ACCOMMODATION_RESPONSE_THEORY.md)  
**User clarification:** gaze and accommodation can change during a calibration fixation. RMS is not a hard accuracy or acceptance requirement.

## 0. Execution boundary: do not interrupt the ongoing audit implementation

This accommodation-law workstream defines a separate response-family experiment. Candidate fitting and evaluation use fixed schedules, disjoint calibration and evaluation observations, common cohorts, and explicit coverage. The implementation and saved results are summarized in [CURRENT_STATUS.md](CURRENT_STATUS.md).

This revision changes only this plan. The proposed workstream keeps local stage names **AR0-AR4**, not new claims that an existing numbered phase is complete. Within this workstream, the explicit metric policy below supersedes older instructions to require absolute RMS/state-disagreement ceilings before comparing or selecting candidates for evaluation. It does not alter the companion theory equations or the active selector implementation.

When implementation is requested later, pin an integrated commit and verify both the population/selection-contract repairs and the new non-threshold metric policy. Use an isolated branch or worktree and new run directories. The source snapshot above is not a guarantee that the selector already supports this policy; Section 2.3 identifies the remaining integration requirement.

Existing `Theory.md`, `ESTIMATOR_PLAN.md`, audit files, frozen baseline, saved detections and all prior results stay authoritative for their own scope and unchanged by this documentation addition. Captures 5/6 remain untouched by this proposed comparison.

### 0.1 What the latest results change in this plan

The latest direct baseline27 differential-y-minus-xy comparison reports paired changes in E-squared / G-theta-squared / G-A-squared of -0.641 / +0.0649 / +0.0403 for gaze and -3.258 / -0.0013 / -0.0109 for capture, in their respective squared units. Better excluded-point prediction can coexist with worse same-frame subset-state agreement. No one of these quantities is a universal pass/fail target. [R1,R2]

The shared-y covariance trial used a training-residual scale of 1.267-1.712 pixels as a weighting sensitivity, not an accuracy tolerance or validated noise variance. Neither that scale nor the observed E/G values should become hard thresholds for the accommodation-law study. The current reference remains baseline27 with xy. No power candidate has been evaluated in the inspected results. [R1,R2]

Therefore keep measurements, weighting and coefficient count fixed for the initial log-versus-power comparison; compare prediction quality and tradeoffs instead of requiring a particular RMS number. Do not force a new law to reproduce the old model's latent trajectory or make fixation trajectories flatter.

## 1. Scientific question and locked first comparison

> Does changing accommodation curvature, without changing the measured geometry or global coefficient count, improve the three-way excluded-P4 cross-check?

The inspected model uses a logarithm in its non-$D_x$ accommodation responses and gaze/accommodation interactions. The experiment replaces those terms with

$$
\phi_\lambda(A)=\begin{cases}
((1+A/A_*)^\lambda-1)/\lambda,&\lambda\ne0,\\
\log(1+A/A_*),&\lambda=0,
\end{cases}\qquad A_*=1\,\mathrm D.
$$

| Configuration | First-pass requirement |
|---|---|
| Response candidates | `ar27_log`, `ar27_shifted_sqrt`, `ar27_linear`, `ar27_shifted_quadratic` |
| Fixed exponents | 0, 0.5, 1, 2, respectively |
| Coefficients | 27 for each fixed exponent; two states per frame |
| Exponent policy | One common exponent for every response component, fixed during fitting/inversion of a candidate |
| State | Horizontal gaze and accommodation, in degrees and diopters |
| Calibration gaze | Horizontal -10, -5, 0, 5, 10 degrees; nominal vertical gaze zero |
| Measurements | All three P1 references; full retained x/y for every P4 exclusion |
| Normalizer | Square root of P1 triangle area; area ratio unchanged as a diagnostic |
| Basis | `Dx: [1,a,t,t*phi,t^2,t^2*phi,t^3]`; other five functions: `[1,t,phi,t*phi]` |
| Basis/state scales | `t=theta_deg/10`; `a=A_diopters/1`; optimizer `[theta_deg/10,A_diopters/4]` |
| Bounds | Gaze [-20,20] degrees; A [0,6] D; not claims about calibrated support |
| Calibration | Joint global coefficients and one free `(theta_x,A)` state per valid frame; no constant-fixation state; temporal regularization zero |
| Initial soft mean-anchor scales | 0.10 degree and 0.25 D as finite penalty weights, shared across candidates; not allowed movement, RMS limits, or measured uncertainties |
| Initial coefficient prior | Same declared column-normalized policy, strength 0.001; extra-curvature penalty zero |
| Pilot/noise | One training-only log27 pilot/reference covariance shared within each split |
| Reference | Fresh same-split log27 under the same implementation and settings |
| Main criterion | Paired inner-group excluded-P4 prediction loss; mandatory same-frame state-agreement and coverage/axis/tail/scope reporting; no absolute accuracy gate |

The numerical scales are proposed controls inherited from the existing comparison, not physical uncertainty measurements. A change to any locked factor creates a separately named ablation, not an undocumented improvement to one exponent.

Do not remove y information while comparing response laws. Do not fit a continuous exponent, change accommodation demand labels, or reuse a log model's coefficients under a different exponent.

### 1.1 Fixation is a calibration condition, not a constant physiological state

For each frame i in fixation k, estimate its own state

$$
x_i=(\theta_{x,i},A_i),\qquad x_i\text{ need not equal }x_{i'}\text{ for }i,i'\in k.
$$

Nominal horizontal target and accommodation demand identify the stimulus condition. They are not instantaneous measured gaze/accommodation. Even the actual fixation-mean accommodation may differ from demand. Nominal vertical gaze remains zero as a protocol label; do not set measured image-y to zero or claim instantaneous vertical eye motion was measured to be zero.

Do not replace a fixation with one state, assign every frame its target/demand as truth, penalize its within-fixation state variance, reject frames for target deviation, or add smoothing solely to reduce reported error. Fit the framewise optical responses first; use finite mean-anchor penalties only to establish the calibration convention.

For training groups only, the inherited anchor term is

$$
J_{\rm anchor}=\frac{1}{2K}\sum_k\left[
\frac{(\bar\theta_{x,k}-\theta^{nom}_{x,k})^2}{s_\theta^2}
+\frac{(\bar A_k-A^{demand}_k)^2}{s_A^2}\right],
\quad
\bar x_k=\frac{1}{n_k}\sum_{i\in k}x_i.
$$

Here the initial s-theta=0.10 degree and s-A=0.25 D are soft penalty scales. They are neither framewise tolerances nor requirements that the means fall within those distances. Larger mean discrepancies are allowed at a finite cost. These scales must not be increased in strength merely because a trajectory is not flat or its nominal-target RMS is large. A later anchor-weight sensitivity must treat every exponent identically and select by inner cross-checks, not nominal-label proximity. Do not remove all scale-identifying information without a separate identifiability analysis.

For any scalar state z, a constant nominal value u, and equal frame weights,

$$
\frac1{n_k}\sum_{i\in k}(z_i-u)^2
=(\bar z_k-u)^2+\frac1{n_k}\sum_{i\in k}(z_i-\bar z_k)^2.
$$

Thus framewise RMS against a fixed nominal target includes actual within-fixation variation as well as mean offset; it is not an estimator-accuracy requirement. Mean discrepancy, temporal spread and optical cross-check error must be reported separately. This identity does not prove that every estimated fluctuation is physiological signal; interpretation still requires optical consistency and, for accuracy, an independent reference.

### 1.2 What must agree: measurements of the same frame

At frame i, the three P4 exclusions infer x-hat(i,-1), x-hat(i,-2), and x-hat(i,-3). Cross-check asks whether those simultaneous subsets explain the same instantaneous state and predict the point each omitted. It does not require that state to match an earlier/later frame or the nominal label.

A state trajectory may move throughout a fixation while all three subsets agree at each frame. Conversely, three flat or boundary-clipped trajectories may agree while predicting the excluded points badly. Temporal variation is not automatically error, but it is also not an excuse for unexplained same-frame subset disagreement. Retain both prediction and agreement metrics without requiring either to be zero.

### 1.3 Metric policy: relative evidence, not hard RMS requirements

| Quantity or condition | Role in this study |
|---|---|
| Excluded-P4 cross-prediction squared loss and its RMS summary | Primary paired comparison and inner ranking; no fixed pixel target |
| Same-frame subset gaze/accommodation disagreement | Required companion evidence; no universal degree/diopter ceiling |
| Signed axes, worst-point error, median and tail summaries | Required tradeoff reporting; no automatically imported absolute thresholds |
| Nominal-target/demand RMS, mean offsets and within-fixation spread | Descriptive calibration references only; never acceptance or selection gates |
| State outside empirical training extrema or at an optimizer bound | Separate scope/bound diagnostics; not automatic proof of a wrong state |
| Identity/correspondence, finite geometry, valid covariance, numerical certification, leakage and manifest integrity | Hard implementation/data requirements; not physiological accuracy specifications |
| Adequate scheduled and exact shared exposure coverage | Predeclared comparability requirement, independent of nominal error and candidate loss |

No absolute E/G/axis/worst/tail accuracy thresholds are required by this plan. An application-specific accuracy limit may be added only as a separately documented, explicitly authorized requirement; do not derive one from the current results. Passing a numerical tolerance or coverage check is not proof of physiological accuracy.

## 2. AR0 — Prerequisites and immutable experimental manifest

### 2.1 Consume, do not duplicate, the ongoing audit repairs

Before a selection-bearing run, verify the integrated implementation passes these contracts from the latest audit:

1. The scheduled population is independent of returned predictions. Exact frame identities and exactly three unique held-P4 slots are checked; missing outputs remain explicit unavailable records or raise a contract failure.
2. Exact shared-cohort ranking preserves the original expected exposures. A vanished exposure or insufficient shared support prevents promotion; it does not improve coverage by shrinking denominators.
3. Inner grouping matches the intended held-out gaze-condition or capture/demand question. Evaluation labels, records and all-three states cannot enter fitting or subset branch selection.
4. Invalid, ambiguous, weak-rank, bound, out-of-scope and failed-calibration cases are retained distinctly. Protocol-zero vertical targets are not asserted as framewise ground truth.

The latest status reports population repairs, but the execution checkpoint must still pass their regressions. Mathematical tests may proceed in isolation; a selection-bearing run may not bypass manifest, leakage or numerical-integrity checks. The new non-threshold metric-policy integration in Section 2.3 is a separate prerequisite.

### 2.2 Freeze a run manifest before evaluating candidates

Record source commit, exact design copies, implementation hashes, detector/interval hashes, correspondence, software versions, seeds, all candidate specifications and planned task IDs. Read accommodation demands from stored metadata. Record actual source-grid geometry only when available; do not infer it from gaze targets.

Freeze frame sampling from original rows before inspecting candidate outputs. For a first historical comparability screen, the existing 48 training / eight evaluation rows per fixation and central-80% rule may be reused on captures 1-4, but label this a sampled development screen. A denser confirmation population must be predeclared in a new manifest, used identically for every exponent, and never chosen from low-error survivors.

Keep gaze-condition and capture/demand split families separate. Adjacent frames and overlapping folds are not independent replicate experiments. Captures and accommodation demands are confounded in the existing recordings; more rows do not create more independent demands.

Predeclare the coverage/comparability policy: expected exposures, complete/scored and shared-frame fractions, per-exposure representation, grouping, missing-data handling and uncertainty summaries. Predeclare how state-agreement, axis, tail, bounds and scope tradeoffs will be reported and reviewed. Do not require maximum RMS, maximum G, nominal-error, temporal-variance, or absolute tail limits to run the comparison. An explicit absence of accuracy thresholds is a valid policy, not missing configuration. Missing population/grouping rules can prevent defensible selection; missing arbitrary accuracy cutoffs cannot.

### 2.3 Explicit integration with the current selector

At the revision snapshot, `full_position/selection.py` still requires numerical maxima named `maximum_G_theta_deg`, `maximum_G_A_D`, `maximum_x_axis_rms_px`, `maximum_y_axis_rms_px`, `maximum_worst_point_px`, and tail limits before `choose()` can select. Its population repairs do not remove these accuracy gates. Do not silently use that mandatory-threshold mode for this study. [R3]

When implementation is authorized, add a versioned policy to the existing selector, for example `paired_crosscheck_no_absolute_accuracy_gates_v1`. Keep the legacy policy for historical reproduction. Reuse its manifest, leakage and shared-exposure validation, but separate:

- hard numerical/data and scheduled/shared-population checks;
- primary paired cross-prediction ranking;
- mandatory descriptive companion outcomes and tradeoff flags;
- optional deployment-specific acceptance limits, disabled by default for this study.

The policy must run without absolute E/G/axis/tail ceilings. Do not simulate disabled gates by passing enormous values or infinity. It must distinguish `selected_for_outer_evaluation` from `promoted_for_deployment`; the former can be determined by a predeclared inner comparison without the latter being justified. Regression tests must show both that no-accuracy-threshold comparison works and that missing frames/exposures still cannot disappear.

This is a future implementation requirement, not a source-code change made by this plan revision. Missing this policy means the software needs integration, not that scientific RMS thresholds should be invented.

**AR0 exit:** a versioned manifest, explicit non-threshold metric policy and passing population/leakage contracts; otherwise record the specific implementation or comparability gap.

## 3. AR1 — Basis/model implementation and numerical acceptance

### 3.1 Add a separate response type, not a global log replacement

A possible implementation layout is a new response-basis module, explicit power-response model adapter, and experiment runner using existing generic calibration/inversion interfaces. Names are proposals, not existing functions. Reuse the completed audit infrastructure after checking its contracts; avoid a second scorecard or selector.

The adapter must expose `design`, `predict`, `components`, physical `state_hessian`, coefficient count/order, intercept indices and immutable response metadata. Values, derivatives and Hessians must agree for every exponent. The theory supplies the formulas.

For $\lambda=0$, compare against existing conditional27 at identical coefficients, contexts and states, including A=0/6, theta=0 and bounds. The target is equality within declared floating-point tolerances. Keep the frozen scale-15 two-channel baseline in its own adapter; it is not the new reference model.

### 3.2 Metadata and compatibility are mandatory

Use a distinct artifact family/schema, for example `conditional_power_response_v1`, with at least:

```json
{
  "model_family": "conditional_power_response_v1",
  "capacity": 27,
  "accommodation_basis": "shifted_boxcox",
  "accommodation_exponent": 0.5,
  "accommodation_unit": "diopter",
  "accommodation_scale_D": 1.0,
  "direct_linear_A_term": true,
  "theta_scale_deg": 10.0,
  "optimizer_state_scale": [10.0, 4.0],
  "retained_channels": "xy",
  "nominal_vertical_gaze_deg": 0.0,
  "vertical_gaze_interpretation": "calibration protocol, not framewise truth"
}
```

Also store exact coefficient order, normalization, pilot family/exponent, covariance convention, bounds, finite mean-anchor weights, framewise-state policy, metric-policy version, coverage rules, manifest IDs and failed alternatives. Record `absolute_accuracy_thresholds: null` and `nominal_error_is_acceptance_gate: false` for this study. Reject missing or incompatible power metadata. Never silently interpret a legacy model as a power model or assign a missing exponent from a current default.

Legacy acceleration and polynomial-profile code may hard-code log terms. Dispatch only to verified implementations of the requested response or raise an explicit unsupported-family error. Initial real-data comparisons use the shared scalar reference; no unverified GPU path may supply a candidate-specific result.

### 3.3 Required tests

The acceptance suite must cover:

- Four exact basis cases, logarithmic limit, finite values/derivatives/Hessians at A=0, and invalid-domain rejection without implicit clamping.
- Physical and optimizer-scaled derivatives, mixed derivatives, model Hessians, profiled Jacobian products and transpose products, all with nonzero residuals and coefficient priors.
- Twenty-seven coefficients for all four candidates; lambda=0 parity with the legacy27 response and compatible fixed-covariance inverse results.
- General shifted-power state-scale identities from the theory, and the limit that state disagreement can change without different point predictions.
- The lambda=1 conditional accommodation solution against numerical bounded minimization, including an unidentifiable denominator and both accommodation boundaries.
- Fixed-A gaze-polynomial consistency, candidate retention/polishing and multiple-branch diagnostics under every response law.
- Raw-input noninterference for all excluded points and axes: arbitrarily perturbing the excluded coordinate changes only later scoring, not retained preprocessing, covariance, starts, branches or predictions.
- Schema round trips, rejection of family/exponent/scale/order mismatches, unchanged historical model behavior, and all manifest/shared-exposure regression cases.
- Nonconstant synthetic theta/A trajectories within each fixation: use framewise generating states, keep mean anchors soft, and verify that valid time variation is not filtered, averaged away, clamped to labels or penalized solely for making target RMS large.
- Anchor-unit tests: a within-fixation zero-mean state perturbation leaves the mean-anchor term unchanged, though the optical term may change; larger mean offsets change a finite penalty, not a hard validity flag.
- Same-frame agreement tests: identical moving subset trajectories have G=0 despite nonzero temporal spread/nominal RMS; an injected mismatch in one simultaneous subset changes G and its held-point predictions. Separate clipping and state-scale counterexamples remain required.
- Selector tests: comparison without absolute accuracy limits; after training and split/schedule manifests are frozen, changing reporting-only evaluation nominal labels cannot affect states, branch choice or primary cross-check ranking; changing them may update reporting-only label/scope diagnostics. Integrity and frozen shared-exposure checks remain active. No arbitrary RMS threshold is manufactured from a result.

Synthetic shared-state recovery should be tested under each generating exponent with fixed calibration conventions. Do not require the selection machinery to identify the generating exponent when several candidates have indistinguishable predictive geometry or latent-scale ambiguity.

**AR1 exit:** validated basis/model and generic solver compatibility; not a claim about real-data benefit.

## 4. AR2 — Controlled joint calibration of each fixed exponent

For every actual training split:

1. Load only its training groups and validate the scheduled training rows.
2. Fit or construct one logarithmic pilot and estimate coordinate covariance from that training subset alone. Freeze its reference-state policy and per-frame residual weights for all exponents in this split.
3. For each exponent, build the 27-column design and recompute its training-only nominal-state coefficient initializer, column scales and prior using the same policy. Nominal states are initializers only. Jointly optimize coefficients and distinct framewise states with the same finite mean-anchor penalties and numerical bounds; never replace them by one state per fixation.
4. Preserve every start, certified/uncertified outcome, coefficient/state checkpoint, objective decomposition and declared continuation history.

An existing pilot or starting trajectory may be reused only when its training provenance exactly matches the current split and it contains no validation observations. In particular, an outer-fold pilot or latent trajectory that saw an inner validation group is prohibited inside that inner fit.

Use common nominal and seeded perturbed-nominal state starts across candidates. An additional same-split log27 training-state warm start can be used for every candidate if declared beforehand; it is an initializer, not a state penalty. Coefficients must be solved for the candidate's basis, not copied as if equivalent.

Keep covariance fixed during each frame inverse and identical within the exponent comparison. Retain shared-P1 propagation. The recently completed shared-y covariance trial remains a separate weighting sensitivity; do not introduce its covariance modification at the same time as the law comparison. Its training-residual RMS is neither a nominal-state tolerance nor a bound on admissible eye motion.

The existing column-normalized prior offers a reproducible comparison policy, not identical function-space regularization across exponents. Report prior and optical contributions and training-only response/state shifts. A small predeclared prior-strength sensitivity may be a later inner-loop factor, applied to every exponent; do not adjust only a favored candidate after seeing outer errors.

A nine-fold screen of four exponents would have 36 planned fit outcomes before any extra starts or nested inner fits. It is a development screen, not a complete nested study. Failed fits contribute no manufactured predictions and retain every scheduled denominator.

**AR2 exit:** complete calibrated or explicitly failed candidate records, with framewise trajectories and separate optical/mean-anchor/prior objective components. Numerical termination is not an RMS accuracy gate. No exponent is selected using training cost, fixation flatness, nominal RMS or outer measurements.

## 5. AR3 — Cross-check evaluation and exponent selection

### 5.1 The same three-way test for every candidate

For each held-out evaluation frame and each excluded P4 point $j$:

- Use all P1 points and only the other two P4 x/y coordinates.
- Form the retained covariance marginal before whitening; do not slice a full whitening operator that mixes excluded data.
- Infer only horizontal gaze and accommodation under that fixed candidate, using the common bounded multistart and polishing policy.
- Freeze the state and all retained plausible branches. Predict both coordinates of P4 point $j$.
- Read the excluded point only in the scorer, then store vector/axis errors and the subset state. Repeat for all three exclusions.

A full P4 centroid, area, observed affine map, all-three state, excluded-point confidence gate, excluded error, or validation label must not influence the subset solve. All-three estimates may be computed for normal application and separate diagnostics, never as hidden holdout inputs.

The lambda=1 analytic conditional solution is an independent numerical check. Initially do not give only lambda=1 a fundamentally different acceptance/branch policy that confounds response-law and solver changes. Any corrected solver policy must be evaluated consistently across candidates.

### 5.2 One scorecard, with scientific names

Use the companion theory's primary loss $L_{cross}$ and display `E_cross_px = sqrt(L_cross)`. RMS describes aggregation, not a hard acceptance target. Specifically, for frame i,

$$
e_{i,j}=q_{i,j}-\hat q_{i,j}(\hat x_{i,-j}),\qquad
E_i^2=\frac13\sum_{j=1}^3\|e_{i,j}\|^2,
\qquad
L_{cross}=\frac1K\sum_k\frac1{|I_k|}\sum_{i\in I_k}E_i^2,
$$

where $I_k$ is the exact shared complete-frame set in exposure k; all expected exposures and unavailable outcomes remain accounted for. Use normalized errors alongside pixels, not as a different hidden population.

For one candidate c versus reference r, compare $\Delta L=L_{cross,c}-L_{cross,r}$ on those same identities. Negative means lower cross-prediction squared loss, not a passed absolute accuracy requirement. Show distributions, exposure-level paired changes and an appropriate grouped uncertainty/stability assessment, rather than only the aggregate. Correlated frames, paired subsets and overlapping folds are not independent replications.

Compute G-theta and G-A from the three simultaneous subset states at each frame before any temporal aggregation. Do not compare those states to the nominal target when computing G and do not aggregate states across a fixation before cross-checking them.

Every scorecard must include `G_theta_cross_deg`, `G_A_cross_D`, signed per-point/axis residuals, worst-point and tail metrics, raw and normalized point errors, condition/rank, branch ambiguity, bounds/clipping, per-subset nominal/empirical support and P1 context/parity. Separately show nominal mean offsets, framewise nominal RMS, within-fixation temporal spread and retained-data optimizer cost as descriptive diagnostics, never as true-error estimates or acceptance/selection gates. A large recorded cross-check residual must remain visible rather than being discarded to improve RMS.

Report scheduled/eligible/scored slots, complete triples, failed calibrations, expected/contributing exposures and exact memberships. Coverage denominators come from the frozen manifest, not from whatever the evaluator returns. For equal-exposure loss, missing exposure contributions mean an incomplete comparison, not silent omission or a zero loss.

Compare every power candidate directly with the same-split log reference on exact common frame IDs. Preserve full-population reporting, jointly interior and jointly supported cohorts separately. Do not infer a paired result by subtracting separately filtered RMS tables. Unsupported-by-extrema and intended endpoint extrapolation are distinct from numerical invalidity; do not force states into extrema to improve percentages.

### 5.3 Nested selection and honest development evidence

For a selection-bearing study, construct explicit inner and outer grouped split manifests. For held-out gaze-condition performance, inner validation should reflect that question across captures. For capture/demand transfer, use entire capture groups when remaining calibration support is adequate. Record infeasible nested splits; do not secretly replace them with random frame splits.

Within an outer training pool, fit each candidate only on inner training groups. Rank numerically valid candidates satisfying the predeclared population/comparability policy by paired inner cross-check loss, with no absolute accuracy ceilings. The minimum-loss candidate may be selected for the sealed outer evaluation under this predeclared rule, while companion outcomes carry explicit tradeoff flags. Use a fixed numerical-tie rule favoring the log reference; do not prefer flatter trajectories or smaller nominal RMS. Refit the selected exponent on all outer training groups, then perform one sealed outer evaluation. A separately predeclared log-reference outer score is a comparator, not another opportunity to change the selected exponent.

Do not choose a global winner by examining all outer candidate results and then call its minimum loss independent validation. Historical folds already used for these design decisions remain development evidence. Nested processing improves the procedure but cannot erase that history. New independent conditions or reserved transfer data address a different evidence question. [E2]

No absolute accuracy thresholds are needed for inner ranking or an outer evaluation. If numerical/data contracts fail or shared exposure coverage is insufficient, report no valid comparative selection with the actual reason. If paired predictive improvement is uncertain or accompanied by worse state agreement, tails, bounds or transfer behavior, report `inconclusive` or `predictive_gain_with_tradeoff`, not an unconditional improvement. Retain the reference operationally until the declared review/confirmation procedure supports a replacement. This is not an assertion that the reference meets a physiological accuracy requirement. Deployment promotion is distinct from running the scientific comparison.

## 6. AR4 — Interpretation, confirmation and later extensions

The first decision concerns the accommodation law under fixed x/y, not whether y should be retained. A useful exponent should improve excluded-point prediction across relevant groups without merely compressing the accommodation scale, flattening actual within-fixation dynamics or causing hidden numerical failures. The latest gaze/differential-y tradeoff shows why a lower aggregate E alone is not a complete account of model behavior. Report all companion changes; do not demand that every frame or every summary improve.

Report exponent stability across inner splits, calibration gains/residual trajectory differences using training rows only, and uncertainty or ambiguity in the choice. Four nominal demand levels and reused frames provide limited independent support for curvature; a sharply named exponent is not a physiological measurement.

After the law comparison is frozen, a later controlled experiment may compare the selected exponent and log reference under x/y versus x plus differential-y. Keep that a separate factor and do not attribute combined changes solely to the power law. Common-y discrepancy diagnosis from the ongoing audit remains relevant; a new accommodation basis is not presumed to solve it.

Only if residuals and cross-checks justify it, consider a 37-coefficient response, selective extra-curvature penalties, or a small independently weighted $a,a^2$ basis. Literal $a^p$, per-component exponents, fitted shift/scale parameters, nonlinear exponent optimization, direct masked-prediction training, vertical-gaze states and detector changes are out of scope for this initial plan.

Before transfer, freeze the response, fitting rules, non-threshold comparison policy, numerical/population checks, uncertainty/review rules and any application-mask policy. Captures 5/6 remain untouched until then. Their unlabeled geometric agreement does not supply independent physiological accommodation accuracy.

## 7. Artifacts, reproducibility and concurrency

Use a new directory such as `experiments/full_position/accommodation_response_v1/`; this path is proposed and does not assert an existing run. Keep at least:

```text
config.json                       # pinned inputs, candidates, metric policy, runtime
population_manifest.json          # scheduled rows and three slot IDs per frame
split_manifest.json               # explicit inner/outer groups and exposure IDs
design_snapshot/                  # these documents and inherited contracts
implementation_snapshot/          # or immutable source commit plus verified hashes
fits/<split>/<candidate>/         # model or failed checkpoint, states, all starts
crosscheck/<split>/<candidate>/   # predictions, scores, branches, support, masks
selection.json                    # inner ranking, tradeoffs, selection vs promotion
RESULTS.md                        # cross-check-first tables and limitations
verification.json                 # tests, hashes, cohorts, numerical checks
```

Exact layout may reuse the integrated runner's conventions; avoid duplicating infrastructure merely to match a new directory sketch. A run manifest must contain the resolved settings, not only references to mutable defaults. Resume is allowed only when source/data/candidate/metric-policy/coverage/manifest hashes match. Never overwrite or append new-law scores into historical phase directories.

Parallelize independent fits with a declared worker limit and one BLAS/OMP thread per worker, subject to the environment's measured behavior. Use verified GPU execution only after value/derivative/branch/certificate parity for the power family. Record realized runtime and failures; do not claim the historical GPU speedup applies to this new experiment.

Before implementation commits, re-read the current branch. Preserve concurrent work and use new paths or coordinated additive patches. Do not force-push or reset to the documentation snapshot. This proposal adds no required source changes to the ongoing audit task.

## 8. Completion and promotion checklist

**Documentation complete** means the response family, isolated factors, derivatives, cross-check definition and staged plan are specified. That is the status of this addition.

**Implementation complete** additionally requires AR0/AR1 contracts, compatible metadata, verified profiling/inversion and unchanged legacy behavior.

**Experiment complete** requires every scheduled fit outcome, prediction slot, expected exposure, and selected/failed candidate to be accounted for; independent numerical/aggregation checks and historical preservation hashes must be recorded.

**Scientific improvement supported** means evidence of paired cross-check gains under the declared calibration and coverage policy, appropriate grouped evaluation, and clear reporting of state-agreement, axis/tail, support and state-scale tradeoffs. It does not require achieving a universal RMS, making nominal error small or making fixation trajectories constant. No automatic deployment promotion follows from a lower training cost, smaller temporal variance or a new exponent. Absolute physiological accuracy requires a separate reference and claim.

## Source navigation

- [Companion accommodation-response theory](ACCOMMODATION_RESPONSE_THEORY.md): equations and interpretation.
- [Existing estimator plan](ESTIMATOR_PLAN.md) and [response model](../full_position/model.py): inherited geometry, basis scale and interfaces at the inspected snapshot.
- [Calibration implementation](../full_position/calibrate.py): profiled coefficient and prior conventions.

- **R1:** [Current status at the revision snapshot](https://github.com/rueijrwu/gaze_acc_joint_estimator/blob/335660aa61b2b31785e40833a073d58fce8b1851/CURRENT_STATUS.md): saved paired comparisons, implemented repairs and shared-y trial.
- **R2:** [Latest frozen follow-up results at the revision snapshot](https://github.com/rueijrwu/gaze_acc_joint_estimator/blob/335660aa61b2b31785e40833a073d58fce8b1851/experiments/full_position/latest_results_followup_v1/RESULTS.md): development evidence, not power-law results.
- **R3:** [Selector at the revision snapshot](https://github.com/rueijrwu/gaze_acc_joint_estimator/blob/335660aa61b2b31785e40833a073d58fce8b1851/full_position/selection.py): existing mandatory accuracy guards requiring the separate policy in Section 2.3.
- **User protocol/metric clarification (2026-10-07):** gaze/accommodation may change during fixation; RMS is not a hard requirement. This is an explicit project instruction, not an inference of framewise ground truth from target labels.
- [Current status](CURRENT_STATUS.md): execution history, not evidence that this new workstream has run.
- **E1:** [SciPy `boxcox1p`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.boxcox1p.html).
- **E2:** [scikit-learn nested cross-validation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).

This revision changes the plan only. No selector/source change, estimator implementation, calibration, transfer evaluation or new performance result is delivered here. Verify the new metric-policy implementation before executing a selection-bearing run.
