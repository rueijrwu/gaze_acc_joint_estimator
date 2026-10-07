# Accommodation-response experiment plan: isolated log-versus-power comparison

**Status:** PROPOSED; documentation only. Implementation and real-data runs have not started under this plan.  
**Repository / branch:** `rueijrwu/gaze_acc_joint_estimator` / `exp5_full`  
**Source snapshot inspected:** `b1a37f92727304da5726b54351545d32421b56bf`  
**Date:** 2026-10-07  
**Theory:** [ACCOMMODATION_RESPONSE_THEORY.md](ACCOMMODATION_RESPONSE_THEORY.md)

## 0. Execution boundary: do not interrupt the ongoing audit implementation

The user states that implementation of [LATEST_RESULTS_AUDIT.md](LATEST_RESULTS_AUDIT.md) is still underway. This separate plan does not supersede, renumber or mark that work complete. Do not edit its active source, experiments, manifests, status documents, or outputs to implement this proposal implicitly.

The present deliverable is only this plan and its companion theory. The proposed workstream uses local stage names **AR0-AR4**, not new claims that an existing numbered phase is complete.

When implementation is requested later, start from a pinned, integrated commit containing the completed population/selection-contract repairs. Use an isolated implementation branch or worktree and new run directories. The exact checkpoint must be selected then; the documentation snapshot above is not a guarantee of execution readiness.

Existing `Theory.md`, `ESTIMATOR_PLAN.md`, audit files, frozen baseline, saved detections and all prior results stay authoritative for their own scope and unchanged by this documentation addition. Captures 5/6 remain untouched by this proposed comparison.

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
| Calibration | Joint coefficients and latent states; temporal regularization zero |
| Initial soft mean-anchor scales | 0.10 degree and 0.25 D; same across candidates |
| Initial coefficient prior | Same declared column-normalized policy, strength 0.001; extra-curvature penalty zero |
| Pilot/noise | One training-only log27 pilot/reference covariance shared within each split |
| Reference | Fresh same-split log27 under the same implementation and settings |
| Main criterion | Inner grouped excluded-P4 prediction loss with separate state/coverage/tail/scope guards |

The numerical scales are proposed controls inherited from the existing comparison, not physical uncertainty measurements. A change to any locked factor creates a separately named ablation, not an undocumented improvement to one exponent.

Do not remove y information while comparing response laws. Do not fit a continuous exponent, change accommodation demand labels, or reuse a log model's coefficients under a different exponent.

## 2. AR0 — Prerequisites and immutable experimental manifest

### 2.1 Consume, do not duplicate, the ongoing audit repairs

Before a selection-bearing run, verify the integrated implementation passes these contracts from the latest audit:

1. The scheduled population is independent of returned predictions. Exact frame identities and exactly three unique held-P4 slots are checked; missing outputs remain explicit unavailable records or raise a contract failure.
2. Exact shared-cohort ranking preserves the original expected exposures. A vanished exposure or insufficient shared support prevents promotion; it does not improve coverage by shrinking denominators.
3. Inner grouping matches the intended held-out gaze-condition or capture/demand question. Evaluation labels, records and all-three states cannot enter fitting or subset branch selection.
4. Invalid, ambiguous, weak-rank, bound, out-of-scope and failed-calibration cases are retained distinctly. Protocol-zero vertical targets are not asserted as framewise ground truth.

These are prerequisite tests, not a claim that the ongoing implementation is already fixed. Mathematical unit tests may proceed in isolation; a nested selection result may not bypass these gates.

### 2.2 Freeze a run manifest before evaluating candidates

Record source commit, exact design copies, implementation hashes, detector/interval hashes, correspondence, software versions, seeds, all candidate specifications and planned task IDs. Read accommodation demands from stored metadata. Record actual source-grid geometry only when available; do not infer it from gaze targets.

Freeze frame sampling from original rows before inspecting candidate outputs. For a first historical comparability screen, the existing 48 training / eight evaluation rows per fixation and central-80% rule may be reused on captures 1-4, but label this a sampled development screen. A denser confirmation population must be predeclared in a new manifest, used identically for every exponent, and never chosen from low-error survivors.

Keep gaze-condition and capture/demand split families separate. Adjacent frames and overlapping folds are not independent replicate experiments. Captures and accommodation demands are confounded in the existing recordings; more rows do not create more independent demands.

Predeclare guards for complete/scored fraction, per-exposure common support, physical state disagreement, axis errors, worst-point/tail errors, ambiguity, bounds and intended extrapolation. Store their provenance. This plan does not invent numerical scientific pass thresholds. Missing guards allow descriptive results only and no promotion decision.

**AR0 exit:** a versioned manifest and passing population/leakage contracts; otherwise record `not_ready_for_selection`.

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

Also store exact coefficient order, normalization, pilot model family/exponent, covariance convention, bounds, guards, manifest IDs and failed alternatives. Reject missing or incompatible power metadata. Never silently interpret a legacy model as a power model or assign a missing exponent from a current default.

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

Synthetic shared-state recovery should be tested under each generating exponent with fixed calibration conventions. Do not require the selection machinery to identify the generating exponent when several candidates have indistinguishable predictive geometry or latent-scale ambiguity.

**AR1 exit:** validated basis/model and generic solver compatibility; not a claim about real-data benefit.

## 4. AR2 — Controlled joint calibration of each fixed exponent

For every actual training split:

1. Load only its training groups and validate the scheduled training rows.
2. Fit or construct one logarithmic pilot and estimate coordinate covariance from that training subset alone. Freeze its reference-state policy and per-frame residual weights for all exponents in this split.
3. For each exponent, build the 27-column design and recompute its nominal-state coefficient initializer, column scales and prior using the same policy. Jointly optimize coefficients and training states with the same soft mean-anchor scales and bounds.
4. Preserve every start, certified/uncertified outcome, coefficient/state checkpoint, objective decomposition and declared continuation history.

An existing pilot or starting trajectory may be reused only when its training provenance exactly matches the current split and it contains no validation observations. In particular, an outer-fold pilot or latent trajectory that saw an inner validation group is prohibited inside that inner fit.

Use common nominal and seeded perturbed-nominal state starts across candidates. An additional same-split log27 training-state warm start can be used for every candidate if declared beforehand; it is an initializer, not a state penalty. Coefficients must be solved for the candidate's basis, not copied as if equivalent.

Keep covariance fixed during each frame inverse and use the same covariance within the exponent comparison. Retain the current shared-P1 propagation convention. Do not introduce the proposed shared-y discrepancy covariance at the same time.

The existing column-normalized prior offers a reproducible comparison policy, not identical function-space regularization across exponents. Report prior and optical contributions and training-only response/state shifts. A small predeclared prior-strength sensitivity may be a later inner-loop factor, applied to every exponent; do not adjust only a favored candidate after seeing outer errors.

A nine-fold screen of four exponents would have 36 planned fit outcomes before any extra starts or nested inner fits. It is a development screen, not a complete nested study. Failed fits contribute no manufactured predictions and retain every scheduled denominator.

**AR2 exit:** complete calibrated or explicitly failed candidate records; no exponent selected using training cost or outer measurements.

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

Use the companion theory's primary loss $L_{cross}$ and display `E_cross_px = sqrt(L_cross)`. RMS describes aggregation, not the target of comparison.

Every scorecard must include `G_theta_cross_deg`, `G_A_cross_D`, signed per-point/axis residuals, worst-point and tail metrics, raw and normalized point errors, condition/rank, branch ambiguity, bounds/clipping, per-subset nominal/empirical support and P1 context/parity. Physical means versus nominal gaze/demand and retained-data optimizer costs remain secondary calibration diagnostics.

Report scheduled/eligible/scored slots, complete triples, failed calibrations, expected/contributing exposures and exact memberships. Coverage denominators come from the frozen manifest, not from whatever the evaluator returns. For equal-exposure loss, missing exposure contributions mean an incomplete comparison, not silent omission or a zero loss.

Compare every power candidate directly with the same-split log reference on exact common frame IDs. Preserve full-population reporting, jointly interior and jointly supported cohorts separately. Do not infer a paired result by subtracting separately filtered RMS tables. Unsupported-by-extrema and intended endpoint extrapolation are distinct from numerical invalidity; do not force states into extrema to improve percentages.

### 5.3 Nested selection and honest development evidence

For a selection-bearing study, construct explicit inner and outer grouped split manifests. For held-out gaze-condition performance, inner validation should reflect that question across captures. For capture/demand transfer, use entire capture groups when remaining calibration support is adequate. Record infeasible nested splits; do not secretly replace them with random frame splits.

Within an outer training pool, fit each candidate only on inner training groups. Select the exponent using inner cross-check loss among candidates satisfying the predeclared guards and shared-exposure coverage. Break only numerical ties by a fixed rule favoring the log reference, rather than arbitrary inspection of trajectories. Refit the selected exponent on all outer training groups, then perform one sealed outer evaluation of that selected procedure. A separately predeclared log-reference outer score is a comparator, not another opportunity to change the selected exponent.

Do not choose a global winner by examining all outer candidate results and then call its minimum loss independent validation. Historical folds already used for these design decisions remain development evidence. Nested processing improves the procedure but cannot erase that history. New independent conditions or reserved transfer data address a different evidence question. [E2]

If guards are missing, shared support is inadequate, all candidates fail, or improvement depends on an unacceptable tail/coverage tradeoff, record no promotion. The reference may remain operational, but that fallback is not an assertion that it passed every scientific guard.

## 6. AR4 — Interpretation, confirmation and later extensions

The first decision concerns the accommodation law under fixed x/y, not whether y should be retained. A useful exponent should improve excluded-point prediction across relevant groups without merely compressing the accommodation scale or causing hidden numerical failures.

Report exponent stability across inner splits, calibration gains/residual trajectory differences using training rows only, and uncertainty or ambiguity in the choice. Four nominal demand levels and reused frames provide limited independent support for curvature; a sharply named exponent is not a physiological measurement.

After the law comparison is frozen, a later controlled experiment may compare the selected exponent and log reference under x/y versus x plus differential-y. Keep that a separate factor and do not attribute combined changes solely to the power law. Common-y discrepancy diagnosis from the ongoing audit remains relevant; a new accommodation basis is not presumed to solve it.

Only if residuals and cross-checks justify it, consider a 37-coefficient response, selective extra-curvature penalties, or a small independently weighted $a,a^2$ basis. Literal $a^p$, per-component exponents, fitted shift/scale parameters, nonlinear exponent optimization, direct masked-prediction training, vertical-gaze states and detector changes are out of scope for this initial plan.

Before transfer, freeze the response, fitting and selection rules, guards and any application-mask policy. Captures 5/6 remain untouched until then. Their unlabeled geometric agreement does not supply independent physiological accommodation accuracy.

## 7. Artifacts, reproducibility and concurrency

Use a new directory such as `experiments/full_position/accommodation_response_v1/`; this path is proposed and does not assert an existing run. Keep at least:

```text
config.json                       # pinned inputs, candidates, guards, runtime
population_manifest.json          # scheduled rows and three slot IDs per frame
split_manifest.json               # explicit inner/outer groups and exposure IDs
design_snapshot/                  # these documents and inherited contracts
implementation_snapshot/          # or immutable source commit plus verified hashes
fits/<split>/<candidate>/         # model or failed checkpoint, states, all starts
crosscheck/<split>/<candidate>/   # predictions, scores, branches, support, masks
selection.json                    # inner-only decisions and reference fallback
RESULTS.md                        # cross-check-first tables and limitations
verification.json                 # tests, hashes, cohorts, numerical checks
```

Exact layout may reuse the integrated runner's conventions; avoid duplicating infrastructure merely to match a new directory sketch. A run manifest must contain the resolved settings, not only references to mutable defaults. Resume is allowed only when source/data/candidate/guard/manifest hashes match. Never overwrite or append new-law scores into historical phase directories.

Parallelize independent fits with a declared worker limit and one BLAS/OMP thread per worker, subject to the environment's measured behavior. Use verified GPU execution only after value/derivative/branch/certificate parity for the power family. Record realized runtime and failures; do not claim the historical GPU speedup applies to this new experiment.

Before implementation commits, re-read the current branch. Preserve concurrent work and use new paths or coordinated additive patches. Do not force-push or reset to the documentation snapshot. This proposal adds no required source changes to the ongoing audit task.

## 8. Completion and promotion checklist

**Documentation complete** means the response family, isolated factors, derivatives, cross-check definition and staged plan are specified. That is the status of this addition.

**Implementation complete** additionally requires AR0/AR1 contracts, compatible metadata, verified profiling/inversion and unchanged legacy behavior.

**Experiment complete** requires every scheduled fit outcome, prediction slot, expected exposure, and selected/failed candidate to be accounted for; independent numerical/aggregation checks and historical preservation hashes must be recorded.

**Scientific improvement supported** requires paired cross-check gains under the declared calibration/coverage/tail/scope rules, appropriate grouped evaluation, and clear treatment of state-scale uncertainty. No automatic deployment promotion follows from a lower training cost or a new exponent. Absolute physiological accuracy is a separate validation claim.

## Source navigation

- [Companion accommodation-response theory](ACCOMMODATION_RESPONSE_THEORY.md): equations and interpretation.
- [Existing estimator plan](ESTIMATOR_PLAN.md) and [response model](full_position/model.py): inherited geometry, basis scale and interfaces at the inspected snapshot.
- [Calibration implementation](full_position/calibrate.py): profiled coefficient and prior conventions.
- [Latest audit](LATEST_RESULTS_AUDIT.md): ongoing scheduled-population/shared-exposure repairs and common-y diagnosis. Read its implementation status anew before AR0 exit.
- [Current status](CURRENT_STATUS.md): execution history, not evidence that this new workstream has run.
- **E1:** [SciPy `boxcox1p`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.boxcox1p.html).
- **E2:** [scikit-learn nested cross-validation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).

No estimator implementation, calibration, transfer evaluation, or new performance result is delivered by these two design documents.
