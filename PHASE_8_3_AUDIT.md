# Phase 8.3 audit — three-pair cross-check as the primary criterion

**Repository:** `rueijrwu/gaze_acc_joint_estimator`  
**Branch:** `exp5_full`  
**Inspected commit:** `07b789341e5d1e77e9fe4afba31e39d4b4968373`  
**Date:** 2026-10-07  
**Scope:** implementation review, saved-result interpretation, and independent algebra checks; this document does not implement or rerun an estimator.  
**Protocol clarification supplied by the user during this audit:** every calibration fixation has nominal vertical gaze **0 degrees**; horizontal targets are **-10, -5, 0, 5, 10 degrees**.

## Executive decision

**Retain both image-x and image-y P4 measurements and keep baseline27 as the present development reference.** Phase 8.3 supplies controlled evidence that retained image-y improves excluded-P4 prediction and shared-state agreement under the tested frozen responses. This is a result about the three-pair cross-check, not about fitting nominal accommodation demand more closely.

The comparison is strongest because the two methods within a response use identical coefficients, calibration convention, P1 context, and tested P4 point. The retained observations change; model capacity does not. Matched complete-interior frame comparisons favor x/y in all four response/split-family cells. However, capture-family coordinate improvements are smaller, and baseline27 has worse held-point y-axis and worst-point summaries despite better overall prediction. These exceptions must remain visible. [S1–S3]

**Cross-check is the scientific metric; RMS is only a declared aggregation of its errors.** Every phase should report and make scientific decisions using the same cross-check scorecard: excluded-point prediction, subset-state agreement, and coverage/support/tail safeguards. Nominal-label discrepancy and retained-data optimizer cost are secondary diagnostics, not substitute rankings. Appendix A specifies how to apply this rule throughout calibration, model selection, numerical changes, information tests, and final transfer.

Vertical gaze at zero does **not** imply zero or constant image-y coordinates. The present state remains `(horizontal gaze, accommodation)`, conditional on the zero-vertical-gaze calibration protocol. Do not introduce a vertical-gaze state or erase image-y responses merely because all vertical target labels are zero. Section 1 makes that distinction explicit.

## 1. Measurement and calibration contract

### 1.1 What is measured and what is estimated

There are three P1 points and three corresponding P4 points, each two dimensional: twelve raw coordinate measurements. After the recorded correspondence mapping, let them be `p_j` and `q_j`, for `j=1,2,3`.

The conditional model uses all three measured P1 points:

$$
\mathbf c_1=\tfrac13\sum_j\mathbf p_j,\qquad
\ell_1=\sqrt{\mathcal A_1},\qquad
\mathbf r_j=(\mathbf p_j-\mathbf c_1)/\ell_1,\qquad
\mathbf v_j=(\mathbf q_j-\mathbf c_1)/\ell_1.
$$

Its prediction is

$$
\widehat{\mathbf q}_j(x)
=\mathbf c_1+\ell_1\{\mathbf D(x)+\mathcal T(x)\mathbf r_j\},
\qquad x=(\theta_x,A).
$$

This is six P4 coordinate responses conditional on measured P1 context, not twelve independent state equations. P1 localization uncertainty remains part of residual covariance. The current conditional model does not separately fit P1 shape as another state response. The prescribed summaries remain `d_x=(c4_x-c1_x)/sqrt(area(P1))` and `rho4=area(P4)/area(P1)`, but they must not replace the full coordinates or be counted again as independent observations. [S2, S3, S5, S7]

### 1.2 Vertical gaze is fixed at zero in the calibration protocol

The user's clarification establishes the intended calibration slice:

$$
\theta_x^{nom}\in\{-10,-5,0,5,10\}^{\circ},\qquad
\theta_y^{nom}=0^{\circ}\quad\text{for every fixation}.
$$

This is protocol information supplied in this conversation, not a new independent measurement of each frame's true gaze. Record its provenance accordingly. The following are different quantities:

- `theta_y`: physical vertical gaze angle; not varied by this calibration protocol.
- `q_j,y` and `v_j,y`: observed image coordinates; not constrained to zero.
- `D_y(theta_x,A)` and the second row of `T(theta_x,A)`: calibrated vertical-image responses to horizontal gaze and accommodation.

For the current model,

$$
\widehat v_{j,y}=D_y(\theta_x,A)
+T_{21}(\theta_x,A)r_{j,x}+T_{22}(\theta_x,A)r_{j,y}.
$$

Its derivatives with respect to `theta_x` and `A` need not vanish when physical vertical gaze is zero. Horizontal field changes, accommodation-dependent mapping, and the geometry of the sampled source pattern can all be represented by those image-y responses. Phase 8.3's `x` versus `xy` therefore means **retained image channels**, not one-dimensional versus two-dimensional gaze estimation. [S2, S7]

Recommended metadata additions are `state_variables=[theta_x_deg,A_D]`, `calibration_theta_y_nominal_deg=0`, `calibration_theta_y_role=protocol_constraint_not_framewise_ground_truth`, and `image_axis_convention`. Do not fabricate per-frame vertical truth labels, overwrite measured image-y, or infer vertical-gaze calibration from constant zero targets. Small fixation deviations or alignment changes can still be unmodeled effects; their size has not been independently established here. Extension to varying vertical gaze requires new calibration support and identification tests, not another unconstrained per-frame variable.

## 2. The primary cross-check and its aggregation

### 2.1 Three excluded-P4 predictions

For each frame `n` and excluded point `j`, retain all three P1 points and estimate the shared state using only allowed coordinates of the other two P4 points:

$$
\widehat x_{n,-j}^{(m)}
=\arg\min_{x\in\mathcal B}
 e_{n,I_j^{(m)}}(x)^T R_{n,I_j^{(m)}I_j^{(m)}}^{-1}
 e_{n,I_j^{(m)}}(x),\qquad m\in\{x,xy\}.
$$

`x` retains two P4 x coordinates. `xy` retains four P4 x/y coordinates. **Both predict and score both coordinates of the same excluded point.** Its error is

$$
\boldsymbol\epsilon_{nj}^{(m)}
=\mathbf q_{nj}-\widehat{\mathbf q}_{nj}(\widehat x_{n,-j}^{(m)}).
$$

The excluded point, its full-triangle centroid/area, all-three inverse, and its eventual prediction error must not affect the subset solve or branch choice. Stored all-three states may accompany the report, but are not inputs to the subset inverse. Coefficients and noise policies must come from groups outside the scored evaluation group. [S2, S3]

At a regular fixed-coefficient two-state inverse, retained x has two observations for two states; retained xy has four observations for two states. The former can fit retained observations exactly without thereby validating the state. For example, the inspected baseline27/gaze_+0 row 11090 record has zero retained-x cost but a nonzero excluded-point error `(-0.489, 3.691)` pixels. This directly illustrates why retained cost must not rank cross-check quality. [S8]

### 2.2 Complete-frame scorecard

For a frame with three scored, identifiable, unambiguous exclusions, define

$$
E_n=\sqrt{\tfrac13\sum_{j=1}^{3}\|\boldsymbol\epsilon_{nj}\|^2},\qquad
W_n=\max_j\|\boldsymbol\epsilon_{nj}\|.
$$

For the three subset states, separately calculate

$$
G_{\theta,n}=\sqrt{\tfrac13\sum_{j<k}(\widehat\theta_{n,-j}-\widehat\theta_{n,-k})^2},
\qquad
G_{A,n}=\sqrt{\tfrac13\sum_{j<k}(\widehat A_{n,-j}-\widehat A_{n,-k})^2}.
$$

`E` is excluded-measurement prediction disagreement; `G_theta` and `G_A` are shared-state disagreement. The square root of mean squares is a summary operation in each definition, not a different scientific objective. Preserve all point/axis signed errors, ranges, and the worst-point result rather than reporting only `E`. [S4]

For fixation/capture exposures `k`, the existing equal-exposure summary is

$$
L_{cross}=\frac1K\sum_k\frac1{N_k}\sum_{n\in k}E_n^2,
\qquad E_{cross,eq}=\sqrt{L_{cross}}.
$$

On an identical paired cohort, compare `xy minus x` squared errors before exposure averaging. A negative mean change favors xy. Record `N_k`, exposure IDs, absent exposures, and scheduled denominators. An average over surviving groups is not a score for missing groups.

A partial frame may have useful individual-point errors, but must not be silently converted into a complete triple by dividing by three. Boundary cases stay in the full scorecard; interior and empirical-support cohorts are separately labeled. Shared clipping or calibration-scale changes can make `G` small without improving point prediction. These correlated subset states are not three independent physiological references. [S4, S7]

## 3. Implementation audit

| Contract | Finding at the inspected commit |
|---|---|
| Same trained response within x/xy comparison | Correct: baseline27 and strong-anchor37 are frozen separately; neither is refit for its x-only inverse. |
| Exact retained observation masks | Correct: `phase83.task` supplies two retained x values; `predict_holdout(channels='x')` constructs the matching indices. |
| No excluded-P4 leakage in reviewed estimator path | No direct leak found: subset inversion precedes scoring; the helper has no excluded-measurement argument. |
| Shared measured P1 reference | Correct: all three P1 points define the same normalization and context for both methods. |
| Correct subset weighting | Correct: covariance is marginalized before inversion/whitening; it is constructed from P1 and the frozen training pilot/noise policy. |
| Comparable numerical policy | X uses the 49-start scalar solver with polishing; xy reuses the frozen certified holdouts. Numerical objective scaling preserves statistical cost units in the reviewed inverse. |
| Failure/coverage records | Correct in the inspected reporting contract: invalid, weak-rank, ambiguous, boundary, and complete-triple counts remain distinct. |
| Cross-check aggregates | Reviewed formulas use excluded-point error and subset-state disagreement, with paired identities and equal-exposure means. |
| Training-support metadata | Incomplete in inspected Phase 8.3 joined output; see finding F1 below. |

Sources: [S2–S6, S8].

The saved verifier reports 1,280 joined frames, 1,127 independent complete-frame metric checks, 1,699 x holdout recomputations, 1,716 selected-branch certificate checks, 640 frozen-xy reuse checks, and four paired-membership checks. The status reports 59 regression tests. These are **repository-reported executions**, not reruns performed for this audit. Certificates establish checked local numerical conditions, not global branch completeness. [S1, S6, S9]

The retained-x regression reviewed here checks one synthetic excluded index with capacity27 and exact covariance marginalization. Extend this specific x-path check to all three indices, both capacities, and perturbations of unused retained-y coordinates as well as the excluded point. Existing general xy leakage tests do not substitute for every new x-mask integration contract. [S10]

## 4. Results interpreted strictly as cross-check performance

### 4.1 Full available populations

These are saved equal-fixation complete-frame summaries. In every cell, 160 frames and 480 checks are scheduled, including 17 input-invalid frames and 51 invalid check slots. `E` is P4 cross-prediction, not target-label error. [S1]

| Response | Family | Retained | Scored / 480 | Complete / 160 | E (px) | G_theta (deg) | G_A (D) | Worst point (px) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| baseline27 | gaze | x | 429 | 143 | 4.203 | 0.930 | 1.010 | 5.417 |
| baseline27 | gaze | xy | 429 | 143 | 3.421 | 0.269 | 0.311 | 4.385 |
| baseline27 | capture | x | 429 | 143 | 4.149 | 0.721 | 0.906 | 5.130 |
| baseline27 | capture | xy | 429 | 143 | 4.014 | 0.281 | 0.319 | 5.222 |
| strong-anchor37 | gaze | x | 423 | 137 | 5.550 | 1.141 | 1.379 | 7.706 |
| strong-anchor37 | gaze | xy | 429 | 143 | 4.764 | 0.440 | 0.530 | 6.199 |
| strong-anchor37 | capture | x | 418 | 132 | 4.692 | 0.745 | 1.135 | 6.137 |
| strong-anchor37 | capture | xy | 429 | 143 | 4.293 | 0.276 | 0.366 | 5.593 |

Baseline27 has identical scored support between methods. From the rounded values, adding y reduces gaze `E` by about 18.6% and capture `E` by only 3.3%; state disagreement reductions are much larger, approximately 61–71%. Thus the most consistent benefit is stabilization of the shared state, accompanied by positive but uneven coordinate-prediction gains.

Strong-anchor37's raw x and xy rows have different support. Do not subtract those RMS values and call the difference a paired improvement. Its x-only path loses six gaze and eleven capture scores to rank/ambiguity; xy retains all 429 eligible checks. Improved availability is a separate benefit, and exact matched comparisons are needed for error claims. Rank and ambiguity categories overlap, so their counts must not be blindly added. [S1, S6]

### 4.2 Exact paired comparisons

Saved equal-fixation mean changes below are `xy minus x`. All columns are squared cross-check metrics; negative values favor xy. [S1]

| Response | Family | Complete-frame cohort | n | Delta E² (px²) | Delta G_theta² (deg²) | Delta G_A² (D²) |
|---|---|---|---:|---:|---:|---:|
| baseline27 | gaze | all paired | 143 | -5.962 | -0.792 | -0.924 |
| baseline27 | gaze | jointly interior | 121 | -4.911 | -0.630 | -0.751 |
| baseline27 | capture | all paired | 143 | -1.100 | -0.442 | -0.718 |
| baseline27 | capture | jointly interior | 112 | -0.259 | -0.359 | -0.482 |
| strong-anchor37 | gaze | all paired | 137 | -8.762 | -1.107 | -1.633 |
| strong-anchor37 | gaze | jointly interior | 84 | -8.386 | -1.016 | -1.370 |
| strong-anchor37 | capture | all paired | 132 | -3.532 | -0.468 | -1.140 |
| strong-anchor37 | capture | jointly interior | 110 | -1.519 | -0.147 | -0.558 |

These comparisons support incremental retained-y value beyond an explanation based solely on bound recovery. They do not demonstrate that every frame, point, axis, or unseen recording improves.

**Do not mix point-level and complete-frame masks.** For example, strong-anchor37 gaze has 334 jointly interior point tests with point-level delta squared error -0.233 px², whereas its 84 jointly interior complete frames have delta E² -8.386 px². These are different populations: three eligible points are required for the latter. Publish both membership sets and the exposure counts; the two numbers are not interchangeable measures of one cohort.

### 4.3 The capture-family tradeoff is scientifically important

For baseline27 capture, adding y changes held-point x-axis RMS from 2.363 to 1.846 pixels, but held-point y-axis RMS from 3.411 to 3.565 pixels. Worst-point RMS rises from 5.130 to 5.222 pixels. These are increases of roughly 4.5% and 1.8% in the latter two rounded summaries despite improved overall `E`. Support is the same for this comparison. [S1]

Do not call this uniform geometric improvement. Inspect paired signed residuals by excluded point, axis, fixation, and capture. A plausible explanation to test is that extra y observations stabilize the state while a systematic y-response mismatch remains; the aggregate table does not prove that cause. Do not suppress the worsening axis with a new weight chosen from these evaluation errors.

### 4.4 What the experiment establishes—and does not

Supported: under two already calibrated full-coordinate responses and the sampled grouped conditions, using retained P4 image-y as well as image-x improves complete-frame excluded-point and state-agreement averages on exact shared interior support.

Not established: superiority over an independently trained x-only system; a controlled two-source-versus-three-source hardware comparison; full-coordinate superiority over the historical two-channel baseline; a uniquely identified aberration mechanism; calibrated artifact detection; or physiological gaze/accommodation accuracy.

Both methods already use three-source P1 context, and both frozen responses were calibrated using full-coordinate training data. This is intentionally a **fixed-response inference-information ablation**, not a comparison of independently optimized end-to-end x-only and xy pipelines. Calling it simply “full position beats the old estimator” would overstate the result.

## 5. Remaining findings and targeted recommendations

### F1 — Fill training-support diagnostics before using support gates

The inspected `baseline27/gaze_+0/x_frames.json` row 11090 has `interior=true` but null gaze/accommodation anchor and empirical-support flags. The reviewed `phase83.task` uses `join_records` directly and does not enrich these fields from the frozen model and training states as the sensitivity pipeline does. This is a confirmed reporting gap in that record and code path, not evidence that its state is actually out of support. [S2, S4, S8]

Populate support for **each subset state** from the corresponding frozen response, its declared training groups, and the available P1 context. Record separate nominal-anchor, empirical-state, P1-context, and parity checks with explicit unavailable reasons. Do not copy the all-three state's support to every subset. Keep solver-interior masks distinct from training-support masks and preserve all historical scores. Componentwise training extrema are descriptive bounds, not a validated multidimensional support envelope.

### F2 — Identify which retained-y information helps

The x/xy result does not distinguish common vertical-image displacement from relative shape change. For retained points `a,b`, define

$$
y_c=(v_{a,y}+v_{b,y})/2,\qquad y_d=v_{a,y}-v_{b,y}.
$$

Compare, under the same frozen response, `x`, `x + y_c`, `x + y_d`, and `x + y_c + y_d`. No excluded point enters these quantities. The last representation is invertibly equivalent to retained xy and should reproduce its objective and predictions when covariance and derivatives are transformed consistently.

Use an explicit linear transform `H` on the retained observations, predictions, and Jacobians, with covariance `H R_ret H^T`. Never assign independent weights to the transformed channels or whiten the complete six-vector before masking. Check rank, branch sets, coverage, and the same excluded-point scorecard for each mask. Common/differential labels describe geometry; neither is automatically a pure physiological signal.

This small diagnostic is more informative than immediately expanding the polynomial. With zero vertical nominal gaze, it asks whether useful horizontal-gaze/accommodation information is mainly common vertical-image motion, differential sampled distortion, or both.

### F3 — Keep branch and uncertainty limitations visible

Retained x has fewer constraints and more ambiguity in strong-anchor37. The fixed plausible-cost threshold is a heuristic, not a calibrated confidence interval, and certification of discovered minima is not proof all branches were found. Preserve prediction sets for ambiguous cases; do not choose the branch nearest the excluded measurement. Inspect low-rank and small-gap cases with the independent profile method before concluding that a new weighting or model change adds information. [S3, S6]

The current covariance propagates shared-P1 noise but excludes established total model discrepancy and global coefficient uncertainty. Its correctness as propagation mathematics is not proof its predictive ellipses are calibrated. Cross-check pixel errors and coverage remain usable without calibrated Gaussian probabilities. [S5]

### F4 — Interpret state agreement under a fixed calibration convention

The state-scale symmetry identified in Phase 8.2 remains relevant across refitted models: optical predictions can stay the same while physical-unit state spreads change. In the Phase 8.3 **within-response** x/xy comparison, coefficients and state convention are fixed, so that specific refitting confound is controlled. It returns when comparing capacity/anchor/prior changes. Keep training-scale diagnostics and bounds alongside `G`; do not minimize `G` alone. [S7]

### F5 — Strengthen regression coverage and aggregation contracts

Add x-mask noninterference tests for every excluded point and both capacities. Perturb the excluded x/y, the unused retained-y values, nominal evaluation labels, and stored all-three state; x-path preprocessing, weights, starts, branches, and predictions must remain unchanged conditional on an unchanged validity mask. Altering an input-validity flag is a different test and may legitimately change availability. Add integration tests at the raw-array entry point, not only helper tests.

Add the invertible common/differential-y equivalence test, unknown-support preservation, duplicate-ID rejection, absent-exposure reporting, and incomplete-triple tests. In `audit_followup.transitions`, transition labels are enumerated from point records; a frame-only `bound_to_bound` category can exist even when no individual point has that transition. Enumerate the union of point and frame labels (or all declared labels) to avoid omitting such a frame stratum in future data. This code-path risk was derived from source; it has not been shown to change the saved Phase 8.3 tables. [S10, S11]

## 6. Immediate next-step order

**P0: Make one cross-check scorecard authoritative for every phase.** Backfill it from saved predictions where possible; fix F1 support metadata and publish paired per-axis/worst-point diagnostics. Append the vertical-zero protocol metadata without altering historical detections or claiming new labels were measured.

**P1: Run the retained-y common/differential information test.** Freeze each response and evaluate all masks under the same leakage-safe cross-check and scheduled population. Include baseline27 and strong-anchor37 as contrasting development references, not as a newly selected winner.

**P2: Diagnose reproducible response mismatch before another capacity experiment.** Inspect training-only and group-held-out signed residuals, especially baseline27 capture y and worst-point worsening. If the extra curvature is justified, test selective shrinkage of the ten added37 terms with inner-group cross-check selection. The available penalty implementation is not itself evidence the penalty improves predictions. [S7, S10]

**P3: Use nested grouped selection and denser evaluation.** Current folds have guided several development decisions. Do not relabel them untouched final validation. Keep captures 5/6 untouched until choices are frozen; their later geometric cross-check does not require known demand labels, but it does not establish physiological accuracy.

---

# Appendix A — Make cross-check the main metric throughout the full workflow

This appendix incorporates the user's request that **the full set of phases—not only Phase 8.3—use cross-check as the main metric**. It is a proposed execution/selection contract; it does not claim these pipeline changes are already implemented.

## A1. One scientific success definition

> A candidate succeeds when a shared gaze/accommodation state inferred without a P4 point predicts that excluded point more reliably, with compatible subset-state agreement and acceptable coverage, support, and tail behavior.

Use `E_cross` as the primary coordinate outcome and `G_theta_cross`, `G_A_cross` as required shared-state outcomes. No raw sum mixes pixels, degrees, and diopters. Keep a vector of outcomes and declared guards instead of inventing one weighted scalar that hides tradeoffs.

Recommended headline columns for every phase are:

| Mandatory outcome | Reporting meaning |
|---|---|
| P4 cross-prediction | Equal-exposure `E_cross`, per-point/axis bias, normalized companion, distribution, and worst-point/tail summaries |
| Shared-state agreement | `G_theta_cross` and `G_A_cross` separately; state ranges, bound clipping, calibration-scale context |
| Coverage | Scheduled, input-eligible, certified, identifiable, unambiguous, scored, and complete-triple counts |
| Support | Computational bounds versus nominal/empirical state support and P1 context; unknown is not true |
| Paired change | Exact IDs, squared changes, exposure counts, and uncertainty evidence appropriate to grouped data |

Nominal fixation-mean discrepancy, optical training residual, prior/anchor contributions, and numerical stationarity move to a secondary diagnostics panel. They are still required to understand calibration and numerical correctness; they cannot replace the primary cross-check ranking. A two-channel model without an individual-P4 decoder is `not applicable` on the coordinate cross-check, never zero-error or the winner by default.

RMS may remain the main historical summary to preserve comparability. Rename ambiguous displays to **“excluded-P4 cross-prediction error, equal-exposure RMS”** and **“cross-subset gaze/accommodation disagreement”**. Add medians, upper quantiles, worst-point and signed errors; do not switch aggregates after inspecting which one makes a candidate look best.

## A2. Apply the same criterion at every phase

| Phase or decision | What cross-check must decide | Necessary constraints / what not to do |
|---|---|---|
| Geometry and normalization | Whether a proposed representation preserves or improves group-held-out prediction at declared support | Keep correspondence and uncertainty propagation; do not reject non-similarity simply because it is distorted |
| Numerical implementation | Whether certified evaluation coverage and honest predictions are preserved after a solver change | Solver correctness is judged by the retained objective/KKT checks, not by selecting a favorable excluded-point branch |
| Phase 8.1 reporting | Whether all three point checks and all three subset-state comparisons are recorded correctly | Post-processing is not new predictions or independent validation |
| Phase 8.2 calibration controls | Which anchor/prior/noise setting has better inner-group cross-check performance | Retain physical scale anchors; do not rank settings by nominal-demand fit, training cost, or shrunken state spread alone |
| Phase 8.3 information masks | Whether added retained channels improve the same excluded point and shared-state scorecard | Freeze response and calibration; use exact observation masks and covariance marginals |
| Response/context/curvature changes | Whether added terms transfer across groups on the cross-check | Add only supported terms; do not let arbitrary per-frame nuisance variables erase residuals |
| Final model/threshold selection | Whether a frozen candidate meets the predeclared cross-check/coverage requirements on outer or new data | No repeated tuning on the final evaluation; no universal physiological claim from agreement |
| Application and monitoring | Whether the stored model continues to explain three responses when all are measured | Store all-three state and separate cross-checks; do not overwrite bad predictions or treat a failed check as a proven detector error |

This changes **scientific selection and reporting across phases**. It does not authorize using excluded observations inside an individual subset solve.

## A3. Calibration remains an inner fit; cross-check drives outer selection

The present variable-projection objective fits training coordinates, soft fixation means, and a coefficient prior. That objective is a valid **candidate-construction method**, not the final scientific score. Replacing its name with “cross-check loss” would not make it cross-validation.

For each outer grouped fold:

1. Seal the outer groups before fitting any pilot, noise model, prior center, feature transform, or threshold.
2. Within the remaining training groups, create inner group folds. For each declared candidate setting, calibrate using only inner-training groups, then freeze all global parameters and policies.
3. On each inner-validation frame, run three independent P4 exclusions. Its subset states may use that frame's all-three P1 context and allowed retained P4 measurements; they may not use nominal evaluation labels, its excluded point, or a coefficient fit that included that validation group.
4. Aggregate the common cross-check scorecard, retaining unavailable cases. Select using the predeclared prediction/coverage/state/tail rule.
5. Refit the selected setting on the outer-training groups, freeze it, and evaluate the outer groups once using the same cross-check contract.

The following pseudocode states the separation; function names describe proposed responsibilities, not existing APIs:

```python
for outer in grouped_outer_splits:
    develop, final_test = split_groups_before_preprocessing(outer)
    for candidate in predeclared_candidates:
        cards = []
        for inner in grouped_inner_splits(develop):
            model = calibrate(inner.train, candidate)  # pilot, noise and priors also train-only
            cards.append(crosscheck_all_three_exclusions(model, inner.validation))
        candidate.card = aggregate_with_scheduled_denominators(cards)
    choice = select_by_crosscheck_with_predeclared_guards(predeclared_candidates)
    model = calibrate(develop, choice)
    save_frozen_outer_crosscheck(model, final_test)
```

The stored Phase 8.1/8.2/8.3 outputs can be rescored consistently without refitting, but their prior use for development does not become nested validation retroactively. Genuine inner-selected claims need appropriately generated predictions.

Do not remove the nominal anchors just because label discrepancy is secondary. They establish physical state scales and prevent the empirical scale freedom from making smaller `G` look like added information. Vertical nominal gaze stays zero as the declared calibration condition; no independent vertical state is estimated.

## A4. A concrete candidate-promotion rule

Before a future selection run, specify the unit/normalizer, grouping weights, uncertainty method, minimum availability, and tolerable changes in `G`, worst-point/axis behavior, and support. These tolerances must come from experimental requirements or separate development evidence; the current audit supplies **no validated universal pass threshold**.

A practical rule is:

- First require the candidate to satisfy numerical, identifiability, coverage, and protocol/support requirements; record every failure instead of dropping it.
- Among eligible candidates, prefer lower paired inner-validation `L_cross` (mean squared excluded-point error). Use `G_theta`, `G_A`, tails and worst-point behavior as explicit guards or a Pareto comparison. An improvement in only one component is reported as a tradeoff, not an overall winner.
- If the required guards are not defined or results conflict, report **no promotion decision**, not a fabricated combined score. Preserve the simpler reference until evidence supports the change.

For optional operational thresholds, report separately: (i) fraction of scheduled frames that are testable, (ii) fraction of testable triples passing all three point thresholds and the declared state-agreement requirements, and (iii) passing triples divided by all scheduled frames. Distinguish unavailable, evaluated-and-failed, and passed cases. Bound-clipped agreement alone cannot count as a pass. Do not use inflated or uncalibrated Mahalanobis thresholds as a substitute for established tolerances.

A matched-score improvement and a coverage reduction are two outcomes. Never compute the decision only on a candidate's surviving easy frames. Preserve full fixed-population coverage, exact paired cohorts, and per-exposure counts; absent groups require an explicit unavailable status.

## A5. Cross-check-aware training is possible, but is a separate algorithm

The immediate recommendation is **cross-check-based grouped model selection with the existing calibration solver**. This is sufficient to make cross-check the main criterion throughout the workflow without destabilizing a verified inner optimizer.

A later masked cross-prediction training objective could differentiate through

$$
\widehat x_{n,-j}(\beta)
=\arg\min_x \|e_{n,I_j}(x;\beta)\|_{R_{I_jI_j}^{-1}}^2
$$

and penalize the predicted excluded training point. Those training targets are allowed to influence fitted coefficients, but such training errors are **not untouched cross-check evidence**; separate validation groups are still necessary. Preserve state-scale anchoring, shared global coefficients, and proper treatment of ambiguous/bounded inner solves. A loss forcing the subset states to be equal is especially unsafe in isolation because clipping or state-scale compression can reduce it without better optical prediction.

This is a bilevel/nonlinear coefficient problem, not automatically the same linear variable-projection solver. It needs a new objective schema, implicit/explicit derivative verification, branch policy, and grouped outer evaluation. Do not silently bolt the masked prediction loss onto the current profiler or claim a same-frame training reconstruction is a held-out test.

## A6. Reporting and code changes that implement this policy

Create a shared scorecard contract used by `crosscheck.py`, `sensitivity.py`, `phase83.py`, any future curvature/context runner, and application reporting. Reuse existing formulas rather than maintaining slightly different primary metrics in each script.

Every scored slot should identify run/candidate, split/fold, capture/fixation/row/frame, excluded point, retained image channels, normalization, frozen model/noise hashes, subset state, branch/certificate, pixel and normalized prediction/error, rank, support, and reason for any unavailable field. Complete-frame records add `E_cross`, `G_theta_cross`, `G_A_cross`, worst point, clipping, and the exact contributing slots. Existing names can remain aliases for compatibility.

Every comparison should declare direction (`candidate minus reference`), exact point/frame memberships, aggregation weights, cohort rule, exposure counts, and whether it is exploratory, inner-selection, outer evaluation, or final transfer. Store same-cohort per-axis changes with the coordinate/state scorecard, not only overall RMS.

Add acceptance regressions for:

- excluded-point and unused-channel noninterference through preprocessing, weighting, starts, branch selection, and predictions;
- equal-fixation versus pooled aggregation, missing exposures, duplicate identities, partial triples, and point/frame cohort differences;
- zero vertical nominal gaze without zeroing observed image-y or creating a vertical state;
- shared clipping and state-scale symmetry, proving that zero `G` alone does not imply a correct prediction;
- model selection cannot inspect outer-test labels, excluded coordinates during inversion, or outer scores while tuning hyperparameters;
- numerical changes preserve the scientific cost definition and cannot choose a branch using its excluded error;
- support unknowns remain unknown, and report headers always distinguish cross-check from anchor and fitting diagnostics.

## A7. Why retained-y can help on a zero-vertical-gaze slice

At fixed response, context, and covariance, partition retained observations into x and y. For a regular local model with Jacobians `J_x,J_y`, define

$$
S=R_{yy}-R_{yx}R_{xx}^{-1}R_{xy},\qquad
J_{y|x}=J_y-R_{yx}R_{xx}^{-1}J_x.
$$

Then the local information identity is

$$
J^T R^{-1}J
=J_x^T R_{xx}^{-1}J_x+J_{y|x}^T S^{-1}J_{y|x}.
$$

The additional term is positive semidefinite when the covariance is positive definite. The derivatives are with respect to **horizontal gaze and accommodation**, so no varying vertical-gaze parameter is needed. This is a local mathematical explanation of potentially useful extra measurements, not a promise that a misspecified real model cannot get worse held-out errors. Phase 8.3's cross-check supplies the empirical test, including its documented exceptions.

Independent audit checks verified this identity and common/differential-y covariance equivalence on 100 random positive-definite cases, with maximum discrepancies about `2.84e-14` and `1.95e-14`. A separate synthetic two-state response with no vertical-gaze variable had nonzero image-y derivatives for both horizontal gaze and accommodation. These checks validate the algebra and interpretation only; they are not new real-data estimator performance results.

## Evidence and source map

Repository links below refer to content inspected at the pinned commit in this report; branch files may advance. The vertical-zero protocol is explicitly the user's clarification during this audit, not a claim extracted from a detection file.

- **[S1]** [Phase 8.3 results](experiments/full_position/phase83_retained_channels_v1/RESULTS.md): metric tables, paired support, counts, and repository-reported verification scope.
- **[S2]** [Phase 8.3 runner](full_position/phase83.py): frozen response selection, exact retained-x inputs, xy reuse, scoring and joins.
- **[S3]** [Inverse and holdout code](full_position/invert.py): marginal weighting, masks, certification, branch handling, separated prediction/scoring.
- **[S4]** [Cross-check aggregation](full_position/crosscheck.py): slot eligibility, complete triples, support fields, and exposure weighting.
- **[S5]** [Noise propagation](full_position/noise.py): P1-shared covariance, marginalization, and conditional prediction covariance.
- **[S6]** [Phase 8.3 verification](experiments/full_position/phase83_retained_channels_v1/verification.json): saved branch/rank/coverage and snapshot checks.
- **[S7]** [Theory](Theory.md), [estimator plan](ESTIMATOR_PLAN.md), and [model](full_position/model.py): shared two-state response, scale convention, state-scale symmetry, and staged design.
- **[S8]** [Inspected baseline27 gaze-zero x records](experiments/full_position/phase83_retained_channels_v1/baseline27/gaze_+0/x_frames.json): concrete retained-cost and missing-support example, capture 1 row 11090.
- **[S9]** [Current status](CURRENT_STATUS.md): completed phases, reported 59-test suite, and untouched transfer status.
- **[S10]** [Audit-follow-up regressions](tests/test_audit_followup_regressions.py): x-mask, scale, objective, continuation, and curvature tests reviewed.
- **[S11]** [Paired transition code](full_position/audit_followup.py): point/frame membership and squared-change aggregation.

**Verification boundary:** this audit read implementation and saved results through the repository connector and ran independent self-contained algebra/aggregation checks. It did not rerun the repository's 59-test suite, all raw-frame joins, 18 Phase 8.3 tasks, or the calibration studies. Saved verifier claims are attributed to the repository. No model, detection, prediction, historical result, or computational threshold is changed by this document.
