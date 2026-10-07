# Implementation and results audit: three-pair predictive agreement

**Repository:** `rueijrwu/gaze_acc_joint_estimator`  
**Branch:** `exp5_full`  
**Historical audit snapshot:** `c2584292d13393a4c5a8bb34afdea9c19f90e67f`
**Revision date:** 2026-10-07  
**Revision scope:** audit and evaluation contract, Phase 8.1 frozen-record cross-check, Phase 8.2 grouped joint calibration and its audit follow-up, and Phase 8.3 frozen-response retained-channel comparison. Phase 8.2 refits shared states and coefficients; Phase 8.3 keeps responses fixed and computes retained-x inverses. No physiological measurements are added.

## Phase 8.2 audit response and Phase 8.3 completion

The recommendations in [PHASE_8_2_AUDIT.md](PHASE_8_2_AUDIT.md) are addressed by
the exact state-scale symmetry and real branch-cost scaling regressions,
[saved transition/gauge diagnostics](experiments/full_position/phase82_audit_transitions_v1/RESULTS.md),
[strict preconditioned continuation](experiments/full_position/phase82_strict_retry_v1/RESULTS.md),
[two anchor-axis conditions](experiments/full_position/axis_anchor_sensitivity_v1/RESULTS.md),
and [Phase 8.3](experiments/full_position/phase83_retained_channels_v1/RESULTS.md).
All 36 added joint fits are certified, retaining 429 scored slots per family and
condition/capacity; the training-only failed-fit retry passes the original gate
without changing the historical failure or its evaluation denominator.

Phase 8.3 keeps all P1 references and excludes the same P4 point for retained-x
and retained-x/y inversion. The two methods receive exactly their retained
coordinates and covariance marginals. Retained-y information reduces E and raw
state disagreement on exactly paired interior frames for both frozen responses.
The richer x-only response has additional rank/ambiguity failures (6 gaze and
11 capture slots). The independently verified run retains all scheduled slots,
84,084 x start candidates, and all branch/bound evidence. Its 18 tasks took 57
seconds with ten CPU workers; the 36 joint fits took 634 seconds with eight.
BLAS/OMP are one per worker, consistent with the measured benchmark.

Gaze-only strengthening appears more useful for conditional37 capture-state
agreement than accommodation-only strengthening. Neither establishes a new
overall winner: baseline27 still has lower held-point error. Selective shrinkage
of only the ten extra conditional37 curvature terms is implemented and tested,
but its experiment remains conditional on residual evidence supporting richer
capacity, as the audit requires. Captures 5/6 remain untouched; this is reused
development evidence rather than physiological or independent final validation.
All 59 regression tests pass. [CURRENT_STATUS.md](CURRENT_STATUS.md) provides
the weighted tables, paired squared changes, verification links, and next work.

## Executive assessment

**The main question is whether different parts of the three-pair measurements support one shared gaze/accommodation state and predict the remaining measured geometry.** Agreement means consistency with the same optical state, not equality of the three P4 positions and not preservation of triangle similarity.

The setup contains **three P1 points and three corresponding P4 points**, each measured in x and y: six image points and twelve raw coordinates. There are **two unknown states**, horizontal gaze and accommodation. The historical **two-channel** estimator compresses the same measured geometry into displacement and area ratio; it does not mean that this experiment measured only two pairs.

The implementation substantially follows the conditional full-position theory. It preserves the three-point P1 reference, fits all six P4 coordinate responses, and predicts each excluded P4 point from a state inferred without that point. Inverse certification and the principal reporting/compatibility findings from the original audit have been addressed. The Phase 8.1 saved-model results support `conditional27` over `conditional37` on the measured cross-prediction criterion. The later Phase 8.2 joint-refit sensitivity found positive full-population `conditional37` strong-anchor results in both split families, while other settings show tradeoffs; neither study selects a deployment model. They do not establish adequate uncertainty calibration, superior physiological accuracy, or a direct cross-prediction advantage over a two-channel model that does not itself predict individual P4 points.

The evaluation priorities are now:

1. **Primary predictive evidence:** three-way held-out P4 prediction errors on matched, declared populations.
2. **Complementary shared-state evidence:** disagreement among the three subset gaze/accommodation estimates, with rank, ambiguity, bounds, and coverage.
3. **Supporting diagnostics:** all-three fit residuals, nominal fixation-mean agreement, calibration sensitivity, and uncertainty calibration.

Nominal accommodation demand is not measured accommodation. Better agreement with demand is not the primary definition of success for this experiment. Equally, low subset-state disagreement alone is insufficient: a biased or uninformative model can make all subsets agree while predicting the measurements poorly.

## 1. Evidence, provenance, and historical boundary

This revision reconciles the [current status](CURRENT_STATUS.md), [polished frozen-model results](experiments/full_position/audit_polished_v1/RESULTS.md), the previous audit, and the current inversion/reporting implementation. The original numerical audit concerned commit `88f3ac5556f1a4aa3ca1e42db356853ab8c48f53`. Its report is preserved in [the pre-rewrite file at the evidence snapshot](https://github.com/rueijrwu/gaze_acc_joint_estimator/blob/c2584292d13393a4c5a8bb34afdea9c19f90e67f/AUDIT_REPORT.md).

The original audit's targeted numerical findings and commit are historical evidence, not the provenance identifier for the later Phase 8.2 run. The three-way scorecard has a separately versioned post-processing result in [crosscheck_v1](experiments/full_position/crosscheck_v1/RESULTS.md), computed from frozen frame and holdout records without recalibration, refitting, or prediction. The original 21-test count remains historical evidence; the Phase 8.1 42-test verification and source-tree check are recorded in [crosscheck verification](experiments/full_position/crosscheck_v1/verification.json). Phase 8.2's separate 52-test check and joint-refit provenance are recorded in its [verification file](experiments/full_position/joint_sensitivity_v1/verification.json) and [config](experiments/full_position/joint_sensitivity_v1/config.json), including exact copies of the design documents used during that run.

The old report's recommendations to implement polishing and fix basic reporting are resolved. The saved-record reporting step in Section 8.1 and joint latent-state/coefficient sensitivity in Phase 8.2 are complete; see Sections 8 and 9. Phase 8.3 controlled information ablations and later scientific experiments remain open. Completing the scorecard does not complete those studies.

Original `grouped_v2` outputs remain historical. Use `audit_polished_v1` for the latest fixed-model comparison; do not silently combine different supports or original and revised branches.

## 2. What is measured, estimated, and cross-checked

### 2.1 All three pairs remain in the experiment

After applying correspondence metadata, write the measured positions as

$$
\mathbf p_j=\mathbf P_{1,j},\qquad
\mathbf q_j=\mathbf P_{4,\pi(j)},\qquad j=1,2,3.
$$

The recorded zero-based correspondence is `pair_index=[2,1,0]`. Coordinates and validity flags must be reordered together. The reference quantities are

$$
\mathbf c_1=\frac13\sum_j\mathbf p_j,\qquad
\mathcal A_1=\frac12\left|\det[\mathbf p_2-\mathbf p_1,\mathbf p_3-\mathbf p_1]\right|,
\qquad \ell_1=\sqrt{\mathcal A_1},
$$

$$
\mathbf r_j=\frac{\mathbf p_j-\mathbf c_1}{\ell_1},\qquad
\mathbf v_j=\frac{\mathbf q_j-\mathbf c_1}{\ell_1},\qquad
\mathbf u_j=\mathbf v_j-\mathbf r_j.
$$

The required summaries remain exactly

$$
\boxed{d_x=\frac{c_{4,x}-c_{1,x}}{\sqrt{\mathcal A_1}},\qquad
\rho_4=\frac{\mathcal A_4}{\mathcal A_1}.}
$$

Area means the triangle formed by reflection centers, not blob area. Keep this normalization while isolating the value of additional coordinate information.

The current nominal gaze targets are **-10, -5, 0, 5, 10 degrees**. New models use `theta_deg/10`. The frozen baseline retains its original `theta_deg/15` coefficient convention; 15 is not a calibration endpoint. The computational box `theta in [-20,20]`, `A in [0,6]` is not a statement of calibrated coverage. Sources: [Theory.md](Theory.md), [ESTIMATOR_PLAN.md](ESTIMATOR_PLAN.md), and [current status](CURRENT_STATUS.md).

### 2.2 One state predicts different pair responses

The implemented conditional prediction is

$$
\boxed{\widehat{\mathbf q}_j(x)
=\mathbf c_1+\ell_1[\mathbf D(x)+\mathcal T(x)\mathbf r_j]},
\qquad x=(\theta,A).
$$

The coefficients defining `D` and `T` are shared across calibration observations and frozen during application. Only two state components vary in a frame; there are no freely fitted framewise displacement/deformation coefficients.

P1 and P4 sample different, state-dependent images of the source pattern. Expected unequal magnification, orientation, and shape changes belong in the prediction. The residual is disagreement with the **predicted distorted pattern**, not departure from a similarity transform. Three sampled pairs admit an effective affine representation without establishing a globally affine optical mapping or identifying unique physical aberration coefficients.

The model implies `dhat=D` and `rhohat=abs(det(T))`. Area and centroid summaries must not be appended as independent observations to the six coordinate residuals. The predictive formula uses the P1 centroid, not the measured P4 centroid.

### 2.3 What the conditional model does not claim

Twelve raw coordinates have nine continuous geometric degrees of freedom after removal of common translation and positive scale on a nondegenerate branch. The current model treats normalized P1 geometry as noisy measured context and predicts six P4 components. It is a **six-residual conditional model**, not nine independent state observations or twelve independent residuals.

All three P1 points are used, but this procedure does not separately hold out and validate every P1 point. Shared P1 error can affect all three cross-checks. The optional joint nine-component model, or a genuine P1-point holdout with a different reference construction, is a separate experiment; neither is required to define the current P4 test.

## 3. Primary experiment: three-way P4 cross-prediction

For each evaluation frame, freeze the calibrated model and perform all three tests:

| Test | P1 reference | P4 inputs to the state solve | Predicted/tested measurement |
|---|---|---|---|
| Exclude P4_1 | All three P1 points | P4_2 and P4_3, x and y | P4_1, x and y |
| Exclude P4_2 | All three P1 points | P4_1 and P4_3, x and y | P4_2, x and y |
| Exclude P4_3 | All three P1 points | P4_1 and P4_2, x and y | P4_3, x and y |

**Two retained P4 points are used only inside one cross-check. The experiment still has all three pairs, and every P4 point is tested in turn.** All P1 points remain available, so the prescribed normalizer is unchanged.

Let `I_j` contain the four coordinates of the other two P4 points. Solve

$$
\widehat x_{-j}=\arg\min_{x\in\mathcal B}
 e_{I_j}(x)^\top R_{I_jI_j}^{-1}e_{I_j}(x),
\qquad e(x)=v-F(x,r),
$$

then, without changing the selected state/branch using the excluded observation, compute

$$
\boxed{\mathbf e_j^{\rm cross}
=\mathbf q_j-\widehat{\mathbf q}_j(\widehat x_{-j}).}
$$

This is the relevant cross-check agreement between measurements: a state supported by the retained measurements predicts the excluded response. Four retained scalar observations can locally identify two states, but subset rank, conditioning, and branch uniqueness must be checked, not assumed.

### 3.1 Two different holdouts must remain distinct

An **outer fixation/gaze/capture holdout** keeps evaluation observations out of coefficient fitting, noise estimation, prior construction, and model selection. The **within-frame P4 holdout** keeps one measurement out of that frame's state solve. A strong predictive test uses both.

An in-sample P4 exclusion can still reveal internal inconsistency, but its global calibration may already have seen the tested frame. Do not label that as the same evidence as prediction on a genuinely withheld condition.

### 3.2 Leakage and branch rules

The excluded P4 point must not influence its subset solve through the full P4 centroid, area, observed map, all-three initializer, weights, validity gates, state penalties, or branch choice. Use the retained covariance marginal before whitening. Changing the excluded raw coordinate must leave preprocessing, starts, subset branches, and predictions unchanged until scoring.

**Do not select a subset branch by making it agree with the other subset estimates or the all-three estimate.** Those estimates generally use the excluded P4 point, so that would leak it back into its own test. State agreement is evaluated after independently freezing the subset results.

Ambiguous subsets retain their prediction sets and are marked inconclusive for a single-prediction score. Weakly identified subsets are not evidence that the excluded point is wrong. A branch selected using all measurements may be useful operationally, but it is a different result and not an untouched holdout.

The checks are conditional on stored detections. Upstream detector selection can already share point history or geometric assumptions; leave-one-P4-out inference does not make the image-localization pipeline independent.

## 4. Metric contract: prediction, state agreement, and coverage

The current code reports point errors and retains subset states. It does not yet provide the explicit per-frame three-way shared-state scorecard specified here. These are reporting additions to derive from saved outputs first, not a request to retrain or force subset agreement.

### 4.1 Point-prediction agreement

For a frame with three testable, unambiguous checks, report

$$
E_{\rm cross,px}
=\sqrt{\frac13\sum_{j=1}^3\|\mathbf e_j^{\rm cross}\|^2},
\qquad
E_{\rm cross,norm}=E_{\rm cross,px}/\ell_1,
$$

$$
E_{\max,px}=\max_j\|\mathbf e_j^{\rm cross}\|,
\qquad
E_x=\sqrt{\frac13\sum_j(e_{j,x}^{\rm cross})^2},\quad
E_y=\sqrt{\frac13\sum_j(e_{j,y}^{\rm cross})^2}.
$$

`E_cross,px` is RMS of three **2D point distances**, not coordinate-component RMS; its divisor is 3, not 6, and `E_cross,px^2=E_x^2+E_y^2`. Keep signed point/axis errors to distinguish repeatable bias from random spread. Use pixel and normalized versions together; numerical thresholds must identify which units they use.

Across a fixed complete-frame set, the pooled point RMS is `sqrt(mean(E_cross,px^2))`, not `mean(E_cross,px)`. Also report medians, tails, and results by point, gaze, fixation, and capture/demand. Preserve the original pooled statistic for reproducibility and add a clearly labeled equal-fixation aggregate so unequal valid-frame counts do not silently change weighting. Record membership for every aggregate.

### 4.2 Shared-state agreement

The three tests produce

$$
\widehat x_{23}=\widehat x_{-1},\qquad
\widehat x_{13}=\widehat x_{-2},\qquad
\widehat x_{12}=\widehat x_{-3}.
$$

For three identifiable, unambiguous results, define descriptive centers

$$
\bar\theta=\frac13\sum_j\widehat\theta_{-j},\qquad
\bar A=\frac13\sum_j\widehat A_{-j},
$$

and report separately

$$
\boxed{S_\theta=\sqrt{\frac13\sum_j(\widehat\theta_{-j}-\bar\theta)^2}},
\qquad
\boxed{S_A=\sqrt{\frac13\sum_j(\widehat A_{-j}-\bar A)^2}}.
$$

Also report the ranges `max(theta_-j)-min(theta_-j)` and `max(A_-j)-min(A_-j)`, every signed pairwise state difference, and subset-versus-all-three differences. The all-three estimate is an operational reference, not ground truth. The descriptive centers above are not a replacement estimator.

`S_theta` is in degrees and `S_A` in diopters. Do not add their raw squares into one unitless score. A combined score requires predeclared physical reference increments and separate component reporting. The divisor 3 describes dispersion of these three results; it is not an independence-based variance or standard-error estimate.

These inverses share P1 and overlapping P4 measurements, so their errors are correlated. For example, uncertainty of a difference requires

$$
\operatorname{Cov}(\widehat x_a-\widehat x_b)
=C_a+C_b-C_{ab}-C_{ba}.
$$

Do not add two marginal state covariances and silently assume zero cross-covariance. Simple descriptive differences can be reported now; calibrated statistical consistency probabilities require the joint error model, calibration uncertainty, and model discrepancy.

A separately inverted single pair could be another diagnostic only when its two coordinate responses identify both states. The current reference intentionally uses two retained P4 points for redundancy; it does not claim that every single pair has a regular two-state inverse.

### 4.3 Coverage and failure are inseparable from agreement

Retain the entire preselected population, all raw-valid subsets, and all solver outcomes. Report counts of complete three-check frames, partial checks, missing geometry, numerical failure, weak rank, ambiguity, bound-active solutions, and state/context extrapolation. A missing or ambiguous check is not zero error and not successful agreement.

A frame missing any of the three tests has no complete-frame `E_cross`, `S_theta`, or `S_A`; keep its individually available errors with explicit partial support. Do not silently recompute a three-way score from two surviving tests.

Bound-active unique predictions may retain geometric errors, but must have a separate stratum. Shared clipping at `A=0` or `A=6` can manufacture small state spread. Local Gaussian scores must remain unavailable where their regular-interior assumptions fail. Declare physical-unit sensitivity and conditioning policies before judging state agreement; rank two alone need not mean useful precision.

For the saved study, 429 scored point tests refer to the 143 complete valid rows per split family. Each family originally selected 160 rows; 17 invalid rows remain part of coverage reporting. Do not present 429/429 valid-point scoring as complete coverage of every originally selected frame.

### 4.4 Metrics that support, but do not replace, the cross-check

All-three fitted residuals describe how well all observations can be fitted together, not untouched prediction. Retained-subset optimization cost assesses the solver's own inputs, not the excluded measurement. Nominal fixation means assess calibration conventions and plausibility, not independently measured accommodation. Small uncertainty estimates, smooth traces, or agreement with the baseline are not standalone success criteria.

Select with a scorecard: predictive agreement, state consistency, testable coverage, identification, and robustness across grouped conditions. Do not train a constant/overconstrained state response, impose equality between subset estimates, or apply strong state priors merely to reduce the displayed agreement metric. Calibration anchors still provide approximate physical state meaning; they are not discarded, but their influence must be measured.

## 5. Latest results under the corrected objective

The [frozen-model reevaluation](experiments/full_position/audit_polished_v1/RESULTS.md) reused all 27 saved models, their coefficients/covariances, and the original sampled populations. No retraining was performed. The following are saved results, not recalculated metrics from this rewrite.

| Split family | Common scored P4 tests | conditional27 cross-prediction RMS, px | conditional37 cross-prediction RMS, px |
|---|---:|---:|---:|
| Held-out gaze condition | 429 | **3.423** | 6.620 |
| Held-out capture/demand | 429 | **4.062** | 4.353 |

Both coordinate models now score all 429 valid-population tests per family. `conditional37` full-frame coverage recovered from 139 to 143 gaze frames and from 142 to 143 capture frames. Previously reported 426-point comparisons must not be mixed with this expanded support. On its unchanged 426-point gaze support, `conditional37` RMS changed from 6.197 to 6.632 px; support expansion does not explain that deterioration.

**The relevant comparison favors `conditional27` because it predicts excluded measurements better, not because it is closer to nominal accommodation demand.** It is the current development reference, not a physiologically validated winner or proof that extra curvature is always harmful.

The saved-record three-way state disagreement is now reported alongside coordinate prediction in [crosscheck_v1](experiments/full_position/crosscheck_v1/RESULTS.md). On the full eligible population and exact matched interior population, `conditional27` has lower equal-fixation RMS for both state components in both split families. These are descriptive, correlated subset comparisons with no calibrated agreement tolerance; they do not establish physiological accuracy.

### 5.1 Better retained fit can mean worse cross-check agreement

Polishing recovered eight lower-cost gaze-subset branches. One improved the excluded-point prediction; seven worsened it. Six worsened predictions concentrate in the capture-2, -10-degree fixation, with errors approximately 23.9-25.3 px.

The original targeted success is preserved: capture 1, row 1380, excluding zero-based P4 index 1, improved from **5.959 to 2.695 px**, while retained-coordinate cost fell from **26,295.893 to 24,441.697**. But that one case cannot represent the other transitions.

The lesson is **not** to retain a known inferior numerical solution because its withheld prediction happens to look better. The corrected solver more reliably exposes that the calibrated response and retained observations can support a state inconsistent with another measured point. Keep the honest worse prediction, then investigate the response/calibration assumptions. A lower training or retained cost is not the cross-check metric.

Independent profile checks agree with the eight recovered transition minima within a reported maximum cost difference of `4.1e-10`. Finite profile grids and local certificates still do not prove global inverse completeness.

### 5.2 Nominal-anchor comparisons have a secondary role

The latest common-support gaze-fold fixation-mean discrepancy RMS is 0.761/0.748/0.564 degrees and 1.135/1.407/0.536 D for `conditional27`/`conditional37`/`two_channel13`, respectively. These numbers belong in the calibration-diagnostic section, not as the primary shared-state success ranking.

The two-channel control is not an individual-coordinate predictor and has no directly comparable held-out-P4 error in these tables. It cannot be declared the cross-check winner from its smaller demand discrepancy or near-zero two-channel residual. Conversely, coordinate models have not established superior absolute state accuracy merely by offering a cross-check the control lacks.

### 5.3 What has improved and what remains unproven

**Demonstrated in the saved audit:** numerical certification and valid-population coverage improved, and the cross-prediction diagnostic is operational. **Favored on the current development comparison:** `conditional27` over `conditional37` for excluded-point prediction. **Not established:** population-level benefit of each added coordinate, calibrated pass/fail thresholds, accurate localization of a faulty point, or independent physiological gaze/accommodation accuracy.

The within-frame tests are correlated. The gaze and capture split families reuse observations. Endpoint gaze holdouts are extrapolations relative to their remaining anchors. Do not pool all point tests as independent recording replications or infer that poor endpoint holdout performance proves failure after training on the full five-target grid.

## 6. Implementation compliance and resolved audit findings

The mathematical core remains aligned with the theory: correspondence-safe six-point input, prescribed area normalization, shared two-state coordinate prediction, state-dependent deformation, fold-local calibration, correlated shared-P1 residual propagation, and leakage-safe P4 exclusion. The optional joint P1-response model remains deferred rather than accidentally omitted.

| Original finding | Status at the evidence snapshot | Remaining boundary |
|---|---|---|
| Better candidates discarded before certification | Addressed: candidates retained and bounded exact-Hessian polishing precedes acceptance/clustering | Maintain independent branch checks; no global completeness claim |
| Per-frame versus fixation-mean metrics confused | Addressed in `validate.py` and `report.py` | Preserve aggregation labels and matched-support membership |
| Report could use a rejected start's cost | Addressed: selected and rejected costs separated | Preserve selection provenance in every new run |
| Compatibility validation and application diagnostics incomplete | Reported addressed in the polished audit; shared support/parity/uncertainty contract | Unavailable uncertainty components remain explicitly unavailable |
| Numerical regressions and convergence semantics incomplete | Earlier saved suite reported 21 tests; the current full pytest suite passes 42, including cross-check contract and perturbation regressions | Retain certification and leakage checks for future changes |

Sources: [current inversion](full_position/invert.py), [reporting](full_position/report.py), [evaluation summaries](full_position/validate.py), and [latest audit results](experiments/full_position/audit_polished_v1/RESULTS.md).

The exact-Hessian and polynomial-profile findings from the original audit remain useful numerical safeguards. At fixed accommodation/context, the implemented gaze response is cubic or lower, so the fixed-weight objective is at most degree six in scaled gaze. Stationary-root enumeration plus accommodation refinement provides an independent targeted check; it must not be relabeled a complete global certificate.

The earlier implementation audit reports byte-identical copies of 27 models, nine round-tripped populations, 199 matching source hashes, and 21 tests. The current cross-check verification reports 42 passing tests and an unchanged frozen source tree. These preservation checks establish reproducibility of the reported updates, not physiological correctness or adequacy of the learned response surface.

## 7. What the disagreement teaches us

### 7.1 Separate expected distortion from unexplained disagreement

The optical hypothesis remains appropriate: different source sites can have different gaze/accommodation responses. A correct P4 triangle need not be a uniformly scaled P1 triangle. Investigate error relative to predicted translation and deformation; do not gate away non-similarity or add unrestricted framewise affine parameters to make errors vanish.

The concentrated capture-2 endpoint errors are a useful diagnostic target. Examine signed point/axis residuals, all three subset states, branch transitions, and P1 context at those frame IDs. Neither the concentration nor a large single-point error alone identifies the cause: calibration/extrapolation, response capacity, capture-dependent geometry, and detector selection can all contribute.

### 7.2 Measurement noise is not total prediction uncertainty

The implemented shared-input covariance propagation is consistent with the conditional model. Its short-timescale noise input does not include all response discrepancy or global coefficient uncertainty. Revised median squared Mahalanobis scores remain approximately 1,531/2,315 in gaze folds and 2,521/2,743 in capture folds for `conditional27`/`conditional37`. They are not calibrated artifact probabilities.

Report raw predictive disagreement before deciding how to model its uncertainty. Do not enlarge noise until every cross-check passes, nor discard high-error evaluation rows and report only survivors. Uniform covariance scaling leaves an otherwise fixed, no-prior framewise minimizer unchanged, while calibration anchor/prior tradeoffs can change. Noise propagation, noise assumptions, and calibration uncertainty are separate issues.

### 7.3 Current sensitivity results have a narrower meaning than a calibration audit

Saved training diagnostics report median absolute signed group bias 0.497 px and a 95th percentile of 1.845 px. These are aggregated in-sample fixation/point/axis biases with observations reused across folds, not directly comparable to pooled held-out 2D point RMS and not independent evidence of generalization.

The earlier saved coefficient-sensitivity refits hold latent states fixed. A tenfold prior-strength change modifies training predictions by 0.053-0.142 px RMS for `conditional27` and 0.083-0.174 px for `conditional37`; changing reference-state quartiles gives smaller reported changes. Those fixed-state results do not establish that jointly refitted states, anchors, or cross-prediction results are insensitive. The completed Phase 8.2 joint refits are summarized below and in the [versioned results](experiments/full_position/joint_sensitivity_v1/RESULTS.md).

The two original difficult audit frames had well-conditioned P1 triangles, so they did not motivate replacing square-root area first. That local observation does not rule out normalization sensitivity elsewhere. Preserve normalization in the next controlled comparison.

## 8. Completed reporting update and next experiments

### P0 — Three-way agreement scorecard (complete)

The versioned [crosscheck_v1 report](experiments/full_position/crosscheck_v1/RESULTS.md) joins frozen holdouts by split, fold, model, capture, fixation, row, and excluded point. It reports per-point errors and support, complete-frame prediction/state metrics, scheduled coverage, bounds, and full and exact matched interior populations. The [summary JSON](experiments/full_position/crosscheck_v1/crosscheck_summary.json) retains exact membership IDs; [verification](experiments/full_position/crosscheck_v1/verification.json) records source immutability and the 42-test run. This is post-processing only: no predictions, branch selection, or refits were rerun.

Each family/model has 160 scheduled frames and 480 scheduled point slots, with 17 invalid frames retained as 51 unscored slots; the valid population has 143 complete frames and 429 scored points. Equal-fixation complete-frame `E`/`G_theta`/`G_A` RMS values are 3.421 px/0.269°/0.311 D and 6.700 px/0.509°/0.929 D for gaze `conditional27`/`conditional37`; capture values are 4.014 px/0.281°/0.319 D and 4.307 px/0.325°/0.392 D. On the exact shared interior complete-frame support (110 gaze, 125 capture frames), the corresponding values are 3.198/0.241/0.258 versus 4.699/0.408/0.377 for gaze, and 3.924/0.276/0.317 versus 4.097/0.339/0.413 for capture. See the linked report for boundary summaries, pooled point RMS, and matched point-level masks.

The report keeps prediction error and state disagreement as separate metrics. It does not introduce a pass threshold or combine degrees with diopters. Shared clipping can produce small state spread while coordinate prediction remains poor.

### P1 — Joint calibration sensitivity (complete; reported as Phase 8.2)

The [Phase 8.2 study](experiments/full_position/joint_sensitivity_v1/RESULTS.md)
refit latent states and coefficients over eight predeclared anchor, prior, and
reference-covariance settings on the frozen grouped folds. Its
[verification record](experiments/full_position/joint_sensitivity_v1/verification.json)
retains 144 task outcomes, including one uncertified calibration and its failed
checkpoint; the report separates all-testable and exact matched-interior
support. Strong anchors with `conditional37` lower all three full-population
metrics in both split families at the same 429 scored slots, and also improve
the capture/37 exact shared-interior metrics. Other settings show tradeoffs;
there is no final prior choice or deployment model selected from these
development folds. The results preserve the corrected solver and all failed
starts/checkpoints. The [paired-interior plot](experiments/full_position/joint_sensitivity_v1/paired_interior_vs_baseline.png)
uses each comparison's own matched complete-interior frames; the
[calibration diagnostics table](experiments/full_position/joint_sensitivity_v1/calibration_diagnostics.csv)
preserves the selected start and weighted objective components for all 144
tasks. Do not use these evaluation outcomes to set thresholds or branch rules
for final claims.

The supplemental numerical stop check continued the selected failed
`prior_weak/capture_1/conditional27` training trajectory for 100 additional
evaluations with `gtol` disabled. Physical projected stationarity decreased
from `0.0031837354` to `0.0018793984` but remained above the `0.001` gate; the
fit remains unaccepted. The rejected continuation is preserved in
[supplemental_stop_checkpoint.json.gz](experiments/full_position/joint_sensitivity_v1/supplemental_stop_checkpoint.json.gz),
with its summary in
[supplemental_stop_check.json](experiments/full_position/joint_sensitivity_v1/supplemental_stop_check.json).

### P2 — Phase 8.3 controlled information tests without leakage

The saved same-response ablations are useful but targeted: on two audit rows, coordinate versus derived-summary inversion changed the selected states substantially; retained x/y improved five of six excluded-point predictions relative to retained x only and worsened one. That is not yet a population conclusion.

Extend the retained-x versus retained-x/y comparison across development populations using the same frozen calibrated response, with rank and coverage checked for both. Both methods must exclude the tested P4 point. A derived full-triangle area that contains that point is not a valid input to its holdout inverse.

Use full-coordinate versus same-response centroid/area inversion separately as a measurement-compression/state-sensitivity experiment. It does not supply an independent held-out-P4 baseline when the summaries contain the excluded point. Declare model-capacity and weighting changes rather than attributing all differences to extra geometry.

### P3 — Add only response/context terms supported by residual evidence

After P0-P2, inspect reproducible signed-gaze curvature, accommodation coupling, and P1-context dependence. A targeted mixed term such as `t^3 log(1+A)` in common displacement is an example of a hypothesis to test, not an established missing optical law or a predetermined fix. Compare it within grouped training; do not expand every basis indiscriminately or fit arbitrary per-frame distortion states.

Audit stored detector selection and capture/source/eye geometry. A pattern selected under geometric constraints may already have its deformation biased or truncated. This remains a measurement-pipeline question, not a demonstrated detector bug. Capture and demand are confounded in the current recordings.

### P4 — Denser grouped evaluation and final transfer

Use nested grouped selection and denser evaluation, distinguish interpolation from endpoint extrapolation, and keep fixations/recordings rather than adjacent frames as grouping units. Predeclare tolerable predictive errors, state disagreement, and coverage tradeoffs using development evidence or an experiment-specific precision requirement; no calibrated acceptance threshold is supplied by the present results.

Keep captures 5/6 untouched by the full-position study until choices are frozen. Their lack of independent target/reference labels does not prevent predictive geometry checks, but does prevent claiming physiological error from those checks alone. Independent reference acquisitions are a separate requirement for absolute physiological accuracy, not a prerequisite for improving the internal cross-check.

## 9. Final interpretation

**The purpose of three pairs is that the same gaze and accommodation must explain several different measured responses.** The primary improvement criterion is better prediction of each excluded P4 measurement; the companion criterion is agreement among the states inferred from different retained subsets, with adequate identification and coverage.

The current implementation already performs the essential P4 cross-prediction. Its numerical certification is materially improved. `conditional27` currently cross-predicts better than `conditional37`, but their remaining errors and uncertainty mismatch require further diagnosis. Better solver cost, better nominal-demand agreement, and better cross-measurement agreement are different claims and must remain separate.

Phase 8.2 has completed the joint calibration sensitivity refits. The next
planned work is Phase 8.3/P2: extend controlled information tests across
development populations, with rank, coverage, and excluded-point leakage
checked. Keep response/context hypotheses and denser transfer evaluation in
their stated order. Do not use current evaluation errors to select priors or
thresholds.

## Source map

Historical numerical facts refer to the commit identified above; Phase 8.2 uses the linked run configuration, source hashes, and pre-run design snapshots. Relative links provide navigation.

- [CURRENT_STATUS.md](CURRENT_STATUS.md): latest execution status, fixes, verification, and remaining work.
- [Theory.md](Theory.md) and [ESTIMATOR_PLAN.md](ESTIMATOR_PLAN.md): normalization, source-pattern interpretation, model scope, calibration, and holdout contracts.
- [audit_polished_v1/RESULTS.md](experiments/full_position/audit_polished_v1/RESULTS.md): latest paired/expanded-support errors, state diagnostics, sensitivity, and verification boundaries.
- [comparison.json](experiments/full_position/audit_polished_v1/comparison.json) and [verification.json](experiments/full_position/audit_polished_v1/verification.json): saved transition and preservation evidence.
- [full_position/invert.py](full_position/invert.py): retained candidates, polishing, subset prediction, and separate scoring.
- [full_position/validate.py](full_position/validate.py) and [report.py](full_position/report.py): implemented aggregates and the reporting extension point.
- [audit_checks_v1](experiments/full_position/audit_checks_v1/): targeted profiles, controlled measurement ablations, and conditional training-sensitivity evidence.
- [Phase 8.2 joint-training sensitivity](experiments/full_position/joint_sensitivity_v1/RESULTS.md), [verification](experiments/full_position/joint_sensitivity_v1/verification.json), [configuration and run hashes](experiments/full_position/joint_sensitivity_v1/config.json), archived [Theory.md](experiments/full_position/joint_sensitivity_v1/design_snapshot/Theory.md) and [ESTIMATOR_PLAN.md](experiments/full_position/joint_sensitivity_v1/design_snapshot/ESTIMATOR_PLAN.md) design copies, and the [paired-interior figure](experiments/full_position/joint_sensitivity_v1/paired_interior_vs_baseline.png).
- [grouped_v2/RESULTS.md](experiments/full_position/grouped_v2/RESULTS.md): archived original results, not the latest corrected population.
- [Original audit before this rewrite](https://github.com/rueijrwu/gaze_acc_joint_estimator/blob/c2584292d13393a4c5a8bb34afdea9c19f90e67f/AUDIT_REPORT.md): historical numerical findings and targeted reconstruction methods.

The three-way metric definitions and results are in the linked `crosscheck_v1`
report. Joint calibration sensitivity is reported in the completed Phase 8.2
study; Phase 8.3 information ablations and physiological accuracy remain
untested. No physiological accuracy claim is made here.
