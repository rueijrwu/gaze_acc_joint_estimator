# Latest cross-check results audit

**Repository:** `rueijrwu/gaze_acc_joint_estimator`  
**Branch / inspected commit:** `exp5_full` / `bc08fb30734fd0ae70aeeaf5239a6d0de0eafb86`  
**Date:** 2026-10-07

## Executive assessment

The latest results justify making **baseline27 + x + differential-y** the next candidate to compare directly against the current **baseline27 + full x/y** development reference. They do not justify adding more polynomial capacity yet.

The scientific criterion remains the three-pair cross-check:

1. retain all three measured P1 points as context;
2. exclude one P4 point;
3. infer the shared horizontal-gaze/accommodation state from the other two P4 measurements;
4. predict both coordinates of the excluded P4 point;
5. rotate the exclusion through all three P4 points;
6. report excluded-point prediction together with disagreement among the three subset-inferred states, coverage, support, bounds and tails.

RMS is only an aggregation of these cross-check quantities. Training cost, nominal-demand discrepancy, or support percentages are not substitutes for the cross-check.

The transformed-measurement implementation is coherent and leakage-safe in the reviewed path. However, two reproducible population-handling defects were found in the new nested-selection coordinator. They should be corrected before a real nested calibration/model-selection study is run.

## 1. What the latest results support

The full-population scorecards compare the same 143 complete frames and 429 scored slots for baseline27 masks.

| Response / split | E cross-check, px | G_theta, deg | G_A, D | Worst point, px |
|---|---:|---:|---:|---:|
| baseline27 / gaze / xy | 3.421 | 0.2694 | 0.3111 | 4.385 |
| baseline27 / gaze / x+differential-y | **3.326** | 0.3708 | 0.3703 | **3.954** |
| baseline27 / capture / xy | 4.014 | 0.2806 | 0.3194 | 5.222 |
| baseline27 / capture / x+differential-y | **3.586** | **0.2782** | **0.3019** | **4.146** |
| strong-anchor37 / gaze / xy | 4.764 | 0.4397 | 0.5297 | 6.199 |
| strong-anchor37 / gaze / x+differential-y | **4.762** | 0.5831 | 0.7434 | **5.817** |
| strong-anchor37 / capture / xy | 4.293 | 0.2764 | 0.3665 | 5.593 |
| strong-anchor37 / capture / x+differential-y | **3.773** | **0.1820** | **0.2356** | **4.290** |

### Capture results

For baseline27 capture, differential-y improves excluded-point E by about 10.7% and worst-point error by about 20.6%, while slightly improving both subset-state disagreement measures.

For strong-anchor37 capture, differential-y improves all four displayed cross-check quantities. This is evidence that differential vertical geometry is useful, but it does not make conditional37 the overall reference because its coordinate error remains higher than baseline27 differential-y.

### Gaze results

For baseline27 gaze, differential-y improves excluded-point prediction and worst-point error but worsens subset-state agreement. For strong-anchor37 gaze, average coordinate prediction is almost unchanged while state disagreement worsens.

Because the frozen response and physical-state convention are unchanged within each mask comparison, these differences reflect the information supplied to the inverse rather than a recalibration of state units.

The correct interpretation is therefore:

> Differential-y provides useful geometric information. The additional common-y constraint can improve state agreement in some settings, but under the current response/weighting it can also pull the inferred state toward worse excluded-point predictions.

This is a tradeoff, not evidence that one scalar metric alone should select the model.

## 2. Direct differential-y versus xy comparison is still required

The saved follow-up reports each transformed mask against x-only. The next decision-relevant comparison should be

[
	ext{x + differential-y} quad	ext{versus}quad 	ext{full x/y}
]

on exact shared identities.

Report paired changes in:

- excluded-point squared vector error and E_cross squared;
- G_theta squared and G_A squared;
- x- and y-axis squared prediction error;
- worst-point and tail error;
- coverage, rank, ambiguity and bounds;
- empirical-state and measured-P1 support;
- signed gaze/capture/point strata.

Run this comparison on the full common population and on exact jointly interior / jointly supported populations. Do not rank candidates by comparing separately filtered cohort summaries with different memberships.

## 3. Important interpretation correction: retained-pair common-y is not pure translation

The implemented common-y observable for retained P4 points a,b is

[
y_c=rac{v_{a,y}+v_{b,y}}{2}.
]

With

[
v_{j,y}=D_y(	heta,A)+T_y(	heta,A)r_j
]

and (r_1+r_2+r_3=0), when point j is excluded,

[
oxed{
y_c=D_y-rac12 T_y r_j.
}
]

The retained differential observable is

[
oxed{
y_d=v_{a,y}-v_{b,y}=T_y(r_a-r_b).
}
]

Therefore retained-pair common-y contains both common displacement and deformation; its deformation contribution depends on which P4 point is excluded. Differential-y cancels common displacement and retains differential pattern deformation.

Consequences:

- poor common-y performance does not prove that common vertical image motion is inherently useless;
- it can reflect response-model error in D_y, deformation error, inappropriate weighting, capture/context dependence, or state confounding;
- differential-y's success shows that relative vertical deformation is informative under the current model;
- the calibration protocol's nominal vertical gaze of zero does not make image-y zero or irrelevant. The estimated state remains horizontal gaze plus accommodation.

## 4. Implementation audit

### Retained-information transform

The reviewed transformed-measurement path is mathematically consistent:

[
z=Hv_I,qquad hat z=HF_I(	heta,A),qquad R_z=HR_{II}H^	op.
]

The same transform is applied to observations, forward values, state Jacobians, state Hessians and the retained covariance. Both coordinates of the excluded P4 point are predicted afterward with the original response.

The combined common+differential-y representation is invertible with full retained x/y. The saved verification reports matching finite-multistart branch clusters over all 1,920 comparisons, with selected-state differences below about (5.3	imes10^{-8}) in physical units. This is strong implementation-consistency evidence, though not a proof of globally complete branch enumeration.

No new direct excluded-P4 leakage was identified in this path.

## 5. Nested-selection coordinator: two population-contract defects

These findings concern the future nested-selection mechanism, not the saved Phase 8.3 frozen-mask results.

### Finding A — Missing scheduled frames can disappear from the denominator

The nested coordinator verifies only that returned frames carry the requested fixation/group ID. The scorecard then defines its scheduled-frame denominator from the frames actually returned.

A targeted regression using the exact selector/scorecard logic returned one of eight scheduled frames for each validation group. The evaluator passed the group check. Guards requiring 100% scored and complete coverage also passed because the other seven scheduled frames were absent from the denominator.

The represented coverage was only 12.5%, yet the candidate remained eligible.

**Required fix:** provide the scorecard/selector an immutable scheduled-population manifest. Validate exact frame IDs and the expected three held-P4 slots before consulting errors. Missing outcomes must remain explicit unavailable records or cause validation failure; they must never disappear from the denominator.

### Finding B — Shared-frame ranking can silently lose an exposure

Each candidate can individually satisfy all-exposure coverage, yet their exact intersection of complete-frame IDs can remove an exposure if the candidates fail on different frames.

A targeted regression reproduced this: both candidates had scheduled/scored data in two exposures, but the common complete-frame intersection contained only one exposure. The selector still ranked them on that reduced cohort without flagging the lost exposure.

**Required fix:** preserve the original expected exposure set in the paired/shared-cohort scorecard. Require a predeclared minimum shared-frame fraction per exposure and overall. If an exposure disappears from the paired cohort, make the comparison ineligible or explicitly incomplete.

### Grouping should match the intended transfer question

The current generic nested coordinator leaves one supplied group out at a time. If the supplied groups are fixation IDs, inner selection is leave-one-fixation-out—not automatically whole-capture or whole-gaze-condition transfer.

For capture-transfer selection, use explicit whole-capture inner groups where calibration support permits. For gaze-transfer selection, hold the same horizontal-gaze condition across captures. The validation grouping must match the deployment/generalization question.

## 6. Support diagnostics: useful guard, not the scientific target

The latest support reporting is improved because every subset state is checked against its own training-state extrema and P1 context rather than copying support from the all-three estimate.

Still, componentwise empirical extrema are only diagnostics. They are not a validated multidimensional state/context envelope.

For example, baseline27 capture differential-y has fewer complete triples inside empirical-state support than xy. This is a real warning but does not establish that differential-y estimates are wrong. Whole-capture holdouts can deliberately create extrapolation relative to the remaining training demands.

Keep three concepts separate:

1. **Numerical validity:** certified, rank-2, unambiguous inverse.
2. **Calibration scope:** interpolation, intended extrapolation, or unexpected context shift.
3. **Cross-check performance:** ability to predict the excluded P4 measurement and maintain subset-state consistency.

Do not force estimates into empirical ranges merely to improve a support percentage; clipping can manufacture apparent agreement.

## 7. A targeted common-y discrepancy experiment

Do not add broad polynomial capacity yet. First test whether the common-y constraint is affected by a shared vertical discrepancy.

For normalized six-coordinate residual covariance R0, define

[
a=[0,1,0,1,0,1]^T
]

and a candidate covariance

[
oxed{
R_	au=R_0+left(rac{	au_{px}}{ell_1}ight)^2aa^T.
}
]

This represents an additional shared vertical discrepancy rather than independent localization noise.

- tau=0 recovers current x/y weighting.
- increasing tau reduces the influence of common-y while retaining x and differential-y.
- choose/estimate tau from training or inner-validation data only.
- continue scoring raw excluded-P4 coordinate prediction as the primary cross-check; covariance inflation cannot directly improve that raw score by definition.

This is a proposed model test, not an established correction.

There is also a useful algebraic interpretation. For the four retained coordinates, with (a_I=[0,1,0,1]^T), eliminating an unrestricted shared-y residual offset b,

[
min_b(e-a_Ib)^T R^{-1}(e-a_Ib),
]

is equivalent to evaluating the residual after a transform that retains x coordinates and differential-y. Thus the successful differential-y mask can be viewed as discarding one shared-y residual direction while preserving differential vertical deformation.

This does not prove that the real discrepancy is a free common offset. It motivates a small, interpretable continuum between full xy and removing common-y completely.

## 8. Training residual diagnosis before correction

The saved training residuals show capture-dependent mean y structure, approximately -0.848/+0.529/+0.324 px for baseline27 captures 2/3/4, with similar signs under strong-anchor37.

This is a clue, not proof of cause. Before changing covariance or response terms:

- regress signed training residuals against measured P1 context;
- inspect point-specific and signed-horizontal-gaze interactions;
- separate common and differential residual components;
- inspect whether capture effects remain after conditioning on P1 context and state;
- avoid free held-out-capture corrections because capture and accommodation demand are confounded in the present data.

If the residual is repeatable and predictable from measured context, a context-dependent response correction may be more appropriate than covariance inflation. If it behaves as unpredictable shared variation, the discrepancy-covariance candidate is more plausible.

## 9. Recommended next sequence

### P0 — Direct saved-result comparison

Compute differential-y versus xy on exact shared frame identities and shared support. No new fitting is required.

### P0 — Repair nested selection before using it

Add regression tests for missing scheduled frames and lost shared exposures. Introduce immutable population manifests and shared-cohort coverage guards.

### P1 — Training-only common-y diagnosis

Characterize common/differential residuals versus capture, horizontal gaze, accommodation state and measured P1 context.

### P2 — Small nested candidate set

Use cross-check selection on a predeclared candidate set:

1. baseline27 + xy — current reference;
2. baseline27 + x+differential-y — strongest current coordinate-prediction challenger;
3. baseline27 + justified shared-y discrepancy treatment — only if training evidence supports it.

Strong-anchor37/differential-y may remain a secondary challenger because its capture-state agreement is strong, but do not expand capacity simultaneously with changing the information/noise model.

### P3 — Independent evaluation

After candidates, guards, grouping and uncertainty policy are frozen, perform nested grouped development evaluation. Keep captures 5/6 untouched for later transfer checks. Historical development folds cannot become independent validation retroactively.

## Bottom line

The latest evidence supports a narrower conclusion than “full x/y is best”:

> **Differential vertical geometry is consistently informative. The present common-y constraint creates a prediction-versus-state-agreement tradeoff and deserves targeted diagnosis rather than automatic inclusion or deletion.**

Keep the excluded-P4 cross-check as the main scientific criterion. Use state agreement, axis/tail behavior, coverage, support and bounds as separate safeguards. Do not rank candidates by training fit, nominal-demand proximity, or candidate-specific surviving cohorts.

Before the next real nested study, fix the scheduled-population and shared-exposure contracts in the selector.
