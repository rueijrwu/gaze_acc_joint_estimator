# Subplan 5 — Test full-model agreement; decide whether an extension is warranted

**Covers:** S8/G8 and optional GX. **Status:** G8 COMPLETE_WITH_LIMIT / GO_WITH_LIMIT; `comparison_complete=false`.
**Requires:** a frozen certified full DM0 model, complete input schedule, and G7's unresolved-assumption list.  
**Parent:** [Gate policy](../STAGE_GATES.md); [mathematics](../Theory.md).

## Executed G8 outcome

G8 was authorized and completed on the full reviewed schedule using the frozen certified Stage 4 attempt04 model. The result is COMPLETE_WITH_LIMIT / GO_WITH_LIMIT for the empirical cross-check; it is not a candidate-model comparison and does not establish physiological accuracy. There are 100,090 scheduled rows, 300,270 held-point slots, and 89,175 complete eligible triples across all 20 exposures. Equal-exposure E/Gtheta/GA are 12.40853 px / 0.477866 degrees / 1.101718 D. The three omissions show materially different state agreement, especially omission of P4_1; conditional accommodation information for omission of P4_2 is weak. See [Stage 5 results](../../experiments/distortion_model/stage_05_crosscheck_and_model_decision/docs/RESULTS.md) and the saved [attempt report](../../experiments/distortion_model/stage_05_crosscheck_and_model_decision/results/attempt_01/REPORT.md).

Alignment and identifiability remain unresolved. One operational-reference sensitivity comparison is proposed using the adjacent P4 omega4 convention documented in the scientific review; it was not executed or authorized by this G8 result.

## S8 / G8 — Do different measurements support the same instantaneous state?

For every scheduled frame, omit P4_1, P4_2 and P4_3 in turn. Each solve retains all three P1 points and only two P4 points, recomputes g at its trial visual gaze, and runs the complete estimator. Freeze all global coefficients, offsets, templates, weighting, reference gauges and model definition.

Select covariance marginals before whitening. Do not use the measured full P4 centroid, its full-fit state, area, unmasked warm start or omitted-point-dependent validity/weight for the subset inference. Perturb the omitted point with calibration fixed to verify noninterference through the entire loop. All-three **predicted** means are allowed.

Score in original relative camera units:

`e_j = (q_j-c1) - g_subset * (D + F4_j - mu1)`.

Keep one outcome for every expected frame and slot, including `held_point`, identifiers, unavailable/failure reason and numerical status. A missing measurement, ambiguous inverse, boundary solution, numerical failure and model mismatch are different conditions, not one rejected-row category.

## Minimum result, not just a scalar ranking

Report E_cross, G_theta and G_A on complete eligible triples, with within-exposure squared averaging followed by equal exposure weights. Retain partial-slot summaries separately. State agreement compares three estimates **of the same frame** in visual gaze and diopters, not nominal labels or different timestamps. RMS is descriptive, not an accuracy gate.

The report needs five compact views:

1. Coverage/status by all twenty exposures, point and model; both scheduled and usable denominators.
2. Native-pixel cross-error, signed x/y biases, each P4, tails/worst point, and exposure contributions.
3. Same-frame state disagreement with bounds, branch/rank information, optical response derivatives and accommodation–gaze separation.
4. Raw centroids, mean distortion correction and center polynomial; P1 g and residual associations at comparable gaze/demand.
5. The unresolved-assumption ledger from G2–G7, updated with supporting/contradicting empirical evidence and explicit remaining confounding.

A curve against inferred A is model-dependent. Include the corresponding raw geometry versus recorded conditions. Associations with g can indicate a normalization issue, but g, gaze, accommodation, time and capture may be correlated. Neither such an association nor its absence alone proves or disproves equal P1/P4 axial scaling. No free Z or second P4 scale is introduced to force agreement.

## G8 decision

**GO / supported effective response:** the check is complete, all expected exposure counts are visible, same-frame agreement and response information make DM0 a defensible effective description within its fixed conventions, and no specific unresolved contradiction prevents that interpretation. State the scope and remaining uncertainty. A particular RMS value is not required.

**GO_WITH_LIMIT / tradeoff or model_assumption_unresolved:** calculations are valid but some references, scale transfer, response components, conditions or states remain weakly determined. Report the result as completed internal analysis with limitations, not a universal winner. Small G_A caused by compression/clipping is not a positive finding. A low pixel error cannot rescue unobservable A.

**REPAIR:** leaks, missing slots, wrong units/offsets, mismatched populations or faulty aggregation. Rerun affected inference/scoring from the same compatible frozen model; do not refit unchanged coefficients unless the defect affected calibration.

**PAUSE:** valid computation reveals no usable shared-state inference under the declared assumptions, or coverage cannot support the intended all-exposure conclusion. Preserve the negative/incomplete result and propose one discriminating check. This can complete the experiment without supporting the optical model.

No requirement exists that every frame, exposure or companion metric improve. Use signed residual structure, conditional information, actual effect size and concentration of gains rather than a universal percentage cutoff. Empirical uncertainty estimates must acknowledge correlated frames and overlapping subsets; the three omissions are not independent experiments.

## Optional GX — Which single extra mechanism earns implementation?

DM1 is not automatically required after DM0. Open an extension only after a completed G8 report (or a separately approved model change resolving an earlier structural block) states:

**Observed signature:** which points/axes/conditions disagree, how much, and under which actual frame identities?  
**Hypothesis:** what single mechanism predicts that signature?  
**Competing cause:** which reference, numerical, scale, detector or capture explanation was checked?  
**Identifiability:** does the extra response remain distinct after allowing scale, gaze, center separation and existing parameters?  
**Decision:** what matched result would support the addition, and what result would leave DM0 preferable or the mechanism unresolved?

For DM1, calculate the centered radial response and its separation from effective magnification and allowed displacement/gaze directions at the actual source geometry. Equal or near-equal radii can make the radial column redundant. If so, record `radial_unidentifiable` and retain DM0; do not release a free center to manufacture rank. If distinct, add only the shared incremental kA term and run a **fresh full DM1 calibration** through the affected stages, with the same D degree, noise, anchors, populations and starts policy.

Compare each candidate against DM0 on exact paired frames and, where possible, one common complete-frame cohort. Retain all-scheduled coverage. Missing shared exposures make the all-exposure comparison incomplete; do not silently rank survivor means. Independently reconstruct squared-loss changes and show where they arise. Improved training fit alone does not support the radial interpretation.

DM2 requires the same one-mechanism argument. Gaze polynomial degree changes, nonlinear accommodation curvature, and an extra spatial term are different experiments; do not change them together. A broad power-law sweep, differential nuisance scale, optional P1 omission, second GPU backend or new calibration protocol is not an automatic prerequisite.

**Stop rules:** retain DM0 when its effective response is the best-supported conclusion; report an unidentifiable mechanism when the geometry cannot separate it; request matched additional evidence when needed. More parameters and more compute are not evidence by themselves. Physical accommodation accuracy still needs independent support beyond this internal calibration.

## Completion and recovery

Set `fit_complete`, `fit_certified`, `crosscheck_complete`, `comparison_complete` and the scientific disposition separately. Keep compact native-frame states, errors and status arrays plus the input/model hashes so summaries can be recalculated after a chat or machine switch. Do not delete the only evidence behind a score during cleanup.

G8 has completed with linked full-schedule evidence; its saved checkpoint records `crosscheck_complete=true`, `diagnostic_three_way_complete=true`, and `comparison_complete=false`. See the [Stage 5 execution report](../../experiments/distortion_model/stage_05_crosscheck_and_model_decision/docs/STAGE_REPORT.md) and [scientific review](../../experiments/distortion_model/stage_05_crosscheck_and_model_decision/results/attempt_01/SCIENTIFIC_REVIEW.md). Optional GX remains unrun. The unresolved assumptions and one proposed operational-reference sensitivity follow-up are recorded in the handoff.
