# Subplan 4 — Correct centers and finish the first full DM0 calibration

**Covers:** S6/G6 and S7/G7. **Current status:** G6 GO_WITH_LIMIT; G7 attempt 04 COMPLETE_CERTIFIED_WITH_LIMIT / GO_WITH_LIMIT. G8 has not run and awaits separate authorization. See the [Stage 4 live progress pointer](../../experiments/distortion_model/stage_04_center_and_dm0_calibration/docs/PROGRESS.md) and [G7 attempt 04 report](../../experiments/distortion_model/stage_04_center_and_dm0_calibration/results/g7_attempt_04/STAGE_REPORT.md).
**Parent:** [Gate policy](../STAGE_GATES.md). **Requires:** compatible G0–G5 checkpoints.
**Equations:** [Theory](../Theory.md), corrected-center law and the single relative-coordinate objective.

## S6 / G6 — Are centroid shifts and optical-center separation accounted for once?

Use the current per-frame states and optical predictions. With mu1=mean(F1) and mu4=mean(F4), calculate

```text
D_obs = (c4-c1)/g - mu4 + mu1
h_cent_predicted = D(theta,A) + mu4 - mu1
```

Keep the P4-minus-P1 sign. The individual C1/C4 reconstructions are model-informed origins, not independently measured positions. Do not fit them as additional framewise states.

**Free block:** D's coefficients at fixed states and optical/reference parameters. Keep its declared visual-gaze degree and accommodation terms unchanged for this comparison. Degree 2 is the parent plan's first configuration; degree 1/3 are later named ablations, not a remedy applied only to a favored accommodation candidate.

Solve the linear D block against the **same complete relative residual and covariance**, not a second independent loss on D_obs. A center-corrected fit is algebraically related to its point fit; reducing that auxiliary discrepancy is not new evidence. Use the optical means and g from one immutable snapshot.

**Output:** one correction ledger per representative frame/exposure: raw centroid separation, P1 g, mu4-mu1, corrected D_obs, and D prediction. Plot its two image components versus visual gaze and demand. Report actual point-loss, anchor and regularization components before/after the D solve. Verify the center/centroid identity, sign, native units and polynomial origin conversion on synthetic cases.

**GO:** the predicted centroid response equals D+mu4-mu1 exactly, the weighted coefficient solve behaves correctly, and the complete model can explain its own bookkeeping without double-counting or a free center trajectory. Empirical residuals and polynomial adequacy remain for G7/G8.

**GO_WITH_LIMIT:** the polynomial or mean corrections have weakly identified terms. Fix/shrink unsupported terms within the declared policy; name the uncertainty to review after joint refinement. Do not insist that every corrected frame lies on its nominal fixation label.

**REPAIR:** wrong sign, shifted polynomial without coefficient conversion, independent h_cent and D fits, or acceptance based only on a separate center-loss decrease.

**PAUSE:** corrected-center interpretation changes arbitrarily with allowed template gauges or requires unsupported framewise centers. Revisit the earliest reference/geometry issue. Do not compensate by increasing gaze degree until the origin convention is fixed.

**Next:** full DM0 optimization. G6 does not claim improved physiological gaze accuracy.

## S7 / G7 — Can the complete DM0 model be fitted to all empirical calibration data?


### User-authorized G7 physical coupling bound

For G7 attempt 02, the user authorized a one-degree bound on horizontal accommodation-induced apparent gaze shift. At normalized \(g=1\), define \(H_x=D_x+\mu_{4,x}-\mu_{1,x}\). The continuous constraints are \(H_{x,\theta}\ge s_{min}\) and \(|H_{x,A}|\le s_{min}/4\), with \(s_{min}\) equal to half the outward-interval lower slope from G6 (10.027 reference px/degree). Interval constraints cover theta [-20,20] degrees and A [0,6] D; for the 0-to-4 D accommodation change they imply at most a one-degree inverse-equivalent horizontal shift where the monotone inverse remains in the declared theta domain. This is a user-assumed coupling bound, not a measured optical calibration or individual theta-error bound. It neither clamps theta labels nor adds a loss term. See the [bounded implementation](../../distortion_model/centroid_bound.py), [preflight record](../../experiments/distortion_model/stage_04_center_and_dm0_calibration/results/preflight_1degree/preflight.json), and [attempt 02 report](../../experiments/distortion_model/stage_04_center_and_dm0_calibration/results/g7_attempt_02/STAGE_REPORT.md).

This is the first full-calibration result, not another initializer. Use all valid frames in every declared full interval with the same source IDs, shared functions, two free states per frame, and finite full-fixation-mean anchors. Keep zero temporal-flatness penalty and the P1-only g definition. Refit global parameters and frame states; do not merely apply bootstrap states or a historical empirical fit.

Execute the existing S7 block order: accommodation, visual gaze with fresh scale, permitted P1/global parameters, P4 baseline, K4/omega4, and D. All proposals are judged against the same full J, with damping or joint refinement as needed. Compare one common initialized and one reproducible perturbed start. A warm-start from a preceding checkpoint is allowed only with matching provenance and does not certify a fit by itself.

The all-row objective must include the exact group sums/counts even when execution is chunked. Test small CPU and chosen-GPU loss/gradient agreement before launching expensive fitting. Chunking may reduce memory, not the scientific population. Avoid a second backend or dense global Jacobian unless a demonstrated need changes the approved implementation scope.

**Checkpoint evidence:** full loss decomposition, scaled projected stationarity and curvature/branch checks from the parent plan; parameter and state updates; scale distributions; boundary/rank/ambiguity counts; all-row counts per exposure; actual compute time/memory; M(A) and native/common-visual-angle response curves. Saved trajectories remain dynamic. Numerical tolerance is not an RMS accuracy target.

At completed outer checkpoints, use one predeclared compact P4-check schedule to detect failure of the whole pipeline early. Select it before seeing residuals, include all conditions, and preserve missing outcomes. It is a progress diagnostic, not the final G8 score or a license to select the best checkpoint by cherry-picked frames. The final comparison evaluates the full declared schedule. No extra global fit is performed on this compact check.

## G7 decision and controlled recovery

**GO:** at least one full DM0 fit meets the declared numerical certificate, the full input population is accounted for, and frozen-model masking is operational. Retain competing certified solutions rather than claiming uniqueness. This authorizes G8 even when empirical cross-error is large or a scientific premise remains unresolved.

**GO_WITH_LIMIT:** a certified fit exists but references, branches, or physical A scale remain ambiguous. Carry these into G8's hypothesis ledger and report alternative solutions. Do not use this status to waive failed numerical certification or rank checks; affected inversions remain unresolved.

**REPAIR:** a demonstrated implementation issue, such as a missed g derivative, chunk-local mean anchor, state label clamping, mutated reference snapshot, or GPU mismatch. Fix it and rerun only the dependency-affected work with preserved prior attempts.

**PAUSE:** no certified fit after the declared budget/starts, or the objective is undefined on the required model domain. Save an uncertified checkpoint. First diagnose numerical cause versus structural mismatch. One justified continuation from a compatible checkpoint may test budget exhaustion; repeated blind budget increases or new optical terms are not the response.

G7 explicitly revisits G2–G5 provisional assumptions using the refined gaze and full model. Keep unresolved items open; a low J cannot certify common scale, an absolute distortion center, or a physiological accommodation law. Ordinary parameter refinement does not reopen every gate; changing a model/reference convention does.

**Current result:** G7 attempt04 is COMPLETE_CERTIFIED_WITH_LIMIT / GO_WITH_LIMIT. It records one reviewed constrained saved-point correction to the immutable attempt03 parent; it does not represent a new fit campaign. The full 89,175-row certificate passes the unchanged 1e-6 stationarity and complementarity thresholds, and the independent public-optics reconstruction matches components within 2.28e-13 and predictions within 3.41e-13 px. Direct objective change is −5.9962e-12 and maximum predicted movement is 6.19e-6 px, so no material empirical improvement is claimed. Attempt04 flags `crosscheck_complete=false` and `comparison_complete=false`; the existing compact records remain attempt03 results. Fixed operational omega1/omega4 references, unquantified omega1 uncertainty, G3 alternatives, and physical zeros remain unresolved. G8 awaits separate authorization. See the [full result summary](../../experiments/distortion_model/stage_04_center_and_dm0_calibration/docs/RESULTS.md), [attempt04 report](../../experiments/distortion_model/stage_04_center_and_dm0_calibration/results/g7_attempt_04/STAGE_REPORT.md), [independent audit](../../experiments/distortion_model/stage_04_center_and_dm0_calibration/results/g7_attempt_04/independent_public_optics_audit.json), and [live progress pointer](../../experiments/distortion_model/stage_04_center_and_dm0_calibration/docs/PROGRESS.md). Attempts01–03 remain immutable historical records.
