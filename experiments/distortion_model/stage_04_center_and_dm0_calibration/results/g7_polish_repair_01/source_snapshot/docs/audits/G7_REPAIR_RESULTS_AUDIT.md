# G7 repair-results audit: constrained polishing and empirical cross-agreement

**Date:** 2026-10-10  
**Repository / branch:** `rueijrwu/gaze_acc_joint_estimator / exp5_distortion_model`  
**Inspected commit:** `cb6d1902c2dc334c7b830f988ad40220bde33189`  
**Scope:** Follow-up to [G7_AUDIT.md](G7_AUDIT.md), reviewing repair01 and G7 attempt03. This is a documentation-only source/result review, not a new numerical experiment, implementation, or authorization to resume fitting.  
**Gate status remains:** `COMPLETE_UNCERTIFIED / PAUSE`; G8 is not authorized [R0].

## 1. Executive decision

**Keep the derivative repair. Diagnose the remaining constrained polishing step at the saved solution, then prioritize interpretation of the existing three-pair agreement. Do not launch another broad calibration campaign or expand the accommodation model merely to reduce the training objective.**

The repair resolved the reported interval-derivative artifact and separated previously conflated curvature statuses. The subsequent bounded continuation substantially reduced the correctly evaluated global stationarity residual, but changed the objective and optical agreement only negligibly. Numerical closure and empirical model adequacy are therefore separate questions [R1,R2,R3].

The first follow-up finding is a source-confirmed mismatch: the certificate tests reduced Lagrangian curvature on the active-constraint tangent space, while the polishing proposal requires positive curvature in the entire shared-parameter space and omits the same constraint-Hessian correction from its proposal matrix. This explains why the two checks need not agree. It does **not** establish that this is the sole cause of every rejected step; an instrumented saved-point comparison is still required [C1].

## 2. What is established by the saved results

### 2.1 Derivative repair: resolved at the inspected smooth branch

Repair01 reevaluated the exact selected attempt02 state without changing the states, globals, objective, or saved fit arrays. It differentiated the selected endpoint algebra while preserving outward-rounded interval values for feasibility. The saved regression reports 22 directions on both signs of the near-zero coefficient, maximum gradient error `5.36e-9`, and Hessian error `1.72e-8`. The old large finite-difference Hessian artifact is reproduced, whereas the smooth-branch contribution in the affected diagonal is zero. Actual ties are marked nonsmooth rather than given an unjustified smooth certificate [R1].

The repaired same-state global KKT residual was `0.0035570`, not the old finite-difference-based `2.8122e-4`. Use the repaired value when describing the effect of continuation. These numbers refer to different derivative assessments of the same initial point; they are not interchangeable measures of progress [R1].

### 2.2 Attempt03: better stationarity, effectively unchanged prediction

| Quantity | Saved result |
|---|---:|
| Full calibration population | 89,175 complete valid frames, all 20 exposures |
| Original scheduled / unavailable rows | 100,090 / 10,915 |
| Shared optimization coordinates | 19 |
| Warm-start objective | 1925.6013387428595 |
| Final objective | 1925.601338676003 |
| Objective decrease | Approximately 6.69e-8 |
| Frame-state projected gradient | Approximately 1.23e-11 |
| Global scaled KKT residual | Approximately 2.5103e-5 |
| Declared stationarity tolerance | 1e-6 |
| Constraint complementarity | Approximately 1.89e-13 |
| Constrained profile curvature | Positive, rank 16 |
| Native relative-coordinate RMS | 4.678857434 px; previously 4.678851016 px |
| Final compact pooled coordinate RMS | Approximately 29.367297 px |

These are reported results, not independently rerun measurements in this audit [R2]. The declared eight outer updates completed. Only one observed-polish proposal was attempted and it was rejected; six proposals were a maximum budget, not six completed iterations. Across joint proposals the report records 810 objective line-search failures and five nonpositive shared-proposal-curvature failures [R2].

Positive constrained curvature does not replace stationarity. Conversely, a remaining stationarity failure does not make saved residuals uninterpretable. Keep the snapshot uncertified while examining its empirical behavior. Do not reinterpret the authorized one-degree accommodation-induced centroid-equivalent shift as a one-degree gaze accuracy requirement or a fixation-label tolerance [R0,R2].

## 3. Numerical findings and the narrow repair

### N1 — Certificate and polishing step use different curvature conditions

**Confirmed by source inspection.** In `BoundedJointDM0.certificate()`, the code eliminates free frame-state response, subtracts multiplier-weighted physical-constraint Hessians, constructs a null space from strongly active constraint rows, and tests the projected matrix [C1].

For inequalities `c(p) >= 0`, define the Lagrangian as

$$
\mathcal L(X,p,\lambda)=J(X,p)-\sum_k\lambda_k c_k(p),\qquad\lambda_k\ge0.
$$

Since these physical inequalities depend only on the shared parameters, the corresponding reduced matrix is

$$
S_{\mathcal L}=H_{pp}-H_{pX}H_{XX}^{-1}H_{Xp}
-\sum_k\lambda_k\nabla^2 c_k(p).
$$

State elimination must include the full-fixation mean coupling and the applicable state-bound active set. The reported certificate tests `N.T @ S_L @ N` in a 16-dimensional tangent space [R2,C1].

In contrast, `direction()` constructs a shared objective-Hessian Schur complement, with damping, and `quadratic_step()` rejects it if its full-space normalized minimum eigenvalue is nonpositive. The proposal does not subtract the multiplier-weighted constraint Hessians used by the certificate [C1].

These tests are not logically equivalent. The elementary constrained problem `f(u,v)=(u^2-v^2)/2, v=0` has a positive tangent Hessian and an indefinite full Hessian. Positive full-space approximations can be sensible during broad optimization, but are stronger than necessary for local constrained polishing.

**Recommended diagnosis:** compare the present proposal and a constraint-consistent proposal at the same saved attempt03 point. Use the same state elimination, parameter scaling, active constraints, and Lagrangian curvature as the certificate. Do not merely delete the positivity guard and pass an uncontrolled indefinite problem to SLSQP. Preserve the existing broad-fit safeguard until a tested local alternative exists.

### N2 — Measure the small subproblem's accuracy independently

**Confirmed reporting gap, not proof that SLSQP is inaccurate.** `quadratic_step()` checks solver success and linearized feasibility but does not return an independently reconstructed full subproblem KKT ledger [C1].

For one saved-point diagnostic, record the QP gradient/stationarity, constraint violation, multipliers, complementarity, active-row rank, proposed step, and predicted reduction in consistent scaled coordinates. Check the small system by a second linear-algebra route where appropriate.

A local null-space or bordered KKT solve is a candidate implementation, not a new optimizer framework. If the active Jacobian is `B`, a local equality working set uses `B*delta_p=-c_active`, with a particular feasibility correction plus tangent displacement. Reconstruct the frame-state step using the same coupled inverse. Check multiplier signs, inactive inequalities, parameter boxes, and branch changes; do not freeze the current active set unconditionally. Weakly active constraints and endpoint ties require explicit treatment rather than an ordinary smooth equality certificate.

### N3 — Separate curvature rejection, nonlinear feasibility, and cost acceptance

**Hypotheses to test, not established causes.** A linearly feasible direction may violate curved active constraints at second order. Also, subtracting two accumulated objectives near 1925 can obscure a very small change. The current history does not by itself determine which acceptance issue dominates [R2,C1].

Record predicted versus actual constraint slack, directional decrease, accepted-step size, and component-wise objective changes. A small second-order feasibility restoration is appropriate only if that specific failure is demonstrated. It must preserve the same continuous one-degree condition, not replace it by a softer penalty or a different domain.

For the fixed raw-coordinate weighting, calculate a small objective difference directly. With old residual `r_i=y_i-f_i`, prediction change `delta_f_i=f_i_new-f_i_old`, and the same exposure weight `w_i`,

$$
\Delta J_{data}=\sum_i w_i\left[-r_i^T R_i^{-1}\delta f_i
+\tfrac12\delta f_i^T R_i^{-1}\delta f_i\right].
$$

Add exact changes of the full-mean anchors and priors. This identity evaluates the same objective; it does not change the cost or justify accepting an increase outside the declared numerical policy. Use it to diagnose cancellation rather than presume that roundoff explains all rejections.

### N4 — Do not solve a scientific problem by relaxing a numerical gate

The existing `1e-6` stationarity threshold remains unmet. Preserve that fact and all earlier attempt artifacts. Do not retrospectively relax the threshold to report a certified fit. If a numerical floor is demonstrated, quantify the remaining correction in predicted native pixels and framewise degrees/diopters and separately document any proposed certificate revision.

Small objective change is not sufficient evidence that all parameter corrections are negligible, especially along weakly constrained directions. No new full-data Hessian, proposal, or numerical-floor calculation was performed for this document.

## 4. The empirical question is now explicit

The saved compact aggregation reports the following complete-triple results [R3]:

| Metric | Result |
|---|---:|
| Scheduled / scored slots | 300 / 270 |
| Complete eligible frames | 90 |
| Represented exposures | 20 of 20 |
| Equal-exposure P4 vector-error RMS E | 41.8138 px |
| Same-frame gaze disagreement Gtheta | 1.50113 degrees |
| Same-frame accommodation disagreement GA | 1.31867 D |

The same rounded values appear at outer04, outer08, and the final checkpoint, with identical usable membership. Thus the numerical continuation has not demonstrated a material change in this compact measurement agreement [R3].

The `41.8138 px` value is not a sudden increase from `29.367297 px`: the former uses squared 2D vector errors averaged within complete frames and equally across exposures; the latter is a pooled scalar-coordinate RMS. Weighting as well as vector-versus-coordinate convention differs. Neither is directly comparable with the ten-channel full-fit coordinate RMS [R2,R3].

These are not hard error limits, and the compact schedule is not full G8. It is also not independent physiological accuracy: the global calibration already used the recordings, and the retained subsets overlap. GA compares different subset estimates at the same timestamp, not nominal demand or another frame. Real accommodation and gaze changes during fixation remain allowed.

### E1 — Explain where disagreement occurs before adding optical capacity

Use the existing records to produce a single matched diagnostic table with E, Gtheta, and GA, retaining every scheduled identity and unavailable outcome. Include point/axis, exposure, interval-position, median/tail, and bound/interior partitions. Report the largest squared-error contributors so a few exceptional rows cannot masquerade as typical performance. The saved report notes 48 bound-active compact inversions; a bound-active solution is not automatically erroneous [R2].

The known compact schedule includes interval endpoints. Preserve them and the original aggregate. Compare already recorded endpoint and interior groups as diagnostics, not outcome-driven trimming. Do not assert that transitions explain the error before examining the records, or assume that the compact sample estimates the full-period distribution without bias.

If an appropriate breakdown is already saved, inspect it rather than implement a duplicate reporting subsystem. No new inference or full calibration is required to summarize existing subset records.

### E2 — A later small inference diagnostic can separate anchor dependence from subset inconsistency

Subject to explicit authorization, freeze the same model and compare on the same timestamps:

$$
X_{calibration}\quad\leftrightarrow\quad
X_{all\ three,\ no\ labels}\quad\leftrightarrow\quad
X_{-1},X_{-2},X_{-3}.
$$

Calibration versus label-free all-three inversion diagnoses dependence on the mean anchors or differences between calibration and application inference. All-three versus two-P4 inversion diagnoses information loss, subset disagreement, or model mismatch. These are hypotheses, not established conclusions.

Use retained-only initialization, covariance marginals, and branch selection for every omission. Do not warm-start subset inference with an all-three state or use an omitted observation through a centroid. Do not add temporal priors, clamp states to nominal labels, or change the shared calibration. This proposed diagnostic is not an executed or authorized G8 campaign.

### E3 — Let a reproducible residual mechanism select the next model experiment

Keep separate operational P1/P4 zeros, P1-only common scale, relative coordinates, center-based gaze, and the authorized physical bound. The live progress record still lists unknown physical zeros, fixed operational offsets, and unresolved reference sensitivity [R0]. A successful derivative repair has not resolved those assumptions.

A repeatable scale-response mismatch across demands suggests examining M(A). A point-dependent deformation not explained by scale may motivate a radial increment only if it is identifiable. A systematic gaze/reference-dependent pattern points first to alignment or the rotation map. An isolated extreme row calls for inspection of its measurements before adding a global coefficient.

Do not automatically promote DM1 or DM2, release per-frame centers/scales, or search more exponents. Do not treat a model-implied curve against its own fitted A as independent confirmation of the accommodation law. Capture/demand confounding and the lack of independent accommodation truth remain interpretation limits.

## 5. Bounded follow-up plan and decision gates

These are proposed next actions, not completed work or a modification of the live PAUSE decision. Reuse the existing stage-report/progress mechanism; do not create a workflow engine.

| Step | Minimum evidence | Decision |
|---|---|---|
| P0: freeze evidence | Attempt03 states/globals, source and parent hashes, fixed objective/bound/reference conventions | REPAIR any mismatch; never overwrite prior attempts |
| P1: saved-point proposal audit | Present versus constraint-consistent proposal; independent small-system residuals; active/branch status; predicted and actual cost/slack changes | Identify one demonstrated repair or retain PAUSE; no blind continuation |
| P2: compact empirical interpretation | Existing-record matched E/Gtheta/GA distributions and failure contributors | Name the dominant contradiction or report evidence insufficient; no trimming or automatic model expansion |
| P3: optional numerical closure | Only after review, one narrowly scoped compatible correction with the same objective and physical constraint | Recompute the full declared certificate; do not change tolerance after seeing the result |
| P4: full empirical evaluation | A certified G7 checkpoint and separate authorization for G8 | Full schedule and cross-agreement; numerical success does not automatically mean optical support |

P1 and P2 address different questions. Read-only interpretation of existing empirical evidence need not wait for the final stationarity digit. Further optimization, new inference, and G8 require their own explicit authorization under the current handoff [R0]. Stop after the smallest defensible diagnosis rather than repeatedly increasing iteration budgets.

**Exactly one immediate next action:** inspect the saved attempt03 polishing subproblem and determine whether the certificate/proposal curvature mismatch, small-system accuracy, or acceptance calculation explains the rejected correction. Preserve all data, model parameters, anchors, and the one-degree physical bound during that diagnosis.

## 6. Scope, verification, and precedence

This follow-up adds an audit document only. It leaves `G7_AUDIT.md`, fitting code, source snapshots, saved results, theory, plans, and live gate status unchanged. The earlier audit remains the history of the original branch-crossing defect; [R1] is the evidence that this defect was repaired at the selected smooth branch.

Verification performed for this document: fresh branch/status reads, inspection of the current proposal/certificate code, and comparison with saved repair/attempt/compact reports. Numerical values above are attributed to those reports. No fitting, inference, derivative regression, new probe, automated test suite, or full-data numerical reconstruction was executed as part of this document creation.

The working conclusion remains: **numerical repair substantially clarified the certificate but has not established a better accommodation model. Close the specific polishing inconsistency and use the three-pair empirical evidence to decide the next optical question.**

## Sources pinned to the inspected commit

All repository evidence below is pinned so later commits cannot silently change the basis of this audit.

- [R0: Stage 4 live handoff at audit time](https://github.com/rueijrwu/gaze_acc_joint_estimator/blob/cb6d1902c2dc334c7b830f988ad40220bde33189/experiments/distortion_model/stage_04_center_and_dm0_calibration/docs/PROGRESS.md).
- [R1: Repair01 same-state derivative and curvature reassessment](https://github.com/rueijrwu/gaze_acc_joint_estimator/blob/cb6d1902c2dc334c7b830f988ad40220bde33189/experiments/distortion_model/stage_04_center_and_dm0_calibration/results/g7_repair_01/STAGE_REPORT.md).
- [R2: G7 attempt03 execution, numerical evidence, and limits](https://github.com/rueijrwu/gaze_acc_joint_estimator/blob/cb6d1902c2dc334c7b830f988ad40220bde33189/experiments/distortion_model/stage_04_center_and_dm0_calibration/results/g7_attempt_03/STAGE_REPORT.md).
- [R3: Saved compact cross-agreement aggregation](https://github.com/rueijrwu/gaze_acc_joint_estimator/blob/cb6d1902c2dc334c7b830f988ad40220bde33189/experiments/distortion_model/stage_04_center_and_dm0_calibration/results/g7_attempt_03/compact_diagnostics/REPORT.md).
- [C1: Current constraint derivatives, QP proposal, update, and certificate](https://github.com/rueijrwu/gaze_acc_joint_estimator/blob/cb6d1902c2dc334c7b830f988ad40220bde33189/distortion_model/centroid_bound.py).
