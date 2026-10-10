# Staged empirical checks for the distortion estimator

**Branch:** `rueijrwu/gaze_acc_joint_estimator / exp5_distortion_model`  
**Design base:** `e4d4ca933a65c2121fbca1a51928d9d308bf6938`  
**Status:** Proposed subplans. Every gate is **NOT_RUN** until an implementation produces linked evidence. No calibration, test result, or model agreement is asserted here.  
**Authority:** [Theory.md](Theory.md) defines the equations; [ESTIMATOR_PLAN.md](ESTIMATOR_PLAN.md) defines S0–S8. This document adds execution checkpoints, not another model, optimizer, or phase numbering system.

## 1. Objective and shortest path

Establish, one testable assumption at a time, whether the empirical P1/P4 recordings support the proposed optical calibration. Each step must leave a small result, its interpretation, a resumable checkpoint, and exactly one next action. Do not complete the entire implementation before looking at real measurements.

The first target remains **DM0-M1 full calibration and its complete internal cross-agreement report**. DM1 radial deformation is conditional, and DM2 is not an automatic next phase. Finding that radial distortion is unidentifiable can be a successful reason to keep DM0.

All twenty reviewed conditions from captures 1–4 remain in the final calibration. Use full reviewed intervals with the declared validity policy. Plots, block summaries, and a bounded numerical rehearsal are not condition-held-out calibration or a replacement for the full fit. Captures 5/6 stay outside this workstream.

## 2. Gate map and subplans

| Existing step / gate | Smallest result to inspect | What the gate authorizes |
|---|---|---|
| S0 / G0 | Census, immutable frame/slot manifest, correspondence, relative-map checks | Trust the inputs and measurement convention |
| S1 / G1 | Five-condition centroid bootstrap and framewise initial states | Use approximate gaze to initialize P1 |
| S2 / G2 | Independent P1 symmetry evidence, constrained reference fit, P1-only scale and residuals | Use one declared nuisance-scale correction |
| S3 / G3 | Independent P4 symmetry evidence and dual-zero conversion | Choose a supported P4 reference, not assume visual/P1 zero |
| S4 / G4 | Near-reference P4 effective-scale response and uncertainty/ambiguity notes | Initialize DM0 accommodation, not declare its physical law established |
| S5 / G5 | P4 keystone response across gaze; revisit S4 with individual local angles | Assemble the complete optical shape model |
| S6 / G6 | Center/centroid correction ledger and forward D polynomial | Start full joint refinement with one consistent residual |
| S7 / G7 | Fresh full DM0 fit, numerical certificate, trajectories and provisional-check review | Freeze a usable model for full cross-check, or retain an uncertified checkpoint |
| S8 / G8 | Whole-loop P4 omissions on the full schedule and native-pixel/state agreement | Judge empirical compatibility and remaining model deficiencies |
| Optional GX | Specific residual mechanism, identifiable extra response, matched comparison | Permit DM1 or one DM2 change; otherwise keep/report DM0 |

Read only the next needed subplan:

1. [Data and gaze bootstrap: S0–S1](stages/01_DATA_AND_BOOTSTRAP.md).
2. [P1 reference and common scale: S2](stages/02_P1_REFERENCE_AND_SCALE.md).
3. [P4 reference, accommodation, and gaze deformation: S3–S5](stages/03_P4_BASELINE_AND_DEFORMATION.md).
4. [Corrected centers and full DM0 fit: S6–S7](stages/04_CENTER_AND_DM0_CALIBRATION.md).
5. [Cross-agreement and model decision: S8, GX](stages/05_CROSSCHECK_AND_MODEL_DECISION.md).

## 3. Numerical gates are not accuracy gates

**Hard execution contracts:** correct source identities, finite admissible inputs, valid covariance, positive P1 scale where used, correct derivatives/reference conversions, complete scheduled-outcome accounting, and the stated numerical certification. Set floating-point and solver tolerances before comparing models. A numerical failure is not evidence against the optical hypothesis.

**Empirical decisions:** quantify the response, ambiguity, coherent residual patterns, coverage, and sensitivity at each step. There is no universal pixel/degree/diopter RMS ceiling, no requirement to lie within the soft-anchor scales, no minimum percentage improvement, and no fixation-flatness test. A decline in training cost is not a cross-check result. A tiny p-value from many correlated frames is not proof of a useful response.

Use the same named comparison population and fixed weighting policy. Report effect sizes and signed residuals by point, axis, gaze, and capture; include the spread across several contiguous time blocks as a stability diagnostic. Block summaries do not create independent physiological ground truth or remove blocks from final calibration. Never use a candidate's fitted temporal spread as the denominator that defines success.

Each gate has one of four decisions:

| Decision | Meaning and next action |
|---|---|
| **GO** | Required integrity evidence exists and the result is usable for the next stated task. It does not validate the whole model. |
| **GO_WITH_LIMIT** | A bounded, explicit initialization uncertainty remains. Freeze the declared convention, name the later gate that will revisit it, and continue without claiming the uncertainty resolved. |
| **REPAIR** | A specific implementation/input/convention problem is identified. Fix that problem and repeat this gate; do not add optical capacity. |
| **PAUSE** | Evidence is missing, contradictory, or too weak for the proposed next inference. Preserve the result and identify the smallest discriminating measurement or controlled ablation. Do not silently change the model. |

A large residual alone does not force PAUSE. A broad symmetry minimum or unidentifiable radial coefficient need not prevent DM0: use GO_WITH_LIMIT only when a legitimate fixed reference or reduced model leaves the next calculation well-defined. It cannot waive bad correspondence, nonexistent reference evidence, invalid scale, numerical certification, or unavailable model rank. Preserve such failures explicitly.

## 4. One small record per gate; one live resume pointer

Reuse the main plan's config, population, arrays, checkpoints, and result files. Do not build a workflow engine. A run needs only two additional small Markdown records:

- `STAGE_REPORT.md`: append one section per gate attempt, with links to evidence already saved by the estimator.
- `PROGRESS.md`: current gate, last usable checkpoint, unresolved limitations, and exactly one authorized next action.

Every stage-result section uses this template:

```text
Gate / attempt / candidate:
Status: NOT_RUN | RUNNING | COMPLETE | INTERRUPTED
Decision: none | GO | GO_WITH_LIMIT | REPAIR | PAUSE
Source commit and dirty-source hash, theory/plan/config hashes:
Parent gate record and checkpoint:
Question and proposed optical assumption:
Parameters fixed / fitted / derived:
Population: scheduled / input-valid / solved / unresolved, by exposure:
Actual evidence: table values, residual/curve plot paths, record IDs:
Evidence kind: synthetic numerical / empirical initialization / full fit / cross-check:
What the data support; what they contradict; what remains unknown:
Deferred check and owning gate (GO_WITH_LIMIT only):
One next action; retry scope; last usable checkpoint:
Reviewer and decision timestamp:
```

Templates and proposed plots are not completed evidence. Do not enter invented placeholder numbers. A script exiting successfully sets neither GO nor optical agreement automatically. The execution owner reviews the result and records the decision; adjacent gates may proceed within the same run after that review. Human input is required when changing an approved scientific assumption, not for every routine file write.

At every handoff, read `PROGRESS.md`, verify the remote branch and checkpoint compatibility, then continue. A stale chat must not reset the plan or overwrite newer work.

## 5. Deliberate limits on iteration and complexity

For each failure, inspect the first documented contradiction and perform **one targeted repair or one named alternative**, not several simultaneous changes. A default of one local retry per diagnosed issue prevents blind loops; more attempts require a recorded new hypothesis and evidence, not simply raising iteration budgets repeatedly. This is a workflow budget, not a scientific rejection threshold.

Implement only the code needed to reach the next gate. Retained-only input masking and the relative forward contract start at G0; expensive full cross-checks wait for G7. One CPU reference and one tested GPU backend are sufficient. No dashboard, generic plugin system, second accelerator implementation, large hyperparameter grid, or whole-condition held-out campaign is on the critical path.

Only a declared future comparison may change an optical assumption. Do not respond to poor agreement by adding free P4 scale, physical Z, per-frame centers/exponents, a common P1/P4 zero, silent trimming, or a stronger nominal-state penalty. The existing relative observations, separate zeros, P1-only common scale, dynamic states and P4-minus-P1 sign remain locked.

## 6. Checkpoint dependencies and revisits

Store the actual upstream checkpoint IDs, not just stage names. Normal parameter refinement within S7 does not require rerunning every initialization gate at every iteration. Review provisional findings at completed S7/S8 checkpoints.

Changes to data selection, correspondence, units or covariance reopen G0 and all dependent comparisons. Changes to the bootstrap require review of dependent reference initialization, not blind reuse. A new P1 template/scale or zero convention reopens G2 and dependent P4/center/full-fit checks. A new P4 reference, zero or baseline reopens G3/G4 and downstream checks. A changed accommodation family or D degree requires a fresh full calibration and G8 comparison. A masking/scoring-only code fix reruns the affected inversions/report from a compatible frozen model; it need not refit unchanged coefficients.

Never overwrite previous attempts. Stop safely after a gate or a compatible solver checkpoint. Interrupted is not failed science; uncertified is not a calibrated model. Keep source/model hashes and enough native-row evidence to reproduce every published score.

## 7. Final scientific disposition

The gate ledger must distinguish:

**protocol_ready**: bookkeeping and computation are trustworthy; **supported_effective_response**: completed full-calibration/internal agreement supports DM0 within its declared conventions; **supported_increment**: an identifiable extra mechanism improves the matched evidence; **tradeoff**: benefits and limitations differ; **model_assumption_unresolved**: current evidence cannot settle a required optical premise; **numerically_unavailable**: the mathematics was not adequately solved.

An experiment can be complete with an unresolved or negative scientific result. Do not equate `crosscheck_complete` with `model_supported`. Common-scale compatibility in these recordings does not independently prove equal P1/P4 axial responses; capture and accommodation demand are confounded. Independent accommodation accuracy is outside these gates.

**Stop once the simplest defensible conclusion is available.** A usable effective model does not need an identifiable absolute barrel coefficient, and a contradictory dataset does not need an ever-larger polynomial.
