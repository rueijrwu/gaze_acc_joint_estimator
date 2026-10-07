# Full-calibration accommodation-law comparison — execution and verification plan

**Status:** PROPOSED execution plan for the already committed `full_position/accommodation_full.py`; **no real-data full-calibration results are claimed here**.  
**Repository/branch:** `rueijrwu/gaze_acc_joint_estimator` / `exp5_full`  
**Companion:** [full-calibration runner](full_position/accommodation_full.py), [accommodation-response theory](ACCOMMODATION_RESPONSE_THEORY.md), [original law-comparison plan](ACCOMMODATION_RESPONSE_PLAN.md).  
**Scope:** fit all reviewed calibration conditions together; compare log and fixed shifted powers by **same-frame three-pair cross-agreement**, not transfer to unseen conditions.

## 1. Scientific objective and distinctions

The main task is **one complete joint calibration per accommodation law**, using all 20 reviewed fixation conditions from captures 1–4. Do not hold out a gaze condition, fixation, or capture to construct the calibration. There are four laws, **four global fits total**, not the historical nine-fold × four-law screen.

For each law `lambda`, calibrate 27 global response coefficients `beta_lambda` and a separate latent state `x_i=(theta_x_i,A_i)` for every valid frame. Every frame contributes all three corresponding P4 x/y measurements, conditional on all three measured P1 points. Different frames within a fixation are allowed to have different gaze and accommodation. Nominal target and demand are **soft fixation-mean anchors**, not instantaneous ground truth.

After fitting a law, freeze its coefficients. Within each calibration frame, perform three **temporary, per-frame P4 exclusions**: estimate the state from all P1 points plus the other two P4 points, then predict both coordinates of the excluded P4. This is an **internal agreement** check using already calibrated conditions, not independent validation. No coefficient refit occurs during these checks.

The principal evidence is whether one shared state explains three simultaneous P1/P4 pair responses. Keep distinct:
- **Full-model fit residual:** prediction minus measured P4 using all three P4 in the fitted frame.
- **Within-frame cross-prediction error:** prediction of one temporarily omitted P4 with coefficients frozen.
- **Subset-state disagreement:** differences among the three state estimates of that same frame.
- **Temporal variation:** actual or estimated gaze/accommodation changes across frames during fixation. This is not automatically error.
- **Nominal-label discrepancy:** descriptive difference to stimulus labels, not measured physiological accuracy.

No absolute RMS accuracy ceiling, nominal-target error cutoff, or fixation-flatness criterion is used to accept or rank laws. **Hard gates concern data integrity, numerical certification, and honest accounting**, not physical accuracy.

## 2. Locked four-model comparison

Use the current `PowerResponseModel` names and registry exactly:

| Candidate | lambda | phi_lambda(A) |
|---|---:|---|
| `ar27_log` | 0 | `log(1 + A)` |
| `ar27_sqrt` | 0.5 | `2*(sqrt(1+A)-1)` |
| `ar27_linear` | 1 | `A` |
| `ar27_quadratic` | 2 | `A + A*A/2` |

`A` is in diopters, with the dimensionless 1-D scale from the accompanying theory. All four models use `t=theta_x/10`, the same conditional27 geometry, 27 coefficients, 2 free states/frame, and image channels x/y. `Dx` retains its separate linear-`A` term. Do not change P1 correspondence, P1 centroid, `sqrt(P1 triangle area)` normalizer, coefficient prior policy, detector, or measurement mask in this study.

Calibration settings: same training-only log27 pilot and reference covariance across all four candidates; coefficient prior strength 0.001 (column-normalized); soft fixation-mean scales 0.10 degrees and 0.25 D; temporal strength 0; computational bounds theta in [-20,20] degrees and A in [0,6] D. These are **experimental conventions and numerical bounds**, not physiological tolerances. Use the same starts, convergence certification, and multistart subset inverse policy for all laws.

Nominal horizontal gaze targets are [-10,-5,0,5,10] degrees within each capture. The calibration protocol's vertical target is nominally zero; pixel y remains a valid image measurement and **there is no vertical-gaze state**.

## 3. Stage FC0 — Audit and harden the committed runner before any costly run

Start by fetching the current branch head, re-reading `full_position/accommodation_full.py` and imported `accommodation_study._fit_task`, and recording actual source/data hashes. Treat the runner as an initial implementation, **not as previously verified full-scale execution**.

Required checks and likely integration issues:

1. **All-condition/all-row semantics:** verify `training_data(..., count=0)` actually uses all baseline-valid frames within the reviewed central-80% intervals. Verify `schedule(..., agreement_per_fixation=0)` produces every original central-80% row, including invalid rows as explicit unscored identities; `fixed_sample(rows,0)` is intended to return all rows. Preserve counts per capture/fixation. “Full calibration” means all valid frames in these declared windows; the excluded edge 20% is a pre-existing window policy, not an outcome-driven holdout.
2. **No inadvertent data leakage:** the full fit consumes all three P4 observations and may use all calibration conditions, by design. During each temporary P4 holdout, the withheld point must not enter the inverse, starts, covariance marginal, branch selection or reconstruction except the final scorer; global coefficients have already seen the point, so do not call this independent prediction.
3. **Shared reference/noise:** `pilot_fit`, `noise_blocks` and `reference_covariance` must be computed once and reused identically for all laws; verify frozen pilot, reference state, coordinate covariance and actual frame covariance.
4. **Runner/module contract:** `_fit_task` was originally written for grouped development screens. Confirm it accepts `full_calibration`, a full set of training-group IDs, all rows, and the policy metadata. Check any assumptions that evaluation IDs are disjoint from training IDs; the deliberate overlap here must be clearly recorded as **internal agreement**, never presented as a sealed evaluation.
5. **Exact candidate registry:** `tuple(CANDIDATES) == NAMES` is order-sensitive. Assert both the exact IDs/exponents and a stable declared order; do not silently map wrong coefficients to exponent labels.
6. **Memory/runtime risk:** the profile solver works with a latent state for *each* valid frame. Previous 48-row/fixation fits do not establish memory or runtime feasibility for many thousands of frames. Before production, measure number of states, weighted-design workspace, Jacobian-vector products, QR workspace, pilot construction and actual peak resident memory. Four simultaneous full fits can multiply memory pressure; serial or reduced concurrency is acceptable.
7. **Source integrity and outputs:** the current runner writes a new output directory and later checks historical hashes. Confirm any output/scratch paths are excluded from source-hash scans; compare exact source commit and live working-source hashes. On interrupted runs preserve failed checkpoints and do not claim completion.
8. **Rerun/recovery:** assess whether `_fit_task` can be safely restarted without overwriting partial `dest` directories. The committed runner creates task directories with `exist_ok=False`, so a failed run is **not automatically resumable**; document a safe new-run / explicit verified-resume procedure before launch.

**FC0 exit:** code contracts, memory feasibility, run provenance and failure/restart policy checked. Fix genuine script defects in a separate clearly described implementation commit if necessary. Preserve original research data and previously saved results.

## 4. Stage FC1 — Tests and bounded rehearsal

### Unit/synthetic requirements
- For lambda=0, identical coefficients, states and P1 context reproduce existing conditional27 values, Jacobians and Hessians.
- Four-law basis/derivative/Hessian finite-difference tests, including A=0 and A=6.
- All framewise states remain free; zero-mean within-fixation state motion changes optical residuals but not the mean-anchor term.
- Exact three-pair mapping, normalization and withheld-point noninterference on every P4 point and x/y coordinate.
- All scheduled frame IDs and exactly three slot IDs per frame; no missing/duplicate outcomes disappear from the denominator.
- Equal-exposure aggregation and paired differences reconstruct saved point/frame residuals; absent exposures are explicitly marked inconclusive rather than averaging over survivors.
- Candidate log-vs-log paired difference is identically zero on the same masks (numerical rounding aside).
- All four laws use identical pilot/covariance policies and solver thresholds; rank/ambiguity/bounds remain visible.
- Nominal-label RMS and temporal spread cannot trigger hard failure or select a law.

### Rehearsal requirements
Run a representative **bounded** end-to-end rehearsal of the actual full-run path (not merely syntax or `--help`) using a separately named scratch output. Choose a bounded subset of rows spread across all 20 conditions *only for the rehearsal*, with exactly the same four response laws and unchanged code path. This subset is **not** reported as full calibration. Observe peak memory, wall-clock, certified convergence, logging, and independent recomputation of a few P4 cross-checks.

Then check enough real full-window metadata to make an informed resource decision before full fitting. Do not extrapolate the previous 246-second 36 small-fit benchmark to a full-data run as a promise.

**FC1 exit:** all numerical/contracts tests pass and a representative rehearsal completes or produces actionable failure diagnostics. Inability to run the entire data here is a blocker to claiming measured full-data results, not a reason to invent them.

## 5. Stage FC2 — Four full joint calibrations

Freeze the source commit and one run manifest before fitting. Use exactly the same all-condition calibration population and noise/pilot for each law. Store frame IDs, fixation IDs, source hashes, correspondence and validity reasons.

For each of the four models:

1. Fit all 27 shared coefficients and the two per-frame latent states, with **all three P4 coordinate pairs per valid frame**.
2. Keep the inherited finite soft mean anchors and 0 temporal regularization. Gaze and accommodation may vary within fixation.
3. Preserve starts, objective components, stationarity, curvature/physical correction certificates, unsuccessful checkpoints and full-frame state trajectories.
4. Require numerical certification before labeling a fit usable. An optimizer stopping by budget is not automatically a certified calibration.
5. Do not discard a law because of large nominal-target RMS, temporal variability, bounds count or relatively large descriptive cross-check RMS. Those are observed outcomes and must be included.
6. If a calibration fails, preserve the failure and its entire scheduled population; do not borrow another law's coefficients/states or fill prediction scores by interpolation.

Four full fits are the only main experiment tasks. If compute/time pressure is excessive, implement a documented staged all-row solver or limited-iteration checkpointing that preserves the same declared final objective; do **not** quietly revert to the old 48-row screen and call it full calibration.

**FC2 exit:** four certified full calibrations or explicit individual failures with status/provenance. No law is promoted by training cost alone.

## 6. Stage FC3 — Same-frame three-pair agreement

For each certified frozen full-calibration model, apply the three-way point exclusion across the **same complete manifest of calibration-window frames**, including explicit unavailable/invalid slots. In every frame i, for j=1,2,3:

- use all P1 as measured context;
- use only P4 points k != j (both x/y coordinates), their exact covariance marginal and the same 49-start scalar inversion policy;
- freeze the chosen state and branch diagnostics;
- reconstruct the excluded P4 x/y and compare with the measured excluded point only in scoring.

For all three complete subset estimates `xhat_i,-1`, `xhat_i,-2`, `xhat_i,-3`, report:

`E_i^2=(||e_i1||^2+||e_i2||^2+||e_i3||^2)/3`

`G_theta,i^2=((theta_i,-1-theta_i,-2)^2+(theta_i,-1-theta_i,-3)^2+(theta_i,-2-theta_i,-3)^2)/3`

and the corresponding `G_A,i^2` in D squared. These compare **simultaneous estimates of the same frame**, not states from different frames.

Primary reporting: equal-fixation/exposure mean of E-squared, plus RMS `sqrt(mean(E^2))` as a readable summary; separately report `G_theta`, `G_A`, signed x/y error, per-P4 error, worst-point and tail distributions. No absolute RMS value is a pass/fail threshold.

Critically, the full calibration's in-sample residual and frozen-coefficient leave-one-P4 cross-reconstruction are **different metrics**, and the latter reuses observations during coefficient fitting. Report both distinctly. Neither gives independently measured gaze/accommodation accuracy.

## 7. Stage FC4 — Compare laws on exact common frames

Use `ar27_log` as the declared **reference**, not a presumed winner. For each alternative, compare on exact same frame identities where both have a complete triple. Keep all scheduled, input-valid, scored, interior, boundary, ambiguous and failed counts. Do not subtract separately filtered candidate RMS values.

Calculate candidate-minus-log paired differences in equal-exposure E-squared, `G_theta^2`, `G_A^2`, axes and worst-point squared. A negative difference means better internal cross-prediction/agreement in the corresponding quantity, not a universal physiological success. Report `N` and exposure counts for every pair. If an exposure has no shared complete triple, mark the equal-exposure comparison incomplete; do not silently omit it or treat it as zero.

Show per-capture, nominal horizontal gaze, fixation, accommodation-demand label and P4 index. Preserve dynamic time traces: compare all-three and three subset states at **the same timestamps**. Include mean-anchor deviations and temporal spread separately as descriptive context only, never as accuracy/flatness gates.

If one law wins on P4 reconstruction but loses on state agreement, describe the tradeoff rather than forcing a scalar mixture of px, degrees and diopters. Do not use residual thresholds to remove difficult observations post hoc. State bounds are numerical flags and do not establish incorrect physical behavior.

**Decision language:** `internal_agreement_preferred`, `tradeoff`, `inconclusive`, or `numerically_unavailable`, based on recorded paired evidence. The plan defines **no automatic deployment promotion** and no hard performance threshold. Avoid claiming that fitting and checking the same calibration population proves transfer or a physical accommodation law.

## 8. Outputs and verification

Use a fresh directory, e.g. `experiments/full_position/accommodation_full_v1/`, with:

- `config.json`: exact version/parameters, all-condition IDs, coefficient/noise/pilot hashes and data source hashes.
- `splits/full_calibration/`: all-row calibration inputs, full agreement schedule and immutable manifest (the directory name is legacy runner structure, not a held-out split).
- `fits/full_calibration/<candidate>/`: accepted model or failed checkpoint, complete training states, numerical candidates/certificates, cross-check frame and holdout records, and trajectory diagnostics.
- `summary.json` and `RESULTS.md`: cross-agreement scorecards and paired comparisons, with absent outcomes and no hard RMS gates.
- `verification.json`: independent source hashes, model/derivative invariants, per-condition row counts, valid/invalid point accounting, 4 fit outcomes, same-frame subset noninterference and paired aggregation checks.
- `completion.json`: completed versus certified counts and actual resource usage, without claiming physiological validation.

Before publication, independently recompute several full-frame model predictions and three rotating holdouts, verify all frame/slot IDs and the 20 exposure roster, and check equal-exposure paired squared changes from saved raw records. Run full repository tests, record actual results, and verify frozen input files remain unchanged.

## 9. Suggested command after FC0/FC1

From a checkout whose current source matches the pinned manifest:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m full_position.accommodation_full \
  --output experiments/full_position/accommodation_full_v1 \
  --workers 1 \
  --agreement-per-fixation 0
```

`--agreement-per-fixation 0` requests all calibration-window rows for internal agreement. **It does not mean no P4 points are excluded within a frame.** The runner's current `--workers` setting applies to whole candidate fits; increase concurrency only after peak-memory evidence. The output path must not exist.

**Do not execute a full experiment based on this plan alone if FC0 reveals a runner contract defect or memory infeasibility.** Fix the script, retest, and rerun in a new hashed output directory. The script is the implementation starting point; the plan does not claim it has already completed real full-data calibration.

## 10. Scope boundaries

- No whole-gaze or whole-capture holdouts in this experiment.
- No 37-coefficient model, y-only/differential-y mask, shared-y covariance trial or learned exponent in the first comparison.
- No framewise constant fixation assumption, hard nominal RMS criterion or temporal flattening.
- Captures 5/6 remain untouched.
- No independent physiological accommodation accuracy or generalization claim.
- Subsequent validation, alternative masks and capacity changes require separately declared experiments; they are not prerequisites for reporting the full-calibration internal-agreement comparison.

**Successful deliverable:** a reproducible, numerically certified all-condition comparison of log/sqrt/linear/quadratic calibrations, with full-population same-frame cross-agreement for each law and exact paired comparisons, including failures and tradeoffs.