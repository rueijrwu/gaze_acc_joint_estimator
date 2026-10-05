# Full-information calibration handoff

**PAUSED by the user. Do not automatically resume implementation, algorithms, fits, tests, or scientific investigations.** This document records the state for an explicit future handoff. Existing outputs must be preserved; no commit, reset, or result selection is requested.

The scientific question is whether retaining full relative geometry `[m, S1, S4]` adds useful information for joint gaze/accommodation calibration beyond normalized `[m/S1, S4/S1]`, and whether the observed mean-gaze behavior comes from measurement/model assumptions or regularization. Work began from [CURRENT_PLAN.md](CURRENT_PLAN.md), but increasingly drifted toward coefficient solvers, trust-region stopping and numerical refinement. Conceptual measurement/model decisions should be reconsidered before continuing those algorithms.

## Implemented workflow

The isolated maintained root is [exp2/full_information](../exp2/full_information/). Existing full, reduced and relative-calibration outputs were preserved. The user requested **Sol medium for implementation and Luna low for all mechanical reads, searches, scans, investigations and execution**; root handled design/audit. Shell commands use the repository's `rtk` wrapper and single-thread Python `-B` environment.

| Source | Implemented responsibility |
|---|---|
| [experiment.py](../exp2/full_information/experiment.py) | Frozen-solution diagnosis; guarded mean probes/continuations; anchor and initialization sensitivity; profile/sensitivity reports; validation CLI |
| [profile_problem.py](../exp2/full_information/profile_problem.py) | Optional anchor multipliers and mean probe, derivatives, physical envelope and original-objective constrained KKT |
| [scale_diagnostics.py](../exp2/full_information/scale_diagnostics.py) | S1 structure/confounding and correlated-covariance/free-scale controls |
| [joint_problem.py](../exp2/full_information/joint_problem.py) | Conditional full-relative model and exact normalized free-scale control |
| [joint_inverse.py](../exp2/full_information/joint_inverse.py) | Anchor-free bounded multistart inference with physical/branch flags |
| [matched_experiment.py](../exp2/full_information/matched_experiment.py), [matched_predict.py](../exp2/full_information/matched_predict.py) | Training-only matched cohorts, frozen shared preprocessing, guarded checkpoints/predictions and phase lineage |
| [matrix_summary.py](../exp2/full_information/matrix_summary.py) | Explicit case aggregation, convergence/coverage/flags and same-support nominal consistency/spread |
| [validation.py](../exp2/full_information/validation.py) | Independent-reference contract and fail-closed evaluator |
| [coefficient_newton.py](../exp2/full_information/coefficient_newton.py) | Experimental optional coefficient solver; see status below |

The shared [calibrate_continuation.py](../exp2/calibrate_continuation.py) has backward-compatible optional physical-gradient/preconditioner hooks and verified resume initialization. The legacy default path remains intact. [COMMANDS.md](../exp2/full_information/COMMANDS.md), [PROTOCOL.md](../exp2/full_information/PROTOCOL.md) and [matrix_protocol.json](../exp2/full_information/matrix_protocol.json) describe the executable protocol. [RESULTS.md](../exp2/full_information/RESULTS.md) is an earlier partial narrative; the reports below and this handoff take precedence for current status.

## Accepted evidence and unresolved results

**Frozen diagnosis:** Seven original objectives reconstructed within `1e-15`; fourteen model/checkpoint hashes matched. Optical loss was much smaller than accommodation-anchor cost. That allocation alone does not prove a weak optical direction or rank models with different cohorts/capacities. See [current-solution diagnostics](../exp2/full_information/diagnostics/current_solutions/current_solution_diagnostic.json).

**Scale:** The retained cohort has 77,756 frames and 20 fixations. Small within-fixation S1 variation does not establish known scale. Demand is nested in capture, nominal gaze is ordered within recording, and time/drift/detector effects are confounded. No independently calibrated scale, independent gaze/accommodation references or verified randomized repeats are present. Adjacent-frame covariance includes motion and detector variation. See [scale structure](../exp2/full_information/diagnostics/scale_structure/scale_structure.json) and [scale controls](../exp2/full_information/diagnostics/scale_structure/scale_controls.json).

**Matched internal holdout:** All cases withhold fixation 17, the demand-2 center fixation. Common three-demand training IDs are `0–9,15,16,18,19`; the four-demand arm additionally trains on `10–14`. The normalized coefficient map is the same 14-by-12 interpolation map in both arms. Exponent, noise, transform anchor, initial function center/ranges and prior precision are frozen from the common training cohort; held-out rows do not enter preprocessing.

| Cases | Current scientific/numerical status |
|---|---|
| N3, N4 | Converged; held-17 predictions are conditional nominal-target consistency/stability results, not physiological accuracy |
| R3, R4 | Initial full-relative fixed-scale runs reached their budgets without convergence; no accepted full-three-channel comparison |
| U3_tau001, U3_tau01 | Finite-scale-uncertainty runs also budget-limited/nonconverged; no accepted constrained-scale comparison |

N3 converged in approximately 583.67 s (562 evaluations/483 accepted steps); N4 in 293.22 s (131/74). N3 held-17 inference retained all 4,080 frames, with no reported failure/ambiguity/unverified/bound/extrapolation flags. Its nominal mean offsets were approximately `+0.07542 deg` and `-0.20327 D`, with within-fixation SD `0.05757 deg` and `0.30904 D`. These are a **single withheld fixation**, not independent accuracy errors. Exact case results are in [matched matrix Markdown](../exp2/full_information/matched_summary/production_20261004/matrix_summary.md), [JSON](../exp2/full_information/matched_summary/production_20261004/matrix_summary.json), [CSV](../exp2/full_information/matched_summary/production_20261004/matrix_summary.csv) and plot (plot omitted from this repository). Models remain under `exp2/full_information/matched/{N3,N4,R3,R4,U3_tau001,U3_tau01}`.

Full-relative observations are `[d, rho, log(S1)]`, with delta-method correlated covariance and a linear log-shape assumption `c0 + c1*(theta/15) + c2*A**p`. This `[1,t,a]` shape is a modeling assumption, not established optics. The full-relative model adds three shape coefficients/prior dimensions. Fixed scale is a hypothesis, not established calibration. Finite uncertainty implements assumed covariance inflation `C33 += log-scale SD²` within the radial robust measurement model. It does not fit or validate a physical per-recording/temporally varying scale process, and is not equivalent to profiling a separate Gaussian scale prior under robust loss. Cross-scale costs omit covariance log-determinants and have different prior dimensions, so they must not select scale assumptions or rank likelihoods. Free framewise scale analytically recovers the matched normalized objective and drops the log-shape prior.

**Mean profiles:** Negative H0/FC0 probes were verified. Positive probes remain unresolved after guarded continuations and tighter inner solves: H0 original constrained KKT is about `1.014e-5`, FC0 about `7.535e-5`, above the unchanged `1e-5` acceptance threshold. Do not accept or connect them as verified profile samples. Original/continued/refined phases are retained under [profiles](../exp2/full_information/profiles/); summary code uses achieved means, open markers for unverified points, and avoids counting ancestor phases as independent samples.

The latest [actual gradient audit](../exp2/full_information/validation/protocol/profile_actual_gradient_audit.json) passed checkpoint/source reconstruction and found augmented `Jᵀr` and the correctly normalized encoded/physical envelope agreeing to approximately `1e-13`. There were no exact-knot/bound states under the current masks, but the worst accommodation coordinates lie roughly `2e-10` above the `A=2` knot. **True coefficient-reprofiled one-sided finite-difference investigation is still pending. Do not execute it during the pause.** The stall cause is not established; no further restarts or acceptance-threshold relaxation were approved.

**Sensitivity:** Half-strength nominal anchors and the deterministic perturbed start converged. Double-strength anchors stalled at `xtol` and are not accepted as converged. See [sensitivity Markdown](../exp2/full_information/sensitivity_summary/production_current_20261004/sensitivity_summary.md), [JSON](../exp2/full_information/sensitivity_summary/production_current_20261004/sensitivity_summary.json), [case CSV](../exp2/full_information/sensitivity_summary/production_current_20261004/sensitivity_cases.csv) and plot (plot omitted from this repository). These measure local regularization/initialization sensitivity, not truth or global uniqueness.

The current sensitivity summary includes all three actual directories: `sensitivity/H0_anchor0p5`, `sensitivity/H0_anchor2`, and `sensitivity/H0_perturbed_seed20261004` (seed `20261004`, converged). Earlier example directory names in command documentation should not be mistaken for the actual production paths.

## Numerical controls and experimental solver status

Production used radial robust loss `kappa=2`, curvature off, previous-mean prior off, unchanged nominal anchors/temporal terms, and initial-function prior strength `0.1` with common training-only inverse-squared-range precision. State bounds are gaze `[-20,20] deg`, accommodation `[0,6] D`; encoded coordinates are `theta/15` and `A**p/4**p`. Residual normalization is `sqrt(n)` with equal-fixation weighting.

Initial budgets were 600 evaluations/900 s, LSMR maximum 300 with absolute/relative tolerance `1e-7`, 24 robust outer iterations and 25 inner evaluations. TRF uses `ftol=xtol=1e-12`, `gtol=None`; callback acceptance remains physical gradient/step `1e-5`, relative cost `1e-10`, robust-weight change `1e-7`. Positive-profile refinements used LSMR tolerances `1e-10`, maximum 2,000 and a 600-s phase cap, without changing those physical gates. Per-phase checkpoint/history/LSMR artifacts and cumulative runtime/evaluation counts are preserved; consult each `solver_settings.json` rather than assuming identical numerical controls across phases.

Optional Newton is **experimental and not enabled in production**. Synthetic objective/gradient/QR/weight checks passed; see [Newton synthetic checks](../exp2/full_information/validation/protocol/full_information_newton_synthetic_checks.json). The real R3 fixed-state trial fell back to unchanged cold IRLS (69 iterations) and showed no speedup: [IRLS phase](../exp2/full_information/validation/protocol/r3_fixed_state_coefficient_default_irls_phase.json), [Newton phase](../exp2/full_information/validation/protocol/r3_fixed_state_coefficient_newton_phase.json). Warm-started fallback and further QR-centering changes are **proposals only**, not implemented or approved for production. Solver selection is separate numerical provenance, not a change to objective/policy guards.

## Before any explicit future resumption

1. Revisit the measurement question: what independent evidence would justify fixed/constrained scale, and can current capture/demand/time confounding be resolved without new acquisition?
2. Reassess the linear log-shape assumption and extra prior/capacity before attributing changes to a third measurement channel. Current incomplete R/U fits cannot answer that comparison.
3. Interpret the verified negative mean probes and anchor sensitivity as local regularized behavior; distinguish optical information from nominal target constraints. Resolve the near-knot positive stall scientifically/numerically only after explicit authorization, without treating a tolerance adjustment as evidence.
4. Decide what independently referenced randomized repeats/whole-recording validation are needed. The evaluator must remain `not_evaluable_without_independent_references` for the current archive.
5. Only then decide whether numerical acceleration or additional production fits serve the original question. Preserve all losing/incomplete cases and phase lineage; do not select a final result from held-out nominal agreement or cross-scale objective values.

No task-owned numerical processes remain. The existing temporary scripts `/tmp/profile_actual_gradient_audit.py`, `/tmp/r3_coefficient_phase.py`, and `/tmp/full_information_newton_checks.py` are complete, not half-written; they were preserved. No `__pycache__` is present under the new full-information root. Nothing was deleted during the pause.

Do not automatically extend the existing gradient audit with pending one-sided reprofiled finite differences, implement warm fallback, or refresh/select results. New task-owned untracked code, reports and outputs must be preserved. Temporary scripts/logs/caches should be distinguished from durable deliverables when the user explicitly authorizes cleanup or further work.
