# Phase 8.2 audit — joint calibration sensitivity

**Repository:** `rueijrwu/gaze_acc_joint_estimator`  
**Branch / inspected commit:** `exp5_full` / `13baf946c8159234d6c4d1d45632ae0b5d658a12`  
**Date:** 2026-10-07

## Executive assessment

Phase 8.2 substantially follows the intended joint latent-state/coefficient sensitivity experiment. It genuinely refits global coefficients and framewise gaze/accommodation states, preserves grouped training/evaluation separation, retains failed fits, and evaluates the same three-P1/three-P4 cross-check contract. I found no new direct held-out-P4 leakage in the reviewed path.

The positive result is real but specific: stronger calibration constraints materially improve `conditional37` relative to its own baseline. This does **not** establish `conditional37` as superior to baseline `conditional27`, and it does not establish physiological accuracy.

The most important new theoretical audit finding is that the implemented conditional optical families have an exact state-scale reparameterization: coordinate predictions can remain unchanged while raw subset-state disagreement in degrees/diopters changes. Therefore excluded-P4 coordinate prediction must remain the primary metric; state agreement is a companion metric under a declared calibration convention.

## 1. Implementation audit

The sensitivity runner verifies frozen source hashes, training rows/group ordering, and disjoint training/evaluation groups. It jointly refits coefficients and latent states. Changed anchor scales are part of the calibration objective, not post-processing. Frozen same-fold state trajectories are used only as training initializers.

Every P4 holdout keeps all three P1 points as measured reference context. One P4 point is excluded, the other two P4 points estimate the shared state, and the excluded point is scored only afterward. The cross-check reports:

- `E`: RMS of the three excluded-P4 2D prediction errors.
- `G_theta`: RMS disagreement among the three subset gaze estimates.
- `G_A`: RMS disagreement among the three subset accommodation estimates.
- Bounds, coverage, worst-point error, rank/ambiguity, and complete-interior masks separately.

The experiment contains eight predeclared one-factor settings over nine outer folds and two capacities: 144 task outcomes. The saved run reports 143 accepted calibrations and one uncertified calibration. This is development sensitivity, not nested model selection.

## 2. What Phase 8.2 actually improves

Published equal-fixation RMS values:

| Comparison | E (px) | G_theta (deg) | G_A (D) |
|---|---:|---:|---:|
| Gaze/27 baseline | **3.4207** | **0.2694** | **0.3111** |
| Gaze/37 baseline | 6.7005 | 0.5093 | 0.9290 |
| Gaze/37 strong anchors | 4.7637 | 0.4397 | 0.5297 |
| Gaze/37 strong prior | 4.7012 | 0.4285 | 0.4615 |
| Capture/27 baseline | **4.0144** | 0.2806 | **0.3194** |
| Capture/37 baseline | 4.3066 | 0.3250 | 0.3917 |
| Capture/37 strong anchors | 4.2928 | **0.2764** | 0.3665 |

All rows above retain 429 scored slots.

For gaze/37, strong anchors reduce full-population E by about 29%, G_theta by 14%, and G_A by 43%, while bound slots fall from 79 to 29. This is a genuine same-population improvement relative to baseline37.

However, on the exact 107 shared-interior gaze frames, the change is much smaller:

- E: 4.664 -> 4.571 px
- G_theta: 0.412 -> 0.409 deg
- G_A: 0.375 -> 0.375 D

This is consistent with much of the full-population gain occurring in difficult/bound-related regimes rather than a large uniform interior improvement. A boundary-transition decomposition is needed before making that causal claim.

For capture/37 strong anchors, shared-interior E is nearly unchanged (4.075 -> 4.058 px), while G_theta improves 0.338 -> 0.278 deg and G_A improves 0.413 -> 0.371 D. Here the main benefit is improved subset-state consistency at almost unchanged coordinate error.

Weakening the prior illustrates why E and state agreement must remain separate. Capture/37 E improves 4.3066 -> 3.7595 px, but G_theta worsens 0.3250 -> 0.6978 deg and G_A worsens 0.3917 -> 0.5537 D. This is not an overall shared-state improvement.

**Conclusion:** keep baseline `conditional27` as the development reference. Treat strong-anchor `conditional37` as the most interesting richer challenger, not a replacement.

## 3. New finding: optical state-scale freedom

The implemented bases use

[
t=\theta/10,\qquad L=\log(1+A).
]

For positive constants `g,h`, define

[
\theta'=g\theta,\qquad A'=h(1+A)-1.
]

Then `t'=g t` and `L'=L+log(h)`. The implemented polynomial basis is closed under this transformation: global coefficients can be transformed so that

[
F(\theta',A';\beta',r)=F(\theta,A;\beta,r)
]

for every P1 context while corresponding subset-state differences scale as

[
G_\theta'=gG_\theta,\qquad G_A'=hG_A.
]

Independent synthetic algebra checks reproduced this for both capacities to floating-point precision (maximum normalized prediction difference below 1.8e-15).

This does **not** mean an already calibrated estimator can arbitrarily rescale its state. Fixed nominal anchors, the coefficient prior, computational bounds, and calibration conventions break/restrict this freedom. It means the optical observations alone do not determine every physical state scale when coefficients and states are jointly free.

### Consequence

Excluded-P4 coordinate prediction `E` should remain primary. `G_theta` and `G_A` are valuable companion consistency metrics, but they must be interpreted under the declared calibration convention and should not be optimized in isolation.

Add this symmetry to `Theory.md` and add a regression test. On saved matched training rows, also estimate descriptive positive gains between baseline and sensitivity `theta`, and between baseline and sensitivity `1+A`, then report the residual trajectory difference. Do not posthoc rescale evaluation states to improve reported agreement.

## 4. Covariance x4 and strong anchors are the same objective direction

Under the current normalized coefficient prior, multiplying coordinate covariance by four and halving both anchor scales produce objectives differing only by an overall factor:

[
J_{cov4}=(O+P)/4+H,
]

[
J_{anchor-strong}=O+P+4H=4J_{cov4}.
]

The saved verification confirms essentially identical fitted states across all 18 fold/capacity pairs.

Therefore `covariance_4x` and `anchor_strong` are **not independent evidence** that two different interventions help. Covariance x4 is useful as an algebra/control check, not as validation that noise inflation is appropriate.

Also note that identical minimizers do not imply identical branch/uncertainty diagnostics: the inverse currently uses absolute cost-gap thresholds. Uniform covariance scaling changes those cost gaps. Add a regression test distinguishing objective rescaling from a genuine statistical covariance change.

## 5. Remaining calibration certification issue

`prior_weak/capture_1/conditional27` was correctly rejected. Its selected start stopped with physical projected stationarity 0.0031837, above the required 0.001. A preserved 100-evaluation continuation reduced it to 0.0018794 but still did not pass.

Do not weaken the acceptance gate. Add an explicit continuation/preconditioning path for solver-stop versus independent-stationarity mismatches. The failed fold removes 117 otherwise-valid slots, so its aggregate capture/27 prior-weak score must not be compared as if coverage matched baseline.

## 6. Recommended next work

### P0 — Existing-result diagnostics first

Before new fitting, decompose candidate-vs-baseline changes into:

1. interior -> interior,
2. bound -> interior,
3. interior -> bound,
4. bound -> bound.

Preserve exact memberships and fixation/signed-gaze/capture strata. Compute paired squared-error changes, not differences between unrelated RMS summaries.

Also compute the descriptive state-scale gain diagnostics above.

### P1 — Separate gaze and accommodation anchoring

Phase 8.2 changed both anchor scales together. Add only the two missing conditions:

| Condition | Gaze anchor scale | Accommodation anchor scale |
|---|---:|---:|
| baseline | 0.10 deg | 0.25 D |
| **gaze-only strong** | **0.05 deg** | 0.25 D |
| **accommodation-only strong** | 0.10 deg | **0.125 D** |
| both strong (existing) | 0.05 deg | 0.125 D |

These are soft fixation-mean penalty scales, not measured physiological uncertainties. This isolates whether richer-model stabilization comes mainly from gaze calibration, accommodation-demand anchoring, or both.

### P2 — Proceed with Phase 8.3 information ablation

Use at least:

- baseline `conditional27`
- strong-anchor `conditional37`

as contrasting frozen responses.

For every excluded P4 point compare retained x-only versus retained x/y using the same trained response, all three P1 references, and the same excluded point. Report E, G_theta, G_A, worst-point/axis error, rank, branches, bounds, and coverage separately.

Do not allow a full-P4 centroid or triangle area containing the excluded point into its inverse.

### P3 — Selectively regularize the extra curvature

Before adding new response terms, test a nested penalty that shrinks only the ten extra `t^2`/`t^2 L` coefficients of conditional37 toward zero (the conditional27 nested case), while jointly refitting shared coefficients/states.

This is more interpretable than changing the global prior on every coefficient. Run it only after residual and Phase 8.3 evidence justifies retaining the richer capacity.

### P4 — Preserve independent evaluation

The current folds are development data because their results now guide model choices. Final ranking requires nested/grouped or new independent evaluation. Keep captures 5/6 untouched until the choices are frozen.

## Bottom line

Phase 8.2 is useful and largely faithful to the plan. It demonstrates that the richer response can cross-predict substantially better when calibration is constrained more strongly, while weak constraints expose large state instability. It does not establish a new overall winner or physiological accuracy.

The next work should isolate calibration-scale/anchor effects and the incremental information carried by x/y geometry rather than increasing model capacity indiscriminately.

Maintain the central three-pair criterion:

> A shared gaze/accommodation state inferred without one P4 point should predict that excluded P4 point, while subset states remain mutually consistent under a clearly specified calibration convention.

## Source navigation

Audit evidence is based on commit `13baf946c8159234d6c4d1d45632ae0b5d658a12`:

- `full_position/sensitivity.py`
- `full_position/calibrate.py`
- `full_position/model.py`
- `full_position/crosscheck.py`
- `full_position/invert.py`
- `tests/test_sensitivity.py`
- `experiments/full_position/joint_sensitivity_v1/RESULTS.md`
- `experiments/full_position/joint_sensitivity_v1/config.json`
- `experiments/full_position/joint_sensitivity_v1/verification.json`
- `experiments/full_position/joint_sensitivity_v1/calibration_diagnostics.csv`
- `experiments/full_position/joint_sensitivity_v1/supplemental_stop_check.json`
- `ESTIMATOR_PLAN.md`

Verification boundary: this audit reviewed source and saved evidence and ran independent algebra checks; it did not rerun all 144 calibrations, raw-frame aggregation, or the repository's reported 52-test suite.
