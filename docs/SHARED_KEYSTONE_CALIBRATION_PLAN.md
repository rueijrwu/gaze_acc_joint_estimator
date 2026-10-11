# Corrective experiment plan: globally shared raw P1/P4 keystone

**Branch:** `exp5_distortion_model`  
**Basis commit:** `980d6d3640f06bf5823e8e908f31d422c94006a7` (Stage 06 complete, CPU audit PASS)  
**Status:** SUPERSEDED EXECUTION SEQUENCE / MODEL-COMPARISON COMPANION<br>
**Paired audit:** [SHARED_KEYSTONE_CALIBRATION_AUDIT.md](SHARED_KEYSTONE_CALIBRATION_AUDIT.md)

The current execution specification is [NEW_PROTOCOL_PLAN.md](NEW_PROTOCOL_PLAN.md), Revision 2, incorporating [NEW_PROTOCOL_PLAN_AUDIT.md](NEW_PROTOCOL_PLAN_AUDIT.md). Mean initialization now supplies frozen anchors, followed by framewise alternation and joint S0 calibration; S1 is conditional on the completed S0 evidence. This companion retains the shared-law hypothesis and comparison contracts. Neither plan has been implemented.

## 0. Objective and acceptance question

**Primary hypothesis:** Current Stage-03 capture-specific \(K_{4,c}\) may have absorbed some of the accommodation-dependent P4 deformation; a globally shared \(K_4(\theta,A)\) could preserve that signal for per-frame \(A\) estimation.

**Primary test:** With equivalent data, weighting, empirical references, nuisance scale, gaze policy, and radial modeling, does a shared \(K_4(\theta,A)\) yield a reproducible, identifiable accommodation response that a shared \(K_4(\theta)\) cannot explain?

The objective is **not** to force a flatter inferred \(A\) curve or achieve the lowest training SSE. The key is a physically coherent *single response law* across captures, with defensible out-of-sample geometry and state sensitivity.

Historical raw Stages 01–06, normalized controls, source snapshots and reports remain immutable. This document authorizes no implementation or change to canonical results by itself.

## 1. Model invariants

All model candidates must use the same:

1. P1/P4 three-point correspondence and the same complete-frame population: 89,175 frames over four captures, 20 nominal fixations; preserve missing-frame accounting and source-frame identities.
2. Capture-1 nominal zero-gaze **empirical non-equilateral P1 and P4 references** (nominal \(A_{ref}=0.36036036036036034\,D\)); never regularize triangles toward an equilateral shape.
3. **Raw**, unnormalized projective keystone: apply baseline radial deformation, then keystone, then center only for a translation-free observation. No keystone RMS/area normalization, no independently fitted P4 magnification.
4. One positive frame scale \(g_i\), profiled **only from P1**, shared unchanged with P4 and recalculated at every trial gaze.
5. A declared fixed/native angle basis common to every capture. Preserve units when converting the existing four scaled keystone coefficients; in particular, Stage-03 \(u_{y,c}\) differs by capture. Do not pool raw scaled coefficients directly.
6. An explicitly anchored reference radial gauge: \(\kappa(A_{ref})=0\). Refit the slope when \(K_4\) changes.
7. Positive forward projective denominator, positive stretch, valid radial branch, and exact restoration of the modeled optical centroid during inversion.
8. Native corresponding-vertex forward residual as the primary fit/evaluation metric; reverse-reference residuals as secondary diagnostics.

No free per-capture P4 keystone coefficient, isotropic P4-only scale, arbitrary gaze-dependent scalar normalization, or free framewise radial coefficient is allowed in the **final shared-law candidates**.

## 2. Parameterization

Let \(t_x=\theta_x/10^\circ\) and \(t_y=\theta_y/10^\circ\) in every capture. For P4, start with

\[
B_{4j}(A)=
[1+\kappa(A)\|B_{4j}^{ref}\|^2]B_{4j}^{ref},
\qquad
\kappa(A)=k_A(A-A_{ref}).
\]

For a minimal accommodation-dependent projective scale, define

\[
a(\theta,A)=
\big[b_{0,ref}+b_{0,A}(A-A_{ref})\big]t_x^2-b_1t_y^2,
\]

\[
d_j(\theta,A)=1+
b_2t_x\,B_{4j,y}(A)/R_4+
b_3t_y\,B_{4j,x}(A)/R_4,
\]

\[
F_{4j}(\theta,A)=
\frac{\operatorname{diag}(e^a,e^{-a})B_{4j}(A)}
{d_j(\theta,A)},
\qquad
\widehat X_{4ij}=g_i\,[F_{4j}-\overline F_4].
\]

All \(b\) and \(k_A\) are **global, across-capture parameters**. A single function is evaluated for each frame. There is no capture index in this optical response.

This matches the existing reciprocal-exponential/geometric-denominator raw-keystone structure, except for the carefully controlled shared \(b_{0,A}\) term. More explicit \(A\) dependence in other coefficients is a later model expansion, not the initial ablation.

Keep the empirical reference gauge and all declared optical-origin conventions fixed through the paired test. Optical-zero offsets or image-axis rotations, if changed, require their own documented ablation.

## 3. Required model comparisons

| ID | P4 keystone | Radial response | Role |
|---|---|---|---|
| H — historical | Four independent \(K_{4,c}(\theta)\) | Historical fitted shared \(\kappa(A)\) | Read-only Stage 03–06 benchmark; potentially overflexible, not an authoritative state law |
| S0 — shared baseline | One \(K_4(\theta)\), all captures | Refitted shared \(\kappa(A)\) | **Mandatory primary control** |
| S1 — minimal \(A\) coupling | One \(K_4(\theta,A)\) with **only** global \(b_{0,A}\) added | Refitted shared \(\kappa(A)\) | Conditional hypothesis test after S0 |
| S2 — optional second coupling | Add exactly one further identified shared \(A\)-dependent term | Refitted shared \(\kappa(A)\) | Only if S1 leaves reproducible geometry that the extra term can distinguish |
| Diagnostic free fit | Historical independent \(K_{4,c},\kappa_c\) | Independent per capture | In-sample flexibility ceiling only; not a valid final \(A\) estimator |

For the paired S0/S1 comparison, use the same references, frame population, initial gaze/optical/accommodation anchor targets, widths, weighting and constraints from the current protocol. Both candidates may jointly update states under their common declared policy. A separately labeled fixed-gaze comparison can isolate the added optical term; freezing Stage-06 gaze and scales is an optional historical control and is not the primary execution path.

Compare the same native-coordinate data error, not the numerical value of incomparable optimizer objectives. Because S0/S1 have fewer parameters than H, their in-sample error may be greater; a lower H residual is **not** by itself evidence against the shared law.

## 4. Current execution sequence

The exact operator, numeric defaults, anchor construction, constraints and stopping rules are defined in [NEW_PROTOCOL_PLAN.md](NEW_PROTOCOL_PLAN.md). Follow this order:

1. Fingerprint the existing raw Stage 01–06 controls and preserve their canonical results.
2. Initialize gaze polynomials from arithmetic fixation means; calibrate provisional shared P1/P4 keystone from Capture 1 while profiling P1 magnification.
3. Initialize radial response from all captures' zero-gaze observations using a trial-consistent centered forward fit. Any inverse mean initializer uses a declared provisional radial seed and restores its modeled optical mean.
4. Fit the initial shared radial slope using nominal labels to establish the diopter scale. Construct one mean-accommodation target from each capture's zero-gaze P4 estimate, reuse it for that capture's five fixation-mean anchors, and freeze all targets and penalty widths. Primary width is 0.25 D; nominal-demand targets are a declared control.
5. Alternate Capture 1 shared keystone/gaze proposals, all-capture zero-gaze radial-law proposals, and **all-frame accommodation updates**. Accept/damp proposals against the full objective. This allows nonzero-gaze Capture 1 accommodation to feed back into keystone calibration.
6. Jointly refine shared P1/P4 keystone, radial slope, and framewise gaze/A across all captures. Reprofile P1 magnification and modeled optical means consistently; certify global and state stationarity under the same objective and constraints.
7. Evaluate anchor sensitivity and known data-only scale symmetries. Retain separate data and regularized information diagnostics.
8. If S0 evidence supports an additional response, fit S1 with one global horizontal accommodation-dependent keystone term, refitting the remaining optics and states. Include radial and explicit keystone contributions in the A derivative and inverse.
9. Run the declared training/test and retained-input P4 checks, with training-only references/anchors and no omitted-coordinate leakage. A full relative-displacement D model remains a separate comparison.

The first joint model centers P1 and P4 separately and uses polynomial gaze anchors. This is conditional shape-plus-gaze-anchor inference. It does not implement Theory's full relative-center observation model. Historical fit coefficients and nominal demand are not framewise physical truth.

## 5. Numerical and scientific audit contracts

### Exactness / reproducibility

- Synthetic raw forward/inverse closure at the declared domain and fixed \(A\).
- Independent CPU replay of all candidate forward predictions and inverses.
- Finite-difference checks of global \(b\), \(k_A\), state \(\theta,A\), and P1 profiled \(g\) derivatives, including \(\partial_AK_4\).
- Trial consistency: a proposed \(A\) updates radial shape, keystone, optical centroid and inverse together.
- Same complete-frame masks, references, correspondence, fixation weighting, initial gaze anchors, state-update policy and state-anchor widths across S0/S1.
- Positive P1 scale, finite denominators, monotonic radial branch, bound/active-constraint reporting; do not silently drop invalid measured inversions.

### Identification / sensitivity

- Joint global Jacobian/SVD or profile objective for \(b_{0,A}\) versus \(k_A\), including nuisance parameters; show tradeoffs and confidence/curvature caveats.
- Local state Jacobians \(\partial X/\partial\theta\), \(\partial X/\partial A\) for P4-only and P1+P4; report scale-invariant column angles and unit-dependent singular values/condition numbers.
- Compare per-capture fit residuals, per-fixation signed vertex vectors, recovered/reference P4 triangle radii, gaze bias and inverse tails.
- Test stability of shared coefficient estimates under multiple initializations and reasonable fixed calibration variants. Any coefficient pegged at its bound is a warning, not a calibrated physical value.
- Analyze whether nominal demand's perfect alignment with capture index makes any claimed accommodation parameter indistinguishable from drift; explicitly report unresolved confounding.

### Outcome categories

| Result | Interpretation / action |
|---|---|
| S0 reproduces the camera data with defensible held-out predictions and stable \(A\) | Prefer simpler globally shared \(K_4(\theta)\); independent \(K_{4,c}\) was unnecessary flexibility |
| S1 improves held-out prediction and adds a stable, identifiable \(A\) derivative | Retain globally shared \(K_4(\theta,A)\); refit A and retest gaze bias |
| S1 improves only in-sample SSE or is highly correlated with \(k_A\) | **Do not promote** \(A\)-dependent keystone yet; retain S0 or report a combined effective response |
| Both shared models underfit with systematic capture residuals | Check alignment, optical configuration, gaze calibration and measurement policy; do **not** immediately restore unrestricted per-capture K as a physiological model |
| \(A\) fixation bias remains despite better optical prediction | Investigate missing shared geometry/relative displacement, P1 scale/gaze calibration, or unmodeled optics in an explicitly separate experiment |

No hard physiological accuracy claim is permitted from same-recording point error or optimizer convergence alone.

## 6. Reporting requirements

Produce a new saved run directory for each candidate (S0/S1, optional S2), each including:

- `protocol.json`: model ID, frozen/free coefficients, native units/gauges, reference, weighting, mean anchors, validity rules, frame count.
- `provenance.json` and immutable source snapshot: full parent hashes and complete implementation inputs.
- `summary.json`: objective decomposition, coefficients and physical units, convergence and multistart results, capture/fixation statistics, sample counts, forward and inverse metrics.
- `audit.json`: independent CPU replays, derivative/closure checks, bounds, conditioning, omission-data noninterference.
- `frames.npz`: matched row/record IDs, inputs, inferred states, predicted and recovered coordinates, residuals, per-frame Jacobian diagnostics and validity masks.
- `RESULTS.md` and `SCIENTIFIC_REVIEW.md`: explicit S0-vs-S1 matched comparison, separate capture/demand confounding statement, and decision to accept/reject added \(A\)-dependence.

Suggested plots: per-capture P4 signed residual vectors versus gaze; \(b_0(A)\) versus nominal demand with no misinterpretation of four points as independent physiology; \(k_A/b_{0,A}\) profile; Capture-1 inferred A vs gaze; inverse-tail distribution; omit-one-P4 predictions.

## 7. Implementation boundary and artifacts

The current sequence uses new folders under `experiments/reverse_transform/shared_keystone/`: `stage_01_mean_initialization`, `stage_02_framewise_alternation`, `stage_03_joint_refinement`, optional `stage_04_a_dependent_keystone`, and `stage_05_validation`. Detailed manifests and acceptance criteria are in the current protocol.

Preserve raw Stages 01–06. Start with mean initialization, explicit frozen anchors and the repaired all-frame accommodation feedback, then complete the shared S0 joint objective. A fixed-Stage06-input P4 ablation may be reported as an additional control; it does not replace this revised sequence. No new implementation or numerical results are claimed by these document changes.
