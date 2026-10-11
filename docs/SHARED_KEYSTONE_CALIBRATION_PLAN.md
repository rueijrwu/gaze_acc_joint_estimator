# Corrective experiment plan: globally shared raw P1/P4 keystone

**Branch:** `exp5_distortion_model`  
**Basis commit:** `980d6d3640f06bf5823e8e908f31d422c94006a7` (Stage 06 complete, CPU audit PASS)  
**Status:** PROPOSED / NOT IMPLEMENTED  
**Paired audit:** [SHARED_KEYSTONE_CALIBRATION_AUDIT.md](SHARED_KEYSTONE_CALIBRATION_AUDIT.md)

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

Let \(t_x=\theta_x/10^\circ\); choose one common fixed \(u_y\) or use native degrees for \(t_y\). For P4, start with

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
| S1 — minimal \(A\) coupling | One \(K_4(\theta,A)\) with **only** global \(b_{0,A}\) added | Refitted shared \(\kappa(A)\) | **Mandatory hypothesis test** |
| S2 — optional second coupling | Add exactly one further identified shared \(A\)-dependent term | Refitted shared \(\kappa(A)\) | Only if S1 leaves reproducible geometry that the extra term can distinguish |
| Diagnostic free fit | Historical independent \(K_{4,c},\kappa_c\) | Independent per capture | In-sample flexibility ceiling only; not a valid final \(A\) estimator |

For the paired S0/S1 comparison, **freeze the same provisional gaze array** to isolate changes to the optical model. Then refit \(A_i\) consistently under each accepted candidate. Do not silently change gaze calibration during this first comparison.

Compare the same native-coordinate data error, not the numerical value of incomparable optimizer objectives. Because S0/S1 have fewer parameters than H, their in-sample error may be greater; a lower H residual is **not** by itself evidence against the shared law.

## 4. Execution sequence

### P0 — freeze and fingerprint controls

Archive or reference immutable hashes of the active raw Stage 03, 04, 05 and 06 `population.npz`, `frames.npz`, `summary.json`, `audit.json`, `protocol.json`, and parent optical/reference artifacts.

Record a single matched population and identical vertex order, including P4 permutation already applied once. Use separate new output paths, e.g. `experiments/reverse_transform/shared_keystone/stage_01_global_p1/` through dedicated global P4 and state-test stages. Do not overwrite `results/run` under existing stages.

### P1 — P1 model and common scale policy

For a physically shared system, the target is one \(K_1(\theta)\) shared by all captures, plus one \(g_i\) per frame. Fit it with the fixed P1 empirical reference using the same fitting population, native gaze and vertex metric.

To isolate the P4 issue, an initial **controlled S0/S1 optical ablation may temporarily freeze the existing Stage-06 gaze array and P1 scales**, provided both candidates use the *same* frozen inputs and results are labeled conditional. That is not the final globally consistent calibration, because Stage 06 inherited per-capture P1/gaze solutions.

After the first P4-only comparison, rerun/reconcile P1 globally and reprofile \(g_i\) at every gaze update. Quantify whether P1 itself exhibits capture effects; do not bury those inside a falsely universal \(K_1\).

### P2 — fit S0 as an all-capture shared P4 keystone

Fit one common four-coefficient \(K_4(\theta)\) to all four captures simultaneously using the same reference, current P1 scale, and a newly optimized shared \(k_A\).

As a reproducible initialization, set \(A_i\) to the nominal demand per capture, *explicitly as a starting approximation, not known physiological accommodation*. Then, holding the global optics fixed, fit framewise \(A_i\) using the same 0.25 D fixation-mean soft anchors and operational [0,6] D bounds as the raw baseline. If necessary, alternate shared optical updates with state updates under **one explicitly recorded joint objective**.

Use equal fixation weighting across all 20 fixations (and hence avoid the longest capture dominating), not an accidentally per-capture or frame-count-weighted cost. Enforce the valid forward domain on all frames.

### P3 — fit S1 with one shared accommodation-dependent keystone coefficient

From S0, release only \(b_{0,A}\) in

\[
b_0(A)=b_{0,ref}+b_{0,A}(A-A_{ref}).
\]

**Refit global \(K_4\) coefficients and radial \(k_A\)** together under S1. Do not initialize S1 with the frozen historical per-capture \(K_{4,c}\) as if they were physical constants; old coefficients may be used only as transparently labeled optimizer starts.

At each frame, the derivative required for \(A_i\) is

\[
\boxed{
\partial_A\widehat X_{4ij}
=
g_i\,C\!\left[
(\partial_B K_4)\,\partial_A B_4
+
(\partial_A K_4)_{B_4}
\right]_j ,
}
\]

for fixed gaze and P1-only scale. The second term is **absent** in the historical Stage-04/06 A-only model. It must enter optimization, gradient audits, and the inverse evaluation for S1.

The inverse is evaluated using the **same global coefficients at the same trial \(A_i\)**. Restore model optical centroid, divide by \(g_i\), undo the projective stretch/denominator and then invert the radial response. Do not add an RMS-size factor.

### P4 — state recovery under globally frozen optical laws

For each accepted S0/S1, freeze the global optical parameters and estimate framewise \(A_i\), initially at the same provisional gaze, with matched fixation-mean anchors.

Compute per-fixation A mean ± SD, bounds, native-camera P1/P4 residuals and inverse-reference residuals. In particular, compare Capture 1's Stage-06 means

\[
[0.56664,\,0.58709,\,0.36226,\,0.21374,\,0.24386]\,D
\]

over gaze \([-10,-5,0,+5,+10]^\circ\), against nominal 0.36036 D. Measure the range and signed dependence, but **do not force** the means to the nominal demand by tuning model terms beyond the declared soft anchors.

### P5 — only then reconcile gaze and full relative displacement

If a shared model improves coordinate behavior and provides an identifiable \(A\) derivative, consider a separate joint gaze+\(A\) iteration. At every trial gaze, reevaluate \(K_1\), \(g_i\), \(K_4(\theta_i,A_i)\), and the inverse center.

The historical Stage-06 per-capture gaze polynomials use P4–P1 information; they remain conditional initializations, not independent truth. Any replacement with a global relative-displacement \(D(\theta,A)\) law (as proposed in `docs/Theory.md`) is a **separate ablation after S0/S1**, not a covert modification inside the same test.

### P6 — held-P4 and cross-condition checks

Once the global model is frozen, perform three independent omit-one-P4-vertex trials. Infer state/scale using retained inputs only (all P1 points plus two P4 points), then predict the omitted P4 point in original relative pixels. No use of its measured coordinate through a full-three-point P4 centroid, cached inverse, label, initializer, gate, or covariance.

Also compare fits when withholding one gaze fixation from optical calibration (interpolation/extrapolation must be identified) and, where feasible, withholding an accommodation/capture condition for a stress test. With only four demands and one per capture, a leave-one-capture-out result is **weak/extrapolative** and cannot independently separate capture effects from \(A\).

## 5. Numerical and scientific audit contracts

### Exactness / reproducibility

- Synthetic raw forward/inverse closure at the declared domain and fixed \(A\).
- Independent CPU replay of all candidate forward predictions and inverses.
- Finite-difference checks of global \(b\), \(k_A\), state \(\theta,A\), and P1 profiled \(g\) derivatives, including \(\partial_AK_4\).
- Trial consistency: a proposed \(A\) updates radial shape, keystone, optical centroid and inverse together.
- Same complete-frame masks, references, correspondence, fixation weighting, gaze inputs and state-anchor widths across S0/S1.
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

## 7. Proposed implementation order (smallest useful next step)

1. Create a versioned global raw-keystone model class with **one shared native coefficient basis** (no per-capture K lookup).
2. Build an S0 all-capture P4 fitter with provisional gaze/P1 scale frozen as a controlled comparison; jointly refit radial law.
3. Add a *single* S1 global \(b_{0,A}\) option, correct \(\partial_A K\) and deterministic inverse.
4. Run CPU derivative/closure audits and matched S0-vs-S1 native-residual comparisons.
5. Reconcile shared P1 calibration and reprofile its framewise scale; rerun S0/S1.
6. Refit framewise A under globally frozen optics; evaluate angular bias and bounds.
7. Advance to joint gaze/relative-displacement inference and held-P4 prediction only if global response identification is credible.

**Do not change Stages 01–06 in place.** The first implementation deliverable should be a *new* controlled all-capture shared-keystone ablation, not a wholesale rewrite of the gaze estimator.