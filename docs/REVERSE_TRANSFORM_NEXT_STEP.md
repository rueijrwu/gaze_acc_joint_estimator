# Reverse-transform next-step plan: remove keystone size normalization

> **Latest handoff (2026-10-10):** The raw-keystone rebuild described below **has already been implemented** through raw Stage 06 at commit `980d6d3`. This file is retained as the historical pipeline-normalization correction plan. The **next unimplemented optical-model change** is a controlled global/shared P1/P4 keystone calibration: [audit](SHARED_KEYSTONE_CALIBRATION_AUDIT.md) and [implementation plan](SHARED_KEYSTONE_CALIBRATION_PLAN.md). Do not rerun the normalization removal as if raw Stages 01–06 were still pending, and do not promote independently fit `K_{4,c}` as a final physiological accommodation model.

**Branch:** `exp5_distortion_model`  
**Purpose:** Replace the size-normalized keystone convention with the direct physical/projective keystone transform across the **entire reverse-transform pipeline**, then rebuild Stages 01–05 in order before drawing further accommodation conclusions.  
**Status:** Planning document only. Existing Stage 01–05 results remain historical evidence under the old normalized-keystone convention.

## 1. Revised conclusion

The previous next-step plan assumed the existing keystone parameterization should remain fixed while testing gaze versus accommodation.

Stage 05 showed that freeing framewise gaze did not remove the gaze-dependent accommodation trend. A deeper implementation review identified a more fundamental issue:

> **The current keystone operator is explicitly normalized to remove its own RMS-size change.**

That normalization is not required by the optical transformation and prevents gaze from producing total P1/P4 size changes through the keystone/projective mapping.

The next experiment should therefore **not add another differential gaze-scale term yet**.

The first priority is:

\[
\boxed{
\text{remove the artificial keystone RMS normalization}
}
\]

and rebuild the full reverse-transform chain with the direct keystone transformation.

The correction is pipeline-wide: every stage whose coefficients, radial law, framewise A, or joint states were fit under the normalized-keystone convention must be treated as historical and recalibrated in sequence. Only after the raw P1/P4 transforms are rebuilt should accommodation be evaluated again.

## 2. Why the current normalization is problematic

The current implementation forms a projective/keystone-transformed triangle \(F\), centers it, and then rescales it back to the RMS radius of its pre-keystone input.

Schematically,

\[
F=K(\theta;B),
\]

\[
C(F)=F-\bar F,
\]

\[
s_K(\theta)=\frac{R[C(F)]}{R[B]},
\]

\[
H(\theta)=\frac{C(F)}{s_K(\theta)}.
\]

Therefore

\[
\boxed{R[H(\theta)]=R[B].}
\]

For P4 with a radial/accommodation pre-transform \(P(A)\),

\[
\boxed{R[H_4(\theta,A)]=R[P(A)].}
\]

After multiplying by the P1-derived frame scale \(g_i\),

\[
\widehat X_{4i}=g_iH_4(\theta_i,A_i),
\]

so

\[
\boxed{
R[\widehat X_{4i}]
=
g_iR[P(A_i)].
}
\]

Under that convention, gaze can change P4 shape but cannot change total P4 size after the common P1 scale is removed.

The measured Capture-1 data still show systematic P4/P1 relative size variation with gaze, while Stage 04/05 accommodation estimates vary strongly with gaze. This creates a direct route for the accommodation/radial state to absorb a real gaze-dependent size component that the normalized keystone is forbidden from representing.

## 3. Correct keystone model

The keystone/projective transform should be used directly.

For a local reference point

\[
\mathbf b=
\begin{bmatrix}
b_x\\b_y
\end{bmatrix},
\]

use

\[
\boxed{
K_r(\theta;\mathbf b)
=
\begin{bmatrix}
\dfrac{s_{x,r}(\theta)b_x}
{1+q_r(\theta)b_y}\\[2mm]
\dfrac{s_{y,r}(\theta)b_y}
{1+q_r(\theta)b_y}
\end{bmatrix}.
}
\]

The exact 2D implementation may retain the current horizontal/vertical generalized form, but **no post-keystone RMS-size rescaling is allowed**.

The output used for fitting is simply the centered physical/projective result:

\[
\boxed{
H_r(\theta)=C\!\left(K_r(\theta;B_r)\right).
}
\]

For P4 with relative radial deformation,

\[
\boxed{
H_4(\theta,A)
=
C\!\left(
K_4\!\left(\theta;R_A(B_4)\right)
\right).
}
\]

There is no division by

\[
R[C(K)]/R[B].
\]

Keystone is allowed to change both:

- triangle shape;
- total triangle size.

Both are legitimate consequences of the projective transformation.

## 4. Common P1 nuisance scale remains necessary

Removing keystone normalization does **not** mean introducing an uncontrolled frame scale.

The common nuisance magnification should still be estimated from P1.

For measured centered P1 coordinates \(X_{1i}\) and the raw gaze-dependent P1 prediction

\[
H_1(\theta_i)
=
C(K_1(\theta_i;B_1)),
\]

profile one positive scalar

\[
\boxed{
g_i(\theta_i)
=
\frac{
\langle X_{1i},H_1(\theta_i)\rangle
}{
\langle H_1(\theta_i),H_1(\theta_i)\rangle
}.
}
\]

Then

\[
\boxed{
\widehat X_{1i}
=
g_iH_1(\theta_i).
}
\]

The same \(g_i\) is applied to P4:

\[
\boxed{
\widehat X_{4i}
=
g_iH_4(\theta_i,A_i).
}
\]

This preserves the intended separation:

\[
\underbrace{g_i}_{\text{common external/frame magnification}}
\]

versus

\[
\underbrace{K_1,K_4}_{\text{gaze-dependent optical/projective size and shape}}
\]

versus

\[
\underbrace{R_A}_{\text{accommodation-dependent P4 deformation}}.
\]

Do not normalize \(K_1\) or \(K_4\) to constant RMS radius.

## 5. Gauge interpretation

Without the artificial keystone normalization, P1 has a scale gauge unless the global P1 keystone calibration is fixed.

In principle,

\[
K_1(\theta)\rightarrow c(\theta)K_1(\theta),
\qquad
g_i\rightarrow g_i/c(\theta_i)
\]

can preserve the P1 prediction.

This gauge must be resolved by the **global calibrated definition of the P1 keystone**, not by deleting its gaze-dependent size response.

The practical rule is:

- fit one shared P1 keystone function across the Capture-1 gaze conditions;
- fix its reference identity at zero gaze;
- use that fixed global function to define \(g_i\);
- then apply the same \(g_i\) to P4.

Once \(K_1\) is fixed, relative P1/P4 gaze magnification becomes observable through the difference between the two calibrated transforms.

## 6. Existing Stage 01–05 results become historical controls

Do not patch old coefficients by merely deleting the normalization.

The previous coefficients were fitted under a different model:

\[
\text{normalized keystone}+\text{profiled scale}.
\]

Removing the normalization changes the meaning of the keystone coefficients and the profiled frame scale.

Therefore:

- Stage 01 P1 coefficients must be refit;
- Stage 02 Capture-1 P4 coefficients must be refit;
- Stage 03 relative radial coefficients must not be reused as though unchanged;
- Stage 04 framewise A results remain a historical baseline;
- Stage 05 A0/G/GA results remain evidence that ordinary framewise gaze correction did not solve the bias under the old normalized model.

Do not compare new coefficients numerically with old coefficients as if they represented the same parameterization.


## 6A. Pipeline-wide replacement map

The raw-keystone correction applies to every active empirical stage. The corrected pipeline should be versioned as a new chain rather than editing historical outputs in place.

| Historical stage | Required corrected action |
|---|---|
| Stage 01 — Capture-1 P1 | Refit raw P1 keystone; profile one positive common scale from P1 under the raw gaze-dependent transform. |
| Stage 02 — Capture-1 P4 | Refit raw P4 keystone using the new Stage-01 P1 scale; keep reference relative radial increment at zero gauge for Capture 1. |
| Stage 03 — independent captures | Refit all P1/P4 raw-keystone coefficients and relative radial terms from scratch. Do not reuse old kappa values or the old beta law. |
| Stage 04 — framewise A | Rerun only after the corrected Stage-03 model and new accommodation/radial law are frozen. |
| Stage 05 — gaze/A ablation | Treat as historical evidence under the normalized model; rerun only if the corrected Stage-04 result still requires state-separation testing. |

No old fitted optical coefficient should be carried into the raw-keystone chain except as an optional numerical starting point that is explicitly labeled as such and fully reoptimized. Saved historical states must not be used as fixed truth in the corrected pipeline.

### Corrected Stage-03 accommodation/keystone policy

The corrected Stage 03 should not assume that accommodation enters only through a radial coefficient. After removing the keystone normalization, independently fitted capture coefficients should be inspected for systematic accommodation dependence.

The first shared P4 candidate remains

\[
K_4(\theta;B_4(A)),
\]

with A entering the P4 baseline/radial model. If coordinate residuals show systematic remaining changes in the keystone coefficients with accommodation, promote only the necessary terms to

\[
\boxed{K_4(\theta,A)}.
\]

Examples of minimal supported coupling are

\[
s_{x,4}(\theta,A)=1+[\alpha_{40}+\alpha_{4A}(A-A_{ref})]\theta^2,
\]

\[
q_4(\theta,A)=[\gamma_{40}+\gamma_{4A}(A-A_{ref})]\theta.
\]

The goal is not to maximize flexibility. It is to prevent the newly fitted radial/accommodation state from absorbing a reproducible accommodation-dependent keystone error.

Use a staged comparison:

1. raw keystone with A-independent P4 coefficients;
2. inspect coefficient/residual trends versus demand;
3. add one A-dependent keystone term at a time;
4. retain only terms that improve original-coordinate prediction and remain identifiable;
5. refit the radial/accommodation law after the accepted K4 model is fixed.

## 7. New experiment sequence

The new sequence should restart with **Capture 1 only**.

### R1 — raw-keystone P1 refit

Use the same Capture-1 complete-frame population, reference P1 triangle, gaze labels, correspondence, and equal-fixation policy.

Replace the P1 shape model with

\[
\boxed{
H_1(\theta)=C(K_1(\theta;B_1))
}
\]

with no RMS normalization.

For every trial global P1 keystone parameter set:

1. evaluate \(H_1(\theta_i)\);
2. profile one positive \(g_i\) from P1;
3. minimize original centered P1 camera-coordinate residuals.

Fit only the shared P1 keystone coefficients.

Do not add accommodation or P4 information to this stage.

#### R1 required checks

- positive \(g_i\) for all fitted frames;
- denominator/domain validity;
- analytic versus finite-difference gradients;
- synthetic forward/inverse closure;
- fixation-specific residuals;
- fixation-specific fitted \(g_i\);
- predicted raw-keystone P1 RMS size versus gaze.

The key result is the physical gaze-dependent size response encoded by \(K_1\), separately from \(g_i\).

### R2 — raw-keystone Capture-1 P4 refit

Freeze:

- the R1 P1 keystone;
- gaze calibration/state policy;
- P1-derived \(g_i\);
- empirical P4 reference.

Fit the Capture-1 P4 raw keystone:

\[
\boxed{
H_4(\theta)
=
C(K_4(\theta;B_4)).
}
\]

Use

\[
\boxed{
\widehat X_{4i}=g_iH_4(\theta_i).
}
\]

For Capture 1, keep the reference-relative radial increment fixed at

\[
\kappa=0.
\]

Do not add accommodation or an independent P4 scale.

#### R2 primary diagnostic

After the fit, compute the residual P4/reference size ratio after the common P1 scale and raw P4 keystone are accounted for.

The old normalized model produced approximately

\[
0.9914,\ 0.9904,\ 1.0000,\ 1.0071,\ 1.0064
\]

across the five gaze conditions.

The crucial question is:

> Does raw keystone substantially flatten this residual gaze-dependent P4/P1 size pattern?

If yes, the old accommodation-vs-gaze trend was at least partly caused by the normalization convention.

### R3 — inverse recovery validation

Implement the exact inverse of the **raw** fitted keystone.

Do not reuse the old inverse-size convention.

For the forward projective form

\[
\mathbf z=K(\theta;\mathbf b),
\]

the inverse should undo only the projective/scale terms actually present in \(K\), with no synthetic RMS factor.

Validate:

- synthetic forward/inverse closure;
- measured reference recovery;
- original-coordinate forward residuals;
- inverse-reference residuals;
- no hidden radius normalization.

The raw forward and inverse operators must be algebraic inverses under the declared model.

### R4 — rebuild Stage 03 and refit accommodation-dependent optics

Only after R1–R3 pass should the cross-capture accommodation-dependent model be rebuilt.

Do not reuse the old Stage-03 kappa values or beta directly.

Refit Captures 2–4 against the new raw-keystone model. First estimate the raw P1/P4 keystone and relative radial deformation independently per capture. Then inspect whether P4 keystone coefficients themselves vary systematically with accommodation/demand. If they do, fit the smallest shared K4(theta,A) dependence supported by coordinate residuals. Only after the accepted K4(theta,A) structure is fixed should the shared radial/accommodation law be fit.

The reference-relative radial model remains

\[
R_A(B)
=
\left(1+\kappa(A)r^2\right)B
\]

or the existing equivalent pixel-coordinate form.

But the newly fitted \(\kappa\) now operates in a model where gaze is allowed to contribute genuine size through keystone.

The recovered \(\kappa\)-versus-demand trend must be re-estimated from scratch.

### R5 — framewise A rerun

After the new radial law is frozen, rerun the framewise-A experiment with gaze initially fixed to the recalibrated gaze model.

The key result is not merely lower residual.

Check whether the fixation-mean A pattern becomes flatter across the five Capture-1 gaze conditions.

Compare directly with the old Stage-04 means:

\[
0.566,\ 0.592,\ 0.359,\ 0.196,\ 0.207\ {\rm D}.
\]

If the new raw-keystone model removes much of that gaze trend while preserving P4 fit quality, that supports the hypothesis that old A was absorbing normalized-away gaze size.

### R6 — only then reconsider joint gaze+A

Joint framewise gaze+A should **not** be the first rerun.

First establish the corrected raw optical transform.

If framewise A still shows substantial gaze bias after R1–R5, then repeat a controlled joint horizontal-gaze+A test using the raw-keystone model.

As before:

- recompute P1 \(g_i\) at every trial gaze;
- keep vertical gaze frozen initially;
- freeze global optical coefficients;
- report local gaze/A Jacobian conditioning;
- compare original-coordinate and inverse-reference residuals.

## 8. Do not add a differential gaze-scale term yet

A separate P4/P1 gaze-size term such as

\[
s_{41}(\theta)
=
1+c_1\theta+c_2\theta^2
\]

was previously proposed as a next ablation.

Defer this term.

The raw keystone already contains a legitimate gaze-dependent size response. Adding \(s_{41}\) before testing the unnormalized keystone would risk fitting the same missing effect twice.

Only introduce a separate differential gaze-scale function if:

1. raw P1/P4 keystone is fitted correctly;
2. the common P1 scale is recomputed consistently;
3. a reproducible residual P4/P1 size-vs-gaze pattern remains;
4. that residual cannot be explained by radial/accommodation deformation or known model mismatch.

## 9. Fitting and reporting metric

Use original centered camera coordinates as the primary fitting space.

For P1:

\[
e_{1i}
=
X_{1i}
-
g_i(\theta_i)H_1(\theta_i).
\]

For P4:

\[
e_{4i}
=
X_{4i}
-
g_i(\theta_i)H_4(\theta_i,A_i).
\]

Use equal-fixation aggregation as before.

Do not normalize measured triangles by area, RMS radius, or a fitted P4 scale.

Continue reporting inverse-reference recovery as a major validation diagnostic, but the fitting metric should remain in original centered camera coordinates unless an equivalent covariance-corrected inverse metric is explicitly established.

## 10. Required ablations

At minimum compare:

### Old model

Normalized keystone + P1 scale.

### New model

Raw keystone + P1 scale.

Use the exact same Capture-1 frames and reference.

Report by gaze fixation:

- P1 forward RMS/median/P95;
- P4 forward RMS/median/P95;
- P1 inverse RMS/median/P95;
- P4 inverse RMS/median/P95;
- fitted P1 \(g_i\) mean/SD;
- predicted raw-keystone P1 size factor;
- predicted raw-keystone P4 size factor;
- observed residual P4/P1 radius ratio;
- signed vertex residuals;
- domain/bound failures.

The central diagnostic is the residual P4/P1 size curve after the raw transform.

## 11. Decision logic

### Case A — raw keystone flattens P4/P1 size-vs-gaze and A bias falls

Interpret the old A-vs-gaze trend as partly caused by the artificial keystone size normalization.

Continue with the raw transform and refit the accommodation law.

### Case B — raw keystone improves size-vs-gaze but A remains biased

Then both effects may be present.

Proceed to the newly refitted accommodation deformation and, if needed, a later joint gaze+A analysis.

### Case C — raw keystone does not flatten the residual gaze-size curve

Only then consider a separate P4/P1 differential gaze-scale function.

Do not reintroduce normalization.

### Case D — raw keystone makes P1 scale/gaze poorly identifiable

Then the P1 keystone parameterization itself needs stronger global constraints or matched optical priors.

Do not solve this by returning to per-frame RMS normalization, because that deletes the physical size response being tested.

## 12. Immediate implementation order

\[
\boxed{
\begin{array}{l}
\textbf{1. Remove RMS normalization from P1 keystone.}\\
\textbf{2. Refit Capture-1 P1 raw keystone and profile common }g_i.\\
\textbf{3. Remove RMS normalization from P4 keystone.}\\
\textbf{4. Refit Capture-1 P4 raw keystone using the same }g_i.\\
\textbf{5. Validate exact raw forward/inverse closure.}\\
\textbf{6. Inspect residual P4/P1 size versus gaze.}\\
\textbf{7. Refit relative radial/accommodation response from scratch.}\\
\textbf{8. Rerun framewise A and inspect gaze bias.}\\
\textbf{9. Only if needed, rerun joint horizontal gaze+A.}\\
\textbf{10. Only after state separation, run held-P4 cross-agreement.}
\end{array}
}
\]

## 13. Compatibility and preservation

Preserve all existing Stage 01–05 outputs and source snapshots.

Do not overwrite canonical results.

The new raw-keystone pipeline should use new stage/result directories so the old normalized-keystone results remain available as controlled historical comparisons.

The scientific change is specifically:

\[
\boxed{
\text{keystone is a physical/projective transformation, not a shape-only transformation.}
}
\]

One common P1-derived nuisance scale remains, but no additional normalization is inserted inside the keystone operator.
