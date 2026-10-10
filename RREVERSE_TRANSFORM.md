# Reverse-transform / empirical-reference audit

**Repository / branch:** rueijrwu/gaze_acc_joint_estimator / exp5_full  
**Audit basis:** estimator commit 37d2181ee0fbd43c1eaa6f38b26b0ab5df32475c  
**Optical evidence checked:** rueijrwu/distortion_tracking main at 7b3f28e7a7817aa701c84d1ecba13dbdd229d96d  
**Status:** design audit only. No reverse-transform estimator, new calibration, or physiological validation is claimed here.

## 1. Executive conclusion

The reverse-transform idea is scientifically useful and is compatible with the current joint optical theory if it is defined as **recovery of one fixed empirical reference pattern through the inverse of the same calibrated forward model**.

The correct target is the measured, non-equilateral empirical triangle, or one globally shared refinement of that triangle. It should not be forced into an equilateral triangle or perfect grid.

The intended interpretation is

\[
\boxed{
\text{measured relative pattern at }x_i
\;\xrightarrow{\mathcal T_{x_i}^{-1}}\;
\text{latent reference coordinates}
\;\xrightarrow{\mathcal T_{x_{\rm ref}}}\;
\text{the same empirical reference pattern}.
}
\]

where

\[
x_i=(\theta_i,A_i).
\]

This formulation is attractive because it makes the accommodation hypothesis concrete: after correcting nuisance scale and gaze-dependent deformation, the accommodation state should be the one whose shared optical response returns the P4 geometry toward the same reference pattern.

However, the audit recommends **not replacing the forward native-coordinate objective with an unweighted inverse-space error**. The safest initial architecture is:

1. fit and compare models with the forward residual in original same-frame relative image coordinates;
2. use reverse-to-reference recovery as the physical interpretation and diagnostic;
3. allow an inverse-space metric only after its covariance/Jacobian propagation is validated.

The central scientific risk is identifiability. A non-equilateral measured triangle is helpful, but unequal side lengths alone do not prove that accommodation-dependent magnification and radial distortion are independently observable. The decisive geometry is the set of point radii and directions relative to a constrained distortion center, after gaze, nuisance scale, reference gauge, and center uncertainty are included.

## 2. Current repository state

The latest docs/Theory.md already contains most of the correct ingredients:

- P1-reference scale fitting is the primary nuisance-scale normalization.
- Triangle area and the P4/P1 area ratio are optional historical diagnostics rather than required observations.
- P4 accommodation response is decomposed into accommodation-dependent magnification, radial distortion, gaze/keystone deformation, and axial nuisance scale.
- The empirical P4 baseline is recognized as already distorted.
- Relative P4-P1 centroid displacement is retained rather than erased by independent centering.
- Framewise state is one horizontal gaze and one accommodation value.
- Shared optical functions are calibrated globally.
- Held-P4 cross-agreement freezes the global calibration and reruns state inference.

These are compatible with a reverse-reference interpretation.

There is, however, an important documentation conflict:

- docs/Theory.md uses gaze-conditioned P1 reference-scale normalization.
- docs/ESTIMATOR_PLAN.md still states that P1 triangle-area normalization is required and describes changing the P1 normalizer as a later experiment.
- docs/ACCOMMODATION_FULL_CALIBRATION_PLAN.md still defines the older area-normalized conditional six-coordinate model as the working invariant.

This mismatch is a blocking implementation hazard. The historical area-normalized model should remain available as a control, but the new reverse-transform work should not silently mix the two contracts.

## 3. Preserve the empirical non-equilateral reference

Let the empirical P4 reference be

\[
\mathcal B^{\rm ref}
=
\left\{
\mathbf B^{\rm ref}_1,
\mathbf B^{\rm ref}_2,
\mathbf B^{\rm ref}_3
\right\}.
\]

The desired recovery condition is

\[
\boxed{
\mathbf B^{\rm rec}_{ij}
\approx
\mathbf B^{\rm ref}_j.
}
\]

It is **not**

\[
\|\mathbf B^{\rm rec}_1-\mathbf B^{\rm rec}_2\|
=
\|\mathbf B^{\rm rec}_2-\mathbf B^{\rm rec}_3\|
=
\|\mathbf B^{\rm rec}_3-\mathbf B^{\rm rec}_1\|.
\]

The fixed asymmetry can contain source placement, camera alignment, baseline optical distortion, and other reproducible geometry. Those effects should not be reassigned to framewise accommodation.

### 3.1 Do not force a regular grid or equilateral triangle

A fixed map that makes the empirical reference look regular can be used as a plotting coordinate convention. If the complete optical model and covariance are transformed consistently, it creates no new information.

It must not be treated as evidence that the physical reference itself was a perfect grid.

A regular equal-radius configuration can actually reduce radial-distortion observability. For the radial model

\[
\mathbf B_j(A)
=
\left[
1+\kappa(A)\|M(A)\mathbf u_j\|^2
\right]
M(A)\mathbf u_j,
\]

if all three points have equal radius \(R\) from the distortion center,

\[
\mathbf B_j(A)
=
M(A)
\left[
1+\kappa(A)M(A)^2R^2
\right]
\mathbf u_j.
\]

Magnification and radial distortion then collapse into one effective scalar.

The empirical triangle is non-equilateral, which may help. But non-equilateral side lengths do not by themselves guarantee distinct radii around the distortion center. Identifiability must be evaluated with the actual reference coordinates and center convention.

### 3.2 Prefer a consensus empirical reference over one noisy frame

One high-quality frame can initialize the reference, but a final calibration should test whether the result depends materially on that choice.

Prefer either:

1. one fixed empirical reference estimated robustly from several appropriate observations; or
2. one globally shared reference refinement under explicit origin, orientation, and scale gauges.

Do not allow a different reference triangle for each frame, accommodation value, or held-P4 subset.

## 4. What the reverse transform should mean

Let the local P4 forward map be

\[
\mathcal F_{\theta,A}
=
K_4(\theta,A)\circ\mathcal R_A,
\]

with

\[
\mathcal R_A(\mathbf u)
=
\left[
1+\kappa_4(A)\|M(A)\mathbf u\|^2
\right]
M(A)\mathbf u.
\]

The full relative observation also contains nuisance scale and the relative optical-origin / centroid-displacement law. Therefore the physical inverse cannot be applied directly to raw P4 image points as though the P4 centroid were the radial center.

After same-frame translation removal, define the P1-scale-corrected relative measurement

\[
\widetilde{\mathbf q}_{ij}
=
\frac{
\mathbf q_{ij}-\mathbf c_{1i}
}{
\widehat g_{P1,i}(\theta_i)
}.
\]

In the common-scale starting model, current theory predicts

\[
\widetilde{\mathbf q}_{ij}
\approx
\mathbf h(\theta_i,A_i)
+
\mathbf S_{4j}(\theta_i,A_i).
\]

The conceptual reference recovery is

\[
\boxed{
\mathbf B^{\rm rec}_{ij}
=
\mathcal T_{x_{\rm ref}}
\left[
\mathcal T_{x_i}^{-1}
\left(
\widetilde{\mathbf q}_{ij}
\right)
\right].
}
\]

The transformation \(\mathcal T\) must include the declared relative-origin convention, not only a radial warp around an observed centroid.

## 5. Inverse order is mandatory

For the local P4 forward transformation

\[
\mathbf P^{\rm local}_{4,j}
=
g_4\,
K_4
\left(
\theta,A;
\mathcal R_A(\mathbf u_j)
\right),
\]

the reverse path must undo operations in reverse order:

1. remove the external/common nuisance scale using the P1-derived scale policy;
2. remove the modeled P4-P1 relative-origin / centroid displacement;
3. apply the inverse keystone or projective transformation;
4. invert the accommodation-dependent radial/magnification transformation;
5. apply the reference-state radial/magnification transformation;
6. apply the reference-state keystone transformation;
7. restore the reference-state relative-origin convention.

Do not commute these stages without proof. Scaling inside a projective denominator is not generally equivalent to scaling outside it.

### 5.1 Radial inverse

For a radial map such as

\[
\mathbf y
=
\left(
1+\kappa r^2
\right)
\mathbf x,
\]

the inverse is not obtained by simply replacing \(\kappa\) with \(-\kappa\).

The inverse should solve the one-dimensional radial equation using a bounded monotone root solve or another certified method. The accepted domain must satisfy the one-to-one conditions, including checks associated with

\[
1+\kappa r^2
\]

and

\[
1+3\kappa r^2.
\]

Synthetic round-trip tests must verify

\[
\mathcal T_x^{-1}
\left(
\mathcal T_x(\mathbf u)
\right)
\approx
\mathbf u
\]

over the full accepted gaze, accommodation, and field domain.

## 6. A critical requirement: reverse transformation needs an optical origin

The current forward theory can be written with a directly fitted relative centroid law. That is sufficient for predicting measured relative coordinates.

A physical radial inverse of individual P4 points needs more: it needs a defined local coordinate system and radial center.

Therefore, before implementing a reverse radial transform, choose one explicit parameterization:

1. fit a shared relative optical-origin displacement and derive the centroid law;
2. use an equivalent shared local-coordinate parameterization that reconstructs the uncentered local P4 coordinates; or
3. use matched optical constraints that fix the required radial center / origin.

Do not silently use the measured P4 centroid as the radial center.

The center must be fixed or globally constrained. A different radial center per frame would be flexible enough to absorb state-dependent deformation and undermine the physical interpretation.

## 7. Do not invert a centered triangle as if centering commuted with optics

This is a major mathematical point.

The centered P4 shape is

\[
\mathbf S_{4j}
=
\mathbf F_{4j}
-
\overline{\mathbf F}_4.
\]

For a nonlinear radial/projective transformation \(K\), in general

\[
K^{-1}
\left(
\mathbf F_j-\overline{\mathbf F}
\right)
\neq
K^{-1}(\mathbf F_j)
-
\text{constant}.
\]

Therefore it is not generally valid to:

1. center the measured P4 triangle;
2. treat the centered coordinates as if they were raw local optical coordinates;
3. apply the physical inverse barrel/keystone transformation directly.

The reverse transform must reconstruct or parameterize the appropriate local optical coordinates first.

This is one reason the forward model should remain authoritative and the reverse diagnostic should be derived from it.

## 8. Magnification versus barrel/radial distortion

For the accommodation-dependent radial model

\[
\mathbf B_j(A)
=
M(A)
\left[
1+
\kappa(A)M(A)^2r_j^2
\right]
\mathbf u_j,
\qquad
r_j=\|\mathbf u_j\|,
\]

uniform magnification and radial deformation become distinguishable only through differences in the point radii and the complete spatial response.

### 8.1 Equal radii are the degeneracy, not equal side lengths

If

\[
r_1=r_2=r_3=R,
\]

then

\[
\mathbf B_j(A)
=
M(A)
\left[
1+\kappa(A)M(A)^2R^2
\right]
\mathbf u_j,
\]

so only one effective scale is observed.

The empirical triangle being non-equilateral is encouraging, but the actual radii relative to the radial center must be evaluated.

### 8.2 Different radii help, but a free center can reintroduce ambiguity

If the radii differ, the responses to \(M\) and \(\kappa\) can become distinct.

However, if the radial center is also free, center shifts can mimic part of the same differential deformation.

The actual-geometry audit should therefore inspect the Jacobian columns for at least

\[
\partial_\theta\mathbf y,\quad
\partial_A\mathbf y,\quad
\partial_g\mathbf y,\quad
\partial_{c_x}\mathbf y,\quad
\partial_{c_y}\mathbf y,\quad
\partial_M\mathbf y,\quad
\partial_\kappa\mathbf y.
\]

Use rank, singular values, principal confounded directions, and sensitivity to plausible shared center offsets.

If \(M\) and \(\kappa\) are not separately identifiable, report an identifiable effective accommodation deformation rather than claiming a physical barrel coefficient.

## 9. P1 scale must remain gaze-conditioned

The nuisance scale estimate depends on the P1 reference evaluated at the trial gaze:

\[
\widehat g_{P1}(\theta)
=
\frac{
\mathbf a_1(\theta)^\mathsf T
W_1
\mathbf e_1
}{
\mathbf a_1(\theta)^\mathsf T
W_1
\mathbf a_1(\theta)
}.
\]

Therefore scale-corrected P4 coordinates cannot be precomputed once before the state solve.

Every trial or accepted gaze update must update:

1. the P1 reference pattern;
2. the P1 scale;
3. the corrected P4 relative coordinates used by the accommodation step.

This coupling is part of the estimator, not an implementation detail.

## 10. Do not use a free per-frame barrel coefficient as accommodation

An unconstrained per-frame radial coefficient can be useful during model discovery.

It should not define the final accommodation estimator.

The scientific model should use globally shared response laws such as

\[
M_i=M(A_i;\beta_M),
\]

and

\[
\kappa_i=\kappa(A_i;\beta_\kappa).
\]

Then \(A_i\) is the framewise state and \(\beta_M,\beta_\kappa\) are shared calibration parameters.

If each frame independently chooses \(M_i\) and \(\kappa_i\), and accommodation is assigned afterward, then accommodation has not been inferred from a common optical law.

## 11. Recommended metric: forward fit, reverse diagnostic

### 11.1 Primary optimization metric

Use the translation-free native observation

\[
\mathbf y_i
=
\begin{bmatrix}
\mathbf p_{i2}-\mathbf p_{i1}\\
\mathbf p_{i3}-\mathbf p_{i1}\\
\mathbf q_{i1}-\mathbf c_{1i}\\
\mathbf q_{i2}-\mathbf c_{1i}\\
\mathbf q_{i3}-\mathbf c_{1i}
\end{bmatrix}.
\]

Let the joint forward model predict

\[
\widehat{\mathbf y}_i
=
\widehat{\mathbf y}
(
\theta_i,A_i;\Psi
).
\]

Use the covariance-weighted forward objective

\[
\boxed{
J_{\rm data}
=
\sum_i
\left(
\mathbf y_i-\widehat{\mathbf y}_i
\right)^\mathsf T
R_i^{-1}
\left(
\mathbf y_i-\widehat{\mathbf y}_i
\right).
}
\]

Add only the declared soft fixation-mean anchors and necessary global/gauge constraints.

Do not add a cost rewarding equal triangle sides.

### 11.2 Why raw inverse-space RMS is unsafe

If a local inverse map changes residual coordinates according to

\[
\mathbf e'=J\mathbf e,
\]

then its covariance changes as

\[
R'=JRJ^\mathsf T.
\]

With consistent covariance propagation, the local quadratic metric can remain equivalent. Without it, a candidate transformation can appear better merely because its inverse contracts coordinates.

Therefore:

- native forward residual is the primary fitting and ranking metric;
- reverse-reference residual is a physical diagnostic;
- inverse-space ranking is allowed only after Jacobian and covariance propagation are validated.

### 11.3 Reverse-reference diagnostic

After fitting the frame state, compute

\[
\mathbf B^{\rm rec}_{ij}
=
\mathcal T_{x_{\rm ref}}
\left[
\mathcal T_{x_i}^{-1}
\left(
\widetilde{\mathbf q}_{ij}
\right)
\right].
\]

Report:

- signed x/y residuals for each P4;
- side-length ratios relative to the empirical reference;
- orientation and area relative to the empirical reference;
- residual maps versus gaze and accommodation;
- dependence on reference-frame selection;
- dependence on shared distortion-center assumptions.

Side-length diagnostics should compare to the empirical reference:

\[
\frac{L^{\rm rec}_{12}}{L^{\rm ref}_{12}},
\quad
\frac{L^{\rm rec}_{23}}{L^{\rm ref}_{23}},
\quad
\frac{L^{\rm rec}_{31}}{L^{\rm ref}_{31}}
\rightarrow 1.
\]

They should not compare the three sides to each other.

## 12. Recommended iterative estimator

With global optical parameters fixed during application:

1. initialize gaze from the calibrated relative P4-P1 centroid relationship;
2. fit P1 nuisance scale from the P1 reference evaluated at that gaze;
3. update accommodation using the shared accommodation-dependent P4 model;
4. update gaze using the accommodation-corrected centroid displacement and full relative geometry;
5. recompute P1 scale at the new gaze;
6. use damping or a final joint two-state refinement;
7. report ambiguity and conditioning rather than forcing a unique state when the local problem is weakly identified.

During calibration, alternate frame-state updates with updates of the permitted global optical/reference parameters under one coherent objective.

Do not optimize independent unrelated costs for gaze, scale, radial coefficient, and accommodation.

## 13. Cross-agreement / omitted-P4 validation

After calibration, freeze:

- the empirical reference;
- distortion-center and alignment convention;
- P1 reference and scale policy;
- accommodation-dependent magnification and radial laws;
- keystone and relative-origin laws;
- covariance policy;
- response-law parameters.

For each frame, omit P4 \(j\) in turn:

1. retain all three P1 points;
2. retain only the other two P4 points;
3. initialize without any all-three P4 centroid, area, inverse warp, or held-point-derived quantity;
4. rerun the full gaze -> P1 scale -> accommodation -> gaze solver;
5. predict the omitted P4 in original relative image coordinates.

Use

\[
\mathbf e^{\rm cross}_{ij}
=
(\mathbf q_{ij}-\mathbf c_{1i})
-
(\widehat{\mathbf q}_{ij|-j}-\mathbf c_{1i})
\]

as the primary held-point error.

Also compare the subset-derived gaze and accommodation estimates on the same frame.

Internal cross-agreement remains an internal consistency test, not independent physiological validation.

## 14. Proposed gates before implementation

### RT0 — Freeze reference and coordinate conventions

Declare:

- empirical reference construction;
- point correspondence;
- reference origin;
- reference orientation;
- reference scale;
- distortion-center parameterization;
- P1 scale policy;
- common-scale versus differential-scale policy.

**Pass:** no hidden per-frame or per-subset reference freedom remains.

### RT1 — Synthetic forward/inverse closure

Test scale, keystone, radial/magnification, relative-origin displacement, and all compositions.

**Pass:** forward -> inverse round trips recover the latent reference to numerical precision over the accepted domain, with one-to-one radial inversion.

### RT2 — Actual-geometry identifiability audit

Use the actual empirical non-equilateral triangle and plausible shared center/alignment uncertainty.

Evaluate rank and condition for:

- \(M\) versus \(\kappa\);
- gaze versus accommodation;
- center versus radial response;
- differential nuisance scale versus accommodation;
- reference gauge versus optical coefficients.

**Pass:** every parameter claimed as separately physical has adequate local distinction over meaningful calibration regions. Otherwise collapse to an identifiable effective deformation.

### RT3 — Reference sensitivity

Repeat with several valid empirical reference frames or a robust consensus reference.

**Pass:** inferred optical response and state trajectories are stable to reference choice within declared uncertainty.

### RT4 — Real-data ablation under one metric

With identical data, covariance, starts, and state policy, compare:

1. accommodation-dependent P4 magnification only;
2. magnification plus radial deformation.

**Pass:** the radial term explains reproducible shape change that cannot be absorbed by common scale, gaze, center, or reference gauge and remains identifiable.

### RT5 — Full all-condition calibration

Run all reviewed calibration conditions with independent framewise gaze/accommodation and soft fixation-mean anchors.

**Pass:** convergence, ambiguity, rank, and failures are explicitly recorded; no regular-triangle constraint or per-frame deformation map is introduced.

### RT6 — Three-way cross-agreement

Freeze calibration and run the full omitted-P4 solver.

**Pass:** retained pairs infer compatible states and predict the omitted point without held-point leakage. No universal RMS threshold is imposed.

## 15. Required documentation reconciliation

### docs/Theory.md

Add a dedicated reference-recovery section that:

- defines the empirical non-equilateral reference;
- states \(\mathcal T_{\rm ref}\circ\mathcal T_x^{-1}\);
- gives inverse order explicitly;
- explains why a physical inverse needs a declared local optical origin;
- warns that centering does not commute with nonlinear optical inversion;
- keeps the forward native-coordinate objective primary;
- identifies reverse recovery as a diagnostic / validated secondary metric;
- requires actual-geometry \(M/\kappa/\)center identifiability tests.

### docs/ESTIMATOR_PLAN.md

Reconcile the old area-normalization requirement with current Theory.

Move the historical conditional area-normalized estimator into an explicitly historical/control section.

Define the new implementation contract around:

- P1 reference-scale fitting;
- the 10-coordinate translation-free observation;
- the joint forward optical model;
- paired forward/inverse operators;
- reverse-reference diagnostics;
- complete held-P4 iterative cross-agreement.

### docs/ACCOMMODATION_FULL_CALIBRATION_PLAN.md

Replace the old area-normalized six-coordinate model as the invariant of the new workstream.

Preserve its important scientific requirements:

- no framewise nominal-demand truth;
- no fixation-flatness assumption;
- one shared optical response law;
- identifiability/gauge analysis;
- three-way P4 cross-agreement;
- external evidence required for physiological diopter accuracy.

Add explicit tests of:

- empirical-reference choice;
- distortion-center gauge;
- \(M\)-versus-\(\kappa\) rank;
- reference-recovery stability.

### docs/CURRENT_STATUS.md

Do not rewrite historical empirical results as results of the reverse-transform model.

Add the new method only after actual implementation/calibration evidence exists.

## 16. Failure modes to watch

1. **Reference leakage:** using the omitted P4 to choose or refine the reference during cross-check.
2. **Centroid-as-center error:** treating the measured P4 centroid as the radial center.
3. **Wrong inverse order:** undoing radial before keystone when the forward composition is radial then keystone.
4. **Double distortion:** applying a baseline radial correction to an empirical reference that already contains it.
5. **Free per-frame barrel:** fitting \(\kappa_i\) independently and relabeling it accommodation.
6. **Scale removal of signal:** fitting an unconstrained P4 scale that removes accommodation-dependent magnification.
7. **Regular-shape bias:** forcing the non-equilateral empirical triangle toward an equilateral target.
8. **Gauge drift:** simultaneously freeing reference geometry, center, reference radial coefficient, accommodation magnification, and per-frame scale.
9. **Inverse-space metric bias:** ranking models by unweighted reverse RMS when inverse mappings change coordinate scaling.
10. **Shape-only information loss:** centering P4 and discarding the P4-P1 centroid displacement.
11. **Centered-inverse error:** applying a physical inverse directly to centered nonlinear/projective coordinates.
12. **Plan mismatch:** implementing a hybrid of current Theory and old area-normalized plans without a declared contract.
13. **Overclaiming:** treating internal cross-agreement as independent physiological accommodation accuracy.

## 17. Recommended implementation architecture

Do not start with a separate reverse-only estimator.

Implement one authoritative forward optical model with paired tested inverse operators:

- P1 forward transformation;
- P1 gaze-conditioned scale fit;
- P4 accommodation radial/magnification forward operator;
- P4 accommodation radial/magnification inverse operator;
- keystone forward operator;
- keystone inverse operator;
- relative-origin forward model;
- reference-recovery adapter.

The inverse operators should be verified against the forward operators, not independently tuned to data.

One forward likelihood/objective should drive calibration and state inference. Reference recovery should consume the same fitted model.

## 18. Recommended decision

Proceed with the reverse-transform concept, but define it as **empirical reference recovery under one shared optical model**.

The scientific question should not be:

> Which barrel coefficient makes the triangle look most regular?

It should be:

> Which shared gaze/accommodation optical state makes the measured P1/P4 geometry consistent with the same fixed empirical reference while preserving relative P4-P1 displacement and predicting withheld P4 measurements?

The immediate priority is:

1. reconcile the plan documents with current Theory;
2. prove forward/inverse round-trip correctness;
3. establish a valid local optical-origin/distortion-center convention;
4. test magnification versus radial identifiability on the actual empirical triangle;
5. test reference-frame sensitivity;
6. only then run full calibration and three-way held-P4 cross-agreement.

If the three-point geometry cannot separately identify radial coefficient and magnification, that does not invalidate reference recovery. The correct scientific result is to use the identifiable effective deformation as the accommodation signal and avoid claiming a physically unique barrel coefficient.
