# Reverse-transform / empirical-reference audit

> **Current implementation status (2026-10-10):** Reverse-transform Stages 01–05 were implemented and numerically audited under a **size-normalized keystone convention**. Subsequent review found that the keystone operator was rescaled to preserve RMS triangle size, which removes a legitimate gaze-dependent size component from the projective transform. Therefore Stages 01–05 are now historical controls and must be rebuilt under the raw keystone transform before further accommodation interpretation. Do not reuse their fitted keystone coefficients, radial-law slope, framewise A values, or joint gaze/A states as calibration inputs to the corrected pipeline. See [REVERSE_TRANSFORM_NEXT_STEP.md](REVERSE_TRANSFORM_NEXT_STEP.md).

**Repository / branch:** `rueijrwu/gaze_acc_joint_estimator` / `exp5_distortion_model`  
**Original audit basis:** `37d2181ee0fbd43c1eaa6f38b26b0ab5df32475c`; updated after Stages 01–05 and the keystone-normalization finding.  
**Optical reference repository:** `rueijrwu/distortion_tracking` / `main`, inspected at `7b3f28e7a7817aa701c84d1ecba13dbdd229d96d`  
**Scope:** This document preserves the original design audit and its scientific cautions. Current implementation status and stage links are summarized in the banner above; the implemented stages do not establish physical optical calibration or physiological validation.

## 1. Executive conclusion

The proposed idea is scientifically useful if it is stated as:

> **Estimate gaze, P1 nuisance scale, and accommodation such that removal of the state-dependent optical effects maps every frame back to one fixed empirical P1/P4 reference pattern.**

The important target is the **same empirical non-equilateral triangle**, not an equilateral triangle, regular grid, or arbitrary aesthetically regularized pattern.

This formulation has four advantages:

1. it uses only relative changes from a fixed reference;
2. it makes the accommodation hypothesis visually and mathematically testable as reference recovery;
3. it preserves the actual source/alignment geometry rather than declaring fixed imperfections to be physiological signal;
4. it provides a strong internal validation question: after estimating a state from retained measurements, does the omitted P4 return to its reference-consistent location?

However, the reverse transformation should **not initially replace the forward relative-coordinate cost**. The safest design is:

- use the physical forward model in the original relative camera-coordinate metric for calibration and state estimation;
- use the reverse-to-reference transformation as a primary diagnostic/visualization and, only after covariance/Jacobian validation, as an equivalent weighted metric.

The main unresolved scientific issue is **identifiability**. A non-equilateral measured triangle is helpful, but unequal side lengths alone do not prove that accommodation-dependent magnification and barrel/radial distortion can be separated. What matters is the set of radii relative to a constrained distortion center and the rank of the complete local sensitivity after nuisance scale, gaze, center, and reference gauges are included.


## 1A. Pipeline-wide correction after implementation audit

The implementation review changed one important conclusion of this audit: the extra RMS-size normalization inside the keystone operator must be removed **everywhere**, not only in a future stage.

The corrected rule is:

\[
\boxed{
\text{keystone/projective transform} = K(\theta,A;B)
\quad\text{with no post-transform RMS or area normalization.}
}
\]

Centering the transformed triangle for a translation-free observation is still allowed:

\[
H_r(\theta,A)=C\!\left(K_r(\theta,A;B_r)\right),
\]

but do not rescale \(H_r\) to the input/reference RMS radius.

One common positive nuisance scale remains and is estimated from P1:

\[
g_i(\theta)=
\frac{\langle X_{1i},H_1(\theta)\rangle}
{\langle H_1(\theta),H_1(\theta)\rangle}.
\]

The same \(g_i\) is then applied to P4. This preserves the intended separation between common external magnification, gaze-dependent projective size/shape, and accommodation-dependent P4 deformation.

### Consequence for existing stages

All empirical reverse-transform stages that used the normalized keystone must be corrected as a chain:

1. **Stage 01 / P1:** refit P1 keystone from scratch using the raw transform; reprofile framewise P1 scale under that transform.
2. **Stage 02 / Capture-1 P4:** refit P4 keystone from scratch using the new P1-derived scale; keep the reference radial increment at its gauge value for the reference capture.
3. **Stage 03 / independent captures:** refit P1/P4 raw-keystone coefficients and relative radial deformation from scratch. The old per-capture radial coefficients are not transferable because they may have absorbed size removed by the normalization.
4. **Stage 04 / framewise A:** rerun only after the new Stage-03 optical model and radial/accommodation law are frozen. Old framewise A values remain historical controls.
5. **Stage 05 / joint gaze+A:** rerun only if needed after the corrected Stage-04 result. Old joint states remain diagnostic evidence under the obsolete normalized-keystone parameterization.

Do not patch old coefficients by simply deleting the normalization factor. Removing the normalization changes the model and therefore the meaning of all fitted coefficients and profiled states.

### Accommodation dependence of P4 keystone

When the corrected raw-keystone Stage 03 is rebuilt, inspect whether P4 keystone coefficients vary systematically with accommodation/capture. If supported, promote the minimum necessary dependence into a shared \(K_4(\theta,A)\), e.g. selected terms such as

\[
s_{x,4}(\theta,A)=1+[\alpha_{40}+\alpha_{4A}(A-A_{ref})]\theta^2,
\]

or

\[
q_4(\theta,A)=[\gamma_{40}+\gamma_{4A}(A-A_{ref})]\theta.
\]

Do not force all coefficients to depend on A. Add only dependencies that reduce reproducible coordinate residuals and remain identifiable. Otherwise the radial/accommodation term can again absorb accommodation-dependent keystone error.

### Preservation rule

Preserve all existing Stage 01–05 outputs and source snapshots unchanged. The corrected raw-keystone pipeline must write to new stage/result directories so the normalized-keystone chain remains reproducible as a historical comparison.

## 2. Current repository state

The latest `docs/Theory.md` already contains several pieces required by this proposal:

- P1-reference scale is the primary normalization; triangle area is no longer required.
- P4 has accommodation-dependent magnification `M(A)`, radial coefficient `kappa4(A)`, gaze-dependent keystone deformation, and a separate axial nuisance scale.
- The empirical reference is explicitly allowed to be already distorted.
- The theory warns that inverse keystone about an observed centroid is not generally equivalent to inversion about the true local optical origin.
- It already recognizes equal-radius magnification/radial degeneracy.
- It uses a forward relative-data cost in original coordinates and a three-way held-P4 cross-agreement.

Those are all compatible with the reverse-reference interpretation.

The documentation is not yet fully reconciled. In particular:

- `docs/ESTIMATOR_PLAN.md` still describes the historical area-normalized conditional six-residual model as the required normalization and treats a changed P1 normalizer as a later experiment.
- `docs/ACCOMMODATION_FULL_CALIBRATION_PLAN.md` still defines `r=(P1-c1)/sqrt(area(P1))`, `v=(P4-c1)/sqrt(area(P1))`, and the older `D(theta,A)+T(theta,A)r` model.

Therefore implementation should not be started by silently mixing the latest theory with those older contracts. The reverse-transform work should first pass the gates in Section 11, then the authoritative plan documents should be reconciled explicitly.

## 3. Recommended reference: preserve the empirical triangle

The empirical P4 pattern is a triangle but **not an equilateral triangle**. That asymmetry should be preserved.

Let the fixed empirical reference be

[
mathcal B^{ref}={mathbf B^{ref}_1,mathbf B^{ref}_2,mathbf B^{ref}_3}.
]

The desired result after correcting a frame is

[
oxed{mathbf B^{rec}_{ij}approxmathbf B^{ref}_j,}
]

not

[
|mathbf B^{rec}_1-mathbf B^{rec}_2|
=
|mathbf B^{rec}_2-mathbf B^{rec}_3|
=
|mathbf B^{rec}_3-mathbf B^{rec}_1|.
]

An unequal reference is not a defect. It contains fixed source placement, camera alignment, baseline optical distortion, and any other reproducible reference geometry. Those fixed effects should not be reassigned to framewise accommodation.

### 3.1 Do not force a regular grid/equilateral target

A fixed map that transforms the empirical reference into a convenient regular plotting coordinate system is allowed as a **coordinate convention**. If the complete forward/inverse model and covariance are transformed consistently, this creates no new information.

It must not be interpreted as evidence that the physical reference was an equilateral triangle or perfect grid.

A "random barrel" selected only to make the reference look regular is therefore not a physical calibration. It can be:

- an initialization;
- a fixed gauge/convention;
- or a visualization transform.

It should not be a fitted per-frame degree of freedom and should not be reported as the physical reference barrel coefficient.

### 3.2 Prefer a consensus empirical reference over one noisy frame

One frame can initialize the pattern, but the final reference should preferably be estimated from multiple suitable near-reference frames or as one shared template during calibration.

A practical initialization is:

1. choose low-demand, near-zero-gaze frames;
2. preserve P1/P4 point identities;
3. estimate trial gaze and P1 scale;
4. map those frames toward a common reference state;
5. form a robust shared reference;
6. hold that reference convention fixed while evaluating frame states.

This does **not** impose constant gaze or accommodation inside a fixation. It only estimates one shared reference geometry from multiple observations.

## 4. What the reverse transform should mean

Write the local P4 forward model schematically as

[
mathcal F_{	heta,A}
=
K_4(	heta,A)circ mathcal R_A,
]

where (mathcal R_A) contains the accommodation-dependent baseline magnification/radial response, and (K_4) is the gaze/keystone transformation.

The complete measured relative coordinate also contains:

- the P1-derived external scale (g);
- possible P4/P1 differential scale (eta);
- and the relative P4/P1 optical-origin displacement.

The present theory writes

[
mathbf q_j-mathbf c_1
=
gleft[
oldsymboldelta_g
+etamathbf F_{4j}(	heta,A)
-overline{mathbf F}_1(	heta)

ight].
]

Therefore, before applying a physical inverse (K_4^{-1}) or (mathcal R_A^{-1}), reconstruct the local P4 coordinate:

[
oxed{
mathbf z_j
=
rac{
(mathbf q_j-mathbf c_1)/g
+overline{mathbf F}_1(	heta)
-oldsymboldelta_g
}{eta}
=
mathbf F_{4j}(	heta,A).
}
]

Then the source/reference coordinate is recovered in reverse order:

[
oxed{
widehat{mathbf u}_j
=
mathcal R_A^{-1}
left(
K_4^{-1}(	heta,A;mathbf z_j)

ight).
}
]

To compare with the empirical reference state (x_{ref}=(	heta_{ref},A_{ref})), reapply the reference-state transformation:

[
oxed{
mathbf B^{rec}_{ij}
=
K_4left(
	heta_{ref},A_{ref};
mathcal R_{A_{ref}}(widehat{mathbf u}_j)

ight).
}
]

At a convention where (	heta_{ref}=0) and (K_4(0,A)=I), this simplifies, but the general form should be retained in the specification.

### 4.1 Inverse order is mandatory

If the forward order is

[
	ext{reference/source}

ightarrow
mathcal R_A

ightarrow
K_4

ightarrow
	ext{external scale/relative placement},
]

the inverse order is

[
	ext{remove external scale/placement}

ightarrow
K_4^{-1}

ightarrow
mathcal R_A^{-1}.
]

Do not change the sign of the radial coefficient and call that the radial inverse. The inverse of

[
mathbf r'=(1+kappa|mathbf r|^2)mathbf r
]

is generally a nonlinear scalar-radius solve.

### 4.2 A critical requirement: reverse transformation needs an optical origin

The current theory permits either:

- fitting the relative centroid law (mathbf h_g) directly; or
- fitting the relative optical-origin displacement (oldsymboldelta_g) and deriving (mathbf h_g).

For ordinary forward prediction, direct (mathbf h_g) can be sufficient.

For a **physical reverse radial transformation of individual P4 points**, (mathbf h_g) alone is generally insufficient because radial inversion requires coordinates about a defined local radial center/origin. The reverse-transform implementation therefore needs one of these:

1. a constrained/shared (oldsymboldelta_g) plus distortion-center convention;
2. an equivalent shared local-coordinate parameterization that reconstructs (mathbf F_{4j});
3. or a deliberately empirical relative-deformation inverse that makes no claim to recover absolute radial coordinates.

This is the most important structural issue to resolve before coding the reverse transform.

## 5. Do not invert a centered triangle as if centering commuted with optics

A tempting shape-only construction is

[
mathbf q_j-mathbf c_4=getamathbf S_{4j}.
]

This removes relative P4/P1 displacement and is useful for inspecting shape.

But (mathbf S_{4j}=mathbf F_{4j}-overline{mathbf F}_4) is a **centered output of a nonlinear/projective transformation**. In general,

[
K^{-1}(mathbf F_j-overline{mathbf F})

eq
K^{-1}(mathbf F_j)-	ext{constant},
]

and similarly for a radial transform. Centering and projective/radial inversion do not commute.

Therefore:

- centered P4 edges/triangles are valid relative measurements;
- they can be forward-predicted and compared;
- but they should not be pointwise "undistorted" by simply applying the physical inverse about the measured centroid.

For a true reverse optical transform, reconstruct the local coordinate about the calibrated origin first.

## 6. Magnification versus barrel/radial distortion

The empirical triangle being non-equilateral is potentially helpful, but it does not by itself guarantee identifiability.

For

[
mathbf B_j(A)
=
M(A)
left[
1+kappa_4(A)M(A)^2r_j^2

ight]
mathbf u_j,
qquad
r_j=|mathbf u_j|,
]

uniform magnification changes all points through the common factor (M), while radial distortion changes them according to (r_j^2).

### 6.1 Equal radii are the degeneracy, not equal side lengths

If all three radii relative to the radial center are equal,

[
r_1=r_2=r_3=R,
]

then

[
mathbf B_j
=
M(1+kappa M^2R^2)mathbf u_j,
]

and only one effective scale is observable from the three centered points.

A triangle can be non-equilateral and still have all three vertices on a circle about the chosen radial center. Therefore **unequal side lengths are not the test**.

The required test is the actual local Jacobian/rank at the empirical geometry and plausible shared center.

### 6.2 Different radii help, but a free center can reintroduce ambiguity

If the three radii differ, the responses to (M) and (kappa) can become distinguishable. However, if the radial center is also free, center shifts can mimic part of the same differential deformation.

Before claiming separate accommodation magnification and barrel coefficients, evaluate the rank and conditioning of a local sensitivity matrix containing at least the relevant columns from

[
left[
partial_	hetamathbf y,
partial_Amathbf y,
partial_gmathbf y,
partial_{c_x}mathbf y,
partial_{c_y}mathbf y,
partial_Mmathbf y,
partial_kappamathbf y

ight],
]

with only the actually free quantities included.

If (M) and (kappa) remain poorly separated, use an **effective accommodation-dependent deformation/scale** rather than assigning physical meaning to an unstable absolute barrel coefficient.

## 7. Reference gauge and relative radial model

The strongest scientific version of the proposal is reference-relative.

The empirical reference is already distorted. Therefore it is safer to model the transformation **from the reference state to another state** than to pretend the empirical reference is a known paraxial grid.

Conceptually,

[
oxed{
mathcal R_{rel}(A)
=
mathcal R_Acircmathcal R_{A_{ref}}^{-1}.
}
]

Then

[
mathcal R_{rel}(A_{ref})=I.
]

If the absolute paraxial geometry and (kappa(A_{ref})) are not identifiable, parameterize the **relative deformation** with the reference constraints

[
M_{rel}(A_{ref})=1,
qquad
Deltakappa(A_{ref})=0,
]

while stating clearly that (Deltakappa) is a reference-relative coefficient, not an independently measured absolute barrel coefficient.

This directly matches the scientific question: how does the P4 pattern change with accommodation relative to the selected reference?

## 8. Recommended metric: forward fit, reverse diagnostic

### 8.1 Primary optimization metric

Use the existing translation-free observation vector

[
mathbf y_i=
egin{bmatrix}
mathbf e_{1i}\
mathbf q_{i1}-mathbf c_{1i}\
mathbf q_{i2}-mathbf c_{1i}\
mathbf q_{i3}-mathbf c_{1i}
end{bmatrix}
]

and minimize the fixed raw-relative coordinate metric

[
oxed{
Q_i=
(mathbf y_i-widehat{mathbf y}_i)^T
R_{y,i}^{-1}
(mathbf y_i-widehat{mathbf y}_i).
}
]

This metric has three important properties:

1. it does not reward a candidate merely for contracting coordinates through its inverse;
2. its units and covariance are tied to the measurements;
3. held-P4 prediction can be scored in the original camera-coordinate convention.

### 8.2 Reverse-to-reference residual

Also compute

[
oxed{
mathbf e^{rev}_{ij}
=
mathbf B^{rec}_{ij}
-
mathbf B^{ref}_j.
}
]

This should be a first-class diagnostic: plot it by gaze, accommodation, point, capture, and recovered coordinate.

If it is later used as a formal comparative cost, propagate the measurement covariance through the reverse transform:

[
R^{rev}
approx
J_{rev}R_{local}J_{rev}^T
+
R_{template},
]

where (J_{rev}) is the reverse-transform Jacobian and (R_{template}) represents reference uncertainty when relevant.

An unweighted inverse-space RMS should not rank models because candidate transforms can change the scale of the residual space.

### 8.3 Desired interpretation

A successful model should satisfy both views:

[
	ext{forward state model predicts the raw relative coordinates}
]

and

[
	ext{inverse state correction collapses observations back onto one shared empirical reference.}
]

These are two representations of the same hypothesis, not separate degrees of freedom.

## 9. Framewise estimation and shared calibration

With global parameters fixed, the recommended framewise sequence is:

1. initialize gaze from the relative P4-P1 displacement;
2. evaluate the gaze-conditioned P1 reference;
3. estimate the one positive P1 nuisance scale (g_{P1}(	heta));
4. remove that common nuisance scale;
5. estimate accommodation using the complete P4 reference-relative forward model;
6. update gaze using the accommodation-corrected displacement and P4 geometry;
7. repeat with (g_{P1}) recomputed at every trial gaze;
8. optionally perform a final joint ((	heta,A)) refinement.

The key point is that "find the barrel coefficient that best matches the reference" must not mean a free framewise (kappa_i).

The final estimator should use shared functions, e.g.

[
M_i=M(A_i;eta_M),
qquad
kappa_i=kappa(A_i;eta_kappa),
]

or a shared reference-relative effective deformation law.

A free per-frame best-fit radial coefficient can be useful as a **diagnostic experiment** to see whether a repeatable response curve exists. It is not yet an accommodation estimator.

After frame-state updates, update the allowed shared reference, displacement, rotation, magnification/radial response, and alignment parameters using all calibration frames, then repeat under the same objective.

## 10. Cross-agreement / omitted-P4 validation

The strongest internal test remains the three-way held-P4 procedure.

After calibration, freeze:

- empirical reference geometry;
- distortion center/origin convention;
- all shared response functions;
- noise/weighting policy;
- reference gauge;
- and any reference refinement.

For each frame and omitted P4 (j):

1. retain all three P1 points;
2. retain only the other two P4 points;
3. rerun the complete gaze -> P1 scale -> accommodation -> corrected gaze inference;
4. do not use the measured three-P4 centroid, P4 area, affine fit, reverse transform, or initialization containing the omitted point;
5. predict the omitted P4 in the original relative camera-coordinate system;
6. score its original-coordinate error;
7. separately report subset agreement in (	heta) and (A).

The reverse diagnostic can then ask whether the omitted prediction and retained measurements all map to the same empirical reference state.

This test is internal consistency because calibration used the full dataset. It is not independent physiological validation.

## 11. Proposed gates before implementation

The reverse-transform idea should be developed in short gates so a failure does not trigger a large rewrite.

### RT0 — Freeze reference and coordinate conventions

Deliverables:

- exact empirical P1/P4 reference construction;
- source correspondence;
- reference origin/rotation/scale convention;
- distortion-center convention;
- (A_{ref}), (	heta_{ref}), and (M(A_{ref})=1);
- statement of whether radial coefficients are absolute or reference-relative.

**Gate:** the same input data always produce the same reference and gauges. No per-frame regularization target is allowed.

### RT1 — Synthetic forward/inverse closure

Generate synthetic states from the proposed forward model and verify:

[
mathcal T_{ref}circmathcal T_x^{-1}circmathcal T_x
]

returns the reference to numerical precision.

Test:

- transform order;
- radial inverse;
- keystone inverse;
- P1 scale cancellation;
- nonzero relative-origin displacement;
- differential (eta);
- non-equilateral reference.

**Gate:** exact noiseless closure and analytic/numeric Jacobian agreement.

### RT2 — Actual-geometry identifiability audit

Using the actual empirical reference triangle, evaluate local rank/conditioning over:

- gaze range;
- accommodation range;
- plausible shared center offsets;
- P1 scale variation.

Compare at minimum:

1. accommodation magnification only;
2. magnification + radial deformation;
3. magnification + radial + optional accommodation-dependent keystone term.

**Gate:** do not promote a separate radial coefficient unless it contributes a distinguishable response after gaze/scale/center nuisances. Otherwise use effective deformation.

### RT3 — Reference sensitivity

Repeat the initialization with:

- several individual candidate reference frames;
- one multi-frame consensus reference;
- modest allowed shared reference refinement.

Compare state trajectories and predicted original-coordinate P4 values.

**Gate:** scientific conclusions must not depend strongly on which acceptable reference frame happened to be chosen. If they do, reference uncertainty must enter the model.

### RT4 — Real-data ablation under one metric

On one identical calibration population and noise policy, compare:

- scale + gaze only;
- accommodation magnification;
- accommodation magnification + radial/effective deformation;
- then only if needed, explicit accommodation/keystone coupling.

Do not change response family, center policy, reference construction, and covariance simultaneously.

**Gate:** each added mechanism must produce reproducible image-space improvement and acceptable identifiability, not merely lower training cost.

### RT5 — Full all-condition calibration

Run the accepted model on all 20 reviewed calibration conditions with free framewise gaze/accommodation and soft fixation-mean anchors.

Store:

- global parameters;
- per-frame states;
- convergence/branch/bound status;
- forward residuals;
- reverse-reference residuals;
- conditioning/identifiability diagnostics.

**Gate:** numerical certification and complete scheduled accounting.

### RT6 — Three-way cross-agreement

Freeze the calibration and perform the complete omitted-P4 procedure.

Report:

- original-coordinate P4 cross error;
- signed axes and point-specific residuals;
- (	heta) subset disagreement;
- (A) subset disagreement;
- ambiguity/conditioning;
- reverse-reference collapse plots;
- exact common-frame coverage.

**Gate:** classify the result as supported, tradeoff, identifiability unresolved, or insufficient evidence. Do not define an arbitrary pixel/degree/diopter combined score.

## 12. Required documentation reconciliation

If RT0-RT3 support this design, update the authoritative documents before implementation:

### `docs/Theory.md`

Add a dedicated **reference-recovery interpretation**:

[
mathbf B^{rec}
=
mathcal T_{ref}circmathcal T_x^{-1}(mathbf y)
]

with:

- explicit local-origin reconstruction;
- inverse order;
- reference-relative radial option;
- forward-vs-reverse metric distinction;
- empirical non-equilateral reference;
- statement that a regular-grid transform is a gauge/visualization unless independently known.

The current warning against inverting about an observed centroid should remain.

### `docs/ESTIMATOR_PLAN.md`

This file is currently historical/conditional and conflicts with the latest theory at the top-level normalization contract.

Either:

1. mark the old area-normalized conditional design as a historical baseline section and add a new authoritative implementation plan for reference-P1 scale + joint optical model; or
2. create a clean new implementation plan and explicitly state which document controls the new work.

Do not silently edit the old coefficients/normalization and claim the same estimator.

### `docs/ACCOMMODATION_FULL_CALIBRATION_PLAN.md`

The plan still assumes the old `r/v` area normalization and `D+Tr` six-residual model. Preserve its historical results, but revise the future work program to use:

- the reference-P1 scale;
- the new relative observation vector;
- the shared empirical reference;
- physical/reference-relative accommodation mechanisms;
- the same identifiability and held-P4 evidence hierarchy.

### `docs/CURRENT_STATUS.md`

Do not rewrite historical empirical results as results of the reverse-transform model. Add the new method only after an actual implementation/calibration exists.

## 13. Failure modes to watch

1. **Reference leakage:** choosing/refining the reference using the held P4 during cross-check.
2. **Centroid-as-center error:** treating the measured P4 centroid as the radial center.
3. **Wrong inverse order:** undoing radial before keystone when the forward composition is radial then keystone.
4. **Double distortion:** applying a baseline radial correction to an empirical reference that already contains it.
5. **Free per-frame barrel:** fitting (kappa_i) independently and then relabeling it accommodation.
6. **Scale removal of signal:** fitting a separate unconstrained P4 scale that removes accommodation-dependent magnification.
7. **Regular-shape bias:** forcing the empirical non-equilateral triangle toward an equilateral target.
8. **Gauge drift:** simultaneously freeing reference geometry, center, reference radial coefficient, (M(A)), and per-frame scale.
9. **Inverse-space metric bias:** ranking by unweighted reverse RMS when the candidate inverse changes coordinate scaling.
10. **Shape-only information loss:** centering P4 and discarding P4-P1 centroid displacement.
11. **Plan mismatch:** implementing a hybrid of the new theory and the old area-normalized conditional plan without a declared contract.
12. **Overclaiming:** interpreting internal cross-agreement as independent physiological accommodation accuracy.

## 14. Recommended decision

Proceed with the reverse-transform concept, but define it as **empirical reference recovery under one shared optical model**.

The recommended scientific statement is:

[
oxed{
	ext{A valid gaze/accommodation state is one whose calibrated optical inverse returns the measured P1/P4 geometry to the same fixed empirical reference.}
}
]

For implementation and fitting:

[
oxed{
	ext{Optimize in original relative coordinates; use reverse-to-reference consistency as a physically interpretable diagnostic and validated secondary metric.}
}
]

Do not force the empirical triangle to become equilateral. Do not require an absolute barrel coefficient when only relative deformation is identifiable. Resolve the local optical-origin/distortion-center parameterization first, because a physical reverse radial transform cannot be defined uniquely from the relative centroid law alone.

The first practical work should therefore be **RT0 -> RT1 -> RT2**, not a full estimator rewrite.
