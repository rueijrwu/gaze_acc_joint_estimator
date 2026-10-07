# Three-pair theory: shared gaze/accommodation and predictive geometric consistency

## Purpose and status

**One physical state must explain all three P1/P4 pairs simultaneously.** The reason to retain three pairs is not merely to average more points. It is to obtain several geometric responses to the same gaze/accommodation state, so that observations not used to estimate that state can test its predictions.

The three pairs need not move equally. Their response functions can differ, but they share one horizontal gaze angle `theta` and one accommodation value `A` in each frame. The central test is:

> Estimate gaze and accommodation from part of the measured geometry, then ask whether that same state predicts the remaining P4 point.

This document defines the measurement theory, shared-state model, and meaning of that test. [ESTIMATOR_PLAN.md](ESTIMATOR_PLAN.md) proposes a concrete calibration/inversion algorithm. The full-position estimator is **proposed, not implemented or validated**. The frozen 13-coefficient two-channel estimator remains the baseline in [HANDOFF.md](HANDOFF.md) and [models/quadratic_model.json](models/quadratic_model.json).

The required observables remain exactly

$$
\boxed{d_x=\frac{\operatorname{mean}(P4_x)-\operatorname{mean}(P1_x)}
 {\sqrt{\operatorname{area}(P1)}}},\qquad
\boxed{\rho_4=\frac{\operatorname{area}(P4)}{\operatorname{area}(P1)}}.
$$

The square root belongs in the displacement denominator, not in `rho_4`. Area is the triangle formed by the three reflection centers, not the summed areas of the detected blobs. The extension preserves these definitions while retaining the geometry they compress.

## 1. Correspondence, coordinates, and physical state

Let

$$
\mathbf p_j=\mathbf P_{1,j},\qquad
\mathbf q_j=\mathbf P_{4,\pi(j)},\qquad j=1,2,3,
$$

be corresponding two-dimensional image points in pixels, with x right and y down. The permutation `pi` converts stored P4 slots to P1 correspondence. The handoff records zero-based `pair_index=[2,1,0]`. Read each payload's metadata and reorder coordinates and flags consistently. Sorting points by x does not itself establish persistent illuminator identity.

The state is

$$
x=(\theta,A),
$$

where horizontal gaze is in degrees and accommodation is in diopters. Using vertical image coordinates does not automatically identify vertical gaze. A later `(theta_x,theta_y,A)` model would need appropriate calibration and a new rank analysis.

Use `eta` for explicitly declared geometry/pose/imaging nuisance variables. They are not arbitrary per-frame freedoms that may be added merely to make a residual disappear.

Let

$$
\mathbf c_1=\tfrac13\sum_j\mathbf p_j,\qquad
\mathbf c_4=\tfrac13\sum_j\mathbf q_j,
$$

$$
E_1=[\mathbf p_2-\mathbf p_1\;\;\mathbf p_3-\mathbf p_1],\qquad
E_4=[\mathbf q_2-\mathbf q_1\;\;\mathbf q_3-\mathbf q_1].
$$

Signed and unsigned triangle areas and the P1 length normalizer are

$$
a_i=\tfrac12\det E_i,\qquad \mathcal A_i=|a_i|,\qquad
\ell_1=\sqrt{\mathcal A_1}.
$$

The notation `mathcal A_i` denotes image area and must not be confused with accommodation `A`. Retain signed areas and correspondence parity as diagnostics; the absolute value discards orientation sign and is nondifferentiable at a zero-area crossing.

Finite coordinates and reliable correspondence are prerequisites. A P1 triangle must be sufficiently nondegenerate relative to localization uncertainty, not merely nonzero in floating-point arithmetic. Thin triangles can also make edge-map inversion poorly conditioned. Any stronger validity gate than the existing implementation must be separately specified and compared on matched support.

## 2. Complete relative measurements and the baseline reduction

Define the P1-centered, area-normalized reference geometry and P4 positions:

$$
\mathbf r_j=\frac{\mathbf p_j-\mathbf c_1}{\ell_1},\qquad
\mathbf v_j=\frac{\mathbf q_j-\mathbf c_1}{\ell_1}.
$$

The matched displacement is

$$
\mathbf u_j=\frac{\mathbf q_j-\mathbf p_j}{\ell_1}
=\mathbf v_j-\mathbf r_j.
$$

These are dimensionless. The P1 triangle satisfies

$$
\sum_j\mathbf r_j=0,\qquad
\operatorname{area}(\mathbf r_1,\mathbf r_2,\mathbf r_3)=1.
$$

The common displacement is

$$
\boxed{\mathbf d=\frac{\mathbf c_4-\mathbf c_1}{\ell_1}
=\tfrac13\sum_j\mathbf u_j=\tfrac13\sum_j\mathbf v_j}.
$$

Write the differential response as

$$
\mathbf w_j=\mathbf u_j-\mathbf d,\qquad \sum_j\mathbf w_j=0.
$$

Three normalized displacement vectors contain six scalar components: two common-motion components and four differential-motion components. A nonredundant displacement basis is `(d,w1,w2)`, with `w3=-w1-w2`.

Since P1 normalized area is one,

$$
\boxed{\rho_4=\operatorname{area}(\mathbf v_1,\mathbf v_2,\mathbf v_3)
=\operatorname{area}(\mathbf r_1+\mathbf w_1,
                     \mathbf r_2+\mathbf w_2,
                     \mathbf r_3+\mathbf w_3)}.
$$

A common P4 translation changes `d` but not triangle area. Differential motion can alter area, orientation, or shape. The area ratio preserves only one scalar aspect of that differential motion.

The baseline retains only

$$
y_0=[d_x,\rho_4]^\top.
$$

Adding `d_y` gives a third channel but still not the full geometry. Calling `d_x` a gaze channel and `rho_4` an accommodation-sensitive channel is descriptive, not a claim of separability: both may depend on both states and nuisance geometry.

**Three pairs reduced to two scalar summaries still provide only two channels for two states.** The intended predictive cross-check requires retaining individual positions/displacements or an equivalent complete representation. Do not add `rho_4` as an independently weighted observation when it is already determined by all retained P4 positions and P1 geometry.

## 3. What normalization removes—and what it does not

For a common positive isotropic scale and additive translation,

$$
\mathbf p'_j=\lambda\mathbf p_j+\mathbf b,\qquad
\mathbf q'_j=\lambda\mathbf q_j+\mathbf b,\qquad \lambda>0,
$$

area scales by `lambda^2` and length by `lambda`. Consequently `r,v,u,d,rho_4` are unchanged. This is the reason for using square-root P1 area for displacement and P1 area for an area ratio.

For a shared proper image rotation `R`, the vectors rotate while the area ratio remains unchanged. In particular, horizontal displacement remains tied to the camera x-axis. Optional P1-derived rotation normalization changes the retained information and must be evaluated rather than assumed harmless.

For an identical nonsingular affine transformation `B` of all six image points,

$$
\rho'_4=\rho_4,\qquad
\mathbf d'=B\mathbf d/\sqrt{|\det B|}.
$$

Thus the area ratio has common-affine invariance, whereas the displacement vector generally does not. These statements concern transformations applied identically to all measured points. They do not establish immunity to physical eye motion, differential P1/P4 magnification, perspective, changes in optical surfaces, or detector bias.

P1 area need not be constant or independent of gaze/accommodation merely because it is a normalizer. Its measured variations and uncertainty remain important. Common-scale normalization also does not supply an independently measured physical eye-camera distance.

## 4. Exact geometry versus a predictive optical model

For a nondegenerate P1 triangle, define the observed triangle map

$$
T_{obs}=E_4E_1^{-1}.
$$

At the three measured vertices,

$$
\mathbf q_j=\mathbf c_4+T_{obs}(\mathbf p_j-\mathbf c_1),
$$

so

$$
\boxed{\mathbf u_j=\mathbf d+(T_{obs}-I)\mathbf r_j},\qquad
\boxed{\rho_4=|\det T_{obs}|}.
$$

This is an exact representation of any three corresponding target points when the source triangle is nondegenerate. It is not a claim that the full optical system is globally affine. Three-point affine construction is also the construction documented by OpenCV.

For nonzero `rho_4`,

$$
T_{obs}=\sqrt{\rho_4}\,K,\qquad |\det K|=1.
$$

The four continuous map components separate into one area coordinate and three area-preserving deformation coordinates. On a fixed orientation branch these can describe relative rotation and two stretch/shear components. Their anatomical interpretation is not established by this decomposition.

For example, `T=I` and `T=diag(2,1/2)` have the same area ratio but different shapes. Only under a similarity relation `T=kR` do all corresponding lengths have a common ratio `k=sqrt(rho_4)`. The observable itself remains the area ratio.

### 4.1 Why a free map cannot validate its own points

A free affine map plus translation has six parameters and fits the six P4 coordinates exactly. An incorrectly selected P4 point can simply change that fitted map. Zero reconstruction residual then says only that the representation is complete.

This is analogous to fitting a line through exactly two points: the fit is exact even when one point is wrong. The observed map is useful for describing geometry, but its self-reconstruction is not a detection-quality test.

### 4.2 The shared-state restriction creates the test

Instead of fitting a new unrestricted map in each frame, calibrate shared coefficient functions

$$
\mathbf D=\mathbf D(x;\beta),\qquad
\mathcal T=\mathcal T(x;\beta).
$$

Then predict

$$
\boxed{\widehat{\mathbf v}_j=\mathbf D(x;\beta)
                     +\mathcal T(x;\beta)\mathbf r_j},
$$

$$
\boxed{\widehat{\mathbf q}_j=\mathbf c_1+\ell_1
 [\mathbf D(x;\beta)+\mathcal T(x;\beta)\mathbf r_j]}.
$$

After calibration, `beta` is fixed and only the two state components vary per frame. Different pairs can have different responses because their reference positions differ. The shared state, not equal pair displacement, is the physical constraint.

**The predictive formula uses the P1 centroid and a predicted displacement. It must not use the measured P4 centroid.** That would absorb common displacement and leak excluded P4 observations.

The predicted summaries are necessarily

$$
\widehat{\mathbf d}=\mathbf D(x;\beta),\qquad
\widehat\rho_4=|\det\mathcal T(x;\beta)|.
$$

Allowing arbitrary per-frame `D,T` would remove the restriction again. Their flexibility belongs in a globally calibrated, capacity-controlled response model, not unconstrained framewise parameters.

## 5. Geometric dimensions and scope of “full information”

Six 2D points supply 12 coordinates. Removing common translation removes two dimensions and common positive scale removes one. The remaining normalized geometry has nine continuous dimensions on a nondegenerate correspondence branch.

One complete representation is

$$
(\mathbf d,H_1,T_{obs}),\qquad H_1=E_1/\ell_1,\qquad |\det H_1|=2.
$$

| Component | Continuous geometric dimensions |
|---|---:|
| Normalized common displacement `d` | 2 |
| Relative triangle map `T_obs` | 4 |
| Normalized P1 triangle orientation/shape | 3 |
| Total | 9 |

Equivalently, retain normalized P1 geometry and the six components of `v` or `u`. A common-rotation quotient would remove one additional dimension; it is not part of the required normalization.

Two model scopes are legitimate but must not be confused:

**Conditional full-P4 model:** predict six P4 components given measured P1 geometry. Every point is used, but the three P1-shape dimensions are context, not separately predicted state observations. Their localization noise still contributes to the residual uncertainty.

**Joint full-normalized-geometry model:** also predict three nonredundant P1-shape/orientation coordinates, with joint covariance and supported state/nuisance dependence. This can use nine response components. It requires more calibration assumptions and is not automatically superior. A concrete constrained P1-shape chart and the extension are specified in the estimator plan.

Adding all crossed pairs or edges does not exceed these ranks. For example,

$$
(\mathbf q_j-\mathbf p_k)/\ell_1=\mathbf d+T_{obs}\mathbf r_j-\mathbf r_k.
$$

Geometric dimension is not a count of statistically independent errors or independently validated physiological information. Redundant feature copies and covariance regularization do not create new measurements.

## 6. Shared-state estimation and why additional measurements help

Stack a chosen nonredundant observation vector as `y`, and write

$$
y=F(x;\beta,\text{P1 context})+\epsilon.
$$

With calibrated coefficients fixed, the possible predictions form a two-dimensional state-dependent surface when the state Jacobian has rank two. The observations need not lie exactly on that surface.

For the conditional six-component model,

$$
\widehat x=\arg\min_{x\in\mathcal B}
 [y-F(x)]^\top R_e^{-1}[y-F(x)].
$$

The six observed components must agree with one two-state prediction. Locally this leaves four residual directions when no additional nuisance states are fitted. In contrast, a regular two-channel/two-state inverse leaves no residual direction; a near-zero optical residual is then unsurprising.

This supports two different checks:

**All-pair consistency:** fit the state with all three pairs and inspect the remaining geometry residual. This is a fitted-data model check, not an untouched prediction test.

**Held-out P4 prediction:** estimate the state without one P4 point and test its predicted position. This is the stronger test of the intended shared-state hypothesis, provided the remaining observations identify the state and calibration excludes the evaluation data.

Residuals can flag an incorrect point, a poor forward model, unmodeled geometry, or noise. They do not automatically identify the cause. Errors along the state-response directions can move the estimated state while leaving a small residual, so even additional channels cannot expose every common bias.

## 7. Optical response functions and the existing baseline

The geometry above is exact; its dependence on gaze/accommodation must be calibrated or modeled. One possible local raw-coordinate approximation is

$$
\mathbf P^*_{i,j}(\theta,A;\eta)
=[G_{i,j}(\eta)+a\,\Gamma_{i,j}(\eta)]
 [\theta^3,\theta^2,\theta,1]^\top,\qquad a=\phi(A).
$$

This is an optional empirical model, not a universal optical law. Choosing a logarithmic or power accommodation coordinate requires evidence; the two are not generally equivalent.

Let `M` be the raw centroid displacement and `Delta_i=det(E_i)` under this model. The induced summaries are

$$
\mathbf d=\mathbf M/\sqrt{|\Delta_1|/2},\qquad
\rho_4=|\Delta_4|/|\Delta_1|.
$$

Even when each edge is affine in `a`, its triangle determinant generally contains an `a^2` term:

$$
\det[\mathbf e_0+a\mathbf e_1,\mathbf f_0+a\mathbf f_1]
=\det[\mathbf e_0,\mathbf f_0]
+a\{\det[\mathbf e_1,\mathbf f_0]+\det[\mathbf e_0,\mathbf f_1]\}
+a^2\det[\mathbf e_1,\mathbf f_1].
$$

Cubic coordinate responses in gaze can produce determinant polynomials of degree six. Thus the normalized displacement generally involves a square-root denominator, and the area ratio a ratio of determinants. The old two-pair separation equations cannot be transferred by simply renaming the separation ratio as an area ratio.

For either state coordinate `z`, on a smooth nonzero-area branch,

$$
\partial_z\mathbf d=(\partial_z\mathbf m)/\ell_1
 -\frac{\mathbf d}{2\mathcal A_1}\partial_z\mathcal A_1,
$$

$$
\partial_z\rho_4=(\partial_z\mathcal A_4)/\mathcal A_1
 -\frac{\rho_4}{\mathcal A_1}\partial_z\mathcal A_1.
$$

P1-area changes can therefore affect both summaries. Assuming their state derivatives vanish requires evidence.

The frozen exp5 model has 13 coefficients, with numerical `t=theta/15` and `L=log1p(A)`:

$$
f_d=(b_0+b_1A)+(s_0+s_1L)t+(c_{20}+c_{21}L)t^2+c_3t^3,
$$

$$
f_\rho=(r_0+r_1t+r_2t^2)+L(r_3+r_4t+r_5t^2).
$$

The logarithm means `log(1+A/(1 D))` in physical units. These are empirical forward models of the two summaries, not exact reductions of the raw-point polynomial. The proposed full-position model predicts `D,T` and derives its area ratio from the determinant rather than fitting an unrelated extra area channel.

The historical 14-coefficient knot model and reduced-demand experiments are not the current exp5 estimator. Their old implementation links and numerical conclusions must not be treated as new full-position results.

## 8. Noise: shared P1 geometry is not exact input

Let `P` stack the 12 measured coordinates with covariance `Sigma_P`. For a differentiable feature map `h`, first-order uncertainty propagation is

$$
\Sigma_h\approx J_h\Sigma_PJ_h^\top.
$$

This is the multivariate first-order uncertainty propagation described by NIST. Nonlinear normalization can introduce bias and non-Gaussian errors, especially near degenerate triangles.

For the conditional model, P1 noise enters both the observed `v` and the predicted response through `r`. Therefore propagate the **complete residual**

$$
e(P;x,\beta)=v(P)-F(x,r(P);\beta),
$$

$$
\boxed{R_e\approx J_{e,P}\Sigma_PJ_{e,P}^\top},\qquad
J_{e,P}=J_{v,P}-J_{F,r}J_{r,P}.
$$

Propagating only `v` and treating the measured P1 context as noiseless misses shared-input correlations. All three residual blocks generally have correlated errors. The estimator plan specifies a fixed reference-covariance approximation so that weights do not covertly depend on held-out P4 values or change the profiled objective.

For reference, signed-area gradients for a triangle are

$$
\nabla_{p_1}a=\tfrac12(y_2-y_3,x_3-x_2)^\top,
\quad\nabla_{p_2}a=\tfrac12(y_3-y_1,x_1-x_3)^\top,
\quad\nabla_{p_3}a=\tfrac12(y_1-y_2,x_2-x_1)^\top.
$$

For unsigned area multiply by the sign of `a`, away from zero. These derivatives expose the common-denominator noise explicitly.

Noise estimation and preprocessing must use training data only. Adjacent-frame or second differences may contain true eye motion, temporally correlated detector error, and selection changes; they are not automatically pure localization noise. A covariance estimated from residuals after fitting free states can underestimate noise in fitted state directions.

A fixed invertible linear feature-basis change with correctly transformed covariance preserves Mahalanobis residuals exactly. General nonlinear changes with Jacobian-propagated covariance give only local equivalence. A robust loss does not correct a systematic model bias, and redundant residual copies must not be treated as independent.

## 9. Identifiability, nuisance geometry, and calibration uncertainty

At fixed calibration, define the prediction Jacobians

$$
J_x=[\partial_\theta F\;\partial_A F],\qquad J_\eta=\partial_\eta F,
\quad L^\top L=R_e^{-1}.
$$

Without fitted nuisance states or state priors, the regular first-order inverse response is

$$
\delta x\approx(J_x^\top R_e^{-1}J_x)^{-1}J_x^\top R_e^{-1}
 (\delta y-\delta F_{model}).
$$

Inspect the singular values of `L J_x diag(u_theta,u_A)` with declared degree/diopter reference increments. Small or nearly parallel noise-scaled state responses make the inverse unstable. A raw condition number without state-unit scaling is not interpretable across model choices.

When nuisance variables are fitted, inspect

$$
I_{x\mid\eta}=\widetilde J_x^\top
(I-\widetilde J_\eta\widetilde J_\eta^+)\widetilde J_x,
\qquad \widetilde J_x=LJ_x,\quad\widetilde J_\eta=LJ_\eta.
$$

The projection removes measurement changes that allowed nuisance variation can explain. Added geometry informs gaze/accommodation only to the extent that their remaining responses are distinguishable. It can still be useful as a failure diagnostic even when it adds little state precision.

For `k` nonredundant whitened components, the regular local residual dimension is

$$
k-\operatorname{rank}[\widetilde J_x\;\widetilde J_\eta].
$$

This count is local, with calibrated coefficients fixed; it is not a guarantee of global inverse uniqueness or a count of physiologically independent references. Priors add assumptions rather than optical channels. A scale-gauge convention does not identify freely varying physical scale.

Jointly fitting global coefficients and latent states introduces further coupling. Calibration uncertainty, model capacity, and nuisance assumptions must be propagated or assessed with grouped refits and sensitivity analysis. A flexible model trained on the evaluation frame can make a subsequent apparent point check circular.

## 10. The central experiment: predict a withheld P4 point

Assume coefficients were calibrated outside the evaluation fixation/capture. For point `j`, let `I` denote the other two P4 points. Keep all three P1 points as the reference.

Estimate

$$
\widehat x_I=\arg\min_{x\in\mathcal B}
 e_I(x)^\top [R_e]_{II}^{-1}e_I(x),
$$

then predict

$$
\widehat{\mathbf q}_{j\mid I}=\mathbf c_1+\ell_1
 [\mathbf D(\widehat x_I)+\mathcal T(\widehat x_I)\mathbf r_j].
$$

Compare this prediction with the observed `q_j` only after the state/branch result has been determined. Repeat for all three P4 points. A successful test says that a state inferred without that measurement predicted its response within the declared uncertainty and support.

Four retained scalar P4 components can locally identify two states and leave two fitted residual directions, but rank and branch checks are mandatory. Two points do not automatically give a unique state. If several plausible states remain, retain their predictions without choosing the one that best matches the excluded point. Report ambiguity or an inconclusive check where necessary.

### 10.1 What may and may not enter the state solve

All three P1 points and their area may be used: only a **P4 point** is withheld. A whole-pair holdout would remove a P1 point too and would require a different, explicitly defined normalization.

The excluded P4 measurement must not enter through full P4 centroid, area, triangle map, an all-three baseline inverse, initialization/warm starts, state priors, residual-based gates, weights, or branch selection. Nor may calibration or preprocessing have used the evaluation frame.

Whiten the retained residuals with their covariance marginal. Taking a principal block of the full precision matrix is generally different; selecting rows after full whitening may mix excluded coordinates into the solve.

A direct leakage test is to change the withheld P4 coordinate arbitrarily. The subset estimate, weights, starts, branches, and predicted point must stay unchanged. Only the final comparison with that point may change. Detector selection itself may use joint geometry; the test is conditional on those stored detections unless the detector is independently audited too.

### 10.2 Predictive uncertainty with correlated errors

At a regular interior subset solution, define

$$
K_I=(J_I^\top R_{II}^{-1}J_I)^{-1}J_I^\top R_{II}^{-1},\qquad M_j=J_jK_I.
$$

To first order the excluded-point error is `epsilon_j-M_j epsilon_I`. Its covariance is

$$
\boxed{V_j=R_{jj}+M_jR_{II}M_j^\top-R_{jI}M_j^\top-M_jR_{Ij}}.
$$

The cross terms account for shared P1 uncertainty. They must not be discarded merely because a P4 measurement was withheld. This expression conditions on calibrated coefficients; calibration uncertainty and model discrepancy need additional treatment.

Bounds, multiple branches, or weak identification can invalidate a single local Gaussian ellipse. Use prediction sets, profiles, or simulation in those cases. Calibrate alarm thresholds on training-only held-out predictions, with a stated policy for three simultaneous checks. A small residual is not a confidence guarantee, and disagreement does not by itself prove which point is wrong.

## 11. Calibration and evaluation principles

Nominal gaze and accommodation demand are **soft fixation-mean anchors**, not independent framewise physiological measurements. The proposed calibration estimates both global coefficients and framewise states with declared regularization. Demand should not be forced onto every frame as actual accommodation.

A prior based on an earlier model is model-derived evidence. Initialization alone does not add an objective term, but penalizing deviations from prior model states does. Declare that distinction and evaluate sensitivity, including removal of such penalties. Temporal smoothing may stabilize a trajectory, but cannot substitute for an optical cross-check.

Separate three levels of evidence:

**Fitted geometric consistency:** the all-three observations lie close to the fitted shared-state model.

**Held-out predictive consistency:** states inferred without a P4 point predict it on data excluded from calibration/model selection.

**Physiological accuracy:** inferred states agree with independent gaze/accommodation references.

The first is weaker than the second, and neither establishes the third. Shared biases can survive agreement across all pairs. Conversely, extra geometry can be valuable for rejection/uncertainty diagnostics even when it does not materially improve state precision.

Keep entire fixations/captures together when splitting data; adjacent frames are not independent test examples. The frozen all-data baseline is a historical reference, not a clean held-out control on captures it trained on. Retrain controls within folds with matched support and declared capacity, anchor, noise, and temporal policies.

Captures 5/6 are unlabeled in the handoff and therefore support geometric/trajectory checks, not physiological error estimates. Whole-capture and demand effects can be confounded. Report inverse branches, bounds, conditioning, extrapolation in both state and P1 context, numerical failure, rejected/missing observations, and coverage. Do not equate computational bounds with calibrated support or physiological limits.

## 12. Design conclusion and proposed algorithm

The shared-state construction is

$$
\boxed{
\ell_1=\sqrt{\mathcal A_1},\quad
\mathbf r_j=(\mathbf p_j-\mathbf c_1)/\ell_1,\quad
\widehat{\mathbf v}_j=\mathbf D(\theta,A)+\mathcal T(\theta,A)\mathbf r_j,
\quad \widehat\rho_4=|\det\mathcal T(\theta,A)|.
}
$$

Fit the same two states to all pair responses for the primary estimate. For validation, withhold each P4 point in turn and predict it from the remaining geometry. This is the principal reason to retain three corresponding pairs.

[ESTIMATOR_PLAN.md](ESTIMATOR_PLAN.md) specifies the recommended first implementation: a 27-coefficient conditional geometry model, an explicit 37-coefficient capacity comparison, shared-input covariance, bounded variable-projection calibration with soft mean anchors, independent multistart state inversion, and leakage-safe P4 holdouts. It also specifies a genuine nine-component extension when P1 shape can be justified as additional state evidence.

No estimator implementation, new calibration, or new physiological result is asserted by this theory revision.

## Sources and historical boundary

Repository sources:

- [HANDOFF.md](HANDOFF.md): data scope, correspondence, frozen estimator, and historical experiment boundaries.
- [joint_m2.py](joint_m2.py): current triangle measurements, validity, training setup, and independent bounded inversion.
- [models/quadratic_model.json](models/quadratic_model.json): frozen coefficients, covariance, support, and recorded training diagnostics.
- [Saved capture summaries](experiments/captures_5_6_direct/summary.json): existing application results; not newly regenerated here.

Method references:

- [OpenCV: Geometric Image Transformations, getAffineTransform](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html): affine construction from three corresponding points.
- [NIST: Combining uncertainty components](https://physics.nist.gov/cuu/Uncertainty/combination.html): first-order uncertainty propagation with covariance.
- [SciPy: least_squares](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html): bounded least-squares optimization and solver diagnostics for the proposed algorithm.
- [scikit-learn: Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html): separation of training/model selection from grouped evaluation.

The triangle identities, conditional residual model, information accounting, and correlated held-out covariance are derived in this document. General noise/identification/validation principles are retained from the earlier theory. The old two-pair horizontal document remains in Git history (blob `cace05e50b242052d3d8b7727a90c592b3dc2f06`); its separation ratios, crossed-pair identities, knot models, and historical outcomes are not definitions or results of this three-pair estimator.
