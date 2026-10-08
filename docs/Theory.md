# Three-pair theory: state-dependent grid distortion and shared gaze/accommodation

## Purpose and status

**P1 and P4 are two differently distorted images of the same fixed source pattern. One physical gaze/accommodation state must explain their translation and deformation simultaneously.** The reason to retain three pairs is not merely to average more points or to estimate a more precise area. It is to sample several responses of the changing optical system, so that measurements not used to estimate the state can test its predictions.

The three pairs need not move equally or preserve triangle similarity. Their response functions can differ, but they share one horizontal gaze angle `theta` and one accommodation value `A` in each frame. Reproducible state-dependent deformation is a candidate measurement signal, not an artifact to remove before estimation. The central test is:

> Estimate gaze and accommodation from part of the measured geometry, then ask whether that same state predicts the remaining P4 point, including its expected distorted position.

This document defines the optical interpretation, measurement geometry, shared-state model, and meaning of that test. [ESTIMATOR_PLAN.md](ESTIMATOR_PLAN.md) records the design and implementation scope. The conditional six-residual full-position estimator has been implemented and evaluated as an exploratory prototype; see [CURRENT_STATUS.md](CURRENT_STATUS.md) for execution status and results. The optional nine-component joint P1/P4 estimator remains deferred. Neither implementation nor geometric agreement establishes physiological accuracy. The frozen 13-coefficient two-channel estimator remains the comparison baseline in [models/quadratic_model.json](../models/quadratic_model.json).

The required observables remain exactly

$$
\boxed{d_x=\frac{\operatorname{mean}(P4_x)-\operatorname{mean}(P1_x)}
 {\sqrt{\operatorname{area}(P1)}}},\qquad
\boxed{\rho_4=\frac{\operatorname{area}(P4)}{\operatorname{area}(P1)}}.
$$

The square root belongs in the displacement denominator, not in `rho_4`. Area is the triangle formed by the three reflection centers, not the summed areas of the detected blobs. Keep this normalization for the first full-position comparison; retain the coordinate information that the two summaries compress.

## 1. Correspondence, calibration convention, and physical state

The retained y components must be interpreted using their actual geometry.
For excluded point j and retained a,b, the centered P1 contexts satisfy
`r_a+r_b=-r_j`. Thus retained-pair common y is
`(v_a,y+v_b,y)/2 = D_y - T_y*r_j/2`; it is not pure translation. Differential
y is `v_a,y-v_b,y = T_y*(r_a-r_b)` and cancels the shared displacement.
Poor common-y cross-prediction may reflect displacement, deformation, context,
weighting or state confounding; it does not establish that shared image motion
is intrinsically useless.

A targeted discrepancy candidate adds a shared vertical mode to the full
normalized residual covariance: `R_tau=R_0+(tau_px/ell1)^2 a a^T`, with
`a=[0,1,0,1,0,1]`. Marginalize this covariance before each subset transformation.
Tau is declared/estimated from training or inner validation, never held-point
error. Eliminating an unrestricted retained shared-y residual offset in the
quadratic cost is equivalent to retaining x and differential y. This algebra
motivates a small covariance continuum; it does not prove a free offset is the
true discrepancy. Keep raw excluded-coordinate prediction as the scientific test.

Let

$$
\mathbf p_j=\mathbf P_{1,j},\qquad
\mathbf q_j=\mathbf P_{4,\pi(j)},\qquad j=1,2,3,
$$

be corresponding two-dimensional image points in pixels, with x right and y down. The permutation `pi` converts stored P4 slots to P1 correspondence. The stored detections use zero-based `pair_index=[2,1,0]`. Read each payload's metadata and reorder coordinates and flags consistently. Sorting points by x does not itself establish persistent illuminator identity.

The state is `x=(theta,A)`, where horizontal gaze is in degrees and accommodation is in diopters. The user clarified the calibration protocol: every fixation has nominal vertical gaze zero degrees. This is a protocol constraint, not measured framewise vertical truth. Image-y observations and their derivatives with respect to horizontal gaze and accommodation remain unrestricted. Artifacts record `state_variables=[theta_x_deg,A_D]`, `calibration_theta_y_nominal_deg=0`, the protocol role, and the stored image-axis convention. A later `(theta_x,theta_y,A)` model would need appropriate calibration and a new rank analysis. Use `eta` for explicitly declared geometry/pose/imaging nuisance variables, not arbitrary per-frame freedoms introduced to eliminate residuals.

### 1.1 Current gaze targets, scaling, and bounds are different quantities

The current experiment's five nominal gaze calibration targets are

$$
\boxed{\theta^{nom}\in\{-10,-5,0,5,10\}\;\mathrm{degrees}.}
$$

Read their assignments from [fixation_intervals.json](../data/fixations/fixation_intervals.json). Do not import the historical `[-15,-7.5,0,7.5,15]` target array from older experiments or use it to relabel these intervals. The gaze targets are not the source-grid coordinates or illumination angles.

For the proposed full-position models define

$$
t=\theta/(10\;\mathrm{degrees}),\qquad
L_A=\log(1+A/(1\;\mathrm D)).
$$

Numerically this is `t=theta_deg/10` and `L_A=log1p(A_diopters)`; the nominal gaze targets become `[-1,-0.5,0,0.5,1]`. The frozen baseline separately uses `t_b=theta_deg/15`. Its divisor is a historical coefficient/basis convention, **not a claim that this experiment calibrates to 15 degrees**. Never evaluate its unchanged coefficients with the new divisor. New model artifacts must record their gaze scale and schema.

The full-data nominal gaze anchor range is `[-10,10]` degrees. Each training fold has its own, possibly smaller, support. The proposed computational inversion box remains `theta in [-20,20]` and `A in [0,6]`; it allows diagnostic solutions outside the calibration range, not a claim of validated coverage there. Nominal labels, basis scaling, computational bounds, and physiological limits must not be conflated.

### 1.2 Triangle geometry

Define centroids and edge matrices

$$
\mathbf c_1=\tfrac13\sum_j\mathbf p_j,\qquad
\mathbf c_4=\tfrac13\sum_j\mathbf q_j,
$$

$$
E_1=[\mathbf p_2-\mathbf p_1\;\;\mathbf p_3-\mathbf p_1],\qquad
E_4=[\mathbf q_2-\mathbf q_1\;\;\mathbf q_3-\mathbf q_1].
$$

Signed and unsigned areas and the P1 length normalizer are

$$
a_i=\tfrac12\det E_i,\qquad \mathcal A_i=|a_i|,\qquad
\ell_1=\sqrt{\mathcal A_1}.
$$

`mathcal A_i` denotes image area, not accommodation `A`. Preserve signed areas and correspondence parity: the absolute value discards orientation sign and is nondifferentiable at a zero-area crossing.

Finite coordinates and reliable correspondence are prerequisites. A P1 triangle must be sufficiently nondegenerate relative to localization uncertainty, not merely nonzero in floating-point arithmetic. Thin triangles can make edge-map inversion poorly conditioned. Any stronger validity gate than the baseline must be separately specified and evaluated on matched support. Non-similarity to a reference triangle is not, by itself, an invalidity criterion.

## 2. Complete relative measurements and the baseline reduction

Define P1-centered, area-normalized reference geometry and P4 positions:

$$
\mathbf r_j=\frac{\mathbf p_j-\mathbf c_1}{\ell_1},\qquad
\mathbf v_j=\frac{\mathbf q_j-\mathbf c_1}{\ell_1}.
$$

The matched displacement is

$$
\mathbf u_j=\frac{\mathbf q_j-\mathbf p_j}{\ell_1}
=\mathbf v_j-\mathbf r_j.
$$

These are dimensionless, and the P1 triangle satisfies

$$
\sum_j\mathbf r_j=0,\qquad
\operatorname{area}(\mathbf r_1,\mathbf r_2,\mathbf r_3)=1.
$$

The common displacement is

$$
\boxed{\mathbf d=\frac{\mathbf c_4-\mathbf c_1}{\ell_1}
=\tfrac13\sum_j\mathbf u_j=\tfrac13\sum_j\mathbf v_j}.
$$

Write differential motion as

$$
\mathbf w_j=\mathbf u_j-\mathbf d,\qquad \sum_j\mathbf w_j=0.
$$

Three normalized displacement vectors contain six scalar components: two common-motion components and four differential-motion components. A nonredundant displacement basis is `(d,w1,w2)`, with `w3=-w1-w2`.

Since normalized P1 area is one,

$$
\boxed{\rho_4=\operatorname{area}(\mathbf v_1,\mathbf v_2,\mathbf v_3)
=\operatorname{area}(\mathbf r_1+\mathbf w_1,
                     \mathbf r_2+\mathbf w_2,
                     \mathbf r_3+\mathbf w_3)}.
$$

A common P4 translation changes `d` but not triangle area. Differential motion can alter area, orientation, or shape. The area ratio preserves only one scalar aspect of that motion. State-dependent differential motion is potentially useful even when it preserves area.

The baseline retains `y_0=[d_x,rho_4]^T`. Adding `d_y` supplies a third channel but still not the full geometry. Calling `d_x` a gaze channel and `rho_4` an accommodation-sensitive channel is descriptive, not a separability assumption: both can depend on both states and nuisance geometry.

**Three pairs reduced to two scalar summaries still provide only two channels for two states.** Predictive cross-checks require individual positions/displacements or an equivalent complete representation. Do not append `rho_4` as an independently weighted residual when all retained P4 positions and P1 geometry already determine it.

## 3. What normalization removes—and what it must preserve

For a common positive isotropic scale and additive translation,

$$
\mathbf p'_j=\lambda\mathbf p_j+\mathbf b,\qquad
\mathbf q'_j=\lambda\mathbf q_j+\mathbf b,\qquad \lambda>0,
$$

area scales by `lambda^2` and length by `lambda`. Consequently `r,v,u,d,rho_4` are unchanged. This explains square-root P1 area for displacements and P1 area for an area ratio.

For a shared proper image rotation, the vectors rotate while the area ratio remains unchanged. Horizontal displacement remains tied to the camera x-axis. Optional P1-derived rotation normalization removes an additional component and must be evaluated rather than assumed harmless. Do not normalize each P4 pattern to a similarity template before fitting: doing so can remove the sought state-dependent response.

For an identical nonsingular affine transformation `B` of all six image points,

$$
\rho'_4=\rho_4,\qquad
\mathbf d'=B\mathbf d/\sqrt{|\det B|}.
$$

The area ratio has common-affine invariance, whereas the displacement vector generally does not. These are statements about a transformation applied identically to every measured point. They do not establish invariance to physical eye motion, differential magnification, perspective, optical-surface changes, or detector bias. In particular, genuine gaze/accommodation-induced distortion need not be shared by P1 and P4 and is not something this normalization is intended to cancel.

P1 area need not be constant or state-independent. Its variation and uncertainty remain relevant even though it is the denominator. Normalization also does not supply an independent eye-camera distance measurement.

Other size measures, such as P1 RMS spread, may be evaluated later under separate schemas. A different normalizer is a separate hypothesis from retaining more geometry; do not change both in the first comparison and attribute all improvement to additional pairs.

## 4. Exact sampled geometry versus a predictive optical model

For a nondegenerate P1 triangle define the observed map

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

This is the exact three-correspondence affine construction, not a claim that the continuous optical mapping is affine. See the [OpenCV construction](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html). For finite triangles, `T_obs` is an effective map of those three samples, not necessarily the spatial Jacobian of the optical mapping at the triangle center. Its determinant is their exact area ratio, not automatically a pointwise optical magnification determinant.

For nonzero `rho_4`, write

$$
T_{obs}=\sqrt{\rho_4}\,K,\qquad |\det K|=1.
$$

The four continuous map components separate into one area coordinate and three area-preserving deformation coordinates. On a fixed orientation branch these can describe relative rotation and two stretch/shear components. Such descriptors are not automatically anatomical parameters or Seidel aberration coefficients.

For example, `T=I` and `T=diag(2,1/2)` have the same area ratio but different shapes. Only for a similarity relation `T=kR` do all corresponding lengths share the ratio `k=sqrt(rho_4)`. Fitted similarity magnification is therefore a reduced comparison, not a preferred replacement for full coordinates when nonuniform distortion is part of the signal.

### 4.1 A free map cannot validate its own samples

A free affine map plus translation has six parameters and fits the six P4 coordinates exactly. An incorrect P4 point can simply change that map. Zero reconstruction error then establishes completeness of the representation, not correct detection. This is analogous to fitting a line through exactly two possibly erroneous points.

Spatially nonlinear distortion can also produce three vertices that this affine representation fits exactly. Three points alone cannot reveal the distortion between them or uniquely identify unrestricted higher-order field coefficients in one frame. Additional source samples or a justified shared optical model across calibrated conditions are needed for those stronger claims.

### 4.2 A shared-state response creates the prediction test

Instead of a new unrestricted map per frame, calibrate

$$
\mathbf D=\mathbf D(x;\beta),\qquad
\mathcal T=\mathcal T(x;\beta)
$$

and predict

$$
\boxed{\widehat{\mathbf v}_j=\mathbf D(x;\beta)
                     +\mathcal T(x;\beta)\mathbf r_j},
$$

$$
\boxed{\widehat{\mathbf q}_j=\mathbf c_1+\ell_1
 [\mathbf D(x;\beta)+\mathcal T(x;\beta)\mathbf r_j]}.
$$

After calibration `beta` is fixed; only `theta,A` vary per frame. Different source samples can have different responses while sharing the same state. `D` describes common relative displacement and `T` the sampled differential response, including state-dependent distortion.

**Use the P1 centroid and predicted displacement, not the measured P4 centroid.** Using the latter would absorb displacement error and leak a withheld P4 point. The implied summaries are

$$
\widehat{\mathbf d}=\mathbf D(x;\beta),\qquad
\widehat\rho_4=|\det\mathcal T(x;\beta)|.
$$

A matrix that depends only on `x` is a calibrated conditional approximation for the declared P1 context and deployment conditions. Its transfer to arbitrary P1 shapes/poses is not guaranteed by the exact three-point identity. Test context support and residual dependence; introduce additional context or nuisance dependence only when justified and identifiable.

### 4.3 Expected distortion is signal; unexplained distortion is residual

The meaningful error is observed position minus the position predicted by the complete calibrated response—not departure from an undistorted triangle. Large but reproducible deformation can be a valid observation. Conversely, a visually similar triangle can be inconsistent with the state model.

Do not force `T` to be a similarity, force determinant-one deformation to be constant, or discard angle/edge changes by default. Compare restricted models explicitly when testing whether such components matter. Do not add free per-frame distortion coefficients to make the full model's residual disappear: the expected distortion must be determined by the same two states and fixed coefficients.

## 5. Geometric dimensions and scope of “full information”

Six 2D points supply 12 coordinates. Removing common translation removes two dimensions; removing common positive scale removes one. The normalized geometry has nine continuous dimensions on a nondegenerate correspondence branch.

A complete representation is

$$
(\mathbf d,H_1,T_{obs}),\qquad H_1=E_1/\ell_1,\qquad |\det H_1|=2.
$$

| Component | Continuous geometric dimensions |
|---|---:|
| Normalized common displacement `d` | 2 |
| Relative triangle map `T_obs` | 4 |
| Normalized P1 orientation/shape | 3 |
| Total | 9 |

Equivalently, retain normalized P1 geometry and the six components of `v` or `u`. Removing common rotation would remove one additional dimension; that is not the required normalization.

**Conditional full-P4 model:** predict six P4 components given measured P1 geometry. Every point is used, but the three P1-shape dimensions are context rather than separately predicted state evidence. P1 localization noise still contributes to residual uncertainty. This does not assert that the P1 optical mapping is constant.

**Joint full-normalized-geometry model:** also predict three nonredundant P1 shape/orientation coordinates with joint covariance and supported state/nuisance dependence. This supplies nine response components but needs additional calibration assumptions. The estimator plan provides a constrained P1-shape chart for this optional extension.

Appending every crossed pair or edge does not exceed these ranks. For example,

$$
(\mathbf q_j-\mathbf p_k)/\ell_1=\mathbf d+T_{obs}\mathbf r_j-\mathbf r_k.
$$

Geometric dimension is not a count of statistically independent errors or physiological references. Redundant residuals and covariance regularization do not create new information.

## 6. Shared-state estimation and predictive consistency

Stack a nonredundant selected observation vector as `y` and write

$$
y=F(x;\beta,\text{P1 context})+\epsilon.
$$

At fixed calibration and P1 context, predictions form a two-dimensional state surface when the state Jacobian has rank two. The observations need not lie on it. For the conditional six-component model,

$$
\widehat x=\arg\min_{x\in\mathcal B}
 [y-F(x)]^\top R_e^{-1}[y-F(x)].
$$

Six observed components must agree with one two-state prediction, leaving four local residual directions if no nuisance states are fitted. A regular two-channel/two-state inverse has no residual direction, so a near-zero optical residual is unsurprising.

**All-pair consistency:** fit all three pairs and inspect remaining geometric error. This is a fitted-data check.

**Held-out P4 prediction:** estimate the state without one P4 point and test its predicted, potentially distorted position. This is stronger when the remaining measurements identify the state and calibration excludes evaluation data.

A residual may indicate incorrect selection, inadequate state-response capacity, unmodeled geometry, localization bias, or random noise. It does not automatically identify the cause. A bias along a state-response direction can change the estimate while leaving a small residual; extra measurements cannot detect every shared error.

## 7. Source-grid imaging and field-dependent optical responses

### 7.1 A fixed source pattern through two changing mappings

Let `s_j` identify a fixed source-grid sample in a declared source plane or angular parameterization. Model its images as

$$
\boxed{\mathbf p_j=\mathcal F_1(\mathbf s_j;\theta,A,\eta),\qquad
\mathbf q_j=\mathcal F_4(\mathbf s_j;\theta,A,\eta)}.
$$

The maps include the relevant eye/reflection path and imaging relay. Source-grid coordinates are not the five gaze targets. If physical source coordinates are unavailable, retain sample identity and leave the coordinates unspecified; do not infer a metric grid by independently sorting distorted images.

P1 and P4 originate at the anterior cornea and posterior lens surface, respectively. Their optical paths are different. The dDPI optical study discusses refraction omitted by the simple two-mirror approximation and ray-traced changes in reflection positions/shapes with eye and illumination angle; its simulations concern an unaccommodated eye, not validation of this joint estimator. See [Wu et al. (2023)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10166114/).

The present modeling premise is that rotation changes the field sampled relative to the eye, while accommodation changes lens geometry/optical power and therefore the P4 mapping, including possible field-dependent deformation. Neither mechanism is restricted to uniform translation or magnification. P1 measured geometry also need not be state-independent. Their exact strengths, symmetry, and reproducibility in these captures remain to be established.

An abstract decomposition is

$$
\mathbf h_j=\mathbf h(\mathbf s_j,\theta,\eta),\qquad
\mathcal F_i=\mathcal G_i(\mathbf h_j;\theta,A,\eta).
$$

Allowing explicit `theta` in `G_i` avoids assuming that changing eye orientation only translates field coordinates in an otherwise fixed camera mapping. The chain rule gives

$$
\partial_\theta\mathcal F_i
=(\partial_{\mathbf h}\mathcal G_i)\partial_\theta\mathbf h_j
 +\left.\partial_\theta\mathcal G_i\right|_{\mathbf h}.
$$

The field derivative can differ across source samples. A fixed off-axis illuminator also means `theta=0` need not be zero optical field. Do not require distortion to vanish at straight-ahead gaze, grow monotonically with absolute gaze, or have perfect even/odd symmetry without evidence.

### 7.2 Illustrative field distortion: translation changes spacing too

For illustration only, use one dimension and small angular coordinates `s_j,vartheta,h_j` expressed consistently in radians:

$$
h_j=s_j-\vartheta,\qquad
X_j=b(\vartheta,A)+m(A)h_j+k(A)h_j^3.
$$

This is a toy field model, not the fitted eye model or a universal third-order aberration law. For two field samples `h_0-a` and `h_0+a` with fixed half-spacing `a`, subtracting their positions gives

$$
\Delta X=2a\{m(A)+k(A)[3h_0^2+a^2]\}.
$$

Therefore

$$
\boxed{\Delta X/(2a)=m(A)+k(A)[3h_0^2+a^2]}.
$$

Moving the sampled pattern across the field changes its apparent spacing even when accommodation and the distortion coefficients remain fixed. Changing accommodation can alter both magnification `m` and distortion coefficient `k`. The toy point responses are

$$
\partial_\vartheta X_j=b_\vartheta-m-3kh_j^2,\qquad
\partial_A X_j=b_A+m'h_j+k'h_j^3.
$$

Different source points thus have different state sensitivities. These equations motivate testing gaze curvature and gaze/accommodation interactions in the calibrated deformation, not assuming that shape is constant or using one scalar as the sole accommodation channel. Their angular convention is separate from the degree-valued estimator basis; any implemented derivative must convert units explicitly.

In two dimensions, a local expansion may contain linear, quadratic, and cubic functions of the two field coordinates with coefficients depending on optical state. Do not assume a radial-only camera-distortion formula applies unchanged to the ocular reflection paths. Three samples do not uniquely determine all such coefficients.

### 7.3 Spatial nonlinearity versus state-response nonlinearity

The continuous grid mapping can be spatially nonlinear, while an effective affine map still reproduces its three observed samples exactly. Across frames, that effective map can vary nonlinearly with `theta,A`. Thus predicting `D(theta,A)` and `T(theta,A)` is compatible with distorted optics; it does not assume a distortion-free grid.

At fixed measured P1 context, the conditional prediction derivative is

$$
\left.\partial_z\widehat{\mathbf v}_j\right|_{\mathbf r}
=\partial_z\mathbf D+(\partial_z\mathcal T)\mathbf r_j.
$$

Along a physical trajectory where P1 geometry changes with the state, the total derivative additionally includes `T partial_z r_j`. Do not equate a fixed-context inverse Jacobian with this total optical response or count P1-shape information twice. A joint P1/P4 optical model must handle both responses explicitly.

Low-degree response functions are approximations to these sampled mappings. Compare the 27- and 37-coefficient candidates under grouped prediction tests, especially at the five gaze conditions. A finite-degree polynomial and three sampled sites do not establish a unique optical prescription or aberration decomposition.

### 7.4 Relation to raw-coordinate polynomials and the frozen baseline

One optional local point approximation is

$$
\mathbf P^*_{i,j}(\theta,A;\eta)
=[G_{i,j}(\eta)+a_A\Gamma_{i,j}(\eta)]
 [\theta^3,\theta^2,\theta,1]^\top,\qquad a_A=\phi(A).
$$

This is empirical; power and logarithmic accommodation coordinates are not generally equivalent. If `M` is the raw centroid displacement and `Delta_i=det(E_i)`, the induced summaries are

$$
\mathbf d=\mathbf M/\sqrt{|\Delta_1|/2},\qquad
\rho_4=|\Delta_4|/|\Delta_1|.
$$

Even edges affine in `a_A` generally give a quadratic determinant:

$$
\det[\mathbf e_0+a_A\mathbf e_1,\mathbf f_0+a_A\mathbf f_1]
=\det[\mathbf e_0,\mathbf f_0]
+a_A\{\det[\mathbf e_1,\mathbf f_0]+\det[\mathbf e_0,\mathbf f_1]\}
+a_A^2\det[\mathbf e_1,\mathbf f_1].
$$

Cubic gaze-coordinate responses can produce degree-six determinants. The normalized displacement generally has a square-root denominator and the area ratio a ratio of determinants. The old pair-separation equations cannot be transferred by simply renaming the separation ratio.

For either state coordinate `z`, on a nonzero-area branch,

$$
\partial_z\mathbf d=(\partial_z\mathbf m)/\ell_1
 -\frac{\mathbf d}{2\mathcal A_1}\partial_z\mathcal A_1,
\qquad
\partial_z\rho_4=(\partial_z\mathcal A_4)/\mathcal A_1
 -\frac{\rho_4}{\mathcal A_1}\partial_z\mathcal A_1.
$$

P1-area state derivatives may not be set to zero merely because P1 is the reference.

The **frozen baseline only** uses `t_b=theta_deg/15` and `L_A=log1p(A_diopters)`:

$$
f_d=(b_0+b_1A)+(s_0+s_1L_A)t_b+(c_{20}+c_{21}L_A)t_b^2+c_3t_b^3,
$$

$$
f_\rho=(r_0+r_1t_b+r_2t_b^2)+L_A(r_3+r_4t_b+r_5t_b^2).
$$

These 13 coefficients model summaries empirically, not as exact reductions of the raw-point polynomial. The proposed model uses `t=theta_deg/10`, fits new coefficients, and derives area ratio from its predicted matrix. The historical 14-coefficient knot model and older 15-degree target protocol are not this experiment. Preserve baseline coefficients and evaluation conventions unchanged.

## 8. Measurement definition, shared-input noise, and model discrepancy

The observables are detector-reported image locations. A ray-traced chief-ray intersection, intensity centroid, fitted spot center, and peak-intensity position are distinct measurement definitions. Freeze and document the location definition and detector configuration. In ray-tracing comparisons, distinguish geometric position from the position that the actual image/localization pipeline would report. Do not automatically interpret a point-location shift as a particular aberration coefficient.

Expected state-dependent centroid shifts belong in the calibrated response. Random localization noise and systematic detector/model errors require separate treatment. Increasing covariance or robustly downweighting peripheral points must not be used to conceal reproducible field-dependent model error. Raw spot shape, asymmetry, saturation, and clipping can be useful diagnostics when available; they are not additional estimator channels in this proposal.

Let `P` stack all 12 coordinates with covariance `Sigma_P`. For a differentiable feature map, first-order propagation is

$$
\Sigma_h\approx J_h\Sigma_PJ_h^\top.
$$

This is the multivariate first-order uncertainty rule described by [NIST](https://physics.nist.gov/cuu/Uncertainty/combination.html), not an exact distributional statement for nonlinear ratios.

For the conditional model, P1 noise affects both observed `v` and predicted response through `r`. Propagate the complete residual:

$$
e(P;x,\beta)=v(P)-F(x,r(P);\beta),
$$

$$
\boxed{R_e\approx J_{e,P}\Sigma_PJ_{e,P}^\top},\qquad
J_{e,P}=J_{v,P}-J_{F,r}J_{r,P}.
$$

Propagating only `v` while treating P1 context as exact omits shared-input error. The three residual blocks are generally correlated. The plan specifies a frozen reference-covariance approximation, avoiding dependence on excluded P4 measurements and preserving a fixed profiled objective.

Signed triangle-area gradients are

$$
\nabla_{p_1}a=\tfrac12(y_2-y_3,x_3-x_2)^\top,
\quad\nabla_{p_2}a=\tfrac12(y_3-y_1,x_1-x_3)^\top,
\quad\nabla_{p_3}a=\tfrac12(y_1-y_2,x_2-x_1)^\top.
$$

For unsigned area multiply by `sign(a)`, away from zero. Near degeneracy, first-order approximations can fail through ratio bias, heavy tails, or orientation changes.

Estimate noise and preprocessing from training data only. Temporal differences can contain true motion, detector correlation, and selection changes. Covariance of residuals after fitting free states can underestimate noise along fitted directions. A fixed invertible linear basis change preserves Mahalanobis residuals with correctly transformed covariance; nonlinear transformations give only local equivalence under Jacobian propagation. Robust losses do not remove systematic bias, and redundant features must not be weighted as independent observations.

## 9. Identifiability, nuisance geometry, and distortion sensitivity

At fixed calibration and declared P1 context define

$$
J_x=[\partial_\theta F\;\partial_A F],\qquad
J_\eta=\partial_\eta F,\qquad L^\top L=R_e^{-1}.
$$

Without fitted nuisance states or state priors, the regular local inverse response is

$$
\delta x\approx(J_x^\top R_e^{-1}J_x)^{-1}J_x^\top R_e^{-1}
 (\delta y-\delta F_{model}).
$$

Inspect singular values of `L J_x diag(u_theta,u_A)` using declared physical increments. Basis scaling `theta/10` is not a substitute for declaring those degree/diopter units. Nearly parallel or weak noise-scaled responses make the inverse unstable.

For fitted nuisance variables use

$$
I_{x\mid\eta}=\widetilde J_x^\top
(I-\widetilde J_\eta\widetilde J_\eta^+)\widetilde J_x,
\qquad \widetilde J_x=LJ_x,\quad\widetilde J_\eta=LJ_\eta.
$$

A distortion component need not be exclusively gaze- or accommodation-sensitive to help. Its additional response is useful when it helps distinguish the two states after allowing supported nuisance changes. Conversely, a strongly changing feature can be uninformative if other effects reproduce the same signature.

For example, let `j_theta,j_A` be whitened, physically scaled state columns and let `N=[j_theta, Jeta_w]`. The accommodation response not reproduced by gaze or allowed nuisance variation is

$$
j_{A,\perp}=(I-NN^+)j_A.
$$

This is a local information diagnostic, not a physiological validation or permission to assume the noise model correct. Compare state precision and conditioning as well as withheld-point predictions; small training residuals alone do not establish useful sensitivity.

For `k` nonredundant whitened components the local residual dimension is

$$
k-\operatorname{rank}[\widetilde J_x\;\widetilde J_\eta].
$$

This count fixes calibrated coefficients and is not a guarantee of global uniqueness. Priors add assumptions rather than optical channels. A scale-gauge convention does not identify freely varying physical scale. Calibration uncertainty, capacity, and nuisance assumptions require grouped refits or explicit propagation. A flexible model trained on an evaluation frame can make a subsequent apparent point test circular.

## 10. Central experiment: predict a withheld P4 point

Calibrate outside the evaluation fixation/capture. For P4 point `j`, let `I` contain the other two P4 points; keep all three P1 points as the reference.

Estimate

$$
\widehat x_I=\arg\min_{x\in\mathcal B}
 e_I(x)^\top[R_e]_{II}^{-1}e_I(x),
$$

then predict

$$
\widehat{\mathbf q}_{j\mid I}=\mathbf c_1+\ell_1
 [\mathbf D(\widehat x_I)+\mathcal T(\widehat x_I)\mathbf r_j].
$$

Compare with observed `q_j` only after determining the subset state and branches. Repeat for all three points. The prediction includes expected distortion; it is not a prediction under triangle similarity unless evaluating that explicitly restricted model.

Four retained scalar coordinates can locally identify two states and leave two residual directions, but rank and branch checks are mandatory. If several plausible states remain, retain their prediction set without using the excluded point to choose among them. Weak rank or ambiguity yields an inconclusive test, not evidence that the excluded point is wrong.

### 10.1 Leakage restrictions

All three P1 points and their area are permitted because only a **P4 point** is withheld. A whole-pair holdout removes a P1 point too and requires a different normalization.

The excluded P4 value must not enter through full P4 centroid, area, observed map, similarity scale, an all-three baseline inverse, initialization/warm starts, state priors, residual gates, weights, or branch selection. Calibration and learned preprocessing must exclude the evaluation frame.

Use the retained covariance marginal before inversion. A principal block of the full precision is generally different, and selecting rows after full whitening can mix excluded coordinates into the solve.

A mandatory test changes the withheld P4 coordinate arbitrarily: subset preprocessing, weights, starts, states, branch sets, and predicted point must remain unchanged. Only the subsequent prediction error may change. Upstream detector selection may already depend on triangle assumptions; this is conditional validation of stored detections unless that selection is audited independently.

### 10.2 Correlated predictive uncertainty

At a regular interior subset inverse, define

$$
K_I=(J_I^\top R_{II}^{-1}J_I)^{-1}J_I^\top R_{II}^{-1},\qquad M_j=J_jK_I.
$$

To first order the withheld error is `epsilon_j-M_j epsilon_I`, with covariance

$$
\boxed{V_j=R_{jj}+M_jR_{II}M_j^\top-R_{jI}M_j^\top-M_jR_{Ij}}.
$$

The cross terms reflect shared P1 uncertainty and may not be discarded. This formula conditions on calibration; coefficient uncertainty and model discrepancy require additional treatment. Near bounds, weak rank, or multiple branches, use profiles, simulation, or prediction sets instead of one Gaussian ellipse. Calibrate thresholds using training-only held-out predictions and a declared policy for three simultaneous checks. Without clean references, empirical exceedance rates are not validated detector false-positive rates.

## 11. Calibration and evaluation principles

Nominal gaze and accommodation demand are **soft fixation-mean anchors**, not independent framewise physiological measurements. The five nominal gazes are `[-10,-5,0,5,10]` degrees. Framewise states remain free within computational bounds, and actual accommodation is not forced to equal demand.

Initialization from a previous model is not an extra objective term; penalizing deviation from that model is. Declare model-derived priors and assess sensitivity including their removal. Temporal regularization can stabilize trajectories but cannot substitute for optical evidence.

Separate three levels of evidence:

**Fitted geometric consistency:** all-three observations agree with the calibrated state-dependent mapping.

**Held-out predictive consistency:** a state inferred without a P4 point predicts it on data excluded from calibration/model selection.

**Physiological accuracy:** states agree with independent gaze/accommodation references.

The first is weaker than the second; neither establishes the third. Agreement can retain shared biases. Extra geometry can still improve diagnostics even without measurable improvement in state precision.

Split by whole fixations/captures rather than random adjacent frames. Retrain matched controls within folds; the frozen all-data baseline is not an untouched control for data it trained on. Freeze preprocessing, covariance, model capacity, priors, and thresholds without outer-test information.

Evaluate predictions separately at `-10,-5,0,5,10` degrees and by accommodation-demand/capture. Examine central versus peripheral gaze, signed-gaze asymmetry, and systematic point/axis residuals before attributing peripheral disagreement to detection failures. Holding out an endpoint target is extrapolation relative to that fold's remaining anchors. Straight-ahead gaze is not necessarily on-axis illumination. Source-field claims require actual source/imaging geometry, not gaze labels alone.

Captures 5/6 are unlabeled transfer checks, not physiological references. Capture, demand, time, and drift may be confounded. Report branches, bounds, conditioning, state/P1-context extrapolation, numerical failures, missing/rejected points, and coverage. Do not erase expected deformation with geometric prefilters or report only rows surviving new model-dependent rejection.

## 12. Design conclusion and algorithm boundary

The shared-state construction is

$$
\boxed{
\ell_1=\sqrt{\mathcal A_1},\quad
\mathbf r_j=(\mathbf p_j-\mathbf c_1)/\ell_1,\quad
\widehat{\mathbf v}_j=\mathbf D(\theta,A)+\mathcal T(\theta,A)\mathbf r_j,
\quad\widehat\rho_4=|\det\mathcal T(\theta,A)|.
}
$$

Use all coordinate responses for the primary estimate; withhold each P4 point to test predictions. Preserve reproducible deformation rather than collapsing the pattern to one magnification or rejecting deviations from similarity. Additional shape responses can help separate shared states, but must earn that interpretation through noise-aware, grouped predictive evidence.

[ESTIMATOR_PLAN.md](ESTIMATOR_PLAN.md) specifies the 27-coefficient implementation reference and the required 37-coefficient curvature comparison, both with `theta_deg/10` scaling and new coefficients. Neither capacity is declared sufficient or superior in advance. The plan retains shared-input covariance, bounded variable-projection calibration with soft means, independent multistart inversion, and leakage-safe P4 holdouts. A genuine nine-component P1/P4 extension is separate.

No new calibration, changed detection output, or physiological result is asserted by this design document. See [CURRENT_STATUS.md](CURRENT_STATUS.md) for the implemented conditional estimator and exploratory evaluation; the optional joint nine-component extension remains deferred. Independent physiological references are required for physiological-accuracy claims.

## Optical state-scale symmetry and calibration convention

For either conditional response basis, positive gains $g,h$ give an exact
optical reparametrization: $\theta'=g\theta$ and $A'=h(1+A)-1$. Writing
$t=\theta/10$ and $L=\log(1+A)$, the inverse substitutions are
$t=t'/g$, $A=A'/h+(1/h-1)$ and $L=L'-\log h$. Each declared polynomial
basis is closed under these substitutions, so transformed coefficients satisfy
$F'(\theta',A';\beta',\mathbf r)=F(\theta,A;\beta,\mathbf r)$ for every P1 context.
The raw disagreements consequently transform as $G_\theta'=gG_\theta$ and
$G_A'=hG_A$. Optical residuals alone cannot establish these physical scales.

Soft fixation anchors, coefficient priors, and the fixed computational bounds
break or restrict this freedom; an already calibrated model cannot be freely
rescaled while claiming the same calibration objective. Descriptive positive
gains fitted on exactly matched training trajectories can diagnose scale motion,
with remaining trajectory residuals reported in degrees/diopters. They must never
be used to correct evaluation states after observing held-out results. Bounds and
the log domain also restrict admissible transformed states. The implementation
tests this algebra for both capacities; it is not physiological validation.

## Sources and historical boundary

Repository sources:

- [Fixation intervals](../data/fixations/fixation_intervals.json): current gaze/demand labels; the five-gaze protocol is also explicitly confirmed by the user.
- [joint_m2.py](../joint_m2.py): baseline measurements, validity, calibration setup, and independent inversion.
- [Frozen model](../models/quadratic_model.json): coefficients, historical `theta/15` convention, covariance, support, and saved diagnostics.
- [Saved capture summaries](../experiments/captures_5_6_direct/summary.json): prior results, not rerun here.

Optical and method references:

- [Wu et al. (2023), High-resolution eye-tracking via digital imaging of Purkinje reflections](https://pmc.ncbi.nlm.nih.gov/articles/PMC10166114/): distinct reflection paths and angular dependence in an unaccommodated optical model. It does not validate this accommodation/distortion estimator.
- [OpenCV, getAffineTransform](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html): exact affine construction from three correspondences.
- [NIST, Combining uncertainty components](https://physics.nist.gov/cuu/Uncertainty/combination.html): first-order propagation with covariance.
- [SciPy, least_squares](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html): bounded least-squares methods for the proposed algorithm.
- [scikit-learn, Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html): separation of training/model selection and grouped evaluation.

The grid-imaging interpretation adopts the user-stated physical premise; the illustrative field algebra, normalized geometry, state/nuisance accounting, and predictive covariance are derived here. They are hypotheses and mathematical relations, not measured coefficients or a complete aberration prescription. Earlier two-pair derivations remain in Git history (original blob `cace05e50b242052d3d8b7727a90c592b3dc2f06`); their separation ratios, old knot models, 15-degree target protocol, and numerical outcomes must not be substituted for this three-pair experiment.
