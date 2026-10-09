# Joint P1/P4 optical transformation for gaze and accommodation

**Status:** Proposed optical-model specification; not a claim of implementation, completed calibration, or physiological validation.  
**Repository / branch:** `rueijrwu/gaze_acc_joint_estimator` / `exp5_full`.  
**Revision date:** 2026-10-09.  
**Estimator snapshot:** `080d3a0386494737d3f7754c911103e8cf2aa293`.  
**Optical source:** `rueijrwu/distortion_tracking` at `1d2a0874c79f3f174b36a197c3fbbffd8a32fe60` [D1-D4].  
**Revision:** Make a gaze-conditioned reference-P1 scale fit the primary normalization. Describe the complete joint P1/P4 distortion transformation, not merely its normalization. Triangle areas and the P4/P1 area ratio are optional legacy diagnostics, not required observations or residuals.

## 0. Scientific objective and scope

**One calibrated optical model must explain the relative P1 and P4 patterns through one horizontal gaze and one accommodation value in each frame.** Relative centroid displacement supplies the dominant initial gaze signal. P1 supplies a reference pattern and nuisance magnification. P4 supplies accommodation-dependent magnification and deformation. The calibrated responses correct one another iteratively.

All observations are formed from **same-frame differences**. Absolute coordinates are input bookkeeping, not an eye-position observation. Common additive image translation cancels before fitting; a fixed camera position measured during zero-gaze calibration cannot serve that purpose for subsequent frames.

The main iteration is

$$
\boxed{\theta^{(t)}\ \longrightarrow\ \widehat g_{P1}(\theta^{(t)})
\ \longrightarrow\ A^{(t+1)}\ \longrightarrow\ \theta^{(t+1)}\ \longrightarrow\cdots.}
$$

During full calibration, shared optical parameters are also updated using all conditions. During application and cross-checks those parameters are frozen. Gaze and accommodation may change throughout fixation. No hard RMS accuracy requirement, constant-fixation state, or temporal-flatness penalty is introduced.

This revision replaces the previous requirement to divide by the square root of P1 triangle area. It does **not** remove any of the three measured P1/P4 correspondences or compress the P4 pattern to a scalar. Historical area-based and conditional27/37 models remain unchanged controls. Existing code and plans need explicit reconciliation before implementing this different observation/model adapter; no stopped experiment is restarted by this document.

## 1. Optical evidence, adopted approximations, and limits

The source separates baseline distortion, gaze-dependent deformation, accommodation dependence, and axial magnification [D1,D2].

| Component | Initial model | Qualification |
|---|---|---|
| P1 baseline | Fixed real, already distorted pattern | Do not apply its baseline barrel distortion again |
| P1 gaze response | Rotation-dependent transformation `K1(theta)` | Weak deformation is not exact invariance |
| P1 accommodation response | No direct A dependence | Modeling premise; the cited P1 sweep has no A dimension |
| P4 accommodation response | Baseline magnification `M(A)` and radial coefficient `kappa4(A)` | Both are shared functions to calibrate, not unrelated per-frame coefficients |
| P4 gaze response | `K4(theta,A)` acting on its accommodation-dependent baseline | Start with A-independent rotation coefficients; the composition still couples theta and A |
| P1 axial scale | `g1(Z) approximately 1+alpha1Z*Z` | Reference is the source's recorded Z=0 plane |
| P4 axial scale | `g4(A,Z) approximately 1+alpha4Z(A)*Z` | Separate Z study uses matched A/theta reference grids |
| Common-scale approximation | `g4/g1 approximately 1` | Similar fractional scale changes, not equal absolute P1/P4 image sizes |
| Relative centroid displacement | A separately calibrated law | Center-relative simulations do not measure the P4-P1 center separation |

The reported P1 slope is `alpha1Z=0.00295165270232 mm^-1`. P4 slopes are [D2,D3]:

| Accommodation (D) | alpha4Z(A) (mm^-1) |
|---:|---:|
| 0 | 0.00295232758 |
| 1 | 0.00295172708 |
| 2 | 0.00295112891 |
| 3 | 0.00295052435 |
| 4 | 0.00294991207 |

**Linear in Z is not linear in A.** These per-accommodation slopes span 0.0818504% of their mean. The P4 Z study covers 1,275 states and 11,475 records: five accommodations, five rotations (-10,-5,0,5,10 degrees), 51 Z planes, and nine fields. Every scale is referenced to the real grid at the **same A and theta at Z=0**. The constrained linear scale has worst-state coordinate RMSE about 0.249 micrometers and maximum Euclidean point error about 0.511 micrometers. Direct per-state scale fits have different, smaller residuals. Maximum reported scale variation with rotation is 0.00294912% for P4 and 0.00019129% in the P1 five-angle check. These are simulation results over specified domains, not hardware accuracy or exact invariance [D1-D3].

The P4 Z study has LEN SHA-256 `fb3937e6763f27e331faa10a4863533e994030cbc0ae74364b186ba0d06d474d`; the earlier P4 accommodation/rotation data have hash `5e0715c70016fbcec8a49956dd2ec824a50f5c4c52ca472b0225a2523db614ac`. They motivate the composition below but are not an already validated combined theta/A/Z calibration. Reconcile optical revisions and experimental geometry before transferring coefficients [D1,D4].

Z denotes the recorded absolute `Cornea_ENT_D` THI coordinate, not an experimentally measured eye-camera displacement. The P1 sequence baseline is 1 mm and the P4 z1 readback is 0 mm, but both reported ratios use the recorded Z=0 reference. Do not silently change that reference. P1 scale is sufficient for the initial estimator; no free physical Z state is required.

Common additive image translation cancels exactly. Physical eye translation may additionally change magnification, perspective, illumination field, or the two reflection paths differently. Only the modeled common component is removed. Near-equal pattern scales do not prove a Z-independent relative-centroid law. The simulation's approximately 1-micrometer simplification reference is not an RMS acceptance threshold for the recordings.

## 2. Measurements, state, and a translation-free observation vector

Let i denote a frame and j=1,2,3 a persistent source identity:

$$
\mathbf p_{ij}=P1_{ij},\qquad \mathbf q_{ij}=P4_{i,\pi(j)}.
$$

Read correspondence and validity flags from each payload; the recorded zero-based permutation is `pair_index=[2,1,0]`. Do not independently sort the two patterns. The source grid is not the gaze-target grid [G1,G2].

The states of interest are `x_i=(theta_i,A_i)`, in degrees and diopters. The nuisance scale `g_i>0` is the P1 scale; it is not a third physiological output. All 20 reviewed conditions from captures 1-4 participate in full calibration. Horizontal targets are -10,-5,0,5,10 degrees. Nominal vertical gaze is zero; measured image-y remains nonzero and informative. No vertical-gaze state is added. Demand labels and nominal targets are soft fixation-mean references, not instantaneous truth. The lowest stored demand is approximately 0.36036 D, not measured zero accommodation [G1].

Define same-frame centroids and two independent P1 edges:

$$
\mathbf c_{1i}=\tfrac13\sum_j\mathbf p_{ij},\quad
\mathbf c_{4i}=\tfrac13\sum_j\mathbf q_{ij},\qquad
\mathbf e_{1i}=\begin{bmatrix}\mathbf p_{i2}-\mathbf p_{i1}\\\mathbf p_{i3}-\mathbf p_{i1}\end{bmatrix}.
$$

The complete linear relative observation is

$$
\boxed{\mathbf y_i=\begin{bmatrix}
\mathbf e_{1i}\\
\mathbf q_{i1}-\mathbf c_{1i}\\
\mathbf q_{i2}-\mathbf c_{1i}\\
\mathbf q_{i3}-\mathbf c_{1i}
\end{bmatrix}\in\mathbb R^{10}.}
$$

Its linear map from the 12 native coordinates has rank 10 and removes precisely the two common-translation degrees of freedom. No triangle area is needed. The two P1 edges share a point and therefore generally have correlated errors.

**Preserve the relative centroid difference.** Centering P4 separately is allowed as a display or shape decomposition only when `c4-c1` is retained. Do not fit centroids, all edges, all relative points, and area as independent extra data.

Use the same fixed length unit for measured coordinates and reference templates. Reference-camera pixel units are convenient, making g dimensionless. If templates come in millimeters, convert measurements or templates with an explicit fixed camera calibration; do not hide pixel/mm conversion inside a purported dimensionless axial scale. Framewise corrected coordinates retain the reference length unit; they are not intrinsically dimensionless.

A shared optical-axis offset may define `vartheta=theta-theta_opt`. Even/odd optical symmetries refer to this alignment, not automatically to nominal fixation zero. There is no new zero angle per frame or capture.

## 3. Full local optical transformations

### 3.1 Rotation and keystone operator

For a local input `b=(b_x,b_y)`, use the restricted source-inspired transformation

$$
K_r(\vartheta,A;\mathbf b)=
\frac{1}{1+q_r(\vartheta,A)b_y}
\begin{bmatrix}s_{x,r}(\vartheta,A)b_x\\s_{y,r}(\vartheta,A)b_y\end{bmatrix},
\qquad r\in\{1,4\}.
$$

An initial minimal model is

$$
s_{x,r}=1+\alpha_r\vartheta^2,\qquad
s_{y,r}=1+\beta_r\vartheta^2,\qquad
q_r=\gamma_r\vartheta.
$$

K1 has no A dependence in the initial model. K4 may later use, for example, `q4=(gamma40+gamma41*a)*vartheta` or `sx4=1+(alpha40+alpha41*a)*vartheta^2`, where `a=(A-A_ref)/(1 D)`. Add only supported coupling terms. Always preserve `K4(0,A;b)=b`, so accommodation-dependent baseline scale is not duplicated in a free zero-gaze rotation scale.

If optical axes differ from camera axes, conjugate K by one fixed calibrated orthonormal alignment `Q_r`: `Q_r K_r(vartheta,A;Q_r^T b)`. This is a coordinate convention, not a per-frame rotation that erases measured deformation. In length units L, q has units L^-1, alpha/beta degree^-2, and gamma L^-1 degree^-1. Require positive scales and denominators bounded away from zero on the actual source-field/state domain.

This is a restricted spatial hypothesis, not a universal aberration model. The source reports spatial residuals it cannot express, such as a nonzero output y from an input with `b_y=0`. Raising only the polynomial order of its angle coefficients cannot create that missing spatial component [D1].

### 3.2 P1: fixed distorted baseline, gaze deformation, axial scale

Let `b1_j` be the real P1 reference template at optical zero and reference magnification, expressed relative to its local optical origin. It already contains reference barrel distortion. Define

$$
\boxed{\mathbf F_{1j}(\theta,A)=K_1(\vartheta;\mathbf b_{1j}),\qquad
\partial_A\mathbf F_{1j}=0\quad\text{in the initial model}.}
$$

When a matched paraxial P1 template and an adequate radial description are available, the fixed baseline can alternatively be written

$$
\mathbf b_{1j}=\mathcal R_{1,ref}(\mathbf u_{1j})
=[1+\kappa_{1,ref}\|\mathbf u_{1j}\|^2]\mathbf u_{1j},\qquad
\mathbf P_{1,j}^{local}=g_1(Z)K_1(\vartheta;\mathcal R_{1,ref}(\mathbf u_{1j})).
$$

This is an alternative representation of the baseline, not another correction applied to the measured b1. The empirical real template is the default and can retain baseline structure that a one-term radial approximation misses. No free per-frame P1 barrel coefficient is added.

The full local P1 pattern before eliminating translation is `g1(Z)*F1_j(theta)`. The transformation order is baseline -> gaze deformation -> external axial scale, or `S(g1)H1`, not `H1S(g1)`. Scaling inside the projective denominator is a different transformation. P1 accommodation independence is an explicit approximation, not evidence missing from the joint model.

### 3.3 P4: accommodation magnification, radial distortion, gaze deformation, axial scale

Let `u4_j` be a known or separately constrained paraxial reference template about the local radial center. Define

$$
\mathbf z_j(A)=M(A)\mathbf u_{4j},\qquad M(A_{ref})=1,
$$

$$
\mathcal R_A(\mathbf u)=\left[1+\kappa_4(A)\|M(A)\mathbf u\|^2\right]M(A)\mathbf u,
\qquad \mathbf B_{4j}(A)=\mathcal R_A(\mathbf u_{4j}),
$$

$$
\boxed{\mathbf F_{4j}(\theta,A)
=K_4\!\left(\vartheta,A;\mathbf B_{4j}(A)\right).}
$$

Thus the complete local P4 pattern is

$$
\boxed{\mathbf P_{4,j}^{local}(\theta,A,Z)
=g_4(A,Z)\,K_4\!\left(\vartheta,A;\mathcal R_A(\mathbf u_{4j})\right).}
$$

M(A) is the accommodation-dependent P4 magnification; g4 is axial nuisance magnification. They must not be conflated. The radial coefficient kappa4 has units L^-2 and its radius is evaluated before radial distortion. Check both `1+kappa*r^2` and the radial derivative `1+3*kappa*r^2` when requiring a one-to-one cubic radial mapping.

Even when the K4 coefficients are independent of A, `B4_y(A)` enters its denominator. Consequently gaze and accommodation are coupled by the composition; separate coefficient functions do not imply additive or separable image responses. The full accommodation derivative is

$$
\partial_A\mathbf F_4=(\partial_{\mathbf b}K_4)\,\partial_A\mathbf B_4
+\left.\partial_A K_4\right|_{\mathbf b},
$$

where the second term is zero only for the A-independent K4 candidate. Both terms, when present, belong in estimation and observability calculations.

Begin with constant/linear baseline functions, such as `M=1+m1*a` and `kappa4=kappa_ref+kA*a`, where identifiable and positive on the domain. Log, shifted-power, or literal-power shapes may parameterize these specific mechanisms later. Each shape parameter is shared across the full calibration, never independently fitted per frame or P4 subset. Fit new parameters and frame states for every candidate.

### 3.4 Real reference templates and distortion centers

A measured zero-gaze P4 template is already distorted. Do not insert it as the paraxial u4 and apply the same radial distortion a second time. If the reference radial map is known and invertible, use

$$
\mathbf B_4(A)=\mathcal R_A\!\left(\mathcal R_{A_{ref}}^{-1}(\mathbf b_4^{ref})\right).
$$

Otherwise use a declared relative baseline-deformation model, or constrain the paraxial geometry and reference distortion with matched optical data. Its coefficients are not automatically absolute barrel coefficients. Neither empirical zero-gaze centroids nor the P1 scale fit measures the distortion center [D1,D2].

## 4. Complete joint transformation in the actual relative reference

Define model-derived means, centered optical shapes, and P1 reference edges:

$$
\overline{\mathbf F}_r=\tfrac13\sum_j\mathbf F_{rj},\qquad
\mathbf S_{rj}=\mathbf F_{rj}-\overline{\mathbf F}_r,\qquad
\mathbf a_1(\theta)=\begin{bmatrix}\mathbf F_{12}-\mathbf F_{11}\\\mathbf F_{13}-\mathbf F_{11}\end{bmatrix}.
$$

Let `g=g1` and `eta=g4/g1`. A relative-origin displacement `delta_g(theta,A,g)` can connect the two local optical origins. For deriving cancellation only, introduce an arbitrary common origin t:

$$
\mathbf p_j=\mathbf t+g\mathbf F_{1j},\qquad
\mathbf q_j=\mathbf t+g\boldsymbol\delta_g+g\eta\mathbf F_{4j}.
$$

These auxiliary absolute equations do not add an absolute-position observation. Subtracting the same-frame P1 centroid gives

$$
\boxed{\widehat{\mathbf p_j-\mathbf c_1}=g\mathbf S_{1j}(\theta),}
$$

$$
\boxed{\widehat{\mathbf q_j-\mathbf c_1}
=g\left[\boldsymbol\delta_g+\eta\mathbf F_{4j}(\theta,A)-\overline{\mathbf F}_1(\theta)\right].}
$$

Define the relative centroid law in reference-scale units:

$$
\boxed{\mathbf h_g(\theta,A,g)
=\boldsymbol\delta_g+\eta\overline{\mathbf F}_4-\overline{\mathbf F}_1.}
$$

Calibrate either h_g directly or delta_g and derive h_g. **Do not fit independent free laws for both.** The simulation's center-relative transformations alone cannot supply their separation. In the initial common-scale mode, set eta=1 and use `h_g=h(theta,A)`; its absence of residual axial dependence is a separate hypothesis to inspect.

The complete joint observation model is therefore

$$
\boxed{\widehat{\mathbf y}(\theta,A,g;\Psi)
=\begin{bmatrix}
g\mathbf a_1(\theta)\\
g[\mathbf h_g+\eta\mathbf S_{41}(\theta,A)]\\
g[\mathbf h_g+\eta\mathbf S_{42}(\theta,A)]\\
g[\mathbf h_g+\eta\mathbf S_{43}(\theta,A)]
\end{bmatrix}.}
$$

Psi contains the shared reference geometry, optical transformations, alignment, displacement, and response-law parameters. It is not a free per-frame deformation map. P1 and P4 are both predicted; P1 is not merely treated as exact external context. All translations common to both images have canceled.

The corresponding scale-corrected transformations, with no triangle denominator, are

$$
\boxed{\widehat{\widetilde{\mathbf p}}_j=\mathbf S_{1j}(\theta),\qquad
\widehat{\widetilde{\mathbf q}}_j=\mathbf h_g(\theta,A,g)+\eta(A,g)\mathbf S_{4j}(\theta,A).}
$$

In expanded optical form these are

$$
\widehat{\widetilde{\mathbf p}}_j
=K_1(\vartheta;\mathbf b_{1j})-\tfrac13\sum_l K_1(\vartheta;\mathbf b_{1l}),
$$

$$
\widehat{\widetilde{\mathbf q}}_j
=\boldsymbol\delta_g
+\eta\,K_4(\vartheta,A;\mathcal R_A(\mathbf u_{4j}))
-\tfrac13\sum_l K_1(\vartheta;\mathbf b_{1l}).
$$

These equations explicitly connect **gaze, accommodation, both reflection transformations, and the relative reference**. P1 has no direct A dependence under the initial premise; the complete joint observation nevertheless depends on both states through its distinct blocks. Common scale is not the accommodation-specific M(A).

The retained gaze-sensitive and paired-displacement predictions are

$$
\widehat{(\mathbf c_4-\mathbf c_1)/g}=\mathbf h_g,\qquad
\widehat{(\mathbf q_j-\mathbf p_j)/g}
=\mathbf h_g+\eta\mathbf S_{4j}-\mathbf S_{1j}.
$$

An initial centroid law is `h_x(theta,A)=b(A)+s(A)*theta`: accommodation can change offset and gaze gain. Retain a calibrated h_y when needed, despite nominal vertical gaze being zero. This law is relative displacement, not an absolute distortion-center trajectory.

## 5. Primary normalization: fit scale to reference P1 edges

At each trial gaze, compare measured P1 edges with the calibrated P1 edges **at that gaze**:

$$
\mathbf e_{1i}\approx g_i\mathbf a_1(\theta_i).
$$

For a declared positive-definite P1-edge covariance R11, fixed during the trial fit, let W1=R11^-1. The unconstrained scalar least-squares solution is

$$
\boxed{\widehat g_{P1,i}(\theta)=
\frac{\mathbf a_1(\theta)^{\mathsf T}W_1\mathbf e_{1i}}
{\mathbf a_1(\theta)^{\mathsf T}W_1\mathbf a_1(\theta)}.}
$$

With identity weighting this is a dot-product ratio. There is no determinant, triangle area, P4 area, or measured distortion center in the calculation. The reference must have nonzero weighted edge energy, and the scale must be finite and positive. Flag an invalid scale or use an explicitly declared constrained fit; do not silently clip and claim successful calibration.

Changing the choice of edge origin should leave generalized least-squares results unchanged when the entire edge covariance is transformed consistently. The reference template/alignment is calibrated from multiple frames or matched optics, not a noisy single-frame template assumed exact. Template uncertainty is shared calibration uncertainty.

One positive scalar is fitted. Do not replace it with an unconstrained affine or per-point correction that absorbs the gaze deformation or P4 accommodation signal. A collinear but noncollapsed P1 configuration can still determine this scalar; useful joint gaze/distortion identifiability remains a separate question. Removing area division does not remove all geometric degeneracies.

Define the corrected relative coordinates

$$
\boxed{\widetilde{\mathbf p}_{ij}(\theta)
=\frac{\mathbf p_{ij}-\mathbf c_{1i}}{\widehat g_{P1,i}(\theta)},\qquad
\widetilde{\mathbf q}_{ij}(\theta)
=\frac{\mathbf q_{ij}-\mathbf c_{1i}}{\widehat g_{P1,i}(\theta)}.}
$$

They have reference-image length units. Both common additive translation and estimated common magnification are removed, but P4-specific M(A), kappa4(A), relative centroid displacement, and residual eta remain.

The corrected centroid difference and centered P4 shape are

$$
\widetilde{\mathbf d}_i=(\mathbf c_{4i}-\mathbf c_{1i})/\widehat g_i,\qquad
\widetilde{\mathbf s}_{4ij}=(\mathbf q_{ij}-\mathbf c_{4i})/\widehat g_i.
$$

The latter is a full-frame decomposition, not a permitted observation when one P4 is omitted. Never normalize P4 by its own independently fitted scale: that could remove M(A). Never divide an already scale-corrected observation by g again.

### 5.1 Scale depends on trial gaze

P1 need not have constant shape or size with gaze. Comparing every frame only with an undeformed zero-gaze template would mix gaze and axial scale. Reevaluate a1(theta) and g at trial gaze, not just once before fitting.

For fixed W1 and observations, the scale derivative is

$$
\partial_\theta\widehat g=
\frac{\mathbf a_{1,\theta}^{\mathsf T}W_1\mathbf e_1
-2\widehat g\,\mathbf a_{1,\theta}^{\mathsf T}W_1\mathbf a_1}
{\mathbf a_1^{\mathsf T}W_1\mathbf a_1}.
$$

Include this derivative, and analogous global-parameter derivatives, in the complete residual. The corrected observations themselves depend on trial gaze; treating them as permanently fixed changes the objective. If weights vary, their derivatives require separate treatment rather than reuse of this fixed-weight expression.

### 5.2 Equivalence to legacy area normalization only under its assumptions

For a nondegenerate noiseless common-scale pattern, let `ell1=sqrt(area(P1))` and `L1(theta)=sqrt(area(F1(theta)))`. Then

$$
\widehat g_{P1}=g=\ell_1/L_1(\theta),\qquad
(\mathbf q-\mathbf c_1)/g=L_1(\theta)(\mathbf q-\mathbf c_1)/\ell_1.
$$

Thus the two representations carry the same ideal information. Under noise or shape mismatch, the weighted reference-scale fit and area estimator differ; neither is automatically more accurate. The scale approach uses all selected P1 edge components with explicit covariance and avoids dependence on a small area denominator.

`area(P4)/area(P1)` and `sqrt(area(P1))` may be saved to compare historical reports. They are **not required estimator channels, normalization factors, independent residuals, or default acceptance gates**. The three point identities and their full coordinate responses remain required.

## 6. Differential P1/P4 axial scale without a free Z state

Under matched-reference linear models,

$$
\eta(A,Z)=\frac{g_4(A,Z)}{g_1(Z)}
\approx\frac{1+\alpha_{4Z}(A)Z}{1+\alpha_{1Z}Z},\qquad
\eta-1=\frac{[\alpha_{4Z}(A)-\alpha_{1Z}]Z}{1+\alpha_{1Z}Z}.
$$

Linear numerator and denominator do not make eta linear or constant. Near-equal reported slopes motivate eta=1 initially; slope differences alone do not bound approximation residuals, lens-revision differences, or noise.

Where matched calibration supports the slope ratio and reference plane, set `c(A)=alpha4Z(A)/alpha1Z`. Eliminate Z using `g=g1`:

$$
\boxed{g_4=1+c(A)(g-1),\qquad
\eta(A,g)=c(A)+\frac{1-c(A)}{g}.}
$$

This uses the P1-derived scale and does not add a free physical Z state. The source tabulates c only at five accommodations; any continuous interpolation or fitted c(A) must be declared and calibrated, not invented as exact physiology. Both scale factors must remain positive within the used domain.

For differentiable c,

$$
\partial_g\eta=(c-1)/g^2,\qquad
\left.\partial_A\eta\right|_g=c'(A)(1-1/g),\qquad
\partial_\theta\eta=\partial_g\eta\,\partial_\theta\widehat g
$$

when c has no explicit gaze dependence. The shared-scale limit c=1 yields eta=1. If the experimental reference differs from the source's Z=0 plane, renormalize both scale functions consistently; the above intercepts cannot simply be reused.

Use the extended relative prediction `q-c1 = g[h_g+eta*S4]`. **Do not multiply the entire `h+S4` by eta.** Center-relative Z sweeps constrain pattern scaling, not the relative optical-origin motion. Either learn h_g directly or delta_g and derive h_g; neither the full relative observation nor its centroid can be corrected by blindly dividing by eta. No independent framewise P4 scale is allowed to absorb accommodation.

## 7. Reference centers, gauges, and identifiable optical mechanisms

### 7.1 Origins and centers

Near-zero-gaze P1/P4 centroids can initialize separate reference origins. They do not establish exact distortion centers. Under radial distortion,

$$
\overline{\mathbf B}_4=M\overline{\mathbf u}_4
+\kappa_4M^3\overline{\|\mathbf u_4\|^2\mathbf u_4},
$$

so even a centered paraxial template need not have a centered distorted centroid. Use multiple near-zero-gaze frames, preserve source identities, and allow only identifiable shared center/alignment offsets. P1 symmetry is supporting alignment evidence, not a reason to choose one noisy frame as exact optical zero or assume an equilateral source triangle.

Center offsets belong inside the radial/projective template construction. A free additive output-center shift is absorbed by the relative displacement law and is not independently identifiable. Do not fit both redundantly. Inverse keystone applied about the observed centroid is not generally equivalent to inverting the true center-relative map; centering and a projective transformation need not commute.

### 7.2 Fix the scale and state gauges

Fix the reference length unit, source/template geometry convention, M(A_ref)=1, and optical zero. Constrain the isotropic gaze dependence of K1 with matched optics or an explicit reference-scale calibration: arbitrary K1 gaze scaling and independent g_i can compensate. P1-reference scale is relative to this convention, not an independent measurement of distance.

Do not jointly free the paraxial template, radial center, reference radial coefficient, M(A), and per-frame scales without a rank/gauge analysis. Nominal demand means help anchor the accommodation coordinate but do not establish instantaneous physiological diopters. Changing a latent A coordinate may be compensated by a different response law; compare state-scale sensitivity, not just image error.

### 7.3 Magnification versus radial distortion

If the three reference radii are equal to R,

$$
\mathbf B_{4j}=M(A)[1+\kappa_4(A)M(A)^2R^2]\mathbf u_{4j}.
$$

Only the combined effective scale is determined by that centered spatial pattern, not separate M and kappa4. Different sampled radii, a constrained optical baseline, or additional source evidence are needed for separation. With free centers or further nuisance terms, different radii alone do not guarantee identifiability. A useful accommodation estimate may rely on effective magnification even when an absolute barrel coefficient is unresolved.

A free affine map through three P4 points can fit those points exactly. It is not a cross-check or evidence of a physiological distortion law. All distortion functions here are globally shared responses to theta and A, not freely fitted maps for individual frames.

## 8. Cost function and scale-uncertainty policy

### 8.1 Fixed relative-data metric with P1-only scale

Let `Sigma_P` be the native 12-coordinate covariance and L the fixed relative-coordinate map from Section 2. Then `R_y=L Sigma_P L^T` includes correlated P1 edges and shared P1-centroid noise in all P4 blocks. It is not a diagonal collection of independent point errors.

The primary P1-scale mode defines

$$
\mathbf m_i(\theta,A;\Psi)=\widehat{\mathbf y}
(\theta,A,\widehat g_{P1,i}(\theta;\Psi);\Psi),
$$

$$
\boxed{Q_i=(\mathbf y_i-\mathbf m_i)^{\mathsf T}R_{y,i}^{-1}
(\mathbf y_i-\mathbf m_i).}
$$

R_y and the P1-edge weights are frozen under one declared noise policy for the first comparison. This is a **P1-scale-constrained estimating objective**, not the unconstrained maximum-likelihood profile of g using both reflections. It keeps scale estimation P1-only while assessing all P1 shape and P4 relative-position residuals together. The data-derived g makes the composite residual's covariance different from R_y; R_y here is the declared raw-data weighting metric, not a claim of ten independent postfit residuals.

Do not instead minimize unweighted corrected-coordinate errors: dividing by a large fitted g could reduce that objective without improving raw-coordinate predictions. At fixed g, equivalent corrected weighting uses `R_y/g^2`; when g is data-derived, propagate its dependence for uncertainty and do not double-count it as an independent measurement.

### 8.2 Optional joint-scale profile is a distinct estimator

For common-scale mode with h independent of g, write `yhat=g*f(theta,A)`. If a later explicit comparison allows the full data to estimate the **same single shared** scale, the unconstrained positive solution is

$$
g_{joint}(\theta,A)=
\frac{\mathbf f^{\mathsf T}R_y^{-1}\mathbf y}
{\mathbf f^{\mathsf T}R_y^{-1}\mathbf f},
$$

when this value is positive. This profiles the fixed-covariance joint relative least-squares objective. It is not generally equal to g_P1 under noise and must not silently replace the P1-reference policy. P4 still cannot receive a separate unconstrained scale. For eta(A,g) or g-dependent h_g, minimize the actual declared scalar problem; the common-scale formula is not automatically valid.

### 8.3 Corrected-coordinate rank and uncertainty

At fixed trial theta and fixed W1, `ytilde=y/g_P1` has the constraint

$$
\mathbf a_1^{\mathsf T}W_1\widetilde{\mathbf e}_1
=\mathbf a_1^{\mathsf T}W_1\mathbf a_1.
$$

Its local feature Jacobian has rank nine for a valid scale. This replaces the old area-normalized constraint; the P1 area is not fixed to one. The ten entries of ytilde are not ten independent normalized observations. Storing all centered P1/P4 entries adds further centering redundancy. Use an independent local chart or the correct covariance range for uncertainty, not diagonal noise or a covariance ridge that invents a missing information direction. Retaining g as bookkeeping does not make it an independent observation.

For uncertainty in the P1-scale mode, differentiate the complete residual `epsilon(P;x,Psi)`, including the scale fit and shared P1 terms: `R_e approximately J_e,P Sigma_P J_e,P^T`. The reference template and global calibration add shared parameter uncertainty when not conditioned on. Overlapping two-P4 inversions are correlated. Near bounds, weak geometry, or multiple branches, local Gaussian precision can be misleading; report profiles or prediction sets.

A legacy covariance for `v-D-T*r` is not this new joint residual. If a future likelihood uses candidate/state-dependent covariance, include its normalization and validate the model; do not let covariance inflation conceal optical mismatch. The current revision does not declare such a likelihood implemented or calibrated.

### 8.4 Full-calibration objective

With K exposures and N_k valid calibration frames in exposure k, use

$$
\boxed{J=\frac1{2K}\sum_k\frac1{N_k}\sum_{i\in k}Q_i
+\frac1{2K}\sum_k\left[
\frac{(\bar\theta_k-\theta_k^{nom})^2}{s_\theta^2}
+\frac{(\bar A_k-A_k^{demand})^2}{s_A^2}\right]
+J_{global\ prior}.}
$$

Inherited starting mean-anchor scales of 0.10 degree and 0.25 D are finite penalty weights, not acceptable error, allowed motion, or physiological uncertainty. There is no penalty forcing within-fixation constancy. Regularization and reference gauges must be shared consistently across model comparisons. Do not add E, G_theta, and G_A in incompatible units as arbitrary fitting penalties.

The optimizer can reuse bounded least-squares and variable-projection machinery for genuinely linear parameter blocks. The composed radial/projective model is not the previous linear 27-column basis: new forward functions, derivatives, residual/covariance interfaces, and CPU/GPU parity checks are required. Reuse numerical algorithms, not incompatible coefficients or an asserted unchanged design matrix.

## 9. Full calibration and iterative correction

### 9.1 Initialize, then use every calibration condition

Use the lowest-demand condition to initialize a centroid-to-gaze relationship; do not label it measured A=0. Use multiple near-zero-gaze frames to form the relative templates and set the reference scale. P1 shape/symmetry may refine a shared alignment subject to the gauge restrictions above.

Initialize g from the P1 reference at nominal/initial gaze. Fit the five fixation-mean values of `(c4_x-c1_x)/g` to a linear gaze mapping. Assign each frame its own initial gaze, not the nominal fixation value. Initialize A from demand only as a starting value. Initialize M(A) and identifiable radial/effective-deformation terms from near-zero-gaze frames across demands, then refine them on all frames after gaze correction. This staged initialization does not withhold any condition from full calibration.

### 9.2 Per-frame alternating updates

With global parameters fixed, define Q_i by Section 8.1, including g_P1 at every trial gaze. The iteration is

$$
\widehat g_i^{(t)}=g_{P1,i}(\theta_i^{(t)}),\qquad
A_i^{(t+1)}=\arg\min_A Q_i(\theta_i^{(t)},A),
$$

$$
\theta_i^{(t+1)}=\arg\min_\theta Q_i(\theta,A_i^{(t+1)}),\qquad
\widehat g_i^{(t+1)}=g_{P1,i}(\theta_i^{(t+1)}).
$$

The corrected pattern is `(q-c1)/g`, and the prediction is the full `h_g+eta*S4`; accommodation is not inferred from a required area ratio. If differential correction is enabled, recompute eta(A,g) and h_g at every trial state.

For common-scale `h_x=b(A)+s(A)*theta`, a useful gaze proposal is

$$
\theta_{proposal}=
\frac{(c_{4,x}-c_{1,x})/\widehat g^{(t)}-b(A^{(t+1)})}{s(A^{(t+1)})}.
$$

Accept or refine this proposal with the complete Q, reevaluating g at trial gaze. Do not freeze corrected observations, invert a projective map about an assumed centroid-center, or use a different ad hoc cost for each block. During calibration, conditional updates must also include their contribution to the soft fixation-mean terms of J. During application there are no nominal-frame anchors.

Use damping or line search and joint two-state refinement when appropriate. Weak shape-gaze sensitivity supports this initialization strategy but does not guarantee convergence or uniqueness. A fixed number of alternations, a flat trajectory, or low RMS is not numerical certification.

### 9.3 Shared optical updates

After updating frame states, update the permitted shared baseline, displacement, M(A), kappa4(A), rotation, and alignment parameters using all valid full-period rows in all 20 reviewed conditions. Repeat state/global updates under the same J, holding reference gauges fixed. Do not estimate unrelated per-frame distortion functions or center trajectories.

Each alternative optical response law receives a fresh full calibration. No whole-condition/capture holdout is required for this internal-agreement task. Balanced rehearsal populations can test numerical integration, but must not be reported as full-period calibration. Preserve numerical failures, actual valid/scheduled counts, starts, source hashes, and convergence status.

## 10. Cross-agreement of the entire joint estimator

Freeze every global optical parameter, center/template convention, weighting policy, and response law after calibration. For each frame omit P4 j in turn, retain all three P1 points and the other two P4 points, and rerun the complete iterative state/scale estimator.

The selected relative vector has four P1-edge components plus four retained P4 components. Select its covariance marginal **before** whitening. Compute the P1 scale at that subset's trial gaze; do not copy the all-three state or its scale-dependent correction. No measured all-three P4 centroid, area, affine map, unmasked warm start, or held-point-derived weight may enter initialization or inversion. A mean of all three **predicted** F4 points is permitted because it is a function of the trial state and fixed calibration.

The omitted-point reconstruction is

$$
\boxed{\widehat{\mathbf q}_{ij\mid-j}=\mathbf c_{1i}
+\widehat g_{i,-j}\left[
\mathbf h_g(\widehat x_{i,-j},\widehat g_{i,-j})
+\eta(\widehat A_{i,-j},\widehat g_{i,-j})\mathbf S_{4j}(\widehat x_{i,-j})\right].}
$$

Restoring c1 only restores the arbitrary image origin for comparison. Score the relative error in original camera-coordinate units:

$$
\mathbf e_{ij}^{cross}=(\mathbf q_{ij}-\mathbf c_{1i})
-(\widehat{\mathbf q}_{ij\mid-j}-\mathbf c_{1i}).
$$

If calculations use millimeters, convert both terms to the same native-pixel convention before reporting pixel error. Do not rank candidates by an error divided by their own fitted g; a change in scale units must not manufacture an improvement.

For complete three-way checks,

$$
E_i^2=\tfrac13\sum_j\|\mathbf e_{ij}^{cross}\|^2,\qquad
G_{A,i}^2=\tfrac13\sum_{j<k}(\widehat A_{i,-j}-\widehat A_{i,-k})^2,
$$

with analogous G_theta. Compare simultaneous subset estimates, not previous frames or nominal demand. Average squared quantities within each exposure and equally across exposures. RMS is a readable summary, not a hard threshold. Vector-error RMS differs from scalar-coordinate RMSE by sqrt(2) for the same 2D population.

Compare exact matching frames/slots, and show raw signed axes, each P4, worst point, tails, bounds, ambiguity, conditioning, and coverage. Missing or failed outcomes must not disappear from denominators. Full-fit residual, internal cross-reconstruction, nominal-mean discrepancy, and temporal variation remain different outputs.

Calibration used these same observations, so this is internal consistency, not independent physiological validation. Held-point noninterference means changing only the omitted P4 **after fixing calibration** cannot change the subset states, scale, branches, or predictions. The requirement does not assert invariance if the entire model is recalibrated on changed data.

## 11. Selecting a useful accommodation model

The goal is a defensible accommodation response, not an extra fraction of a pixel. Use optical cross-prediction and same-frame state agreement together with parameter identifiability, response curves, gauge sensitivity, and failure patterns. Log is a reference, not a presumed winner. No arbitrary mixed-unit sum or universal RMS cutoff defines success.

For fixed calibration, quantify accommodation sensitivity not reproduced by gaze or permitted scale/nuisance variation. In regular nonredundant whitened coordinates with state Jacobian J, the two-state Schur complement is

$$
\mathcal I_{A\mid\theta}=F_{AA}-F_{A\theta}^2/F_{\theta\theta},\qquad F=J^{\mathsf T}R^{-1}J.
$$

Use the full composed response derivatives, including the P1 scale fit, optical centering, M/kappa, projective denominator, and any eta/h_g dependence. This expression assumes an appropriate nonsingular error representation. If g is jointly fitted, project its response out as an additional nuisance. In the P1 plug-in mode, propagate the composite estimator/noise dependence rather than presenting R_y as its postfit residual covariance. Rank, branches, and empirical perturbation checks remain necessary.

Information per diopter and raw G_A can change when latent accommodation is rescaled. Small G_A can arise from compression or shared clipping. Test common anchor/prior/bound sensitivities, and report whether only effective magnification is determined. A missing spatial deformation or unmatched axial correction must not be disguised by forcing a large accommodation exponent. Detector bias, uncertain center alignment, and capture/demand confounding remain alternative causes of residuals. Physiological accuracy needs an independent reference or further justified optical constraints.

## 12. Required checks and compatibility

Before claiming implementation, verify the following properties with synthetic and recorded-data tests:

- Arbitrary per-frame common translations leave all relative inputs, inference, and cross-errors unchanged.
- The weighted reference-P1 scale recovers a known positive scale, is translation-free, and uses no P4 measurements at fixed trial gaze.
- A change of edge origin with correctly transformed covariance gives the same scale; the analytic gaze derivative agrees with finite differences.
- Positive common scaling cancels in corrected coordinates, while P4-only accommodation magnification and differential eta remain.
- Both full P1 and full P4 transformations are evaluated in the stated order; centering, rotation, radial response, and units are consistent.
- An empirical distorted baseline is not radially corrected twice; equal-radius scale/radial degeneracy and center/reference gauges are recognized.
- P1-only and joint-scale profiling agree in exact noiseless common-scale data but need not agree under noise; policy metadata distinguishes them.
- Relative-coordinate rank is 10 and fixed-trial scale-normalized rank is nine without an area constraint; singular corrected covariance is not treated as independent noise.
- Differential linear-scale elimination of Z and its derivatives are correct, with equal slopes recovering eta=1; h_g is not blindly multiplied by eta.
- Withholding a P4 changes neither P1 scale inputs nor subset initialization through hidden all-three measurements; the whole iterative solve passes noninterference with calibration fixed.
- Dynamic within-fixation trajectories remain free; zero-mean state changes leave mean anchors unchanged, though optical residuals may change.
- Failed or ambiguous fits retain scheduled identities; comparisons use exact common populations and original-coordinate error units.

These are implementation requirements, not claims of a completed repository test suite or new real-data calibration. The area-normalized baseline remains available for controlled comparison, with its coefficients and conventions untouched. Only this theoretical specification changes in the present revision.

## 13. Sources and implementation boundary

The relative cancellation, reference-scale estimator, and composed joint equations are the proposed synthesis and algebraic consequences of the stated assumptions. Optical simulation evidence supports model structure and near-common scaling, not exact cancellation in the experimental setup or independent accommodation accuracy.

- **[D1]** [distortion_tracking Theory.md](https://github.com/rueijrwu/distortion_tracking/blob/1d2a0874c79f3f174b36a197c3fbbffd8a32fe60/Theory.md): local optical composition, reference conventions, P1/P4 linear Z evidence, and spatial-model limits.
- **[D2]** [distortion_tracking Summary.md](https://github.com/rueijrwu/distortion_tracking/blob/1d2a0874c79f3f174b36a197c3fbbffd8a32fe60/Summary.md): fitted scale results, tested domains, and lens-revision qualifications.
- **[D3]** [P4 Z magnification report](https://github.com/rueijrwu/distortion_tracking/blob/1d2a0874c79f3f174b36a197c3fbbffd8a32fe60/data/p4_z_magnification/p4_z_magnification.md): matched same-A/theta grids, slope table, direct versus linear residuals.
- **[D4]** [P4 Z metadata](https://github.com/rueijrwu/distortion_tracking/blob/1d2a0874c79f3f174b36a197c3fbbffd8a32fe60/data/p4_z_magnification/metadata.json): collection and LEN provenance, not hardware calibration.
- **[G1]** [Reviewed fixation metadata](../data/fixations/fixation_intervals.json): gaze/demand labels, correspondence to captures, reviewed periods, validity, timestamps, and reliability flags.
- **[G2]** [Superseded theory snapshot](https://github.com/rueijrwu/gaze_acc_joint_estimator/blob/080d3a0386494737d3f7754c911103e8cf2aa293/docs/Theory.md): historical area-normalized formulation, retained as a comparison rather than the primary normalization.

Reconcile [ESTIMATOR_PLAN.md](ESTIMATOR_PLAN.md) and the accommodation plans with this model before implementation. Preserve [the frozen estimator](../models/quadratic_model.json), its theta/15 convention, detection data, current code, test fixtures, and historical results. Captures 5/6 remain outside this calibration workstream. [CURRENT_STATUS.md](CURRENT_STATUS.md) and [EXPERIMENTS.md](EXPERIMENTS.md) describe execution evidence; this theory does not silently promote a model or restart an experiment.
