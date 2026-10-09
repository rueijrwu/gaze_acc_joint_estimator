# Distortion-based joint gaze and accommodation estimator

**Branch:** `exp5_distortion_model` in `rueijrwu/gaze_acc_joint_estimator`.  
**Status:** Authoritative design for this branch; implementation and new real-data results are not claimed.  
**Date:** 2026-10-09.  
**Starting tree:** `bc3e75b595bcf28486d23881354b3fa3c1daacde` (data and documentation only).  
**Optical evidence:** `rueijrwu/distortion_tracking` at `1d2a0874c79f3f174b36a197c3fbbffd8a32fe60` [D1–D3].  
**Companion:** [ESTIMATOR_PLAN.md](ESTIMATOR_PLAN.md).

## 1. Goal and non-negotiable distinctions

Find a useful **accommodation response model**, not just another exponent that lowers image RMS. One globally calibrated optical model must explain the three P1/P4 pairs with one horizontal gaze and one accommodation value **per frame**. The main estimation cycle is

$$
\boxed{\theta\ \longrightarrow\ g_{P1}(\theta)\ \longrightarrow\ A
\ \longrightarrow\ \theta\ \longrightarrow\cdots.}
$$

Relative P4–P1 displacement supplies the strong initial gaze signal. P1 supplies reference geometry and nuisance magnification. P4's remaining magnification and deformation supply accommodation information. Gaze and accommodation correct each other; neither is assumed perfectly separable.

All observations are **same-frame differences**. We do not estimate absolute eye position. Common additive image translation cancels exactly; physical translation that also changes scale, field sampling, or the two reflection paths differently requires the corresponding model, not merely a subtraction.

The primary normalization is a **positive scalar fitted to reference P1 edges at the trial gaze**. Neither P1 triangle area nor the P4/P1 area ratio is required. Keep all three point identities and full x/y coordinates. Do not replace them with a free per-frame affine map or normalize P4 by its own size, which would remove accommodation information.

All reviewed calibration conditions participate in a fresh full fit for each candidate optical law. Temporary omission of one P4 is an internal same-frame cross-check **after** fitting, not exclusion of a fixation/capture from calibration. No absolute pixel/degree/diopter RMS gate, constant-fixation state, or temporal-flatness penalty is used. Numerical validity and honest coverage accounting remain mandatory.

These two documents supersede inherited area-normalization and held-condition selection requirements for this branch. Historical documents and scores are background, not evidence that this new branch already contains an estimator or an optimum accommodation law.

## 2. What the optical results allow us to simplify

The source models center-relative grids, not the measured separation of P1 and P4 centers. It supports the following starting hypotheses; the evidence levels must remain distinct [D1–D3].

| Component | Initial treatment | Evidence/limit |
|---|---|---|
| P1 reference barrel structure | Keep an empirical distorted template; no extra radial correction | Reference distortion is already present |
| P1 dependence on accommodation | None | Modeling premise; P1 sweep did not vary A |
| P1 and P4 rotation deformation | Keep explicit keystone operators | Weak change is not zero change |
| Axial scale versus Z | Approximately linear for each reflection | Supported separately over the reported domains |
| Gaze dependence of axial scale | Omit initially | Five-angle scale variation: P1 0.00019129%, P4 0.00294912% |
| P4 axial slope dependence on A | Omit initially; retain as an optional correction | Slopes span 0.0818504% of their mean over 0–4 D |
| P1/P4 fractional scale difference | Set `eta=1` initially | Near-equal slopes motivate this; they do not prove exact cancellation |
| A dependence of P4 keystone coefficients | Omit initially | A hypothesis, not an established small-term result |
| P4 accommodation magnification/radial response | Keep and calibrate | These are the mechanisms of interest |

The separate source scale laws are

$$
g_1(Z)\simeq1+\alpha_{1Z}Z,\qquad
g_4(A,Z)\simeq1+\alpha_{4Z}(A)Z.
$$

The P1 slope is `0.00295165270232 mm^-1`; P4 slopes for A=0,1,2,3,4 D are respectively `0.00295232758`, `0.00295172708`, `0.00295112891`, `0.00295052435`, `0.00294991207 mm^-1`. Linear in Z does not mean linear in A. The P1 linear-scale model has worst coordinate RMSE 0.29549 micrometers at zero rotation; the P4 linear-scale study has worst-state coordinate RMSE about 0.249 micrometers and maximum point error about 0.511 micrometers. These are simulation-model residuals, not experimental error limits [D1–D3].

Do not delete keystone based on the small *Z-scale* variation. A constant P1 pattern differs from the simulated pattern at -20 degrees by about 30.42 micrometers coordinate RMSE. The source's identity/parity simplification changes existing P1 quadratic-transform predictions by only about 0.0521 micrometers pooled coordinate RMS, but does not remove the spatial residual of the keystone family [D1].

The P4 Z-study lens hash is `fb3937e6763f27e331faa10a4863533e994030cbc0ae74364b186ba0d06d474d`; the older P4 accommodation/rotation study uses `5e0715c70016fbcec8a49956dd2ec824a50f5c4c52ca472b0225a2523db614ac`. Do not splice their numerical fits into a claimed matched joint calibration. Transfer the **structure**, calibrate the actual setup, and label any simulation-constrained parameters. Z is the recorded absolute THI coordinate, not measured eye-camera distance. A shared-scale model removes the need for a free Z state.

## 3. Measurement contract and units

Let `i` identify a frame, `j=1,2,3` a source correspondence, and

$$
\mathbf p_{ij}=P1_{ij},\qquad \mathbf q_{ij}=P4_{i,\pi(j)}.
$$

Read correspondence/validity from each payload; the inherited convention is zero-based `pair_index=[2,1,0]`. Never independently sort patterns. Define

$$
\mathbf c_{1i}=\frac13\sum_j\mathbf p_{ij},\qquad
\mathbf c_{4i}=\frac13\sum_j\mathbf q_{ij},\qquad
\mathbf e_{1i}=\begin{bmatrix}\mathbf p_{i2}-\mathbf p_{i1}\\\mathbf p_{i3}-\mathbf p_{i1}\end{bmatrix}.
$$

Use the nonredundant, linear, translation-free observation

$$
\boxed{\mathbf y_i=
\begin{bmatrix}\mathbf e_{1i}\\\mathbf q_{i1}-\mathbf c_{1i}\\
\mathbf q_{i2}-\mathbf c_{1i}\\\mathbf q_{i3}-\mathbf c_{1i}\end{bmatrix}\in\mathbb R^{10}.}
$$

The map `y=L P` from the 12 ordered native coordinates has rank 10; its nullspace is precisely the two shared translations. Two edges share P1 noise. No area determinant occurs. Do not append centroid/edge/area summaries as independent residuals. Centering P4 separately is allowed for diagnostics only if the relative centroid displacement is retained.

Use measured pixels and **reference-image pixels** for templates, making the nuisance scale `g` dimensionless. A fixed documented conversion is required for millimeter optical templates. Optical coefficients in inverse millimeters cannot be copied into reference-pixel equations. Corrected coordinates retain reference length units; they are not necessarily dimensionless.

State is `x_i=(theta_i,A_i)` in degrees and diopters. All 20 reviewed conditions in captures 1–4 participate. Nominal horizontal targets are `[-10,-5,0,5,10]` degrees, nominal vertical gaze is zero, and image y remains an observation. The lowest stored demand is about 0.36036 D, not measured A=0. Read labels from the fixation file [G1]. Use all valid rows in its declared full intervals; any trimming/sampling is a separately named population. Captures 5/6 remain reserved.

Define `vartheta=theta-theta_opt` using one shared optical alignment, and `a=(A-A_ref)/(1 D)`. An implementation may encode `t=vartheta/(10 degrees)`, but must transform coefficient units and derivatives consistently. Gaze targets, state bounds, and basis scales are different quantities.

## 4. Explicit optical operators: radial distortion and keystone

### 4.1 Keystone is a rational map, not an unspecified correction

For reflection `r=1,4`, in its declared fixed optical axes,

$$
\boxed{
K_r(\vartheta,A;\mathbf b)=
\begin{bmatrix}
\dfrac{s_{x,r}(\vartheta,A)b_x}{1+q_r(\vartheta,A)b_y}\\[4pt]
\dfrac{s_{y,r}(\vartheta,A)b_y}{1+q_r(\vartheta,A)b_y}
\end{bmatrix},\qquad
H_r=\begin{bmatrix}s_{x,r}&0&0\\0&s_{y,r}&0\\0&q_r&1\end{bmatrix}.}
$$

The same projective denominator changes both coordinates. The minimal functions are

$$
s_{x,r}=1+\alpha_r\vartheta^2,\quad
s_{y,r}=1+\beta_r\vartheta^2,\quad
q_r=\gamma_r\vartheta.
$$

P1 has no direct A dependence; the first P4 keystone has no explicit A dependence either. Optional couplings are, for example, `q4=(gamma40+gamma41*a)*vartheta` or `sx4=1+(alpha40+alpha41*a)*vartheta^2`. Add only a demonstrated necessary term. Preserve `K4(0,A;b)=b` so a free zero-angle scale cannot duplicate the accommodation baseline.

For reference length L, alpha/beta have units degree^-2, gamma has L^-1 degree^-1, and q has L^-1. When axes differ, use one fixed orthonormal `Q_r` and `Q_r K_r(vartheta,A;Q_r^T b)`. Do not rotate every observed triangle into a fitted orientation.

Require positive directional scales and `1+q_r b_y >= delta_K > 0` on the declared domain. This is a model-domain constraint, not an accuracy limit. Keystone cannot generate nonzero output y from input `b_y=0`; the source shows such a residual. Higher-degree angle coefficients alone cannot repair that spatial deficiency [D1].

### 4.2 Full P1 local transformation

Use a real, already distorted zero-gaze template `b1_j`:

$$
\boxed{\mathbf F_{1j}(\theta)=K_1(\vartheta;\mathbf b_{1j}),\qquad
\mathbf P_{1j}^{local}(\theta,Z)=g_1(Z)\mathbf F_{1j}(\theta).}
$$

Expanded in aligned axes,

$$
\mathbf P_{1j}^{local}=(1+\alpha_{1Z}Z)
\begin{bmatrix}
(1+\alpha_1\vartheta^2)b_{1j,x}/(1+\gamma_1\vartheta b_{1j,y})\\
(1+\beta_1\vartheta^2)b_{1j,y}/(1+\gamma_1\vartheta b_{1j,y})
\end{bmatrix}.
$$

If a matched paraxial P1 template is known, an alternative representation is `b1_j=[1+kappa1_ref*||u1_j||^2]u1_j`. It is not an extra correction to the real template. No per-frame P1 radial coefficient is fitted. External scaling acts **after** keystone: `S(g1)H1`, not `H1S(g1)`.

### 4.3 Full P4 local transformation

For a known/constrained paraxial template `u4_j` about the local radial origin, set

$$
\mathbf z_j=M(A)\mathbf u_{4j},\qquad M(A_{ref})=1,
$$

$$
\boxed{\mathbf B_{4j}(A)=
[1+\kappa_4(A)\|\mathbf z_j\|^2]\mathbf z_j,\qquad
\mathbf F_{4j}(\theta,A)=K_4(\vartheta,A;\mathbf B_{4j}(A)).}
$$

Writing `R_j^2=||u4_j||^2` and `B_j(A)=M(A)[1+kappa4(A)M(A)^2 R_j^2]`, the complete aligned-axis local map is

$$
\boxed{
\mathbf P_{4j}^{local}(\theta,A,Z)=[1+\alpha_{4Z}(A)Z]
\begin{bmatrix}
\dfrac{s_{x,4}(\vartheta,A)B_j(A)u_{4j,x}}
{1+q_4(\vartheta,A)B_j(A)u_{4j,y}}\\[5pt]
\dfrac{s_{y,4}(\vartheta,A)B_j(A)u_{4j,y}}
{1+q_4(\vartheta,A)B_j(A)u_{4j,y}}
\end{bmatrix}.}
$$

The order is **accommodation magnification -> radial distortion -> gaze keystone -> external axial scale**. Accommodation changes the keystone denominator even with A-independent alpha4/beta4/gamma4. This is the essential joint coupling. The derivative is

$$
\partial_A\mathbf F_4=(\partial_b K_4)\partial_A\mathbf B_4
+(\partial_A K_4)_b.
$$

The second term vanishes only for the minimal keystone; the first does not. `M(A)` is the desired accommodation signal, not the nuisance `g4(A,Z)`. Never normalize it away. Kappa has units L^-2. For a nonfolding cubic radial map, check `1+kappa*r^2` and `1+3*kappa*r^2` across the used radii.

### 4.4 The practical empirical-template route

The recordings do not supply a paraxial grid or a measured distortion center. Do not assume the observed P4 reference is undistorted. If a reference radial map is known, use `B4(A)=R_A(R_Aref^-1(b4_ref))`. Otherwise the first executable model uses an **incremental empirical deformation**:

$$
\boxed{\mathbf B^{rel}_{4j}(A)=
M(A)\left[1+\Delta\kappa(A)M(A)^2\|\mathbf b_{4j}^{ref}\|^2\right]
\mathbf b_{4j}^{ref},\quad M(A_{ref})=1,\quad\Delta\kappa(A_{ref})=0.}
$$

Here `b4_ref` is the real template expressed about a declared shared local origin. The increment is exactly identity at the reference, so it does not double-apply baseline barrel distortion. Its delta-kappa is an **effective incremental coefficient**, not an absolute physical radial coefficient. It shares the same keystone, relative-reference and calibration machinery. Record `baseline_mode=empirical_incremental` versus `paraxial_absolute`; never interchange artifacts between these modes.

## 5. Joint transformation in the actual relative reference

Define model means and centered shapes

$$
\overline{\mathbf F}_r=\tfrac13\sum_j\mathbf F_{rj},\quad
\mathbf S_{rj}=\mathbf F_{rj}-\overline{\mathbf F}_r,\quad
\mathbf a_1(\theta)=\begin{bmatrix}\mathbf F_{12}-\mathbf F_{11}\\\mathbf F_{13}-\mathbf F_{11}\end{bmatrix}.
$$

Let `g=g1`, `eta=g4/g1`, and `delta_g(theta,A,g)` be a **relative optical-origin displacement** in P1-reference units. Introduce an arbitrary `t_i` only to derive cancellation:

$$
\mathbf p_j=\mathbf t+g\mathbf F_{1j},\qquad
\mathbf q_j=\mathbf t+g\boldsymbol\delta_g+g\eta\mathbf F_{4j}.
$$

Subtraction removes t. Define the relative-centroid law

$$
\mathbf h_g=\boldsymbol\delta_g+\eta\overline{\mathbf F}_4-\overline{\mathbf F}_1.
$$

Fit **h_g or delta_g, not both independently**. Center-relative simulations do not provide their separation. The full measured-vector prediction is

$$
\boxed{\widehat{\mathbf y}(\theta,A,g;\Psi)=
\begin{bmatrix}
g\mathbf a_1(\theta)\\
g[\mathbf h_g+\eta\mathbf S_{41}]\\
g[\mathbf h_g+\eta\mathbf S_{42}]\\
g[\mathbf h_g+\eta\mathbf S_{43}]
\end{bmatrix}.}
$$

Both reflections are modeled, not just P4 conditioned on an exact P1. Psi contains shared templates, alignment, keystone, accommodation functions and displacement coefficients. No free framewise deformation map is present.

The **default model** is `eta=1`, `h_g=h(theta,A)`. The scale-corrected expressions are then

$$
\boxed{\widehat{(\mathbf p_j-\mathbf c_1)/g}=\mathbf S_{1j},\qquad
\widehat{(\mathbf q_j-\mathbf c_1)/g}=\mathbf h+\mathbf S_{4j}.}
$$

Equivalently, the second expression is `delta+K4(vartheta,A;B4_j(A))-mean_l K1(vartheta;b1_l)`: the complete relative barrel/keystone transformation is retained.

For initialization use

$$
\mathbf h(\theta,A)=\mathbf b_0+\mathbf b_A a+
(\mathbf s_0+\mathbf s_A a)t,\qquad t=\vartheta/(10\text{ degrees}).
$$

The four vector coefficients allow accommodation-dependent centroid offset and gaze gain (eight scalars). H_y is not vertical gaze. Avoid unrestricted capture-specific offsets when capture and accommodation demand are confounded. This centroid law is calibrated independently of the center-relative optical shape, but the common six-coordinate fit couples their estimates.

## 6. P1-reference scale: no triangle normalization

For P1-edge covariance `R11`, fixed within an objective, put `W1=R11^-1`. At each trial theta,

$$
\boxed{\widehat g_{P1}(\theta)=
\frac{\mathbf a_1^T W_1\mathbf e_1}{\mathbf a_1^T W_1\mathbf a_1}.}
$$

Require nonzero reference edge energy and a finite positive solution. Invalid scale is a reported invalidity, not silently clipped success. Fit one scalar only. Changing edge origin with consistently transformed covariance must preserve this generalized least-squares solution. This estimate does not require a measured center or a nonzero triangle determinant; joint optical identifiability remains a separate requirement.

Correct observations for interpretation as

$$
\widetilde{\mathbf p}_j=(\mathbf p_j-\mathbf c_1)/\widehat g,
\qquad\widetilde{\mathbf q}_j=(\mathbf q_j-\mathbf c_1)/\widehat g.
$$

Do not divide twice, and do not infer scale from a varying fitted P1 barrel coefficient. Include the trial-gaze/global-parameter dependence of g. For fixed W1,

$$
\partial_\theta\widehat g=
\frac{\mathbf a_{1,\theta}^T W_1\mathbf e_1-
2\widehat g\,\mathbf a_{1,\theta}^T W_1\mathbf a_1}
{\mathbf a_1^T W_1\mathbf a_1}.
$$

An old area ratio is only an optional diagnostic. In a noiseless nondegenerate common-scale example, the fitted g equals `sqrt(area(P1))/sqrt(area(F1(theta)))`; under noise it is a different estimator. Removing the area denominator does not create extra information or justify three independent edge weights.

### Optional differential scale, not another free state

If matched evidence requires unequal fractional scales,

$$
\eta(A,Z)=\frac{1+\alpha_{4Z}(A)Z}{1+\alpha_{1Z}Z}.
$$

For a common reference and nonzero alpha1Z, eliminate Z using `c(A)=alpha4Z(A)/alpha1Z`:

$$
\boxed{g_4=1+c(A)(g-1),\qquad \eta(A,g)=c(A)+[1-c(A)]/g.}
$$

Use `g[h_g+eta*S4]`, not `g*eta*(h+S4)`. Pattern-scale evidence does not establish centroid scaling. Do not fit eta independently per frame; it can absorb accommodation. Its initial value is one. Any c(A) or residual g dependence of h requires a separately calibrated ablation, with derivatives through g and a declared common reference plane. Source slopes from differing lens revisions are not ready-made experimental corrections.

## 7. Reference geometry and identifiability before coefficient expansion

Zero-gaze centroids initialize **separate reference origins**, not exact radial/keystone centers. A distorted centroid can shift even with a centered paraxial template because `mean(kappa*M^3*||u||^2*u)` need not vanish. Use multiple near-zero-gaze frames, correspondence and weak P1 symmetry evidence; do not equate the most symmetric noisy triangle with exact visual/optical zero.

Initially freeze center offsets and template geometry after initialization. Refining a shared center requires a constrained, identifiable model and sensitivity analysis; never fit a center per frame. The paraxial template, absolute barrel coefficient, center and magnification cannot all be inferred freely from three reference points.

Fix the reference length, optical zero, `M(A_ref)=1`, and P1 scale convention. A gaze-dependent isotropic rescaling of all reference predictions can trade off against g. Use matched optical constraints or a declared empirical P1 rotation reference and freeze that scale convention; otherwise call g an **effective reference scale**, not a measured pure axial factor. If a gauge changes, transform P4 and the displacement law consistently as well. Fit P1 residuals under the joint state model, but do not let an arbitrary isotropic P1 gauge drift during optimization.

If all three reference radii equal R,

$$
\mathbf B_{4j}=M(A)[1+\kappa_4(A)M(A)^2R^2]\mathbf u_{4j}.
$$

Only the combined scale is identifiable from this spatial pattern. More frames at the same radii do not automatically separate M and kappa. Near-equal radii lead to weak separation. Check the centered scale/radial design rank after permitting centroid motion. Use an **effective-scale-only model** if needed, or fix one mechanism from matched optical evidence. A future source at a different nonzero radius helps; a center point alone does not separate these two effects.

Nominal demands constrain a numerical accommodation convention, not framewise physiological truth. Monotone A warps can be partly absorbed by response functions. Raw G_A or information per diopter cannot alone select a law. Preserve common anchor, bound, center and prior sensitivity tests, and distinguish independent physical accuracy from internal agreement.

## 8. A single, explicit calibration cost

Let native covariance be Sigma and `R_i=L Sigma_i L^T`. Freeze a common covariance policy across candidate fits. Estimate it from appropriate contiguous records or independent localization evidence; temporal differences can contain real motion. Shared P1 errors and cross-reflection correlations remain. Do not compute a P4 holdout covariance from the omitted observed coordinate.

**Default policy: `p1_profile_v1`.** Substitute the P1-only scale into the raw relative forward model,

$$
\mathbf r_i(x_i,\Psi)=\mathbf y_i-
\widehat{\mathbf y}(x_i,\widehat g_{P1,i}(\theta_i);\Psi),
$$

$$
\boxed{J=\frac1{2K}\sum_{k=1}^K\frac1{N_k}
\sum_{i\in k}\mathbf r_i^T R_i^{-1}\mathbf r_i
+\frac1{2K}\sum_k\left\|
\begin{bmatrix}(\bar\theta_k-\theta_k^{nom})/s_\theta\\
(\bar A_k-A_k^{demand})/s_A\end{bmatrix}\right\|^2
+\lambda\,\mathcal P(\Psi).}
$$

The inherited mean-anchor scales `s_theta=0.10 degree`, `s_A=0.25 D` are finite weights, not error limits. There is no temporal penalty. The global regularizer is stated in dimensionless, physically declared parameter scales; use weak shrinkage of optional departures, not arbitrary raw-coefficient norms across incomparable bases.

This is a **declared P1-constrained least-squares criterion**, not the maximum likelihood obtained by optimizing g over all ten coordinates. Because g is estimated from noisy P1, R is a fixed metric for this criterion, not automatically the covariance of the plug-in residual. Do not interpret its minimized value as a calibrated chi-square statistic.

For illustration in common-scale mode, write `f=yhat/g`, `g=l y` at fixed state, with `l=(a1^T W1 E)/(a1^T W1 a1)` and E selecting the four P1 edges. Then `l f=1` and the plug-in residual covariance is `(I-f l)R(I-f l)^T`, of rank at most nine. Treating corrected/normalized components as independent or giving that singular covariance arbitrary diagonal floors would invent information. Propagate measurement dependence for uncertainty diagnostics, or use a nonredundant representation [U1].

**Optional comparison policy: `joint_profile_v1`.** With common scale and h independent of g, the joint raw Gaussian profile is `g_joint=(f^T R^-1 y)/(f^T R^-1 f)`. This is an analytically convenient nuisance fit, but allows retained P4 to affect scale. It is not silently equivalent to P1 normalization under noise. Keep it as a separately named ablation, initialize from P1, mask it correctly, and do not make it the default. Differential eta or h(g) generally removes the simple joint scalar profile.

Use a new adapter for the composed model. The old 27-column affine-response design is not this model. Linear centroid coefficients can still be solved exactly at fixed states/nonlinear optics; radial terms inside projective denominators are generally nonlinear. Retain the established bounded least-squares/variable-projection approach only where its separability actually holds.

## 9. Full calibration and iterative state correction

### Initialization

Use the lowest-demand capture to initialize a linear mapping from fixation-mean P4–P1 displacement to the five nominal gaze targets, after provisional P1 scale correction. Initialize each frame separately; do not assign its fixation label as truth. Use `A_ref` from the low-demand convention, not a fictitious measured zero-A capture.

Initialize reference geometry from many near-zero-gaze frames. Freeze the length/center/P1-scale convention. Initialize M and the identifiable radial increment using near-zero-gaze samples across all demands. Begin with linear dependence in `a`; estimate actual frame A with soft mean anchors. This staged initialization uses available conditions, not held-condition training.

### Iteration

With shared parameters fixed, recompute g at every trial gaze. A convenient gaze proposal for `h_x=b(A)+s(A)*vartheta` is

$$
\theta_{proposal}=\theta_{opt}+
\frac{(c_{4,x}-c_{1,x})/\widehat g-b(A)}{s(A)}.
$$

Accept/refine updates using the **same J**:

1. Update P1 reference scale at current theta.
2. Update A conditional on theta through the complete forward radial/keystone model.
3. Update theta conditional on A, recomputing g and its derivatives.
4. Revisit scale and use a joint two-state refinement if alternation stalls.
5. During calibration only, update shared optical/centroid parameters using all frames; repeat.

Apply damping/line search to J. Conditional calibration steps include their contribution to global fixation means; an anchor on every frame is a different objective. Application and per-frame cross-checks have no nominal-label anchor. Do not inverse-warp about an observed centroid as though it were the optical center: centering, radial mapping and keystone generally do not commute.

Stop on recorded scaled stationarity, objective/step stabilization and domain validity, not an arbitrary number of alternations or an RMS target. Multistart can expose competing minima; local convergence is not proof of global uniqueness. Real within-fixation motion remains allowed.

## 10. Compare accommodation models by cross-agreement

After full calibration, freeze all global parameters, templates, centers and shape laws. For each frame omit P4_j and run the **entire** scale/state inference with all P1 and only the other two P4. Select the covariance marginal before whitening. With P1-profile scale there are eight retained scalar coordinates and two states; scale is a P1-derived nuisance, not an independent measured truth.

The omitted point must not enter the initial gaze, measured P4 centroid/area, weights, global-template updates, all-three warm starts, branch choice or diagnostics used to accept that subset. Means over three **predicted** points are allowed. Fixed calibrated parameters have previously seen all frames: the result is internal calibration agreement, not independent validation.

Score in native relative-pixel units:

$$
\mathbf e_{ij}^{cross}=(\mathbf q_{ij}-\mathbf c_{1i})-
\widehat g_{i,-j}[\mathbf h_g+\eta\mathbf S_{4j}](\widehat x_{i,-j}).
$$

$$
E_i^2=\frac13\sum_j\|\mathbf e_{ij}^{cross}\|^2,\quad
G_{A,i}^2=\frac13\sum_{j<l}(\widehat A_{i,-j}-\widehat A_{i,-l})^2,
$$

with analogous G_theta. These compare simultaneous subset states, not different times or nominal demand. Average squares within exposure then equally across exposures; RMS is a display summary. Score all laws on identical populations, retain missing/failed slots, and report signed axes, per-point/worst errors, tails, bounds and branches.

E measures optical reconstruction; G_theta/G_A measure compatibility in the declared state convention. Neither alone determines a good accommodation model. In a raw model with explicit scale, accommodation distinguishability can be diagnosed by projecting its whitened derivative away from gaze and scale: `jA_perp=(I-N N^+)W J_A`, `N=[W J_theta,W J_g]`. Report information/rank/conditioning with fixed physical scales and gauge sensitivity. For P1-profile uncertainty, propagate the complete estimator, including its data-dependent g, rather than treating g as noiseless or importing a two-channel formula.

Subset differences share P1 and overlap in P4. Uncertainty of a difference includes cross-covariance; three differences are not independent replicates. Low raw G_A can be produced by compression or shared clipping. Keep raw E/G, optical response curves and observability diagnostics together. Do not add px^2, degrees^2 and D^2 into an arbitrary cost.

Candidate-dependent predictive scoring such as `0.5*(e^T V^-1 e+log det V)` is optional only after validating V, its shared correlations and its reference units. It is not the first deliverable and cannot replace raw reconstruction errors or hide model mismatch by covariance inflation.

## 11. Model hierarchy and deliverable

Start with the same fixed reference/center convention, keystone, centroid law and scale policy:

- **DM0:** accommodation-dependent effective magnification only. This is the identifiable fallback/control.
- **DM1:** magnification plus one linear radial increment (or absolute radial slope with a separately constrained paraxial/reference model).
- **DM2, conditional:** one additional global curvature/shape parameter in the mechanism that leaves reproducible residual structure. Log-to-linear shifted power, literal positive-A power, or one quadratic term are alternative parameterizations of M or the radial increment, not simultaneous freedoms on every output.

For each candidate refit all global parameters and framewise states on all conditions. If the data cannot distinguish M and radial change, report the effective response instead of manufacturing absolute barrel coefficients. Do not resume a broad exponent sweep or add arbitrary nuisance states merely to shave pixel error. The scientific deliverable is a defensible accommodation-sensitive optical response, its cross-agreement, and clearly identified limitations.

## Sources and scope

- [D1: Distortion theory, pinned](https://github.com/rueijrwu/distortion_tracking/blob/1d2a0874c79f3f174b36a197c3fbbffd8a32fe60/Theory.md).
- [D2: Optical result summary, pinned](https://github.com/rueijrwu/distortion_tracking/blob/1d2a0874c79f3f174b36a197c3fbbffd8a32fe60/Summary.md).
- [D3: P4 Z-magnification report, pinned](https://github.com/rueijrwu/distortion_tracking/blob/1d2a0874c79f3f174b36a197c3fbbffd8a32fe60/data/p4_z_magnification/p4_z_magnification.md).
- [G1: Reviewed fixation metadata](../data/fixations/fixation_intervals.json). Detection arrays/correspondence are in the trusted repository pickles, not regenerated by this document.
- [U1: NIST, law of propagation of uncertainty](https://physics.nist.gov/cuu/Uncertainty/combination.html).

Numerical simplification evidence applies only to the source domain and conventions. Centers, source geometry and response coefficients remain experimental calibration choices. This document does not claim a completed combined lens model, a GPU implementation, or physiological accuracy.
