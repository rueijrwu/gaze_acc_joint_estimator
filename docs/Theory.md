# Joint distortion estimator: separate P1/P4 optical zeros and center-based gaze

**Branch:** `exp5_distortion_model` in `rueijrwu/gaze_acc_joint_estimator`.  
**Status:** Authoritative proposed mathematics for this branch, not an implemented estimator or a new experimental result.  
**Revision:** 2026-10-09; replaces the specification at `2fbd742f7cb2ca26eb84ea0a6029dd9de1e10b60`.  
**Companion:** [ESTIMATOR_PLAN.md](ESTIMATOR_PLAN.md).  
**Optical evidence:** `distortion_tracking` at `1d2a0874c79f3f174b36a197c3fbbffd8a32fe60` [D1-D3].

## 1. Scientific objective and required calibration order

Find an accommodation-sensitive optical model whose three simultaneous P1/P4 pairs support the same horizontal gaze and accommodation. Do not reduce the objective to a marginal improvement in pixel RMS. The calibration proceeds from crude centroids to model-corrected reflection centers:

1. Use fixation-mean **P4 minus P1** centroid displacement in the reference-accommodation recording to initialize a mostly linear, up-to-cubic gaze mapping.
2. Find P1's own most symmetric gaze, fit P1 distortion about that optical zero, and calculate a P1-derived scale for every frame across all accommodation conditions.
3. Independently find P4's most symmetric gaze. It is P4's optical alignment zero; it need not equal visual zero or P1's symmetry zero.
4. At P4's symmetry reference, initialize accommodation-dependent magnification and barrel deformation across accommodation conditions.
5. With that baseline available, fit P4 gaze-dependent keystone across the five gaze conditions.
6. Estimate all frame states, reconstruct distortion-corrected P1/P4 centers, and refit the gaze polynomial to their **P4 minus P1** separation.
7. Repeat the full calibration cycle, accepting changes against one relative-coordinate objective and assessing complete iterative cross-agreement.

The final fit uses all reviewed calibration conditions. The stages are initialization and block-update order, not held-out-condition experiments. Each frame retains its own gaze and accommodation. Fixation labels provide soft mean anchors, never framewise truth or a requirement for flat trajectories.

One positive scale, derived from P1 at each trial gaze, is applied to both reflection patterns. No free axial Z state, second P4 nuisance scale, differential-scale parameter, or required triangle-area normalizer belongs to the active estimator. P4 accommodation magnification remains signal.

## 2. Measurement contract: P4 minus P1 and only relative observations

Let i identify a frame and j=1,2,3 a persistent source correspondence. Write

$$
\mathbf p_{ij}=P1_{ij},\qquad \mathbf q_{ij}=P4_{i,\pi(j)}.
$$

Read correspondence and flags from the payload; the inherited zero-based permutation is `[2,1,0]`. Do not independently sort the patterns. Define

$$
\mathbf c_{1i}=\tfrac13\sum_j\mathbf p_{ij},\quad
\mathbf c_{4i}=\tfrac13\sum_j\mathbf q_{ij},\quad
\mathbf e_{1i}=\begin{bmatrix}\mathbf p_{i2}-\mathbf p_{i1}\\\mathbf p_{i3}-\mathbf p_{i1}\end{bmatrix}.
$$

The ten-component linear relative observation is

$$
\boxed{\mathbf y_i=\begin{bmatrix}
\mathbf e_{1i}\\
\mathbf q_{i1}-\mathbf c_{1i}\\
\mathbf q_{i2}-\mathbf c_{1i}\\
\mathbf q_{i3}-\mathbf c_{1i}
\end{bmatrix}.}
$$

The map from the twelve native coordinates to y has rank ten. Adding an arbitrary same-frame translation to all six points leaves y unchanged. Absolute image coordinates are input bookkeeping, not absolute eye-position observations. A fixed camera-origin reference recorded earlier cannot remove later translation.

The initial gaze signal has the fixed sign

$$
\boxed{\mathbf d_i^{cent}=(\mathbf c_{4i}-\mathbf c_{1i})/g_i.}
$$

Do not discard the centroid difference when displaying separately centered shapes. Do not append centroid, area and redundant edge summaries to y as independent extra observations. Neither square-root P1 area nor P4/P1 triangle-area ratio is required; historical values may be logged as diagnostics only.

Use camera pixels and explicitly defined reference-image pixels, making g dimensionless. Optical coefficients in inverse millimeters require a fixed documented unit conversion. Corrected coordinates retain reference-length units. Common additive image translation cancels exactly; real translation that changes field sampling, perspective or the two optical paths differently is not thereby guaranteed to cancel.

## 3. Three distinct angular zeros

### 3.1 Visual gaze and two reflection-specific optical alignments

Let theta be horizontal gaze in the experiment's **visual calibration convention**. Its nominal fixation targets remain -10, -5, 0, 5 and 10 degrees. Nominal vertical gaze is zero, but image-y coordinates remain observations. There is no vertical-gaze latent state.

Define two globally shared angular offsets:

$$
\omega_1=\text{visual gaze at P1's symmetry zero},\qquad
\omega_4=\text{visual gaze at P4's symmetry zero}.
$$

The distortion arguments are

$$
\boxed{\xi_1=\theta-\omega_1,\qquad \xi_4=\theta-\omega_4.}
$$

There is one physical theta per frame, not separate P1 and P4 gaze states. In general

$$
\omega_1\ne\omega_4,\qquad \omega_1\ne0,\qquad \omega_4\ne0.
$$

At visual theta=0 the distortion inputs are -omega1 and -omega4, not zero. At P4's optical zero theta=omega4, P1 usually remains off its own optical zero. The offsets can be substantial; do not impose a near-zero prior solely because the fixation protocol contains a zero-degree target.

Each offset is shared across frames and captures, initially independent of accommodation. The most symmetric pattern defines an **operational optical alignment zero** for that reflection. This is not automatically an independently measured anatomical axis. If a reproducible accommodation-dependent P4 symmetry shift remains, report it and test a separately declared extension rather than assigning a new zero to each frame.

### 3.2 Converting P1 evaluation to P4's angular reference

Define

$$
\boxed{\Delta_{14}=\omega_4-\omega_1.}
$$

Then

$$
\boxed{\xi_1=\xi_4+\Delta_{14},\qquad
\theta=\xi_1+\omega_1=\xi_4+\omega_4.}
$$

Thus, when calculations are expressed using P4-local gaze, evaluate P1 at **xi4 + Delta14**, not xi4. For example, omega1=-3 degrees and omega4=+4 degrees imply xi1=+7 degrees and xi4=0 at visual theta=+4 degrees. These are illustrative offsets, not measured results.

Use visual theta as the canonical optimizer/output state to avoid unnecessary conversions. The offsets are two global calibration parameters, not two extra variables in every frame. Camera-axis alignment matrices and radial-center positions are different quantities from these angular offsets.

### 3.3 How the symmetry zeros are initialized

Evaluate P1 and P4 symmetry separately over the five gaze conditions, using many valid frames and the actual source correspondence. P1 can pool across accommodation because its initial model has no direct A dependence. Select P4's reference after P1 scale correction, evaluating the candidate gaze conditions across accommodation recordings.

The symmetry criterion must reflect the real source arrangement, such as its known mirrored source pairing and a fixed optical image axis. Do not equate symmetry with an equilateral triangle or use a freely fitted affine transform that makes every pattern appear symmetric. A dimensionless shape diagnostic may be used for this reference-selection step, but its rescaling is not the estimator's P4 normalization.

The best of five conditions initializes the offset; it need not be the exact continuous symmetry minimum. Preserve the scores, uncertainty, ties and endpoint selections. Refine the two shared offsets against the constrained full model when identifiable. Do not set every frame in a selected fixation to xi=0. If a minimum is broad, keep competing offset candidates or fix a declared reference and report weak alignment information. A possible zero outside the sampled range is not validated merely by extrapolating a symmetry curve.

## 4. Optical evidence and the common-scale simplification

The optical source supports approximately isotropic, nearly common axial magnification for P1 and P4. P1's reported linear Z slope is 0.00295165270232 mm^-1. The separately measured P4 slopes range from 0.00295232758 at 0 D to 0.00294991207 at 4 D. Both studies checked five gaze angles. These findings motivate using the same P1-derived scalar for both reflections [D1-D3].

The active estimator does **not** compute g from Z or fit those published slopes. It obtains g directly from reference P1 geometry. The remaining P4 accommodation magnification M(A) is a different mechanism and must not be normalized away. P1 and P4 retain distinct distortion maps and coefficients; sharing nuisance scale does not mean sharing all optical responses.

Retain P1's empirical barrel-distorted baseline and both keystone operators. Omit extra Z-dependent distortion/scale coefficients in the initial estimator. P1's lack of direct A dependence is a modeling premise, not a conclusion from a P1 accommodation sweep. P4 keystone coefficients initially lack explicit A dependence, but this is an initial hypothesis rather than an established negligible coupling.

The P4 Z dataset and earlier accommodation/rotation dataset use different lens revisions. Their reported residuals, near-equal slopes and the source's approximately 1-micrometer simplification reference are not experimental error guarantees. Do not treat the small difference between slope fits as a bound on all normalization error or assume it establishes Z invariance of the relative center displacement. Audit residual dependence on P1 scale without freeing a second per-frame scale [D1-D3].

## 5. Full distortion transforms about each reflection's own symmetry zero

### 5.1 Explicit keystone operator

In fixed aligned image axes, define for r=1 or 4

$$
\boxed{K_r(\xi_r,A;\mathbf b)=
\begin{bmatrix}
\dfrac{s_{x,r}(\xi_r,A)b_x}{1+q_r(\xi_r,A)b_y}\\[4pt]
\dfrac{s_{y,r}(\xi_r,A)b_y}{1+q_r(\xi_r,A)b_y}
\end{bmatrix},\quad
H_r=\begin{bmatrix}s_{x,r}&0&0\\0&s_{y,r}&0\\0&q_r&1\end{bmatrix}.}
$$

The minimal symmetry-centered functions are

$$
s_{x,r}=1+\alpha_r\xi_r^2,\qquad
s_{y,r}=1+\beta_r\xi_r^2,\qquad
q_r=\gamma_r\xi_r.
$$

For P1 these functions have no A dependence. Initially the P4 coefficients are also independent of A, while its baseline depends on A. Optional couplings such as q4=(gamma40+gamma41*a)*xi4 are separate controlled extensions. Identity is enforced at each **native optical zero**: K1(0;b)=b and K4(0,A;b)=b. It is not enforced at visual theta=0 for both reflections.

If optical axes differ from camera axes, conjugate by a fixed calibrated orthonormal matrix Qr. These image-axis rotations do not replace omega1/omega4 and must not be fitted independently per frame. With xi in degrees and reference length L, alpha/beta have degree^-2, q has L^-1 and gamma has L^-1 degree^-1. Record conversions when using xi/10 numerically.

Require positive scales and denominators bounded away from zero over the actual local-angle and field domain. Large offsets shift that domain: a visual bound of +/-20 degrees does not imply both xi ranges are +/-20 degrees. Keystone is a restricted family, not a complete aberration model; higher polynomial order cannot repair spatial components the rational map cannot express [D1].

### 5.2 P1: its own baseline, its own angular offset, common scale

Let b1j be the real, already distorted P1 template at theta=omega1 and reference magnification. Then

$$
\boxed{\mathbf F_{1j}(\theta)=K_1(\theta-\omega_1;\mathbf b_{1j}),}
$$

$$
\boxed{\mathbf P_{1j}^{local}=g_i
\begin{bmatrix}
\dfrac{[1+\alpha_1(\theta_i-\omega_1)^2]b_{1j,x}}
{1+\gamma_1(\theta_i-\omega_1)b_{1j,y}}\\[4pt]
\dfrac{[1+\beta_1(\theta_i-\omega_1)^2]b_{1j,y}}
{1+\gamma_1(\theta_i-\omega_1)b_{1j,y}}
\end{bmatrix}.}
$$

Do not apply P1's reference barrel distortion again. An absolute paraxial/radial representation is an alternative only if matched reference geometry is available. No free per-frame P1 barrel coefficient is introduced. The external common scale acts after keystone: S(g)H1, not H1S(g).

### 5.3 P4: accommodation baseline at P4 zero, then P4 keystone

For a known or separately constrained paraxial reference template u4j about the P4 local radial origin at theta=omega4, define

$$
\mathbf z_j(A)=M(A)\mathbf u_{4j},\qquad M(A_{ref})=1,
$$

$$
\boxed{\mathbf B_{4j}(A)=[1+\kappa_4(A)\|\mathbf z_j(A)\|^2]\mathbf z_j(A),\qquad
\mathbf F_{4j}(\theta,A)=K_4(\theta-\omega_4,A;\mathbf B_{4j}(A)).}
$$

Writing Bj(A)=M(A)[1+kappa4(A)M(A)^2||u4j||^2], the full aligned-axis local transformation is

$$
\boxed{\mathbf P_{4j}^{local}=g_i
\begin{bmatrix}
\dfrac{s_{x,4}(\theta_i-\omega_4,A_i)B_j(A_i)u_{4j,x}}
{1+q_4(\theta_i-\omega_4,A_i)B_j(A_i)u_{4j,y}}\\[4pt]
\dfrac{s_{y,4}(\theta_i-\omega_4,A_i)B_j(A_i)u_{4j,y}}
{1+q_4(\theta_i-\omega_4,A_i)B_j(A_i)u_{4j,y}}
\end{bmatrix}.}
$$

The order is accommodation magnification, radial distortion, gaze keystone, common external scale. The radius is evaluated before radial distortion. Kappa has L^-2 units. Inspect both 1+kappa*r^2 and the radial derivative 1+3*kappa*r^2 when requiring a nonfolding radial map.

Accommodation affects the keystone denominator through B4 even when alpha4/beta4/gamma4 are A-independent. Use the full chain derivative

$$
\partial_A\mathbf F_4=(\partial_b K_4)\partial_A\mathbf B_4+(\partial_A K_4)_b.
$$

### 5.4 Practical empirical accommodation baseline

The recordings do not directly supply paraxial templates or measured distortion centers. The default executable route starts from a real P4 template b4ref at its symmetry reference and Aref and fits a **relative increment**:

$$
\boxed{\mathbf B_{4j}^{rel}(A)=M(A)
[1+\Delta\kappa(A)M(A)^2\|\mathbf b_{4j}^{ref}\|^2]\mathbf b_{4j}^{ref},\quad
M(A_{ref})=1,\quad\Delta\kappa(A_{ref})=0.}
$$

This preserves the already distorted baseline rather than applying its original radial distortion twice. Delta-kappa is an effective incremental coefficient, not an independently measured absolute barrel coefficient. If the reference radial map is known, an alternative is B4(A)=R_A(R_Aref^-1(b4ref)). Record `empirical_incremental` versus `paraxial_absolute` explicitly.

Begin with M=1+m1*a and Delta-kappa=kA*a, where a=(A-Aref)/(1 D). Use effective magnification alone when scale and radial response are not separable. Shared curvature/power/log parameters may later describe these mechanisms; do not reintroduce an arbitrary independent deformation for every frame.

### 5.5 Re-referencing a P1 transform to P4 zero without changing predictions

Usually no spatial rewarp is necessary: simply evaluate K1 at xi1=xi4+Delta14. If a P1 template explicitly referenced to P4 zero is needed, use homogeneous composition:

$$
\mathbf b_{1}^{(4)}=\operatorname{dehom}\{H_1(\Delta_{14})[\mathbf b_1;1]\},
$$

$$
\boxed{H_{1|4}(\xi_4)=H_1(\xi_4+\Delta_{14})H_1(\Delta_{14})^{-1}.}
$$

Then H1|4(0)=I and applying H1|4 to b1^(4) yields exactly the original P1 prediction. This is a reference change, not a claim that P1 is symmetric at P4 zero. The native P1 coefficients retain parity about omega1; the re-referenced coefficient curves need not have that parity about omega4. Do not set the old coefficients' argument to xi4 and call the result equivalent.

The general same-reflection transfer between two angles is Hr(xi_b,A)Hr(xi_a,A)^-1 at fixed A. A source template at a different accommodation additionally needs the baseline accommodation mapping. Do not use an angle-only transfer to silently change A.

## 6. P1-reference magnification, shared by both reflections

Construct reference P1 edges

$$
\mathbf a_1(\theta,\omega_1)=
\begin{bmatrix}\mathbf F_{12}(\theta)-\mathbf F_{11}(\theta)\\\mathbf F_{13}(\theta)-\mathbf F_{11}(\theta)\end{bmatrix}.
$$

With the fixed P1-edge covariance R11 and W1=R11^-1,

$$
\boxed{\widehat g_{P1,i}(\theta,\omega_1)=
\frac{\mathbf a_1^\mathsf TW_1\mathbf e_{1i}}
{\mathbf a_1^\mathsf TW_1\mathbf a_1}.}
$$

Use solves instead of forming an inverse in implementation. Require finite positive g and nonzero weighted edge energy; record invalid cases rather than clipping silently. Only P1 is used to derive the scale. At fixed theta, omega1 and P1 observations, changing A or a P4 measurement cannot directly alter g.

Reevaluate g at every trial theta or P1/global-offset update. With fixed weights and template,

$$
\partial_\theta g=
\frac{\mathbf a_{1,\theta}^\mathsf TW_1\mathbf e_1
-2g\mathbf a_{1,\theta}^\mathsf TW_1\mathbf a_1}
{\mathbf a_1^\mathsf TW_1\mathbf a_1},\qquad
\partial_{\omega_1}g=-\partial_\theta g.
$$

Additional template dependencies need their chain terms. At fixed visual theta, omega4 does not enter the P1 scale. When a program uses xi4 as its state instead, changing omega4 changes theta=xi4+omega4; that chain rule must be included.

The interpretation-only corrected coordinates are (p-c1)/g and (q-c1)/g. They use no area denominator and must not be divided by g a second time. Do not normalize P4 by its own fitted size. The scale fit does not locate the distortion center and does not independently measure physical Z.

## 7. From point centroids to model-corrected centers

Let mu_r=(1/3)sum_j F_rj and S_rj=F_rj-mu_r. For deriving relative geometry, write the auxiliary local-origin model

$$
\mathbf p_{ij}=\mathbf C_{1i}+g_i\mathbf F_{1j},\qquad
\mathbf q_{ij}=\mathbf C_{4i}+g_i\mathbf F_{4j}.
$$

C1 and C4 are not independent extra states or measured absolute eye positions. The shared optical model yields the center estimates

$$
\widehat{\mathbf C}_{1i}=\mathbf c_{1i}-g_i\boldsymbol\mu_{1i},\qquad
\widehat{\mathbf C}_{4i}=\mathbf c_{4i}-g_i\boldsymbol\mu_{4i}.
$$

With the required **P4 minus P1** convention, define

$$
\boxed{\mathbf D(\theta,A)=\frac{\mathbf C_4-\mathbf C_1}{g},\qquad
\widehat{\mathbf D}^{obs}_i=
\frac{\mathbf c_{4i}-\mathbf c_{1i}}{g_i}-\boldsymbol\mu_{4i}+\boldsymbol\mu_{1i}.}
$$

Thus the total measured centroid law is derived, not independently fitted:

$$
\boxed{\mathbf h_{cent}(\theta,A)=\mathbf D(\theta,A)+\boldsymbol\mu_4(\theta,A)-\boldsymbol\mu_1(\theta).}
$$

The initial polynomial uses uncorrected centroid difference. The refined polynomial uses model-corrected center separation D. Once D is fitted, do not also fit a free total-centroid law hcent or add the distortion mean offsets twice. Corrected centers are functions of the same data/state/model; agreement with them is not independent validation.

An angular symmetry zero is not a measured pixel distortion-center coordinate. Use separate local origins for P1/P4 templates, initially estimated from their own reference patterns. Only identifiable shared center/template corrections may be refined. Do not fit unrestricted centers per frame, or simultaneously free all radial centers, templates, absolute barrel coefficients and scale conventions.

## 8. Gaze polynomial: visual convention, up to order three

Use one global forward center-separation law, with t_v=theta/(10 degrees) and a=(A-Aref)/(1 D):

$$
\boxed{\mathbf D(\theta,A)=
(\mathbf b_0+\mathbf b_A a)+(\mathbf s_0+\mathbf s_A a)t_v
+\mathbf c_2t_v^2+\mathbf c_3t_v^3.}
$$

The vector form covers both image axes; D_y is not vertical gaze. Start with dominant linear behavior, a small quadratic correction and optional cubic correction. Add A dependence to curvature only for supported residual structure. Compare degree-1, degree-2 and degree-3 nested variants with consistent weights and accommodation models. Regularize higher-order contributions in this declared visual-angle basis, not arbitrary raw coefficients after an angular shift. There is no hard coefficient hierarchy or RMS limit.

Keep the polynomial in visual gaze by default. Optical parity constraints belong to xi1/xi4, not t_v. No zero polynomial intercept is forced at visual theta=0 or either optical zero. A nonlinear forward cubic and a separately fitted inverse cubic are not generally exact inverses; the inverse polynomial is only a bootstrap/initialization tool.

If the same polynomial is evaluated in P4-local coordinates t4=xi4/(10 degrees), then t_v=t4+w4 with w4=omega4/(10 degrees). For D=sum_n c_n(A)t_v^n, the transformed coefficients must be

$$
\boxed{\widetilde{\mathbf c}_m(A)=
\sum_{n=m}^3 {n\choose m}\mathbf c_n(A)w_4^{n-m}.}
$$

Shifting the polynomial origin without this conversion changes the mapping. A cubic can produce quadratic/linear/intercept terms under a shift. Keeping the canonical polynomial in visual theta avoids repeatedly transforming it when optical-offset estimates change.

## 9. Complete relative forward model with both offsets

Use F1j(theta)=K1(theta-omega1;b1j), F4j(theta,A)=K4(theta-omega4,A;B4j(A)), and a1(theta,omega1) exactly as defined above. F1/F4 accept **visual theta**; the lower-level K1/K4 accept **local xi**. Subtract each offset exactly once.

With mu1(theta)=mean_j F1j(theta), the complete prediction is

$$
\boxed{\widehat{\mathbf y}_i=g_i
\begin{bmatrix}
\mathbf a_1(\theta_i,\omega_1)\\
\mathbf D(\theta_i,A_i)+\mathbf F_{41}(\theta_i,A_i)-\boldsymbol\mu_1(\theta_i)\\
\mathbf D(\theta_i,A_i)+\mathbf F_{42}(\theta_i,A_i)-\boldsymbol\mu_1(\theta_i)\\
\mathbf D(\theta_i,A_i)+\mathbf F_{43}(\theta_i,A_i)-\boldsymbol\mu_1(\theta_i)
\end{bmatrix},\qquad g_i=\widehat g_{P1,i}(\theta_i,\omega_1).}
$$

In fully expanded relative form, the P4 block is

$$
\boxed{\widehat{\mathbf q_{ij}-\mathbf c_{1i}}=g_i\left[
\mathbf D(\theta_i,A_i)+K_4(\theta_i-\omega_4,A_i;\mathbf B_{4j}(A_i))
-\tfrac13\sum_lK_1(\theta_i-\omega_1;\mathbf b_{1l})\right].}
$$

The P1 block is g_i times the differences between its K1 outputs. Equivalently,

$$
\widehat{\mathbf p_j-\mathbf c_1}=g(\mathbf F_{1j}-\boldsymbol\mu_1),\qquad
\widehat{\mathbf q_j-\mathbf c_1}=g[\mathbf D+\mathbf F_{4j}-\boldsymbol\mu_1].
$$

Consequently,

$$
\widehat{(\mathbf q_j-\mathbf p_j)/g}=\mathbf D+\mathbf F_{4j}-\mathbf F_{1j},\qquad
\widehat{(\mathbf c_4-\mathbf c_1)/g}=\mathbf D+\boldsymbol\mu_4-\boldsymbol\mu_1.
$$

The P1 and P4 distortions are each symmetry-centered, while the relative observation remains in one common camera-axis and visual-gaze convention. Applying both operators at the same local angle would be wrong when omega1 differs from omega4. Absolute common translation cancels; neither optical offset creates an absolute eye-position observation.

## 10. One calibration objective, not separate conflicting fits

Let R_i=L Sigma_i L^T be a fixed common weighting covariance for the ten relative coordinates, derived from a declared native-coordinate noise policy. With residual e_i=y_i-yhat_i(theta_i,A_i;Psi), use

$$
\boxed{J=\frac1{2K}\sum_{k=1}^{K}\frac1{N_k}\sum_{i\in k}\mathbf e_i^\mathsf TR_i^{-1}\mathbf e_i
+\frac1{2K}\sum_k\left[
\frac{(\bar\theta_k-\theta_k^{nom})^2}{s_\theta^2}+
\frac{(\bar A_k-A_k^{demand})^2}{s_A^2}\right]+\lambda\mathcal P(\Psi).}
$$

Psi includes the two shared angular offsets, fixed/constrained reference geometry, optical coefficients and center polynomial. The only independent frame states are theta_i,A_i. Inherited initial mean-anchor scales 0.10 degree and 0.25 D are finite weights, not allowed-motion or accuracy limits. Temporal-flatness regularization is zero. Weak optional-coefficient shrinkage must use declared dimensionless parameter scales.

The policy `p1_profile_v1` derives g from P1 only and substitutes it in the raw relative-pixel cost. This is a P1-constrained criterion, not the Gaussian maximum likelihood obtained by optimizing g over all ten coordinates. Do not detach g when differentiating theta, omega1 or other P1 parameters. A jointly fitted P4-influenced nuisance scale is not an allowed silent fallback.

P1 noise is shared by its edges, every relative P4 coordinate and the plug-in scale. The residual covariance is not automatically R. At a fixed state, if g=l*y and yhat=g*f with l*f=1, its first-order covariance is (I-f*l)R(I-f*l)^T and has rank at most nine. Use R as the declared fixed metric; do not interpret the minimized value as a calibrated chi-square, invent independent normalized errors or add diagonal floors to manufacture information. Center/scale uncertainty and cross-subset correlations must be propagated for uncertainty diagnostics [U1].

At fixed frame states and nonlinear optical parameters the D coefficients are linear, so solve them by weighted linear least squares against the complete relative objective. Shape means and g are fixed in that block. Do not fit a second center-derived-data loss as independent evidence. Radial coefficients inside a projective denominator and angular offsets are nonlinear. Reuse least-squares/variable-projection methods only for genuinely separable blocks.

## 11. Staged initialization and iterative full calibration

The reference condition is denoted Aref. The retained metadata label for the lowest demand is approximately 0.36036 D; do not silently relabel it measured A=0. A relative-accommodation origin may be defined there. The method also applies to a genuine independently specified zero-accommodation reference if such data are supplied [G1].

**Bootstrap gaze:** form the spatial centroids in each frame, then the temporal fixation means of c4x-c1x in the reference recording. Fit an inverse polynomial of degree at most three to the five nominal gaze positions, favoring its linear part. Apply it separately to each frame. Provisional transfer to other accommodation recordings supplies starts, not exact gaze. Once g is available, repeat using the mean of the framewise corrected displacement, not the ratio of two independently averaged quantities.

**P1 first:** select its own most symmetric gaze omega1, build its native baseline there, and fit its gaze response using initial states. Extend the shared P1 fit and scale calculation across all accommodation conditions, using soft gaze means while allowing individual states to change. P1's accommodation independence does not make accommodation-biased bootstrap gaze exact.

**P4 symmetry reference:** after scale correction, separately select/refine omega4 from its five gaze conditions across accommodation. Do not inherit omega1 or nominal zero. At P4 zero, explicitly evaluate the P1 map at xi1=omega4-omega1. Preserve both native baselines rather than relabeling P1's as a symmetric P4-zero template.

**Accommodation baseline, then P4 keystone:** initialize M(A) and the identifiable barrel increment near theta=omega4 across demands, initially treating deviations from that gaze as small. Then fit P4 rotation deformation across all five gazes. Revisit the reference frames with their individual estimated xi4 as soon as keystone is available; no reference fixation remains permanently fixed at xi4=0.

**Centers and gaze recalibration:** estimate all states, calculate the model-informed center differences with Section 7, and refit D. The current optical corrections provide proposals; accept them using J. Only the first bootstrap permanently uses crude centroids as its target. The final forward model uses D plus explicit distortion-mean offsets.

**Repeat the full process:** alternate frame states, P1 response/scale, P4 baseline, P4 keystone, permitted shared alignment/geometry updates and D until the full objective and parameters stabilize. Recompute scales whenever their inputs change. Global offset updates require full derivatives and domain checks; never independently reset each frame's gaze zero. If a template is literally re-referenced, apply the homogeneous baseline/operator conversion together. Do not mix reference relabeling with a physical parameter update.

Use damping, line search and joint two-state refinement when conditional updates stall. For a gaze proposal, first correct the centroid by -mu4+mu1 and use D's dominant linear inverse; then evaluate the complete cubic-capable forward model with new g at every trial theta. During calibration, conditional state updates include their contribution to **full fixation means**. Application has no nominal-frame anchor. Numerical convergence is not proof of a unique physical solution.

## 12. Cross-agreement and identifiability

After each complete calibration checkpoint, freeze all parameters, including both offsets and center/template conventions. For each scheduled frame omit each P4 point in turn, retain all P1 and the other two P4, and rerun the complete state/scale inference. Never use the all-three measured P4 mean, full-fit state, area, or an unmasked warm start. Model-predicted means over all points are permitted.

For retained P4 set I, its model-informed center is

$$
\widehat{\mathbf C}_{4,I}=\overline{\mathbf q}_I-g\overline{\mathbf F}_{4,I}.
$$

The retained mean is not the full centroid. Predict the omitted point in native relative pixels using g[D+F4j-mu1], with the state selected without that measurement. Select covariance marginals **before** whitening, not principal blocks of a full precision matrix. Changing the omitted coordinate with calibration fixed must not change starts, scale, state, branch or prediction.

Primary reporting is omitted-P4 reconstruction and same-frame agreement of the three visual-theta/A estimates. For complete triples,

$$
E_i^2=\tfrac13\sum_j\|\mathbf e_{ij}^{cross}\|^2,\qquad
G_{A,i}^2=\tfrac13\sum_{j<k}(\widehat A_{i,-j}-\widehat A_{i,-k})^2,
$$

with analogous G_theta. Average squared quantities within exposure, then equally across all twenty exposures. RMS is a readable summary, not a hard threshold. Preserve exact frame/slot identities, axes, points, tails, bounds, branches, conditioning and missing outcomes. Compare models on identical complete cohorts; absent exposures make that comparison incomplete. The three subset estimates overlap and are correlated.

An additional **P1-omission** diagnostic requires rebuilding scale from the two retained P1 points, using their centroid/reference covariance and the corresponding predicted P1 mean. Do not use the omitted P1 in a full centroid or scale. Predict it from the retained reference after solving the state. This is a separate rank-checked diagnostic, not a substitution for the primary P4 schedule or an independent extra physical truth channel.

Fixed calibration already used the tested observations, so these checks measure internal consistency. Do not force cross-errors to zero, demand monotonic cross-score improvement during optimization, or stop solely because correction loops stop moving. Report numerical convergence and cross-agreement stability separately.

Equal-radius P4 samples can make M and radial change indistinguishable; a free radial center can conceal rather than resolve that problem. Weak P1 gaze deformation can confound angular alignment, reference scaling and framewise g. Fix template/length gauges, profile offset sensitivity, and retain an effective-scale model when separate radial terms are unidentifiable. Symmetry-zero parameters and model-corrected centers are not independent physiological measurements.

Inspect the full two-state Jacobian, including offsets, optical means and the P1 scale derivative. Shared-offset/template uncertainty affects interpretation beyond a fixed-calibration local inverse. Raw G_A can decrease under a compressed A coordinate; optical cross-prediction, state agreement and accommodation-vs-gaze information must be interpreted together. No arbitrary mixed-unit sum establishes the best law.

## 13. Controlled model hierarchy and boundary

DM0 uses accommodation-dependent effective magnification; DM1 adds an identifiable radial increment; DM2 adds one justified curvature/coupling/spatial mechanism. The separate omega1/omega4 and P4-minus-P1 center-separation polynomial are part of every active candidate, not optional corrections reserved for a favored accommodation law. Hold polynomial degree/regularization and alignment policy fixed when comparing accommodation mechanisms; test degree changes separately.

No extra physical Z state, free P4 scale, mandatory area ratio, per-frame center/alignment, or automatic restoration of legacy coefficients is introduced. Each candidate gets a fresh full calibration with natural within-fixation state variation. A defensible effective accommodation response is useful even when its absolute barrel coefficient is not uniquely identified. Independent physical references would be needed for physiological accuracy claims.

This specification supersedes the old single-theta_opt, nominal-zero reference initialization and permanently centroid-based gaze polynomial for this branch. It does not modify data or assert that any new optimizer, GPU run or calibration has been completed.

## Sources

- [D1: Optical distortion theory, pinned](https://github.com/rueijrwu/distortion_tracking/blob/1d2a0874c79f3f174b36a197c3fbbffd8a32fe60/Theory.md).
- [D2: Optical summary, pinned](https://github.com/rueijrwu/distortion_tracking/blob/1d2a0874c79f3f174b36a197c3fbbffd8a32fe60/Summary.md).
- [D3: P4 axial magnification report](https://github.com/rueijrwu/distortion_tracking/blob/1d2a0874c79f3f174b36a197c3fbbffd8a32fe60/data/p4_z_magnification/p4_z_magnification.md).
- [G1: Reviewed fixation metadata](../data/fixations/fixation_intervals.json).
- [U1: NIST, law of propagation of uncertainty](https://physics.nist.gov/cuu/Uncertainty/combination.html).
