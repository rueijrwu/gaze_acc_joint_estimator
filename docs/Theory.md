# Joint distortion theory: P1 reference scale, dual optical zeros, and shared gaze/accommodation

**Branch:** `rueijrwu/gaze_acc_joint_estimator / exp5_distortion_model`.  
**Revision:** 2026-10-09; reconciled against `38029d7eb52cb95a8e62f0f812bf513d42e520aa`.  
**Status:** Authoritative proposed mathematics, not an implemented model or a new calibration result.  
**Companion:** [ESTIMATOR_PLAN.md](ESTIMATOR_PLAN.md). These two files define this branch's active workstream.  
**Optical source:** `distortion_tracking` at `1d2a0874c79f3f174b36a197c3fbbffd8a32fe60` [D1-D3].

## 0. Objective, current state, and non-negotiable decisions

Find an accommodation model for which three corresponding P1/P4 measurements support **one visual horizontal gaze and one accommodation value in each frame**. The objective is not a marginal improvement in P4 RMS alone. The model must explain relative displacement, deformation, and the useful accommodation information after gaze and nuisance magnification are accounted for.

The inspected branch contains reviewed data and documentation, but no executable `distortion_model/`, historical `full_position/`, or test package. Inherited status/experiment documents describe historical results; their path references do not establish that those artifacts exist in this branch. The prior log/power outcomes are motivation, not results of this optical model. No stopped experiment is resumed by this specification.

The agreed design is:

| Item | Active decision |
|---|---|
| Measurement sign | P4 minus P1 |
| Observations | Same-frame relative coordinates; no absolute eye-position observation |
| Frame states | Visual horizontal gaze `theta_i` and accommodation `A_i`, both free within fixation |
| Optical references | Independent shared P1 and P4 symmetry zeros `omega1`, `omega4` |
| Nuisance magnification | One positive P1-reference scale; the same scalar multiplies both reflections |
| Area normalization | Neither square-root P1 area nor P4/P1 area ratio is required |
| Gaze model | Centroid bootstrap, then a forward model of distortion-corrected P4-P1 center separation |
| Accommodation model | DM0 effective magnification first; DM1 adds an identifiable radial increment; DM2 adds one supported mechanism |
| Calibration | Fresh full calibration of each candidate on all twenty reviewed conditions |
| Evaluation | Entire iterative inference repeated under rotating P4 omissions; internal cross-agreement |
| Requirements | Numerical/data integrity, not a hard RMS, nominal-target, or fixation-flatness cutoff |

Do not add a free physical Z state, another P4 nuisance scale, a differential-scale coefficient, independent framewise optical centers, or framewise exponents. Do not replace the three individual point pairs with one area or size channel. P4 accommodation-dependent magnification must remain signal.

## 1. Relative measurement model

Let `i` index frames and `j=1,2,3` index fixed source correspondences. Measured camera coordinates are

$$
\mathbf p_{ij}=P1_{ij},\qquad \mathbf q_{ij}=P4_{i,\pi(j)}.
$$

Read the source permutation and flags from the payload; the inherited zero-based map is `[2,1,0]`. Never infer source identity by independently sorting two distorted patterns. The source pattern is distinct from the five visual gaze targets.

Define

$$
\mathbf c_{1i}=\frac13\sum_j\mathbf p_{ij},\quad
\mathbf c_{4i}=\frac13\sum_j\mathbf q_{ij},\quad
\mathbf e_{1i}=\begin{bmatrix}\mathbf p_{i2}-\mathbf p_{i1}\\\mathbf p_{i3}-\mathbf p_{i1}\end{bmatrix}.
$$

The active observation is the ten-component linear vector

$$
\boxed{\mathbf y_i=\begin{bmatrix}
\mathbf e_{1i}\\
\mathbf q_{i1}-\mathbf c_{1i}\\
\mathbf q_{i2}-\mathbf c_{1i}\\
\mathbf q_{i3}-\mathbf c_{1i}
\end{bmatrix}=L\mathbf z_i,}
$$

where `z_i` stacks the twelve native coordinates. `rank(L)=10`; adding any common same-frame translation to all six points leaves `y_i` unchanged. A zero-gaze camera position measured earlier is not a replacement for the same-frame reference.

Separately centered shapes may be displayed, but retain `c4-c1`: discarding it removes the dominant initial gaze signal. Do not count centroids, all edges, area, and the original coordinates as independent measurements. Triangle-area quantities remain optional historical diagnostics, not inputs to the active scale or accommodation estimator.

Use measured pixels and the same declared reference-pixel unit in templates, so the scale below is dimensionless. An optical coefficient in inverse millimeters must be explicitly converted before use. Scale-corrected coordinates retain reference-length units; they are not intrinsically dimensionless.

Relative subtraction removes common additive image translation exactly. Physical eye translation may additionally change scale, field sampling, or P1/P4 differently. The common-scale model is a declared approximation for those effects, not a claim of invariance to every physical translation.

## 2. Visual gaze, accommodation, and separate optical zeros

The only independently optimized frame states are

$$
\boxed{x_i=(\theta_i,A_i).}
$$

Theta is visual horizontal gaze in degrees. Nominal targets are `[-10,-5,0,5,10]` degrees; nominal vertical gaze is zero, but image-y remains informative and is not set to zero. A is accommodation in diopters. The lowest demand label is approximately 0.36036 D, not measured physiological zero. Read all labels from [the intervals](../data/fixations/fixation_intervals.json), without relabeling them [G1].

Fixation means supply finite anchors. Individual theta and A values may vary throughout a fixation. The reference accommodation `A_ref` is a declared baseline convention, initialized from the low-demand recording, with actual frame states still free.

Let omega1 and omega4 be the visual gazes at each reflection's operational symmetry zero. They are independent shared calibration parameters, initially constant across captures and accommodation:

$$
\boxed{\xi_1=\theta-\omega_1,\qquad \xi_4=\theta-\omega_4,\qquad
\Delta_{14}=\omega_4-\omega_1.}
$$

Consequently,

$$
\boxed{\xi_1=\xi_4+\Delta_{14},\qquad \theta=\xi_1+\omega_1=\xi_4+\omega_4.}
$$

There is still only one gaze per frame. Neither omega need be zero, and they need not equal each other. At P4's optical zero, P1 is evaluated at `xi1=omega4-omega1`, not at zero. Visual target labels never move when these offsets change. Do not impose a near-zero offset prior merely because the target grid includes visual zero.

Initialize each zero independently from many frames across the five gaze conditions, using symmetry appropriate to the actual illuminator arrangement. A known mirrored pairing and a fixed image axis may define a score. Do not assume an equilateral triangle, or use a free per-frame affine fit to make all triangles symmetric. Symmetry-only rescaling is a diagnostic, not P4 normalization for inference.

The best sampled condition initializes an offset; it does not set every frame in that fixation to its optical zero. Preserve broad minima, ties, endpoint choices, and profiles. If the source arrangement or data do not identify a continuous zero, fix a declared operational reference or retain alternatives and report the limitation. Do not manufacture anatomical-axis certainty from a three-point pattern. A reproducible accommodation-dependent P4 zero is a separately declared extension, not permission for a zero per frame/capture.

## 3. Optical evidence and scope of the shared scale

The pinned optical source separates existing baseline distortion, rotation deformation, accommodation, and axial magnification. P1's reported Z slope is 0.00295165270232 mm^-1. Separately collected P4 slopes range from 0.00295232758 to 0.00294991207 mm^-1 across 0-4 D. The studies checked five gaze angles. This supports starting with nearly common fractional image scaling [D1-D3].

The active model does not estimate Z or use those slopes to convert image scale into physical distance. It estimates the scale from P1 itself. P1 has no direct accommodation response in the first model; this is a premise, not a P1 accommodation-sweep result. P4 has distinct baseline and rotation coefficients, even though nuisance magnification is shared.

The P4 Z study and the older P4 accommodation/rotation study have different lens revisions. Their coefficients are not an already matched full theta/A/Z calibration. Center-relative grids also do not establish the Z dependence of the relative optical-center separation. Preserve those qualifications and inspect residual dependence on P1 scale; do not silently add another scale to improve fit. The approximately 1-micrometer simulation simplification reference is not a hardware-accuracy or estimator RMS gate.

## 4. Complete optical transformations

### 4.1 Keystone about the native zero

In declared fixed optical image axes, use

$$
K_r(\xi_r,A;\mathbf b)=\frac{1}{1+q_r(\xi_r,A)b_y}
\begin{bmatrix}s_{x,r}(\xi_r,A)b_x\\s_{y,r}(\xi_r,A)b_y\end{bmatrix},
\quad
H_r=\begin{bmatrix}s_{x,r}&0&0\\0&s_{y,r}&0\\0&q_r&1\end{bmatrix}.
$$

The initial functions are

$$
s_{x,r}=1+\alpha_r\xi_r^2,\qquad
s_{y,r}=1+\beta_r\xi_r^2,\qquad q_r=\gamma_r\xi_r.
$$

K1 has no A dependence. Initially the P4 rotation coefficients also have no explicit A dependence; its accommodation baseline still couples A into the projective denominator. Preserve `K1(0)=I` and `K4(0,A)=I` at their own native zeros. Neither must be identity at visual theta=0.

If needed, conjugate the mapping by a fixed calibrated orthonormal image-axis matrix `Q_r`. A camera-axis rotation is not the angular offset omega and must not be freely fitted per frame. With xi in degrees and length unit L, alpha/beta have degree^-2, gamma has L^-1 degree^-1, and q has L^-1. Record any numerical angle scaling.

Require positive scales and a projective denominator bounded away from zero on the actual local-angle/field domain. Shifted local domains must be checked even when visual theta remains in its bounds. These are model/numerical validity conditions, not image-error thresholds. Missing spatial deformation cannot be fixed merely by increasing polynomial order in xi [D1].

### 4.2 P1 baseline and transformation

Let `b1j` be a fixed or explicitly constrained empirical real P1 template at theta=omega1 and reference scale:

$$
\boxed{\mathbf F_{1j}(\theta)=K_1(\theta-\omega_1;\mathbf b_{1j}).}
$$

It already contains reference barrel distortion. Do not apply that distortion twice or introduce a P1 radial coefficient in every frame. The nuisance scale acts after keystone, `S(g)H1`, not `H1S(g)`.

A centroid can initialize a local template origin, but is not an independent distortion-center measurement. Hold template length/origin gauges fixed initially. Weak P1 deformation cannot identify unrestricted gaze-dependent isotropic scaling, all framewise scales, and arbitrary template changes simultaneously.

### 4.3 P4 baseline: two distinct conventions

With an independently known/constrained paraxial template `u4j`, an absolute radial model is

$$
\mathbf z_j(A)=M(A)\mathbf u_{4j},\qquad
\mathbf B_{4j}(A)=[1+\kappa_4(A)\|\mathbf z_j(A)\|^2]\mathbf z_j(A),
\qquad M(A_{ref})=1.
$$

The default recording-based model instead uses the real reference template `b4ref` at theta=omega4 and Aref:

$$
\boxed{\mathbf B_{4j}^{rel}(A)=M(A)
[1+\Delta\kappa(A)M(A)^2\|\mathbf b_{4j}^{ref}\|^2]\mathbf b_{4j}^{ref},}
$$

$$
M(A_{ref})=1,\qquad \Delta\kappa(A_{ref})=0.
$$

This is an **empirical incremental** deformation. It preserves the already distorted reference at Aref. Delta-kappa is not automatically the difference between two physical absolute barrel coefficients. If a reference radial map is known and invertible, the alternative `R_A(R_Aref^-1(b4ref))` has a different, explicit physical interpretation. Never label an empirical distorted template as paraxial merely to fit an absolute kappa.

For either baseline convention,

$$
\boxed{\mathbf F_{4j}(\theta,A)=K_4(\theta-\omega_4,A;\mathbf B_{4j}(A)).}
$$

The order is accommodation magnification, radial baseline deformation, gaze keystone, and finally the same external scale g as P1. M is accommodation signal; g is nuisance scaling. A radial coefficient has reference-length^-2 units and uses the pre-radial radius. Inspect `1+kappa*r^2` and `1+3*kappa*r^2` for a nonfolding absolute radial map; apply the corresponding domain check to an incremental map.

For example,

$$
\partial_A\mathbf F_4=(\partial_{\mathbf b}K_4)\partial_A\mathbf B_4
+(\partial_AK_4)_{\mathbf b}.
$$

Do not omit the first term merely because the keystone coefficients initially lack explicit A dependence.

### 4.4 The first model comparison

Let `a=(A-Aref)/(1 D)`. The initial mechanisms are:

| Candidate | M(A) | Delta-kappa(A) | Purpose |
|---|---|---|---|
| DM0-M1 | `1+m1*a` | 0 | Smallest accommodation-dependent effective-scale model |
| DM1-M1K1 | `1+m1*a` | `kA*a` | Add a distinguishable relative radial response |
| DM2 | One declared extra mechanism | Only the specified change | Test a supported curvature, coupling, or missing spatial component |

Both DM0 and DM1 retain **both optical zeros, both keystone operators, the P1 scale, and the same center-separation gaze model**. DM0 is not a distortion-free eye model: its empirical reference patterns retain baseline distortion and its gaze operators remain active.

If `||b4ref_j||=R` for every point, then

$$
\mathbf B_{4j}^{rel}(A)=M(A)[1+\Delta\kappa(A)M(A)^2R^2]\mathbf b_{4j}^{ref}.
$$

Only the effective scale is visible from that radial baseline. Applying the same deterministic keystone cannot distinguish two parameter sets that give identical baselines. Near-equal radii can make separation weak. Check the global and state-conditioned response rank, allowing for center displacement and other fitted parameters, before interpreting kA. More frames or flexible centers do not automatically remove this structural ambiguity.

Obtain a complete DM0 fit and cross-agreement result first. Attempt DM1 with declared identical controls, reporting whether its added mechanism is determined. Failure to separate radial strength is a valid reason to retain effective magnification, not a failure to measure accommodation at all. Fit each candidate afresh; do not relabel previous coefficients. Log/power/saturating shapes are later functions of these mechanisms, not an open-ended exponent sweep on all coordinates.

### 4.5 Native-reference transformations

Usually retain both native templates and evaluate `xi1=xi4+Delta14`. If P1 is re-expressed at P4's zero, transform both baseline and operator:

$$
\mathbf b_1^{(4)}=\operatorname{dehom}\{H_1(\Delta_{14})[\mathbf b_1;1]\},\qquad
H_{1|4}(\xi_4)=H_1(\xi_4+\Delta_{14})H_1(\Delta_{14})^{-1}.
$$

This preserves predictions and makes H1|4 identity at the new reference, but does not make P1 physically symmetric there. Native parity about omega1 need not survive as parity about omega4. At fixed A, same-reflection transfer is `H(xi_b,A)H(xi_a,A)^-1`; changing A also requires the baseline transformation. A reference re-expression is not a physical offset update; actual changes of omega must be evaluated against the calibration objective.

## 5. P1-only reference scale

At trial visual theta construct

$$
\mathbf a_1(\theta)=\begin{bmatrix}
\mathbf F_{12}(\theta)-\mathbf F_{11}(\theta)\\
\mathbf F_{13}(\theta)-\mathbf F_{11}(\theta)
\end{bmatrix}.
$$

For fixed positive-definite P1-edge covariance `R11`, let `W1=R11^-1`. The adopted policy `p1_profile_v1` is

$$
\boxed{g_i(\theta)=\frac{\mathbf a_1^T W_1\mathbf e_{1i}}{\mathbf a_1^T W_1\mathbf a_1}.}
$$

Use linear solves in code. Require finite positive g and nonzero weighted template energy; mark invalid proposals instead of clipping or substituting another model's scale. The two edges share one point, so their covariance is not generally diagonal. Changing edge origin with the properly transformed covariance must preserve g.

At fixed visual theta, P1 observations, and P1 parameters, neither A nor a P4 coordinate directly changes g. Recompute g for every trial theta and relevant P1/global update. With fixed weights/template,

$$
\partial_\theta g=\frac{\mathbf a_{1,\theta}^T W_1\mathbf e_1
-2g\mathbf a_{1,\theta}^T W_1\mathbf a_1}{\mathbf a_1^T W_1\mathbf a_1},\qquad
\partial_{\omega_1}g=-\partial_\theta g.
$$

Include additional template/weight dependencies when allowed. At fixed visual theta omega4 does not enter g. At fixed xi4, however, theta=xi4+omega4 adds its chain derivative.

Display corrected coordinates as `(p-c1)/g` and `(q-c1)/g`; do not divide by g again or normalize P4 by its own size. Estimate g relative to the P1 template at the current gaze, not an unchanged zero-gaze triangle. These operations need no distortion-center measurement, triangle area, or physical Z estimate.

This normalization does not create information: it fixes the common length gauge using P1. If P1 reference length is changed, the entire optical/displacement parameterization must transform consistently. Freeing isotropic P1 gaze scaling without constraining that gauge can make g and the optical parameters compensate.

## 6. Corrected centers and the forward gaze polynomial

For algebra only write `p_j=C1+g*F1_j` and `q_j=C4+g*F4_j`. Let `mu_r=mean_j(F_rj)`. The model-informed origins are

$$
\widehat{\mathbf C}_1=\mathbf c_1-g\boldsymbol\mu_1,\qquad
\widehat{\mathbf C}_4=\mathbf c_4-g\boldsymbol\mu_4.
$$

They are not independent frame states or measurements of absolute eye position. The observable model-corrected separation is

$$
\boxed{\mathbf D^{obs}=\frac{\mathbf c_4-\mathbf c_1}{g}-\boldsymbol\mu_4+\boldsymbol\mu_1,\qquad
\mathbf D(\theta,A)=\frac{\mathbf C_4-\mathbf C_1}{g}.}
$$

Therefore the total centroid law is **derived**:

$$
\boxed{\mathbf h_{cent}(\theta,A)=\mathbf D(\theta,A)+\boldsymbol\mu_4(\theta,A)-\boldsymbol\mu_1(\theta).}
$$

The centroid bootstrap is only an initializer. The final forward model fits D and includes optical mean offsets once. Do not also fit an independent h_cent, or add Dobs as a separate observation on top of the point residuals. Corrected centers use the same model and measurements and are not additional calibration truth.

With `t=theta/(10 degrees)` use the cubic-capable family

$$
\boxed{\mathbf D(\theta,A)=(\mathbf b_0+\mathbf b_A a)
+(\mathbf s_0+\mathbf s_A a)t+\mathbf c_2t^2+\mathbf c_3t^3.}
$$

Use degree 1, 2, or 3 as a declared control, shrinking unnecessary curvature in the visual-angle basis. Both image axes are included; Dy is not vertical gaze. No zero intercept is imposed at either optical zero or visual zero. Initial curvature terms lack A dependence. Keep degree and regularization policy identical when comparing accommodation mechanisms; test degree changes as a separate paired ablation. A low training residual alone does not justify another term.

A forward cubic and an independently fitted inverse cubic are not exact inverses. At fixed A, if evaluating D in P4-local coordinates `t4=xi4/10`, then `t=t4+w4`, `w4=omega4/10`, and

$$
\widetilde{\mathbf c}_m(A)=\sum_{n=m}^{3}{n\choose m}\mathbf c_n(A)w_4^{n-m}.
$$

Keep D in visual coordinates by default. Shifting its input without coefficient conversion changes the model. Optical parity applies to xi1/xi4, not automatically to this polynomial.

## 7. Single joint relative prediction

Public optical functions accept visual theta and subtract their own offsets exactly once. Low-level K functions accept local xi. The complete prediction is

$$
\boxed{\widehat{\mathbf y}_i=g_i(\theta_i)\begin{bmatrix}
\mathbf a_1(\theta_i)\\
\mathbf D(\theta_i,A_i)+\mathbf F_{41}(\theta_i,A_i)-\boldsymbol\mu_1(\theta_i)\\
\mathbf D(\theta_i,A_i)+\mathbf F_{42}(\theta_i,A_i)-\boldsymbol\mu_1(\theta_i)\\
\mathbf D(\theta_i,A_i)+\mathbf F_{43}(\theta_i,A_i)-\boldsymbol\mu_1(\theta_i)
\end{bmatrix}.}
$$

Equivalently,

$$
\widehat{\mathbf p_j-\mathbf c_1}=g(\mathbf F_{1j}-\boldsymbol\mu_1),\qquad
\boxed{\widehat{\mathbf q_j-\mathbf c_1}=g[\mathbf D+\mathbf F_{4j}-\boldsymbol\mu_1].}
$$

This preserves both independently centered distortion responses, shared scale, and relative center separation. The predicted pair displacement is `g*(D+F4_j-F1_j)`. The predicted centroid difference is `g*(D+mu4-mu1)`. Absolute common translation cancels without discarding the gaze signal.

At P4 zero, F4 equals its accommodation baseline while F1 generally remains deformed. No use of the same local angle in both operators is permitted unless the fitted offsets actually coincide.

## 8. One calibration objective and its statistical interpretation

Derive fixed `R_i=L*Sigma_i*L^T` from a declared native-coordinate weighting policy. With `e_i=y_i-yhat_i`, K exposures and Nk valid frames in exposure k, use

$$
\boxed{J=\frac1{2K}\sum_k\frac1{N_k}\sum_{i\in k}e_i^T R_i^{-1}e_i
+\frac1{2K}\sum_k\left[
\frac{(\bar\theta_k-\theta_k^{nom})^2}{s_\theta^2}
+\frac{(\bar A_k-A_k^{demand})^2}{s_A^2}\right]
+\mathcal P(\Psi).}
$$

Psi includes shared offsets, optical coefficients, center polynomial, and only the permitted constrained geometry. Initial `s_theta=0.10 degree` and `s_A=0.25 D` are inherited finite mean-penalty scales, not allowed errors or motion limits. Framewise states remain free; temporal-flatness regularization is zero. Priors use declared dimensionless parameter scales, with fixed/free gauges recorded. Do not assign an unsupported prior that forces both optical zeros near visual zero.

The scale is substituted from P1, not optimized using all points. This is a **P1-constrained fixed-metric criterion**, not unconstrained Gaussian maximum likelihood over an additional g variable. Differentiate through g. At fixed state/calibration, g is linear in y: write `g=ell*y`, `yhat=g*f`, `ell*f=1`. Then the plug-in residual map is `P=I-f*ell`, with

$$
\operatorname{Cov}(e)=P R P^T,\qquad \operatorname{rank}(P)=9
$$

for a regular ten-coordinate model. R is the declared weighting metric, not automatically the residual's covariance. Do not invert the singular plug-in covariance as though it were full rank or attach a calibrated chi-square interpretation. Retain P1/P4 cross-correlations in uncertainty diagnostics [U1]. Noise estimates based on temporal differences can contain real movement; do not reinterpret them as independently measured hardware localization noise.

At fixed frame states and nonlinear optical parameters, D's coefficients are linear. Solve that block by weighted QR/SVD against the **full relative residual and its cross-covariance**, not a second independent fit to Dobs. Radial parameters inside keystone and offsets are nonlinear. Profile only genuinely linear blocks, with proper derivatives. Reuse established bounded least-squares methods and verified GPU ideas; the old 27-column design is not this model.

Do not mix raw pixel, degree, and diopter disagreement terms into an arbitrary extra calibration cost. The same-frame coordinate model already enforces a shared state. A changed covariance, robust loss, or direct cross-agreement training objective is a separately specified experiment, not a silent response to imperfect results.

## 9. Staged bootstrap, then iterative full calibration

The following order uses all conditions in the final fit, with no condition holdout:

1. **Centroid bootstrap:** in the low-demand recording, spatially average P1/P4 in each frame, then temporally average P4-minus-P1 in the five fixations. Fit a primarily linear inverse, with supported curvature up to cubic, and initialize each frame individually. This is not the final forward D polynomial.
2. **P1 first:** find P1's own symmetry reference, preserve its empirical baseline distortion, fit its native-angle gaze map, and derive g across all accommodation recordings. Provisional gaze transferred from the low-demand recording is a start, not truth.
3. **P4 zero:** independently estimate its symmetry reference after P1 scale correction. Keep visual theta canonical and convert P1 with Delta14; do not relabel P1's native baseline.
4. **Accommodation baseline:** use near-P4-zero frames across demands to initialize DM0, or the identifiable incremental baseline of DM1. A starts near the demand labels but remains free with soft means.
5. **P4 keystone:** fit gaze deformation across the five gaze conditions, then revisit the reference-angle frames using their own xi4. No entire reference fixation remains fixed at xi4=0.
6. **Correct centers:** infer frame states, calculate Dobs, and solve the global forward D block against the same J.
7. **Iterate:** alternate frame states and allowed shared parameters, recomputing scales and optical means at every relevant update. Accept/damp steps using the one full objective, not unrelated per-stage residuals.

After scale initialization, use the temporal mean of **framewise** corrected displacement. It is not generally the ratio of separately averaged displacement and scale. Likewise, a nonlinear inverse of a mean is not the mean of the framewise inverse.

For fixed shared parameters a frame update is conceptually

$$
\theta^{(t)}\rightarrow g_{P1}(\theta^{(t)})\rightarrow A^{(t+1)}
\rightarrow\theta^{(t+1)}\rightarrow g_{P1}(\theta^{(t+1)}).
$$

Use conditional minimizations of J during calibration, including changes to the **full fixation means**. For a gaze proposal, subtract `mu4-mu1` from `(c4-c1)/g` and invert D's dominant linear term. Refine with the complete cubic-capable forward model, re-evaluating g and both local angles for every trial. Do not freeze normalized observations or inverse-keystone about an assumed measured centroid-center.

Global updates include P1 distortion/permitted omega1, P4 baseline, P4 keystone/permitted omega4, and the center polynomial. Maintain one consistent snapshot during a trial; update derived quantities whenever dependencies change. A true reference re-expression preserves predictions; an actual offset change is accepted against J. Templates, scales, centers, and offset gauges cannot all drift freely.

Use damping/line search and joint two-state refinement as needed. Numerical stationarity and valid local domains determine convergence; cross-agreement stability is reported separately. Neither a fixed number of iterations nor flat trajectories establishes a solution. Cross-agreement need not decrease monotonically with the training objective.

## 10. Cross-check the entire inference, not only its final prediction

Freeze every global parameter, symmetry/template convention, and weighting policy at a completed checkpoint. Omit each P4 point in turn and rerun the whole state/scale solve from **retained-only** inputs. Use all three P1 points plus the other two P4 points. Select the eight-coordinate covariance marginal before whitening; a block of full precision is not equivalent.

The omitted point must not enter through the all-three measured P4 centroid, area, state, scale-dependent correction, start, gate, weight, or branch selection. Means of three predicted optical points are allowed. At fixed calibration, changing only the omitted measurement must leave the inferred states, g, branches, and prediction unchanged.

For a retained P4 set I, the corrected center is `mean_I(q)-g*mean_I(F4)`. Its mean is not the full centroid. Reconstruct the omitted point and score in native relative pixels:

$$
\boxed{e_{ij}^{cross}=(\mathbf q_{ij}-\mathbf c_{1i})
-g_{i,-j}[\mathbf D(\widehat x_{i,-j})+\mathbf F_{4j}(\widehat x_{i,-j})-\boldsymbol\mu_1(\widehat\theta_{i,-j})].}
$$

Do not divide the reporting error by a candidate's own scale to manufacture improvement. The three subset estimates are compared in **visual theta**, not xi1 versus xi4.

For complete eligible triples,

$$
E_i^2=\frac13\sum_j\|e_{ij}^{cross}\|^2,\qquad
G_{A,i}^2=\frac13\sum_{j<k}(\widehat A_{i,-j}-\widehat A_{i,-k})^2,
$$

with the analogous G_theta. Average squared values within exposure and equally across exposures. RMS is a display summary, not a hard requirement. Vector-error RMS differs by sqrt(2) from scalar-coordinate RMSE on the same 2D population.

Keep exact identities, missing outcomes, partial-slot summaries, signed axes, each P4, tails, worst point, bounds, rank, and branches. Expected populations and all three slot IDs remain visible even when calibration fails. An absent exposure makes a paired all-exposure comparison incomplete. Overlapping subsets are correlated; uncertainty of their differences must include cross-covariances.

The coefficients were fit using this calibration data. These tests measure **internal consistency**, not independent physiological accuracy. Full-fit residuals, subset cross-reconstruction, nominal-mean discrepancy, and temporal variation are different outputs. Agreement need not be zero.

Optional P1 omission is separate: rebuild the reference from the two retained P1 points, recompute one-edge scale, covariance, and the predicted retained-P1 mean, then test rank and branches. Never reuse a full P1 centroid/scale containing the omitted P1. This is not required before the first DM0 result and is not pooled with the P4 metrics.

## 11. What constitutes a better accommodation model

The primary evidence is a **cross-agreement scorecard**, not a single mixed-unit cost or a universal winner rule. On identical frames report P4 reconstruction, G_theta, G_A, coverage, tails, and optical response structure. A lower E with higher G is a tradeoff; a model need not improve every exposure. Identify concentrated gains/failures without dropping difficult conditions.

Check whether accommodation can be distinguished from gaze. At fixed calibration use full derivatives through g, both offsets, optical means, and baselines. For a regular appropriately noise-propagated two-state linearization,

$$
I_{A\mid\theta}=F_{AA}-F_{A\theta}^2/F_{\theta\theta},\qquad F=J_x^T R^{-1}J_x,
$$

where the chosen representation/weighting interpretation must be stated. With the plug-in P1 scale, R of the raw observation is not automatically the postfit residual covariance; use the composite estimator and nonredundant uncertainty propagation for precision claims. Numerical rank and perturbation checks remain necessary. Do not set an arbitrary physical-information threshold from the current sample.

Raw G_A and information per diopter change under latent-A rescaling. Inspect scale compression, clipping, anchor/prior/bound sensitivity, alternative offset solutions, and conditional M/radial rank. More pixels or a smaller residual do not by themselves identify an absolute optical center or a physical barrel coefficient. A useful effective accommodation response remains meaningful even when its radial decomposition is unresolved.

Start with the complete DM0 result, then compare a freshly fitted DM1 under the same data, degree, anchors, covariance, starts, and reference policy. DM2 requires a specific reproducible deficiency and one explicit change. Do not respond to a missing spatial term or scale-reference issue with an increasingly large global accommodation exponent. Absolute physiological accuracy requires independent references or additional justified optical constraints.

## 12. Mathematical contracts and evidence boundary

Implementation must test translation invariance; ten-coordinate rank; P1-only positive scale and its derivatives; edge-origin/covariance equivalence; dual zeros and visual/local conversions; homography and cubic reference changes; center-versus-mean signs; radial baseline identity; transformation order; equal-radius degeneracy; common-scale versus P4-only scale behavior; dynamic fixation means; correlated noise; complete masked-inference noninterference; and immutable scheduled frame/slot accounting.

Tests validate mathematics and software, not the optical truth of the approximation. Actual implementation, test runs, data counts, and completed calibrations must be reported separately. This revision changes only the specification; source data and other branches remain untouched.

## Sources

- [D1: Optical Theory.md, pinned](https://github.com/rueijrwu/distortion_tracking/blob/1d2a0874c79f3f174b36a197c3fbbffd8a32fe60/Theory.md).
- [D2: Optical Summary.md, pinned](https://github.com/rueijrwu/distortion_tracking/blob/1d2a0874c79f3f174b36a197c3fbbffd8a32fe60/Summary.md).
- [D3: P4 Z magnification report, pinned](https://github.com/rueijrwu/distortion_tracking/blob/1d2a0874c79f3f174b36a197c3fbbffd8a32fe60/data/p4_z_magnification/p4_z_magnification.md).
- [G1: Reviewed interval metadata](../data/fixations/fixation_intervals.json).
- [Previous branch specification](https://github.com/rueijrwu/gaze_acc_joint_estimator/blob/38029d7eb52cb95a8e62f0f812bf513d42e520aa/docs/Theory.md): design continuity, not a completed implementation.
- [U1: NIST uncertainty propagation](https://physics.nist.gov/cuu/Uncertainty/combination.html): covariance propagation, not an optical validation.
