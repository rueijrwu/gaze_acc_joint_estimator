# Joint relative P1/P4 optical model for gaze and accommodation

**Status:** Proposed replacement theory for the next optical-model calibration; not a claim that the existing estimator implements or validates it.  
**Repository / branch:** `rueijrwu/gaze_acc_joint_estimator` / `exp5_full`.  
**Revision date:** 2026-10-09.  
**Estimator snapshot inspected:** `459041d82e8b4743b7da4e3788e184e2c9e1f8ff`.  
**Optical source:** `rueijrwu/distortion_tracking`, `1d2a0874c79f3f174b36a197c3fbbffd8a32fe60`, including the new P4 Z magnification study [D1-D4].  
**Revision focus:** both P1 and P4 have approximately linear axial magnification. Keep common-scale normalization as the first approximation, with an explicit differential-scale correction and a separately calibrated relative-centroid law.

## 0. Objective and observation contract

**Calibrate one shared optical transformation that explains both P1 and P4, with an independently varying horizontal gaze and accommodation in every frame. Use relative displacement for the dominant gaze signal, P1 geometry for nuisance-scale correction, and the corrected P4 magnification/deformation for accommodation.**

The observations are exclusively **same-frame relative geometry**. Absolute image positions are retained only as the input from which differences are formed; neither an absolute eye position nor an absolute distortion-center trajectory is measured or fitted as a physiological output. Subtracting a fixed zero-gaze camera position would not remove subsequent eye/image translation. Subtracting a reference measured in the **same frame** removes the common additive component.

The intended iteration is

$$
\boxed{\theta^{(t)}\ \longrightarrow\ g_{P1}^{(t)}\ \longrightarrow\ A^{(t+1)}
\ \longrightarrow\ \theta^{(t+1)}\ \longrightarrow\ g_{P1}^{(t+1)}\ \longrightarrow\cdots.}
$$

During full calibration, the shared optical parameters are updated as well. During application and cross-checks they are fixed. Every new accommodation-law candidate receives a fresh full calibration; previous coefficients are not reinterpreted under a different law.

This theory replaces the earlier emphasis on empirical `D(theta,A), T(theta,A)` coefficient searches. It preserves three-pair correspondence, P1-area normalization, dynamic fixation states, correlated-noise treatment, and internal cross-agreement. The historical conditional27/37 and frozen two-channel estimators remain unchanged controls, not implementations of the new joint model. Existing plans must be reconciled with this theory before implementation; rewriting this document does not restart any stopped experiment. See [CURRENT_STATUS.md](CURRENT_STATUS.md) and [EXPERIMENTS.md](EXPERIMENTS.md) for retained evidence.

## 1. Optical evidence and assumptions

The optical source separates a real, already distorted reference pattern from subsequent rotation, accommodation, and axial magnification. **P4 now has its own Z sweep:** approximately linear axial magnification is no longer inferred from P1 alone. This supports near-common P1/P4 pattern scaling over the tested domain, not exact equality or a validated experimental cancellation [D1-D3].

| Element | Interpretation adopted here | Evidence boundary |
|---|---|---|
| P1 baseline | Fixed real pattern including reference barrel distortion | Do not apply the same radial correction again |
| P1 rotation | Small but nonzero deformation `K1(theta)` | Weak sensitivity is not exact invariance |
| P1 axial scale | `g1(Z) approximately 1+alpha1Z*Z`, after rotation | P1 scale separability checked at five angles |
| P4 baseline | Accommodation-dependent `M(A)` and radial distortion | Distinct from magnification caused by changing Z |
| P4 rotation | Initially a separate `K4(theta)` | Its shared accommodation dependence still needs calibration |
| P4 axial scale | `g4(A,Z) approximately 1+alpha4Z(A)*Z`, after the same-state Z=0 pattern | New P4 sweep: A=0,1,2,3,4 D; five angles; Z=-5 to +5 mm |
| Near-common P1/P4 scale | `g4/g1 approximately 1` is the first pattern-normalization approximation | Similar separately fitted slopes; not a matched end-to-end proof |
| Relative centroid displacement | Calibrated independently of centered-shape scaling | Center-relative Z grids do not measure the P4-P1 center displacement |
| P1 accommodation independence | No direct A dependence in its first shape model | A premise; the cited P1 sweep did not vary accommodation |

### 1.1 Linear axial magnification for both reflections

In the optical source's absolute THI coordinate Z, with each scale normalized at Z=0,

$$
\boxed{g_1(Z)\approx1+\alpha_{1Z}Z,\qquad
 g_4(A,Z)\approx1+\alpha_{4Z}(A)Z.}
$$

The published P1 slope is `alpha1Z=0.00295165270232 mm^-1`. The new P4 slopes are [D2,D3]:

| A (D) | alpha4Z(A) (mm^-1) |
|---:|---:|
| 0 | 0.00295232758 |
| 1 | 0.00295172708 |
| 2 | 0.00295112891 |
| 3 | 0.00295052435 |
| 4 | 0.00294991207 |

**Linear in Z does not mean linear in A.** Each listed slope is fitted separately at fixed accommodation, from theta=0 ratios. Their span is 0.0818504% of their mean. The source's convenient `alpha4Z approximately 0.00295 mm^-1` approximation is not a separate fit with a certified error bound; retain the tabulated A dependence when testing differential scaling. These are optical-model coefficients, not experimental calibration constants.

The P4 sweep contains 1,275 states and 11,475 field records: five accommodations, rotations -10,-5,0,+5,+10 degrees, 51 Z planes and nine fields. Every Z ratio uses the real grid at **the same A and theta at Z=0**. It must not be formed against one zero-gaze grid for all angles. The constrained linear factor has worst-state coordinate RMSE about 0.249 micrometers and maximum Euclidean point error about 0.511 micrometers. Direct fitted per-state scales have smaller residuals; the two errors must not be mixed. Maximum relative rotation variation of the measured P4 scale is 0.00294912%; the P1 five-angle check gives 0.00019129%. These support approximate separability at the tested angles, not all 401 rotations or an arbitrary optical setup [D1-D3].

### 1.2 Provenance and what the scale results do not establish

The new P4 Z study uses LEN SHA-256 `fb3937e6763f27e331faa10a4863533e994030cbc0ae74364b186ba0d06d474d`; the original P4 accommodation/rotation study uses `5e0715c70016fbcec8a49956dd2ec824a50f5c4c52ca472b0225a2523db614ac`. Their fitted components must not be presented as one validated theta/A/Z model. Establish matched lens-revision evidence or refit the composed model in the actual setup before quantitative transfer. The new evidence motivates the model structure and the common-scale approximation; it does not authorize copying or combining setup-specific fitted coefficients [D1,D2,D4].

Z is the source's absolute `Cornea_ENT_D` THI coordinate. The P4 loaded z1 baseline is 0 mm; the P1 sequence baseline is 1 mm, while its ratios are still referenced to the recorded Z=0 plane. Do not subtract the sequence baseline or relabel Z as measured eye-camera translation. An experimental reference plane requires its own declared scale convention. Estimate relative image scale from P1, not absolute eye location from the published slope.

Common additive image translation is removed exactly by differences. Physical translation may also change scale, perspective, field sampling or the reflections differently. The Z studies characterize **center-relative pattern magnification**, not the full displacement between P1 and P4 optical origins. Near-equal shape scales do not prove that the centroid-difference signal has identical Z dependence. Keep the calibrated relative-centroid law separate (Sections 5 and 6).

The approximately 1-micrometer simulation simplification reference is not an experimental accuracy cutoff. Neither pixel RMS, state RMS, nor fixation flatness is a hard scientific requirement here.

## 2. Measurements, state, and coordinate conventions

Let `i` identify a frame and `j=1,2,3` identify a persistent source correspondence:

$$
\mathbf p_{ij}=P1_{ij},\qquad \mathbf q_{ij}=P4_{i,\pi(j)}\in\mathbb R^2.
$$

Read the permutation and validity flags from each detection payload; the recorded convention is zero-based `pair_index=[2,1,0]`. Do not independently sort distorted patterns to invent correspondence. Three source locations are not the five gaze targets [G1,G2].

The states of interest are

$$
\boxed{x_i=(\theta_i,A_i),}
$$

with horizontal gaze in degrees and accommodation in diopters. A positive nuisance scale `g_i` (the P1 scale `g1_i`) can be inferred from P1 or profiled in a relative-coordinate likelihood; it is not a third physiological variable. P4 uses the same scale in the leading approximation or a constrained ratio `eta=g4/g1` when supported; it does not receive an unrelated free framewise scale. No absolute lateral translation is estimated. No vertical-gaze state is added: nominal vertical gaze is zero in the fixation protocol, but measured image-y and its state dependence remain intact.

All 20 reviewed calibration conditions from captures 1-4 participate in a full calibration. Nominal horizontal gaze is `[-10,-5,0,5,10]` degrees. Read accommodation demands from the fixation metadata: the lowest stored demand is approximately 0.36036 D, not measured zero accommodation. Use a declared `A_ref`, initialized from this low-demand condition when appropriate; do not relabel it as physiological `A=0`. Both states may change inside every fixation [G1].

Use one fixed reference image-length unit for templates, centers and transformations: calibrated image-plane millimeters or declared reference-pixel units. Use `a=(A-A_ref)/(1 D)` for basis functions when convenient. A numerical angle scale, such as 10 degrees, changes coefficient units and must be recorded. Preserve the frozen baseline's separate `theta/15` convention only in its legacy adapter.

A calibrated optical-axis offset `theta_opt` may be used, with `vartheta=theta-theta_opt`. Initially set it from nominal zero-gaze alignment. Even/odd rotation constraints below refer to optical alignment, not a guarantee that nominal fixation zero is precisely the optic axis. Only a shared, identifiable alignment offset may be refined; not a new zero for each frame or capture.

## 3. Relative observations remove translation without losing gaze displacement

Define the two centroids and triangle areas:

$$
\mathbf c_{1i}=\tfrac13\sum_j\mathbf p_{ij},\quad
\mathbf c_{4i}=\tfrac13\sum_j\mathbf q_{ij},\quad
\mathcal A_{ri}=\tfrac12|\det E_{ri}|,
$$

$$
E_{1i}=[\mathbf p_{i2}-\mathbf p_{i1}\ \ \mathbf p_{i3}-\mathbf p_{i1}],
\qquad E_{4i}=[\mathbf q_{i2}-\mathbf q_{i1}\ \ \mathbf q_{i3}-\mathbf q_{i1}],
\qquad \ell_{1i}=\sqrt{\mathcal A_{1i}}.
$$

A complete, nonredundant translation-free vector is

$$
\boxed{y_i=\begin{bmatrix}
\mathbf p_{i2}-\mathbf p_{i1}\\
\mathbf p_{i3}-\mathbf p_{i1}\\
\mathbf q_{i1}-\mathbf c_{1i}\\
\mathbf q_{i2}-\mathbf c_{1i}\\
\mathbf q_{i3}-\mathbf c_{1i}
\end{bmatrix}\in\mathbb R^{10}.}
$$

Adding the same arbitrary vector to all six points leaves `y_i` unchanged. This is the observation contract; a reference measured at a previous time is not equivalent.

Keep the established area-normalized representation:

$$
\boxed{\mathbf r_{ij}=\frac{\mathbf p_{ij}-\mathbf c_{1i}}{\ell_{1i}},\qquad
\mathbf v_{ij}=\frac{\mathbf q_{ij}-\mathbf c_{1i}}{\ell_{1i}}.}
$$

Its derived summaries remain exactly

$$
\boxed{\mathbf d_i=\frac{\mathbf c_{4i}-\mathbf c_{1i}}{\ell_{1i}}=\tfrac13\sum_j\mathbf v_{ij},
\qquad \rho_{4i}=\frac{\mathcal A_{4i}}{\mathcal A_{1i}}.}
$$

Area means the triangle of reflection centers, not blob area. `rho4` is the area ratio, not its square root. Pair displacements `(q_j-p_j)/ell1=v_j-r_j` contain the same relative information when `r` is retained.

**Do not center P1 and P4 independently and then discard their centroid difference.** Independent centering removes translation within each pattern but also removes the dominant gaze signal unless `c4-c1` is retained separately. Conversely, do not treat centroid, all edges, all coordinates and area as independent extra observations.

Six 2D points give 12 coordinates; common translation removes two dimensions and common scale removes one. The full normalized geometry has nine continuous dimensions on a nondegenerate correspondence branch. The stored twelve entries `(r,v)` are constrained by `sum r=0` and `area(r)=1`. Section 8 gives a nonredundant likelihood construction.

## 4. Optical transformations in local, not absolute, coordinates

### 4.1 Rotation operator

In a fixed aligned coordinate system, use the proposed restricted keystone family from the optical source:

$$
K_r(\vartheta;\mathbf b)=
\frac{1}{1+q_r(\vartheta)b_y}
\begin{bmatrix}s_{x,r}(\vartheta)b_x\\s_{y,r}(\vartheta)b_y\end{bmatrix},
\qquad r\in\{1,4\},
$$

$$
s_{x,r}=1+\alpha_r\vartheta^2,\quad
s_{y,r}=1+\beta_r\vartheta^2,\quad
q_r=\gamma_r\vartheta.
$$

This enforces `K_r(0;b)=b`. If the simulated keystone axes differ from camera axes, use a fixed calibrated alignment matrix `Q_r` and the mapping `Q_r K_r(vartheta; Q_r^T b)`. Do not rotate each observed P4 triangle into a fitted frame that removes its informative deformation. The scalar named `q_r` is a keystone coefficient, not a measured P4 point.

With reference length unit L, `q_r` has units L^-1, `alpha_r,beta_r` have degree^-2, and `gamma_r` has L^-1 degree^-1. Require positive scales and a denominator bounded away from zero over the declared field/state domain. These are numerical/model-domain conditions, not RMS accuracy gates.

The restricted family is a starting hypothesis. The source identifies spatial residuals that it cannot express, including nonzero vertical motion from a baseline point on `b_y=0`. Raising the degree of the angle curves cannot fix that missing spatial component. Test the actual three source fields and add a spatial term only when its residual structure supports it [D1].

### 4.2 P1: fixed distorted template, gaze deformation, then nuisance scale

Let `b1_j` be the empirical real P1 template at reference scale and optical zero. It already includes the sampled P1 radial distortion. Define

$$
\boxed{\mathbf F_{1j}(\theta)=K_1(\theta-\theta_{opt};\mathbf b_{1j}).}
$$

The source suggests the unnormalized pattern shape `g1(Z) F1_j(theta_i)`, with approximately linear `g1(Z)` as defined in Section 1.1. In homogeneous notation the order is `S(g1) H1(theta_i)`, not `H1(theta_i) S(g1)`. The latter changes the projective denominator and is generally a different mapping [D1,D2].

P1's normalized orientation/shape may carry gaze information even when weak. Predict that relative response; do not assert `area(P1)` is constant with gaze, and do not infer `g` from changes in a separately fitted barrel coefficient.

### 4.3 P4: accommodation-dependent baseline, then gaze deformation

For a known or separately constrained **paraxial** reference template `u4_j`, define

$$
\mathbf z_j(A)=M(A)\mathbf u_{4j},\qquad M(A_{ref})=1,
$$

$$
\boxed{\mathbf B_{4j}(A)=
\left[1+\kappa_4(A)\|\mathbf z_j(A)\|^2\right]\mathbf z_j(A),}
\qquad
\boxed{\mathbf F_{4j}(\theta,A)=K_4(\theta-\theta_{opt};\mathbf B_{4j}(A)).}
$$

`M(A)` is accommodation-dependent P4 magnification. `kappa4(A)` is its first radial coefficient in L^-2. These are different from common nuisance magnification `g_i`. The radial radius is computed before radial distortion, in the same reference-length system. A coefficient expressed in sensor pixels changes under pixel-to-millimeter conversion; it is not dimensionless.

The proposed composition is radial baseline formation, then rotation deformation, then the external P4 axial scale `g4(A,Z)`. The new Z sweep directly supports scaling its own same-A/theta baseline, not automatic composition with an older lens revision; the latter is a model to fit and verify. In the common-scale approximation, `g4=g1=g_i`. At optical zero `K4=I` for every A, so accommodation-dependent baseline scale cannot be hidden in `sx4(0,A)` or `sy4(0,A)`. Even when the rotation coefficients have no explicit A dependence, their denominator contains `B4_y(A)` and therefore creates gaze/accommodation interaction. Add explicit `A*vartheta` or `A*vartheta^2` coefficient dependence only if the complete model needs it [D1].

Begin with constant/linear response functions around `A_ref`, for example `M=1+m1*a` and `kappa4=kappa_ref+kA*a`, enforcing positive scale and valid radial mapping over the used radii. For a one-to-one cubic radial mapping, inspect both `1+kappa*r^2` and its radial derivative `1+3*kappa*r^2`. Do not require the sign or slope of every component to match an assumed universal accommodation law.

Logarithmic, shifted-power or literal-power functions can later parameterize these **specific mechanisms**. A shape parameter is global across the calibration, not fitted independently per frame or per P4 subset. For each candidate, jointly refit its global parameters and all frame states. Do not interpret the old empirical exponent as a measured barrel-distortion exponent.

### 4.4 An empirical P4 baseline is not a paraxial grid

A measured zero-gaze P4 template `b4_ref` already contains its reference distortion. Do not insert it as `u4` into the absolute radial formula and apply that same distortion again. If the reference radial map is known and invertible, use

$$
\mathbf B_4(A)=\mathcal R_A\!\left(\mathcal R_{A_{ref}}^{-1}(\mathbf b_4^{ref})\right),
$$

where `R_A(u)=[1+kappa4(A)||M(A)u||^2]M(A)u`. Otherwise fit an explicitly **relative baseline-deformation model**, or constrain the paraxial template/reference distortion using matched optical data. Its fitted coefficients then describe relative deformation, not separately measured absolute `kappa4`. An unconstrained paraxial template, center, magnification and reference radial coefficient must not all be declared identifiable from three reference points [D1,D2].

## 5. The joint relative forward model

This section states the leading common-scale model, `g1=g4=g`, with a Z-independent displacement law in reference-scale units. Near-equal linear pattern slopes motivate the first assumption; the displacement assumption remains a separate calibration question. Section 6.3 gives the differential-scale extension without changing the relative observation contract.

Define the optical pattern means, centered shapes, and P1 reference-area scale:

$$
\overline{\mathbf F}_r=\tfrac13\sum_j\mathbf F_{rj},\qquad
\mathbf S_{rj}=\mathbf F_{rj}-\overline{\mathbf F}_r,\qquad
L_1(\theta)=\sqrt{\operatorname{area}(\mathbf F_{11},\mathbf F_{12},\mathbf F_{13})}.
$$

Two optical center-relative models alone do not predict the measured separation of their centroids. Let `delta(theta,A)` be the **relative displacement between their local optical origins**, in reference-length units. Then the predicted relative centroid displacement is

$$
\boxed{\mathbf h(\theta,A)=\boldsymbol\delta(\theta,A)
+\overline{\mathbf F}_4(\theta,A)-\overline{\mathbf F}_1(\theta).}
$$

Either calibrate this relative-origin model and derive `h`, or calibrate `h` directly and derive `delta`. **Do not give both independent free coefficients for the same displacement.** No absolute optical-origin location is needed.

For deriving the cancellation only, an arbitrary nuisance origin `t_i` can be introduced:

$$
\mathbf p_{ij}=\mathbf t_i+g_i\mathbf F_{1j},\qquad
\mathbf q_{ij}=\mathbf t_i+g_i(\boldsymbol\delta+\mathbf F_{4j}).
$$

These auxiliary equations are not absolute-position observations or a new eye-position estimator. Subtracting same-frame references removes `t_i` identically and gives the actual relative predictions:

$$
\boxed{\widehat{\mathbf p_{ij}-\mathbf c_{1i}}=g_i\mathbf S_{1j}(\theta_i),}
$$

$$
\boxed{\widehat{\mathbf q_{ij}-\mathbf c_{1i}}
=g_i\left[\mathbf h(\theta_i,A_i)+\mathbf S_{4j}(\theta_i,A_i)\right].}
$$

Consequently, under the common-scale model, `ell1_i=g_i L1(theta_i)` in the noiseless geometry and the **joint normalized model** is

$$
\boxed{
\widehat{\mathbf r}_{ij}=\frac{\mathbf S_{1j}(\theta_i)}{L_1(\theta_i)},\qquad
\widehat{\mathbf v}_{ij}=\frac{\mathbf h(\theta_i,A_i)+\mathbf S_{4j}(\theta_i,A_i)}{L_1(\theta_i)}.
}
$$

Both reference translation and common magnification have disappeared; accommodation-dependent P4 magnification and deformation remain. P1 has no direct A dependence in the first model, but the **joint observation** depends on both states through its distinct P1 and P4 blocks. This is stronger than treating the measured P1 shape as exact context for a six-coordinate P4 regression.

The familiar summaries follow without extra fitted residuals:

$$
\boxed{\widehat{\mathbf d}=\mathbf h/L_1,\qquad
\widehat\rho_4=\frac{\operatorname{area}(\mathbf F_{41},\mathbf F_{42},\mathbf F_{43})}{L_1^2}.}
$$

A useful initial displacement model is `h_x(theta,A)=b(A)+s(A)*theta`; accommodation may affect both offset and gaze gain. Retain a calibrated vertical displacement component as needed, even though vertical gaze is not an estimated state. This displacement law must be calibrated: the center-relative optical simulation alone does not supply it. Weak gaze dependence of shape motivates this initialization, not removal of gaze dependence from the final shape equations.

If a pixel prediction is needed for display or held-point scoring, use

$$
\widehat{\mathbf q}_{ij}=\mathbf c_{1i}+\ell_{1i}\widehat{\mathbf v}_{ij}.
$$

The measured P1 centroid merely restores the arbitrary image origin. Only the difference from the held measured P4 is scored; no absolute eye location has become observable.

## 6. P1 magnification correction and its boundary of validity

### 6.1 Estimate scale after accounting for gaze

The P1-area estimate is

$$
\boxed{\widehat g_i(\theta)=\ell_{1i}/L_1(\theta).}
$$

Comparing P1 area only with its zero-gaze value would mix gaze-dependent deformation with nuisance magnification. An alternative P1-only weighted scale estimate uses the four edge coordinates. Write `e1_i=vec(E1_i)` and `e1_ref(theta)=vec([F12-F11,F13-F11])`; then, for fixed positive-definite edge covariance,

$$
\widehat g_{P1}=\frac{e_{1,ref}^{\mathsf T}R_{e1}^{-1}e_{1i}}
{e_{1,ref}^{\mathsf T}R_{e1}^{-1}e_{1,ref}}.
$$

It uses no distortion-center measurement and must be positive to satisfy the assumed model. Under noiseless isotropic scaling it agrees with the area ratio. With noise/model mismatch these are different estimators; declare which is used rather than combining them as independent measurements. The established normalized route keeps area normalization and uses the edge estimate as a diagnostic or initializer.

### 6.2 Do not correct magnification twice

The following corrected coordinate is equivalent to the established normalization plus its modeled gaze-dependent reference scale:

$$
\frac{\mathbf q_{ij}-\mathbf c_{1i}}{\widehat g_i(\theta)}
=L_1(\theta)\mathbf v_{ij}.
$$

Therefore, **do not divide `v` by `g` again**. Preserve the denominator `L1(theta)` in the forward model and include its derivative during state estimation. P1 normalization does not mean `L1` is constant with gaze. P4-specific `M(A)` must remain in the model; removing it would remove an accommodation signal.

### 6.3 Residual differential scale after P1 normalization

For a matched-reference linear scale model, define

$$
\boxed{\eta(A,Z)=\frac{g_4(A,Z)}{g_1(Z)}
\approx\frac{1+\alpha_{4Z}(A)Z}{1+\alpha_{1Z}Z}.}
$$

Within that approximation,

$$
\eta-1=\frac{[\alpha_{4Z}(A)-\alpha_{1Z}]Z}{1+\alpha_{1Z}Z}.
$$

Both scales being linear does **not** make eta constant or linear in Z. Exact cancellation requires equal fractional scale functions over the domain, not just similar slopes. The reported slopes make eta near one and justify a common-scale starting model. Small differences between fitted slopes are not an error bound: they omit linear-approximation residuals, residual angle dependence, setup differences and measurement noise. Do not treat their ratio as an experimentally validated correction [D1-D3].

To preserve relative translation explicitly, introduce only for algebra

$$
\mathbf p_{ij}=\mathbf t_i+g_{1i}\mathbf F_{1j},\qquad
\mathbf q_{ij}=\mathbf t_i+g_{1i}\boldsymbol\delta_Z(\theta_i,A_i,Z_i)
                         +g_{4i}\mathbf F_{4j}.
$$

`delta_Z` is a **relative optical-origin displacement in P1 reference-scale units**, not an absolute position. It reduces to Section 5's delta at the declared reference. Subtraction cancels the arbitrary common `t_i`. Define

$$
\mathbf h_Z=\boldsymbol\delta_Z+\eta\overline{\mathbf F}_4-\overline{\mathbf F}_1.
$$

The extended normalized predictions are

$$
\boxed{\widehat{\mathbf r}_{ij}=\frac{\mathbf S_{1j}}{L_1},\qquad
\widehat{\mathbf v}_{ij}=\frac{\mathbf h_Z+\eta\mathbf S_{4j}}{L_1}.}
$$

Thus the centered P4 pattern and area ratio obey

$$
\frac{\mathbf q_{ij}-\mathbf c_{4i}}{\ell_{1i}}
\approx\eta\frac{\mathbf S_{4j}}{L_1},\qquad
\widehat\rho_4=\eta^2\frac{\operatorname{area}(\mathbf F_{41},\mathbf F_{42},\mathbf F_{43})}{L_1^2},
\qquad \widehat{\mathbf d}=\mathbf h_Z/L_1.
$$

The common-scale model is recovered by eta=1 and `h_Z=h(theta,A)`. **Do not multiply the complete `h+S4` by eta without also deriving the appropriate relative-origin law.** The centered Z grids constrain the pattern factor, not that centroid law. Either fit `h_Z` directly or fit `delta_Z` and derive it; do not fit both independently. Do not divide the whole measured `(q-c1)/ell1` by eta as a substitute for this forward model.

P1 normalization removes the dominant shared Z magnification while retaining accommodation-dependent `M(A)` and `kappa4(A)`. A small unmodeled differential factor can still resemble accommodation; keep it a documented approximation or a calibrated correction, never an independent free per-frame P4 magnification.

### 6.4 Optional differential correction without adding a free Z state

Use **common-scale mode** first: infer `g1=ell1/L1(theta)` for diagnostics, set eta=1, and calibrate `h(theta,A)` while inspecting residual dependence on the retained P1 scale. This preserves the two-state relative estimator. Do not require experimental Z or create another state simply because the optical sweep used Z.

If matched optical/experimental evidence supports the two linear scale laws at the same reference plane, eliminate Z algebraically. For `g1=1+alpha1Z*Z` with nonzero alpha1Z, let `c(A)=alpha4Z(A)/alpha1Z`. Then

$$
\boxed{g_4=1+c(A)(g_1-1),\qquad
\eta(A,g_1)=\frac{1+c(A)(g_1-1)}{g_1}.}
$$

This uses a measured relative scale and a **fixed calibrated slope-ratio function**, not a new free exponent or per-frame Z fit. The numerical slopes in Section 1.1 are provenance, not automatic values for this correction. With a different reference plane, derive and save the corresponding relative slopes; do not silently use the source's Z=0 intercepts.

At fixed recorded ell1, `g1(theta)=ell1/L1(theta)` changes with trial gaze, and eta can change with trial accommodation. For the stated ratio model,

$$
\partial_A\eta=c'(A)(1-1/g_1),\qquad
\partial_{g_1}\eta=\frac{c(A)-1}{g_1^2},\qquad
\partial_\theta g_1=-g_1\frac{\partial_\theta L_1}{L_1}.
$$

Include these chain-rule terms, plus the separately declared `h_Z` dependence, in inversion and calibration. Never freeze the correction at a stale state during a candidate objective evaluation.

Because this extension uses P1 length as an input, the corrected prediction is `F(theta,A;ell1)`, not a state-only function of nine normalized coordinates. Retain ell1 and propagate its shared uncertainty. Prefer the ten-coordinate relative likelihood when fitting a scale-dependent correction; a conditional normalized implementation must propagate the complete residual including its dependence on measured ell1, and must not count ell1 again as independent evidence. No absolute image position becomes an observation.

## 7. Centers, reference gauges, and three-point identifiability

### 7.1 Zero-gaze centroids initialize origins; they do not measure distortion centers

Initialize separate P1/P4 reference patterns from many matched near-zero-gaze frames, expressed relatively and with consistent scale. The most symmetric P1 frames may support alignment, but the expected symmetry must follow the actual source layout. Neither one extremal frame nor nominal fixation zero establishes an exact optical axis.

A centroid can differ from a radial center even when the undistorted points have zero mean:

$$
\overline{\mathbf B}_4=M\overline{\mathbf u}_4
+\kappa_4 M^3\overline{\|\mathbf u_4\|^2\mathbf u_4}.
$$

Thus centroids are a practical coordinate gauge; local center offsets can be shared, constrained calibration parameters only if identifiable. There is no need to recover absolute centers. Do not move a center independently in each frame, as that can absorb the accommodation signal.

Center **after** evaluating the optical map: `S=K(B)-mean K(B)`. In general `K(B-mean B)` is different. Similarly, remove external scale before applying an inverse projective map. These noncommuting operations must not be interchanged.

### 7.2 Fix parameter gauges before fitting

Fix one reference P1 scale (for example `L1(theta_ref)` to the declared reference template), source identities, coordinate orientation, and the baseline magnification convention `M(A_ref)=1`. A known or independently constrained paraxial/reference-distortion model is needed for absolute radial-coefficient interpretation.

In a normalized-only likelihood, a common multiplication of all optical lengths can leave predictions unchanged after corresponding changes in inverse-length coefficients. Also, an arbitrary gaze-dependent isotropic factor in the P1 template may trade off against per-frame `g` and other response scales. Do not freely estimate all such factors and then report separately precise parameters. Use matched optical constraints or clearly declare an effective normalized parameterization, and evaluate sensitivity to those constraints.

### 7.3 Equal-radius samples cannot separate scale and barrel strength

If the three paraxial source samples have the same radius R about the distortion center,

$$
\mathbf B_{4j}(A)=
\underbrace{M(A)[1+\kappa_4(A)M(A)^2R^2]}_{M_{eff}(A)}\mathbf u_{4j}.
$$

Only the combined scale `M_eff` is identifiable from that radial pattern. A subsequent deterministic `K4` cannot separate two parameter settings giving identical `B4`. Different nonzero radii improve separation, but an unknown center/template can still cause confounding. A central spot can help locate a center; by itself it does not provide a second nonzero radius for scale-versus-radial separation.

Start with an identifiable effective accommodation-dependent baseline when necessary. Do not declare the approach unusable: accommodation may still be inferred from the combined response, while absolute `M` and `kappa4` remain unresolved. Avoid a free affine map per frame; three correspondences can determine that map exactly without independently validating the state.

## 8. A single relative-coordinate cost with correct uncertainty

### 8.1 Linear relative-coordinate likelihood

Let `w_i` stack the original 12 point coordinates and let `B` be the fixed rank-10 matrix implementing `y_i=B w_i` from Section 3. Then

$$
R_{yi}=B\Sigma_i B^{\mathsf T},\qquad
\widehat y_i=g_i f(\theta_i,A_i;\Psi),
$$

where `f` stacks the two P1 optical edges and the three vectors `h+S4_j`. `Psi` contains globally shared optical parameters. This likelihood uses only relative measurements, but preserves relative length information. Common-translation noise lies in B's null space; no absolute-position penalty is added.

For fixed states/parameters and fixed covariance, an optional full relative-likelihood scale profile is

$$
\widehat g_i=\frac{f^{\mathsf T}R_{yi}^{-1}y_i}{f^{\mathsf T}R_{yi}^{-1}f},
$$

subject to the declared positive-scale domain. This uses both reflection blocks and is **not** the P1-only area estimate. P1 can initialize/constrain it; retain that distinction in artifacts. A common scale parameter is acceptable; separate unrestricted P1/P4 scales can destroy accommodation identifiability. This closed-form scale profile applies to the leading `g*f(theta,A)` model only. With eta or a relative-centroid correction depending on g1, use the extended relative forward function and a consistent constrained nuisance solve; do not reuse this formula as though the prediction remained homogeneous in g.

### 8.2 Nonredundant normalized likelihood

For the area-normalized implementation, do not treat all twelve `(r,v)` components as independent. One local three-coordinate P1 shape chart is obtained from the columns `a,b` of `E1/ell1`:

$$
\chi_1=\left(\operatorname{atan2}(a_y,a_x),\ \log\|a\|,
\ \frac{a^{\mathsf T}b}{a^{\mathsf T}a}\right).
$$

Record the discrete area sign, keep orientation on a consistent local angle branch, and reject genuinely degenerate geometry. Together with six entries of `v`, this gives nine nonredundant relative normalized coordinates. With `det[a,b]=2*sign`, the three shape coordinates reconstruct the normalized P1 edges; no independent area residual is needed.

Use the corresponding predicted chart from `F1/L1` and the six predicted `v` entries. First-order covariance follows from the complete raw-measurement Jacobian, including the shared P1 centroid and area [M1]. Nonlinear normalization and this propagated Gaussian likelihood are a local approximation; they are not claimed to equal the profiled ten-coordinate likelihood for arbitrary noise. Choose one primary objective and retain the other as a diagnostic, rather than adding both as if independent.

If redundant coordinates are retained in code, handle their constrained covariance rank explicitly. Artificial diagonal jitter must not turn deterministic geometric constraints into invented evidence. Near degeneracy use the linear-relative formulation or mark normalization unreliable.

### 8.3 Calibration objective and controls

In the chosen nonredundant coordinates, write `e_i=y_i-F(x_i;Psi)` (with profiled nuisance scale if using Section 8.1). A fixed-covariance calibration objective is

$$
J=\frac{1}{2K}\sum_{k=1}^{K}\frac1{N_k}\sum_{i\in k}e_i^{\mathsf T}R_i^{-1}e_i
+J_{anchor}+J_{global},
$$

$$
J_{anchor}=\frac1{2K}\sum_k\left[
\frac{(\bar\theta_k-\theta_k^{nom})^2}{s_\theta^2}
+\frac{(\bar A_k-A_k^{demand})^2}{s_A^2}\right].
$$

`K=20` when all reviewed conditions are used. Inherited starting scales `s_theta=0.10 degree`, `s_A=0.25 D` are finite mean-penalty weights, not accuracy or motion limits. Declare global priors and test their influence; set the temporal-flatness penalty to zero. Do not add raw `G_A` or `G_theta` in incompatible units as arbitrary penalties. Same-frame optical residuals already enforce a shared state.

Use a frozen covariance policy for the first controlled model comparison. If covariance becomes a fitted/state-dependent likelihood component, include its normalization (such as log determinant) and validate its interpretation; increasing uncertainty cannot be allowed to hide systematic residuals. Measurement covariance estimated from temporal differences may include true motion. Detector centers and CODE V center-ray coordinates are different measurement definitions, so optical mismatch and detector bias remain possible.

When using both P1 shape and P4 relative measurements, their correlated errors must be retained. A legacy covariance routine for `v-D-T*r` is not automatically the correct covariance for this new joint residual. Existing numerical solvers can be reused, but observation maps, Jacobians, parameter profiling and uncertainty interfaces require an explicit new model adapter, not a relabeling of the old 27 coefficients.

## 9. Full calibration and alternating estimation

### 9.1 Initialization

Use all calibration conditions for the final fit. The following staged initialization does not impose held-out calibration conditions:

1. From the low-demand condition, fit an initial linear relation between fixation-mean normalized centroid displacement and nominal horizontal gaze: `mean(d_x)_k ≈ a0+a1*theta_nom_k`. Initialize every frame with `(d_x-a0)/a1`. This does not assign all frames the fixation label.
2. Use many near-zero-gaze samples to initialize the relative templates/alignment. Constrain the P1 reference-scale convention and initialize `K1` from matched optical evidence or a justified fixed-scale fit; symmetry is supporting evidence, not absolute eye-position ground truth.
3. Initialize `M(A)` and the identifiable radial/effective-baseline terms from near-zero-gaze frames across demands. Nominal demand supplies initial values and soft means, not fixed A values. Use the known empirical-versus-paraxial baseline convention.
4. Initialize the relative displacement law `h(theta,A)`; allow accommodation-dependent offset and gaze gain before adding arbitrary higher-order gaze terms.

### 9.2 Frame updates with shared parameters fixed

For common-scale mode, let `Q_i(theta,A)` be the single nine-coordinate optical cost. The sequence is

$$
\widehat g_i^{(t)}=\ell_{1i}/L_1(\theta_i^{(t)}),\qquad
A_i^{(t+1)}=\arg\min_A Q_i(\theta_i^{(t)},A),
$$

$$
\theta_i^{(t+1)}=\arg\min_\theta Q_i(\theta,A_i^{(t+1)}),\qquad
\widehat g_i^{(t+1)}=\ell_{1i}/L_1(\theta_i^{(t+1)}).
$$

The P1-scale-corrected P4 relative coordinates are conceptually `L1(theta)*v`; their prediction is `h+S4` in common-scale mode, or `h_Z+eta*S4` in the extension. The actual minimizer evaluates the composed forward model and its covariance. It does not perform an unqualified inverse keystone about the observed centroid.

In calibrated differential-scale mode, each trial state additionally evaluates `g1(theta)`, `g4(A,g1)`, eta, and the supported relative-centroid correction inside the **same** cost. The order is gaze -> P1 scale -> P4/P1 scale ratio -> accommodation -> corrected gaze, with scale and ratio recomputed at every trial. This is a constrained correction to the optical prediction, not a second normalization of the observations or a third free physiological state.

For `h_x=b(A)+s(A)*theta`, an inexpensive gaze proposal is

$$
\theta_{proposal}=\frac{L_1(\theta^{(t)})d_x-b(A^{(t+1)})}{s(A^{(t+1)})}.
$$

This is only a proposal: the accepted gaze update must reevaluate `L1(theta)` and the complete optical cost at trial gaze values. Include derivatives of the reference-area denominator and optical centering. Weak shape-gaze coupling helps the alternation; weak accommodation sensitivity or nearly parallel state responses can still make it fail.

During calibration, conditional updates minimize the **total J**, including the contribution to soft fixation means, not a framewise surrogate anchor. During application there is no nominal-frame anchor. Use damping/line search and joint two-state refinement when needed; do not claim that a fixed number of alternating steps proves convergence or unique state recovery.

For the linear-relative route, use the same alternating logic with the selected positive-scale profile in common-scale mode, or the properly constrained scale solve for the differential extension, in the same objective. Do not alternate P1-only scale minimization and another unrelated likelihood while claiming guaranteed decrease of a common cost.

### 9.3 Global updates and completion

After updating the frame states, update the shared baseline, displacement, distortion, and permitted alignment/rotation parameters using **all valid full-period calibration rows in all 20 conditions**. Repeat state and global updates to numerical convergence. Preserve natural within-fixation variation. P1 scale gauge, radial/template gauges and reference definitions stay fixed or explicitly constrained throughout.

Reuse bounded least-squares/variable-projection ideas only for genuinely linear parameter blocks. The new radial/projective composition is not automatically the old linear 27-column design. Changing the model requires new derivatives and potentially different profiled blocks, not a new unverified optimization algorithm. Preserve numerical certification and CPU/GPU parity requirements.

Different hypotheses for `M(A)`, `kappa4(A)` or a shared shape exponent require fresh full calibrations with otherwise identical controls. No whole-capture or whole-gaze holdout is required for this immediate internal-agreement task. Numerical stabilization, all scheduled outcomes accounted for, and reproducible cross-agreement determine completion; no fixed RMS or flat-trajectory requirement is imposed.

## 10. Cross-agreement of the complete iterative estimator

Freeze every global optical parameter, template, center offset, response law and any calibrated axial slope-ratio/relative-centroid correction after full calibration. For each frame, omit each P4 point in turn while retaining all three P1 points. Run the **entire** state/scale correction using only those retained measurements, not just a final forward evaluation at the all-three state.

Permitted inputs are the P1 shape/area, two retained P4 coordinates, their proper covariance marginal, fixed calibration and retained-only starts. The measured all-three P4 centroid, area, affine map, all-three state, or an unmasked warm start must not enter. A mean over all three **predicted** optical points is allowed: it is a function of fixed parameters and the trial state, not the omitted measurement. Do not refit global centers, exponents or coefficients inside a subset check.

For normalized inference, score

$$
\mathbf e_{ij}^{cross}=(\mathbf q_{ij}-\mathbf c_{1i})-\ell_{1i}\widehat{\mathbf v}_j(\widehat x_{i,-j}).
$$

This remains a relative error even when displayed in pixels. Define

$$
E_i^2=\tfrac13\sum_j\|\mathbf e_{ij}^{cross}\|^2,\quad
G_{A,i}^2=\tfrac13\sum_{j<k}(\widehat A_{i,-j}-\widehat A_{i,-k})^2,
$$

and the analogous `G_theta`. They compare simultaneous estimates, not a previous frame or nominal demand. Aggregate squared quantities within exposures then equally across exposures; RMS is only a display summary. This vector-error RMS differs by a factor of square-root two from scalar-coordinate RMSE on identical 2D points; do not mix the convention with source optical reports.

Keep exact common model/frame/slot identities, signed axes, each point, worst point, tails, coverage, inverse rank/branches, and bound behavior. Missing measurements remain explicit; a smaller surviving cohort cannot manufacture a better score. Full-fit residuals, cross-reconstruction, temporal variation and nominal-mean discrepancy are reported separately.

Shared P1 noise and overlapping P4 subsets correlate errors and states. Use covariance of **differences**, including cross-covariances, when reporting uncertainty-normalized state agreement. For a regular fixed-calibration inverse, propagate omitted-point prediction uncertainty including retained/omitted correlations; near bounds or multiple branches do not replace prediction sets with an unjustified Gaussian ellipse. If a likelihood-based cross-score is introduced, keep raw errors and validate its covariance rather than treating it as independent truth.

Because global calibration already used these observations, this is **internal calibration consistency**, not independent validation. A raw-input noninterference test changes the omitted P4 only after freezing calibration; it must leave the subset solve unchanged. It is not a claim that recalibrating on changed data would leave coefficients unchanged.

## 11. What counts as a useful accommodation model

The model should improve explanation of common instantaneous states, not merely win a fraction of a pixel. Report optical cross-prediction, same-frame state agreement, identifiable optical response curves, and sensitivity to calibration conventions together. No arbitrary sum of pixels squared, degrees squared and diopters squared defines an overall winner.

For a fixed calibrated model in nonredundant whitened coordinates, let `J=[j_theta,j_A]`. If no other nuisance remains, the local information distinguishing accommodation from gaze is

$$
\mathcal I_{A\mid\theta}=F_{AA}-F_{A\theta}^2/F_{\theta\theta},\qquad F=J^{\mathsf T}R^{-1}J,
$$

when the gaze block is regular. If scale or other supported nuisances are fitted, project their whitened derivative columns out as well. Use the full joint derivatives, including `L1(theta)`, rather than a legacy fixed-P1-context Jacobian. Report finite rank, conditioning and alternative branches; do not manufacture a universal physical-information cutoff.

Information per diopter and raw `G_A` change under latent-A rescaling. Finite demand anchors establish a calibration convention but not actual framewise accommodation. The relative optical model does not by itself eliminate this ambiguity. Test shared anchor/prior/bound sensitivity, explain when only effective magnification is determined, and do not confuse clipped or compressed A trajectories with physiological agreement.

If reproducible residuals require a spatial component missing from keystone/radial geometry, add that mechanism rather than forcing an extreme accommodation exponent to absorb it. If the near-common scale approximation or relative-centroid Z behavior fails in the actual setup, diagnose that before assigning the residual to accommodation. Unknown detector-center bias, capture/demand confounding and uncertain optical alignment remain distinct alternatives. Physical accuracy ultimately requires an independent reference or additional justified optical constraints.

## 12. Required mathematical and implementation checks

Before claiming the new model is implemented, verify:

- Independent arbitrary frame translations leave every relative observable, inference input and cross-error unchanged; no fixed camera-origin assumption leaks in.
- Positive common scaling cancels in normalized predictions, P1-derived scale recovers the known synthetic scale, and accommodation-dependent P4 scaling remains.
- Synthetic unequal linear slopes retain the predicted eta in centered shape and eta-squared in area ratio; equal slopes recover the common-scale model exactly.
- Eliminating Z via the fixed slope ratio reproduces the same correction, with correct gaze/accommodation derivatives and matched reference conventions.
- A synthetic P4-only scale or relative-origin perturbation does **not** cancel and is not silently reported as measured accommodation; shape scaling alone does not determine centroid displacement.
- The linear-approximation error, slope-difference estimate and residual rotation dependence are reported separately; mismatched LEN revisions cannot silently supply a combined fitted model.
- P1 reference-area gaze dependence, quotient derivatives, coordinate units, source permutation and signed-area branches are correct.
- Optical centering and scale/projective composition use the stated order; zero-gaze identity holds; an empirical radial baseline is not corrected twice.
- The relative linear map has rank 10; normalized geometry has nine independent coordinates; propagated covariance and a nonredundant chart do not invent extra constraints.
- Equal-radius scale/radial degeneracy and freely moving-center counterexamples are recognized; reference gauge changes do not alter normalized predictions when transformed consistently.
- Synthetic nonconstant gaze/accommodation trajectories remain free within fixation; zero-mean changes leave soft mean anchors unchanged, though optical residuals may change.
- Every omitted-point solve, including initialization and scale correction, passes retained-only noninterference with calibration fixed; masks select covariance marginals before whitening.
- Failed/ambiguous solutions retain scheduled identities; all model comparisons use the same declared populations and report numerical scope separately from physiological interpretation.

These are required properties, not a claim of a completed repository test run or new real-data fit. Preserve all existing implementation, data and baseline artifacts until the new adapter is explicitly implemented and checked.

## 13. Sources and compatibility boundary

The model and cancellation identities in Sections 3-11 are the proposed synthesis and algebraic consequences of its assumptions. The updated optical source supports approximately linear P1 and P4 axial pattern magnification in separate studies and motivates near-common scaling. It does not validate exact experimental cancellation, the relative-centroid Z law, a merged multi-revision optical model, or three-source identifiability.

- **[D1]** [distortion_tracking / Theory.md, pinned optical source](https://github.com/rueijrwu/distortion_tracking/blob/1d2a0874c79f3f174b36a197c3fbbffd8a32fe60/Theory.md): optical composition, linear P1/P4 Z models, state-matched reference conventions, and limits on combining lens revisions.
- **[D2]** [distortion_tracking / Summary.md, same snapshot](https://github.com/rueijrwu/distortion_tracking/blob/1d2a0874c79f3f174b36a197c3fbbffd8a32fe60/Summary.md): separate optical studies, measured scale ratios versus linear fits, five-angle scope and lens hashes.
- **[D3]** [P4 Z magnification report](https://github.com/rueijrwu/distortion_tracking/blob/1d2a0874c79f3f174b36a197c3fbbffd8a32fe60/data/p4_z_magnification/p4_z_magnification.md): per-accommodation slopes, direct and linear residuals, matched Z=0 grids and measured endpoints.
- **[D4]** [P4 Z collection metadata](https://github.com/rueijrwu/distortion_tracking/blob/1d2a0874c79f3f174b36a197c3fbbffd8a32fe60/data/p4_z_magnification/metadata.json): collection configuration and LEN provenance; not hardware calibration.
- **[G1]** [Reviewed fixation metadata](../data/fixations/fixation_intervals.json): source hashes, actual gaze/demand labels, full periods and validity; read original timestamps/frame indices and their reliability flags.
- **[G2]** [Earlier estimator theory at the inspected snapshot](https://github.com/rueijrwu/gaze_acc_joint_estimator/blob/7f0670ca5ffbe264ad4406b5645dc5908d8d15a3/docs/Theory.md): historical conditional geometry and correspondence. Its independent-holdout requirements and empirical coefficient layout are not the new full-calibration optical model.
- **[M1]** [NIST uncertainty propagation](https://physics.nist.gov/cuu/Uncertainty/combination.html): first-order sensitivity/covariance propagation; it is not an exact nonlinear error distribution.
- **[M2]** [OpenCV camera-calibration conventions](https://docs.opencv.org/4.13.0/d9/d0c/group__calib3d.html): radial/projective coordinate conventions as mathematical background, not validation of this ocular optical model.

[ESTIMATOR_PLAN.md](ESTIMATOR_PLAN.md) and the accommodation plans contain historical implementation choices. Reconcile their relevant sections with this proposed joint relative model before changing code. Preserve [the frozen baseline](../models/quadratic_model.json), its `theta/15` convention, detection data, recorded experiments and test fixtures. Captures 5/6 remain outside the new calibration workstream; their unlabeled trajectories are not accommodation ground truth. Only this theoretical specification is changed by the present revision.
