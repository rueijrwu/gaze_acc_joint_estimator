# P4 theory: accommodation, anamorphic stretch, projective keystone, and alignment

**Revision: 2026-10-14.**

The objective is to parameterize P4 deformation by accommodation and gaze, using shared optical parameters that transfer across captures. Nonzero accommodation offsets are allowed; the aim is to explain their repeatable dependence. Vertex RMS measures geometric agreement and does not by itself identify a physical decomposition.

The working forward order is **relative accommodation barrel → combined tilt transform → centering**. The combined tilt transform contains both **anamorphic stretch** and **projective keystone**. Calling the whole block “keystone” previously obscured this distinction.

## 1. Completed model versus proposed corrections

The constant-demand baseline is the [common-size joint experiment](../experiments/archive/p4_keystone_common_size_joint/RESULTS.md). It fits fifteen shared parameters to twenty mean triangles: ten stretch/projective coefficients, one barrel slope, and two coordinates for each of the barrel and tilt origins. Each origin is shared across captures and gazes; the two origins can differ. The subsequent [planar-demand and shared-orientation test](../experiments/archive/p4_planar_demand_tilt_orientation/RESULTS.md) implements the first two recommended comparisons.

| Component | Completed baseline | Recommended next test |
|---|---|---|
| P1 scale | Frozen per frame | Same saved correction |
| Common P4 reference | Frozen previously refined template | Same template initially |
| Accommodation during calibration | Central demand held constant within each capture | Demand computed from target geometry |
| Optical gaze input | Mean saved polynomial gaze within each fixation | Same saved estimates |
| Barrel | Shared accommodation-dependent radial coefficient and free origin | Refit jointly |
| Anamorphic stretch | Independent axis scales, linear/quadratic in gaze, affine in accommodation | Retain these terms |
| Projective keystone | One gaze-dependent denominator, affine amplitude in accommodation | Retain and rotate consistently with stretch axes |
| Image-axis orientation | Fixed to image coordinates | Test one shared orientation angle |
| Optical zero | Implicit in unrestricted odd/even coefficients | Test a shared zero through a constrained relative tilt law if supported |
| Target-plane normal | Central gaze implicitly treated as normal | Use known geometry; state any assumed plane alignment |
| Objective | Equal-weight forward Cartesian vertex RMS | Same metric and weighting |

**The planar-demand correction and one shared image-axis orientation have now been tested**, assuming beta=0°. A constrained optical-zero law has not been fitted. The earlier baseline results below retain their original constant-demand interpretation; the follow-up results are summarized separately in Section 11.

The older [reference-based experiment](../experiments/archive/p4_reference_based_model_fit/RESULTS.md) used thirteen parameters, one common origin for both transforms, a measured C1 0° reference, and an inverse objective. Those choices describe that experiment, not the latest baseline.

## 2. Observations, mean triangles, and reference

For frame i, preserve the reviewed P4 vertex correspondence, whose zero-based permutation is [2, 1, 0]. Let g_i be its frozen positive P1 similarity scale. Define

$$
\mathcal C[Z]_j=Z_j-\frac13\sum_{m=1}^3 Z_m,\qquad
X_{ij}=\frac{\mathcal C[P4_i]_j}{g_i},
\qquad
\overline X_{cg}=\frac1{N_{cg}}\sum_{i\in(c,g)}X_i.
$$

Center and divide by each frame's P1 scale **before** fixation averaging. The twenty means are observations. They are not twenty adjustable templates.

The reviewed data have 88,978 frames, four captures, and targets at −10°, −5°, 0°, +5°, +10°. Nominal vertical target gaze is zero. These data do not independently calibrate a vertical optical gaze law.

| Capture | Central demand A_c0 (D) |
|---|---:|
| C1 | 0.36036036036 |
| C2 | 4.00000000000 |
| C3 | 3.00000000000 |
| C4 | 2.00000000000 |

Use the [reviewed fixation intervals](../data/fixations/fixation_intervals.json), [P1 scale convention](../data/P1_FRAME_SCALES.md), and the latest experiment's [input manifest](../experiments/archive/p4_keystone_common_size_joint/inputs/manifest.json). The frozen reference Q is the common refined template stored there, in centered P1-corrected P4 pixels.

Q already contains reference-state optical distortion. It is not an independently known paraxial triangle. Its accommodation reference remains

$$
A_{\mathrm{ref}}=0.36036036036\ \mathrm D.
$$

The template's nominal gaze reference is 0°. That label does not establish the optical symmetry zero or the target-plane normal. At A_ref and nominal optical-input angle 0°, the relative model must reproduce Q. A fixation's mean polynomial estimate can differ slightly from its nominal target angle.

For optical evaluation use the saved per-capture quadratic P4−P1 gaze estimator. For the mean experiment, use its mean over each fixation. Use the **known physical target direction** for target-distance geometry; gaze-estimator noise does not move a stationary target.

## 3. Accommodation demand on a flat target plane

Let d_c0 be the distance in millimetres to the nominal 0° target. If the target plane is perpendicular to that direction,

$$
d_{cg}=d_{c0}\sec\theta_g,\qquad
\boxed{A_{\mathrm{dem},cg}=A_{c0}\cos\theta_g.}
$$

Convert degrees to radians when evaluating trigonometric functions. Here theta_g is the target's actual viewing angle. A doubled reflected-ray angle must not be substituted without an explicit target-geometry derivation.

At a central distance of 250 mm, the central demand is 4 D. The demand is 3.984779 D at 5° and 3.939231 D at 10°.

| Capture | Central demand (D) | Demand at ±5° (D) | Demand at ±10° (D) |
|---|---:|---:|---:|
| C1 | 0.360360 | 0.358989 | 0.354886 |
| C2 | 4.000000 | 3.984779 | 3.939231 |
| C3 | 3.000000 | 2.988584 | 2.954423 |
| C4 | 2.000000 | 1.992389 | 1.969616 |

For small angles in radians,

$$
A_{\mathrm{dem}}-A_{c0}
=A_{c0}(\cos\theta-1)
\approx-\tfrac12 A_{c0}\theta^2.
$$

This is a known negative, even quadratic contribution to an offset measured from the central demand. It is modest: approximately −0.061 D at 4 D and ±10°. Its amplitude scales with central demand.

### A plane whose normal is not nominal 0°

Let beta be the nominal gaze angle of the plane normal, with d_c0 still the distance to the nominal 0° target. The perpendicular plane distance is h_c=d_c0 cos(beta). Therefore

$$
d_{cg}=\frac{d_{c0}\cos\beta}{\cos(\theta_g-\beta)},\qquad
\boxed{A_{\mathrm{dem},cg}
=A_{c0}\frac{\cos(\theta_g-\beta)}{\cos\beta}.}
$$

This preserves A_dem,c0=A_c0. It requires a visible forward intersection with the plane.

Equivalently, A_dem=A_c0[cos(theta)+tan(beta)sin(theta)]. For small theta in radians,

$$
A_{\mathrm{dem}}-A_{c0}
\approx A_{c0}\tan\beta\,\theta-\tfrac12 A_{c0}\theta^2.
$$

Plane misalignment therefore introduces an odd component while retaining negative quadratic curvature near nominal zero.

For beta=+5° and A_c0=4 D, demand is 4.015279 D at nominal +5°. **Demand has a maximum at the plane normal**, because that target is closest to the eye. Plane alignment alone does not explain a minimum of a positively curved accommodation-bias curve.

Beta describes target geometry. It must not automatically be equated with the optical-zero parameter introduced below. Prefer measured plane geometry; when unavailable, state beta=0 as an assumption and test its sensitivity separately. If targets are not on a common plane, calculate their distances from their actual coordinates.

Geometric demand is a stimulus label, not a measurement of physiological accommodation. Evaluate both barrel and tilt's accommodation-dependent coefficients at the corrected demand during mean calibration, as done in the follow-up test.

## 4. Correct tilt geometry: stretch and projective keystone

For the pinhole rotation convention in [Tilt.md](Tilt.md), after removing the central target's boresight displacement,

$$
\widetilde x=\frac{x\sec\alpha}{1+(\tan\alpha/f)y},\qquad
\widetilde y=\frac{y\sec^2\alpha}{1+(\tan\alpha/f)y}.
$$

Here alpha is the physical tilt angle in radians and f uses the same image-coordinate units as x and y. The associated local homography factors as

$$
H_c(\alpha)=
\underbrace{\begin{bmatrix}
\sec\alpha&0&0\\
0&\sec^2\alpha&0\\
0&0&1
\end{bmatrix}}_{\text{anamorphic stretch }S}
\underbrace{\begin{bmatrix}
1&0&0\\
0&1&0\\
0&\tan\alpha/f&1
\end{bmatrix}}_{\text{pure projective keystone }P}.
$$

Matrix action applies P first and S second. Their combined map follows barrel in our forward model.

| Effect | Leading spatial displacement | Small-tilt dependence |
|---|---|---|
| Boresight | Constant translation | Linear in tilt |
| Anamorphic stretch | Different multiples of x and y | Quadratic in tilt |
| Projective keystone | −q(xy, y²), before the axis stretch | Linear in tilt |
| Radial barrel | κ(x²+y²)(x, y) | Set by the accommodation law |

At 10°, the pinhole axis stretches are +1.5427% and +3.1091%. Their geometric-mean gain is +2.3229%; that gain is not generally the full transformed triangle's radius gain.

The simple sec/sec² coupling belongs to the stated pinhole geometry. P4 is a reflected pattern formed through an optical system. The [ray-tracing model](SimulationModel.md) supports independent accommodation-dependent axis stretches, but does not establish the pinhole coupling for these experimental triangles. Treat the exact pinhole law as a physical comparison model, not a set of mandatory experimental coefficients.

The distinction between a plane-to-image homography and an image-to-image rotation homography follows the [OpenCV homography derivation](https://docs.opencv.org/4.10.0/d9/dab/tutorial_homography.html). The boresight removal above is an additional image translation.

Boresight removal and triangle-centroid subtraction are different operations. A projective transform moves the triangle centroid even after the central target's boresight has been removed. The forward model must explicitly center its prediction, and coordinate inversion must restore its predicted centroid.

## 5. Relative barrel and the existing stretch/projective model

Let O_B and O_K be the barrel and combined-tilt origins in reference-pixel coordinates. Both are shared across all states. Set

$$
u_j=Q_j-O_B,\quad r_j^2=u_j\cdot u_j,\quad
\Delta A=A-A_{\mathrm{ref}},\quad
\kappa(A)=k_A\Delta A,
$$

$$
\boxed{V_j=B_A(Q_j)=Q_j+\kappa(A)r_j^2u_j.}
$$

k_A has units px⁻²/D. Use each vertex's radius about O_B. The simulation's equal-corner-radius reduction to one scalar multiplier is not valid for an arbitrary experimental triangle.

The barrel is an empirical incremental deformation of Q. A free origin and coefficient can trade off with other optical terms; neither is automatically an anatomical measurement.

For the completed baseline, with theta expressed in degrees,

$$
\begin{aligned}
s_x(A,\theta)&=1+\ell_x(A)\theta+\alpha_x(A)\theta^2,\\
s_y(A,\theta)&=1+\ell_y(A)\theta+\alpha_y(A)\theta^2,\\
q(A,\theta)&=\gamma(A)\theta,
\end{aligned}
\qquad c(A)=c_0+c_1\Delta A
$$

for each of the five amplitudes c.

**Anamorphic stretch is already represented by s_x and s_y.** Adding another independently fitted quadratic diagonal stretch would overlap with existing freedom at leading order. The remaining questions concern its orientation, its coupling to projection, and its parameter laws.

The scales are not constrained to be reciprocal. Their common size contribution must be retained. q has units px⁻¹, ell has degree⁻¹, alpha has degree⁻², and gamma has px⁻¹ degree⁻¹; accommodation slopes add D⁻¹.

## 6. Shared image-axis orientation: the immediate optical extension

The completed model fixes the stretch axes and projective direction to image x/y. Test one shared in-image orientation phi:

$$
z_j=R_\phi^\top(V_j-O_K),\qquad
\boxed{F_j=O_K+
R_\phi\frac{\operatorname{diag}(s_x,s_y)z_j}{1+qz_{jy}}.}
$$

The measured centered prediction is

$$
\boxed{\widehat X_j(A,\theta)=\mathcal C[F]_j.}
$$

Equivalently, the stretch matrix is R_phi diag(s_x,s_y) R_phi^T and the projective direction is R_phi e_y. Rotate both consistently. Phi=0 recovers the completed baseline exactly. Phi changes the optical axes in the image; it is not an extra rigid rotation of every output triangle.

Fit phi globally while refitting all fifteen existing optical parameters jointly. This gives sixteen shared parameters. The follow-up test preserves the fixed reference and saved gaze input. It improves conditional optical transfer, but leaves distant tilt origins and irregular accommodation offsets; see Section 11.

The dependence of accommodation enters both components:

$$
\partial_A F=(\partial_V K)\partial_A B+(\partial_A K)_V.
$$

Both terms belong in accommodation inference. With geometry-corrected demand, barrel also varies across gaze within a capture through A_dem,cg, even though it has no explicit gaze coefficient.

## 7. Optical-zero alignment and an extremum near +5°

Three alignment quantities have different meanings:

| Quantity | Meaning |
|---|---|
| beta | Nominal gaze direction normal to the target plane; affects target distance |
| theta_o | Nominal gaze direction corresponding to an effective optical symmetry zero |
| phi | Orientation of the optical stretch/projective axes in the image |

An observed bias of the form

$$
b(\theta)=b_c+C(\theta-\theta_o)^2
$$

has its minimum at theta_o if C>0, and its maximum there if C<0. Thus an extremum near +5° is consistent with an optical-zero shift. It does not establish the alignment or the sign of the underlying distortion.

For an unconstrained quadratic b_0+b_1 theta+b_2 theta², the candidate symmetry point is −b_1/(2b_2), when b_2 is appreciably nonzero. This is a descriptive initialization, not an independent measurement. The earlier shared-bias calibration explicitly encouraged a quadratic curve; its apparent minimum cannot establish an optical zero.

A local size error epsilon induces approximately

$$
\widehat A-A\approx
\frac{\epsilon(A,\theta)}{\partial_A\log r(\widehat X(A,\theta))},
\qquad
\epsilon=\log r(X)-\log r(\widehat X).
$$

The sign depends on accommodation sensitivity and whether the optical model under- or overpredicts size. A stretch minimum, a demand maximum, and an accommodation-bias minimum must not be treated as interchangeable.

### A consistent reference-relative shifted tilt law

Simply replacing theta by theta−theta_o in the current relative coefficients would destroy identity at the nominal reference. Instead, define an **absolute effective local homography with boresight removed**, H_abs(A,alpha), about a consistent pattern origin, and compose

$$
\boxed{H_{\mathrm{rel}}(A,\theta)
=H_{\mathrm{abs}}(A,\theta-\theta_o)
H_{\mathrm{abs}}(A,-\theta_o)^{-1}.}
$$

This is identity at nominal theta=0 for each A. It remains an empirical relative construction when applied to a measured P4 reference.

For the restricted local form

$$
H_{\mathrm{abs}}(A,\alpha)=
\begin{bmatrix}
s_x^{abs}(A,\alpha)&0&0\\
0&s_y^{abs}(A,\alpha)&0\\
0&q^{abs}(A,\alpha)&1
\end{bmatrix},
$$

let alpha=theta−theta_o and alpha_0=−theta_o. The coefficients to use in Section 6 are

$$
s_x=\frac{s_x^{abs}(A,\alpha)}{s_x^{abs}(A,\alpha_0)},\quad
s_y=\frac{s_y^{abs}(A,\alpha)}{s_y^{abs}(A,\alpha_0)},\quad
q=\frac{q^{abs}(A,\alpha)-q^{abs}(A,\alpha_0)}
{s_y^{abs}(A,\alpha_0)}.
$$

A small-tilt candidate uses even absolute stretches 1+c_x(A)alpha² and 1+c_y(A)alpha², and an odd projective law g(A)alpha. All angles must use the same units. The amplitudes can remain affine in accommodation. This creates a constrained alternative to the unrestricted odd/even baseline.

Do not append theta_o to an otherwise unrestricted linear/quadratic polynomial and claim a newly identified alignment: the shift can be absorbed into its coefficients. For a reference-normalized even quadratic stretch, the effective relative coefficients satisfy ell_x=−2 alpha_x theta_o and ell_y=−2 alpha_y theta_o. A common theta_o is testable through such shared restrictions. The exact reference-relative formula above should be used when fitting, including its accommodation dependence.

## 8. Exact size accounting

For centered RMS radius

$$
r(Z)^2=\frac13\sum_j\|Z_j-\operatorname{mean}Z\|^2,
$$

define, for the actual barrel output and combined tilt prediction,

$$
G_B=\frac{r(V)}{r(Q)},\qquad
G_K=\frac{r(F)}{r(V)},\qquad
m_{\mathrm{axis}}=\sqrt{s_xs_y},\qquad
G_{\mathrm{shape}}=\frac{G_K}{m_{\mathrm{axis}}}.
$$

Then

$$
\frac{r(\widehat X)}{r(Q)}
=G_B\,m_{\mathrm{axis}}\,G_{\mathrm{shape}}.
$$

G_shape includes anisotropy and projection evaluated on the triangle; it is not a measurement of projection alone. These ratios multiply. Percentage changes do not add. Large opposing component gains can leave a small net change and indicate sensitivity to the fitted decomposition.

When comparing gazes, reference each measured size to the **same capture's measured 0° mean**. Reference each modeled component to its corresponding model value for that capture's 0° condition.

With constant A within a capture, G_B cancels from relative gaze size. With corrected planar demands it generally does not. Recompute all contributions after the joint refit; old constant-demand keystone fractions cannot be reused as physical attribution.

Accommodation offsets use a different reference:

$$
b_{cg}=\operatorname{mean}_{i\in(c,g)}\widehat A_i-A_{\mathrm{dem},cg}.
$$

Do not silently replace this by accommodation relative to the estimated 0° value. A repeatable nonzero b(theta) remains an acceptable empirical outcome.

## 9. Inversion and framewise accommodation

The forward order is barrel → combined tilt → centering. The inverse order is centroid restoration → inverse combined tilt → inverse barrel.

For each trial (A,theta), recompute the full prediction and its uncentered centroid

$$
\mu(A,\theta)=\frac13\sum_jF_j.
$$

Rotate restored observed coordinates into the tilt axes:

$$
w_j=R_\phi^\top[X_j+\mu(A,\theta)-O_K].
$$

Invert the combined stretch/projective map analytically:

$$
z_{jy}=\frac{w_{jy}}{s_y-qw_{jy}},\qquad
z_{jx}=\frac{w_{jx}s_y}{s_x(s_y-qw_{jy})},\qquad
\widetilde V_j=O_K+R_\phi z_j.
$$

Now form v_j=V_tilde,j−O_B and r_v=||v_j||. Solve the radial equation numerically:

$$
r_v=r_u(1+\kappa r_u^2),\qquad
\widehat Q_j=O_B+\frac{r_u}{r_v}v_j.
$$

For r_v=0, use Q_hat=O_B. Select the nonnegative inner branch satisfying

$$
1+\kappa r_u^2>0,\qquad 1+3\kappa r_u^2>0.
$$

For negative kappa, reject mapped radii beyond the branch maximum. Check positive axis scales and valid forward/inverse projective denominators throughout the stated domain. Restore the predicted centroid rather than adding an arbitrary translation. Do not apply post-inverse alignment to conceal residual error.

With optics frozen and frame gaze supplied by the polynomial estimate, the current forward-inference convention is

$$
\boxed{\widehat A_i
=\arg\min_{A\ \mathrm{in\ the\ valid\ domain}}
\sum_j\|\widehat X_j(A,\widehat\theta_i)-X_{ij}\|^2.}
$$

This is a numerical one-dimensional state inversion of the calibrated nonlinear model. No new per-frame optical coefficients are fitted. Every trial A updates barrel, stretch, projective terms, and the centroid. State bounds or temporal terms must be declared by the inference experiment; calibration demand corrections do not themselves impose per-frame demand bounds.

## 10. Calibration and evaluation

For geometry-corrected mean calibration, fix A_cg to the target-geometry demand and theta_cg to the mean polynomial gaze. Fit shared optics using

$$
E_{\mathrm{forward}}=
\sqrt{\frac1{60}\sum_{c,g,j}
\|\widehat X_j(A_{cg},\theta_{cg})-\overline X_{cg,j}\|^2}.
$$

This is Euclidean vertex RMS in P1-corrected P4 pixels. Each of the twenty means has equal weight. An inverse RMS is a useful separate recovery diagnostic; it is not interchangeable with the forward objective.

Use CuPy float64 for vectorized optical evaluation and derivatives. The latest solver's fixed numerical units are R=191.56618404715252 px, A_span=3.63963963964 D, a=(A−A_ref)/A_span, and t=theta/(10°). R is an inherited numerical scale; it is not recomputed from every triangle or necessarily equal to the refined template's current radius.

Preserve both converged origin branches. Compare shared parameter behavior, the stability of the orientation/alignment parameters, capture-held-out predictions, residual vertex directions, and resulting accommodation offsets. Curve-only holdout after fitting optics on all captures is descriptive; it is not an independent optical holdout.

The focused sequence and its current status are:

1. Apply the known planar-target demand correction, with a declared plane normal, and refit the existing model jointly. Completed under the explicit beta=0° assumption.
2. Test the one shared image-axis orientation phi while refitting barrel, both origins, and the existing stretch/projective amplitudes. Completed; parameter ambiguity and remaining offsets limit interpretation.
3. If an optical-zero interpretation is needed, compare the constrained reference-relative theta_o law with the unrestricted odd/even model. Keep plane-normal alignment separate.
4. Examine remaining vertex deformation before introducing asymmetric barrel or another spatial term.

Judge these changes by repeatable parameter behavior and transfer across captures, with RMS retained as a geometry metric. Avoid penalizing offsets into a quadratic law and then treating that imposed shape as evidence for an optical mechanism.

## 11. What the completed experiment establishes

These results precede the planar-demand correction and the alignment extensions:

- The best completed joint fit has forward vertex RMS 1.271798 px, local rank 15/15, barrel origin (5.8920, −10.1716) px, and tilt origin (−7.1595, 80.0500) px.
- A second converged barrel-origin branch near (−2.2986, 267.4138) px has RMS 1.288670 px. Similar agreement with different origins limits physical attribution.
- The residual log-size trend does not transfer as a common quadratic: curve-only held-capture error is 0.455%, versus 0.447% for no trend. Both origin branches fail the stated repeatability check.
- Uniform-size direction accounts for 27.13% of Cartesian residual energy. The remaining 72.87% is orthogonal to it. Diagnostic rotation removes 49.35% of the size-restored residual energy.
- Anamorphic-axis gain and the remaining projective/shape size contribution substantially cancel. Their separate fitted values are conditional model quantities.

The [complete report](../experiments/archive/p4_keystone_common_size_joint/RESULTS.md) contains parameter tables and plots, and its [audit](../experiments/archive/p4_keystone_common_size_joint/results/audit.json) documents independent physical-pixel replay. These results motivate inspecting alignment; they do not prove that asymmetric barrel, rotation, or a shifted optical zero is the cause.

One capture per demand also confounds capture effects with accommodation dependence. Known target geometry should be incorporated, but geometric demand is still not physiological accommodation ground truth.

### Follow-up: accommodation-offset evidence

The [new experiment](../experiments/archive/p4_planar_demand_tilt_orientation/RESULTS.md) uses the same twenty means and fixed refined reference. Its calibration accommodation is geometric demand, and its accommodation diagnostics numerically invert the frozen full nonlinear model for each mean triangle with fixed mean gaze.

| Model | Mean accommodation offset RMS (D) | Forward vertex RMS (px) |
|---|---:|---:|
| Original constant-demand model, offsets referenced to corrected demand | 0.176531 | 1.271798 |
| Corrected demand, joint optical refit | 0.173628 | 1.271455 |
| Corrected demand plus shared orientation, joint optical refit | 0.168715 | 0.793512 |

Vertex agreement improves much more than accommodation-offset explanation. Relative to the corrected-demand model, orientation improves absolute offsets at 9/20 fixations and worsens them at 11/20. C3 at −5° retains a +0.551627 D offset. The known geometric demand change is at most 0.060769 D.

The selected orientation is 41.733°, with conditional capture-held-out fits spanning 40.109°–43.713°. However, its tilt origin is (−14579, 16526) px, and five other starts reach the iteration limit while origins drift as far as approximately 2.25 million pixels. Local rank 16/16 does not establish a physical origin or globally unique decomposition.

Descriptive residual-offset laws are evaluated without feeding them back into optics. For the oriented model, a shared gaze quadratic predicts held-capture offsets with RMS 0.195758 D; allowing its amplitudes to be affine in accommodation gives 0.246251 D, compared with 0.168715 D for zero prediction. These curve holdouts are conditional on optics calibrated with all captures. They do not establish a repeatable shared quadratic offset law.

The next inference should focus on physically constrained tilt/reference conventions and the remaining accommodation-offset pattern. The fitted image-axis orientation is distinct from theta_o; this test does not identify the optical-zero shift or explain an extremum at +5° through alignment alone. The [offset audit](../experiments/archive/p4_planar_demand_tilt_orientation/results/offset_audit.json) includes independent CPU state inversion and parameter stability checks.

## 12. Corrections to the tilt notes and numerical checks

The pinhole homography in Sections 1–2 of [Tilt.md](Tilt.md) is consistent with its stated rotation convention. Its exact-frequency expressions and some finite-angle wavefront claims need the following qualifications.

For c=cos(alpha), s=sin(alpha), and the boresight-shifted sensor coordinates (x_tilde,y_tilde), set

$$
E=1-\frac{sc}{f}\widetilde y.
$$

Then the projective denominator in Section 4 is D=1/E, and

$$
x=\frac{c\widetilde x}{E},\qquad
y=\frac{c^2\widetilde y}{E},\qquad
J^{-1}=
\begin{bmatrix}
c/E & sc^2\widetilde x/(fE^2)\\
0 & c^2/E^2
\end{bmatrix}.
$$

Local spatial frequency transforms by J^{-T}. In particular,

$$
f'_x=\frac{c}{E}f_{x0},\qquad
f'_y=\frac{sc^2\widetilde x}{fE^2}f_{x0}
+\frac{c^2}{E^2}f_{y0}.
$$

At the origin, f'_y=c² f_y0, not c³ f_y0. These expressions follow from the stated centered homography and supersede the corresponding frequency equations in Tilt.md.

At finite tilt, the spatial quadratic terms are −q sec(alpha) xy and −q sec²(alpha) y². A single coefficient multiplying (xy,y²) describes their leading small-tilt limit, not the full finite-angle expression. Plane-symmetric aberration terms such as ||H||² e_y are additional spatial modes, distinct from linear anamorphic stretch. Their existence in a general expansion does not establish their magnitude in this P4 dataset.

The separation of anamorphic and higher spatial distortion terms is consistent with [Peterson and Sasián's plane-symmetric aberration analysis](https://wp.optics.arizona.edu/jsasian/wp-content/uploads/sites/33/2023/04/Plane-Symmetric-Systems-Structural.pdf); the experimental coefficients still require calibration.

For any implemented candidate, verify reference identity, parameter units, centering, forward/inverse order, independent CPU/GPU replay, derivative agreement, and physical branches over the actual corrected state range. A corrected demand can fall slightly below A_ref, so re-evaluate domain coverage rather than reusing old limits. Check alternative origins and report whether each validation concerns numerical consistency, predictive transfer, or physical interpretation.

## Sources

- [Tilt derivation notes](Tilt.md), with the corrections stated above.
- [P4 ray-tracing model](SimulationModel.md).
- [Current mean-triangle experiment and model](../experiments/archive/p4_keystone_common_size_joint/README.md).
- [Planar-demand and shared-orientation follow-up](../experiments/archive/p4_planar_demand_tilt_orientation/README.md).
- [Original reference-based experimental model](../experiments/archive/p4_reference_based_model_fit/README.md).
- [OpenCV: plane homographies and pure camera rotation](https://docs.opencv.org/4.10.0/d9/dab/tutorial_homography.html).
- [Peterson and Sasián, Structural aberration coefficients for freeform imaging systems, 2023](https://wp.optics.arizona.edu/jsasian/wp-content/uploads/sites/33/2023/04/Plane-Symmetric-Systems-Structural.pdf), especially the distinction between anamorphic and quadratic spatial distortion terms.
