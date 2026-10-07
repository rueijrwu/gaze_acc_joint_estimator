# Optical Measurement and Identifiability Framework: Three P1/P4 Pairs

## Scope and status

This document extends the earlier two-pair horizontal framework to three corresponding P1/P4 reflections in two image dimensions. It specifies the geometry and identification questions for `exp5_full`; it does not claim that a full-position estimator has been implemented or validated. The frozen two-channel quadratic estimator remains the baseline described in [HANDOFF.md](HANDOFF.md). No new fit, detector rule, state prior, or normalization is introduced into that implementation by this theoretical extension.

The required baseline observables are

$$
d_x=\frac{\operatorname{mean}(P4_x)-\operatorname{mean}(P1_x)}
          {\sqrt{\operatorname{area}(P1)}},
\qquad
\rho_4=\frac{\operatorname{area}(P4)}{\operatorname{area}(P1)}.
$$

Here area means the unsigned area of the triangle formed by the three reflection centers, not the summed area of the three detected blobs. The square root belongs in the displacement denominator; it does **not** belong in the definition of `rho_4`.

The derivations below distinguish exact geometric identities, optional forward-model assumptions, and empirical evidence. Numerical convergence, statistical identification, and physiological validation remain distinct requirements.

## 1. Correspondence, coordinates, and states

Let

$$
\mathbf p_j=\mathbf P_{1,j}(\theta,A;\eta),\qquad
\mathbf q_j=\mathbf P_{4,\pi(j)}(\theta,A;\eta),\qquad j=1,2,3,
$$

be corresponding P1/P4 image positions in pixels, with x right and y down. The permutation `pi` reorders the stored P4 slots into P1 correspondence. The handoff records the zero-based mapping `pair_index = [2,1,0]`; read each payload's metadata rather than independently sorting the two triangles and assuming that order establishes identity. Centroids and unsigned areas do not depend on point order, but pointwise displacements and the triangle map do.

The baseline state is still $x=(\theta,A)$: horizontal gaze in degrees and accommodation in diopters. Measuring vertical coordinates does not, by itself, establish a calibrated vertical-gaze state. A later model for $(\theta_x,\theta_y,A)$ would require the corresponding calibration and identification analysis. The nuisance vector $\eta$ can describe supported geometry, pose, or imaging effects; it is not permission to introduce unrestricted framewise nuisance parameters.

Use $\mathcal A_i$ for triangle area, to distinguish it from accommodation $A$.

Define centroids and edge matrices

$$
\mathbf c_1=\frac13\sum_j\mathbf p_j,\qquad
\mathbf c_4=\frac13\sum_j\mathbf q_j,
\qquad \mathbf m=\mathbf c_4-\mathbf c_1,
$$

$$
E_1=[\mathbf p_2-\mathbf p_1\;\;\mathbf p_3-\mathbf p_1],\qquad
E_4=[\mathbf q_2-\mathbf q_1\;\;\mathbf q_3-\mathbf q_1].
$$

Signed and unsigned areas are

$$
a_i=\tfrac12\det E_i,\qquad \mathcal A_i=|a_i|,
\qquad \ell_1=\sqrt{\mathcal A_1}.
$$

Require finite, appropriately validated coordinates and a P1 triangle whose area is sufficiently separated from zero relative to localization uncertainty. Triangle conditioning also matters: a long, thin triangle can make edge-based inversion unstable. The absolute-area function is not differentiable at a sign reversal. Preserve signed-area/parity diagnostics and work on declared correspondence/orientation branches.

## 2. Baseline observables and their meaning

The normalized displacement vector and area ratio are

$$
\boxed{\mathbf d=\frac{\mathbf m}{\ell_1}}
\qquad\text{and}\qquad
\boxed{\rho_4=\frac{\mathcal A_4}{\mathcal A_1}
              =\frac{|\det E_4|}{|\det E_1|}}.
$$

Thus

$$
\mathbf y_0=[d_x,\rho_4]^\top,
\qquad
\mathbf y_{\rm centroid}=[d_x,d_y,\rho_4]^\top.
$$

The second vector is a possible extension, not the full six-point geometry. All these entries are dimensionless. The old sign convention can be retained as $e=c_{1,x}-c_{4,x}$, $e_n=e/\ell_1$, and $d_x=-e_n$.

Calling $d_x$ the gaze channel and $\rho_4$ the accommodation-sensitive channel does not assert separability. In general,

$$
d_x=f_d(\theta,A;\eta),\qquad
\rho_4=f_\rho(\theta,A;\eta),
$$

and both channels can depend on both states. P1 area need not be constant or state-independent merely because it is used for normalization.

## 3. What the normalization removes

Under the same positive isotropic image scale and additive translation for every point,

$$
\mathbf p'_j=\lambda\mathbf p_j+\mathbf b,\qquad
\mathbf q'_j=\lambda\mathbf q_j+\mathbf b,\qquad \lambda>0,
$$

we have

$$
\mathbf m'=\lambda\mathbf m,\qquad
\mathcal A'_i=\lambda^2\mathcal A_i,\qquad
\ell'_1=\lambda\ell_1.
$$

Consequently $\mathbf d'=\mathbf d$ and $\rho'_4=\rho_4$. This explains why a displacement is divided by square-root area, whereas an area is divided by area.

A shared proper image rotation $R$ instead gives $\mathbf d'=R\mathbf d$ and $\rho'_4=\rho_4$. Therefore horizontal displacement remains tied to the image x-axis: area normalization does not remove rotation.

More generally, for an identical nonsingular affine image transformation $B\mathbf P+\mathbf b$ applied to all six points,

$$
\rho'_4=\rho_4,\qquad
\mathbf d'=\frac{B\mathbf d}{\sqrt{|\det B|}}.
$$

The area ratio therefore has a stronger common-affine invariance than the displacement vector. This does not establish invariance to a general perspective transformation, differential P1/P4 magnification, physical eye translation, detector bias, or different optical changes in the two reflections.

Optional rotation normalization must be explicit. For example, let $Q$ be a proper orthonormal frame constructed from the first P1 edge; use $Q^\top\mathbf d$ and correspondingly rotated geometry. This removes one common rotation degree of freedom, but may also remove useful state-dependent orientation. It is not part of the baseline normalization.

## 4. Three pairwise displacements: common motion and deformation

For each matched pair define

$$
\mathbf u_j=\frac{\mathbf q_j-\mathbf p_j}{\ell_1}.
$$

Their mean is exactly the normalized centroid displacement:

$$
\boxed{\mathbf d=\frac13\sum_{j=1}^3\mathbf u_j}.
$$

Separate common displacement from differential motion by

$$
\mathbf w_j=\mathbf u_j-\mathbf d,\qquad
\sum_j\mathbf w_j=0.
$$

There are six scalar matched-displacement coordinates: two common components and four differential components. A nonredundant representation is $(\mathbf d,\mathbf w_1,\mathbf w_2)$, with $\mathbf w_3=-\mathbf w_1-\mathbf w_2$.

Retain the normalized P1 geometry

$$
\mathbf r_j=\frac{\mathbf p_j-\mathbf c_1}{\ell_1},
\qquad \sum_j\mathbf r_j=0,
\qquad \operatorname{area}(\mathbf r_1,\mathbf r_2,\mathbf r_3)=1.
$$

The centered normalized P4 points then satisfy

$$
\frac{\mathbf q_j-\mathbf c_4}{\ell_1}=\mathbf r_j+\mathbf w_j,
$$

so

$$
\boxed{
\rho_4=\frac12\left|
\det\left[
(\mathbf r_2-\mathbf r_1)+(\mathbf w_2-\mathbf w_1)\;\;
(\mathbf r_3-\mathbf r_1)+(\mathbf w_3-\mathbf w_1)
\right]\right|.}
$$

A common P4 displacement changes $\mathbf d$ but not its triangle area. Differential motion can change area, orientation, and shape; the area ratio retains only one scalar aspect of that differential geometry.

When both normalized P1 geometry and all pair displacements are retained, `rho_4` is already determined. It must not be counted as an additional independent measurement. Conversely, pair displacements alone do not generally determine `rho_4` if the P1 triangle geometry has been discarded.

The old two-pair identity relating a difference of normalized crossed displacements to a separation ratio cannot be obtained by replacing that separation ratio with a triangle-area ratio.

## 5. Exact triangle-map representation

For a nondegenerate P1 triangle define

$$
T=E_4E_1^{-1}.
$$

Then, at the three measured vertices,

$$
\boxed{\mathbf q_j=\mathbf c_4+T(\mathbf p_j-\mathbf c_1)}
$$

and therefore

$$
\boxed{\mathbf u_j=\mathbf d+(T-I)\mathbf r_j}.
$$

This is an exact coordinate representation for three corresponding points, not an assumption that the optical system is globally affine. The existence of an affine map determined by three noncollinear source points is the standard three-point construction described in the OpenCV affine-transformation documentation.

Taking determinants gives the central relation

$$
\boxed{\rho_4=|\det T|}.
$$

The signed ratio $a_4/a_1=\det T$ also records the relative orientation branch; unsigned area discards its sign.

For $\rho_4>0$, factor

$$
T=\sqrt{\rho_4}\,K,\qquad |\det K|=1.
$$

The four continuous components of $T$ separate into one area-ratio coordinate and three area-preserving deformation coordinates. On an orientation-preserving branch, one possible explicit parameterization is

$$
T=\sqrt{\rho_4}\,R(\psi)\exp\!\begin{bmatrix}h_1&h_2\\h_2&-h_1\end{bmatrix}.
$$

Here $\psi$ describes relative rotation, while $h_1,h_2$ describe determinant-one stretch/shear. On an orientation-reversing branch, include a fixed reflection factor rather than silently changing point correspondence. These are geometric descriptors, not automatically identified anatomical parameters.

Only under a similarity relation $T=kR$, with orthogonal $R$ and $k>0$, is every corresponding linear separation multiplied by $k$ and $\rho_4=k^2$. Thus $\sqrt{\rho_4}$ is then a linear scale ratio, but it is not the defined observable `rho_4`. General triangles need not be similar.

For example, $T_1=I$ and $T_2=\operatorname{diag}(2,1/2)$ have the same area ratio. With the same centroid displacement they produce identical $(d_x,d_y,\rho_4)$, yet different P4 configurations. The two-channel baseline cannot distinguish them.

For nonsingular $T$, the first-order area response is

$$
\delta\rho_4=\rho_4\operatorname{tr}(T^{-1}\delta T).
$$

Relative deformation directions with zero trace in this expression are locally invisible to the area channel.

**An unconstrained affine fit to these same three pairs has no residual redundancy.** It can fit all three P4 vertices exactly, including incorrectly selected points. Useful residual checks must instead use a restricted geometry model, a shared calibrated state-dependent model, genuinely withheld observations, or additional measurements.

## 6. Independent geometric dimensions

Six 2D points supply 12 scalar coordinates. Removing a common image translation removes two degrees of freedom. Removing a common positive image scale removes one more. On a nondegenerate correspondence branch, the remaining normalized geometry has nine continuous degrees of freedom. Removing an additional common rotation would leave eight.

An exact translation-free raw basis is

$$
\mathbf z=[\mathbf m^\top,\operatorname{vec}(E_1)^\top,
                          \operatorname{vec}(E_4)^\top]^\top\in\mathbb R^{10}.
$$

A normalized representation is

$$
H_1=E_1/\ell_1,\qquad H_4=E_4/\ell_1,
\qquad (\mathbf d,H_1,H_4),
$$

with the exact constraint $|\det H_1|=2$. Its ten stored components therefore have only nine continuous degrees of freedom. Equivalently use $(\mathbf d,H_1,T)$, retaining that same constraint.

The dimension accounting is

| Component | Continuous geometric dimensions |
|---|---:|
| Normalized centroid displacement $\mathbf d$ | 2 |
| Relative triangle map $T$ | 4 |
| Normalized P1 triangle geometry $H_1$ | 3 |
| Total | 9 |

With P1 geometry retained as context, $(\mathbf d,T)$ describes six relative P4 components. The baseline retains $d_x$ and $|\det T|$; the four additional relative components can be represented by $d_y$ and the three components of $K$. Three further dimensions describe normalized P1 orientation/shape.

These are geometric dimensions, not counts of independent physiological information or statistically independent observations. Conditioning on measured P1 geometry does not make its localization error disappear.

For completeness, all crossed displacements obey

$$
\frac{\mathbf q_j-\mathbf p_k}{\ell_1}=\mathbf d+T\mathbf r_j-\mathbf r_k.
$$

Adding every edge or crossed pair to an already complete representation does not create more information. An implementation can use a nonredundant coordinate chart or a correctly constrained covariance; a numerical ridge does not create an additional measurement dimension.

## 7. Extending the raw optical model

A candidate local point model, extending the old coefficient framework, is

$$
\mathbf P^*_{i,j}(\theta,A;\eta)
=[G_{i,j}(\eta)+a\,\Gamma_{i,j}(\eta)]\boldsymbol\Theta(\theta),
\qquad
\boldsymbol\Theta=[\theta^3,\theta^2,\theta,1]^\top,
\qquad a=\phi(A),
$$

where each coefficient matrix is $2\times4$. A shared image scale and translation may multiply/add to the resulting points. This is an optional empirical approximation, not a claim that every coordinate follows this polynomial or that all its coefficients are identifiable.

Write the resulting centroid displacement as $\mathbf M(\theta,A;\eta)$ and the two columns of each edge matrix as $\mathbf e_i$ and $\mathbf f_i$. Define

$$
\Delta_i(\theta,A;\eta)=\det[\mathbf e_i\;\mathbf f_i].
$$

The induced baseline model is then

$$
\boxed{d_x=\frac{M_x}{\sqrt{|\Delta_1|/2}},\qquad
       d_y=\frac{M_y}{\sqrt{|\Delta_1|/2}},\qquad
       \rho_4=\frac{|\Delta_4|}{|\Delta_1|}.}
$$

Even if every point coordinate is affine in $a$, the triangle determinant is generally quadratic in $a$. Specifically, if

$$
\mathbf e_i=\mathbf e_{i,0}+a\mathbf e_{i,1},\qquad
\mathbf f_i=\mathbf f_{i,0}+a\mathbf f_{i,1},
$$

then

$$
\begin{aligned}
\Delta_i={}&\det[\mathbf e_{i,0}\;\mathbf f_{i,0}]\\
&+a\bigl(\det[\mathbf e_{i,1}\;\mathbf f_{i,0}]
        +\det[\mathbf e_{i,0}\;\mathbf f_{i,1}]\bigr)\\
&+a^2\det[\mathbf e_{i,1}\;\mathbf f_{i,1}].
\end{aligned}
$$

If the point coordinates are cubic in gaze, these determinants can have degree up to six in gaze. Accordingly, on fixed sign branches, the normalized displacement generally has a square-root polynomial denominator, and the area ratio is a ratio of determinant polynomials. Neither is automatically the old two-pair rational model or a low-degree normalized polynomial.

Choosing $\phi(A)=A^p$, $\log(1+A)$, or another coordinate is a modeling decision. They are not universally equivalent. For powers with $0<p<1$, the derivative at $A=0$ requires explicit treatment. A logarithm uses a dimensionless accommodation argument: code written as `log(1+A)` takes the numerical value of accommodation in diopters, equivalently $\log(1+A/(1\,\mathrm D))$.

For either state variable $z\in\{\theta,A\}$, normalization gives

$$
\partial_z\mathbf d
=\frac{\partial_z\mathbf m}{\ell_1}
 -\frac{\mathbf d}{2\mathcal A_1}\partial_z\mathcal A_1,
$$

$$
\partial_z\rho_4
=\frac{\partial_z\mathcal A_4}{\mathcal A_1}
 -\frac{\rho_4}{\mathcal A_1}\partial_z\mathcal A_1.
$$

P1-area variation can therefore affect both normalized channels. Setting its gaze or accommodation derivative to zero requires evidence, not just a corneal-stability argument.

## 8. Frozen baseline versus candidate full-position model

The current frozen exp5 model has 13 coefficients. With the numerical gaze/accommodation conventions in the handoff, $t=\theta/15$ and $L_A=\log(1+A)$,

$$
f_d=(b_0+b_1A)+(s_0+s_1L_A)t+(c_{20}+c_{21}L_A)t^2+c_3t^3,
$$

$$
f_\rho=(r_0+r_1t+r_2t^2)+L_A(r_3+r_4t+r_5t^2).
$$

These are empirical models of the triangle-normalized observables, not exact consequences of the point-level polynomial above. The frozen quadratic model remains the reference; the historical 14-coefficient knot model and its reduced-demand experiments are not the current exp5 baseline.

A candidate relative full-position model could instead predict

$$
\mathbf d=\mathbf D(\theta,A;\eta),\qquad
T=\mathcal T(\theta,A;\eta),
$$

using fixed, globally calibrated coefficient functions. Its predicted area ratio must then be $|\det\mathcal T|$. Alternatively it can parameterize area ratio and determinant-one deformation separately. Do not fit `rho_4` and all entries of $T$ as unrelated independent observations of the same frame.

Preserve P1 geometry to assess nuisance variation or model it explicitly when justified. Model capacity, positivity/orientation restrictions, calibration support, and any nuisance states must be declared before evaluating improvement. No particular capacity is selected by this theory alone.

## 9. Coordinate noise and normalized covariance

Stack the six measured 2D points into $\mathbf P\in\mathbb R^{12}$ with a declared localization covariance $\Sigma_P$. It may contain correlations within or across detected points. For a differentiable feature map $\mathbf y=h(\mathbf P)$, first-order propagation gives

$$
\Sigma_y\approx J_h\Sigma_PJ_h^\top,\qquad
J_h=\frac{\partial h}{\partial\mathbf P}.
$$

This is the multivariate form of the first-order law of propagation of uncertainty described by NIST. It is an approximation for nonlinear normalization, not a guarantee of Gaussian errors.

For a triangle with vertices $(x_j,y_j)$, the signed-area gradients are

$$
\nabla_{\mathbf P_1}a=\tfrac12(y_2-y_3,\;x_3-x_2)^\top,
$$

$$
\nabla_{\mathbf P_2}a=\tfrac12(y_3-y_1,\;x_1-x_3)^\top,
\qquad
\nabla_{\mathbf P_3}a=\tfrac12(y_1-y_2,\;x_2-x_1)^\top.
$$

For unsigned area on a nonzero sign branch, multiply by $\operatorname{sign}(a)$. The baseline perturbations are

$$
\delta\mathbf d=\frac{\delta\mathbf m}{\ell_1}
                -\frac{\mathbf d}{2}\frac{\delta\mathcal A_1}{\mathcal A_1},
\qquad
\delta\rho_4=\frac{\delta\mathcal A_4}{\mathcal A_1}
             -\rho_4\frac{\delta\mathcal A_1}{\mathcal A_1}.
$$

All pair displacements share the same noisy P1-area denominator, and the area ratio shares those coordinates. They cannot generally be treated as independent observations. Near degeneracy, ratio bias, heavy tails, and branch errors can invalidate first-order approximations; preserve validity flags and assess such cases separately.

For a fixed invertible linear basis change $C$, transform covariance as $\Sigma'=C\Sigma C^\top$. This exactly preserves Mahalanobis residuals when precision is transformed accordingly. A general nonlinear feature transformation with Jacobian-propagated covariance gives only local equivalence, not exact equality of arbitrary finite-residual least-squares objectives.

Estimate noise and preprocessing using training data only. Do not add deterministic functions of retained features as independent residual blocks or interpret a covariance ridge as new information.

## 10. Local inversion, nuisance variables, and residual dimensions

Let a nonredundant selected feature vector obey

$$
\mathbf y=F(x,\eta;\beta)+\epsilon,
\qquad x=(\theta,A),
$$

with calibrated coefficients $\beta$ held fixed for framewise inversion. Define

$$
J_x=[\partial_\theta F\;\partial_A F],\qquad W=\Sigma_y^{-1}.
$$

Without fitted nuisance states or priors, the regular local least-squares perturbation is

$$
\delta x\approx(J_x^\top WJ_x)^{-1}J_x^\top W
                 (\delta\mathbf y-\delta F_{\rm model}).
$$

For two channels this reduces to the ordinary inverse-Jacobian expression when $J_x$ is nonsingular. Biased geometry can transfer error into both states. Inspect the singular values of $LJ_x\operatorname{diag}(u_\theta,u_A)$, where $L^\top L=W$ and the degree/diopter increments are declared; unscaled condition numbers depend on units.

The nonlinear two-channel version of the earlier fixed-displacement test is, where $\partial_\theta f_d\ne0$,

$$
\left.\frac{df_\rho}{dA}\right|_{f_d}
=\partial_A f_\rho
 -\partial_\theta f_\rho\frac{\partial_A f_d}{\partial_\theta f_d},
\qquad
\det J_x=(\partial_\theta f_d)
          \left.\frac{df_\rho}{dA}\right|_{f_d}.
$$

This relation does not assume a gaze-linear displacement model. Vanishing determinants or folds require branch diagnostics; local full rank does not ensure global uniqueness.

If nuisance variables are estimated, use

$$
\widetilde J_x=LJ_x,\qquad \widetilde J_\eta=LJ_\eta,
$$

$$
I_{x\mid\eta}=\widetilde J_x^\top
\left(I-\widetilde J_\eta\widetilde J_\eta^+\right)\widetilde J_x.
$$

Added geometry helps separate gaze/accommodation only insofar as its state response cannot be reproduced by allowed nuisance variation. A shared-scale gauge convention fixes a global ambiguity, not an independently unknown framewise scale. Calibration-coefficient uncertainty must also be propagated or assessed rather than hidden by the fixed-coefficient framewise analysis.

For $k$ nonredundant, whitened feature directions, the local residual dimension at fixed calibrated coefficients is

$$
k-\operatorname{rank}[\widetilde J_x\;\widetilde J_\eta].
$$

Two channels and a regular two-state inverse leave no residual direction. Six relative components $(\mathbf d,T)$ can leave four if only two states are fitted and the state Jacobian has rank two. Those are model-checking directions, not four independently validated physiological measurements. Additional nuisance freedoms consume residual directions according to their joint rank; arbitrary framewise triangle maps eliminate the intended restriction.

## 11. Calibration and honest geometric checks

Nominal gaze and accommodation demand remain soft fixation-mean anchors, not independent framewise measurements of actual gaze/accommodation. Correlated whitening, robust loss, regularization, and temporal priors can stabilize a fit but do not remove systematic optical-model error or supply independent physiological evidence. A prior built from an earlier fit to the same data is model-derived information and requires sensitivity analysis, including its removal.

A meaningful comparison keeps training support, fold construction, preprocessing, and model capacity explicit. Exclude withheld observations from covariance estimates, learned initialization policies, and source models used to construct priors. Examine inverse ambiguity, bounds, extrapolation, uncertainty, prediction residuals, coverage, and failure rates rather than selecting on training residual or nominal-anchor agreement alone.

Leave-one-point-out tests must remove the held-out point from every feature used for estimation. Holding out a P4 point prohibits using the full P4 centroid, area, or triangle map that contains it. Holding out a whole corresponding pair also prohibits using the original three-point P1-area normalizer: the remaining P1 pair does not define that triangle. Declare an alternative training-fixed or subset-supported representation and verify that the retained observations identify the states and nuisance parameters. Such a check is not simply the original baseline with one coordinate omitted.

Detector-selected triangle consistency can reuse detector assumptions. Consult raw candidates and selection metadata when interpreting geometric agreement. Captures 5/6, which lack reviewed target intervals according to the handoff, support unlabeled trajectory and geometric-consistency checks, not an independent physiological-accuracy claim.

## 12. Central result and historical boundary

The extended theory is summarized by

$$
\boxed{
\ell_1=\sqrt{\mathcal A_1},\qquad
\mathbf d=\frac{\mathbf c_4-\mathbf c_1}{\ell_1},\qquad
T=E_4E_1^{-1},\qquad
\rho_4=\frac{\mathcal A_4}{\mathcal A_1}=|\det T|,
\qquad
\mathbf u_j=\mathbf d+(T-I)\mathbf r_j.
}
$$

The baseline observes horizontal common displacement and relative triangle area. Three corresponding 2D pairs additionally provide vertical common displacement, relative orientation and area-preserving deformation, together with normalized P1 geometry. Whether those additional components improve state estimation, nuisance discrimination, or failure detection is an empirical identification question; their existence alone does not answer it.

The earlier two-P1/two-P4 horizontal derivations remain available in Git history (previous `Theory.md` blob `cace05e50b242052d3d8b7727a90c592b3dc2f06`). Their separation ratio, crossed-pair identities, 14-coefficient knot model, reduced-demand experiments, and historical implementation links must not be treated as the definitions or status of this three-point exp5 model. Their general covariance, nuisance, calibration, and validation principles are retained above.

### Source notes

- Repository [HANDOFF.md](HANDOFF.md): current triangle observables, correspondence convention, frozen 13-coefficient model, data scope, and implementation status.
- Previous repository `Theory.md`, blob identified above: two-pair framework and general identification/validation principles extended here.
- OpenCV official documentation, *Affine Transformations*: standard affine construction from three corresponding points; the determinant and normalized-pair identities here are derived explicitly above.
- NIST, *Uncertainty of Measurement Results: Combining uncertainty components*: first-order Taylor propagation with sensitivity coefficients and input covariances.
