# Accommodation-response theory: logarithmic and shifted-power alternatives

**Status:** PROPOSED, documentation only; no new estimator or real-data result is asserted.  
**Repository / branch:** `rueijrwu/gaze_acc_joint_estimator` / `exp5_full`  
**Source snapshot inspected:** `b1a37f92727304da5726b54351545d32421b56bf`  
**Date:** 2026-10-07  
**Companion:** [ACCOMMODATION_RESPONSE_PLAN.md](ACCOMMODATION_RESPONSE_PLAN.md)

## 0. Scope and separation from the ongoing work

This is an additive research design for changing the accommodation-response basis. It does not replace [Theory.md](Theory.md), [ESTIMATOR_PLAN.md](ESTIMATOR_PLAN.md), [LATEST_RESULTS_AUDIT.md](LATEST_RESULTS_AUDIT.md), or their current implementation work. The user reports that implementation of the latest audit remains in progress. Do not treat this document as a request to interrupt that work, change its experiments, or declare its fixes complete.

The isolated question is:

> With the same three-pair geometry, calibration constraints, response capacity, and measurement channels, does replacing the logarithmic accommodation basis improve prediction of an excluded P4 point and agreement among subset-inferred states?

The main scientific metric remains the three-way cross-check. Optimizer cost and proximity to nominal accommodation demand are not substitutes. The current source uses `log1p(A)` in both conditional capacities; the proposed alternatives have not been fitted on these captures. [R1-R4]

## 1. Unchanged measurement and physical-state contract

There are three P1 points and three corresponding P4 points, each in two dimensions. Read the payload correspondence; the recorded zero-based map is `[2,1,0]`. Write mapped positions as $\mathbf p_j,\mathbf q_j$, for $j=1,2,3$.

The calibration protocol is

$$
\theta_x^{nom}\in\{-10,-5,0,5,10\}^{\circ},\qquad \theta_y^{nom}=0^{\circ}.
$$

The estimated state remains $x=(\theta_x,A)$, in degrees and diopters. Zero nominal vertical gaze is a protocol constraint, not proof of exactly zero instantaneous vertical eye motion. Image-y measurements are retained; they can respond to horizontal gaze and accommodation. This proposal does not identify a third, vertical-gaze state. [R2,R3]

Define

$$
\mathcal A_1=\tfrac12\left|\det[\mathbf p_2-\mathbf p_1\;\;\mathbf p_3-\mathbf p_1]\right|,
\quad \ell_1=\sqrt{\mathcal A_1},\quad \mathbf c_1=\tfrac13\sum_j\mathbf p_j,
$$

$$
\mathbf r_j=\frac{\mathbf p_j-\mathbf c_1}{\ell_1},\qquad
\mathbf v_j=\frac{\mathbf q_j-\mathbf c_1}{\ell_1}.
$$

P1 remains noisy measured context, not a separately fitted state-response block. Predict

$$
\widehat{\mathbf v}_j=\mathbf D_\lambda(x)+T_\lambda(x)\mathbf r_j,
\qquad
\widehat{\mathbf q}_j=\mathbf c_1+\ell_1\widehat{\mathbf v}_j.
$$

The original summaries remain exactly

$$
d_x=\frac{c_{4,x}-c_{1,x}}{\sqrt{\mathcal A_1}},\qquad
\rho_4=\frac{\mathcal A_4}{\mathcal A_1},\qquad
\widehat\rho_4=|\det T_\lambda|.
$$

They are derived diagnostics, not extra independent residuals. Keep all six P4 coordinate residuals, P1 uncertainty propagation, nondegeneracy checks, and correspondence/parity diagnostics. Do not center on the measured P4 centroid, normalize away expected distortion, or replace the P1 normalizer in this experiment. [R1,R3]

## 2. The accommodation assumption being tested

At fixed horizontal gaze, the present $D_y$ and four entries of $T$ have the form

$$
f(A)=c_0+c_1\log(1+A/A_*),\qquad A_*=1\,\mathrm D.
$$

Thus

$$
f'(A)=\frac{c_1}{A_*(1+A/A_*)}.
$$

Unless $c_1=0$, the magnitude of this conditional sensitivity decreases with accommodation. $D_x$ additionally contains a separate linear-$A$ term, so the same restriction does not apply to its complete response. Moving from 27 to 37 coefficients changes gaze curvature, not this basic accommodation law. [R1]

The hypothesis is that a different accommodation curvature may improve the sampled optical mapping. It is not a claim that eye optics obey a known power law, that the logarithm is wrong, or that more response slope is intrinsically better. Coefficients also set amplitude. A new basis may fail to address capture/context discrepancy or localization bias.

## 3. Proposed shifted-power family

Use dimensionless accommodation $a=A/A_*$ with the fixed physical scale $A_*=1\,\mathrm D$. Define one globally shared exponent $\lambda$:

$$
\phi_\lambda(A)=
\begin{cases}
\dfrac{(1+a)^\lambda-1}{\lambda}, & \lambda\ne0,\\[6pt]
\log(1+a), & \lambda=0.
\end{cases}
$$

This is the shifted Box-Cox family implemented by `scipy.special.boxcox1p(a, lambda)`. Here it is a deterministic forward-response basis, not a normality transform of labels. [E1]

| Exponent | Exact response basis | Conditional sensitivity shape |
|---:|---|---|
| 0 | $\log(1+a)$ | Logarithmic reference |
| 0.5 | $2(\sqrt{1+a}-1)$ | Less rapid flattening than log |
| 1 | $a$ | Linear |
| 2 | $a+a^2/2$ | Increasing slope |

All candidates satisfy $\phi(0)=0$ and $\phi'(0)=1/A_*$. These equalities align the zero-point and local scale, but do not remove calibration ambiguity elsewhere.

The first experiment fixes $\lambda\in\{0,0.5,1,2\}$ per calibration candidate. Do not fit an exponent per frame, point, coordinate, or capture. Choosing among exponents is hyperparameter selection even though each fixed-exponent model has the same coefficient count.

A literal $(A/A_*)^p$ is a different family. At $A=0$, fractional powers can have singular first or second derivatives, and $p>1$ has zero first derivative. The shifted family avoids these issues over the declared nonnegative domain. In particular, the shifted-quadratic candidate is not a pure $A^2$ model. Literal powers and independently adjustable polynomial terms are deferred comparisons, not silent aliases.

## 4. Explicit 27-coefficient response

Let $t=\theta_x/(10^{\circ})$ and $\phi=\phi_\lambda(A)$. Use

$$
b_d=[1,a,t,t\phi,t^2,t^2\phi,t^3]^\top,
\qquad
b_s=[1,t,\phi,t\phi]^\top.
$$

Set $D_x=\beta_d^\top b_d$. Give each of $D_y,T_{11},T_{12},T_{21},T_{22}$ its own four coefficients multiplying $b_s$. This yields $7+5\times4=27$ global coefficients and two framewise states. Keep pair-major output order `[v1x,v1y,v2x,v2y,v3x,v3y]`.

The separate linear-$a$ term in $D_x$ is intentionally retained. Only the former logarithmic terms are replaced. Because $A_*=1\,\mathrm D$, the numerical $a$ term matches the existing numerical diopter term. At $\lambda=0$, the new evaluator must reproduce the existing conditional27 values and derivatives for identical coefficients. This does not reinterpret frozen scale-15 baseline coefficients. [R1,R3]

At fixed states and fixed exponent, all predictions remain linear in the global coefficients. The variable-projection coefficient solve can therefore be retained, with the new state derivatives. Exponent changes do require newly fitted coefficients; changing artifact metadata alone is invalid.

An optional later 37-coefficient version replaces $b_s$ by `[1,t,phi,t*phi,t^2,t^2*phi]`. It is not part of the initial four-candidate comparison. Arbitrary per-frame affine or distortion parameters remain prohibited.

## 5. Derivatives and stable evaluation

For every candidate, including the logarithmic limit,

$$
\phi_A=\frac{(1+a)^{\lambda-1}}{A_*},\qquad
\phi_{AA}=\frac{(\lambda-1)(1+a)^{\lambda-2}}{A_*^2}.
$$

At fixed measured P1 context, a term $t^k\phi(A)$ has derivatives

$$
\partial_{\theta_x}(t^k\phi)=\frac{k t^{k-1}\phi}{10^{\circ}},\quad
\partial_A(t^k\phi)=t^k\phi_A,
$$

$$
\partial_{\theta_x\theta_x}(t^k\phi)=\frac{k(k-1)t^{k-2}\phi}{(10^{\circ})^2},\quad
\partial_{\theta_x A}(t^k\phi)=\frac{k t^{k-1}\phi_A}{10^{\circ}},\quad
\partial_{AA}(t^k\phi)=t^k\phi_{AA}.
$$

Terms with zero polynomial derivatives must be implemented explicitly, not as `0*t**(-1)` at $t=0$. The direct $a$ term has derivative $1/A_*$ and zero second derivatives. Apply the usual optimizer-state scale conversion only after constructing physical-degree/diopter derivatives; do not apply it twice.

For nonzero exponent, evaluate `expm1(lambda*log1p(a))/lambda`, or use a verified `boxcox1p` implementation. At exactly zero, use `log1p(a)`. The expansion

$$
\phi_\lambda=L+\tfrac12\lambda L^2+\tfrac16\lambda^2 L^3+O(\lambda^3),\qquad L=\log(1+a),
$$

provides a limit check. First-iteration candidates do not require fitting or differentiating the exponent. Invalid states must be rejected, not silently clipped inside basis evaluation. Physical optimization bounds handle feasibility.

## 6. Numerical inverse opportunities and limits

Keep physical inversion bounds $\theta_x\in[-20,20]^{\circ}$ and $A\in[0,6]\,\mathrm D$, separate from the calibrated horizontal targets. Freeze coefficients and covariance before each frame/subset inverse. [R3]

### 6.1 Exact conditional solve for the linear candidate

For $\lambda=1$, at fixed gaze and P1 context, any retained-coordinate prediction is

$$
F_I(\theta_x,A)=c_I(\theta_x)+a\,b_I(\theta_x).
$$

With fixed retained covariance $R_I$, the minimizing dimensionless accommodation is

$$
\widehat a(\theta_x)=\operatorname{clip}_{[0,6\,\mathrm D/A_*]}
\frac{b_I^\top R_I^{-1}(v_I-c_I)}{b_I^\top R_I^{-1}b_I},\qquad
\widehat A=A_*\widehat a,
$$

provided the denominator is sufficiently positive. If it vanishes, accommodation is locally unobservable there; return that condition rather than an arbitrary unique state.

This profiles accommodation exactly conditional on gaze. The remaining bounded gaze objective can still have multiple minima or boundary-transition kinks. Use it as an independent check of the common multistart solver, not as proof of a globally unique inverse or an excuse to hide alternative branches.

### 6.2 Gaze-polynomial checks survive the basis change

For fixed accommodation, each coordinate remains cubic or lower in $t$. With fixed covariance, the squared residual objective has degree at most six in $t$. Existing stationary-root/profile ideas can therefore be generalized to the new basis. Every helper must use the selected response, not hard-coded logarithmic coefficients. Finite accommodation grids remain incomplete global-search evidence.

## 7. Identifiability and calibration-scale ambiguity

Adding a power response does not eliminate the need for calibration anchors. For a fixed exponent and positive gains $g,h$, let

$$
\theta_x'=g\theta_x,\qquad a'=h(1+a)-1.
$$

Writing $\phi$ as a function of its dimensionless argument in this identity, for $\lambda\ne0$,

$$
\phi_\lambda(a')=h^\lambda\phi_\lambda(a)+\frac{h^\lambda-1}{\lambda};
$$

for $\lambda=0$, the additive constant is $\log h$. Also $a=a'/h+(1/h-1)$. The response bases are closed under these substitutions, so appropriate global coefficient transformations preserve optical predictions wherever both states are admissible. Corresponding raw disagreements transform as $G_\theta'=gG_\theta$ and $G_A'=hG_A$.

Fixed anchors, priors and bounds break or restrict this optical freedom. An already calibrated model cannot be rescaled while claiming unchanged calibration. A lower accommodation-disagreement score alone does not demonstrate a better optical law. No posthoc rescaling of evaluation states is allowed.

Across exponents, monotonic reparameterization of a latent accommodation coordinate can also mimic some response changes. The direct linear-$a$ term in $D_x$ limits simple cross-exponent equivalence in the complete model, but does not guarantee precise exponent identification. Report exponent stability across grouped refits and training-only state-gain diagnostics; do not interpret a selected exponent as a measured lens property.

One shared $\phi_\lambda$ also remains restrictive: at fixed gaze every non-$D_x$ response is affine in the same function. Different components cannot have freely different accommodation curvature. Failure of this comparison can motivate a later low-order multi-function basis, but not automatically more coefficients everywhere.

## 8. Calibration, priors and uncertainty

For fixed exponent, retain joint calibration of coefficients $\beta$ and training states $x_n$, with equal-fixation optical weighting and soft fixation-mean anchors. A representative objective is

$$
J_\lambda=\frac12\sum_{g=1}^{K}\frac1{K n_g}\sum_{n\in g}
 e_n^\top R_n^{-1}e_n
+\frac12\|P_\lambda(\beta-\beta_{0,\lambda})\|^2
+\frac1{2K}\sum_g\|S^{-1}(\bar x_g-x_g^{nom})\|^2.
$$

The current nominal coefficient initializer and column-normalized penalty are data-derived. Recompute them from training-only inputs using the same declared policy for every exponent. Equal coefficient counts and equal numerical prior strengths are not identical regularization in function space. Keep this limitation visible and test a small, training-only prior sensitivity if a gain depends on it. [R4]

The first comparison shares one fold-local logarithmic pilot and reference covariance across all exponents within that training split. This isolates response shape from a concurrent weighting change. A pilot from an outer training set cannot be reused inside an inner fold if it saw that fold's validation data. Propagate shared P1 errors through the complete residual. Local covariance excludes global coefficient uncertainty and response discrepancy unless separately estimated.

Do not transform the nominal labels into $\phi(A)$ and then treat them as exact truth. Do not add a Box-Cox likelihood-Jacobian term: the response basis is changing, not the random measured variable. Changing covariance, detector selection, or common-y discrepancy treatment is a separate experiment.

## 9. Cross-check is the scientific selection criterion

For each evaluation frame $n$ and each excluded P4 point $j$, estimate $\widehat x_{n,-j,\lambda}$ using all three P1 points and only the other two P4 points. Freeze the subset state and branches before accessing the excluded point. Define

$$
e_{n,j,\lambda}=q_{n,j}-\widehat q_{n,j}(\widehat x_{n,-j,\lambda};\beta_\lambda,P1_n),\qquad
E_{n,\lambda}^2=\tfrac13\sum_{j=1}^{3}\|e_{n,j,\lambda}\|^2.
$$

For the three subset states, define each physical-state disagreement separately:

$$
G_{\theta,n}^2=\tfrac13\sum_{j<k}(\widehat\theta_{n,-j}-\widehat\theta_{n,-k})^2,\qquad
G_{A,n}^2=\tfrac13\sum_{j<k}(\widehat A_{n,-j}-\widehat A_{n,-k})^2.
$$

The primary comparison loss is an equal-exposure average of excluded-point squared errors on a declared matched complete-frame cohort:

$$
L_{cross,\lambda}=\frac1{|\mathcal G|}\sum_{g\in\mathcal G}\frac1{|\mathcal C_g|}
\sum_{n\in\mathcal C_g}E_{n,\lambda}^2.
$$

The displayed $\sqrt{L_{cross}}$ is a pixel summary of cross-check error, not a competing metric. Maintain $G_\theta,G_A$, signed axis/point errors, worst-point and tail outcomes, numerical failures, ambiguity, bounds and calibration scope. Do not combine degrees, diopters and pixels into an undeclared score.

Missing frames, missing points, failed calibrations and lost exposures remain in the scheduled manifest. No empty exposure is silently omitted or given zero loss. Inner comparison requires declared common coverage and per-exposure support. A held-out measurement must not determine branch selection, a weight, a threshold, initialization, or rejection used for its own cross-check. [R3,R5]

Select the exponent on inner grouped cross-checks with predeclared guards, then test the selected procedure on sealed outer groups. Historical development observations remain development evidence even when reanalyzed with nesting. [E2]

## 10. Decision boundary and deferred hypotheses

A power candidate is promising only if it improves excluded-coordinate prediction on appropriate paired groups without unacceptable state disagreement, tails, scope loss or numerical failures. A lower training cost, steeper response derivative, different accommodation scale, or closer demand fit is insufficient.

No power candidate has yet demonstrated improvement on the real captures. Retain the log27 x/y reference. Defer exponent-by-component fitting, extra gaze curvature, shared-y covariance changes, new P1 normalizers, detector changes, and vertical-gaze estimation. An exponent experiment must not be presented as a solution to the common-y discrepancy without evidence.

## Sources and provenance

Repository references describe the inspected snapshot above; implementation may advance independently.

- **R1:** [Current response bases](full_position/model.py).
- **R2:** [Current execution status and protocol](CURRENT_STATUS.md).
- **R3:** [Existing estimator plan](ESTIMATOR_PLAN.md), especially geometry, calibration and P4 exclusion.
- **R4:** [Current coefficient profiling and prior convention](full_position/calibrate.py).
- **R5:** [Latest results audit](LATEST_RESULTS_AUDIT.md), especially scheduled-population and shared-exposure contracts; fixes are ongoing per the user.
- **E1:** [SciPy, `boxcox1p`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.boxcox1p.html): transformation definition. Remaining power-basis identities here are mathematical derivations.
- **E2:** [scikit-learn, nested versus non-nested cross-validation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html): separating hyperparameter selection and evaluation.

This document proposes a research extension. It neither updates live code nor reports a newly fitted model.
