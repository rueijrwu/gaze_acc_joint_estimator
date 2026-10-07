# Optical Measurement and Identifiability Framework

This is a symbolic measurement framework, not a guarantee of physiological
accuracy. [Estimation.md](Estimation.md) distinguishes the implemented legacy
14-coefficient model from the constrained reduced-demand experiment.
Numerical convergence,
statistical identification, and physiological validation are distinct requirements.
The relative-measurement extension below is a mathematical proposal; the saved
fits still use two normalized channels. Its optional normalized curvature and frozen-mean extensions are implemented in
[relative_calibration](../exp2/relative_calibration/README.md); relative pixels remain
quality diagnostics after the scale gate. The staged design is specified
in [EstimationImplementationPlan.md](EstimationImplementationPlan.md).

## Image positions and observables

Let $P_{ij,x}(\theta,A;\eta)$ be the horizontal image position of Purkinje
reflection $i\in\{1,4\}$ from illuminator $j\in\{a,b\}$. Gaze $\theta$ is in
degrees, accommodation $A$ in diopters, and $\eta$ represents geometry and other
nuisance variables. Both state dependencies must be assessed; corneal stability
does not establish that the measured P1 separation is independent of gaze or all
accommodation-related imaging effects. A finite gaze polynomial is an empirical
local approximation whose adequacy must be tested, not a universally accurate
optical law.

Define signed separations $s_i=P_{ib,x}-P_{ia,x}$ and displacement

$$
e=\frac{P_{1a,x}+P_{1b,x}-P_{4a,x}-P_{4b,x}}2.
$$

The implementation uses $\delta_1=|s_1|$, $\delta_4=|s_4|$ and

$$
e_n=e/\delta_1,\qquad d=-e_n,
\qquad\rho_4=\delta_4/\delta_1,\qquad
\mathbf y=[d,\rho_4]^\top.
$$

The denominator must be sufficiently separated from zero. Source ordering/signs
are fixed within a modeled branch; an absolute separation is not globally a
smooth polynomial across a sign reversal. Detection validity is a prerequisite
for an observation, not proof of reliable reflection localization.

## Crossed pairs and independent relative measurements

For this derivation, L/R denotes increasing horizontal image position within
each reflection pair. The detector uses that ordering; it does not establish
persistent illuminator identity through a crossing. On a branch with
$S_1=P1_R-P1_L>0$ and $S_4=P4_R-P4_L>0$, define the crossed displacements

$$
e_1=P4_L-P1_R,\qquad e_2=P4_R-P1_L,\qquad
m=\operatorname{mean}(P4)-\operatorname{mean}(P1)=-e.
$$

Here $m,e_1,e_2,S_1,S_4$ are in pixels; $e$ retains the opposite sign defined
above. Exactly, without assuming linear optical responses,

$$
e_1=m-\frac{S_1+S_4}{2},\qquad
e_2=m+\frac{S_1+S_4}{2},\qquad
S_1=e_2-e_1-S_4.
$$

Four horizontal positions supply at most three independent scalar differences
after removing a common additive image translation. A convenient basis is
$\mathbf z=[m,S_1,S_4]^\top$; $[e_1,e_2,S_4]^\top$ is an equivalent basis:

$$
\begin{bmatrix}e_1\\e_2\\S_4\end{bmatrix}
=C\mathbf z,\qquad
C=\begin{bmatrix}1&-1/2&-1/2\\1&1/2&1/2\\0&0&1\end{bmatrix}.
$$

$C$ is invertible. All components are relative positions; an absolute eye-image
position is unnecessary. Appending every possible pairwise difference does not
increase rank beyond three.

The implemented normalization retains only

$$
d=m/S_1,\qquad \rho_4=S_4/S_1.
$$

For normalized crossed displacements $u_k=e_k/S_1$,

$$
u_1=d-\frac{\rho_4+1}{2},\qquad
u_2=d+\frac{\rho_4+1}{2},\qquad
d=\frac{u_1+u_2}{2},\quad \rho_4=u_2-u_1-1.
$$

Thus $[u_1,u_2]$ and $[d,\rho_4]$ contain the same two normalized measurements.
The crossed-pair difference involves $\rho_4+1$; a same-side pairing would
instead involve $\rho_4-1$. Averaging alone discards differential response, but
the current ratio retains it. Nonlinear and unequal gaze/accommodation responses
do not invalidate these samplewise identities. The missing third relative
dimension is the separation magnitude $S_1$, not absolute image position.

## What normalization removes

Under the restricted image transformation $P^{\mathrm{obs}}_{ij,x}=mP_{ij,x}+q$
with the **same** positive scale $m$ and translation $q$ for all four points,
translation cancels in $e,s_1,s_4$, and $m$ cancels in $d,\rho_4$. Differential
magnification, perspective, optical changes, and localization errors need not
cancel. Variation of $\delta_1$ is not an exclusive measurement of head depth;
gaze dependence and detector variation can remain after normalization.

Using $\mathbf z$ preserves that magnitude and cancels a common additive
translation, while $[d,\rho_4]$ also removes a common positive scale. Physical eye
translation need not be a common additive image shift: depth, perspective and
differential optical changes may alter relative distances. Keeping $S_1$ is
therefore potentially informative, but is not automatically an additional
constraint on gaze or accommodation alone.

At fixed calibrated coefficients, write a candidate relative model as
$\mathbf z=\lambda\mathbf G(\theta,A)+\epsilon$, with $\lambda>0$ a possible
common scale. If scale is independently known and the state Jacobian has rank
two, three relative observations leave one local residual direction beyond the
two states. If scale is freely estimated per frame, three observations instead
serve three unknowns $(\theta,A,\lambda)$; there is no automatic residual
redundancy. Jointly fitting scale and the scale of $\mathbf G$ also introduces a
gauge that requires a declared reference convention. Fixing that global gauge
does not identify a freely varying framewise scale.

More generally, let $\widetilde J_x=LJ_x$ and
$\widetilde J_\eta=LJ_\eta$ be the whitened state and nuisance Jacobians. Inspect
the state information after allowing nuisance variation,

$$
I_{x\mid\eta}=\widetilde J_x^\top
\left(I-\widetilde J_\eta\widetilde J_\eta^+\right)\widetilde J_x.
$$

The superscript $+$ denotes the pseudoinverse. Assess rank and conditioning in
declared physical state units. A constrained scale model or temporal prior can
add information through assumptions; it must not be reported as a newly
independent optical measurement. Constant scale over a recording and P1
separation invariant to gaze/accommodation are hypotheses requiring evidence.

A common translation of both P4 points relative to P1 changes centroid
displacement $d$ but leaves P4 pair separation unchanged. Differential P4 motion
or changing P1 separation changes $\rho_4$. The two channels share a denominator
and point coordinates, so their noise can be correlated. A weak P4 gaze model
or unstable localization can create gaze–accommodation cross-talk; correlation
alone proves neither true accommodation variation nor its absence.

## Rational optical model and current reduction

On a fixed separation-sign branch, an empirical raw model can be written

$$
\mathbf X=[e,\delta_1,\delta_4]^\top
\approx(G+\Gamma a)\boldsymbol\Theta(\theta),\qquad
\boldsymbol\Theta=[\theta^3,\theta^2,\theta,1]^\top,\quad a=A^p.
$$

A common image scale may multiply the entire raw vector. Writing the three rows
as $g_e,g_1,g_4$ and $\gamma_e,\gamma_1,\gamma_4$, normalization gives

$$
d=-\frac{(g_e+\gamma_e a)\boldsymbol\Theta}
         {(g_1+\gamma_1 a)\boldsymbol\Theta},\qquad
\rho_4=\frac{(g_4+\gamma_4 a)\boldsymbol\Theta}
             {(g_1+\gamma_1 a)\boldsymbol\Theta}.
$$

The denominator can remain gaze dependent. Setting $\gamma_1=0$ is an optional
corneal-invariance approximation requiring evidence. Fixing the denominator to
one is a further empirical reduction, not an exact consequence of normalization.
The normalized model is generally nonlinear in coefficients if its denominator
is unknown. Exact linear coefficient profiling applies only when the denominator
and bases are fixed, as in the current reduced implementation.

The power coordinate has an empirically selected, fixed exponent $p>0$; neither
a universal exponent nor equivalence to a logarithmic coordinate is established.
For $0<p<1$, $\partial_A A^p=pA^{p-1}$ is singular at $A=0$. Optimizing
$a=A^p$ can avoid that derivative in the forward model, but physical penalties
require $\partial_a A=p^{-1}a^{1/p-1}$. Endpoints and bounds need explicit handling.

The preserved legacy 14-coefficient model is

$$
d=b(A)+s(A)\theta,\qquad
\rho_4=c_0(A)+c_1(A)t+c_2(A)t^2,\quad t=\theta/15,
\qquad c_k(A)=u_k+v_kA^p.
$$

Intercept $b$ and gain $s$ interpolate linearly on fixed accommodation knots,
with declared endpoint extrapolation. Knot derivatives are one-sided. For fixed
states/exponent/knots the model is $\mathbf F=H(\theta,A)\beta$, linear in shared
coefficients. Coefficients are shared globally; both framewise states remain free.
A smooth accommodation basis is only a deferred candidate if withheld evidence
requires it. Apparent gaze-linked accommodation is not itself a reason to use it.

The reduced-demand experiment keeps this 14-slot representation while
constraining $\beta=E\gamma$. For three trained demand levels, $E$ expands six
free displacement coefficients into eight by interpolating the omitted internal
$b/s$ knot from its trained neighbors; the six ratio coefficients remain free.
There are 12 free coefficients. The matched four-demand control has $E=I_{14}$.
The model remains linear in $\gamma$ at fixed states, with design $HE$; the same
map must enter the coefficient prior and profiled derivatives. This constraint
acts during fitting. A posthoc coefficient patch can change inverse branches
and is not a trained replacement model.

At nominal training means, an omitted demand supplies no direct observation of
its independent knot coefficients. Optimized latent accommodation states can
move into neighboring intervals and give that knot indirect support, so nominal
design rank deficiency does not imply rank deficiency of the trained-state
design. Removing the unsupported degree of freedom addresses this coupling;
it does not prove identifiability or sufficient calibration coverage.

## Nonlinear gaze response: information versus model capacity

For a fixed accommodation, separate pair polynomials have a mean polynomial
with coefficients $(\alpha_i+\beta_i)/2$ and a half-difference polynomial with
coefficients $(\beta_i-\alpha_i)/2$. Both can be nonlinear in gaze. If these
polynomials describe pixel displacements, dividing by gaze-dependent $S_1$
generally produces rational rather than polynomial normalized responses.

The implemented model restricts normalized mean displacement to be linear in
gaze while allowing quadratic gaze dependence in the ratio. Its equivalent
normalized crossed-pair models consequently have equal-and-opposite quadratic
gaze terms. It cannot represent common quadratic curvature in those channels.
This is a restriction of the forward model, not evidence that separate pair
measurements add independent normalized information.

The isolated optional candidate is

$$
d=b(A)+s(A)\theta+q\,t^2,\qquad t=\theta/15,
$$

with one globally shared, regularized curvature coefficient $q$ and the current
ratio model. Only training evidence could justify extending it to
$q(A)=q_0+q_1A^p$. Quadratic curvature is a candidate local approximation, not a
universal description of gaze nonlinearity; any alternative basis, such as odd
cubic curvature suggested by symmetry, must be declared and assessed rather
than added indiscriminately. Independent unrestricted polynomials for every
pair are not required to represent distinct common and differential responses.

## Noise, local inversion, and model cross-talk

Let $\Sigma\succ0$ be measurement covariance, $W=\Sigma^{-1}$ precision, and
$L^\top L=W$ a whitening factor. Switching from $[e_n,\rho_4]$ to $[d,\rho_4]$
requires $S\Sigma S^\top$ and $SWS^\top$, where $S=\operatorname{diag}(-1,1)$.
Covariance estimation, correlated whitening, and robust block losses require
training-only assumptions; they do not remove systematic model bias.

Relative measurements sharing image points have correlated localization noise.
For a fixed linear difference operator $D$ acting on the four horizontal
coordinates, $\Sigma_z=D\Sigma_P D^\top$. This defines a noise model for relative
observations without using absolute position as a state measurement. Under the
basis change $C$, use $\Sigma'=C\Sigma_z C^\top$ and
$W'=C^{-\top}W_zC^{-1}$. Correct transformation preserves the Mahalanobis
residual and a robust loss applied to that complete residual block. Treating
redundant differences as independent produces incorrect weighting and a
rank-deficient augmented covariance. A numerical ridge does not create new
measurement information.

For fixed model coefficients and an interior differentiable state $x=(\theta,A)$,

$$
J_x=[\partial_\theta\mathbf F,\partial_A\mathbf F],\qquad
\delta x\approx J_x^{-1}(\delta\mathbf y-\delta\mathbf F_{\mathrm{model}}).
$$

This first-order relation applies only where the local inverse exists. A biased
P4 gaze response can therefore transfer error into both inferred states. Nearly
collinear Jacobian columns amplify observation/model errors. Inspect
$LJ_x\operatorname{diag}(u_\theta,u_A)$ with declared degree/diopter reference
increments; unscaled condition numbers depend on units. Local rank two is
necessary for a regular two-state inverse, but does not ensure global uniqueness,
correctness outside calibrated support, or physiological accuracy.

For $d=b(A)+s(A)\theta$ and nonzero $s(A)$, holding displacement fixed gives

$$
\left.\frac{d\rho_4}{dA}\right|_d
=\partial_A\rho_4-\partial_\theta\rho_4
 \frac{b'(A)+s'(A)\theta}{s(A)},\qquad
\det J_x=s(A)\left.\frac{d\rho_4}{dA}\right|_d.
$$

At a displacement knot, evaluate each side separately. A slope sign change or
vanishing determinant can expose folds and gaze/accommodation ambiguity even
when two optical residuals can be driven nearly to zero. The new experiment
therefore checks multistart branches, one-sided derivatives, and sampled
noise-scaled physical Jacobians over the relevant gaze/A grid.

## Calibration and three kinds of validity

Nominal gaze and accommodation demand provide **soft fixation-mean anchors**,
not hard framewise labels or measured accommodation. Temporal priors can stabilize
trajectories without adding optical identification. Jointly fitting global
coefficients and local states introduces coupling and a nonconvex problem;
regularization may select a solution without proving it is true.

Previously fitted fixation-mean gaze may also provide a soft, model-derived
prior for a later fit. Reusing states as initialization does not change the
objective; penalizing deviation from those means does. If the previous fit used
the same observations or nominal anchors, its means are not independent
evidence. Freeze the prior centers, declare their strength and uncertainty scale,
and assess sensitivity including a zero-prior case. Within a held-out fold, the
prior's source model must itself exclude the withheld data. Such a prior can
stabilize a fit but cannot create new optical information or validate its own
state estimates. Estimation.md defines the proposed mean-only penalty.

Numerical validity concerns derivatives, solver accuracy, feasible-direction
stationarity, and termination. An evaluation limit is not evidence that the
problem cannot converge. Statistical validity concerns uncertainty, noise/model
assumptions, identifiability, and performance on genuinely withheld conditions.
Physiological validity requires independent gaze/accommodation reference data.
Small optical residuals or nominal-anchor agreement alone establish none of these
three. Support/extrapolation and ambiguity must be reported separately from
computational bounds; a calibration demand endpoint is not a physiological limit.

The reduced-demand experiment is retrospective: compare 0.36D/2D/4D training
with 3D withheld, and complementary 0.36D/3D/4D training with 2D withheld. The
power exponent, noise, and prior must be estimated within each training fold.
A matched full-data fit uses the same prior-construction policy to separate
pipeline differences from the preserved historical reference. Reduced-knot
constraints and a successful complementary holdout do not establish that three
levels suffice for new captures; repeat data or independent physiological
references are still required.

Pair-derived gaze agreement is an internal check only when the two estimates
are not forced to agree by selecting accommodation or other nuisance values.
One pair displacement generally cannot identify both gaze and accommodation.
Retaining a ratio or mean that includes a withheld pair leaks that pair into
the estimate. Independent accommodation information can make separate gaze
estimates comparable, but nominal demand alone does not measure accommodation.
Shared biases can survive even perfect pair agreement.

Repeated recordings test repeatability and sensitivity to conditions. With two
free states and two normalized observations per frame, even unseen observations
can admit near-zero optical residuals, so those residuals alone cannot select
the most accurate state model. The existing full-versus-reduced comparison
changes both demand coverage and coefficient freedom; it does not isolate
either cause of the differing gaze traces. A planned comparison must also fit
all four demands with the same constrained displacement map, then compare that
with the flexible four-demand model and with the three-demand constrained fit.
