# Joint gaze and accommodation estimation

This document distinguishes the preserved 14-coefficient estimator from the new
reduced-demand experiment. [Theory.md](Theory.md) explains the optical assumptions
and identifiability; [the workflow guide](../exp2/README.md) gives legacy commands.
The existing solver is [calibrate_continuation.py](../exp2/calibrate_continuation.py),
backed by [calibrate_profiled.py](../exp2/calibrate_profiled.py). The retained full-fit
checkpoint and trajectories are in [full_calibration/legacy](../exp2/full_calibration/legacy/).
The [reduced-calibration report](../exp2/reduced_calibration/README.md) describes
the implemented experiment, numerical audits and retrospective results. Both
robust holdout fits passed the physical convergence checks. Against the preserved
full-fit trajectory, demand-3 holdout gives 0.3812° gaze and 0.07857D accommodation
RMSE; demand-2 holdout gives 2.1338° and 0.20081D. These are model-state deviations,
not physiological accuracy. The converged fresh matched full control gives
similar results: demand-3 holdout 0.3827°/0.07318D and demand-2 holdout
2.1564°/0.19146D. All comparisons, plots and convergence records are in the report.
The final section specifies proposed relative-measurement and nonlinear-gaze
extensions. They are not present in the preserved full/reduced saved fits. Optional curvature
and frozen previous-mean priors now have an isolated implementation and verification
report in [relative_calibration](../exp2/relative_calibration/README.md). H0, the frozen
holdout3 model, is the primary comparison reference for the new experiment;
matched full is secondary. New H0_all20 predictions apply the same anchor-free
inverse to all20 without refitting, keeping H0 joint training states separate.
These are model-state comparisons, not physiological ground truth. All six new
cases converged; on the shared five 3 D fixations, mean-gaze RMSE from H0 is
.18423° for holdout curvature, .17242° with the weak previous-mean prior, and
.04662°/.20220°/.33908° for full constrained/no-curvature,
full constrained/curvature and full flexible/curvature. Existing flexible full
is .28871°. This shared subset is held out only for holdout3 models; full-model
comparisons are in sample. The [side-by-side report](../exp2/relative_calibration/README.md)
retains all cases, inference flags and the flexible-full numerical continuation. The scale gate
retains the normalized estimator and relative pixels as quality diagnostics. See the staged
[implementation plan](EstimationImplementationPlan.md) before changing code.

## Measurements and calibration support

For each original frame, the horizontal coordinates of the two P1 and two P4
reflections define

$$
\delta_1=|P_{1b,x}-P_{1a,x}|,\qquad
\delta_4=|P_{4b,x}-P_{4a,x}|,
$$

$$
d=\frac{\operatorname{mean}(P_{4,x})-\operatorname{mean}(P_{1,x})}{\delta_1},
\qquad \rho_4=\frac{\delta_4}{\delta_1},
\qquad \mathbf y=[d,\rho_4]^\top.
$$

Both observations are dimensionless. The measurement helper returns the older
sign convention $e_n=-d$; the calibration loader converts it to $d$. When precision
or covariance is expressed in $[e_n,\rho_4]$, transform it with
$S=\operatorname{diag}(-1,1)$: $W_d=SW_eS^\top$, $\Sigma_d=S\Sigma_eS^\top$.

Calibration uses captures 1–4 and five ordered gaze targets per capture:
$[-15,-7.5,0,7.5,15]$ degrees. Nominal accommodation demands are
$[1000/2775,4,3,2]$ diopters. Capture 5 is outside this calibration workflow.

The selected intervals are frozen in
[fixation_intervals.json](../exp2/fixations/fixation_intervals.json).
Trim $\lfloor0.1(r_1-r_0)\rfloor$ original frames from each end of every
half-open interval $[r_0,r_1)$ **before** filtering validity, including at EOF.
Retain finite valid P1/P4 pairs with adequate P1 separation and a valid pupil.
The current selection retains 77,756 frames. No additional event mask is applied.
Source hashes, interval hashes, frame order, and gated counts are checked on load.

The detector requires both native pupil-valid flags before exporting P1/P4 pairs.
The frozen stored detections were gated without rerunning native detection. Pupil
validity does not guarantee accurate P4 localization. Temporal links connect only
consecutive valid original frames inside the same fixation; invalid gaps, trimmed
edges, and fixation boundaries break links.

## Shared forward model

Each frame has free state $x_i=(\theta_i,A_i)$, in degrees and diopters. The preserved
baseline has a shared 14-coefficient model predicting both optical channels:

$$
d_i=b(A_i)+s(A_i)\theta_i,
$$

$$
\rho_{4,i}=c_0(A_i)+c_1(A_i)t_i+c_2(A_i)t_i^2,
\qquad t_i=\theta_i/15,\qquad c_k(A)=u_k+v_kA^p.
$$

The four knot values of $b$ and $s$ interpolate linearly at the sorted nominal
accommodation demands, with endpoint-linear extrapolation. The six ratio
coefficients multiply $[1,t,t^2,a,at,at^2]$, where $a=A^p$. The exponent is fixed
for a specified fit. Thus $\mathbf F_i=H(\theta_i,A_i)\beta$ is linear in the
shared coefficients conditional on frame states.

Legacy full-data initialization uses the retained seed in
[calibration_seed.json](../exp2/calibration_seed.json). Without a seed, fit $p$ to
training fixation-mean ratios. A whole-fixation holdout always estimates its
initialization and noise from training data, ignoring the full-data seed.
Displacement means initialize per-capture intercept/gain; initial gaze comes from
those lines and initial accommodation from nominal demand. Both states are freed
for joint fitting.

The reduced-demand experiment preserves the 14-slot coefficient order and optical
equations, but writes $\beta=E\gamma$ with a fixed linear expansion map. Only the
displacement knot values at **training demands** are free; an omitted internal
knot's intercept and gain are exact linear interpolations of their training
neighbors. Holding out 3D leaves knots $[1000/2775,2,4]$ and 12 free coefficients;
holding out 2D leaves $[1000/2775,3,4]$ and 12 free coefficients. A fresh matched
full-data control uses $E=I_{14}$. The six ratio coefficients remain free, with
the same $A^p$ power coordinate. Thus each coefficient profile solves for
$\gamma$ using $H_iE$, and both prior and variable-projection derivatives use the
same map. Interpolation is enforced throughout fitting, not patched afterward.

Every new fold, including the matched full-data control, estimates $p$ from only
its training fixation-mean ratios over $[0.15,1]$. Mean-line initialization,
noise covariance, and initial-function prior are training-only. No historical
seed, full-fit coefficient, or reference trajectory initializes these fits.

The optimizer coordinates are $q_i=(\theta_i/15,A_i^p/4^p)$. Computational bounds
are $\theta\in[-20,20]$ degrees and $A\in[0,6]$ D. These bounds do not establish
physiological limits or accuracy outside the calibrated knot support.

## Objective and noise weighting

For $J$ training fixations with $n_j$ retained frames, each optical frame has
weight $\alpha_i=1/(Jn_{j(i)})$. This gives each fixation equal total optical weight.
Estimate a frozen, correlated $2\times2$ covariance $\Sigma$ from consecutive
training-frame differences divided by $\sqrt2$, with robust radial winsorization
and an SPD ridge. Real eye motion may contaminate that empirical noise estimate.
Use $W=\Sigma^{-1}$ and a whitening factor satisfying $L^\top L=W$.

Minimize

$$
\mathcal J(x,\beta)=
\frac12\sum_i\alpha_i\varrho_\kappa
  (\|L(H_i\beta-\mathbf y_i)\|^2)
+\mathcal R_{\mathrm{means}}(x)
+\mathcal R_{\mathrm{temporal}}(x)
+\frac12\|M_0\beta-v_0\|^2.
$$

Quadratic fitting uses $\varrho(s)=s$. Robust refinement uses block soft-L1,
$\varrho_\kappa(s)=2\kappa^2(\sqrt{1+s/\kappa^2}-1)$, normally with $\kappa=2$.
The complete correlated two-channel residual is robustified as one block.

Fixation means are ordinary unweighted means of the retained frame states.
The mean penalty is

$$
\mathcal R_{\mathrm{means}}=
\frac1{2J}\sum_j\left[
 (\bar\theta_j-T_j)^2+
 ((\bar A_j-D_j)/0.25)^2\right].
$$

Targets and demands are soft mean anchors, not framewise truth. Robust optical
weights do not redefine those means. For $\ell_j$ consecutive-frame links in a
fixation, the temporal penalty is

$$
\mathcal R_{\mathrm{temporal}}=
\frac{0.1}{2J}\sum_{j:\ell_j>0}\frac1{\ell_j}
\sum_{(i,k)\in\mathcal L_j}\left[
 (\theta_k-\theta_i)^2+((A_k-A_i)/0.25)^2\right].
$$

The coefficient prior preserves the training-only initial forward function on a
$9\times9$ gaze/accommodation grid, with strength 0.1 and frozen prior precision.
The prior, precision, exponent, support, and bounds stay fixed during a
same-objective continuation.

The current gaze and accommodation anchor strengths are both 1, with scales
1 degree and 0.25 D respectively. These are chosen regularization scales, not
uncertainties measured against independent state truth. Every fixation receives
the same anchor strength. Mean anchors remain quadratic even when the optical
loss is robust. Smoothness has strength 0.1 with the same state scales; it acts
only across the declared valid consecutive-frame links.

For all new reduced-demand folds and the matched full-data control, use the same
rule $W_0=\operatorname{diag}(1/\operatorname{ptp}(\mathbf y_{\mathrm{train}})^2)$,
where each range is computed over retained **training-frame observations**, not
fixation means. Strength remains 0.1. This reproduces the existing demand-3
holdout prior policy and ignores the historical full-data seed. The preserved
full reference used a different, stronger seed prior; compare it separately
from the fresh matched full control to expose that pipeline confound.

## Joint optimization and checkpoints

At fixed states and optical weights, QR solves the shared coefficient problem
exactly. A matrix-free variable-projection Jacobian includes the coefficient
response, with exact forward and transpose products. One bounded trust-region
solve updates the complete quadratic trajectory continuously. Local-curvature
coordinate scaling preserves the objective and bounds.

For robust fitting, coefficient IRLS must converge; fixed-weight trajectory
subproblems update the states. Accept and checkpoint only steps that preserve or
lower the true robust objective. Explicit warm-start mode permits a change from
quadratic to robust loss while requiring the same model, data, precision, and priors.

The atomic NPZ checkpoint embeds objective identity, states, coefficients, weights,
and accepted history. Resume checks source/interval and frame/group/observation
hashes, coefficient schema, fixed exponent, priors, precision, bounds, and loss.
It restores the accepted fit state and starts a fresh trust-region solve;
SciPy's internal trust radius is not saved. Evaluation and wall limits preserve
the last accepted checkpoint. Progress and actual LSMR stopping diagnostics are
recorded throughout the run.

Geometry is cached between coefficient IRLS updates at the same state. Prediction
and objective evaluation reuse the forward matrix and avoid derivative allocations.
These performance changes preserve the mathematical objective and derivative rules.

The new checkpoint identity must additionally record and validate $E$, its rank,
training demands, missing-knot constraints, and free coefficient count. Quadratic
fitting followed by robust refinement retains the existing priors, weights,
anchors, bounds, and physical stopping criteria. Larger budgets and same-objective
continuations are permitted; tolerances are not relaxed to manufacture convergence.

## Stopping, interpretation, and validation

Convergence requires all of the following:

- Bound-aware physical stationarity in reference increments of 1 degree and
  0.25 D, with default tolerance $10^{-5}$.
- Maximum step in those reference units below $10^{-5}$ and relative objective
  change below $10^{-10}$.
- For robust fitting, weight changes below $10^{-7}$ relative to the frozen
  subproblem weights.
- Valid feasible-direction checks at accommodation knots and the zero endpoint.

At an exact interior knot, one-sided profiled derivatives must satisfy
$g_A^-\leq0$ and $g_A^+\geq0$. Nearby ordinary points use their actual branch.
Indeterminate physical stationarity at $A=0$ prevents certification. Small gradients,
small residuals, or a solver stopping message alone do not establish convergence.

Report optical residuals separately from estimated observation noise and state
accuracy. With two free states per frame, very small fitted optical residuals can
coexist with uncertain gaze/accommodation. Inspect inverse conditioning,
extrapolation, nominal-mean assumptions, and P4 gaze dependence. Whole-fixation
inversion uses no target/demand anchors; independent reference measurements are
required to establish physiological accuracy. The legacy reference is in
[full_calibration/legacy](../exp2/full_calibration/legacy/); current holdout
comparisons are documented in
[the experiment report](../exp2/reduced_calibration/README.md).

The new experiment evaluates both demand-3 and complementary demand-2 holdouts,
plus their training-mean initializers and a fresh matched full-data control.
Heldout inference freezes coefficients before reading reference states, uses
no nominal gaze/accommodation or baseline-state anchors, and joins original
frame/fixation keys exactly. Report optical residuals, deviations from stored
full-fit states, ambiguity, physical stationarity, extrapolation, and conditioning
separately. Audit sampled inversions and physical Jacobians across the gaze/A
grid; local rank does not establish global inverse uniqueness. All available
comparisons are retrospective and cannot replace independent repeat/reference
validation. Preserve the legacy outputs alongside the new experiment.

## Proposed relative-measurement estimator

Status: the three-channel state estimator remains a conditional specification:
the scale gate does not justify it. The independent relative observation helper
and quality reports, optional normalized curvature and frozen previous-mean
penalty are implemented in [relative_calibration](../exp2/relative_calibration/README.md).
The preserved normalized solver and saved full/reduced results remain the baseline.
Absolute image position is not an estimator input. Form the three-dimensional relative observation

$$
\mathbf z_i=[m_i,S_{1,i},S_{4,i}]^\top,\qquad
m_i=\operatorname{mean}(P4_x)-\operatorname{mean}(P1_x),
$$

with positive, ordered horizontal separations $S_1,S_4$. All entries are in
pixels. Retain crossed displacements $e_1=P4_L-P1_R$ and $e_2=P4_R-P1_L$ as
derived diagnostics or an equivalent three-dimensional observation basis, not
as extra independent residuals. Verify $e_2-e_1=S_1+S_4$ and reproduce the
existing $[d,\rho_4]=[m/S_1,S_4/S_1]$ on exactly the same frames. Invalid pairing,
ordering ambiguity, and insufficient separation remain explicit quality flags.
L/R ordering is not a claim about persistent illuminator identity.

The third relative dimension is separation magnitude. A shared image
translation cancels; common magnification does not. Before treating this
dimension as a third state constraint, declare the scale model. A candidate
$\widehat{\mathbf z}_i=\lambda_i\mathbf G(\theta_i,A_i)$ has three unknowns
per frame when $\lambda_i$ is free. It therefore does not automatically supply
redundant information beyond gaze and accommodation. Assess the state Jacobian
after accounting for scale/nuisance directions, as defined in Theory.md.
Only an independently supported scale constraint can supply the missing
information; a session-constant or temporally regularized scale is an additional
assumption that must be tested and reported. Do not declare P1 separation
independent of gaze/accommodation without evidence.

Use one shared state and a coupled forward model for the relative channels.
Start from the smallest declared gaze/accommodation basis supported by training
data. A factorized relative model, such as
$h(\theta,A)[d(\theta,A),1,\rho_4(\theta,A)]^\top$, introduces products of
unknown coefficients when $h$ is fitted; the current exact linear coefficient
profile cannot be reused without a new derivation. A direct relative basis
linear in shared coefficients may allow profiling conditional on states and
scale, but rank, scale gauge and derivatives still require separate checks.

Estimate the full $3\times3$ covariance of the independent relative basis from
training data, retaining shared-point correlations. Point-localization
uncertainties can inform it when available; current validity flags are not
calibrated uncertainty estimates. Temporal difference estimates can include
real state or scale motion. Transform covariance consistently when changing
relative bases. Do not append redundant normalized and unnormalized features
and assign independent weights. Continue equal-fixation weighting and an
explicitly declared robust block loss; optical objective values across
different measurement dimensions are not directly comparable accuracy scores.

A separate minimal candidate addresses nonlinear gaze while keeping the current
normalized inputs:

$$
d=b(A)+s(A)\theta+q(\theta/15)^2.
$$

Use one shared regularized $q$ initially, with $q=0$ reproducing the baseline.
The ratio already has quadratic gaze dependence and $A^p$ interactions. A
training-supported accommodation dependence of $q$ is a later extension. Do
not change the measurement basis and model flexibility in one unexplained
comparison. New coefficient layouts, nuisance states, priors, covariance and
observation definitions require a new schema and checkpoint identity.

## Proposed reuse of fitted gaze as initialization and soft prior

Existing fitted gaze can stabilize a later calibration. Distinguish three uses:

- Initialization reuses states or compatible coefficients without adding a
  penalty. It can affect which solution a nonconvex fit reaches, but does not
  change the declared objective.
- An optional fixation-mean prior adds an explicit regularization term centered
  on frozen fitted means from a compatible earlier model.
- Independent validation uses separate evidence; agreement with those earlier
  fitted means is not an accuracy test.

For a training fixation $j$, let $\mu_j^{\mathrm{prev}}$ be the ordinary mean of
the previous fitted gaze on the exact retained support. The proposed penalty is

$$
\mathcal R_{\mathrm{prev}}=
\frac{\lambda_{\mathrm{prev}}}{2J}
\sum_j\left(
\frac{\bar\theta_j-\mu_j^{\mathrm{prev}}}
     {\sigma_{\mathrm{prev},j}}
\right)^2.
$$

This anchors means softly and leaves within-fixation variation free apart from
the other penalties. It is not a hard fixation label or a per-frame requirement
to reproduce the previous trajectory. Prior centers stay frozen during a fit;
repeatedly updating them from the current solution would define a different
procedure. Initial implementation must expose strength and scale explicitly,
with $\lambda_{\mathrm{prev}}=0$ recovering the no-prior objective. No positive
strength or uncertainty scale has yet been selected or validated.

The previous estimates may already use the same observations and nominal mean
anchors. Treat their reuse as model-derived regularization, not independent
evidence, and account explicitly for whether the nominal anchor is retained or
replaced. Do not silently stack both at full strength or infer a tiny prior
uncertainty from the number of correlated frames. A strong prior can preserve
previous model bias and obscure the effect of added curvature.

For a strict held-out experiment, the source model must be trained within that
same training fold. A full-data fit or a model-derived prior using the withheld
capture cannot initialize or anchor that fold's candidate. Exact fixation/frame
mapping, model hashes, source fold, mean values, prior scales and strengths must
enter provenance and objective identity. Reference means for unrelated future
recordings are not available merely because a previous calibration exists;
applying a prior requires an explicit correspondence or state-evolution model.
Held-out estimation remains anchor-free unless a separately declared experiment
tests such a prior.

## Planned comparison and interpretation

Use the same constrained displacement family for three-demand and four-demand
fits to test demand coverage, then compare constrained and flexible families
with the same four-demand data to test model capacity. Assess curvature and
relative-scale extensions separately. Use consistent training-only policies
for exponent, noise, priors and initialization; data-dependent values can differ
between folds and must be recorded. Prior-strength comparisons include zero
and declared weak alternatives rather than tuning toward reference traces.

Pair-derived gaze agreement is useful only when accommodation/nuisance values
are constrained independently of that agreement. One pair alone generally
cannot identify both states; retaining the other pair through a mean or ratio
invalidates a claimed pair holdout. Report repeated-recording consistency and
nominal-target checks separately from independent state accuracy. Three relative
channels can support a residual consistency test only to the extent that their
nuisance variables and coefficients are independently constrained. No extension
is validated by a steadier gaze trace, tiny optical residuals, or agreement with
the prior alone. The implementation plan specifies the decision gates and checks.
