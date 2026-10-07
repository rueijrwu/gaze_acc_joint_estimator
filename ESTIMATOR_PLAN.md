# Full-position shared-state estimator: optical grid-distortion algorithm plan

## Status and objective

**This is a proposal, not an implemented or evaluated estimator.** It accompanies [Theory.md](Theory.md). Preserve `joint_m2.py`, `models/quadratic_model.json`, stored detections, and existing result directories as the historical baseline. New code, coefficients, schemas, and outputs must be separate.

Treat P1 and P4 as two differently distorted images of one fixed source pattern. Estimate one horizontal gaze/accommodation state per frame that predicts both common displacement and the different coordinate responses of all three P4 samples relative to P1. Reproducible state-dependent deformation is signal. The diagnostic is unexplained position error under that calibrated response, not departure from an undistorted or similar triangle.

The required normalization is unchanged:

$$
\ell_1=\sqrt{\mathcal A_1},\qquad
 d_x=(c_{4,x}-c_{1,x})/\ell_1,\qquad
 \rho_4=\mathcal A_4/\mathcal A_1.
$$

`rho_4` is an area ratio, not its square root; area is formed by reflection centers, not blob areas. Fit full coordinates rather than compressing them to centroid displacement and area. Similarity-fit magnification is a reduced comparison only. Changing the P1 normalizer is a separate later experiment.

**Current nominal gaze targets: `[-10,-5,0,5,10]` degrees. Proposed candidate gaze scaling: `theta_deg/10`. The frozen baseline's `theta_deg/15` convention is retained only in its own adapter and does not mean this experiment calibrated to 15 degrees.**

## 1. Baseline inspection and calibration contract

The original saved-result inspection was performed at commit `9fabb3cdfe6b4651c7b5b16cb9df573fe91b63cf`. This documentation revision starts from `5017ede8ddd26097960c350bc75090b566cada4c`, adds the grid-distortion interpretation, and incorporates the explicitly confirmed five-target protocol. The following are saved results, not newly regenerated results.

| Item | Stored status |
|---|---|
| Baseline | 13-coefficient model of `[d_x,rho_4]` |
| Calibration | 20 reviewed fixations, captures 1–4, central 80% of each interval |
| Nominal gaze targets in each capture | `[-10,-5,0,5,10]` degrees |
| Retained calibration observations | 71,729 after 55 jump-filter rejections |
| Selected training stage | Quadratic fit; 17 accepted iterations; recorded as converged |
| Later robust stage | Handoff reports weight convergence was not achieved |
| Capture 5 | 25,226 retained estimates / 26,370 rows |
| Capture 6 | 34,080 retained estimates / 35,001 rows |
| Equivalent-minimum ambiguity | Saved counts zero in both captures |
| Accommodation-bound counts | Capture 5: 7 upper; capture 6: 2 lower and 8 upper |
| Gaze outside nominal anchor range | Capture 5: 8,747; capture 6: 12,637 |
| Accommodation outside nominal anchor range | Capture 5: 23; capture 6: 13 |
| Saved projected-stationarity value above `1e-3` | Capture 5: 7; capture 6: 11 |

Sources: [frozen model](models/quadratic_model.json), [capture summaries](experiments/captures_5_6_direct/summary.json), [fixation intervals](data/fixations/fixation_intervals.json), [HANDOFF.md](HANDOFF.md), and [joint_m2.py](joint_m2.py).

A retained estimate is not necessarily a numerically satisfactory inverse. The saved stationarity threshold is implementation-specific. Summaries record reused inverse caches; packaging did not independently rerun the workflow. Historical provenance paths are not necessarily locally available.

Baseline calibration uses soft fixation-mean anchors, a coefficient prior, and temporal regularization. `joint_m2.py` sets gaze-anchor scale to 0.1 degree, accommodation-anchor scale to 0.25 D, and temporal strength to 0.1 with scales 1 degree / 0.25 D. Application inversion is independent per frame. Do not assume legacy `lib/` defaults or two-pair loading rules describe exp5. Captures 5/6 have no reviewed target intervals and cannot establish accommodation accuracy from their stored estimates alone.

### 1.1 Targets, parameterization, and support

Read interval labels from the stored fixation file and verify the configured five-target protocol `[-10,-5,0,5,10]` degrees. Do not import older `[-15,-7.5,0,7.5,15]` constants, manufacture target overrides, or change the stored observations to match a legacy loader. Read accommodation demands from the same metadata; this revision changes no demand labels.

Keep these quantities distinct:

| Quantity | Proposed convention |
|---|---|
| Nominal gaze calibration grid | `[-10,-5,0,5,10]` degrees |
| Full-data nominal gaze anchor range | `[-10,10]` degrees |
| Candidate dimensionless gaze | `t=theta_deg/10` |
| Nominal values of `t` | `[-1,-0.5,0,0.5,1]` |
| Optimizer state encoding | `(theta_deg/10,A_diopters/4)` |
| Computational inversion bounds | `theta_deg in [-20,20]`, `A_diopters in [0,6]` |
| Frozen baseline basis/encoding | Keep its recorded `theta/15` conventions unchanged |

Store fold-specific anchor/support information separately from full-data support and computational bounds. A solution at 15 degrees is outside the full-data nominal gaze anchor range, not an additional calibration point. Holding out -10 or +10 narrows the remaining training-anchor range; report extrapolation relative to that fold. State estimates and nominal anchors need not coincide framewise, so retain both nominal-support and empirical-training-state diagnostics.

The gaze calibration grid is not the illuminator/source grid. Source field angles, source offsets, and camera/eye axis alignment require their own metadata. Do not infer that straight-ahead gaze is on-axis illumination.

### 1.2 Optical interpretation to preserve in implementation

The conceptual mappings are `p_j=F1(s_j;theta,A,eta)` and `q_j=F4(s_j;theta,A,eta)` for fixed source identity `s_j`. Eye rotation changes field sampling and viewing geometry; accommodation changes the P4 optical system, potentially including both magnification and field-dependent deformation. These motivate the model, not predetermined coefficient values.

Do not silently undistort P4 to a similarity template, remove orientation/shape variation, or reject a point merely because it violates constant triangle angles or edge-scale ratios. First model reproducible state dependence. Use residuals after that prediction for consistency checks. The conditional candidate does not assert that P1 shape is state-independent; it uses that measured shape as context.

## 2. Data contract and feature construction

Read each payload's correspondence metadata. The recorded zero-based map is `pair_index=[2,1,0]`. Reorder coordinates and flags together into P1 correspondence; sorting the two images independently does not establish source identity.

Retain original frame index, timestamp, capture/fixation labels, all six image points, validity flags, correspondence metadata, and detector diagnostics. Keep all P1 points as reference geometry and each P4 point as an individually maskable observation. Store physical source coordinates/directions when actually available; otherwise mark them unknown and use the three correspondences without inventing metric source positions.

Define

$$
\mathbf c_1=\tfrac13\sum_j\mathbf p_j,\quad
\mathbf r_j=(\mathbf p_j-\mathbf c_1)/\ell_1,\quad
\mathbf v_j=(\mathbf q_j-\mathbf c_1)/\ell_1,\quad
\mathbf u_j=\mathbf v_j-\mathbf r_j.
$$

Use six residual components in the order `[v1x,v1y,v2x,v2y,v3x,v3y]`, with measured `r_j` as context. This is equivalent to using all six normalized pair displacements `u`: subtracting the same measured `r` from observation and prediction leaves the residual unchanged.

Keep measured P1 centroid and scale for pixel reconstruction and auditing, not as additional state residuals in the first candidate. No per-frame P4 centering, P4 rotation normalization, or P4-area normalization is allowed in this representation. Compute an observed affine map only for separately identified full-frame diagnostics; it is not required to construct the coordinate residual and must not enter a P4 holdout solve.

**Validity:** reproduce the baseline mask separately. It requires finite P1/P4 coordinates, all `p4_found`, and a numerically nondegenerate P1 triangle; it has no pupil gate and does not explicitly gate on `p1_valid`. Any stricter P1 confidence/conditioning gate must have a separate mask and a common-support comparison. Do not silently change baseline coverage.

A missing or uncertainty-unstable P1 triangle invalidates this normalization. With two valid P4 points, a separately labeled partial estimate may be produced if rank and branch checks pass. With fewer than two P4 points, the first implementation rejects for insufficient redundant support; this is a design policy, not a universal rank theorem. Missing P4 points and area ratios must not be fabricated.

The stored detector may already use similarity, triangle geometry, or temporal history to select points. Audit those assumptions and raw candidates before treating a narrow distribution of selected triangle shapes as optical evidence. This plan does not change or rerun the detector. Predictions remain conditional on the stored detection pipeline unless independently localized observations are evaluated.

## 3. Recommended model: a state-dependent map of sampled grid images

Use globally calibrated functions

$$
\widehat{\mathbf v}_j=\mathbf D(x;\beta)+\mathcal T(x;\beta)\mathbf r_j,
\qquad x=(\theta,A),
$$

so

$$
\widehat{\mathbf u}_j=\mathbf D+(\mathcal T-I)\mathbf r_j,
\qquad
\widehat{\mathbf q}_j=\mathbf c_1+\ell_1[\mathbf D+\mathcal T\mathbf r_j].
$$

All predictions share one state. `D` has two components and `T` four entries, but their values are determined by `theta,A` and fixed global coefficients. They are not six additional framewise unknowns.

`D` models relative common translation. `T` is an effective state-dependent deformation map for the three sampled source locations, including unequal directional magnification, shear, and orientation changes when supported. Three samples admit an exact affine representation even if the continuous source-field mapping is nonlinear. This fact does not guarantee a low-degree `T(theta,A)` transfers across arbitrary poses/P1 contexts or identifies physical aberration coefficients.

Use the P1 centroid in predictions, not the measured P4 centroid; the latter absorbs displacement error and leaks a withheld P4 point. The implied summaries are

$$
\widehat d_x=D_x,\qquad \widehat d_y=D_y,\qquad
\widehat\rho_4=|\det\mathcal T|.
$$

Retain them for reporting, but do not add independently weighted centroid/area residuals to the six coordinate residuals. Fit the coordinates directly instead of treating all entries of an observed `E4 E1^-1` map as independent low-noise measurements.

### 3.1 Explicit capacities and new gaze scale

For every new candidate use `t=theta_deg/10` and `L=log1p(A_diopters)`. For `D_x` use the baseline displacement model's seven-term functional family, with newly fitted coefficients:

$$
\phi_d=[1,A,t,tL,t^2,t^2L,t^3]^\top,\qquad D_x=\beta_d^\top\phi_d.
$$

The two required capacity comparisons are:

| Candidate | Basis for each of `D_y,T11,T12,T21,T22` | Global coefficients |
|---|---|---:|
| `conditional27` | `[1,t,L,tL]` | `7+5*4=27` |
| `conditional37` | `[1,t,L,tL,t^2,t^2L]` | `7+5*6=37` |

Start numerical implementation/debugging with `conditional27`; require the paired `conditional37` evaluation before deciding whether state-dependent deformation has been adequately modeled. The cubic-field example in Theory.md shows why field translation can induce quadratic spacing changes, motivating `t^2,t^2L`. It does not prove that either capacity is sufficient or that the larger model should always win.

Select using training-only grouped predictions and stability, then evaluate the selected design on untouched outer folds. Coefficients and sensitivities need not be equal between matrix entries or pairs. Do not enforce mirror symmetry about zero gaze unless justified by actual source/eye/camera alignment. Do not freeze the determinant-one part of `T` or penalize departure from similarity as though it were necessarily noise.

Direct matrix-entry prediction preserves linear coefficient fitting at fixed states. It does not enforce determinant sign. Audit correspondence parity and inspect the predicted determinant over training-supported states and P1 contexts; audit the larger computational box separately. Flag unsupported sign changes, near-singular predictions, and extrapolation. Do not silently repair them by reordering points. A determinant-constrained parameterization is a separately versioned nonlinear coefficient problem, not an unchanged linear profiler.

If both capacities leave repeatable state-dependent residual structure, report model inadequacy. A further basis extension must be specified and selected within development/training data rather than added after examining final transfer results. Do not absorb the pattern with framewise distortion freedoms.

### 3.2 Compatibility with the frozen baseline

`conditional27/37` describe capacity, not compatibility with old coefficient arrays. Store a new model schema with `theta_scale_deg=10`, physical coefficient/basis definitions, state encoding, and the exact target grid. Missing or incompatible scale metadata must cause an explicit load error for new artifacts.

Evaluate `models/quadratic_model.json` with its original adapter, including `t_b=theta_deg/15`. Do not edit that JSON or reinterpret its coefficients at `theta/10`. A nominal-range change is not a coefficient conversion.

If an explicit conversion of a polynomial initializer is ever needed, a term `c_15*(theta/15)^n` becomes `c_10*(theta/10)^n` with

$$
\boxed{c_{10}=(10/15)^n c_{15}.}
$$

Apply the same factor to mixed terms carrying that power of gaze; degree-zero terms are unchanged. Transform coefficient priors/covariance consistently if claiming equivalent fitting assumptions, and verify predictions plus physical-degree derivatives. The preferred first candidate fits fresh coefficients; conversion must never happen implicitly in the baseline path. All new state Jacobians include `dt/dtheta_deg=1/10`.

### 3.3 Meaning of full information and conditional derivatives

This is a **six-residual conditional full-position model**: it consumes every point and retains all P4 geometry relative to measured P1, but does not count P1's three normalized shape/orientation dimensions as separately modeled state evidence. Section 11 defines the optional nine-component model.

For a state derivative at fixed observed P1 context,

$$
\left.\partial_z\widehat{\mathbf v}_j\right|_r
=\partial_z\mathbf D+(\partial_z\mathcal T)\mathbf r_j.
$$

A total physical derivative along a state-dependent P1 trajectory also has `T partial_z r_j`. Keep those derivatives conceptually separate. Source optical sensitivities, conditional inverse sensitivities, and noise derivatives of the shared measured context are not interchangeable.

The conditional mapping can fail when nuisance geometry changes the response beyond what observed P1 context and two states explain. Test such dependence; a complete source-coordinate optical model or extra context terms are later hypotheses requiring calibration, not automatic consequences of having three pairs.

## 4. Measurement noise and shared-input uncertainty

The measurement is the detector-reported point position. Record its definition and detector version/configuration where available; mark missing provenance explicitly. Geometric chief-ray intersections, intensity centroids, peaks, and fitted centers must not be silently interchanged. A synthetic-image/ray-tracing validation should distinguish optical position changes from the actual localization pipeline's response.

The P1 coordinates occur in `v`, `r`, and the denominator. For

$$
e(P;x,\beta)=v(P)-F(x,r(P);\beta),
$$

propagate the complete residual:

$$
R_e\approx J_{e,P}\Sigma_PJ_{e,P}^\top,
\qquad J_{e,P}=J_{v,P}-J_{F,r}J_{r,P}.
$$

Using `cov(v)` while treating `r` as exact omits shared-input error. This is first-order errors-in-variables propagation, not an exact nonlinear likelihood.

A provisional coordinate-noise estimate can use second differences of all 12 coordinates within contiguous, correspondence-stable training-fixation cores. Dividing their covariance by six assumes independent localization noise and locally linear true motion. Acceleration, temporal correlation, and selection errors violate that assumption; label it a short-timescale effective covariance unless independently justified. Never bridge gaps or fixation boundaries. Do not estimate noise from residuals after fitting free states, which can remove variation along fitted state directions.

Use a declared training-only robust covariance/shrinkage/floor policy and report sensitivity to pointwise versus correlated assumptions. A numerical floor stabilizes a noise estimate but does not create information. Systematic field-dependent residuals require model-discrepancy analysis, not automatic inflation of peripheral noise or rejection of distorted triangles.

### 4.1 Fixed-reference weights for the first implementation

To preserve an auditable quadratic/profiled objective:

1. Fit a pilot coefficient model using training data only, initialized at nominal states.
2. Freeze a training-supported reference state, for example componentwise median training anchors, and the pilot coefficients.
3. For each frame's P1 geometry, generate reference P4 points at that state and evaluate `J_e,P` there. Propagate the frozen coordinate covariance. The weights then depend on P1 and frozen calibration, not observed P4 coordinates.
4. Freeze these matrices throughout final calibration and state inversion; store their provenance. Use principal covariance submatrices for subsets.

For capacity-isolation comparisons, use a common fold-local reference-weight construction where feasible, and separately report sensitivity to capacity-specific pilots. Otherwise changes in noise weights and response capacity are confounded.

This is a reference-covariance approximation, not exact state-dependent maximum likelihood. Check sensitivity to other training-supported gaze/reference demands, including central and peripheral training anchors, and to small-noise Monte Carlo. Never use an excluded endpoint or demand to learn a fold's reference weights. A future state-dependent Gaussian likelihood needs the log determinant and derivatives, or an explicitly declared feasible-GLS objective; changing weights silently invalidates the claimed exact profiling of a fixed objective.

The quadratic candidate comes first. Optional robustness is separate. A whole-frame robust loss may use the full Mahalanobis norm with one scalar frame weight. Componentwise robustness after full whitening is not independent physical-point downweighting. Point exclusion is handled by subset inversions. Neither robust weighting nor covariance floors may be used to hide unmodeled reproducible distortion.

## 5. Calibration: global responses and latent framewise states

Split into training and evaluation groups before fitting, covariance estimation, model selection, or learned initialization. Each frame has its own two states; only its fixation mean receives a nominal anchor. Demand is not an exact per-frame accommodation label.

For `K` training fixations with retained counts `n_k`, use

$$
\begin{aligned}
\mathcal J(X,\beta)={}&\tfrac12\sum_i\alpha_i
 \|L_i[v_i-F(x_i,r_i;\beta)]\|^2\\
&+\tfrac1{2K}\sum_k\left[
 ((\bar\theta_k-\theta_k^{nom})/\sigma_\theta)^2+
 ((\bar A_k-A_k^{demand})/\sigma_A)^2\right]\\
&+\tfrac\lambda2\|R_\beta(\beta-\beta_0)\|^2+\mathcal J_{time},
\qquad \alpha_i=1/(K n_{k(i)}),\quad L_i^\top L_i=R_{e,i}^{-1}.
\end{aligned}
$$

The nominal anchors set an approximate physical-state convention; they do not independently measure accommodation. Start anchor-scale sensitivity at 0.1 degree and 0.25 D, then examine weaker/stronger choices. Equal-fixation optical weights prevent long intervals from dominating.

Initialize `beta_0` by regularized regression at nominal states using the new basis and actual five-target assignments. Normalize design columns using training data and document penalties in those coordinates. A provisional prior may shrink non-intercept coefficients toward this initializer, with a zero-prior comparison. Do not shrink shape change to zero merely because it is non-similarity, or use regularization to claim identification of unsupported directions. Select prior strength within training folds; baseline numeric strength is not automatically comparable after changing channel count/scaling.

Initialize states from nominal anchors or a training-fold-only baseline fit. Baseline estimates are initialization, not ground truth or extra per-frame penalties. Use multiple calibration starts to assess dependence on initialization.

For the first geometry-isolation experiment set `J_time=0` and retrain a matched two-channel control without temporal regularization. Separately compare both under a baseline-like temporal policy with declared normalization, original-time gaps, and within-fixation links. Do not compare smoothed new states against unsmoothed controls and attribute the difference to geometry. Application and P4 holdouts remain independent per frame.

### 5.1 Variable projection

At fixed states and measured P1 context, predictions are linear in `beta`. Solve the weighted regularized coefficient problem by QR/SVD including its prior, then optimize the profiled objective over bounded framewise states. The existing `ProfiledProblem` architecture may be reused only after generalizing channel count and auditing old loader/model assumptions. This is not a knot model.

The profiled derivative must include the dependence of optimal coefficients on states. For an augmented linear system `B(X) beta ~= b`, with `r=B beta*-b`, fixed `b`, and full column rank under the declared regularization,

$$
(B^\top B)\,d\beta^*=-[(dB)^\top r+B^\top(dB)\beta^*],
\qquad dr=(dB)\beta^*+B\,d\beta^*.
$$

Use factorized solves rather than explicit inverses. Include derivatives of a changed right-hand side if the implementation alters this formulation. Anchors and temporal penalties have their own derivatives. Verify the complete derivative with finite differences on small problems.

Encode new optimizer states as `(theta_deg/10,A_diopters/4)`, with physical bounds `[-20,20]` degrees and `[0,6]` D. This gives encoded bounds `[-2,2]` and `[0,1.5]`, not the nominal calibration domain. Record the encoding independently of the legacy baseline.

A matrix-free profiled Jacobian avoids a dense matrix over roughly 140,000 latent variables. Profiling introduces global low-rank coupling; its Jacobian is not frame-block-diagonal. Bounded trust-region least squares with Jacobian/vector products is a suitable reference.

Require declared-unit projected stationarity, stable steps/cost, accurate inner solves, and any robust-weight convergence. An evaluation limit is a nonconverged result, not a selected model. Preserve checkpoints and failed alternatives separately.

## 6. Independent framewise estimation

Freeze coefficients, covariance construction, capacity, and preprocessing. For available P4 indices `I`, solve

$$
\widehat x_I=\arg\min_{x\in\mathcal B}
 e_I(x)^\top[R_e]_{II}^{-1}e_I(x).
$$

Use the covariance principal submatrix before inversion: `[R_e^-1]_II` generally differs from `[R_e]_II^-1`. Selecting rows of a fully whitened residual can mix excluded P4 observations into the solve.

An initial offline start grid can use gaze `[-20,-10,-5,0,5,10,20]` degrees and accommodation `[0,1,2,3,4,5,6]` D. The ±20 starts cover the computational box; they are not calibration targets. Refine using bounded trust-region least squares and analytic state derivatives. Test denser/adaptive grids on development cases because a finite grid does not guarantee every branch. Optimize speed only after matching the scalar reference.

Cluster converged solutions by declared physical tolerances. Report minimum cost, numerical ties, other plausible branches, bounds, projected stationarity, and noise-scaled Jacobian singular values. A smallest-accommodation tie-break may provide a reproducible representative but must not conceal ambiguity; numerical ties are not confidence intervals.

The all-three result is the primary unsmoothed estimate. Coordinate residuals and implied centroid/area are diagnostics of the shared optical response. Do not force the result toward the baseline trajectory, repair it to a similarity pattern, or silently reject peripheral states with expected deformation.

## 7. Three P4-point holdout checks

With coefficients learned outside the evaluation fixation/capture, repeat for each P4 point `j`:

1. Build input containing all P1 points and only the other two P4 points. Use their marginal covariance and subset-only validity.
2. Solve for the shared state with rank, stationarity, bounds, and branch diagnostics.
3. Before accessing the excluded value, predict `qhat_j=c1+ell1*(D+T*r_j)` for every retained plausible branch, including the calibrated distortion response.
4. Compare with observed `q_j` and store pixel/normalized errors, predictive uncertainty, and differences from the all-three state.

Do not use the excluded point to choose a branch. If the subset is ambiguous, report its prediction set or an inconclusive result. Weak rank is not proof that the point is wrong. If insufficient P4 observations remain, record an unavailable subset test rather than inventing a prediction.

All three P1 points remain allowed because only a P4 point is held out. Holding out a whole P1/P4 pair removes the original triangle normalizer and is a different experiment.

### 7.1 Leakage prohibition

The excluded point must not affect state estimation through its coordinate, full P4 centroid/area/map, similarity magnification, all-three baseline inverse, all-three warm start, state penalty, full-data whitening, geometry-derived confidence/gates, branch choice, or a model trained on the evaluation frame. Fixed evaluation-population membership may be reported separately from subset-input validity.

A mandatory unit test perturbs the excluded coordinate arbitrarily and checks unchanged subset preprocessing, weights, starts, states, branch sets, and predicted point. Only the later score may change. Perform this test for every point and axis, not only one synthetic case.

Upstream detector selection may already use all points or triangle assumptions. These tests validate predictions conditional on stored detections, not independence of the full image-detection pipeline. A similarity-constrained detector can suppress real optical deformation; raw candidates or independently localized measurements are separate evidence.

### 7.2 Correlated predictive uncertainty

For a regular interior subset inverse, let `J_I,J_j` be state derivatives at fixed P1 context, with its localization error already propagated into `R_e`. Define

$$
K_I=(J_I^\top R_{II}^{-1}J_I)^{-1}J_I^\top R_{II}^{-1},\qquad M_j=J_jK_I.
$$

The first-order held-out error is `epsilon_j-M_j epsilon_I`, with covariance

$$
V_j=R_{jj}+M_jR_{II}M_j^\top-R_{jI}M_j^\top-M_jR_{Ij}.
$$

Cross terms matter because P1 geometry is shared. This excludes coefficient uncertainty and systematic model error; assess them through grouped refits/sensitivity or explicitly propagated calibration covariance. Bounds, weak rank, and multiple branches require profiles/simulation or prediction sets instead of a single Gaussian ellipse.

A Mahalanobis score `e_j^T V_j^-1 e_j` needs training-only held-out threshold calibration and a policy for the maximum of three tests. Without verified clean references, an empirical percentile is an exceedance rate, not validated detector false-positive probability. Large expected deformation is not itself the score.

## 8. Interpretation, model inadequacy, and optional partial estimates

Keep the all-three estimate and every subset result. Distinguish numerical failure, insufficient identification, unexplained geometry, repeatable field-dependent model discrepancy, and suspected point-localized error. A good detection can disagree with an underfit model, especially at an off-center gaze. A distorted pattern can be correct when predicted by the calibrated state.

A later recovery policy may choose a two-P4 subset under a declared rule, preserving and labeling the original all-three result. Choosing a subset after inspecting all three scores is an operational decision, not an untouched validation test. Three points may be insufficient to isolate the faulty point.

Do not introduce unrestricted framewise offsets, scale, shear, radial coefficients, or affine maps to erase errors. A new nuisance state requires identifiable support and consumes residual redundancy. Shared calibration/localization bias or perturbations along state directions can remain mutually consistent but wrong.

## 9. Evaluation design

### 9.1 Separate geometry, capacity, normalization, and priors

Retain the frozen 13-coefficient baseline as historical reference. For captures 1–4 held-out evaluation, retrain the two-channel control within each fold; the all-data frozen model has seen those conditions. Keep its adapter's declared basis scale separate from new candidate scaling.

Compare `conditional27`, `conditional37`, and fold-local two-channel controls under matched support, mean anchors, temporal policies, and declared noise assumptions. Include a reduction of the same full-model predictions to centroid/area when testing which measurements matter; propagate that reduction's covariance rather than reusing independent coordinate weights. Capacity and regularization changes must be reported, not attributed solely to extra geometry.

An optional similarity-restricted model can test whether unequal stretching/orientation response beyond that restriction improves prediction. Declare its state functions, coefficient count, parity, priors, and fitting objective. It is a reduced ablation, not a preprocessing standard or detector validity rule. A nonlinear determinant/similarity parameterization must not be mislabeled as the unchanged linear coefficient profiler.

Keep square-root P1 area in the first comparison. Fitted magnification, singular-value summaries, and RMS-spread normalization are separately labeled later alternatives. Do not treat deterministic summaries as extra independent coordinates.

A geometry decoder fitted at frozen baseline states is a diagnostic of those states, not independent confirmation; it cannot replace joint latent-state calibration.

### 9.2 Grouped splits and exact gaze conditions

Use entire fixations as the minimum split unit. The initial outer scheme holds out one nominal gaze across all four captures, giving five folds at `-10,-5,0,5,10` degrees. Inner grouped splits select capacity, priors, covariance policy, and thresholds without consulting the outer data. Holding out ±10 is extrapolation relative to the remaining gaze anchors; report it accordingly.

Also evaluate whole-capture/demand holdouts, including interior demands. Endpoint-demand holdouts are extrapolation. Capture, demand, time, and drift may be confounded; four captures provide limited evidence to separate them. Read demand labels from metadata rather than old experiment constants.

In every outer fold perform all-three inversion and each P4 holdout. Freeze the selected design before final unlabeled transfer checks on captures 5/6. Repeated tuning on those results makes them development data and must be disclosed.

Report the fixed evaluation population and any baseline-matched subset separately. Do not exclude difficult rows using candidate residuals and report only survivors.

### 9.3 Required outcomes

Report point/axis prediction errors by fixation, capture, nominal gaze, and demand; predictive uncertainty calibration where justified; all-three residuals; subset/full-state differences; nominal fixation means; fixation spread; numerical convergence; rank/conditioning; branches; bounds; state/P1-context support; coverage; rejection; and partial estimates.

The main selection question is whether a state inferred without one P4 point predicts its response reproducibly on unseen conditions without unacceptable loss of stability or coverage. Lower training error, smoother traces, or smaller linear-versus-joint discrepancies are insufficient.

### 9.4 Distortion-specific diagnostics and controlled tests

For each of the five nominal gaze conditions and each capture/demand, report observed and predicted common displacement, all four effective-map entries where measurable, area ratio, and optional stretch/orientation descriptors. These are correlated diagnostics derived from coordinates, not extra likelihood terms. Evaluate expected maps at nominal-state probes separately from estimates at inferred states; nominal demand is not a measured accommodation reference.

Inspect signed-gaze asymmetry, central versus peripheral prediction errors, and residual dependence on P1 shape/context. Source-grid offsets can make zero gaze off-axis, so do not pool positive/negative gaze under an assumed symmetry. Do not infer an optical field angle from a gaze label alone.

Compare `conditional27` and `conditional37` specifically on held-out coordinate errors and noise-scaled state sensitivity/conditioning. Repeatable residual curvature can indicate insufficient response capacity; isolated errors may suggest localization/selection issues, but neither pattern proves its cause. Check sensitivity to priors before claiming the new shape channels add state information.

Required synthetic distinctions:

- Use a known shared-state coordinate model with non-similarity and gaze/accommodation curvature. Correct state predictions should explain its deformation without triggering a similarity-based rejection. Verify rank before expecting accurate state recovery.
- Separately sample a nonlinear source-field mapping at three sites. Demonstrate exact three-site affine reconstruction without claiming recovery of the continuous mapping; an additional simulated site can expose non-affine spatial structure. This is a geometry test, not proof that either empirical capacity fits all optical systems.
- Perturb one P4 point, including area-preserving perturbations, and test held-out noninterference and error sensitivity. Compare with a freely fitted per-frame affine map that can absorb a corrupted triangle exactly.
- Perturb along genuine state-response directions, introduce incorrect correspondence, and apply common translation/scale. Some state-like errors can evade residuals; document that limit rather than promising universal artifact detection.

Synthetic tests assess mathematics and failure sensitivity, not real detector accuracy or physiological validation. Any optical/image simulation must declare source geometry, coordinate units, eye/relay model, and localization definition.

## 10. Proposed implementation and output contract

These are proposed paths, not already implemented files:

```text
full_position/geometry.py       # mapping; P1 context; mask-safe coordinate data
full_position/model.py          # conditional27/37, theta_scale_deg=10; derivatives
full_position/noise.py          # shared-input covariance; subset marginals
full_position/calibrate.py      # fold-local initialization and profiled fitting
full_position/invert.py         # bounded multistart reference and diagnostics
full_position/validate.py       # grouped tests; P4 holdouts; distortion ablations
full_position/schema.py         # basis/scale, provenance, and result validation
experiments/full_position/<run_id>/
```

These responsibilities do not require additional services/processes. Reuse utilities only where semantics match; specifically audit legacy target arrays, degree scaling, two-channel assumptions, pupil gates, and knot-model defaults.

Model artifacts must identify schema/version, source hashes, point order and correspondence/parity, normalization, exact gaze/demand labels, `theta_scale_deg=10`, state encoding, coefficient basis/order, source-grid geometry or its absence, measurement/localization definition, noise/reference weights, prior, anchors, temporal policy, computational bounds, training groups/support, software/seed, convergence, and failed alternatives. Baseline artifacts remain unchanged and separately identified.

Frame outputs preserve every original row with validity/support flags, primary state, predicted coordinates, observed/predicted `d_x,d_y,rho_4`, covariance/conditioning, costs, branches/bounds/stationarity, all subset states/predictions, held-out errors, and reasons for inconclusive tests. Optional map/stretch diagnostics are computed only when available and labeled as such. Missing values remain missing; plot masks do not modify estimator inputs.

Pseudocode:

```python
for outer_fold in grouped_folds([-10, -5, 0, 5, 10]):
    train, test = split_before_preprocessing(outer_fold)
    config = select_inside_training_groups(train)  # compare 27/37; gaze scale 10
    model = calibrate_shared_state_geometry(train, config)
    for frame in test:
        p1 = build_p1_context(frame.p1, frame.p1_flags, model)
        all_result = invert_with_only_available_p4(p1, frame.p4, model)
        for j in range(3):
            kept = build_subset_input(p1, frame.p4_except(j), model)
            subset = invert_or_report_insufficient_support(kept, model)
            prediction = predict_excluded_p4_or_report_inconclusive(p1, j, subset, model)
            score = score_available_point_after_prediction(prediction, frame.p4[j], model)
            save_holdout_result(j, subset, prediction, score)
        save_primary_result_without_repair(all_result)
```

Acceptance tests before real-data claims:

1. Validate exact target labels and normalization; reject legacy target substitutions. Verify nominal encoded gazes `[-1,-0.5,0,0.5,1]`, physical/encoded bounds, and schema-required gaze scale.
2. Test permutation, reconstruction, nonredundant geometry rank, and area identities. Confirm invariance only to the declared common transformations and preserve distortion responses.
3. Compare analytic state and profiled-coefficient derivatives with finite differences, including `dt/dtheta_deg=1/10`. Verify any explicit 15-to-10 coefficient conversion preserves predictions/physical derivatives; preserve frozen baseline outputs through its unchanged adapter.
4. Test complete residual covariance and subset marginals, all-point holdout noninterference, weak-rank/multibranch subsets, P1 degeneracy, bounds, and invalid-row persistence.
5. Run the distortion-specific tests in section 9.4 and compare any accelerated solver with the scalar reference. A documentation-only change does not constitute passing these implementation tests.

## 11. Optional joint P1/P4 and optical extensions

The conditional model retains P1 shape as context. It does not assume its optical mapping is invariant. A genuine nine-component normalized model can also predict three P1 shape/orientation coordinates without double-counting them.

On a fixed branch write `H1=E1/ell1`, with determinant `2s`, `s=+1` or `-1`. A local chart `chi=(phi,kappa,b)` is

$$
H_1=[e^\kappa\mathbf e_\phi\;\;b\mathbf e_\phi+2s e^{-\kappa}\mathbf e_\phi^\perp].
$$

Recover `phi` from the first edge, `kappa=log(norm(first edge))`, and `b` by projecting the second edge onto its unit direction. Handle angle wrapping on a declared branch.

Calibrate `chi=G1(x;gamma)` and predict

$$
y_{joint}=[\chi_{obs};v_{obs}],\qquad
F_{joint}(x)=[G_1(x);\{D(x)+T(x)r_j(G_1(x))\}_{j=1}^3].
$$

This uses nine nonredundant components with full joint covariance and assumes a supported P1 response to the declared states/conditions. If unsupported, retain P1 as context/diagnostics or introduce only identifiable nuisance effects. Do not force accommodation to account for arbitrary P1 changes.

This changes coefficient fitting and derivatives; the conditional six-residual profiler cannot simply be relabeled. Use explicit new basis/scale metadata and test incremental held-out benefit before adoption.

A later source-coordinate/ray-traced model can predict both reflections directly from actual source positions/directions and an optical prescription. It requires source and camera geometry, measurement-definition compatibility, and identifiable global/nuisance parameters. Three sites alone do not justify unrestricted per-frame radial, polynomial, or Seidel coefficients. Denser sources or controlled field variations could test spatial nonlinearity, but are not prerequisites for the current sampled shared-state estimator.

## 12. References and completion boundary

Repository facts are grounded in the linked baseline/interval files. [Theory.md](Theory.md) derives the geometry, illustrative field-distortion example, conditional model, and holdout covariance.

- [Wu et al. (2023), High-resolution eye-tracking via digital imaging of Purkinje reflections](https://pmc.ncbi.nlm.nih.gov/articles/PMC10166114/): reflection paths and angular optical response; not validation of the proposed accommodation model.
- [NIST, Combining uncertainty components](https://physics.nist.gov/cuu/Uncertainty/combination.html): first-order covariance propagation.
- [SciPy, least_squares](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html): bounded trust-region optimization and Jacobian operators.
- [scikit-learn, Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html): training/model-selection separation and grouped evaluation.

This revision changes documents only. It establishes neither fitted candidate coefficients nor new real-data performance, runtime, detector behavior, or physiological accuracy. The next implementation deliverable is a reproducible geometry/noise/inversion prototype with target/scale and leakage tests, followed by fold-local calibration and distortion-aware held-out comparison. Preserve the frozen estimator until comparative evidence supports a replacement.
