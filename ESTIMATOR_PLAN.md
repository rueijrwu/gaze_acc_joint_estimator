# Full-position shared-state estimator: proposed algorithm

## Status and objective

**This is a proposal, not an implemented or evaluated estimator.** It accompanies [Theory.md](Theory.md). Preserve `joint_m2.py`, `models/quadratic_model.json`, stored detections, and existing result directories as the historical baseline. New code, models, and results must have separate paths and schemas.

The objective is to estimate one horizontal gaze/accommodation state per frame while requiring that state to predict the positions of all three P4 reflections relative to their corresponding P1 reflections. Extra geometry provides both potential state information and testable prediction residuals. Do not reduce the new data back to just centroid displacement and triangle area.

The required normalization is unchanged:

$$
\ell_1=\sqrt{\mathcal A_1},\qquad
 d_x=(c_{4,x}-c_{1,x})/\ell_1,\qquad
 \rho_4=\mathcal A_4/\mathcal A_1.
$$

`rho_4` is an area ratio, not its square root. Area means the triangle formed by reflection centers, not blob area.

## 1. Baseline inspection and implications

This proposal was prepared against branch commit `9fabb3cdfe6b4651c7b5b16cb9df573fe91b63cf`. The following are **saved repository results**, not results regenerated for this proposal.

| Item | Stored status |
|---|---|
| Baseline | 13-coefficient model of `[d_x,rho_4]` |
| Calibration | 20 reviewed fixations, captures 1–4, central 80% of each interval |
| Retained calibration observations | 71,729 after 55 jump-filter rejections |
| Selected training stage | Quadratic fit; 17 accepted iterations; recorded as converged |
| Later robust stage | Handoff reports that weight convergence was not achieved |
| Capture 5 | 25,226 retained estimates / 26,370 rows |
| Capture 6 | 34,080 retained estimates / 35,001 rows |
| Equivalent-minimum ambiguity | Saved counts are zero for both captures |
| Accommodation-bound counts | Capture 5: 7 upper; capture 6: 2 lower and 8 upper |
| Gaze outside nominal anchor range | Capture 5: 8,747; capture 6: 12,637 |
| Accommodation outside nominal anchor range | Capture 5: 23; capture 6: 13 |
| Saved projected-stationarity value above `1e-3` | Capture 5: 7; capture 6: 11 |

Sources: [model and training diagnostics](models/quadratic_model.json), [capture summaries](experiments/captures_5_6_direct/summary.json), [HANDOFF.md](HANDOFF.md), and the measurement/training/inversion functions in [joint_m2.py](joint_m2.py).

A retained detection or estimate is not necessarily a numerically satisfactory inverse. The saved stationarity threshold is a diagnostic of that implementation, not a universal tolerance for a new objective. The summaries record reused inverse caches; packaging did not independently rerun the workflow. Historical source paths in provenance are not necessarily available here.

The baseline training uses soft fixation-mean anchors, a coefficient prior, and temporal regularization. `joint_m2.py` sets gaze-anchor scale to 0.1 degree, accommodation-anchor scale to 0.25 D, and temporal strength to 0.1 with scales 1 degree / 0.25 D. Its application inverse is independent per frame. Do not assume the defaults or old two-pair loading rules in `lib/` describe exp5; reuse mathematics only after auditing channel-count and data-policy assumptions.

Captures 5/6 have no reviewed target intervals. Their trajectories and geometric predictions can be assessed, but accommodation accuracy cannot be inferred from these stored estimates alone.

## 2. Data contract and feature construction

Read each payload's correspondence metadata. The recorded zero-based map is `pair_index=[2,1,0]`. Reorder P4 coordinates and their flags together into P1 correspondence; do not separately sort triangles and assume illuminator identity.

For each frame retain:

- Original frame index, timestamp, capture/fixation membership, all six coordinates, selected-point flags, correspondence metadata, and detector diagnostics.
- All three P1 points as the common reference geometry.
- Each P4 point as an individually maskable observation.

Define

$$
\mathbf c_1=\tfrac13\sum_j\mathbf p_j,\quad
\mathbf r_j=(\mathbf p_j-\mathbf c_1)/\ell_1,\quad
\mathbf v_j=(\mathbf q_j-\mathbf c_1)/\ell_1,\quad
\mathbf u_j=\mathbf v_j-\mathbf r_j.
$$

The optimization residual uses six components of `v`, ordered `[v1x,v1y,v2x,v2y,v3x,v3y]`. The model receives the three `r_j` as measured context. This is equivalent to using all six normalized pair-displacement components `u`; subtracting the same measured `r` from observation and prediction leaves the residual identical.

Keep the measured P1 centroid and scale for pixel-space reconstruction and auditing. Do not feed them as additional state observations in the first candidate. No per-frame P4 centering, rotation normalization, or normalization by P4 area is allowed in this representation.

**Validity:** reproduce the baseline mask separately. The baseline requires finite P1/P4 coordinates, all `p4_found`, and a numerically nondegenerate P1 triangle; it has no pupil gate and does not explicitly gate on `p1_valid`. A proposed stricter P1-confidence or triangle-conditioning gate must have its own mask and an explicit common-support comparison. Do not silently change the baseline's support.

A P1 triangle that is missing or unstable relative to localization uncertainty invalidates this normalization. With two valid P4 points the candidate may produce a separately labeled partial estimate if rank and branch checks pass. With fewer than two P4 points, the first implementation rejects for insufficient redundant support; this is a design policy, not a universal rank theorem. Never fabricate a missing P4 point or a missing area ratio.

## 3. Recommended first model: conditional full-P4 geometry

Use globally calibrated functions

$$
\widehat{\mathbf v}_j=\mathbf D(x;\beta)+\mathcal T(x;\beta)\mathbf r_j,
\qquad x=(\theta,A).
$$

Thus

$$
\widehat{\mathbf u}_j=\mathbf D+(\mathcal T-I)\mathbf r_j,
$$

$$
\widehat{\mathbf q}_j=\mathbf c_1+\ell_1\bigl[\mathbf D+\mathcal T\mathbf r_j\bigr].
$$

All three predictions share **one** state. `D` has two components and `T` has four entries, but their per-frame values are determined by that state and fixed calibration coefficients. They are not six additional framewise unknowns.

In particular, the prediction uses the P1 centroid, not the measured P4 centroid. The latter would absorb the displacement being tested and leak a withheld P4 point.

The model implies

$$
\widehat d_x=D_x,\qquad \widehat d_y=D_y,\qquad
\widehat\rho_4=|\det\mathcal T|.
$$

Retain these summaries for comparison. Do not append area or centroid residuals to the six coordinate residuals as independently weighted observations.

### 3.1 A bounded, explicit initial capacity

Let `t=theta/15`, `L=log1p(A)`, where numerical accommodation is in diopters. For `D_x`, use the same seven-term basis as the baseline displacement model:

$$
\phi_d=[1,A,t,tL,t^2,t^2L,t^3]^\top,\qquad D_x=\beta_d^\top\phi_d.
$$

For each of `D_y,T11,T12,T21,T22`, initially use

$$
\phi_0=[1,t,L,tL]^\top.
$$

This gives **27 global coefficients**: seven plus five groups of four. Call this candidate `conditional27`. It does not presume that vertical motion, shear, and area have identical coefficients or sensitivities.

A predeclared capacity extension adds `t^2,t^2L` to those five functions, giving **37 global coefficients**, `conditional37`. Select between these using grouped held-out predictions, not training residual alone. These are proposed empirical bases, not optical laws or evidence that their coefficients are identifiable. Preserve design-rank and regularization diagnostics.

Direct matrix-entry prediction makes coefficient fitting linear at fixed states. It does not automatically preserve determinant sign. Audit correspondence parity and test the predicted determinant over the calibrated support. A model with unsupported sign changes or near-singular predicted triangles must not be silently accepted or fixed by sorting points. Reduce capacity or separately evaluate a determinant-constrained parameterization; that change is a different fitting problem.

### 3.2 What “full information” means here

This candidate consumes every P1/P4 coordinate and retains all P4 information relative to the observed P1 triangle. It conditions on normalized P1 shape/orientation instead of claiming that their three geometric dimensions are separately calibrated physiological state measurements.

It is therefore a **six-residual conditional full-position model**, not a nine-channel joint likelihood. Section 11 specifies the optional nine-dimensional extension. State explicitly which version produced each result; do not advertise six conditional residuals as nine independent constraints.

## 4. Noise and shared-input uncertainty

The P1 coordinates occur in `v`, `r`, and the common denominator. For

$$
e(P;x,\beta)=v(P)-F(x,r(P);\beta),
$$

propagate uncertainty through the entire residual:

$$
R_e\approx J_{e,P}\Sigma_PJ_{e,P}^\top,
\quad J_{e,P}=J_{v,P}-J_{F,r}J_{r,P}.
$$

Using only `cov(v)` while treating measured `r` as exact omits their shared-input error. This is a first-order errors-in-variables treatment, not an exact nonlinear likelihood.

A provisional coordinate-noise estimate can use second differences of the 12 coordinates inside contiguous, correspondence-stable training-fixation cores. Division of their covariance by six assumes independent localization noise and locally linear true motion. Actual eye acceleration, temporal detector correlation, and selection errors violate that assumption. Label the result a short-timescale effective covariance unless independently justified. Never bridge gaps or fixation boundaries, and do not estimate noise from residuals after fitting free per-frame states, which can remove noise along the fitted state directions.

Use robust covariance estimation with a declared, training-only positive-definite shrinkage/floor policy. A numerical floor regularizes a noise estimate; it does not create geometric information. Record sensitivity to pointwise versus correlated coordinate-noise assumptions.

### 4.1 A concrete fixed-weight first implementation

To preserve an auditable quadratic/profiled objective:

1. Fit a pilot coefficient model from training data only, initialized at nominal states.
2. Choose one reference state from the training anchors, for example their componentwise medians, and freeze it with the pilot coefficients.
3. For each frame's P1 geometry, generate noise-free reference P4 points at that state. Evaluate `J_e,P` there and propagate the frozen coordinate covariance. Consequently the frame's weight matrix depends on P1 geometry and frozen calibration information, not observed P4 values.
4. Freeze these matrices throughout the final calibration and all state solves. Store the reference-state/pilot/covariance provenance. Use principal covariance submatrices for P4 subsets.

This is a declared reference-covariance approximation. Check its sensitivity to other training-supported reference states and to small-noise Monte Carlo. It is not maximum-likelihood estimation with an exactly known state-dependent covariance. A future state-dependent Gaussian likelihood must include the log-determinant term and its derivatives, or explicitly document a different feasible-GLS objective; do not update weights silently while calling the old coefficient profiling exact.

The quadratic candidate comes first. Optional robustness must be separately tested. A whole-frame robust loss may operate on the complete Mahalanobis norm with a scalar frame weight. Componentwise robust losses after full whitening do not correspond to independent physical P4-point downweighting. Point exclusion is handled by the subset inversions below, not by pretending correlated point residuals are independent.

## 5. Calibration: shared coefficients, latent framewise states

Split data into training and evaluation groups **before** any fitting, noise estimation, model selection, or learned initialization. Each training frame has its own two states; only its fixation mean receives a nominal anchor. Never assign demand as an exact framewise accommodation label.

For `K` training fixations with retained counts `n_k`, propose

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

The nominal anchors set an approximate physical state convention; they do not independently measure accommodation. Start anchor-scale comparisons at the baseline's 0.1 degree and 0.25 D, then assess weaker/stronger choices. Equal-fixation weighting prevents long intervals from dominating calibration.

Initialize `beta_0` by regularized regression at nominal states. Normalize design columns using training data; document the coefficient penalty in these scaled coordinates and do not penalize unidentifiable directions merely to claim identification. A provisional prior can shrink non-intercept coefficients toward this initializer, with an explicit zero-prior comparison. Prior strength must be selected inside training folds; the baseline's numeric strength is not automatically comparable after changing the number of channels.

Initialize latent states from nominal anchors or a **training-fold-only** baseline fit. Baseline states are initialization, not ground truth and not an extra per-frame penalty. Multi-start calibration is required to investigate dependence on initialization.

For the first geometry-isolation experiment set `J_time=0` and retrain a matched two-channel control with the same no-time policy. Separately compare both models under the baseline-like temporal policy; the latter must define its normalization, original-time gaps, and within-fixation links explicitly. Do not compare a smoothed new model against an unsmoothed control and attribute the difference to extra geometry. Application and P4 holdout inversion remain independent per frame.

### 5.1 Variable projection

At fixed `X` and measured P1 context, the predictions are linear in `beta`. Solve the weighted, regularized coefficient least-squares problem by QR/SVD; include the coefficient prior in that solve. Then minimize the profiled objective over all bounded framewise states.

The existing `ProfiledProblem` architecture is a possible starting point, but it must be generalized from two channels and audited for legacy model/loading assumptions. The residual count and basis are new; there is no historical knot model in this candidate.

The profiled derivative must include the dependence of the optimal coefficients on the states. For a stacked augmented linear system `B(X) beta ~= b`, define `r=B beta*-b`. With fixed `b` and full column rank after the declared regularization,

$$
(B^\top B)\,d\beta^*=-\bigl[(dB)^\top r+B^\top(dB)\beta^*\bigr],
\qquad dr=(dB)\beta^*+B\,d\beta^*.
$$

Use factorized solves, not an explicitly formed inverse. Include any state dependence of a changed right-hand side if the implementation changes this formulation. Anchors and temporal penalties have their own derivatives. Test these derivatives against finite differences on small problems.

Use scaled states `(theta/15,A/4)` with computational bounds `theta in [-20,20]`, `A in [0,6]`. A matrix-free profiled Jacobian avoids a dense matrix over roughly 140,000 latent variables. Profiling introduces global low-rank coupling; do not falsely declare the profiled Jacobian frame-block-diagonal. A bounded trust-region solve with Jacobian/vector products is a suitable reference implementation.

Require physical-unit or declared-scaled projected stationarity, stable steps and cost, accurate inner solves, and any robust-weight convergence. An iteration/evaluation limit is a nonconverged result, not permission to export it as the selected model. Preserve checkpoints and failed fits separately.

## 6. Framewise estimation after calibration

Freeze coefficients, covariance construction, model capacity, and preprocessing. For available P4 indices `I`, solve

$$
\widehat x_I=\arg\min_{x\in\mathcal B}
 e_I(x)^\top[R_e]_{II}^{-1}e_I(x).
$$

Use the covariance principal submatrix before inversion. In general, `[R_e^{-1}]_II` is not `[R_e]_{II}^{-1}`; selecting rows from a fully whitened vector can mix in excluded P4 observations.

A simple offline reference uses a declared Cartesian start grid covering the box, for example gaze `[-20,-10,0,10,20]` and accommodation `[0,1,2,3,4,5,6]`. Refine with bounded trust-region least squares and analytic state Jacobians. This is a proposed starting policy, not a proof that all inverse branches are found. Check denser/adaptive grids on validation cases. Optimize speed only after agreeing with the scalar reference.

Cluster converged solutions by declared physical-state tolerances. Report minimum cost, numerical ties, other plausible branches, bound activity, projected stationarity, and noise-scaled Jacobian singular values. The old smallest-accommodation tie-break may be retained for a reproducible representative, but ambiguity must remain visible. A near-equal-cost threshold is not automatically a confidence interval.

The all-three solution is the primary unsmoothed estimate. Extra coordinate residuals, predicted area ratio, and predicted pixel positions are diagnostics. Do not automatically force the new states toward the baseline trajectory.

## 7. Three P4-point holdout checks

With coefficients trained outside the evaluation fixation/capture, repeat for `j=1,2,3`:

1. Build an estimator input containing all three P1 points and only the other two P4 points. Use their marginal covariance and subset-only validity.
2. Solve for the shared state from that subset, including rank, stationarity, bounds, and branch checks.
3. Before accessing the held-out measurement, predict `qhat_j=c1+ell1*(D+T*r_j)` for every retained plausible branch.
4. Compare with measured `q_j`; store pixel and normalized prediction errors, predictive uncertainty, and the difference from the all-three state.

**Do not select a branch using the held-out P4 point.** When the subset is ambiguous, report its prediction set or an untestable/ambiguous result rather than picking the branch that makes the test pass. A poor subset rank also yields an inconclusive test, not proof that the withheld point is wrong.

The holdout may use all P1 points because only a P4 measurement is withheld. Holding out an entire P1/P4 pair is a different experiment: the original P1 triangle normalizer is then unavailable.

### 7.1 Leakage prohibition

The subset state solve must not depend on the excluded P4 point through any of the following: its coordinate, full P4 centroid/area/map, all-three baseline inverse, all-three warm start or previous-state penalty, full-data whitening, confidence/weight/gating computed from full P4 geometry, branch selection, or a model trained on the evaluation frame. Fixed dataset membership can be reported separately from state-input validity.

A mandatory unit test perturbs the excluded P4 coordinate arbitrarily and verifies unchanged subset preprocessing, weights, starts, fitted states, branch sets, and predicted point. Only the subsequent score may change.

Upstream detector selection may already use the other points or triangle assumptions. These tests establish prediction conditional on stored detections, not independence of the entire image-detection pipeline. Raw-candidate or independently localized checks are separate evidence.

### 7.2 Correlated predictive uncertainty

For a regular interior subset inverse, let `J_I` and `J_j` be prediction derivatives, with P1 context fixed in state differentiation but its noise already included in `R_e`. Define

$$
K_I=(J_I^\top R_{II}^{-1}J_I)^{-1}J_I^\top R_{II}^{-1},\qquad M_j=J_jK_I.
$$

To first order, the withheld prediction error is `epsilon_j-M_j epsilon_I`, with covariance

$$
V_j=R_{jj}+M_jR_{II}M_j^\top-R_{jI}M_j^\top-M_jR_{Ij}.
$$

The cross terms matter because all pairs share noisy P1 geometry. This expression excludes calibration-coefficient uncertainty and systematic model error; assess those with grouped refits/sensitivity or an explicitly propagated calibration covariance. Near bounds, weak rank, or multiple branches, use simulation/profile prediction sets rather than a single Gaussian ellipse.

A score such as `e_j^T V_j^-1 e_j` is useful, but its alarm threshold needs training-only held-out calibration and a policy for the maximum of three tests. Without verified clean reference data, an empirical percentile is a reference exceedance rate, not a validated detector false-positive rate.

## 8. Interpretation and optional partial estimates

Keep the all-three estimate and all three subset results. Do not turn a disagreement into an automatic claim that one point is an artifact. Distinguish numerical failure, insufficient subset identification, geometry inconsistency, and suspected point-localized inconsistency.

A future two-pair recovery policy may select a subset under a predeclared decision rule, but must preserve the original all-three result and label the exclusion. Selecting the best subset after inspecting all three scores is an operational decision, not an untouched held-out validation result. With only three points, evidence can be insufficient to isolate the faulty point.

Do not add unrestricted per-frame offsets, scale, shear, or triangle maps to make the residual vanish. A new nuisance parameter needs supported identifiability and reduces residual redundancy. Shared calibration bias or a perturbation along the model's state directions can still produce mutually consistent but incorrect estimates.

## 9. Evaluation design

### 9.1 Preserve separate comparisons

Keep the frozen all-data 13-coefficient baseline as a historical reference. For honest held-out evaluation on captures 1–4, retrain a two-channel control inside every training fold; the frozen all-data model has already seen those evaluation conditions.

Compare `conditional27`, `conditional37`, and the fold-local two-channel model under matched data support, mean anchors, temporal policy, and noise assumptions. Compare added geometry against added capacity explicitly. An area/centroid-only control is a nonlinear reduction of point predictions; its covariance must be propagated rather than reusing six independent coordinate weights.

As a diagnostic only, a geometry decoder fitted at frozen baseline states can show what those states predict. It is not independent confirmation of the baseline and must not replace latent-state calibration.

### 9.2 Grouped splits

Use entire reviewed fixations as the minimum split unit, never random adjacent frames. A useful initial outer scheme with these 20 intervals withholds one nominal gaze target across all four captures, producing five gaze-condition folds. Inner grouped splits select capacity, priors, and thresholds without consulting the outer evaluation data.

Also evaluate whole-capture/demand holdouts, including the two interior demands. Endpoint-demand holdouts are extrapolation tests. Capture, demand, time, and drift may be confounded: success or failure cannot isolate demand generalization from recording changes. With only four captures, uncertainty estimates and nuisance discrimination remain limited.

Inside each outer evaluation fold, perform both all-three inversion and the three P4-point holdouts. Freeze the selected design before inspecting captures 5/6 as final unlabeled transfer checks. Repeated tuning on their outputs makes them development data, which must be disclosed.

For stringent holdout scores, report both the full fixed evaluation population and any baseline-matched comparison subset. Do not remove difficult evaluation rows using the new model's residuals and then report only the survivors.

### 9.3 Required outcomes

Report held-out point prediction error by point/axis/fixation/capture; predictive-interval calibration where justified; all-three geometry residuals; subset/all-three state discrepancies; nominal fixation-mean agreement; fixation spread; numerical convergence; rank/conditioning; branch ambiguity; bound hits; state/context extrapolation; and coverage/rejection/partial-estimate rates.

Synthetic point perturbations test failure sensitivity and leakage, not real detector accuracy. Separately test area-preserving distortions, incorrect correspondence, common-scale/translation changes, and perturbations that resemble genuine state motion. The last can evade geometric residuals and should be documented.

Do not select a model just because it has smaller training residuals, smoother traces, or smaller linear-versus-joint discrepancy. The main selection question is whether a state inferred without a P4 point predicts that point reproducibly on unseen conditions without unacceptable loss of coverage or stability.

## 10. Proposed implementation and output contract

New implementation paths, not files already implemented:

```text
full_position/geometry.py       # P1 context; pair mapping; mask-safe observations
full_position/model.py          # conditional27/37; derivatives; implied d/rho
full_position/noise.py          # shared-input covariance and subset marginals
full_position/calibrate.py      # fold-local initialization and profiled fitting
full_position/invert.py         # bounded multistart reference and diagnostics
full_position/validate.py       # grouped evaluation and P4-point holdouts
full_position/schema.py         # model, result, and provenance validation
experiments/full_position/<run_id>/
```

These are logical responsibilities, not a requirement for extra services or processes. Reuse existing utilities only when their semantics match.

A model artifact must identify its schema, source hashes, correspondence and parity, measurement order, normalization, bases/coefficient order, noise model/reference covariance, coefficient prior, anchors, temporal policy, computational bounds, training groups, calibration support, software versions, seed, convergence, and failed alternatives.

Frame outputs retain every original row and include validity/support flags; primary state; predicted points; observed/predicted `d_x,d_y,rho_4`; covariance/conditioning; costs; branch/bound/stationarity flags; all three subset states and predictions; withheld-point scores and reasons for inconclusive tests. Missing quantities remain explicitly missing, not interpolated silently. Plot masks never modify estimation data.

Pseudocode:

```python
for outer_fold in grouped_folds:
    train, test = split_before_preprocessing(outer_fold)
    config = select_inside_training_groups(train)
    model = calibrate_shared_state_geometry(train, config)
    for frame in test:
        p1 = build_p1_context(frame.p1, frame.p1_flags, model)
        all_result = invert_with_only_available_p4(p1, frame.p4, model)
        for j in range(3):
            kept = build_subset_input(p1, frame.p4_except(j), model)
            subset = invert_multistart(kept, model)
            prediction = predict_excluded_p4(p1, j, subset, model)
            score = score_after_prediction(prediction, frame.p4[j], model)
            save_holdout_result(j, subset, prediction, score)
        save_primary_result_without_repair(all_result)
```

Acceptance tests before real-data claims: exact normalization and permutation identities; coordinate-reconstruction tests; complete feature-rank tests; analytic/finite-difference derivatives including profiled coefficients; shared-input covariance propagation; subset marginal covariance; held-out perturbation noninterference; ambiguous/rank-deficient subset handling; P1 degeneracy; bound handling; and persistence of invalid rows. Compare any accelerated solver with the scalar reference.

## 11. Optional extension: use P1 geometry as state evidence too

The conditional model preserves P1 geometry but does not predict it. A genuine nine-dimensional normalized joint model can add a calibrated P1-shape response, without double-counting redundant coordinates.

On a fixed P1 orientation branch, write `H1=E1/ell1`, whose determinant is `2s`, `s=+1` or `-1`. A local three-coordinate chart is `chi=(phi,kappa,b)`:

$$
H_1=\left[e^\kappa\mathbf e_\phi\;\;b\mathbf e_\phi+2s e^{-\kappa}\mathbf e_\phi^\perp\right].
$$

Recover `phi` from the first edge, `kappa=log(norm(first edge))`, and `b` by projecting the second edge onto that edge's unit direction. Handle angle wrapping on a declared local branch.

Calibrate `chi=G_1(x;gamma)` and predict

$$
y_{joint}=[\chi_{obs};v_{obs}],\quad
F_{joint}(x)=[G_1(x);\{D(x)+T(x)r_j(G_1(x))\}_{j=1}^3].
$$

This supplies nine nonredundant normalized components with their full joint covariance. It assumes that normalized P1 shape/orientation are predictable from the declared states/conditions. If that assumption fails, retain them as context/diagnostics or introduce only identifiable nuisance effects. Do not force a P1 response to fit by arbitrarily moving accommodation.

This extension changes the calibration problem; shape-chart functions and derivatives must enter the optimizer, and the first conditional model's six-residual profiler cannot simply be relabeled. Evaluate its incremental held-out benefit against the conditional model before adopting it. Without such evidence, the conditional full-P4 model is the recommended first implementation.

## 12. References and completion boundary

Repository facts are grounded in the linked files at the inspected commit. The geometric identities and proposed model/holdout formulas are derived in [Theory.md](Theory.md).

External method references:

- [NIST: Combining uncertainty components](https://physics.nist.gov/cuu/Uncertainty/combination.html), for first-order covariance propagation.
- [SciPy: least_squares](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html), for bounded trust-region optimization, Jacobian operators, and solver termination semantics.
- [scikit-learn: Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html), for separating training/model selection from grouped evaluation.

This document does not claim that candidate coefficients, real-data residuals, runtime, or physiological accuracy have been established. The next implementation deliverable is a reproducible geometry/noise/inversion prototype with the leakage tests above, followed by fold-local calibration and held-out comparison—not replacement of the frozen model before evidence is available.
