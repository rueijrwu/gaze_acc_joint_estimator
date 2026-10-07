# Implementation and results audit — exp5_full

**Repository:** `rueijrwu/gaze_acc_joint_estimator`  
**Audited commit:** `88f3ac5556f1a4aa3ca1e42db356853ab8c48f53`  
**Audit date:** 2026-10-07  
**Repository changes:** None. This report and numerical experiments are local audit artifacts.

## Executive assessment

The implementation substantially follows the mathematical core of `Theory.md` and `ESTIMATOR_PLAN.md`. It is a faithful first **conditional six-coordinate prototype**, not a completed validation program. The optional nine-coordinate joint P1/P4 model was explicitly deferred and its absence is not a defect.

The saved study does not establish improved gaze/accommodation accuracy over the two-channel control. It does establish working overdetermined geometric diagnostics and identifies important model/noise mismatch. A targeted numerical audit additionally demonstrates a concrete improvement: recover and polish lower-cost inverse candidates **before discarding them using the current gradient threshold**.

The most important refinement to CURRENT_STATUS.md is that some apparent inverse-search failures are not failures to enter the correct basin. The current scalar solver can reach the better minimum from several starts, reject every one of those candidates on its final stationarity threshold, and retain a worse branch. Adding starts or GPU candidates alone does not address this certification problem reliably.

## 1. Scope and evidence boundary

Reviewed source includes geometry, model bases/derivatives, coordinate-noise propagation, variable-projection calibration, scalar inversion, P4 holdout prediction/scoring, grouped loading/evaluation, reporting, schema handling, application, optional batched inversion, and all 14 acceptance-test definitions. Reviewed results include RESULTS.md, metrics excerpts, saved model parameters, frame outputs, and uncertainty records.

The repository archive could not be cloned into the execution environment. The complete 14-test suite and 27 fold/model calibrations were **not rerun**. Instead, targeted numerical experiments used:

- The exact `full_position/model.py`, copied from the connector and verified against Git blob SHA `f66f1339de6213432f1bbe60010121a5b77c8112`.
- The saved `gaze_-10/conditional37` coefficients and two frames from capture 1, rows 1380 and 1928.
- P1/P4 coordinates reconstructed algebraically from saved predicted points, predicted translation/map, and point errors. No missing data values were guessed.
- The saved reference residual covariances. No noise model or coefficients were refitted.
- NumPy 2.3.5 and SciPy 1.17.0 in the audit runtime.

The reconstruction uses `p_j-c1 = T^{-1}(qhat_j-mean(qhat))`, obtains `ell1` from that centered P1 triangle, then `c1=mean(qhat)-ell1*D` and `q_j=qhat_j+error_j`. Reconstructed summaries were checked against saved summaries, and the saved row-1928 weighted cost was reproduced. Tiny reconstruction/solver-environment differences can change the current pass/fail decision, which is part of the numerical brittleness being investigated.

The evidence package contains code, fixtures, every scalar-start result, profile candidates, polished results, and derivative verification. These are two targeted rows, not a newly evaluated population or an independent physiological reference.

## 2. What follows the theory and plan

### Geometry and state

`model.py` and `geometry.py` implement the prescribed P1 centroid and square-root P1 triangle-area normalization. All six P4 coordinates are retained. P4 flags and coordinates are reordered together. The new basis uses `theta/10` and the data loader checks `[-10,-5,0,5,10]` targets. The separately fitted summary control uses the equivalent baseline functional family at scale 10, without reinterpreting frozen scale-15 coefficients.

Predictions have the required form `vhat_j=D(theta,A)+T(theta,A)r_j`. The matrix is state dependent, not independently adjusted per frame. P4 centering does not absorb a prediction error. Area and centroids are derived summaries rather than duplicated residual blocks.

### Calibration

`calibrate.py` fits globally shared coefficients and framewise states, with soft fixation-mean anchors and equal-fixation optical weighting. The coefficient solve uses a scaled QR factorization. The profiled derivative includes the dependence of optimal coefficients on states, including the residual-dependent term. Jacobian operators capture the current arrays instead of following a mutable trial-state cache. These are important correct details.

Both coordinate capacities and the fold-local summary control use zero temporal regularization in the primary experiment. This is the intended first geometry-isolation policy. Failed calibration starts remain separate from accepted model artifacts.

### Noise and held-out measurements

`noise.py` propagates the complete residual's dependence on P1 and P4, rather than treating measured P1 context as exact. Reference weights depend on P1 and training-only pilot information, not observed held-out P4 coordinates. Subset solves use covariance marginals before inversion. Predictive covariance includes cross terms due to shared P1 errors.

`predict_holdout` has no argument containing the excluded P4 measurement; scoring is separate. The reviewed integration does not use an all-three baseline estimate to initialize a subset inverse. Prediction branches are determined before observing the excluded point. This is consistent with the intended conditional-on-stored-detections test.

## 3. High-priority numerical finding: certification can discard the better inverse

### Source behavior

`invert.py::invert` runs 49 bounded least-squares solves, then discards a candidate if `result.success` is false or the encoded unit-step projected-gradient mapping exceeds `1e-4`. It stores only a count for discarded starts, not their final states, costs, or termination reasons. Clustering and branch ranking occur after this rejection.

SciPy's small-step termination and the application's post-check are different criteria. A solver can stop with an essentially stable state while still failing a stringent absolute gradient test in a high-curvature, high-residual problem. It is correct not to equate solver success with validation; it is not robust to discard the candidate before an independent polishing/certification attempt.

### Targeted reproduction: capture 1, row 1380, exclude P4 index 1

The audit used only the retained four P4 coordinates in the inverse. In the local 49-start run, **eight starts reached the lower-cost minimum**, with state spreads below approximately `4e-7` in physical units, but none passed the post-check. An accepted higher-cost branch was selected instead.

| Quantity | Selected scalar branch | Lower branch after polishing |
|---|---:|---:|
| Gaze, degrees | -11.114328 | -11.000176 |
| Accommodation, D | 4.516246 | 0.961266 |
| Retained-coordinate weighted cost | 26,295.892702 | 24,441.696646 |
| Withheld-point prediction error, px | 5.958522 | 2.695215 |

Two exact-Hessian Newton polishing steps reduced the lower branch's encoded gradient below `7e-9`, satisfying the **existing** `1e-4` threshold. No tolerance relaxation or held-out-point branch selection was used. This recovers the same better minimum described in the repository GPU audit while exposing a more specific mechanism: a basin can be reached and then rejected by certification.

### Second case and limits

For row 1928, all 49 local starts reached essentially the same full-frame minimum, but none passed the post-check in this environment. Their state spread was below approximately `3e-7`. Polishing recovered the saved state and cost, with gradient around `6e-9`.

Across the two rows' full and three subset solves, all eight best profile candidates passed the original gradient criterion after two polishing steps. The exact model Hessians were checked by finite differences; maximum absolute discrepancy was about `2.4e-12`.

Several polished withheld errors still ranged from roughly 4.7 to 10.2 pixels. Fixing numerical acceptance does not make the larger response model adequate or establish physiological accuracy.

### Recommended numerical change

Retain all finite final candidates and their diagnostics, including rejected candidates. Distinguish budget exhaustion, small-step termination with unresolved stationarity, weak rank, invalid geometry, and a genuinely unlocated branch. Add a safeguarded polishing stage before final rejection and branch clustering. Check feasibility, projected stationarity, local curvature/minimum status, stable cost, and a declared physical state-correction tolerance.

For an interior fixed-covariance least-squares objective, use the exact Hessian

`H = J^T R^-1 J + sum_k (R^-1 e)_k Hessian(e_k)`.

A production polish must handle active bounds and indefinite Hessians; the supplied audit polish is deliberately limited to the tested interior cases.

## 4. Independent branch-check opportunity from the existing bases

At a fixed accommodation value and fixed P1 context, every predicted coordinate is cubic or lower in `t=theta/10`. Thus the fixed-weight cost is polynomial of degree at most six in t, and its derivative has degree at most five.

For each accommodation value, enumerate the real stationary gaze roots in `[-2,2]`, include both endpoints, and evaluate their costs. This checks the best gaze candidate conditional on that accommodation without depending solely on 2D optimizer paths. Sweep and adaptively refine accommodation, track distinct branches, and polish candidates in two dimensions.

The targeted audit implemented this calculation and checked its coordinate polynomials against the exact model. It exposed the low- and high-accommodation minima in row 1380. Root finding has numerical tolerances, and a finite accommodation grid is **not** a proof of global completeness. This should first be a reference audit method, not an unqualified production global solver.

## 5. What the saved real-data results establish

### Reported matched-support prediction errors

| Model | Gaze-holdout RMS, px | Capture-holdout RMS, px | Gaze-holdout nominal gaze-mean RMSE, degrees | Gaze-holdout nominal accommodation-mean RMSE, D |
|---|---:|---:|---:|---:|
| conditional27 | 3.423 | 4.067 | 0.761 | 1.135 |
| conditional37 | 6.197 | 4.354 | 0.742 | 1.450 |
| two_channel13 | Not a coordinate predictor | Not a coordinate predictor | 0.564 | 0.536 |

The coordinate RMS comparisons use 426 matching point tests per split family. The means are saved report values, not recalculated physiological errors. The study has 160 sampled rows per split family, 143 baseline-valid complete rows, only 20 fixation intervals, and four recordings. The two split families reuse observations.

`conditional27` is the more stable current coordinate candidate. `conditional37` is substantially worse at endpoint gaze holdouts, which are extrapolations relative to their training anchors. Its slightly better median at -5 degrees is not uniform improvement; the saved RMS in that fold is worse. Neither model improves overall agreement with nominal anchors in the saved study.

However, nominal accommodation demand is not measured accommodation. These results do not prove that the two-channel estimates are physiologically more accurate. Likewise, two observations and two states can yield near-zero optical residual without validating the physical state. Raw residual magnitudes across different measurement dimensions are not comparable accuracy scores.

The experiment does show that the additional geometry exposes failures of the shared-state response/noise model that two summary channels cannot expose. That is useful diagnostic capability, but not yet a validated detector-error classifier.

## 6. Uncertainty and model discrepancy must be separated

The shared-input propagation formulas are implemented correctly in the inspected code. The statistical problem is that their input covariance describes short-timescale effective coordinate noise, not total prediction error.

Saved median held-out squared Mahalanobis scores around 1,531 and 2,521 indicate that the declared covariance does not describe the observed predictive errors. Calibration-coefficient uncertainty and systematic response discrepancy are excluded from the current local covariance and have not been established by the primary experiment.

Second differences remove slow changes. They cannot alone estimate slow pose drift, repeatable field-dependent bias, or uncertainty of a fitted response surface. A synthetic Monte Carlo test under the assumed covariance checks propagation mathematics, not validity of that covariance for the actual recordings.

Do not simply inflate noise until the scores look acceptable. Uniform covariance scaling does not change a no-prior fixed-model framewise minimizer; it does change uncertainty numbers and can change calibration's optical-versus-anchor/prior tradeoff. Those are different experiments.

Add training-only grouped prediction residual analyses, coefficient refit sensitivity, and a separately declared discrepancy model. Plot signed residuals by point, coordinate axis, gaze, demand/capture, and P1 context. Report optical, anchor, and prior objective components separately. Investigate systematic patterns before treating them as random noise.

There is no immediate evidence that replacing triangle-area normalization fixes this problem. The two targeted difficult frames have P1 edge condition numbers about 1.88 and reported area signal-to-noise about 8,600: they are not near-collinear P1 triangles. This observation is local, not a guarantee about every frame.

## 7. The present comparisons do not isolate extra information

The plan calls for comparisons that separate geometry, model capacity, weighting, and priors. The implementation currently compares separate 13-, 27-, and 37-coefficient model families. It uses a common pilot/noise policy for the two coordinate capacities, which is helpful, but it does not implement the planned reduction of the **same** full-model prediction to centroid/area or the optional similarity-restricted comparison.

Add a fixed-response controlled study: freeze one training-only full response, then compare state inversion from its six coordinates versus its derived summaries, with appropriate transformed covariance. This isolates measurement choice from coefficient changes, although it is a state-estimation diagnostic rather than a valid withheld-point baseline if the summaries include that point.

For an untouched P4 prediction, compare retained-coordinate subsets that all exclude the tested P4 point—for example retained x-only versus retained x/y components, only where both subsets identify the two states. Never use an all-three triangle area in a held-out-P4 inverse.

Nested grouped selection, adequate within-support tests, denser evaluation, and matched state-comparison populations remain unfinished. The current code honestly labels its fixed-setting run exploratory. Choosing conditional27 using these outer folds now makes those results development evidence for the choice; do not relabel them an untouched selection test. Keep captures 5/6 for final transfer after the numerical and modeling choices are frozen.

## 8. Optical interpretation and model updates

The theory's central interpretation should remain: reproducible translation and distortion are state signal, and unexplained residuals test the shared-state model. Do not force similarity or add arbitrary framewise affine parameters to eliminate disagreement.

After correcting inversion and characterizing residuals, test targeted missing response terms rather than increasing every polynomial indiscriminately. For example, both current capacities use an accommodation-independent cubic coefficient in Dx; neither includes `t^3 log(1+A)`. In the illustrative field model with an accommodation-dependent cubic distortion coefficient, such a mixed common-translation term can arise. That motivates a bounded ablation, not proof that it is the cause of the real-data errors.

Inspect P1-context dependence and source/eye/camera geometry before adding nuisance states. The exact ability of three pairs to define an affine map does not establish that a low-degree state-only map transfers across different P1 contexts. Capture and demand are confounded in these recordings.

Audit detector selection before calling the measurements independent optical checks. The estimator preserves selected coordinates; it does not independently localize the underlying reflections. If selection already enforces approximate similarity, the available distortions can be biased or truncated. This is an unresolved measurement-pipeline issue, not a demonstrated current detector bug.

## 9. Reporting and application defects/gaps

### Misleading metric names

`validate.py::summarize` computes the fields named `gaze_mean_anchor_discrepancy_deg` and `accommodation_mean_demand_discrepancy_D` from **individual estimated frames**. Their RMS is not RMS of fixation means. `report.py` separately writes fixation means, and RESULTS.md labels its table as fixation-mean RMSE. Do not assume the published table is wrong; the per-frame summary/metrics names are misleading and invite an incorrect comparison.

Rename per-frame statistics explicitly and emit separately computed fixation-mean statistics, including group membership, weighting, counts, and common-support versions.

### Calibration cost selection in reports

`report.py::generate` reports the minimum cost over every calibration start, including failed ones. The applied model is selected from accepted starts when any exist. Report `alternatives[selected_start].cost`, plus a separately labeled best rejected-start cost. This is a definite code-path risk; the audit did not establish that it changes the saved aggregate result table.

### Incomplete model compatibility checks

`schema.py::load_model` checks schema, gaze/state scales, targets, coefficient shape, and convergence, but does not validate all declared normalization, coordinate/coefficient order, bounds, pilot, or covariance semantics. Execution uses hard-coded definitions. Reject incompatible metadata explicitly rather than silently applying the hard-coded convention. Existing files were not shown to be incompatible.

### Application misses diagnostics required for transfer

`apply.py` preserves rows and inverse diagnostics but does not emit the same anchor/context support and uncertainty fields added by `diagnostics.py` to grouped results. The enrichment tool loads the reviewed capture set; it is not a general replacement for application diagnostics on new captures. Share one evaluator so captures 5/6 and future recordings receive the same support, parity, uncertainty, and failure contract.

### Convergence documentation and tests

The calibration callback can certify stable cost/step, but acceptance also permits `result.success` plus stationarity without that certificate. Either make the acceptance contract explicitly allow sufficient stationarity termination or enforce every stated convergence condition; do not claim every accepted fit has a stable-step/cost certificate.

The 14 existing test definitions cover valuable mathematics and clean synthetic recovery. Add high-residual recorded-candidate regressions, rejected-low-cost candidate retention, bounded polishing, end-to-end raw-input holdout noninterference, compatibility rejection, and report-aggregation tests. RESULTS.md still says twelve tests while CURRENT_STATUS.md and the current test definitions say fourteen.

## 10. Recommended next sequence

1. **Numerical reliability:** retain and polish candidate solutions, use branch profiling as an independent check, add high-residual regressions, and reevaluate the existing fitted models without retraining. Keep original results intact and report transitions in coverage, branches, and point errors.
2. **Reporting consistency:** fix metric names, selected-start costs, acceptance status semantics, and application/schema diagnostics.
3. **Error diagnosis:** separate measurement covariance, coefficient uncertainty, and response discrepancy; decompose objective terms and run grouped training-only sensitivity to anchors/priors/reference covariance.
4. **Information ablations:** compare measurement subsets with controlled forward capacity, and test only physically motivated basis additions supported by residual patterns.
5. **Stronger evaluation:** nested grouped selection, denser matched evaluation, detector/context audits, and then untouched capture-5/6 transfer. Independent references are still needed for physiological accuracy claims.

## Improvement assessment

- **Demonstrated locally in this audit:** a lower-cost, properly certified subset inverse and a withheld error reduction from 5.96 to 2.70 pixels in one case; robust certification of eight targeted interior inverses.
- **Implemented diagnostic improvement:** multiple coordinate responses can disagree with the shared state; the baseline's two channels cannot provide that same residual redundancy.
- **Saved computational improvement:** the repository reports a 14.8x CuPy-versus-batched-NumPy speedup for a large repeated-input timing batch. This audit did not benchmark a GPU and this is not an accuracy or end-to-end calibration-speed result.
- **Not established:** superior full-study gaze/accommodation estimation, physiological accuracy, calibrated artifact thresholds, or universal CPU/GPU inverse equivalence.

## Source map

All repository paths below refer to the audited commit.

- `Theory.md`; `ESTIMATOR_PLAN.md`; `CURRENT_STATUS.md`.
- `full_position/model.py`, `geometry.py`, `noise.py`, `calibrate.py`, `invert.py`.
- `full_position/data.py`, `validate.py`, `report.py`, `schema.py`, `apply.py`, `diagnostics.py`, `accelerated.py`.
- `tests/test_full_position.py`.
- `experiments/full_position/grouped_v2/RESULTS.md` and `metrics.csv`.
- `experiments/full_position/grouped_v2/gaze_-10/conditional37/model.json`, `frames.json`, `frame_uncertainty.json`.
- `experiments/full_position/grouped_v2/gaze_-10/conditional27/frames.json`.

SciPy's official `least_squares` documentation distinguishes small-step (`xtol`) termination, method-dependent gradient (`gtol`) termination, and the returned success/status fields. The audit's polynomial-cost and exact-Hessian formulas are derived from the inspected fixed-covariance model, not claims about arbitrary nonlinear optical models.
