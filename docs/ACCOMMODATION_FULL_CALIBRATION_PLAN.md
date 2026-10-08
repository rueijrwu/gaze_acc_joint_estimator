# Accommodation-model discovery: full calibration, identifiability, and cross-agreement

**Status:** REVISED RESEARCH PLAN — proposed next experiments; not a record of completed fits.  
**Branch:** `exp5_full`  
**Scientific question:** What common optical accommodation-response model allows all three P1/P4 pairs to determine the *same meaningful instantaneous accommodation*, separately from horizontal gaze, across the calibrated stimulus range?  
**Implementation foundation:** Keep the existing joint variable-projection calibration, covariance handling, three-way P4 exclusion, certification, and GPU/CPU parity code. This document supersedes this file's earlier “four fixed laws, choose lowest P4 error” research objective. It does **not** alter historical experiment results.

## 1. Change of objective

**Do not optimize for a marginal reduction of P4 RMS alone.** A model may reconstruct P4 better while using a poorly determined or arbitrary accommodation axis. Conversely, very small disagreement in diopters can be caused by state compression or accommodation-bound clipping. The model must explain P4 geometry *and* allow accommodation to be separately and stably inferred.

The target is a **physically interpretable, globally shared response law**: one response law and one global coefficient set per candidate calibration; each valid frame has an independent latent state `x_i=(theta_i,A_i)`. A fixation is a stimulus condition, not a constant state; no framewise ground truth is invented from nominal labels and no temporal smoothness penalty is introduced to force small errors.

Do not confuse:
1. **Coordinate agreement:** can one common state predict all three P4 positions?
2. **Subset state consistency:** do different two-P4 subsets infer compatible same-frame states?
3. **Accommodation identifiability:** do measured P4 changes actually determine accommodation apart from gaze?
4. **Physical calibration:** is the accommodation axis in diopters grounded by evidence outside mere optical self-consistency?
5. **Response-law adequacy:** does one response shape describe all tested accommodation/gaze/capture conditions without special-case failures?

The first three can be investigated with current data; physical accuracy requires stronger external calibration evidence. A low cross-check error is not an independent physiological validation of A.

## 2. Data, baseline, and scientific invariants

Use all 20 reviewed calibration conditions from captures 1–4, including all valid full-period frames when claiming a full-data run; declare any central-80% alternative as a separate population. The retained full-period reference previously used 89,175 valid calibration frames and 100,090 scheduled checks. Confirm current frozen intervals and validity policy rather than silently mixing protocols. Captures 5/6 are not training inputs and are reserved for later transfer investigations; they have no independently recorded per-frame accommodation truth.

For each frame:
- Use matched, reordered P1/P4 pairs and actual image x/y components. `r_j=(P1_j-c1)/ell1`, `v_j=(P4_j-c1)/ell1`, `ell1=sqrt(area(P1))`.
- Shared state `x_i=(theta_x,i,A_i)` predicts all three responses: `vhat_ij=D(x_i)+T(x_i) r_ij`.
- `theta_x` is the only gaze latent state in this model; nominal vertical gaze is 0 degrees by protocol, but image-y is not zero.
- Fit 27 global coefficients unless capacity is explicitly varied as a separate controlled experiment. Reuse the existing optimizer, same starts, covariance and soft mean-anchor policy across candidate laws. Re-estimate coefficients **and** all framewise states independently for every candidate.
- Allow natural changes in gaze and accommodation within fixation. Nominal gaze and dioptric demand are soft **fixation-mean anchors**, not per-frame truth or RMS gates.
- Keep all scheduled identities and three exclusion slots per frame, including failed/invalid inversions. Compare models on identical paired cohorts with every exposure represented or mark comparison incomplete.
- Never interpret a numerical optimizer bound as independent physiological evidence.

Retain the certified full-data log/sqrt/quadratic comparisons and sampled literal-power search as **historical hypotheses and feasibility evidence**, not a resolved law ranking. Previous results are from different populations/settings and must not be compared by subtracting their RMS scores.

## 3. Fundamental challenge: state reparameterization

The observation model is `v=D(theta,A)+T(theta,A)r`. When A is latent and each frame has a free A value, a monotone change of coordinate `A'=h(A)` may be partially or, for sufficiently flexible response families, exactly compensated by modified response coefficients. Thus agreement of optical predictions alone cannot determine the physical accommodation scale or unique functional shape. Even the currently included direct linear-A term may not fully resolve practical parameter confounding; finite fixation-mean anchors and priors can determine a numerical gauge without constituting independent physiological ground truth.

**Required output is not just the exponent n.** Quantify:
- How different inferred A trajectories are under different laws on the same frames; possible monotone warping and effective gain.
- Sensitivity of law ordering and inferred A to reasonable, identically applied anchor-strength, coefficient-prior and accommodation-bound choices.
- Whether a law's improvement is optical prediction improvement or mainly a change of latent-A parameterization.
- Profile objective/cross-agreement versus shared exponent, including flat minima, boundary choices and competing minima. Do not label an optimal physical exponent if the profile is broad or gauge-dependent.

A nominal stimulus demand is not actual physiological accommodation. Claims of calibrated diopters ultimately need an external known/independently measured accommodation condition or additional justified optical/biomechanical constraints.

## 4. Candidate response models: compare scientific hypotheses, not just powers

Keep the same six-coordinate conditional model and common measurement channels for first-round response-family comparisons. Each candidate has one **global**, shared shape definition across all frames, captures and response components; never a free per-frame exponent.

### 4.1 Required starting families

- **Log control:** `log(1+a)` with `a=A/(1 D)`, freshly calibrated.
- **Shifted Box–Cox:** `phi_n(a)=expm1(n log1p(a))/n`, continuously approaching log as `n -> 0`; `n=1` is linear and n=0.5 and 2 are already implemented. This is a coherent continuous log-to-linear family.
- **Literal powers:** `a^n` for positive A. This is *not* the same family as shifted Box–Cox and is not logarithmic at n=0. Preserve explicitly positive minimum A and report sensitivity to that bound; derivatives can be singular or small near zero.
- **Simple low-complexity saturating/physically plausible response**, if warranted by data: e.g. `a/(1+k a)` or a monotone exponential saturation with **one shared global shape parameter**. Compare only after derivative, conditioning and gauge checks; do not assume accommodation must saturate.
- **Linear and minimal-curvature controls:** establish whether nonlinear curvature actually contributes an identifiable optical response.

Do not automatically prefer literal `A^3`: it won a sampled P4 cross-prediction screen at the search boundary, while subset state disagreement worsened. Expand or refine exponents only after checking profile shape and state observability. Avoid increased per-component freedoms until a simpler law demonstrably cannot capture reproducible optical structure.

### 4.2 How to estimate a shared exponent

At each proposed n, run the **same full joint-fitting algorithm** for a newly instantiated response law, re-estimating `beta_n` and all frame states. The shared n is optimized by an **outer scalar search over fully reoptimized inner calibrations**, or an equivalent carefully certified joint global optimization. Never estimate n separately for each frame or each P4 exclusion. In cross-checks, freeze n and the fitted beta.

Use a bounded pilot search across model families to locate promising regions, but evaluate the actual finalists with **fresh full-period all-condition calibrations** and identical cross-check schedules. A pilot is not a full-data conclusion. Search resolution should follow observed loss/profile curvature, not an arbitrary fixed grid. Preserve all tried values and unsuccessful fits; do not report the best grid boundary as a proved optimum.

## 5. Metrics: what matters scientifically

### 5.1 Primary optical cross-prediction

With coefficients fixed after full calibration, temporarily omit one P4 from state inversion, predict its full 2D location and rotate across all three. For `e_ij=P4_ij - P4hat_ij`, report

`E_i^2 = (||e_i1||^2+||e_i2||^2+||e_i3||^2)/3`.

Aggregate squared loss within each exposure then equally across the 20 exposures; display square-root RMS in pixels, signed x/y bias, each P4 error, tails and worst-point loss. **Never rank based on incompatible frame memberships.** This is **internal** prediction because calibration coefficients were fit using the full dataset, including the point later omitted from a framewise inversion.

### 5.2 Subset-state agreement in the same frame

Let three inversions produce `x_i,-1`, `x_i,-2`, `x_i,-3`. Report `G_theta` (degrees) and `G_A` (diopters), each the RMS of the three unordered pairwise state differences. These are **not** errors versus demand or previous frames.

Raw `G_A` alone must **not** rank laws: a warped/compressed A coordinate or common clipping can artificially reduce state differences. Report absolute/relative sensitivity and inferred-A ranges alongside agreement. Treat disagreement as a cross-check *diagnostic*, not a hard RMS cutoff or an uncalibrated mixed-unit penalty.

### 5.3 Accommodation observability after removing gaze cross-talk

For local response Jacobian `J=[j_theta,j_A]` and declared retained-channel covariance R, use `F=J^T R^-1 J`. Its gaze-conditioned accommodation information is the Schur complement:

`I_A|theta = F_AA - F_A,theta^2/F_theta,theta`.

Evaluate as a distribution by frame, A, gaze, capture and retained two-P4 subset. Show weak-information regions, near-collinearity between gaze and A derivatives, local uncertainty scales, rank/condition indicators and alternative inverse branches. A law cannot count as a well-determined accommodation model simply because it reports very small raw `G_A` while `I_A|theta` is near zero.

**Gauge caution:** the numerical size of `I_A|theta` per diopter also changes under A reparameterization. Compare it under the same justified physical A convention, show coordinate-invariant image-space derivatives and state-warp sensitivity, and do not optimize derivative magnitude alone. A model could artificially boost sensitivity by rescaling A.

### 5.4 Uncertainty-aware cross-prediction

As a secondary candidate score, whiten held-P4 residuals with the **full predictive covariance** including retained-point inversion propagation and shared P1/P4 correlations. The current `noise.predictive_covariance` provides a first-order form. Start with one frozen, common measured-noise policy; document incomplete model-discrepancy and coefficient-uncertainty treatment.

A candidate Gaussian scoring rule is `0.5*(e^T V^-1 e + log det V)` per 2D omitted point, with consistent coordinate units and reference constants. Do **not** omit `log det V` if V varies by candidate: otherwise high declared uncertainty can falsely improve normalized residuals. Check covariance conditioning and calibration first; this model-derived pseudo-likelihood is not independent holdout likelihood because omitted measurements already informed coefficients. Shared exclusions are correlated and cannot be counted as independent statistical replicates.

### 5.5 Reproducibility and optical-response structure

A good law should show interpretable `D(theta,A)` and `T(theta,A)` responses, stable across captures and gaze targets. Plot the accommodation response of all six coordinate components as a function of A at fixed gaze and matched P1 context; inspect first derivative, curvature and plausible branch behavior. Compare **predicted image-space curves**, not only coefficient tables, because coefficients and latent state scales can change.

Check whether consistent residual patterns recur by P4 point, gaze, A range or capture. A lower global RMS with a severe systematic error in one condition should not be called universally better. Report concentration of gains and leave-one-exposure **influence on the score** without refitting or pretending that exposure was excluded from calibration.

## 6. Model-selection objective: replace pixel-only winner with a transparent evidence hierarchy

There is no physically justified single scalar sum of pixels squared, degrees squared and diopters squared. Instead use this hierarchical policy:

**A. Scientific validity and numerical eligibility (hard contracts, not error thresholds).** Complete source/fit provenance, certified optimizer, valid geometry, identically scheduled exposures, finite/correctly computed scores, branch/bound accounting and identical comparable cohorts.

**B. Accommodation identifiability (essential scientific condition).** Evaluate `I_A|theta`, local ambiguity, response degeneracy, clipping, gaze cross-talk and accommodation-scale stability. If A is weakly identifiable or merely rescaled, label the model **unresolved for accommodation**, even when it wins P4 error. Do not invent an absolute information cutoff from the current data; use comparative distributions, sensitivity tests and predeclared numerical-rank guards.

**C. Optical agreement (primary comparative performance score among defensible models).** Compare paired equal-exposure `E_cross^2` and uncertainty-weighted cross-prediction on exact common frames. Report signed axes and tail/exposure structure. Improvements must be judged by magnitude and concentration, not merely numerical sign.

**D. Same-frame latent-state consistency (diagnostic, not a coordinate-scale race).** Report raw `G_theta,G_A` and uncertainty-normalized *joint* gaze/A subset differences when full shared cross-covariances are obtainable. Never assume overlapping subset estimates are independent. Inspect whether lower `G_A` is caused by low sensitivity, A scale changes or bounds.

**E. Model simplicity, robustness and scientific plausibility.** Favor simpler laws when evidence of optical and identification improvement is inconclusive. Probe common anchor/bound/prior/noise sensitivities without individually retuning the winning law. Compare response curves and their derivatives at matched physical states.

The reporting decision must be one of **well-supported candidate**, **tradeoff**, **identifiability unresolved**, **numerically unavailable** or **evidence insufficient**. A single lowest-P4-loss key is a descriptive ranking, **not** an overall “best accommodation model” judgment. No deployment promotion is implied.

## 7. Work program, preserving the existing algorithms

### MD0 — Reinterpret the retained results (no rerun required)

Read `docs/CURRENT_STATUS.md`, retained full three-law report and coarse literal-power frame records. Rebuild a single *separate* evidence ledger with study population, bounds, model basis, candidate, fitted-state distributions, certification, common-cohort coverage, E/G and numerical artifacts. Do not directly compare scores across unmatched full/coarse populations. Inventory which saved frame-level records survived cleanup; never reconstruct missing raw errors from summary RMS.

Identify where `A^3` gains occur, and whether subset-state discrepancies/low accommodation information/ambiguous branches coincide with those gains. Do not claim that clipping **causes** any observed gain or deterioration without a paired controlled comparison.

### MD1 — Instrument observable accommodation information (minimal code extension)

Add **diagnostic reporting**, not a new optimizer: per-frame/per-subset 2-column Jacobians, conditioned information, rank and ambiguity flags, bound classification, sensitivity to coordinate reparameterization, and full 2D P4 cross residuals. Validate derivatives against finite differences and frozen CPU/GPU fixtures. Reuse current covariance and inverse results; do not recalculate numerical state estimates under a different objective merely to generate a metric.

Compute raw and fixed-covariance whitened cross-prediction on the same schedule. If uncertainty-calibrated predictive scoring is desired, first audit `predictive_covariance` and its empirical coverage instead of assuming the measured-noise covariance fully describes model discrepancy.

### MD2 — Candidate family and parameterization study

Compare log, shifted Box–Cox log-to-linear shape, literal positive-A powers and the linear control first. Add a small saturating family only if the existing shapes leave reproducible systematic errors and it is identifiable. Select candidate trial parameters globally by fresh joint calibration, not framewise exponent fitting. Keep the response coefficient count and geometry controlled; record parameter-dependent prior scaling.

Screen broad shape regions on a declared balanced-development population for economy; inspect agreement + identifiability + gauge dependence, not E alone. No automatic confirmation of an extreme exponent just because E continues decreasing there.

### MD3 — Fresh full-data calibration of finalists

Re-use `calibrate.fit` or its numerically checked GPU equivalent and existing inverse/crosscheck. Fit each finalist afresh on the **same full calibration population**, all three P4 x/y per frame, with each frame's own state. No whole-condition held-out calibration is required for this internal-consistency task. Freeze global law/coefficients then apply three-way P4 exclusions to all scheduled frames.

Store full calibration model/certification, state-response curves, data/solver hashes, all exposure/point availability, ambiguity/bounds and metric records. Use one common manifest for paired model comparisons. If a law fails certification, retain its failure without excluding it from denominator accounting or substituting an old fit.

### MD4 — Compare models as accommodation models, not coordinate regressions

On one exact common complete-frame population, present:
1. Optical cross-prediction loss (px² and px RMS), signed axes, tails and per-exposure influence;
2. Gaze and accommodation subset agreement, bound-conditioned and interior subsets;
3. Conditional accommodation information, gaze/A cross-talk, ambiguity and inferred-state sensitivity;
4. Physical-response curves, diopter-scale/gauge sensitivity under common anchor/prior/bounds choices;
5. Model complexity and qualitative failure modes.

Display a Pareto/tradeoff assessment; do **not** collapse all metrics into an arbitrary uncalibrated number. If two laws are optically equivalent under an accommodation warp, report that the current measurements do not identify the response law, regardless of tiny fit-cost differences.

### MD5 — What additional evidence would identify physical accommodation

If the existing dataset cannot fix the accommodation axis, plan a targeted measurement protocol rather than adding more powers: independently measured accommodation (e.g. objective refractive response), well-controlled accommodative demands under repeat captures, known focus perturbations, or an independently justified optical model. Such data can anchor A in diopters and discriminate genuine power-law curvature from a latent reparameterization. This stage is a proposal; the current captures have no independent accommodation ground truth.

## 8. Acceptance/reporting and reproducibility

- No physical error/RMS threshold, fixation-flatness gate, or selected numeric exponent is predefined.
- Preserve original frame IDs, per-exposure expected denominators, all numerical failures, soft anchors, and frozen data hashes.
- Do not change response family, measurement mask, covariance, coefficient capacity and optimizer simultaneously; isolate ablations.
- Audit any post-hoc choice on the same records used to select it; present results as developmental internal consistency.
- Verify both the *mathematics* (analytic/numeric derivatives, propagated covariance, gauge tests) and *implementation* (existing solver parity, subset withholding, score aggregation and row identities).
- Historical `docs/LITERAL_POWER_PLAN.md` and previous results remain evidence of earlier experiments, not the scientific decision policy for this revised workstream.

**Deliverable:** a documented accommodation response family (or an explicit *not identifiable from these data* conclusion), supported by full-calibration optical agreement, stable and interpretable accommodation inference, uncertainty/rank diagnostics, and clear physical limitations. The goal is not winning an extra fraction of a pixel.
