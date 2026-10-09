# Implementation plan: distortion-based gaze and accommodation

**Branch:** `exp5_distortion_model`.  
**Status:** Implementation specification, not execution results.  
**Date:** 2026-10-09.  
**Starting commit:** `bc3e75b595bcf28486d23881354b3fa3c1daacde`.  
**Authoritative mathematics:** [Theory.md](Theory.md). Read both documents before implementation.

## 1. Deliverable and branch boundary

Build a compact optical estimator that (1) fits a scalar magnification from reference P1 geometry, (2) estimates horizontal gaze principally from relative P4–P1 displacement, (3) estimates accommodation from P4 magnification/radial deformation after gaze correction, and (4) iterates both estimates to one shared solution. Calibrate each candidate **afresh on all reviewed conditions** and compare internal three-pair cross-agreement, optical identifiability, and response stability.

This branch currently contains `data/`, `docs/`, a README and ignore rules. It does **not** contain `full_position/`, `lib/`, runnable fitting code, test fixtures, or retained experiment artifacts. Inherited status/experiment documents describe another branch/history; do not treat their paths, tests or completion statements as live here. These two specifications take precedence for this workstream. Do not restore obsolete experiment trees or operate on `exp5_full` to implement this branch.

Reuse established bounded least-squares, variable-projection, multistart and GPU batching methods. Reuse reviewed implementation pieces from a **pinned donor commit** only when their contracts still apply. A possible donor is `54deab5871b4713f67e2ab6ab86f1f950dc95eee` on the historical line: its `full_position/gpu_profile.py` explicitly constructs the old six-coordinate, 27-column response design [I1]. It is not a drop-in engine for the new radial/projective model. Port small generic helpers/tests with provenance and required licenses; do not copy the old coefficient basis, area normalization, monkey-patch integration, or model-selection pipeline wholesale.

CuPy/PyTorch availability is user-provided. It is not a claim that this documentation session ran a GPU. The implementer must inspect installed versions/devices and record them before choosing backend details.

## 2. Locked initial design

| Item | Required first implementation |
|---|---|
| State | One `(theta_x_i,A_i)` per valid frame; no vertical gaze |
| Shared parameters | Reference geometry, optical alignment, allowed keystone parameters, relative centroid law, accommodation response |
| Observation | Ten linear relative coordinates from all three P1/P4 pairs, as in Theory §3 |
| Scale policy | `p1_profile_v1`: trial-gaze-conditioned P1-edge GLS scale |
| Triangle areas | Optional legacy diagnostics only; not residuals, normalizers or automatic gates |
| Optical core | Explicit radial baseline and rational keystone for each reflection |
| Nuisance approximation | Same fractional P1/P4 scale; `eta=1`; no free Z or P4 scale |
| Translation | Removed with same-frame differences; relative centroid signal retained |
| Initial geometry route | `empirical_incremental` unless a matched paraxial/center calibration is supplied |
| Candidates | DM0 effective scale; DM1 scale plus linear radial increment; DM2 only after a supported structural need |
| Fitting cost | Theory §8; raw relative-pixel weighted residual, finite mean anchors, shared regularization |
| Calibration population | All valid frames in all 20 full reviewed intervals, captures 1–4 |
| Cross-check population | Full predeclared interval schedule, three P4 omissions per frame; failures retained |
| Calibration truth | Nominal labels are soft group means, not instantaneous state truth |
| First numerical bounds | `theta=[-20,20] degree`, `A=[0,6] D`; declared numerical exploration, not validated optical coverage |
| Comparison | Exact paired/common frame populations; E, G_theta, G_A and identifiability separately |
| Prohibited success criteria | Hard physical RMS ceiling, flat fixation, or lowest training cost alone |

Candidate models need **not** have 27 coefficients. Their shared parameter count follows the optical mechanisms. Store every free/fixed parameter and gauge. Do not add redundant coefficients merely to equalize the legacy count.

## 3. Minimal code layout and APIs

Create a small `distortion_model/` package, not another general framework:

```text
distortion_model/
  __init__.py, __main__.py
  data.py          # trusted payloads, intervals, immutable schedule, masks
  geometry.py      # correspondence, relative map, P1 scale and covariance
  optics.py        # keystone, baseline modes, full joint prediction
  objective.py     # one declared cost, means, scale policy, derivatives
  calibrate.py     # staged initialization and shared/frame alternating solve
  invert.py        # fixed-calibration batched two-state inference
  crosscheck.py    # three omissions, exact population joins and metrics
  io.py           # versioned artifacts, hashes, checkpoint validation
  report.py       # paired summaries and scientific diagnostics
```

Add `tests/`, a small pinned dependency specification, and configuration examples during implementation. Add a separate CuPy/accelerated helper only if measured performance warrants it. Do not maintain two unrelated forward equations.

Required semantic interfaces (names can be implemented directly):

- `relative_observations(p1,p4)` returns y, source identities and the linear native-to-relative map L.
- `p1_reference(theta,params)` returns reference edges and centered P1 predictions.
- `p1_scale(theta,observed_edges,cov11,params)` returns scale, validity and derivative-compatible values.
- `predict_relative(theta,A,scale,params)` returns ten coordinates with full centering and keystone composition.
- `residual(theta,A,observations,params,mask)` implements **the same scale and covariance policy** in calibration/application/cross-check.
- `fit_full(config)` creates fresh parameters/states, never loads a historical fit under another model definition.
- `infer_retained(calibration,frame,held_point)` receives a masked view excluding the tested P4.
- `summarize(records,manifest)` cannot construct denominators from surviving records.

Serialized model fields must include schema version, branch/commit and source hashes, units, parameter order, baseline mode, correspondence, reference length/center conventions, optical zero, P1 scale policy, state bounds, covariance policy and candidate response functions. Mismatched schemas fail explicitly.

## 4. Stage D0 — Inputs, references and identifiable geometry

### 4.1 Population and data integrity

Read `data/fixations/fixation_intervals.json` and the trusted capture 1–4 pickles. Verify payload hashes and array schema before loading. Preserve native row, source-frame index, timestamps, point flags, exposure/capture/demand labels and correspondence. The inherited permutation is `[2,1,0]`, but read and validate each payload rather than silently overwriting metadata.

Each capture must contain the five labels -10,-5,0,5,10 degrees. Demands come from metadata, including the lowest approximately 0.36036 D. Use `[start_row,end_row_exclusive)` without automatically restoring central-80% trimming. Do not reject a row for target deviation, large residual, missing pupil, or nonsimilar triangles. Retain any newly required scale/domain validity reason separately from the original detection flags. A noncollapsed collinear P1 can estimate scale, but may still be a bad distortion-identification geometry.

Declare all interval rows and exactly three held-P4 slots per frame **before** predictions. Full calibration uses rows with all required coordinates/flags; the schedule also preserves invalid rows as explicit unavailable results. Do not fabricate missing reflections. Captures 5/6 must not enter initialization, noise fitting, parameter selection or calibration.

Audit timestamp reliability; where backward jumps are recorded, use original row/frame order for membership and do not infer trustworthy time intervals or motion derivatives from those timestamps. Full-period input counts are recomputed, not hard-coded from another branch's 89,175/100,090 historical totals.

### 4.2 Reference initialization

Use multiple nominal-zero-gaze samples and the low-demand capture for the first reference. Remove common translations and align by **scalar P1 scale only**; preserve source identity and camera orientation. Fit the coarse displacement-to-gaze initialization to fixation means, then create distinct framewise starting states.

Initialize separate local P1/P4 origins from reference centroids. Treat them as assumed local origins, not measured radial centers. Record a fixed reference length. Freeze reference templates, local-center offsets, optical zero and the P1 isotropic gaze-scale convention in the first fit. P1 symmetry is a secondary alignment diagnostic, not a new framewise target.

Obtain the initial P1 rotation reference from matched optical evidence or a declared empirical reference fit. Without known stable axial geometry, that fit cannot independently determine pure gaze magnification and framewise axial magnification; fix its gauge and label the resulting g **effective reference scale**. Do not absorb P1's isotropic gaze scaling differently from P4's response without transforming the whole model consistently.

Initialize P4 in `empirical_incremental` mode: `M(A_ref)=1`, `delta_kappa(A_ref)=0`, with its actual measured distorted template. This allows progress without inventing a paraxial grid. The absolute-physical-kappa route requires additional matched template/center information and is a distinct artifact mode.

### 4.3 Geometry rank check

Inspect radii about the declared P4 local origin and the centered scale/radial response columns after allowing centroid motion. Test exactly equal-radius and nearly equal-radius configurations synthetically. If separation is rank-deficient, run DM0 and label the radial coefficient unidentifiable; do not initialize DM1 with an unconstrained center to conceal the problem. Retain conclusions for the actual three source fields, not an assumed equilateral or nine-point pattern.

**D0 output:** a frozen data/reference manifest, validity census, source/template units, free/fixed parameter list and geometry-identifiability report. This is setup for fitting, not a scored accommodation result.

## 5. Stage D1 — One correct forward model and cost

Implement the complete expanded transforms in Theory §4, including the keystone matrix and its shared denominator. Predict P1 edges and P4 relative coordinates together. Keep external g **outside** keystone. Use model-predicted means to express shapes in the P1-centroid reference; never substitute the measured P4 mean into a prediction.

First parameterization:

- P1 empirical baseline with a fixed/gauge-constrained gaze-reference transformation; retain keystone.
- P4 M(A)=1+m1*a; DM1 additionally `delta_kappa=k1*a` in the incremental route.
- P4 `sx=1+alpha*t^2`, `sy=1+beta*t^2`, `q=gamma*t/L_ref` in explicitly scaled axes, equivalent to the physical-degree equations.
- Relative centroid `h=b0+bA*a+(s0+sA*a)*t` in both image axes.
- `eta=1`; no new physical Z state, A-dependent axial slope, A-dependent keystone, free affine transform or per-frame center.

Do not set weak rotation distortion to zero without a same-domain ablation. Reject invalid denominator/radial-domain proposals via the optimizer's domain handling; do not replace them with clipped coordinates with false gradients.

Build `R=L Sigma L^T` from a fixed common native-coordinate noise policy. Use Cholesky solves, not explicit matrix inverses. A second-difference estimate from contiguous full-resolution records is an effective-noise assumption, not pure localization truth; never bridge gaps or remove real fixation variation as though it were detection noise. Check covariance stability and record floors/shrinkage in the manifest. No candidate-specific inflation is allowed in the initial comparison.

Implement `p1_profile_v1` exactly as specified, differentiating through its scale numerator/denominator, predicted means and optical transforms. Use the raw relative residual cost. Do not silently use the full-data `g_joint` formula or detach g in the gaze/global derivatives. Do not claim the plug-in residual covariance equals R; uncertainty diagnostics need the induced correlations. The first implementation does not optimize a Gaussian predictive likelihood or append raw G penalties.

Global regularization uses declared dimensionless parameter scales. Fix essential gauges instead of using an opaque huge prior. Optional departures shrink toward their simpler nested model. Choose one common regularization policy before comparing candidates; report sensitivity rather than tuning each law to its best score. A reasonable inherited exploratory strength is 0.001 after the parameter scales are explicitly defined; it is not comparable numerically to old column-normalized priors by itself.

**D1 exit:** analytic/autodiff/finite-difference and invariance tests pass. A tested forward model is not a completed calibration.

## 6. Stage D2 — Full calibration with alternating corrections

Each candidate receives newly estimated global parameters and one free `(theta,A)` state per calibration frame. Reuse the established least-squares strategy, but do not reuse old fitted coefficients/trajectories or relabel a 27-column model as optical calibration.

Initialization sequence:

1. Low-demand fixation means initialize a linear relative-centroid gaze mapping after provisional P1 scale correction.
2. Near-zero-gaze samples across demands initialize M and any identifiable radial increment with soft accommodation means.
3. Revisit the low-demand frames with estimated A, correct the gaze mapping, then extend joint updates to **all conditions**, not only the reference capture.

Optimization cycle:

```text
instantiate fresh model + framewise initial states
repeat:
    evaluate trial-gaze reference P1 and g_P1 for all frames
    propose conditional accommodation updates through full radial/keystone model
    propose gaze updates with accommodation-corrected centroid gain/offset
    reevaluate g_P1 at every trial gaze
    accept/damp against the SAME full objective
    update shared optical parameters using all calibration frames
    update linear centroid coefficients by exact weighted solve when applicable
    evaluate true full-data objective and projected stationarity
freeze the certified global model
```

Use inherited mean-anchor scales 0.10 degree and 0.25 D as finite starting weights. Compute anchors from **full group means**, not a minibatch mean and not a separate framewise target penalty. Zero temporal regularization. If conditional steps stall, use joint two-state damped Gauss–Newton/trust-region refinement. Linear centroid blocks can be eliminated or alternately solved exactly. When differentiating a fully profiled residual, include the profile derivative; an envelope-only objective gradient is not a full residual Jacobian.

Use at least the nominal initialized and a reproducible perturbed-state/global start; preserve competitive solutions. More starts may be required when branches disagree. Certify finite objective, domain, scaled projected gradient and acceptable local curvature. A max-iteration stop is an uncertified checkpoint, not a failed scientific law and not a converged fit. Require full-gradient certification after block convergence, including mean-anchor coupling.

State bounds are numerical exploration limits. No law fails merely because G, E, nominal RMS or within-fixation variation is large. Bound/rank/ambiguity behavior is reported, not used to hide difficult states.

**D2 exit:** independently certified full-calibration artifacts or explicit unsuccessful outcomes. No condition-held-out study or nested model selection is required for this deliverable.

## 7. GPU optimization using PyTorch/CuPy

### 7.1 Default backend and reference

Use **PyTorch float64** for the initial new composed model and derivatives. `torch.func` supplies vmap, jacfwd/jacrev and JVP/VJP transforms; use batched local state derivatives and matrix-free products rather than a Jacobian against every frame state [T1,T2]. Keep a NumPy/SciPy float64 reference for small deterministic tests and independent endpoint checks [S1]. Use analytic formulas where simpler (P1 scale, keystone and radial derivatives), verified against autodiff.

CuPy float64 is an optional production backend for validated array kernels, profile QR/LSMR and batched solves. If a trustworthy donor engine is ported, compare the new optical adapter with the PyTorch/CPU reference first. Do not make a second backend a prerequisite to the first scientifically interpretable full run. Do not run two inconsistent objectives under the same model name.

Record installed Torch/CuPy/CUDA/driver versions, GPU names, FP64 behavior, memory, and available device IDs. No device count, GPU speedup, or memory capacity is assumed from previous runs. Disable automatic mixed precision and TF32 for verified numerical operations; changes of precision are separate tested settings.

### 7.2 Memory-bounded exact full-data objective

Store raw/relative coordinates, covariance factors, state arrays and reference constants on device when feasible. Chunk evaluation if needed; chunks are an execution detail, **not training subsampling**. Do not build a dense `(10N) x (2N+P)` Jacobian or a `(2N)^2` Hessian.

For state steps, compute small per-frame 10x2 Jacobians and damped 2x2 systems in batches, accounting for the low-rank group-mean coupling or using an approximate step accepted against the exact objective. For global steps, accumulate scaled small systems or use matrix-free JVP/VJP. QR/SVD or regularized solves with conditioning checks are preferable to blindly trusting ill-conditioned normal equations.

Group means require exact reductions over all contributing frames. A safe two-pass implementation first computes N_k and full sums, then accumulates per-chunk optical loss/gradients and the correct full-mean anchor gradient. For one component, the anchor derivative is `(mean_k-target_k)/(K*N_k*s^2)`. Never normalize independently by the number of valid frames in a compute chunk. Recompute full means for accepted joint trial states.

Keep gradients through scale and global coefficients; do not detach scale just to reduce memory. Avoid Python loops, `.item()` and host transfers per point/frame. Scalar host decisions once per optimizer iteration are acceptable. Keep line-search trial residuals immutable with respect to previously stored Jacobian linearizations.

### 7.3 Inversion batching and multiple GPUs

Batch independent frames, three omission masks and the declared multistart grid. Start with a common 7x7 state grid (49 starts), plus any separately declared retained-only centroid proposal; clip/deduplicate starts only under a recorded bounds policy. All candidates use the same schedule. Retain unique competitive branches, not only the first converged result.

Use one process/device first. If multiple GPUs are available, distribute independent candidates or complete inference batches. A distributed single calibration must still reduce one common objective, anchor means and global gradient; independently fitting each shard is not a full joint calibration. Do not share mutable GPU tensors between unsynchronized workers.

Measure GPU time with CUDA events or `cupyx.profiler.benchmark`, with warm-up and synchronization; plain host timing does not measure asynchronous kernel execution reliably [C1]. Record transfer, setup, fit, check and reporting times separately, peak device/host memory and batch sizes. No runtime promise is inherited from old small-grid studies.

### 7.4 What constitutes GPU correctness

Compare CPU, PyTorch device and any enabled CuPy path on predictions, objective components, scale derivatives, state/global JVP/VJP, accepted update behavior and final branches. Use scale-aware numerical tolerances chosen before experiment comparison; do not tune them against whichever accommodation law wins. Bitwise identity is not required, but differences large enough to change validity/branch decisions must be explained. Autodiff correctness alone does not validate the optical model.

## 8. Stage D3 — Internal three-pair cross-agreement

After certification, freeze each model's global parameters, centers, reference templates, covariance policy and shape law. For every scheduled frame, exclude P4_j in turn. Use all P1, the two retained P4, the retained covariance marginal, and only retained-derived starts. Reestimate theta and A through the complete iterative scale correction. No calibration labels or temporal priors are used to infer the tested frame.

Never use the all-three P4 centroid, area, unmasked state, observed affine map or warm start from the all-three fit. A predicted mean over the three model points is permitted. Calibration itself used all observations, so this is internal agreement even though the per-frame inverse omits one measurement.

A held-point failure record always includes frame/slot identity and a reason, including calibration failure. Store `held_point=0/1/2` for every scheduled slot. Scoring must not crash just because a candidate never certified. Input-invalid, not-run, unconverged, rank-deficient, ambiguous and bounded outcomes remain distinct.

Score omitted-point error in original relative camera pixels, not divided by that candidate's scale. Compute E, G_theta, G_A and worst-point measures only on complete eligible triples; also retain partial-slot scores with explicit denominators. Do not assign zero disagreement to missing states.

## 9. Stage D4 — Decide whether the accommodation model is useful

DM0 and DM1 share geometry, bounds, scale/noise/anchor policies and initialization protocol. Fresh full fits are required for both. If DM1 has an identifiable radial response, compare image-space response curves, cross-agreement and accommodation–gaze cross-talk. Add one curvature/coupling/spatial term only when it addresses a reproducible residual mechanism; an extreme exponent is not a generic cure for missing spatial deformation.

Required report views:

1. Individual full-population scorecards with all scheduled counts and failure reasons.
2. Each candidate minus DM0 on exact shared complete frames and retained exposures, with per-exposure squared-error contributions.
3. A common all-candidate cohort for ordering, only when all expected exposures contribute. Report any lost exposures and do not let them disappear from the denominator.
4. Signed-gaze/capture/demand/point/axis structure, tails and bound transitions. Restricted cohorts supplement, not replace, the declared full population; do not compare their independently reweighted RMS values as additive contributions.
5. Optical M/radial/keystone responses, conditioned accommodation information, branch distributions and gauge/center sensitivity.

Use **cross-agreement as the main evidence**, with E as the optical reconstruction measure and G_theta/G_A as simultaneous-state compatibility measures. Raw G_A may shrink under an accommodation warp or common clipping; large derivative per diopter can also be manufactured by rescaling A. No mixed-unit scalar sum or arbitrary physical threshold defines a winner.

Outcomes are `supported_effective_model`, `supported_distortion_increment`, `tradeoff`, `identifiability_unresolved`, or `numerically_unavailable`. Report the simplest supported interpretation. No independent physiological accuracy or deployment promotion is implied. If separate magnification/radial coefficients remain inseparable, an effective accommodation response is still a useful deliverable; say what additional matched geometry/reference would be needed.

Do not pursue covariance-likelihood scoring, broad exponent sweeps, extra nuisance states or held-condition experiments before producing the small full-calibration comparison. Later sensitivities are named experiments, not indefinite prerequisites or post-hoc rescues.

## 10. Mandatory tests and acceptance contracts

| Test | Required result |
|---|---|
| Arbitrary per-frame shared translation | All relative inputs, inferred states and cross-errors unchanged |
| Common positive image scaling | P1 scale changes accordingly; corrected geometric prediction preserves accommodation signal |
| P4-only magnification | Not canceled by P1 scale; detected as differential geometry, not automatically identified physiology |
| Keystone expansion/order | Matrix and Cartesian forms agree; moving g inside denominator produces a detectable difference |
| Reference identity | Incremental radial map is identity at A_ref; baseline distortion is not applied twice |
| P1 edge reparameterization | GLS scale unchanged with transformed full covariance |
| Scale derivative | Analytic/autodiff/finite-difference agreement including trial-gaze/global dependence |
| Relative rank and covariance | L has rank 10; induced P1-profile scale constraint is recognized; redundant normalized components not independently weighted |
| Equal-radius source geometry | Magnification/radial degeneracy recognized rather than hidden by regularization |
| Dynamic fixation | Zero-mean trajectory changes do not change mean-anchor cost; no temporal flattening |
| Calibration/full-gradient | Block solution checked against the full objective including mean anchors |
| Withheld-point noninterference | Changing only omitted P4 after freezing calibration leaves its entire subset inference unchanged |
| Population integrity | Missing/duplicate frames or slots cannot reduce scheduled denominators or silently lose an exposure |
| GPU and chunking parity | Same objective/gradients/results within declared numerical tolerance, independent of chunk partition |
| Fresh candidate fits | Different baseline laws construct/recalibrate their own parameters and states |
| Artifact round trip | Wrong units, template mode, scale policy or parameter order fails loudly |

Add counterexamples for near-collapsed P1, projective pole, radial folding, wrong source permutation, active bounds, multiple branches, failed calibration and corrupted resume hashes. A synthetic exactly specified model should be recovered up to declared gauges; include noisy and deliberately misspecified synthetic data so false confidence is visible.

At completion, report which tests actually ran, which backend they exercised and any skipped GPU checks. Syntax checks or mocked orchestration tests are not a substitute for a real-data fit. Numerical tolerances protect implementation correctness, not physiological RMS requirements.

## 11. Artifacts, proposed CLI and reproduction

Create fresh run directories such as `experiments/distortion_model/dm_v1/`. Proposed commands below are **interfaces to implement**, not commands available in the current clean branch:

```bash
python -m distortion_model prepare --config configs/dm_v1.json
python -m distortion_model calibrate --config configs/dm_v1.json --device cuda:0 --dtype float64
python -m distortion_model crosscheck --run experiments/distortion_model/dm_v1 --device cuda:0
python -m distortion_model report --run experiments/distortion_model/dm_v1
```

Configuration must explicitly name candidates, baseline/center/gauge mode, P1 scale policy, full calibration intervals, agreement schedule, covariance, regularization, bounds, multistart, precision and device/batch settings. Default is full-period calibration and all scheduled agreement frames. A sampled engineering rehearsal must be labeled as such and cannot be reported as full calibration.

Store compact reproducibility evidence:

- `config.json`, source/data/reference hashes, immutable frame/slot manifest and validity census;
- per-candidate calibrated model or failed checkpoint, full frame-state arrays, global-parameter trace and numerical certificate;
- cross-check state/error/flag arrays, shared membership lists and exposure contributions;
- `RESULTS.md`, `summary.json`, `verification.json`, and a truthful `completion.json`;
- environment/device/precision data, timing and memory observations.

Use compressed numerical arrays and bounded debug archives rather than full per-iteration/per-start dumps for every frame. Preserve data needed to recompute metrics; do not repeat the prior cleanup loss by keeping only an aggregate RMS. Never overwrite source pickles or historical results. Resume only after matching code, objective, population, templates and parameter hashes; otherwise start a new run. Stopped or failed tasks must not be labeled complete.

Implementation commits should follow D0 data/contracts, D1 optical model/tests, D2 calibration/backend, D3 cross-checks, and D4 reports. Update branch status only from executed evidence. This documentation commit neither launches fitting nor modifies frozen observations.

## References

- [Theory](Theory.md): equations, evidence distinctions, relative model and statistical limitations.
- [G1: Fixation metadata](../data/fixations/fixation_intervals.json).
- [I1: Historical GPU profile implementation, pinned](https://github.com/rueijrwu/gaze_acc_joint_estimator/blob/54deab5871b4713f67e2ab6ab86f1f950dc95eee/full_position/gpu_profile.py): possible donor, not present in this clean branch or compatible unchanged.
- [T1: PyTorch function transforms](https://docs.pytorch.org/docs/stable/func.api.html).
- [T2: PyTorch Jacobians, Hessians and batched transforms](https://docs.pytorch.org/tutorials/intermediate/jacobians_hessians.html).
- [C1: CuPy performance and synchronized benchmarking](https://docs.cupy.dev/en/stable/user_guide/performance.html).
- [S1: SciPy bounded nonlinear least squares](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html).

Library references describe available capabilities, not the version installed in the user's execution environment. Record the versions actually used. GPU speed is an implementation aid; the scientific goal remains an identifiable shared optical accommodation model with full-calibration cross-agreement.
