# G4 / attempt 01 — DM0-M1 accommodation initialization

Status: COMPLETE. Decision: **GO_WITH_LIMIT**. Evidence kind: empirical initialization / synthetic numerical. Reviewer: root implementation/audit agent, 2026-10-10T01:49:57.759846+00:00.

Source commit `efd88bd3b326653aedc3b383143ef10856d22092`; dirty-source hash `8b15bb9a540c3d66e0322fd26d6157cb30a2cb2775d7b9622a37ed1418fda5b3`; config hash `2926c8e864d75afb008fbaad3f2a75d391d72bee4d5afd36db2b4cd81ecb1876`. Exact theory/plan hashes, parent checkpoint/summary byte hashes, source snapshots and runtime: [provenance.json](provenance.json). Parent: [G3 attempt_01](../attempt_01/STAGE_REPORT.md), reviewed GO_WITH_LIMIT. [checkpoint.json](checkpoint.json) pins the parent, native population/covariance, unchanged G2 visual theta/P1 scale and this conditional shape snapshot.

## Question and declared criterion

Does near-reference P4 shape supply a usable effective accommodation initialization? G3's endpoint reference omega4=−10.016974° stays fixed. Build one real centered template from all 4,410 complete low-demand exposure-0 frames in the fixed |xi4|<=2.5° window. Its centroid is a fixed empirical origin convention, not a measured optical center. Native length and source order stay fixed. DM0-M1: `M=1+m1*(A−0.36036036036 D)`; K4 is identity, radial increment is absent, same P1 g multiplies P4 after M. No P4 nuisance scale is introduced.

Data term: four native P4 edges with frozen marginal `T R Tᵀ`, equal capture weights within the declared near window. This is conditional shape initialization, not the final ten-coordinate residual J. The raw marginal is a fixed weighting metric, not the covariance of the plug-in-scale residual; no calibrated precision/chi-square claim follows. Soft anchors use the **full twenty fixation means**, scale s_A=.25 D and zero temporal penalty. Only 16,644 near-window A states are optimized; 72,531 outside-window starts remain provisional at demand and contribute to those full means. All 89,175 complete frames and all 100,090 scheduled rows are saved. Theta and g are unchanged. A bounds [0,6] D and M positivity over that whole domain were declared before fitting.

Fitted: one shared m1 and individual near-window A. Fixed: template/origin/length, G3 zeros, camera axes, G2 theta/g, identity K4, outside-window provisional A. Derived: frame M, predictions/optical means, shape residuals. No physical Z, radial centers, exponents or independently optimized per-frame optical parameters.

## Direct measured response before fitted A

The descriptive Euclidean projection is rho=<centered P4/g,b4>/<b4,b4>; remaining direction is preserved. It is never divided out of measured P4 in inference. [measured_response.json](measured_response.json) preserves actual distributions against demand labels. Mean rho by demand: 0.36036 D ≈1.000002; 2 D ≈0.973704; 3 D ≈0.951485; 4 D ≈0.940095. Coordinate directional RMS is approximately .230, 1.277, 1.001, 1.202 reference pixels, respectively (2D point-vector RMS is sqrt(2) times this coordinate RMS). Reference capture overlaps the template sample.

The negative response is coherent, while directional biases remain. Capture, pose, transferred gaze and demand are confounded. This is consistent with an effective accommodation response under the fixed conventions; it does not establish physiological A accuracy or a physical linear law. [Measured response plot](measured_response.png) plots actual rho against demand before any fitted-A relation.

## Fit, state status and sensitivity

m1 = **−0.0168254613702 D⁻¹**. Matched near-window native P4 edge RMS decreases **9.254938 → 1.922811 px**. Full conditional cost 14.514451 = data 14.489154 + anchor .025297. Two starts converge to matching slope/cost; both certify rank 1 under the soft-anchor gauge. Global range-scaled projected gradient infinity norm 8.79e−11; individual-A projected gradient 6.33e−16. Conditional profiled curvature ≈599.077661. Certificates are global<=1e−6, state<=1e−8; the tighter state solve target is 1e−11. Analytic envelope gradients and coupled mean-aware state Newton steps are used, with bounded stationarity polishing available. No full-model certificate is claimed.

Near A ranges 0–4.385437 D with 17 lower-bound states, zero upper-bound states; all-bound behavior is absent. Frame variation remains. Full fixation-mean deviations include −.161583 D in capture2's −10° condition and +.192695 D in capture3's −10° condition; others with near states are within .001 D. The stated anchor scale is a penalty convention, not an allowed-error gate.

Declared sensitivities, both certified: doubling s_A to .50 D gives m1=−.0168386165 D⁻¹ (small slope change); changing to the bounded G3 −5° reference/template/window gives m1=−.0215282670 D⁻¹ (about 28% magnitude change). The latter is a different initialization population, **not a matched model score comparison**. Sign persists, physical coefficient magnitude is reference-sensitive. Keep the original G3 reference; do not choose by lowest training cost. [fit.json](fit.json) and [sensitivity.json](sensitivity.json) preserve every start and result.

## Full scheduled accounting and residuals

All twenty exposures are retained. Rows outside the near window are displayed with demand-based A and identity K4; their residuals quantify the unmodeled gaze response and do not certify those states. Original unavailable input remains explicit.

| Exposure | Capture | Nominal ° | Scheduled | Valid | Unavailable | Conditional A solved | Native edge RMS px | Full mean A D | Full A SD D |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | -10 | 4800 | 4433 | 367 | 4410 | 0.4331 | 0.36104 | 0.07980 |
| 1 | 1 | -5 | 4800 | 4800 | 0 | 13 | 1.8747 | 0.36070 | 0.01314 |
| 2 | 1 | 0 | 4900 | 4900 | 0 | 0 | 4.0913 | 0.36036 | 0.00000 |
| 3 | 1 | 5 | 5000 | 4993 | 7 | 0 | 7.6472 | 0.36036 | 0.00000 |
| 4 | 1 | 10 | 5200 | 5154 | 46 | 0 | 8.9144 | 0.36036 | 0.00000 |
| 5 | 2 | -10 | 4600 | 3912 | 688 | 3841 | 10.9642 | 3.83842 | 0.26020 |
| 6 | 2 | -5 | 5000 | 4345 | 655 | 5 | 6.8046 | 4.00017 | 0.00581 |
| 7 | 2 | 0 | 5000 | 4553 | 447 | 0 | 2.7847 | 4.00000 | 0.00000 |
| 8 | 2 | 5 | 5000 | 4781 | 219 | 0 | 4.4198 | 4.00000 | 0.00000 |
| 9 | 2 | 10 | 5290 | 4435 | 855 | 0 | 6.9808 | 4.00000 | 0.00000 |
| 10 | 3 | -10 | 4700 | 4505 | 195 | 4500 | 2.0258 | 3.19269 | 0.12763 |
| 11 | 3 | -5 | 4900 | 3902 | 998 | 0 | 3.7973 | 3.00000 | 0.00000 |
| 12 | 3 | 0 | 5000 | 4811 | 189 | 0 | 4.1842 | 3.00000 | 0.00000 |
| 13 | 3 | 5 | 5000 | 4251 | 749 | 0 | 4.6718 | 3.00000 | 0.00000 |
| 14 | 3 | 10 | 5500 | 2326 | 3174 | 0 | 7.4444 | 3.00000 | 0.00000 |
| 15 | 4 | -10 | 4800 | 3914 | 886 | 3875 | 2.4011 | 2.00084 | 0.17841 |
| 16 | 4 | -5 | 4900 | 4899 | 1 | 0 | 2.2550 | 2.00000 | 0.00000 |
| 17 | 4 | 0 | 5000 | 4747 | 253 | 0 | 3.7267 | 2.00000 | 0.00000 |
| 18 | 4 | 5 | 4900 | 4738 | 162 | 0 | 6.0410 | 2.00000 | 0.00000 |
| 19 | 4 | 10 | 5800 | 4776 | 1024 | 0 | 9.9793 | 2.00000 | 0.00000 |

[Fitted native residuals/point-axis biases/100 contiguous block records](fitted_residuals.json), [identity residuals](identity_residuals.json), [state/residual plot](residuals_and_states.png). Coordinates and summaries are original pixels; centered coordinate RMS and four-edge RMS are different diagnostic displays, never extra observations. No block/condition is dropped. [fitted.npz](fitted.npz) saves all scheduled rows, states/status, M/F4/mu4, native predictions/residuals, unchanged theta/g. [reproduction_checks.json](reproduction_checks.json) records independent recomputation.

[tests.json](tests.json): 50 passing checks, zero failures/errors/skips, including ten G4 contracts for marginal covariance, preserved scalar/directional decomposition, composition/state/global derivatives, coupled full-mean penalties, profiled gradient, synthetic m/A recovery, window state status, full-domain bounds and zero-slope unidentifiability. Exact and finite-difference tolerances remain separately recorded. Runtime **11.329 s**, vectorized CPU; CuPy availability recorded, no accelerator required.

## Decision and next action

**GO_WITH_LIMIT**: a nonzero effective response, valid M, dynamic bounded starts and numerical stationarity support G5. Its latent D scale depends on soft demand anchors; physical zero/spatial origin and the 28% reference sensitivity remain unresolved. G5 revisits these frames using each xi4 in the composed K4 and updates M under the same family. G7 reviews joint reference/anchor sensitivity and frame states; G8 reviews common-scale compatibility and whole-loop agreement. Full fit/cross-check/comparison flags remain false.

One next action: implement **S5/G5 conditional P4 keystone and accommodation refinement** on all complete frames, with matched identity-K4 comparison, then review its evidence. Retry scope: none required for G4. Last usable checkpoint: [checkpoint.json](checkpoint.json).
