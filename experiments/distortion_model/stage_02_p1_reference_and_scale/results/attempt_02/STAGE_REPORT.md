# Stage 2 — P1 reference and scale

Gate / attempt / candidate: S2/G2 / attempt_02 / constrained P1 initialization.  
Status: COMPLETE. Decision: **GO_WITH_LIMIT**.  
Evidence kind: empirical initialization and synthetic numerical contracts.  
Reviewer: root implementation/audit agent. Decision timestamp: 2026-10-10T01:18:26.471790+00:00.

## Provenance and parent

Source commit: `efd88bd3b326653aedc3b383143ef10856d22092`; remote branch matched at execution. Dirty source hash: `dc6385ee6379dfb8709b31318bc1d02d02a7b9c62d0d068e134d1749d81d1102`. Config hash: `aa09f77bfc1b574359523c763aa60c7a36c8bdfaf964ec27406d9df9983b9140`. Exact theory/plan/source hashes and runtime are in [provenance.json](provenance.json); source copies are in [source_snapshot](source_snapshot/). Theory SHA256: `8a807332a55c2ac6a786cb2311bbb62895ce37dab9a0c0bf0f3fa884b4200a80`. Estimator plan SHA256: `74c78372474bd453e295590009804fdc2d08833ec1fa705541ab0805c2dde8b7`. Gate policy SHA256: `2b5aadbed756beb0f7d68e29c5ecd47020bc2606fec54a199d5c1c3cf93e7b01`.

Parent: stage 1 attempt_02, reviewed G0/G1 GO_WITH_LIMIT: [report](../../../stage_01_data_and_bootstrap/results/attempt_02/STAGE_REPORT.md). [checkpoint.json](checkpoint.json) identifies unchanged parent population, slot manifest and covariance hashes. Raw source bytes were checked and reconstructed measurements matched the parent. Captures 1–4 and all twenty full intervals remain; captures 5/6 were excluded. Population: 100,090 scheduled, 89,175 input-valid and scale-solved, 10,915 unavailable inputs, zero unresolved valid scales. Expected outcome slots remain 300,270. No trimming or state clipping was used.

## Question, assumptions and parameters

Can an accommodation-independent, gaze-conditioned P1 reference provide a usable positive nuisance scale for G3? Compute `g=(aᵀR11⁻¹e)/(aᵀR11⁻¹a)` from P1 edges at each trial visual gaze. P4 and A are absent from this conditional scale calculation; approximate visual gaze and its refresh still inherit centroid information from both reflections.

Physical illuminator geometry and camera/optical-axis alignment are unavailable. The declared diagnostic is observed fixed-axis bilateral pattern balance, `norm([Lx+Rx−2Mx,Ry−Ly])/norm(R−L)`. It supplies a reproducible operational reference and does not identify physical symmetry or anatomical zero. All five balance profiles and original-row block summaries are preserved in [p1_balance.json](p1_balance.json).

The selected reference is capture 1 nominal −10°, the smallest complete-population mean balance score (0.113382). The other means at −5, 0, +5, +10° are 0.118484, 0.118059, 0.123117, 0.127212. This is an endpoint, and the minimum is neither extrapolated nor claimed sharp. Freeze omega1 at the original visual-gaze mean −10.0209041116°. Build the native-pixel empirical template from all 4,433 complete reference frames, averaging individually centered triples without symmetrizing; radius 251.920815 px.

Fixed: template, native length, centroid origin, camera axes, omega1, visual domain [−20,+20]°, R11, and initial isotropic convention alpha1+beta1=0. Fitted: trace-free anisotropy `delta1=100*alpha1=-100*beta1` and `keystone1=10*radius*gamma1`. Derived: alpha1=2.29725207e−5 deg⁻², beta1=−alpha1, gamma1=1.50554097e−6 px⁻¹deg⁻¹, framewise g and reference means/edges. This two-parameter reduction is provisional and keeps a fixed isotropic gauge. [p1_reference.json](p1_reference.json) contains the full parameter roster and template.

## Actual fit and residual evidence

Identity and constrained K1 use exactly the same 89,175 frames, frozen stage-one visual theta, frozen covariance, and equal exposure weights. Objective is half the mean across exposures of framewise squared whitened edge residuals. Native edge RMS is the square root of the mean across exposures of their mean four-coordinate squared error.

| Snapshot | Objective cost | Equal-exposure native edge RMS (px) |
|---|---:|---:|
| Identity at frozen stage-one theta | 2263.041338 | 1.926628 |
| Constrained K1 at the same theta | 1921.198495 | 1.590827 |
| Same frozen K1 evaluated after one bootstrap refresh | 1918.951587 | 1.589525 |

Cost drops 15.11%; native RMS drops 17.43% at the matched fit snapshot. These are training improvements. Residuals remain structured: for example capture 1 +5° RMS is 3.034 px, with positive signed edge biases; capture 2 +5° is 2.773 px. Five exposure RMS values rise slightly under K1. A scalar and this reduced map leave directional discrepancy. No universal RMS cutoff is imposed. The P1 profile's weighted scale-parallel residual vanishes algebraically and is checked numerically; the four edge coordinates retain their original covariance.

Counts and original-snapshot values below reproduce [p1_residuals.json](p1_residuals.json). Edge order is M−L x/y, R−L x/y. Each valid frame is scale-solved; unavailable means invalid original input. Signed means are in native pixels.

| Exposure | Capture | Nominal gaze ° | Scheduled | Valid/solved | Unavailable | Identity RMS | K1 RMS | Mean g | Signed edge means (Mx,My,Rx,Ry) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 1 | -10 | 4800 | 4433 | 367 | 0.4799 | 0.4799 | 1.000003 | -0.002, -0.001, -0.004, -0.000 |
| 1 | 1 | -5 | 4800 | 4800 | 0 | 0.6113 | 0.6631 | 0.998641 | 1.212, 0.162, 0.215, 0.081 |
| 2 | 1 | 0 | 4900 | 4900 | 0 | 1.2288 | 1.1101 | 0.995830 | 1.191, 0.437, 0.396, 0.721 |
| 3 | 1 | 5 | 5000 | 4993 | 7 | 3.9420 | 3.0341 | 0.985329 | 4.077, 1.545, 3.632, 1.127 |
| 4 | 1 | 10 | 5200 | 5154 | 46 | 2.0447 | 1.7806 | 0.998425 | 1.350, -0.188, -2.818, 1.066 |
| 5 | 2 | -10 | 4600 | 3912 | 688 | 0.4041 | 0.4080 | 1.000598 | 0.596, 0.096, 0.159, 0.050 |
| 6 | 2 | -5 | 5000 | 4345 | 655 | 1.2657 | 1.2992 | 1.002184 | 2.072, 0.273, -0.079, 0.414 |
| 7 | 2 | 0 | 5000 | 4553 | 447 | 1.3160 | 1.2840 | 0.998140 | 2.481, 0.220, -0.091, 0.197 |
| 8 | 2 | 5 | 5000 | 4781 | 219 | 3.1809 | 2.7729 | 0.998638 | 0.933, 1.576, 5.072, 0.882 |
| 9 | 2 | 10 | 5290 | 4435 | 855 | 2.9783 | 2.2042 | 0.993416 | 4.379, 0.201, -0.189, -0.148 |
| 10 | 3 | -10 | 4700 | 4505 | 195 | 0.5029 | 0.5080 | 1.001135 | 0.665, 0.190, -0.130, 0.482 |
| 11 | 3 | -5 | 4900 | 3902 | 998 | 2.1343 | 2.0194 | 1.008041 | -1.332, -0.635, -2.907, 0.375 |
| 12 | 3 | 0 | 5000 | 4811 | 189 | 2.1708 | 2.1209 | 0.998086 | 4.120, 0.420, -0.275, 0.559 |
| 13 | 3 | 5 | 5000 | 4251 | 749 | 1.9675 | 1.5082 | 0.995626 | 2.271, 0.847, 0.758, 1.410 |
| 14 | 3 | 10 | 5500 | 2326 | 3174 | 2.2802 | 1.4123 | 0.993372 | 2.463, 0.364, -0.644, 0.954 |
| 15 | 4 | -10 | 4800 | 3914 | 886 | 0.4726 | 0.4655 | 1.000829 | 0.676, -0.127, -0.473, -0.172 |
| 16 | 4 | -5 | 4900 | 4899 | 1 | 0.7195 | 0.7698 | 0.999274 | 1.485, 0.226, 0.161, 0.238 |
| 17 | 4 | 0 | 5000 | 4747 | 253 | 1.4727 | 1.3677 | 0.996605 | 2.546, 0.484, 0.214, 0.715 |
| 18 | 4 | 5 | 4900 | 4738 | 162 | 2.3162 | 1.7579 | 0.994080 | 3.193, 0.699, 0.358, 1.090 |
| 19 | 4 | 10 | 5800 | 4776 | 1024 | 1.7472 | 1.1879 | 0.995932 | 1.085, 0.081, -1.712, 1.132 |

Plots reviewed:

- [Balance and contiguous blocks](p1_balance.png): axis-specific discrepancies and reference uncertainty.
- [Scale and residuals](p1_scale_and_residuals.png): capture/gaze differences and native-row trajectories; plotted stride 10 only, full rows fit and saved.
- [Measured/predicted patterns](p1_patterns.png): one original-row example per exposure; full residual arrays supply quantitative detail.
- [Bootstrap refresh and coefficient blocks](bootstrap_refresh_and_blocks.png): approximate starts and within-interval stability.

## Numerical evidence and repaired attempt

[tests.json](tests.json) records 31 passing checks, zero errors/failures/skips: 17 stage-one checks and 14 stage-two checks. Stage-two finite-difference tolerances are rtol 3e−6 / atol 1e−5, with parameter-specific steps in [config.json](config.json); the test report's inherited top-level rtol/atol describe the stage-one checks. Stage-two checks cover synthetic scale/coefficient recovery, analytic composition/profile derivatives, fixed-reference and translation invariance, changed edge-origin covariance (stage one), P4/A independence at fixed theta/P1, equal-exposure weighting, admissible domains, rank failure, frozen references and framewise-ratio bootstrap.

Attempt_01 was preserved as REPAIR because ftol stopped above the predeclared projected-gradient certificate. Attempt_02 adds bounded same-objective Gauss–Newton stationarity polishing, at most five steps, accepting only improved projected stationarity with cost within a 64-epsilon rounding allowance. No tolerance relaxation, extra optical capacity, population change or increased optimizer budget. Both selected-reference starts needed two tiny polish steps. Final range-scaled projected gradient infinity norms are 1.13e−8 and 2.36e−7, below fixed 1e−6. Both recover matching coefficients/cost. Conditional Jacobian rank is 2, singular values 22997.0684 and 1746.7548. Full-domain minimum denominator is 0.988841, minimum x scale 1, minimum y scale 0.979296. [reference_fits.json](reference_fits.json) retains every start and accepted step.

All five contiguous-block diagnostic fits certify. Scaled delta ranges 0.001630–0.002617 and keystone ranges 0.001560–0.004942, both positive within this fixed reference. The variation is material and blocks remain in the full fit. These are stability diagnostics, not confidence intervals or independent physiology.

The adjacent −5° operational reference also certifies, with identity/K1 costs 1717.740327/1430.239798. Its scaled delta is 0.0004561 and keystone −0.0041790: the keystone sign changes with reference. Alternative g differs from chosen g by +0.0265% to +0.1550%, RMS relative difference 0.1221% (same original theta). Keep this bounded alternative; do not select omega1 by lower training cost. A useful scale does not establish absolute optical coefficients.

## Consistent export and runtime

Refresh the inverse centroid bootstrap once using the five means of **framewise dx/g** at fitted original theta; then recompute g, F1 and mu1 at the refreshed per-frame theta. No post-refresh refit occurred. [bootstrap.json](bootstrap.json) names both snapshots. Five refreshed reference means are −10.016974, −5.014899, −0.002070, +5.121306, +9.912637°. Theta change standard deviation is 0.024914° (mean −0.004123°, range −0.230923 to +0.249885°); all 89,175 starts are finite and within bounds, exported theta spans −11.025375 to +10.961497°. Demand-based A starts remain provisional.

Exported g spans 0.970479–1.013326, mean 0.997630, with zero nonpositive or unavailable scales among valid frames. [initial_states.npz](initial_states.npz), [p1_at_refreshed_theta.npz](p1_at_refreshed_theta.npz) and [checkpoint.json](checkpoint.json) are the consistent handoff. Hash fields for NPZ arrays are semantic `array_hash`; fields named sha256 identify file bytes. Original theta comparisons and alternative scales are retained separately.

Independent reconstruction checks are recorded in [reproduction_checks.json](reproduction_checks.json): parent and exported semantic hashes, bootstrap reconstruction, all twenty exposure RMS values, aggregate metrics and safe NPZ loading matched.

Runtime: 12.753 s overall, 2.703 s conditional reference fitting. Vectorized NumPy float64 and analytic Jacobians; four threads load sources and run independent reference/block fits, with BLAS/OMP threads 1 to avoid oversubscription. CuPy 14.2.0 detected one GPU. CPU execution is adequate for two shared coefficients; this gate requires no second backend.

## Decision and one next action

**GO_WITH_LIMIT**: the declared empirical reference supplies positive differentiable P1 scales across the full initialization population, rank/stationarity checks pass, and the residuals are interpretable. Retain the constrained K1 as a provisional effective P1 initialization; no extra capacity is released.

Physical source symmetry, omega1, optical spatial origin and camera-axis alignment remain unknown. Coefficients are reference-sensitive and bootstrap gaze is approximate. G5 revisits gaze-dependent deformation with better local angles; G7 revisits P1 reference/gauge sensitivity and structured residuals after joint refinement; G8 evaluates common-scale compatibility and whole-loop agreement. Capture, pose and demand are confounded. Full optical fit/cross-check/comparison flags remain false; physical Z and independent accommodation accuracy are unestablished.

One next action: implement **S3/G3 independent P4 reference and dual-zero conversion** using this reviewed checkpoint and shared P1 scale. Retry scope: none required for G2. Last usable checkpoint: this attempt_02 [checkpoint.json](checkpoint.json).
