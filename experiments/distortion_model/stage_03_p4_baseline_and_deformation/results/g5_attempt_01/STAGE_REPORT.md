# G5 / attempt 01 — Composed P4 gaze and accommodation initialization

Status: COMPLETE. Decision: **GO_WITH_LIMIT**. Scientific disposition: protocol ready; conditional component comparison is a **tradeoff**, full-model support unresolved. Evidence kind: empirical conditional initialization / synthetic numerical contracts. Reviewer: root implementation/audit agent, 2026-10-10T01:57:08.953406+00:00.

## Provenance and parent

Source commit `efd88bd3b326653aedc3b383143ef10856d22092`, remote matched at execution. Dirty-source hash `ecfa8c99729ec48e066fe74196d489278aabb245b0b767ecfb5450520a82623f`; config hash `a4d8cb64ded2edd552a88f2aa79ab31fba2f334aefeabdb22988fa8490bba1a9`. Theory SHA256 `8a807332a55c2ac6a786cb2311bbb62895ce37dab9a0c0bf0f3fa884b4200a80`; estimator-plan SHA256 `74c78372474bd453e295590009804fdc2d08833ec1fa705541ab0805c2dde8b7`; gate-policy SHA256 `2b5aadbed756beb0f7d68e29c5ecd47020bc2606fec54a199d5c1c3cf93e7b01`. [provenance.json](provenance.json) saves exact hashes/runtime and [source_snapshot](source_snapshot/) preserves source bytes.

Parent: reviewed [G4 GO_WITH_LIMIT](../g4_attempt_01/STAGE_REPORT.md). Parent checkpoint SHA256 `c4c4546a6825d22da631a900ad7a6e9f8faed6f2524492734b9a19d91d2faf02`; parent summary SHA256 `fa6e6e41d365c76df02cc9f6439ff1a236a10b210c305921adc5d83c00f26d17`. G3 native angular convention, G4 template and G2 theta/P1 scale remain pinned by [checkpoint.json](checkpoint.json), including unchanged G0 population, slots and covariance through the parent chain.

## Question and fixed/free roster

Does minimal A-independent K4 provide a well-defined composed P4 map for the next joint calculation, and does it explain the remaining directional shape response?

Keep the empirical native P4 reference, length, centroid-origin convention, source order and fixed axes. omega4=−10.016974° and omega1=−10.020904° remain independent and fixed. Use every individual's xi4=theta−omega4. DM0: M=1+m1*(A−Aref), Aref=.36036036036 D, no radial increment. Compose **M*b4 → K4(xi4,M*b4) → same external P1 g**. M stays inside the projective denominator. Center the forward model prediction for displays; no inverse-keystone about a measured centroid occurs.

G4's m/A first seed the three keystone coefficients at fixed G4 states. Then profile individual A for **all 89,175 complete frames** and refine four shared parameters: m1, alpha4, beta4, gamma4. The identity-K4 comparator separately refits m1 and every A on exactly the same rows, fixed P1 scales, covariance, full exposure weights, mean anchors and bounds. Two starts (seed and zero K4) are retained. No template/zero/origin refinements, per-capture zeros, P4 nuisance scale, exponents or optical capacity beyond the prescribed minimal K4 were added.

The conditional criterion is half the equal-exposure mean native four-P4-edge squared error in the frozen shape marginal, plus equally weighted full fixation-mean A anchors with s_A=.25 D; A bounds [0,6] D; temporal penalty zero. Visual theta is the unchanged provisional G2 snapshot. This is **not the final ten-coordinate J**: P1/P4 displacement and cross-covariance enter at G6/G7. The fixed raw marginal is a weighting metric, not an independently established plug-in residual covariance or hardware accuracy measurement.

## Matched evidence: weighted cost and native RMS disagree

| Component | Identity K4 | Minimal composed K4 |
|---|---:|---:|
| Conditional objective | 48.580114 | 44.857264 |
| Data term | 47.949317 | 44.419446 |
| Soft anchor term | .630798 | .437818 |
| Equal-exposure native edge RMS (px) | 5.376403 | 5.388521 |
| m1 (D⁻¹) | −.0178207254 | −.0190779304 |

Weighted cost decreases **7.66%** while native RMS rises **0.23%**. This is a metric tradeoff, not demonstrated overall pixel improvement or a cross-check result. The composed map is retained only as a bounded conditional initializer for G6/G7; the identity comparator remains available. Do not strengthen anchors, change covariance, drop difficult rows or expand spatial capacity to reverse this result.

Fitted K4: alpha4=−2.82212060e−5 deg⁻²; beta4=8.86851027e−5 deg⁻²; gamma4=1.27516914e−5 reference-px⁻¹deg⁻¹. These are conditional effective coefficients under the empirical origin, fixed theta and demand-anchor gauge. Numerical rank does not establish their physical identity. All three belong to the declared minimal map; no extra unsupported freedom is released.

Native residuals remain directional and gaze/capture dependent: often M−L x is negative and edge y is positive at positive targets. Capture1 +10° signed means are approximately [−9.868,+8.921,−1.087,+10.750] px. Capture2 −10° whole-interval RMS is 9.517 px despite near-window RMS 2.312 px; transition excursions outside the near window are retained. P1 scale dependence and extreme residual tails remain visible in [composition_and_scale.png](composition_and_scale.png). Valid flags do not guarantee that every observed shape satisfies the optical family.

## Full population and condition evidence

100,090 scheduled; 89,175 input-valid and conditionally A-solved; 10,915 unavailable original inputs; zero unresolved valid state outcomes. All twenty full intervals and 300,270 expected omission slots remain. No row/block/condition trimming or calculation subsampling. Point order is corresponding source 0/1/2; edge order is q1−q0 x/y, q2−q0 x/y. Below are original native pixel residuals; means/SD of A are descriptive latent-state output.

| Exposure | Capture | Nominal ° | Scheduled | Valid/solved | Unavailable | Identity edge RMS | Composed edge RMS | Mean A D | A SD D | Lower/upper A bounds | Signed edge means px |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0 | 1 | -10 | 4800 | 4433 | 367 | 0.4446 | 0.4491 | 0.36243 | 0.07952 | 6/0 | -0.022, 0.001, -0.011, 0.009 |
| 1 | 1 | -5 | 4800 | 4800 | 0 | 2.1118 | 1.9879 | 0.57750 | 0.10490 | 0/0 | -3.451, 1.821, 0.046, 0.612 |
| 2 | 1 | 0 | 4900 | 4900 | 0 | 3.3151 | 3.7198 | 0.33615 | 0.05263 | 0/0 | -4.038, 5.191, 0.467, 3.348 |
| 3 | 1 | 5 | 5000 | 4993 | 7 | 6.5637 | 6.4058 | 0.23064 | 0.18548 | 891/0 | -5.941, 8.849, 0.294, 7.041 |
| 4 | 1 | 10 | 5200 | 5154 | 46 | 8.4180 | 8.5752 | 0.77418 | 0.10803 | 18/0 | -9.868, 8.921, -1.087, 10.750 |
| 5 | 2 | -10 | 4600 | 3912 | 688 | 9.5614 | 9.5174 | 3.48394 | 0.40593 | 33/0 | -2.469, -1.078, -3.533, -0.860 |
| 6 | 2 | -5 | 5000 | 4345 | 655 | 6.1095 | 5.9338 | 3.90171 | 0.22786 | 13/0 | -4.235, 0.376, -2.030, 0.470 |
| 7 | 2 | 0 | 5000 | 4553 | 447 | 3.0073 | 2.7519 | 3.85649 | 0.08233 | 0/0 | -4.033, 1.679, -1.570, 2.900 |
| 8 | 2 | 5 | 5000 | 4781 | 219 | 4.9643 | 4.8899 | 4.10141 | 0.12049 | 0/0 | -7.257, 3.173, -2.150, 5.282 |
| 9 | 2 | 10 | 5290 | 4435 | 855 | 6.2075 | 6.1972 | 3.72187 | 0.12527 | 0/0 | -8.334, 3.843, -3.488, 7.557 |
| 10 | 3 | -10 | 4700 | 4505 | 195 | 2.0262 | 1.9886 | 2.89198 | 0.11370 | 0/0 | -1.997, -2.240, -1.805, -1.257 |
| 11 | 3 | -5 | 4900 | 3902 | 998 | 2.4622 | 2.1503 | 3.57924 | 0.31126 | 0/0 | -4.086, 0.225, -0.625, 0.989 |
| 12 | 3 | 0 | 5000 | 4811 | 189 | 4.8997 | 4.8110 | 3.20216 | 0.13982 | 0/0 | -7.889, 3.449, -1.490, 3.588 |
| 13 | 3 | 5 | 5000 | 4251 | 749 | 5.0296 | 5.1800 | 3.13315 | 0.09270 | 0/0 | -6.670, 4.873, -1.165, 5.943 |
| 14 | 3 | 10 | 5500 | 2326 | 3174 | 6.5442 | 6.6393 | 2.85508 | 0.20099 | 0/0 | -8.566, 5.494, -2.599, 8.098 |
| 15 | 4 | -10 | 4800 | 3914 | 886 | 2.4257 | 2.4080 | 1.84486 | 0.15819 | 0/0 | -2.981, -0.173, -1.284, -0.387 |
| 16 | 4 | -5 | 4900 | 4899 | 1 | 2.5406 | 2.2932 | 2.11405 | 0.14758 | 0/0 | -4.237, 1.337, -0.733, 0.802 |
| 17 | 4 | 0 | 5000 | 4747 | 253 | 3.8325 | 3.9710 | 2.05467 | 0.11756 | 0/0 | -5.307, 4.305, -0.508, 3.933 |
| 18 | 4 | 5 | 4900 | 4738 | 162 | 5.8114 | 6.0675 | 2.08146 | 0.14059 | 0/0 | -7.685, 6.689, -0.755, 6.524 |
| 19 | 4 | 10 | 5800 | 4776 | 1024 | 9.1900 | 9.2847 | 1.96785 | 0.13060 | 0/0 | -7.618, 6.049, -2.098, 8.997 |

961 lower-bound A states, zero upper-bound states; 891 are in capture1 +5° (17.85% of that condition). No entire exposure is all-bound. Full A spans 0–4.462737 D, mean 2.291841 D. Full-mean deviations include −.516064 D at capture2 −10°, +.579236 D at capture3 −5°, and +.413816 D at capture1 +10°. These deviations and the bound concentration are model/initialization limitations; s_A is a finite penalty scale, not an accuracy threshold. No physiological interpretation is assigned to the latent means.

[fitted_residuals.json](fitted_residuals.json) saves all twenty condition records, signed biases per point/axis and one hundred original-row block summaries; [identity_residuals.json](identity_residuals.json) uses the same accounting. [residuals_and_states.png](residuals_and_states.png) displays the matched comparisons and frame variation.

## Revisit G4 and numerical support

The original 16,644-frame near-reference window is unchanged; each xi4 remains individual. Aggregate native edge RMS there is **1.923954 px**, versus G4 1.922811 px. Per-capture changes: capture1 .423706→.419044, capture2 2.278038→2.312401, capture3 2.025472→1.988664, capture4 2.400116→2.408098 px. Update M and frame A with K4 present, but no claim that off-axis correction resolves the reference ambiguity. [near_reference_revisit.json](near_reference_revisit.json) records states, local-angle support and carried limitations. G4's 28% adjacent-reference slope sensitivity remains open; G5 freezes that convention rather than selecting a new zero by cost.

Both identity starts certify rank 1; both composed starts certify rank 4 and match costs/parameters. Final range-scaled global projected gradient norms are 1.66e−11 and 8.50e−7, below fixed 1e−6; individual-A projected gradient 7.16e−16, below 1e−8 certificate (inner target1e−11). Profiled conditional curvature eigenvalues: 1493.1374, 5708.7212, 54372.1774, 635920.4773. These include the mean-anchor gauge; they are not data-only physiological information. One start required one bounded same-objective stationarity polish at roundoff-level cost, using the already declared 64-epsilon allowance. [fit.json](fit.json), [identity_fit.json](identity_fit.json) and [keystone_seed.json](keystone_seed.json) preserve every outcome.

Full proposed visual/A rectangle, including shifted native angles: M ranges .892407–1.006875; minimum sx .974572, minimum sy 1, minimum denominator .961002. Thus the composed map is admissible across the declared domain, not only at fitted states. Native K4(0,A)=I and template baseline identity are covered by the numerical contracts.

[tests.json](tests.json): **55 passed**, zero failures/errors/skips, including five new G5 contracts for theta/model-mean derivatives, M-before-K4 order, centered prediction, all-four-global profile gradients, full-schedule synthetic global/A recovery and theta derivatives through P1 g. Earlier suites include native references, A derivatives, exact marginalization and dynamic full-mean anchors. Finite differences use rtol3e−6 / atol1e−5; exact invariants remain separately declared. Runtime **69.129 s**, vectorized NumPy float64 CPU. CuPy availability recorded, GPU backend unnecessary for this bounded initialization. [reproduction_checks.json](reproduction_checks.json) records independent hashes, objective/anchor and native residual reconstruction.

## Consistent handoff and gate decision

[fitted.npz](fitted.npz) contains all scheduled rows, individual A/M/F4/mu4, edge/centered native residuals, validity/status and unchanged theta/g. [model.json](model.json) contains the frozen empirical template and optical coefficients. [contributions_and_centers.npz](contributions_and_centers.npz) separates magnification/keystone changes and stores the derived relative-center initializer `Dobs=(c4−c1)/g−mu4+mu1`; this is a handoff diagnostic, not a fitted D polynomial or independent extra measurement. NPZ hashes are semantic array_hash; sha256 fields identify file bytes.

**GO_WITH_LIMIT** authorizes the next consistent relative-center/joint calculation: conversions/order/derivatives, positive full-domain scales, numerical stationarity and all-frame status accounting pass. Structured residuals, weighted/native tradeoff, bound concentration and fixation-mean discrepancies remain substantial. No full optical response support or independent accommodation accuracy is asserted. Preserve both K4 candidates. G7 must review these signatures after visual gaze and relative centers are jointly refined; freeze any freedom unsupported by that fit. G8 tests whole-loop common-scale compatibility and native/state agreement. Do not infer optical adequacy from this conditional objective decrease.

One next action: implement **S6/G6 corrected centers and forward D polynomial** against the complete ten-coordinate metric and cross-covariance using this initialization. Retry scope: none required for computational G5 handoff. Last usable checkpoint: [checkpoint.json](checkpoint.json). Full-fit/certificate/cross-check/comparison flags remain false.
