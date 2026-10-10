# G7 attempt 01 — full DM0 joint calibration

**Status: COMPLETE_UNCERTIFIED. Decision: PAUSE.** Both declared starts completed their budget but failed numerical certification. Preserve this checkpoint as an uncertified result; it does not demonstrate physiological accommodation accuracy or model-law failure.

## Scope and provenance

Source commit `efd88bd3b326653aedc3b383143ef10856d22092`; dirty-source hash `17d7481c0f05b4ae294a8be489a212a7db1d010c845ecdecaec8f60836081d7a`; live remote branch SHA matched HEAD at launch. See [provenance](provenance.json), [configuration](config.json), [G6 attempt02 parent](../g6_attempt_02/checkpoint.json), and [G7 checkpoint](checkpoint.json). The starts were the reviewed G6 state plus a deterministic PCG64 seed-20261010 perturbation.

The fit used all 89,175 complete valid rows from twenty reviewed intervals, with 10,915 unavailable rows retained in the schedule. It used two dynamic states per valid frame, the complete ten-coordinate raw covariance, equal full-exposure weights, fresh P1-only scale, full-fixation mean anchors at 0.10° and 0.25 D, zero temporal penalty, and the declared regularization. The 19 free globals were a trace-free P1 coefficient pair, gamma1, m1, three P4-template tangent coordinates at fixed centroid/RMS radius, three K4 coefficients, and ten degree-2 center coefficients. P1 and P4 offsets remained separate and fixed at empirical operational references. Omega1 continuous uncertainty is unquantified; G3 discrete alternatives and physical zero meanings remain unresolved. No confidence intervals were invented.

## Full-objective outcomes and certificate

| Start | J | Point | Theta anchor | A anchor | Prior | State grad∞ | Global grad∞ | A bound rows | Profile rank field | Certified |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Common | 1522.256449 | 1251.101768 | 263.307547 | 7.652159 | 0.194975 | 0.249354 | 507.554 | 3747 | 0 (sentinel) | No |
| Perturbed, selected lower J | 1516.074461 | 1239.875051 | 268.343358 | 7.627808 | 0.228244 | 0.634107 | 671.004 | 3994 | 0 (sentinel) | No |

The stationarity threshold was 1e−6. Both starts had valid P1/P4 model domains and no global coefficients at bounds. The selected state vector had 3,994 A-bound rows and zero theta-bound rows; the other start had 3747 A-bound rows. Data rank was two for all 89,175 rows. The selected fit's serialized `observed_profile_rank=0` is a sentinel after local curvature prevented Schur elimination, not evidence of global rank deficiency. Both fits report `nonpositive free-state curvature` and are uncertified.

A separate finite-difference curvature census on the saved selected state found 214 negative free local eigenvalues among 174361 free local dimensions, minimum -0.00167650. Negative counts by exposure (ID:count) are `0:0; 1:0; 2:10; 3:0; 4:0; 5:0; 6:168; 7:14; 8:0; 9:0; 10:0; 11:1; 12:0; 13:0; 14:0; 15:0; 16:0; 17:21; 18:0; 19:0`. The rank-two mean-anchor correction per exposure cannot remove more than two negative directions within a group. History contains 144 accepted updates and one unaccepted observed-curvature polish proposal per start; each proposal is at iteration 0, and the loop stopped after that rejection. History does not record an exact rejection reason. The outer joint steps were all accepted (48 per start); maximum trust caps were 33.9432 and 32.1382. Diagnose this curvature and large trust caps, particularly exposure 6 and its outlier/transition rows, before considering one compatible continuation.

## Exposure accounting

| Exposure | Capture | Scheduled | Complete valid | Unavailable | Native relative RMS (px) | Mean theta (°) | Mean A (D) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | capture_1_detections.pkl | 4800 | 4433 | 367 | 2.767689 | -9.761813 | 0.152017 |
| 1 | capture_1_detections.pkl | 4800 | 4800 | 0 | 2.505776 | -7.753574 | 0.585015 |
| 2 | capture_1_detections.pkl | 4900 | 4900 | 0 | 4.994118 | 0.157823 | 1.748235 |
| 3 | capture_1_detections.pkl | 5000 | 4993 | 7 | 5.266118 | 8.085882 | 2.026246 |
| 4 | capture_1_detections.pkl | 5200 | 5154 | 46 | 4.997069 | 6.126037 | 1.343163 |
| 5 | capture_2_detections.pkl | 4600 | 3912 | 688 | 14.015599 | -10.179007 | 3.028349 |
| 6 | capture_2_detections.pkl | 5000 | 4345 | 655 | 8.212037 | -5.563899 | 3.536312 |
| 7 | capture_2_detections.pkl | 5000 | 4553 | 447 | 2.710169 | 2.772715 | 3.444975 |
| 8 | capture_2_detections.pkl | 5000 | 4781 | 219 | 5.500430 | 8.286956 | 2.442667 |
| 9 | capture_2_detections.pkl | 5290 | 4435 | 855 | 4.269461 | 7.008752 | 1.888761 |
| 10 | capture_3_detections.pkl | 4700 | 4505 | 195 | 2.871586 | -10.051347 | 2.723180 |
| 11 | capture_3_detections.pkl | 4900 | 3902 | 998 | 1.834548 | -4.886943 | 4.126847 |
| 12 | capture_3_detections.pkl | 5000 | 4811 | 189 | 2.199221 | 2.969579 | 3.360304 |
| 13 | capture_3_detections.pkl | 5000 | 4251 | 749 | 1.916797 | 5.777727 | 2.445056 |
| 14 | capture_3_detections.pkl | 5500 | 2326 | 3174 | 2.155669 | 6.795140 | 1.757691 |
| 15 | capture_4_detections.pkl | 4800 | 3914 | 886 | 2.368922 | -9.775774 | 1.741799 |
| 16 | capture_4_detections.pkl | 4900 | 4899 | 1 | 2.473771 | -7.186356 | 2.361738 |
| 17 | capture_4_detections.pkl | 5000 | 4747 | 253 | 2.576383 | 3.012222 | 3.069879 |
| 18 | capture_4_detections.pkl | 4900 | 4738 | 162 | 1.843242 | 5.801983 | 2.292368 |
| 19 | capture_4_detections.pkl | 5800 | 4776 | 1024 | 6.268944 | 6.381055 | 1.548977 |

## Compact progress diagnostic

The predeclared compact schedule had 100 rows and 300 held-point slots; all eight saved checkpoint/final compact records retain all 300 slots. At the selected final snapshot, 270 slots had retained inputs, 30 were unavailable, 267 retained-only inferences were individually certified, 3 were unresolved, and 267 were scored. Native progress RMS was 28.394074 px. These slot-level inference certificates are distinct from the failed global calibration certificate; this is not the G8 all-schedule score.

## Independent reconstruction and runtime

The independent NumPy `optics.predict_relative` reconstruction from saved theta/A, native P1 edges, R11 and immutable globals reproduces every valid prediction within 3.41e-13 px and g within 6.66e-16. Point loss, exposure means, anchors and priors match the saved objective; the selected J is 1516.074461. G5/G6 parent checkpoint and summary hashes and every archived source byte match the original provenance. Fitting code is unchanged since the attempt; the frozen authority status line was updated after review. See [mechanical audit](mechanical_audit.json), [audit utility](../../scripts/audit_g7.py), and [console log](console.log).

Total runtime was 156.785 seconds including backend audit, fitting, compact inference, and reporting on CuPy float64 / Tesla P100-PCIE-16GB. The same-equation CPU/GPU derivative and finite-difference audit passed on 40 predeclared actual rows. CPU maximum RSS was 1112944 KiB (measured peak); GPU free-memory/pool values are final snapshots, not peak. No automated test suite ran. The measured native relative RMS was 4.985584 px; compared with G6's 4.560659 px, the decrease in J from 2,134.091666 to 1,516.074461 (~29%) coexists with a roughly 9.3% increase in native RMS to 4.985584 px. Neither quantity certifies physiological accuracy.

Artifacts: [full fitted arrays](fitted.npz), [per-exposure and contiguous block summaries](conditions.json), [compact slots](compact/), [common visual-angle response](response_curves.png), [P4-local-angle response](response_curves_native_xi4.png), [states and residuals](states_and_residuals.png), and [source snapshot](source_snapshot/).

## Frozen G6/G7 center accommodation diagnostic

A post-fit comparison evaluates frozen \(H(\theta,A)=D(\theta,A)+\mu_4-\mu_1\) at \(g=1\), at \(A=0\) and 4 D, and visual \(\theta=-10,-5,0,5,10^\circ\). The calculation uses the public center and optical-reference functions; it does not refit either model. G7 is uncertified, so these values diagnose its selected snapshot only. The user-supplied 50 arcmin (0.833°) expectation is not stated in the located authority documents and is not a fit constraint.

| Visual theta (°) | G6 ΔHx (px) | G7 ΔHx (px) | G7 ΔHy (px) | G7 equivalent local horizontal shift (°) |
|---:|---:|---:|---:|---:|
| -10 | 0.612 | 33.325 | 13.336 | 0.621 |
| -5 | 1.174 | -133.975 | 6.684 | -3.251 |
| 0 | 1.738 | -301.274 | 0.033 | -10.478 |
| 5 | 2.301 | -468.573 | -6.619 | -28.760 |
| 10 | 2.865 | -635.873 | -13.269 | -166.005 |

The equivalent shift divides ΔHx by the local G7 horizontal theta slope at \(A_{ref}=0.36036\) D; it is a local linear diagnostic, not an inverse estimate. The G7 horizontal theta slope reverses sign between \(A_{ref}\) and 4 D at theta 0°, 5°, and 10°. The frozen G7 center coefficients include \(b_{A,x}=-75.317\) px/D and \(s_{A,x}=-83.648\) px per D·(theta/10), compared with G6 values 0.436 and 0.283. This large center accommodation/gain tradeoff joins the existing curvature and trust-cap diagnosis. Neither the user expectation nor this diagnostic validates a physiological law or authorizes a constraint. See [frozen comparison data](centroid_accommodation_comparison.json) and [reproduction script](../../scripts/compare_centroid_g6_g7.py).

## Reviewed disposition

**PAUSE.** `fit_complete=false`, `fit_certified=false`, `crosscheck_complete=false`, `comparison_complete=false`; G8 is not authorized. One next action: diagnose the nonpositive local curvature and large trust caps, starting with exposure 6 and its outlier/transition rows, and review the frozen G6/G7 center accommodation/gain tradeoff against an authoritative source for any expected scale. The user-supplied 50 arcmin expectation is not authority or a constraint. If the diagnoses justify a continuation, a candidate repair is a declared physical constraint or regularization on the full H accommodation contribution and monotone gaze response; its meaning and source, including the user-supplied 50 arcmin expectation, must be resolved first. No fit constraint has been applied.
