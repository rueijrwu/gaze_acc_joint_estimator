# S3/G3 — Independent P4 angular reference

Gate / attempt / candidate: S3/G3 / attempt_01 / fixed operational P4 angular reference.  
Status: COMPLETE. Decision: **GO_WITH_LIMIT**.  
Evidence kind: empirical reference initialization and synthetic numerical contracts.  
Reviewer: root implementation/audit agent. Decision timestamp: 2026-10-10T01:34:43.567283+00:00.

## Source, configuration and parent

Source commit `efd88bd3b326653aedc3b383143ef10856d22092`, matching remote branch at execution. Dirty source hash `bfb292312ace572cc394ba77050ae45d1aed5941a484efa09e3d08466a3a77dd`; config hash `6540b77a4fb16c0abea8e964a30ef61c735be1340c683f9e65c2ec59e454f647`. [provenance.json](provenance.json) lists all exact source/theory/plan hashes and runtime. [source_snapshot](source_snapshot/) preserves original source layout and bytes. Theory SHA256 `8a807332a55c2ac6a786cb2311bbb62895ce37dab9a0c0bf0f3fa884b4200a80`; estimator-plan SHA256 `74c78372474bd453e295590009804fdc2d08833ec1fa705541ab0805c2dde8b7`; gate-policy SHA256 `2b5aadbed756beb0f7d68e29c5ecd47020bc2606fec54a199d5c1c3cf93e7b01`.

Parent: reviewed G2 GO_WITH_LIMIT, stage 2 attempt_02 [audit](../../../stage_02_p1_reference_and_scale/results/attempt_02/STAGE_REPORT.md). Parent checkpoint SHA256 `e23e2cf748977bc50168af0d00fab16389c22f7969399e34b8bbd80fcb158858`; summary SHA256 `041aff80f31d8b70ba7cbdc2969f43a298dba01c3c0c60be24aa6c61bd660aae`. [checkpoint.json](checkpoint.json) pins the original population, slots, covariance, P1 export and frame states. Raw trusted captures were reloaded with pinned hashes; original measurements and manifests matched, and every exported P1 field was independently reevaluated at the frozen visual states.

Population: all twenty reviewed intervals, captures 1–4, 100,090 scheduled rows; 89,175 complete valid/processed diagnostic rows, 10,915 unavailable original inputs, zero unresolved valid diagnostics. All 300,270 expected omission slots remain in the parent manifest. Captures 5/6 excluded. No trimming, new validity gate, zeroing of fixations, frame subsampling for calculations or P4 inference fit occurred.

## Question and fixed conventions

Can P4 independently supply a supported operational angular reference, with correct conversion between visual gaze and P1/P4 native angles, and measured support for G4?

Freeze G2 P1 template, omega1, K1, GLS scale policy, covariance, visual theta and provisional A starts. Divide translation-free centered P4 by the **same exported P1 g**. Retain the corrected P4-minus-P1 centroid displacement separately. P4 size is preserved; there is no P4 fitted scalar, area normalization, affine alignment, source sorting or capture-specific zero.

The fixed-axis diagnostic in corresponding source slots is `norm([q0x+q2x−2*q1x,q2y−q0y])/norm(q2−q0)`. Correspondence `[2,1,0]` was already applied by the loader exactly once. The scale-invariant score describes observed bilateral imbalance; physical source symmetry and camera/optical-axis alignment remain unknown. It is used only for reference selection. The angular zero does not measure a spatial radial center.

Selection was declared before the experiment: choose the smallest equal-capture mean score at the five nominal gaze bins. Assign one shared omega4 from the mean frozen G2 visual theta in the selected **low-demand** exposure. This distinguishes observed nominal bins from approximate inferred gaze. No continuous minimum, outside-range zero, interpolation or offset prior is introduced.

## Reference profiles and independent selection

| Nominal target ° | Equal-capture mean balance | Across-capture SD |
|---|---:|---:|
| -10 | 0.07209194 | 0.00532445 |
| -5 | 0.08691235 | 0.00163700 |
| 0 | 0.09399615 | 0.00944109 |
| 5 | 0.10283209 | 0.00332597 |
| 10 | 0.10996880 | 0.00564729 |

All four individual capture curves have their smallest sampled score at **−10°**, the shared selected bin. Physical symmetry is not reached or localized by this endpoint minimum. Signed horizontal imbalance stays positive across the bins and dominates the norm; vertical imbalance crosses through zero between nominal 0 and +5°, which alone does not identify a bilateral physical zero. Profiles differ by capture/demand, notably capture 3 at nominal 0° and capture 4's broad +10° variability. There is no shift of the winning sampled bin across captures. This cannot exclude continuous or accommodation-dependent shifts.

Selected low-demand exposure 0: n=4,433, score mean 0.065514, frame SD 0.003971. Freeze:

- omega1 = −10.020904111586926° (unchanged G2 operational P1 reference).
- omega4 = −10.016974132845107° (independently selected P4 bin, using G2 visual snapshot).
- Delta14 = omega4−omega1 = +0.00392997874181944°.

Their near-coincidence reflects the same winning endpoint and the upstream bootstrap refresh; it does not establish coincident physical optical axes. Keep independent offsets in code and checkpoint.

All five sampled low-demand reference alternatives are saved in [p4_reference.json](p4_reference.json). The bounded adjacent alternative is nominal −5°, omega4=−5.0148989655383795°; its pooled score is 0.08691235 versus chosen 0.07209194. It shifts every xi4 by −5.0020751673° at fixed visual theta, while P1 angles, scales and original frame states remain unchanged. The five sampled offsets are conventions, not confidence bounds. No P4 template, accommodation curve or keystone coefficients have been fit at this gate.

## Complete exposure and block evidence

Native xi4 is evaluated for each frame. Block ranges below are the five contiguous original-row block mean scores; no block is omitted from the saved population. Near-reference counts use the fixed diagnostic window |xi4| <= 2.5° (half the nominal spacing), not a calibration selection or physiological precision requirement.

| Exposure | Capture | Demand D | Nominal ° | Scheduled | Valid/processed | Unavailable | Mean score | Frame SD | Block mean range | Mean xi4 ° | Near-reference |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 0 | 1 | 0.36036 | -10 | 4800 | 4433 | 367 | 0.065514 | 0.003971 | 0.063492–0.070181 | 0.00000 | 4410 |
| 1 | 1 | 0.36036 | -5 | 4800 | 4800 | 0 | 0.085657 | 0.001226 | 0.084544–0.087022 | 5.00208 | 13 |
| 2 | 1 | 0.36036 | 0 | 4900 | 4900 | 0 | 0.088757 | 0.003632 | 0.083673–0.091047 | 10.01490 | 0 |
| 3 | 1 | 0.36036 | 5 | 5000 | 4993 | 7 | 0.099405 | 0.003122 | 0.097012–0.101261 | 15.13828 | 0 |
| 4 | 1 | 0.36036 | 10 | 5200 | 5154 | 46 | 0.119300 | 0.002800 | 0.117415–0.123276 | 19.92961 | 0 |
| 5 | 2 | 4 | -10 | 4600 | 3912 | 688 | 0.069745 | 0.014889 | 0.064932–0.076982 | 0.15029 | 3841 |
| 6 | 2 | 4 | -5 | 5000 | 4345 | 655 | 0.084934 | 0.006419 | 0.084056–0.086440 | 4.02424 | 5 |
| 7 | 2 | 4 | 0 | 5000 | 4553 | 447 | 0.083755 | 0.002658 | 0.081241–0.086951 | 7.94880 | 0 |
| 8 | 2 | 4 | 5 | 5000 | 4781 | 219 | 0.102394 | 0.002976 | 0.101340–0.103509 | 11.96042 | 0 |
| 9 | 2 | 4 | 10 | 5290 | 4435 | 855 | 0.104110 | 0.000696 | 0.103704–0.104478 | 16.13214 | 0 |
| 10 | 3 | 3 | -10 | 4700 | 4505 | 195 | 0.073034 | 0.004891 | 0.066541–0.076281 | 0.23902 | 4500 |
| 11 | 3 | 3 | -5 | 4900 | 3902 | 998 | 0.088501 | 0.002886 | 0.086253–0.090830 | 4.43109 | 0 |
| 12 | 3 | 3 | 0 | 5000 | 4811 | 189 | 0.108961 | 0.009906 | 0.095434–0.117033 | 8.49832 | 0 |
| 13 | 3 | 3 | 5 | 5000 | 4251 | 749 | 0.101240 | 0.008582 | 0.093782–0.110872 | 12.92932 | 0 |
| 14 | 3 | 3 | 10 | 5500 | 2326 | 3174 | 0.107954 | 0.002772 | 0.106387–0.108625 | 17.14109 | 0 |
| 15 | 4 | 2 | -10 | 4800 | 3914 | 886 | 0.080075 | 0.010287 | 0.077946–0.082081 | 0.47782 | 3875 |
| 16 | 4 | 2 | -5 | 4900 | 4899 | 1 | 0.088557 | 0.001238 | 0.087274–0.090242 | 4.93084 | 0 |
| 17 | 4 | 2 | 0 | 5000 | 4747 | 253 | 0.094512 | 0.003905 | 0.092533–0.097462 | 9.37874 | 0 |
| 18 | 4 | 2 | 5 | 4900 | 4738 | 162 | 0.108289 | 0.002982 | 0.105478–0.109696 | 13.98465 | 0 |
| 19 | 4 | 2 | 10 | 5800 | 4776 | 1024 | 0.108512 | 0.047299 | 0.105624–0.116901 | 18.50070 | 0 |

[Full score/axis/block distributions](p4_balance.json) retain all twenty exposures and one hundred block records, including framewise visual-gaze support. Plots reviewed:

- [P4 balance and signed components](p4_balance.png): per-demand curves and block variability.
- [Native-angle trajectories](native_angle_trajectories.png): per-frame angles in original source rows; valid display stride 10, full arrays saved with availability.
- [Retained P4 size and near-reference support](p4_size_and_support.png): corrected native size remains visible and measured near-zero coverage exists.

The corrected RMS-radius differences across captures remain present in reference pixels (approximately 178–194 px over the plotted condition means). These are descriptive evidence; demand and capture/pose remain confounded. No M(A) fit or physiological accommodation inference is made by this view.

## Native support and identities

| Capture | Demand D | Complete valid | Near-reference frames | Near-frame mean xi4 ° | Near-frame SD ° |
|---|---:|---:|---:|---:|---:|
| 1 | 0.36036 | 24280 | 4423 | -0.032241 | 0.140782 |
| 2 | 4 | 22026 | 3846 | 0.069783 | 0.158107 |
| 3 | 3 | 19795 | 4500 | 0.236369 | 0.253987 |
| 4 | 2 | 23074 | 3875 | 0.437092 | 0.182777 |

Total near-reference frames: 16,644. Every demand/capture has measured support; G4 can use actual individual local angles. All intervals remain in the full schedule. Selected reference fixation xi4 spans −1.004297 to +10.134584° with SD 0.513469°; averaging sets its mean to zero and **does not** set its individual frames to zero. Transition excursions are preserved.

Across the full valid population xi4 spans −1.008400 to +20.978471°; xi1 spans −1.004471 to +20.982401°. For the proposed visual domain [−20,+20]°, future optical denominator checks must use shifted xi4 domain [−9.983026,+30.016974]°, rather than visual bounds directly. These are available angle domains; empirical K4 denominators/rank are deferred until fitting K4 at G5.

Conversions: `xi1=theta−omega1`, `xi4=theta−omega4`, `xi1=xi4+Delta14`, `theta=xi1+omega1=xi4+omega4`. Maximum real-array conversion error is 3.55e−15°; roundtrip error 1.78e−15°. Native P1 identity error is 0 px; P1 at P4 zero matches K1(Delta14,b1) exactly and differs from its own native template by max 0.000360575 px. P1's native template/operator were preserved. [native_support.json](native_support.json) records these values.

[tests.json](tests.json): **40 passed**, zero failures/errors/skips, including nine new G3 contracts. They verify independent dual-zero conversions, native K1/K4 identities including K4(0,A)=I at synthetic A, reference re-expression with baseline and operator transformed together, unchanged P1 g at fixed visual theta under omega4 changes, preserved P4 magnification and correspondence, fixed-axis invariance, invalid-scale/missing/collapsed outcomes, equal-capture reference weighting and framewise near-reference angles. G3 tolerances rtol 1e−10 / atol 1e−9; earlier suite tolerances remain separately declared. Numerical contracts do not validate physical source symmetry. No real K4/M fit is claimed.

## Checkpoint, runtime and decision

[p4_diagnostics.npz](p4_diagnostics.npz) stores all scheduled rows: angle arrays, score/axis components, corrected centered P4, preserved displacement, blocks, availability and diagnostic near-window mask. The [checkpoint](checkpoint.json) points to unchanged G2 state/P1 arrays and frozen G0 manifests/covariance. NPZ hash fields are semantic `array_hash`; fields named sha256 are byte hashes. [reproduction_checks.json](reproduction_checks.json) records independent recomputation of hashes, diagnostics, all exposure/block summaries and conversions.

Runtime 4.408 s, vectorized NumPy float64 CPU with four source-loading and exposure-summary threads. CuPy/GPU availability is recorded in provenance; this diagnostic gate has no optimization requiring acceleration.

**GO_WITH_LIMIT:** independent P4 evidence supports one fixed operational reference; positive P1 scales and complete diagnostics remain valid; conversions/native contracts pass; all captures have near-reference measurements. Endpoint choice, unknown physical symmetry/zero/spatial origin, approximate transferred gaze and capture/demand confounding remain unresolved. Revisit P4 reference and individual local-angle deformation at G5 and joint reference sensitivity at G7. Common-scale optical compatibility belongs to G8. Full-fit/certificate/cross-check/comparison flags remain false.

One next action: implement **S4/G4 DM0-M1 accommodation baseline initialization** using measured near-reference support, the frozen G3 convention and unchanged P1 scale, with provisional A allowed to vary under soft mean demand anchors. Retry scope: none required for G3. Last usable checkpoint: this attempt_01 [checkpoint.json](checkpoint.json).
