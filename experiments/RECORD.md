# Experiment record

Updated 2026-10-11. This is the standalone record of the distortion-model and reverse-transform investigations. Full archived reports, fit arrays, audits, scripts, source snapshots, and plots remain in the archive paths listed below. The active reusable estimator code remains in `distortion_model/`; the shared-keystone run, audit, pipeline, reporting, and roundoff scripts remain in `experiments/reverse_transform/shared_keystone/scripts/`.

### P4 fixation geometry (descriptive, no fit)

`experiments/archive/reverse_transform/superseded_baseline/p4_fixation_geometry/` reports absolute and centered arithmetic mean P4 vertex positions for all 20 reviewed fixation intervals in captures 1–4, using all rows with three available P4 vertices (89,175 rows) without P1 gating. It preserves raw results and a parallel geometry-filtered sensitivity, separates mean-triangle metrics from averages of framewise metrics, and reports 250/500-row contiguous block variability. The filter removed 197 rows, each for edge-length outlier flags; no degenerate or reversed-orientation triangles were found after the corrected shoelace-area calculation. Capture 1's mean-triangle edge ratio rises from 1.0691 at −10° to 1.1016 at +10°. The V1-apex side asymmetry rises from 1.525% to 6.591%, with no zero crossing; raw and filtered isosceles side-asymmetry minima occur at −10° in all captures. The sampled data show no zero-gaze symmetry minimum. These descriptive patterns do not identify a distortion-free origin or physical source symmetry, and symmetry comparisons do not assume equilateral source geometry. The `RESULTS.md`, `summary.csv`, `vertices.csv`, `metrics.json`, plots, saved observations/masks, and reproducible `run.py` document scope and limitations.

### Capture 1 P4 inverse-keystone candidate

[`experiments/archive/reverse_transform/superseded_baseline/capture1_p4_inverse_keystone/RESULTS.md`](archive/reverse_transform/superseded_baseline/capture1_p4_inverse_keystone/RESULTS.md) records a conditional Capture 1 geometric inverse candidate. The frozen gaze initializer is a fixation-mean P4-minus-P1 centroid quadratic with a zero-target anchor; the primary seven-parameter directional-projective symmetric-template fit uses centered P4 vertices with no per-frame scale, P1 magnification, or accommodation estimate. Its filtered weighted point MSE is 0.82614 px², versus 15.46821 px² for the six-parameter horizontal-projective control; independent raw-geometry refit gives 0.83542 px². The fitted conditional origin is −16.679°, all seven normalized Jacobian directions are full rank, and no fit bounds are active. The 17-point origin profile spans 0.84447–24.38379 px²; it is an objective sensitivity profile, not a confidence interval. The inverse is valid on 24,148/24,148 filtered rows with synthetic round-trip RMS below 7e−14 px. Nominal fixation, accommodation demand, and origin remain conditional/confounded; this is not an absolute optical-zero or physiological estimate. Per-frame rows, masks, exact arrays, fit/profile details, plots, and reproducible `run.py` are included.

### P1 scale diagnostic (initial Stage 00 check)

[`p1_scale_analysis`](archive/reverse_transform/superseded_baseline/p1_scale_analysis/results/RESULTS.md) reports P1-only scale candidates and retained affine shape for 88,978 rows in the existing filtered P4-eligible cohort. The common measured reference is C1's nominal-zero fixation; a C1 train-only reference is a sensitivity check. The report keeps all four captures separate, plots fixation means against actual estimated gaze, preserves raw P1 triangles and determinant-one affine shape, and compares area/determinant, fixed-rotation similarity, best-rotation similarity, and RMS-size scales. It does not choose or apply a scale to P4, fit a keystone model, or estimate accommodation.

The area scale response across fixation means spans 1.1949%, 1.0121%, 1.3074%, and 0.6011% in C1–C4; capture-specific nominal demands are 0.36036, 4, 3, and 2 D. Gaze and fixation order remain confounded. The P4-conditioned cohort, reference-sample sensitivity, temporal spread, and fragmented conditional block intervals limit interpretation. Synthetic uniform-scale, determinant-one stretch, rotation, and finite-difference noise-sensitivity checks pass; row CSV/NPZ aggregation agrees exactly, source hashes match, and affine reconstruction/area/determinant-one identities hold to floating-point precision. These measurements do not distinguish physical magnification from isotropic distortion or support accommodation claims. Stage 00's matched P4 scale-effect sensitivity is now complete; next fit `k(A)` and `A` under the working `g=1` primary, retaining frozen P1 `m` only as a conditional sensitivity.

### Matched P4 refit with frozen P1-derived scale (Stage 00 sensitivity)

[`p4_p1_scale_effect/results/RESULTS.md`](reverse_transform/p4_gaze_accommodation_decoupling/stage_00_baseline_sensitivity/p4_p1_scale_effect/results/RESULTS.md) compares a full P4 shared-keystone/template + free `theta04` + capture-specific radial refit at `g=1` against the same refit using frozen bounded learned-P1 `m` on the exactly matched 88,978-row P4-filtered roster. Both branches use the same C1 nominal-zero empirical P4 reference, saved gaze, equal capture/fixation weighting, optimizer protocol, and initialization/multistart seeds. P1 `m` comes from `archive/reverse_transform/superseded_baseline/p1_joint_bounded_magnification/results/arrays.npz:frozen_learned_primary_m`, joined exactly by `(capture, source_row)`; its mean is 0.9878619, range 0.98–0.9993325, and 0.78% of rows sit at the 0.98 floor. In C1, mean `m` is 0.981875 at +5° (about 11% of rows at the floor) and 0.992378 at +10°. These gaze-associated changes are confounded with fixation order and are not causal evidence.

The equal-capture/equal-fixation objective rises from 1.2043393 to 1.7171983 px² (+42.58%; weighted RMS 1.09742 to 1.31042 px). Capture equal-fixation RMS is 1.00666→1.13750 px in C1 (+13.0%), 1.18925→1.51828 in C2 (+27.7%), 1.15835→1.49210 in C3 (+28.8%), and 1.02366→1.02145 in C4 (−0.22%). Per-fixation changes, residual time series/gaps, and mean residual vector triangles are in the report. The selected `g=1` and P1-m checkpoints are stationary; 7/8 and 8/8 near-best starts respectively were stationary. All 88,978 rows pass domain validity; minimum projective denominator is 0.87917/0.94729 and minimum radial derivative is 0.79602/0.78066 for `g=1`/P1-m. There are no active bounds. Fitted `theta04` changes −44.886°→−12.504°; the large offset movement indicates nuisance gaze/scale/template tradeoff, and the offsets have no absolute angle meaning. The row-pooled in-sample inverse diagnostic is 1.2112 px at `g=1`, 3.0124 px when only P1-m is substituted into the fixed `g=1` inverse, and 1.4859 px after full P4 refit with P1-m; all 88,978 frames are valid. The fixed-scale substitution retains the `g=1` parameters/template and reflects an expected mismatch because mean P1-m is 0.98786. It is distinct from the full-refit comparison; capture details are in the report. This likewise shows no improvement from frozen P1 scale under the tested model. It does not select a physical scale or estimate accommodation. Keep `g=1` as working primary for the next `k(A)`/A fit; retain P1-m as conditional sensitivity only until its convention and gaze/offset dependence are disentangled and validated.

## Current interpretation

The accepted current calibration is the Revision 2 shared-keystone S0 centered-shape fit, using the Capture 1 nominal 0° empirical reference. It fits one shared P1 keystone vector, one shared P4 keystone vector, one shared accommodation-to-radial slope, and per-frame gaze and accommodation states. A positive magnification is profiled per frame from P1 and applied to P4. The fitted accommodation states remain conditional on same-recording anchors and model geometry; they are not physiological ground truth.

A matched +5° empirical-reference sensitivity experiment passed its numerical checks, but its observation fit worsened. It reduced lower-bound accommodation contacts and mostly shifted the Capture 1 accommodation offset. It did not remove the within-capture gaze/fixation contrasts. The +5° result is a diagnostic convention comparison, not a replacement calibration or evidence that the +5° pattern is undistorted.

The experiments do not establish an absolute optical center, an absolute physical distortion-free reference, independent gaze or diopter accuracy, causal gaze effects, physiological accommodation, or transfer to another recording or subject. Nominal gaze, capture/demand, and fixation order are confounded in the available recordings. Every reported state and metric should be read within its stated model and anchor convention.

## Data and conventions

The reviewed population contains 100,090 scheduled rows across four captures and 20 fixation intervals: 89,175 complete P1/P4 frames and 10,915 unavailable rows. Source detections and fixation definitions remain in `data/`. Capture 1 source rows are the plotting/order coordinate; timestamps include backward jumps and must not be treated as elapsed time. The reviewed loader applies the P4 vertex correspondence permutation `[2, 1, 0]` once. The canonical Stage 03 provenance records these source identities; use files matching these hashes for exact replay:

| Input | SHA256 |
|---|---|
| `data/fixations/fixation_intervals.json` (roster) | `51c4c7514b31dab44b7eb898e7e640c624bf02133cf8324ae605998b455bf015` |
| `data/detections/capture_1_detections.pkl` | `fd89194a791f7bd88dd86f59f0f8e8e1275416f3a1d0ecda9e268946516dba54` |
| `data/detections/capture_2_detections.pkl` | `66c2fa8170b7140221476755db56ff9cd17dd238cb5ae8a39ccd5c6e1964d04d` |
| `data/detections/capture_3_detections.pkl` | `f8eb76c62d0de7ee7fa66ef99a49d0f424514131227990078a64065b4afdedf9` |
| `data/detections/capture_4_detections.pkl` | `2a307476772098b22b3933617065be3e6d51890754ea21dc9e40d99172ad32cf` |

The Stage 03 population semantic hash is `c29c4b63c74c99fb860019481352407727ebd9204c7e2603753bf617f38b1023`; the provenance and protocol JSON preserve the per-row identity details.

For current S0, the P1/P4 templates are centered arithmetic means from Capture 1's nominal zero-gaze fixation. The operational radial origin is `A_ref = 0.36036036036036034 D`; it sets relative radial increment to zero and does not assert absence of baseline barrel distortion. There is no RMS/area normalization after keystone, no independent P4 scale, and no temporal smoothing. The accommodation prior is a 0.25 D soft penalty on each fixation mean, not a hard per-frame bound. Operational state bounds are model limits, not measurements.

The +5° sensitivity uses centered arithmetic means over all 4,993 complete Capture 1 frames in source rows 15500–20499; seven scheduled rows are unavailable. It anchors the new templates at `(theta_x, theta_y, A, g) = (5°, 0°, 0.36036036036036034 D, 1)`, while retaining the original absolute-gaze homography, frozen gaze anchors, A targets, prior widths, bounds, and nine global optical parameters. The transported templates make the forward map equal the empirical patterns at the reference without applying +5° gaze deformation twice. This is a different empirical reference convention, not a pure reparameterization.

## Experiment history

### Distortion-model stages (historical branch)

These stages developed an earlier full-coordinate/center model and its held-point checks. They are retained for historical reasoning; their statuses and parameters are not the current S0 calibration.

| Stage | Main result | Decision and limit |
|---|---|---|
| 01 — data/bootstrap | 100,090 scheduled, 89,175 complete; linear centroid gaze initializer was usable. | `GO_WITH_LIMIT`; initialization only, not calibrated gaze or accommodation. |
| 02 — P1 reference/scale | Positive P1 GLS scale on all complete rows; matched native edge RMS 1.9266→1.5908 px. | `GO_WITH_LIMIT`; reference endpoint ambiguity and structured residuals remained. |
| 03 — P4 baseline/deformation | G4 effective size slope −0.01682546/D; G5 weighted cost 48.5801→44.8573 (−7.66%), while native RMS rose 0.23%. | `GO_WITH_LIMIT`; weighted/native tradeoff, 961 lower-bound A states, and capture/demand confounding remained. |
| 04 — centers/DM0/G7 | G6 reduced fixed-state equal-exposure relative-coordinate RMS 72.071→4.561 px. G7 attempt 04 received a saved-point numerical certificate with limits. | `GO_WITH_LIMIT`; structured residuals and state/optical ambiguity remained. G8 had not run in this branch. |
| 05 — G8 crosscheck | Frozen-model held-point consistency E=12.40853 px, gaze disagreement 0.477866°, accommodation disagreement 1.101718 D. | `COMPLETE_WITH_LIMIT`; internal consistency, not truth accuracy; model comparison incomplete. |
| 06 — reference sensitivity | Control passed calibration/G8; adjacent candidate failed the unchanged stationarity certificate. | `STOP_UNCERTIFIED_ADJACENT`; comparison incomplete, no optical ranking. |

Full reports and compact machine-readable bundles: `archive/distortion_model/`. These results motivated the later centered-shape shared-keystone S0 experiment but must not be combined numerically with its different objective and parameterization.

### Normalized reverse-transform stages (superseded comparison)

The early reverse-transform path used a normalized keystone shape convention. It is preserved as a historical control and is not a source of coefficients or states for the current raw S0 fit.

| Stage | Scope/result | Status |
|---|---|---|
| 01 | Capture 1 P1 reference, gaze, and frame magnification; four native K coefficients were fit. | Equal-fixation point SSE 1.22869→1.01318 px² (−17.54%) versus scale-only. Same-recording calibration; no accommodation or absolute optical origin. |
| 02 | Capture 1 P4 fit using P1 magnification and zero relative radial coefficient; native K terms include `k_ax=−1.4947e−5`, `k_ay=5.3532e−3 deg⁻²`. | Equal-fixation SSE 6.85869→2.93474 px²/point (−57.21%); pooled shape median/P95 1.8385/3.9370→0.6481/1.6516 px. Reference-capture comparison only; normalized size is a diagnostic. |
| 03 | Independent P1/P4 fits for captures 1–4; relative radial offsets C1–C4 were 0, −1.8569e−6, −1.4728e−6, −8.4772e−7 px⁻². | P4 objectives fell from 6.859→2.935, 197.259→24.217, 114.877→5.464, 48.928→9.533 px². Post-fit radial association was −5.2552e−7 px⁻² per nominal demand D. Capture/demand confounded; not physical barrel calibration. |
| 04 | Framewise accommodation from inverse errors with 0.25 D fixation-mean soft anchors; objective 8.80562 px², with 965 lower-bound A frames and 47 inverse-domain-limited frames. | Framewise inverse RMS C1–C4 was 1.184 / 4.149 / 1.994 / 3.010 px over all complete frames (C2 common-valid RMS 1.984 px). Fitted accommodation retained gaze/fixation patterns; inverse errors and bounds matter. |
| 05 | Capture 1 A0/G/GA state ablation; selected G/GA objectives 3.39212 / 2.49050 px². GA P1/P4 forward RMS 0.834 / 1.170 px; A means changed by at most 0.00286 D from A0. | G/GA had distinct stationary starts, and the approximately 0.39 D A contrast persisted. State/objective comparison only; not independent state accuracy. |

Flat reports and compact bundles: `archive/normalized_reverse_transform/`.

### Raw-keystone reverse-transform chain (historical baseline)

The raw chain removed post-keystone RMS/area normalization, retained one P1-derived scale shared with P4, and applied P4 radial deformation before keystone. It includes the six audited historical stages in `archive/raw_keystone/`. Raw coefficients and state estimates were recalibrated rather than copied from normalized runs.

The chain found a small fitted raw P1 size response in its restricted coefficient family (Capture 1 P1 objective 1.013 px²). Capture 1 P4 K fit objective was 2.735 px² with relative κ fixed at zero. Independent capture P4 fits produced κ = 0, −1.8542103e−6, −1.4746662e−6, and −8.4791740e−7 px⁻²; the descriptive post-fit association was −5.2532312e−7 px⁻²/D. Capture and nominal demand are confounded. Framewise A reduced the objective to 7.340934 px² and P4 forward RMS to 1.162/3.620/1.836/2.984 px by capture, but inverse RMS for Capture 2 was 5.696 px due to radial inverse amplification. In Capture 1 A0/G/GA objectives were 3.39212/2.49050 for the free G/GA candidates, with distinct stationary starts; GA P4 RMS was 1.170 px versus A0 1.162 px. The later conditional center-to-gaze step reached a six-cycle fixed point (next gaze change ≤1.48e−6°) and objective 7.228522 px²; A ranges and fitted accommodation pattern persisted. Its derived centers are gauge-defined origins, not independent physical-center measurements. These stages did not resolve the remaining gaze-conditioned accommodation pattern.

Detailed results, metrics, and limits are in `archive/raw_keystone/`. The old raw scripts and empirical reference are separately preserved in `archive/reverse_transform/raw_keystone/` for provenance; current active reusable code is in `distortion_model/`.

### Shared-keystone Revision 2: current S0 chain

The complete original stage directories are preserved in `archive/reverse_transform/shared_keystone/`.

| Stage | Status | Result |
|---|---|---|
| 01 — mean initialization | `INITIALIZED`; equations audit passes. | Frozen A targets C1–C4: 0.360360, 3.901601, 3.111942, 2.038211 D. Equal-fixation data loss 23.63822745. It is a warm start, not a joint stationarity result. |
| 02 — framewise alternation | `WARM_START_ALL_REJECTED`; 3 cycles; equations audit passes. | Data loss 23.14886764. Keep as an uncertified warm checkpoint, not a successful joint fit. |
| 03 — joint refinement | `PASS`; independent numerical/equation audit passes; seven starts agree. | Accepted S0 fit, summarized below. |
| 05 — held-out validation | `VALIDATION_RECORDED_CONDITIONAL_GEOMETRY`. | Held-point geometry on a same-recording temporal split, summarized below. It is not external validation. |

Stage 04 / S1, the accommodation-dependent P4-keystone extension, was not run. Its gate required review of S0 fixation dependence, bounds, and residual tails; a flat A pattern alone was not enough to promote it.

### S0 joint fit and diagnostics

Stage 03 objective `L=11.645956789131`: data 10.3594949491, gaze prior 0.6255623297, fixation-mean A prior 0.4431920982, optical prior 0.2177074121. Normalized projected-gradient infinity norm is `6.4691e-6`; all seven starts reached the same objective, and independent equation/numerical audits passed. The equal-fixation data loss is 56.17% below Stage 01. The shared radial slope is `s=-5.37586275e-7 px^-2 D^-1`.

Pooled native-pixel forward errors (three vertices per complete frame):

| Pattern | Stage 01 median / P95 / RMS (px) | Stage 03 median / P95 / RMS (px) |
|---|---:|---:|
| P1 | 0.686841 / 2.28312 / 2.58810 | 0.743219 / 2.07470 / 1.39652 |
| P4 | 1.56108 / 3.58416 / 3.91379 | 1.10938 / 2.17875 / 2.81791 |

P1 median worsened while its P95 and RMS improved. There were 0 measured-pattern P1 inverse failures and 13 P4 failures; all complete frames remained in the forward fit. The model has three data-only scale-gauge null modes, so the same-recording anchors select gaze and accommodation scales. Positive curvature with anchors does not mean independent data identify those scales. P4 vertical-projective `q[7]` was at its lower box bound; 958 A states were at the lower bound and 13 at the upper bound. No nonlinear domain constraint was active, but nonlinear-boundary KKT certification is unavailable.

The 0.25 D A-mean prior does not pin the fitted means: their RMS departure from targets is 0.1664 D. Mean A remains fixation-dependent within captures (ranges about 0.392, 0.379, 0.629, and 0.420 D for C1–C4). All captures show lower fitted mean A at nominal +10° than at nominal zero. This repeats across recordings, but nominal gaze, capture/demand, and fixation order are confounded. Saved 0.125/0.5 D anchor-width controls trade lower training errors against larger mean departures and larger +10° contrasts; their original run status remains `UNCERTIFIED_DIAGNOSTIC` despite supplemental saved-point CPU checks. They do not select a physically correct width.

The focused Capture 1 investigation found 891 consecutive lower-bound frames with positive conditional objective derivative (856 contiguous rows 19665–20520), crossing the nominal +5°/+10° interval boundary at row 20500. Row 20000 has fitted gaze `(5.185816°, 0.267434°)`, A=0 D, and P1 scale 0.990271. The fitted model prefers a smaller A than the operational bound allows. Corrected P4 radius changes about 0.63% between the earlier +5° and bound-run groups. This is a model-bound and residual diagnostic, not evidence for physical zero accommodation or a tracking failure.

### Stage 05 held-out validation

The training split used 70,116 complete rows, 875 gap rows, and 18,184 complete test rows from the 100,090 scheduled population. The calibrated training fit had seven starts at `L=9.85074053034`; training, held-state, and retained-input noninterference checks passed. There were 54,552 scored held points. Pooled held-point error was median 2.122946 px, P95 4.906522 px, RMS 6.072948 px, maximum 270.748152 px. The large maximum is retained and is a material tail. Scope is conditional within-recording temporal geometry for this split; not another recording/subject or physiological accommodation validation. Reduced rank checks sampled 400 frames per omitted-point slot rather than all test frames. Nonlinear-boundary KKT certification remains unavailable.

### Capture 1 +5° reference sensitivity

This is a matched diagnostic comparing the S0 zero reference with arithmetic mean P1/P4 templates from Capture 1's +5° fixation (4,993 complete frames, rows 15500–20499). It retained `A_ref=0.36036036036036034 D`, the original gaze anchors/A targets/priors/bounds, one P1-derived scale, and the same nine global parameters. Three starts and independent CPU equation, stationarity, and curvature checks passed. The fit objective was `13.724246525687` (data 11.992319 versus 10.359495 for S0); data loss increased 15.76%, and pooled P1/P4 median errors worsened from 0.743219/1.109385 px to 1.317687/1.266294 px. A lower-bound contacts fell from 958 to 55; at Capture 1 row 20000, fitted A changed from 0 to 0.123489 D.

The change mostly shifts A offsets. The +10° minus zero A contrasts changed only +0.0028 to +0.0071 D across captures; the within-capture contrasts remained. Therefore +5° avoids many lower-bound contacts but does not improve observation fit or remove the main gaze/fixation pattern. It is a sensitivity result, not an improved calibration. The reference still contains unknown intrinsic and gaze-dependent distortion; it does not establish an undistorted optical zero. Full model derivation, per-fixation values, plots, saved arrays, and audits are in `archive/reverse_transform/shared_keystone/stage_03_joint_refinement/investigations/reference_plus5/`.

## Reproduction and archive map

### Independent per-capture P4 inverse-keystone fits

Applied the Capture 1 inverse-keystone protocol independently to captures 1–4, with each capture's own filtered rows, centroid-based nominal-gaze initializer, symmetric reference, and fit. The four fits converged to stable full-row 7-parameter candidates with no active bounds and full-rank scaled Jacobians. Capture 1 reproduced its saved canonical parameters and objective exactly. The comparison preserves capture-specific parameters and does not infer accommodation from nominal demand. Raw controls show substantial loss sensitivity in C2 and C4 (21.027 and 9.242 px²) while origins shift only −0.228° and +0.022° respectively; C2's small cy coefficient changes sign near zero, so component direction there is less stable. See [comparison report](archive/reverse_transform/superseded_baseline/per_capture_p4_inverse_keystone/results/RESULTS.md), [filtered parameters](archive/reverse_transform/superseded_baseline/per_capture_p4_inverse_keystone/results/parameters.csv), [raw controls](archive/reverse_transform/superseded_baseline/per_capture_p4_inverse_keystone/results/raw_control_parameters.csv), and [matched-gaze triangle plot](archive/reverse_transform/superseded_baseline/per_capture_p4_inverse_keystone/results/mean_triangles_matched_gaze.png). The exact vertical initializer diagnostic is `theta_y=b1*(delta_y-y0)` using the native unshifted x-polynomial coefficient; the horizontal model fit is unchanged.

### Fixed-keystone P4 barrel zero-gaze stage (conditional; accepted for now)

The completed zero-gaze stage at [fixed_keystone_barrel_zero_gaze](archive/reverse_transform/superseded_baseline/fixed_keystone_barrel_zero_gaze/results/RESULTS.md) is accepted for now as conditional geometry under a declared shared-reference convention: the Capture 1 frame-first inverse-mean triangle is used for all captures while each capture's fitted keystone remains frozen. It does not establish a physical barrel response. With fixed `K`, shared nuisance magnification `g=1`, and no per-frame radial coefficient or accommodation estimate, fitted `kappa` is `0`, `−1.88307e−6`, `−1.49124e−6`, and `−9.38964e−7 px⁻²` for C1–C4.

| Capture | Pooled frame inverse RMS, baseline → barrel (px) | Mean-triangle inverse RMS, baseline → barrel (px) |
|---:|---:|---:|
| C1 | 0.3744 → 0.3744 | 0.0000318 → 0.0000318 |
| C2 | 13.5654 → 3.3433 | 13.5598 → 3.3089 |
| C3 | 10.7629 → 2.4116 | 10.7255 → 2.1904 |
| C4 | 6.7675 → 1.8763 | 6.7374 → 1.7403 |

The inverse's radial mean residual falls substantially in C2–C4; tangential residual does not fall uniformly, increasing in C3. The separate uniform-size control gives a better forward fit than the barrel model in C3 and C4, while the barrel model is better in C2. Thus the result answers that the conditional barrel inverse reduces the reported triangle RMS, but it is not evidence that a common physical barrel mechanism explains all captures. The subsequent all-fixation joint comparison is complete; see below. Preserve the earlier status handoff at `docs/legacy/CURRENT_STATUS_PRE_PER_CAPTURE_P4.md`.

### All-fixation joint P4 keystone/barrel comparison with learned shared triangle

The completed comparison is at [`archive/reverse_transform/superseded_baseline/joint_keystone_barrel_learned_reference/results/RESULTS.md`](archive/reverse_transform/superseded_baseline/joint_keystone_barrel_learned_reference/results/RESULTS.md), with reproducible runner/report source in the adjacent experiment folder and independent checks in `results/independent_audit.json`. It compares fixed common `B` plus capture-specific radial coefficients, a jointly learned unconstrained common triangle plus radial coefficients, and a learned common triangle plus uniform intrinsic size. The forward objective uses equal total weight for each capture/fixation; the saved gaze, row masks, and P4 geometry filters remain fixed. Each trial recomputes its predicted centroid `mu`; `g=1`, no framewise `g`, `kappa`, or accommodation `A` is estimated, and no output normalization is applied.

All-row equal-group measured P4 MSE was 3.7649746 px² for the historical initialization, 1.3513735 px² for fixed-reference radial, 1.2450076 px² for learned-reference radial, and 1.2560811 px² for learned-reference uniform size. Learning the triangle improved the fixed-radial result by 7.87%; uniform size was 0.89% above learned radial. The all-row models use every row and their train/test subset scores are in-sample. Learned radial reduced forward error but its one-start leave-one-target fits were worse than fixed radial at both endpoint targets: 3.84615 vs 2.58783 px² at −10°, and 4.58719 vs 2.62639 px² at +10°. The endpoint contrast is a single-start diagnostic, not a stable validation result.

The learned-radial inverse frame RMS versus fitted `B` was 1.2704 px, versus 1.3279 px for fixed `B` radial and 1.1962 px for the uniform-size control. Frame-first mean-triangle RMS was 1.0551, 1.1191, and 1.0020 px respectively. These inverse metrics are diagnostics because inversion may compress residuals. Data-only rank was 15/15 for fixed `B` and 21/21 for both learned-template controls; no primary model had active bounds or observed inverse failures. All-roster inputs were filtered complete P1/P4 rows for C1–C4 (24,148 / 21,979 / 19,795 / 23,056 rows). Temporal and training-only-anchor checks remain conditional on frozen gaze initialized from the full roster. The main all-row/sensitivity cases used five starts, conditional and training-only cases three, and each LOO case one.

The report includes `B` side lengths, angles, signed area, RMS radius and V1-apex/V0–V2-base symmetry residuals; per-capture zero-target inverse mean triangles; frame-weighted inverse p50/p90/p95/p99/max; radial/tangential residuals; anchor sensitivity and profiles; and held-out tables. `independent_audit.json` confirms source hashes unchanged, exact frame-CSV reconstruction of primary data losses (maximum difference 2.2e−16 px²), CPU/GPU objective/prediction agreement to floating-point precision, and 88,978 valid synthetic roundtrips per primary branch (maximum point error 5.5e−11 px). All split rows are uniquely partitioned. Capture and nominal demand remain confounded; this is a conditional geometric comparison, not evidence for physical lens or accommodation separation.

Run commands from the repository root with `/home/aplab/.pyenv/versions/venv/bin/python` and the recorded NumPy/SciPy/CuPy float64 environment. A GPU is required for the full-batch estimator. The current scripts are:

```bash
/home/aplab/.pyenv/versions/venv/bin/python experiments/reverse_transform/shared_keystone/scripts/run.py --help
/home/aplab/.pyenv/versions/venv/bin/python experiments/reverse_transform/shared_keystone/scripts/audit.py --help
/home/aplab/.pyenv/versions/venv/bin/python experiments/reverse_transform/shared_keystone/scripts/report.py --help
```

A new run requires an empty output path and its declared stage/parent inputs. Do not overwrite archived canonical results. Inspect the archived stage `protocol.json`, `provenance.json`, `audit.json`, and `RESULTS.md` before reproducing or comparing. The +5 runner lives in the preserved investigation tree; after archival, invoke it at `experiments/archive/reverse_transform/shared_keystone/stage_03_joint_refinement/investigations/reference_plus5/run.py`. Invoke it with `/home/aplab/.pyenv/versions/venv/bin/python`; it accepts `--precheck-only`, or `--out /tmp/<new-results-dir>` for a fresh full run. Its renderer likewise accepts explicit candidate and output directories. Reproduction requires reviewed `data/` files; no data files were moved.

| Archive location | Contents |
|---|---|
| `archive/distortion_model/` | Six compact reports and JSON bundles for the earlier distortion-model stages. |
| `archive/normalized_reverse_transform/` | Five compact normalized reverse-transform stage records. |
| `archive/raw_keystone/` | Six compact raw-keystone stage reports and bundles. |
| `archive/reverse_transform/shared_keystone/stage_01_mean_initialization/` | Full S0 initialization run, audit, protocol, provenance, arrays, plots, and source snapshot. |
| `archive/reverse_transform/shared_keystone/stage_02_framewise_alternation/` | Full uncertified warm checkpoint. |
| `archive/reverse_transform/shared_keystone/stage_03_joint_refinement/` | Canonical fit, controls, investigations (including +5), plots, audits, and source snapshots. |
| `archive/reverse_transform/shared_keystone/stage_05_validation/` | Canonical held-out validation and original attempt/audit evidence. |
| `archive/reverse_transform/raw_keystone/` | Preserved raw-chain active family code and empirical reference, separate from its compact historical reports. |
| `archive/manifests/` | Original cleanup and consolidation manifests, moved byte-for-byte. |

The live `distortion_model/` package and shared-keystone scripts are the reusable implementation. `experiments/README.md` points here. The archive directory is ignored by the current `.gitignore`, so these archive files remain workspace-local unless repository policy is changed later; no files were staged or committed. The archived full run trees are evidence and reproducibility inputs; the compact family reports above are historical summaries. No fit was rerun as part of this consolidation.

### P1 joint keystone and model-relative scale audit (Stage 00)

The completed report and durable row/model outputs are in [`p1_joint_keystone_magnification/results/RESULTS.md`](archive/reverse_transform/superseded_baseline/p1_joint_keystone_magnification/results/RESULTS.md). This P1-only analysis fits shared C1 keystone/template parameters and positive per-frame `m`, with fixed-template control, conditional temporal holdout, and frozen C1 transfer to C2–C4. It does not fit P4 or estimate accommodation.

The wide-offset-envelope learned C1 candidate reaches fitted nuisance offset `theta01=104.2035°`, loss `0.63710765 px²`, rank `7/7`, and unit-scaled condition number about `40,485`; its native `m` spans `0.2379–0.4936`. The ±30° offset-envelope candidate has loss `0.72208095 px²` and `m=0.9757–1.0002`; the relative-by-fixation profile also shifts across these envelopes. The offset values are relative to the saved gaze coordinate and have no physical-angle meaning. Thus full rank does not make `m` physically identified. The P1-to-P4 scale gate is unmet, and this calibration is not transferred to P4.

Conditional C1 temporal-test equal-fixation forward RMS is 0.76788 px for learned-train versus 0.92795 px fixed-train; identity is 1.04851 px and similarity rotation is 0.91446 px (with an extra per-frame degree of freedom). Frozen learned-C1 transfer RMS for C2/C3/C4 is 1.25312/1.12963/0.58394 px versus identity 1.41691/1.20390/0.72272 px. Similarity controls are lower for C3/C4 but add per-frame flexibility. All remain conditional geometry checks, not independent magnification validation.

Profile audit uses saved profile parameters and independent analytic nuisance-gradient recomputation: learned-all has 19/33 numerically valid points, 14 stationary and 5 nonstationary; 14 points are invalid or unsolved. Fixed-all has 29/29 valid stationary points. The profile gaps and nonstationarity are retained and do not justify treating the plotted profile as a confidence interval. Synthetic equation checks pass both the 20-row runner suite and the core 91-frame planted fixture; empirical fit quality does not establish physical scale. `diagnostics.json` records 38/38 source/artifact hashes, the report source SHA-256, and independent CSV/NPZ reconciliation with zero maximum row prediction, scale, and residual-RMS differences.

The positive-only result alone does not support P4 transfer. The bounded sibling below supplies a candidate conditional sensitivity after reviewing its bound and train-only audits. Its matched full-model P4 refit is complete and reported above; a comparison of any different scale convention would still require its own matched refit and cannot establish physical scale. Independent constraints or improved measurements are still needed before physical claims. Do not claim accommodation success from either P1 analysis.

### Bounded P1 scale sensitivity (Stage 00)

[`p1_joint_bounded_magnification/results/RESULTS.md`](archive/reverse_transform/superseded_baseline/p1_joint_bounded_magnification/results/RESULTS.md) adds a user-specified per-frame scale interval around unity: `[0.98,1.02]` primary, `[0.99,1.01]` and `[0.95,1.05]` sensitivities. Learned and fixed templates were fit at all three widths; primary-width train-only fits score C1 temporal splits conditionally, and primary all-C1 fits transfer to C2–C4. This P1-only fit does not itself fit P4; the separate matched P4 refit is documented in the adjacent `p4_p1_scale_effect` results.

For the learned primary fit, C1 equal-fixation RMS is 0.8328 px, fitted P1 nuisance offset `theta01=42.40°`, the fitted bounded `m` range is 0.9800–0.99495, and 2.83% of C1 rows are at the lower bound (none at the upper bound). The fitted P1 offset changes from 28.85° at ±1% to 54.06° at ±5%; the fixed-template offset remains near −5.53°. These are fitted parameters relative to the saved gaze coordinate, not physical angles. The fixed-template `m` spans 0.9843–1.0048. All eight chosen starts are stationary and valid, no cheaper nonstationary start was found, rank is 7/7 for learned primary, no row is invalid, scalar KKT residual is at most 2.11e−10, and independent objective recomputation agrees within 2.22e−16. The original positive-only learned control had `m=0.2379–0.4936` under a wide offset envelope; excluding those values is a consequence of the explicit interval, not independent evidence that the bounded values are physical. Most primary learned scales are interior yet below 1, and widening the interval lowers loss and changes the fitted offset. These results characterize a bounded model-relative calibration and its conditional P1 geometry. The subsequent matched P4 refit is complete and finds that bounded P1 `m` worsens P4 fit under the tested model; the working P4 primary therefore remains `g=1`. This does not resolve physical scale. The saved arrays, row CSV, parameters, multistarts, source hashes, four plots, `run.py`, and `report.py` preserve the complete result.

### Fixed-zero P1 offset sensitivity (Stage 00)

[`p1_zero_origin_comparison/results/RESULTS.md`](archive/reverse_transform/superseded_baseline/p1_zero_origin_comparison/results/RESULTS.md) is a sensitivity constraining P1 nuisance offset `theta01=0°`, retaining `m∈[0.98,1.02]`, and comparing with the active free-offset bounded all-C1 fits. The active protocol fits one free `theta01` shared across P1 frames and plans a separate freely fitted `theta04` shared across P4 fit frames/captures; they are not constrained equal and neither has absolute physical-angle meaning. On C1, equal-fixation RMS increases from 0.83277 to 0.94457 px (+0.11180 px, +13.43%) for learned template and from 0.96874 to 1.02328 px (+0.05454 px, +5.63%) for fixed template. Pooled RMS increases 13.29% and 5.65%, respectively. All 24,148 C1 rows are valid; fixed-offset learned `m` spans 0.98445–1.00492 with no bound contacts, while the free-offset learned fit spans 0.98000–0.99495 with 2.83% at the lower bound.

On matched C1 rows, learned `m`/actual-gaze correlation changes from +0.519 at free offset to −0.587 under the fixed-zero constraint; `m_zero−m_free` has RMS 0.01427 and its fixation-average declines from +0.02049 at F0 to +0.00778 at F4. Fixed-template scales remain nearly unchanged (rowwise correlation 0.999987, difference RMS 0.0000995), though the fit residual rises. Frozen C1 transfer effects vary by capture: the constrained learned model raises RMS for C2–C4 by 0.00831/0.02612/0.06814 px; the fixed-template constraint improves C3 by 0.07161 px and is slightly worse on other captures. All fixed-offset all-C1 and train fits are nuisance-stationary with full conditional nuisance rank and no invalid rows. The same-row profiled scales make these conditional geometry comparisons, not independent scale validation. The experiment measures sensitivity to constraining an arbitrary nuisance offset; it does not determine a physical angle or magnification. This P1-only offset sensitivity itself has no P4 fit; the separate matched P4 fit is documented above.

Under a common rebase of saved gaze, shifting `theta_i`, `theta01`, and `theta04` by the same constant leaves the relative model arguments unchanged. Changing `theta01` alone at fixed saved gaze changes the P1 model and produces the reported RMS and scale differences. The completed Stage 00 P4 fit used separate free offsets without forcing equality; `theta04` shifted −44.886° at `g=1` to −12.504° under P1 `m`, showing a substantial nuisance/scale/template tradeoff. Further P4 accommodation models still require held-out gaze and D-equivalent leakage analysis. `distortion_model/p4_joint.py` retains frozen per-capture offsets and does not implement the shared free `theta04` used in the matched P4 comparison.
