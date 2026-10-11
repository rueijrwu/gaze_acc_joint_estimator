# Experiment record

Updated 2026-10-11. This is the standalone record of the distortion-model and reverse-transform investigations. Full archived reports, fit arrays, audits, scripts, source snapshots, and plots remain in the archive paths listed below. The active reusable estimator code remains in `distortion_model/`; the shared-keystone run, audit, pipeline, reporting, and roundoff scripts remain in `experiments/reverse_transform/shared_keystone/scripts/`.

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
