# Current status and handoff

Updated: 2026-10-10. Repository: `/home/aplab/ACC`.

## Active work: reverse-transform pipeline

The active empirical pipeline is stages 01–05 under `experiments/reverse_transform/`. Stage 05 is **COMPLETE** and its independent saved-results audit is **PASS**. Stage 04 remains the frozen accommodation-only parent for this Capture 1 ablation. Its conditional, same-recording fit reduces corresponding-vertex residuals when a framewise accommodation-like parameter is allowed. It does not establish physical optical calibration, framewise physiological accommodation, or out-of-sample accuracy.

### Goal, data, and correspondence

The goal is to recover each measured P1/P4 triangle to one fixed empirical reference by inverting the fitted gaze, keystone, and relative radial transformations. The score compares the same corresponding vertices in native camera x/y coordinates; it does not make the reference equilateral, discard absolute size, or rotate-align recovered triangles. Gaze calibration uses P4-minus-P1 centroid displacement. The P1 model uses one frame magnification `M_i` with its normalized keystone shape `K`; that one scale absorbs overall size, including size changes that the unnormalized keystone would otherwise introduce. No second P1 scale is fitted.

The reviewed loader in [distortion_model/data.py](distortion_model/data.py) admits exactly captures 1–4 and five nominal horizontal targets (−10°, −5°, 0°, +5°, +10°) per capture. It applies the detector P4 correspondence permutation `[2, 1, 0]` once when loading the raw arrays. The saved populations are already reordered; do not apply it again. All fits use native pixels and source row/frame identity; row order is not a timestamp, and the recording has clock jumps. Capture 1 contains 24,700 scheduled rows; captures 2–4 contain 24,890, 25,100, and 25,400. Across all captures that is 100,090 scheduled, 89,175 complete P1/P4 rows, and 10,915 unavailable rows.

The shared empirical reference is capture 1's nominal zero-gaze fixation, labeled 0.360360 D. Its 4,900 complete frames define the centered mean P1 and P4 triangles. Their reference RMS radii are 250.989832 px (P1) and 191.254 px (P4). The reference may retain baseline distortion; zero relative radial coefficient for capture 1 is a gauge convention, not evidence of zero physical barrel distortion.

### Stage map

| Stage | Work and result | Canonical evidence |
|---|---|---|
| [01 — capture1 P1](experiments/reverse_transform/stage_01_capture1_p1_fit/README.md) | Fresh capture1-only calibration on five reviewed intervals. Fits a quadratic horizontal centroid-to-gaze map; vertical gaze uses the shared first-order x/y slope assumption because no independent vertical targets were measured. Fits four shared 2D keystone shape coefficients and one positive profiled `M_i` per frame. 24,280 complete rows; 420 unavailable. No old model, state, reference, or covariance is loaded. | [Results](experiments/reverse_transform/stage_01_capture1_p1_fit/results/attempt_01/RESULTS.md), [audit](experiments/reverse_transform/stage_01_capture1_p1_fit/results/attempt_01/independent_audit.json) |
| [02 — capture1 P4](experiments/reverse_transform/stage_02_capture1_p4_fit/README.md) | Reuses stage01 gaze and `M_i`; fits four shared P4 keystone coefficients with the P4 scale fixed to P1 `M_i`. No P4 accommodation or radial increment is fit; κ=0 is the capture1 reference gauge. Equal-fixation objective is 6.85869→2.93474 px²/point. Its post-inverse uniform-radius scaling is a shape diagnostic only, not part of the forward fit. | [Results](experiments/reverse_transform/stage_02_capture1_p4_fit/results/run/RESULTS.md), [audit](experiments/reverse_transform/stage_02_capture1_p4_fit/results/run/audit.json) |
| [03 — independent captures](experiments/reverse_transform/stage_03_independent_captures/README.md) | Independently fits captures 1–4 against the capture1 reference, reusing capture1 P1 fit and estimating P1 keystone for captures 2–4. For each capture, P4 fits four shared keystone coefficients and one constant relative radial coefficient with fixed P1 `M_i`; capture1's κ is fixed to zero as the reference gauge. The fitted relative κ values (captures 1–4) are 0, −1.856868e−6, −1.472843e−6, −8.477171e−7 px⁻². Forty-three capture2 radial inverse failures remain in forward fitting. Capture4’s P4 vertical quadratic coefficient is at its declared bound; shape diagnostic worsens versus P1-scale-only. The post-fit κ/demand association is confounded because each demand label occurs in one capture. | [Results](experiments/reverse_transform/stage_03_independent_captures/results/run/RESULTS.md), [audit](experiments/reverse_transform/stage_03_independent_captures/results/run/audit.json), [scientific review](experiments/reverse_transform/stage_03_independent_captures/SCIENTIFIC_REVIEW.md) |
| [04 — framewise accommodation](experiments/reverse_transform/stage_04_framewise_accommodation/README.md) | Holds stage03 gaze, keystone coefficients, P1 magnifications, reference, and linear κ(A) law fixed; estimates only one `A_i` per complete frame. This is the frozen A-only baseline for Stage05. | [Results](experiments/reverse_transform/stage_04_framewise_accommodation/RESULTS.md), [scientific review](experiments/reverse_transform/stage_04_framewise_accommodation/SCIENTIFIC_REVIEW.md), [audit](experiments/reverse_transform/stage_04_framewise_accommodation/results/run/audit.json) |
| [05 — Capture 1 state ablation](experiments/reverse_transform/stage_05_capture1_state_ablation/README.md) | Compares preserved Stage04 A-only states (A0), a horizontal-gaze-only fit (G), and a joint gaze+A fit (GA) on the same 24,280 complete Capture 1 frames. G/GA share equal P1/P4 weights; references, optics, and the κ(A) slope remain frozen. | [Results](experiments/reverse_transform/stage_05_capture1_state_ablation/RESULTS.md), [scientific review](experiments/reverse_transform/stage_05_capture1_state_ablation/SCIENTIFIC_REVIEW.md), [audit](experiments/reverse_transform/stage_05_capture1_state_ablation/results/run/audit.json) |

Stage01 also documents the native coefficient values, polynomial, magnification distributions, and sensitivity concerns. In particular, the assumed small vertical gaze span yields large native vertical coefficients; these require sensitivity analysis and are not evidence of a vertical optical mechanism.

### Frozen stage04 parent contract

The frozen post-fit radial law is

```text
κ(A) = β (A − A_ref)
β = −5.25521030136462e−7 px⁻²/D
A_ref = 0.36036036036036034 D
```

Only `A_i` changes per frame. For each of the 20 reviewed fixation periods, the arithmetic mean is over all complete frames in that period. The objective is the equal-fixation mean of per-frame mean squared distances for the three inverse-recovered corresponding vertices to the fixed reference, plus an equal-fixation mean penalty on each fixation's mean A:

```text
J = mean_fixation(data inverse-vertex MSE)
  + 16 px²/D² · mean_fixation[(mean(A_i) − nominal_demand)²]
```

The equivalent soft-anchor width is 0.25 D with residual scale 1 px. It is a penalty, not a hard ±0.25 D bound or measured uncertainty. There is no per-frame prior, fitted P4 magnification, rotation, temporal smoothing, or empirical measured-radius normalization in this fit. The deterministic model-size normalization inside the frozen stage03 keystone operator remains active. At every trial A, the runner recomputes the trial-dependent model centroid and keystone normalization before inverse mapping; it does not substitute one precomputed inverse-keystone triangle.

Operational `A` bounds are [0,6] D, with a per-frame upper bound reduced only when required to remain on the monotone inverse branch. No complete frame is dropped. The fit used CuPy 14.2.0 float64 on a Tesla P100 GPU; runtime was 6.722 s. The independent saved-results audit used NumPy on CPU.

### Stage04 parent result and numerical audit

| Capture | Complete frames | Common-valid frames | Parent inverse RMS → framewise-A RMS (px) | Original-camera RMS, all complete frames (parent → framewise A, px) |
|---:|---:|---:|---:|---:|
| 1 | 24,280 | 24,280 | 1.697981 → 1.184256 | 1.701860 → 1.185684 |
| 2 | 22,026 | 21,980 | 2.469347 → 1.984455 | 4.739406 → 3.999212 |
| 3 | 19,795 | 19,795 | 2.577231 → 1.994494 | 2.243655 → 1.774445 |
| 4 | 23,074 | 23,074 | 3.311371 → 3.010430 | 3.111584 → 2.869285 |

There are 89,175 complete fitted frames, 89,129 frames valid for all inverse comparisons, zero fitted inverse failures, 43 parent inverse failures, and 46 nominal-law inverse failures. The common-valid inverse RMS is pooled over the three 2D vertex distances and all common-valid frames; original-camera RMS uses every complete frame. On the common population, pooled inverse RMS is 2.569635 px for the parent capture-constant model, 2.605795 px for the frozen nominal-A law, and 2.140766 px for framewise A (reductions of 16.69% and 17.85% versus those baselines). These are absolute relative-vertex errors, not radius-normalized scores. The capture2 framewise inverse RMS over **all** complete frames is 4.149024 px, showing that the common-valid result does not describe its outlier-sensitive full population.

There are 965 frames at the lower A=0 bound: 918, 46, 0, and 1 for captures 1–4. No frame reaches the upper A bound; 47 individual upper bounds were restricted by the inverse domain (0, 46, 0, 1 by capture). The largest fixation-mean shift from its nominal soft anchor is +0.443709 D at capture3, −5°.

Fitted fixation-mean A in D (each cell is mean ± within-fixation standard deviation, not uncertainty of the mean; expected nominal anchors by capture are 0.360360, 4, 3, and 2 D):

| Capture | −10° | −5° | 0° | +5° | +10° |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.566 ± 0.078 | 0.592 ± 0.102 | 0.359 ± 0.060 | 0.196 ± 0.147 | 0.207 ± 0.082 |
| 2 | 3.911 ± 0.435 | 4.095 ± 0.233 | 4.010 ± 0.088 | 4.139 ± 0.138 | 3.651 ± 0.131 |
| 3 | 3.164 ± 0.101 | 3.444 ± 0.271 | 3.103 ± 0.119 | 3.017 ± 0.077 | 2.695 ± 0.206 |
| 4 | 2.069 ± 0.168 | 2.186 ± 0.144 | 2.045 ± 0.127 | 1.893 ± 0.155 | 1.686 ± 0.128 |

The objective is 8.805618 px² (inverse data MSE 8.178932 px² plus mean-anchor penalty 0.626686 px²). Independent CPU saved-results audit passed: projected-gradient infinity norm 1.2548e−11, minimum free-frame curvature 25.212 px²/D², maximum all-frame gradient relative error 6.6162e−8, synthetic forward/inverse closure 1.7053e−12 px, and frozen-parent inverse reproduction difference 5.97e−13 px. These numerical checks certify a stationary conditional solution, not a global or physiological result.

### Stage05 current result: Capture 1 state ablation

Stage05 compares three states on the same 24,280 complete Capture 1 frames: **A0** is the exact saved Stage04 gaze/A/M baseline (not reoptimized under the new objective); **G** frees horizontal gaze while fixing A=Aref and κ=0; **GA** frees horizontal gaze and A while preserving the frozen κ=β(A−Aref) law. Vertical gaze, references/origins, P1/P4 keystone coefficients, population, units, and correspondence are frozen. For G/GA, each trial gaze recomputes P1 shape and profiles one positive common magnification from P1, shared with P4. No P4 scale, differential gaze-size term, β refit, vertical-gaze fit, or framewise prior is added.

The forward score is the equal-fixation mean of P1 plus P4 mean-three-vertex squared residuals in centered camera coordinates, with equal P1/P4 weights. Gaze and A have arithmetic fixation-mean soft anchors of 0.5° and 0.25 D (residual scale 1 px); operational bounds are θx=[−20°,20°], A=[0,6] D. These are not framewise priors or ground truth. A0 preserves its original Stage04 inverse objective, so do not compare its raw optimizer objective with G/GA objectives. All models retain the full population and all 24,280 inverses are valid.

| Model | P1 forward median / P95 / RMS (px) | P4 forward median / P95 / RMS (px) | P1 inverse common median / P95 / RMS (px) | P4 inverse common median / P95 / RMS (px) |
|---|---:|---:|---:|---:|
| A0 | 0.675 / 2.009 / 1.017 | 0.995 / 1.920 / 1.186 | 0.674 / 2.021 / 1.022 | 0.998 / 1.932 / 1.184 |
| G | 0.548 / 1.376 / 0.800 | 1.499 / 2.780 / 1.699 | 0.546 / 1.382 / 0.802 | 1.514 / 2.779 / 1.697 |
| GA | 0.548 / 1.378 / 0.801 | 0.958 / 1.940 / 1.185 | 0.546 / 1.385 / 0.803 | 0.970 / 1.959 / 1.187 |

Inverse metrics use the identical 24,280-frame common-valid set. G improves P1 but does not recover A0's P4 accuracy. GA improves P1 and reduces P4 error relative to G; P4 forward RMS is essentially unchanged from A0 and inverse P4 RMS/P95 are slightly worse. The A fixation pattern persists after gaze is freed: A0 versus GA means (D) are −10° 0.566/0.569, −5° 0.592/0.591, 0° 0.359/0.361, +5° 0.196/0.201, +10° 0.207/0.205. The changes are at most about 0.005 D, while the approximate 0.39-D peak-to-trough pattern remains. State error bars in the plots and tables are within-fixation SD, not uncertainty of the mean.

G and GA selected objectives are 3.554731 and 2.522570 px², with multistart ranges 0.005721 and 0.005408 px². Block-Newton polishing produced stationary selected solutions after optimizer iteration-limit flags; these are not claims of a unique global optimum. The independent CPU audit passed. Median acute gaze/A response-column angles are about 87.8° (P4-only) and 88.4° (joint), so the local columns are not near-collinear. However, weak gaze response, broad state spread, operational bound hits (GA: 1,311 θ lower, 2 θ upper, 909 A lower), and distinct stationary multistart basins prevent validating framewise gaze/A truth or global identifiability. The κ/demand association remains capture-confounded; fitted A is not physiological ground truth.

Canonical Stage05 [results](experiments/reverse_transform/stage_05_capture1_state_ablation/RESULTS.md), [scientific review](experiments/reverse_transform/stage_05_capture1_state_ablation/SCIENTIFIC_REVIEW.md), [summary](experiments/reverse_transform/stage_05_capture1_state_ablation/results/run/summary.json), [audit](experiments/reverse_transform/stage_05_capture1_state_ablation/results/run/audit.json), [per-frame arrays](experiments/reverse_transform/stage_05_capture1_state_ablation/results/run/frames.npz), and [plots](experiments/reverse_transform/stage_05_capture1_state_ablation/RESULTS.md#figures). The next controlled candidate is a differential P4/P1 gaze-size term such as `s41(θx)=1+c1 θx+c2 θx²`; it is not implemented. Held-P4 validation remains a later step.

### Stage04 interpretation and later work

All four +10° fixation means are below their nominal anchors. For example, capture1 means at −5° and +10° are 0.592 and 0.207 D; capture3 means are 3.444 and 2.695 D. The framewise parameter can absorb residual gaze-conditioned size effects. The κ law and demand association are capture-confounded, capture4 inherits a bounded stage03 `k_ay`, and A has no independent framewise ground truth. Do not interpret all fitted A variation as physiological accommodation or as measured physical barrel distortion. Stage05’s Capture 1 state ablation is summarized above. A differential P4/P1 gaze-size term remains a candidate for a separately declared experiment; it is not implemented in Stage05. Held-P4 validation is later work after credible state/model separation.

### Current files and reproduction

- Code: [reviewed data loader](distortion_model/data.py), [P1 shape model](distortion_model/p1_shape.py), [capture shape model](distortion_model/capture_shape.py), [framewise accommodation model](distortion_model/frame_accommodation.py), [joint gaze/A model](distortion_model/joint_state.py), and each stage's `scripts/` directory.
- Canonical stage04 parent outputs: [summary](experiments/reverse_transform/stage_04_framewise_accommodation/results/run/summary.json), [audit](experiments/reverse_transform/stage_04_framewise_accommodation/results/run/audit.json), [per-frame results](experiments/reverse_transform/stage_04_framewise_accommodation/results/run/frames.npz), [figures](experiments/reverse_transform/stage_04_framewise_accommodation/RESULTS.md).
- The stage01–04 fitted artifacts and inputs are hash-frozen parents for the later stages. Preserve them; do not overwrite canonical outputs. The previous incorrect stage02 experiment/results were removed; the corrected stage02 canonical run is the one linked above.
- Run a new replay into a fresh output directory; use the available project interpreter `/home/aplab/.pyenv/versions/venv/bin/python`. Stage05 reproduction commands are in its [README](experiments/reverse_transform/stage_05_capture1_state_ablation/README.md). Stage04 remains replayable with:

```bash
/home/aplab/.pyenv/versions/venv/bin/python experiments/reverse_transform/stage_04_framewise_accommodation/scripts/run.py \
  --output experiments/reverse_transform/stage_04_framewise_accommodation/results/replay \
  --anchor-width 0.25
/home/aplab/.pyenv/versions/venv/bin/python experiments/reverse_transform/stage_04_framewise_accommodation/scripts/audit.py \
  --results experiments/reverse_transform/stage_04_framewise_accommodation/results/replay
/home/aplab/.pyenv/versions/venv/bin/python experiments/reverse_transform/stage_04_framewise_accommodation/scripts/plot_results.py \
  --results experiments/reverse_transform/stage_04_framewise_accommodation/results/replay
```

No automated test suite was added or run. The run, independent audit, and saved-artifact review are the current verification. This handoff update is local; no commit or push was requested or performed this turn. Preserve unrelated working-tree changes.

### Standing workflow preferences

- Root owns implementation and scientific audit; mechanical tasks go to GPT-6-luna at low reasoning effort.
- Run shell commands outside the sandbox using `sandbox_permissions=require_escalated`.
- Prefer vectorized CuPy for appropriate GPU work. Keep scripts and results organized by stage.
- Do not add or run automated test suites unless requested.
- Preserve unrelated working-tree changes. No commit or push has been requested for the current work.

## Historical background: joint distortion-model estimator

This is a separate archived `experiments/distortion_model` pipeline. Its old covariance, area normalization, one-degree accommodation/gaze constraint, model domains, and next actions do not define the current empirical reverse-transform contract above. Preserve its results as history; do not resume its next action as part of Stage05.

The historical Stage6 decision was **STOP_UNCERTIFIED_ADJACENT**: control G8 passed with limits, adjacent G8 was not run, and paired optical comparison remained incomplete. The reviewed records are the [Stage6 scientific review](experiments/distortion_model/stage_06_reference_sensitivity/results/attempt_03/SCIENTIFIC_REVIEW.md), [verification](experiments/distortion_model/stage_06_reference_sensitivity/results/attempt_03/verification.json), [control G8 independent audit](experiments/distortion_model/stage_06_reference_sensitivity/results/attempt_03/g8_control/independent_audit.json), and [Stage5 G8 report](experiments/distortion_model/stage_05_crosscheck_and_model_decision/results/attempt_01/REPORT.md). The historical [Stage4 G7 attempt04 checkpoint](experiments/distortion_model/stage_04_center_and_dm0_calibration/results/g7_attempt_04/checkpoint.json) and related artifacts remain preserved. They do not change the reverse-transform status or authorize a new stage.
