# Corrected barrel-first P4 pipeline

The requested P4 experiment now uses the corrected order: calibrate per-gaze relative barrel from scale-corrected P4 triangles, invert barrel before fitting capturewise keystone, then jointly refine the shared P4 reference, optics, gaze, and accommodation states. The same P1-derived frame magnification `g` is applied to P1 and P4, as required by the ray-tracing model. The old P4 family is preserved at [`experiments/archive/reverse_transform/superseded_p1_scale_capturewise_p4/`](experiments/archive/reverse_transform/superseded_p1_scale_capturewise_p4/) and is superseded for this question. Its different references and priors mean its comparisons cannot isolate the effect of reversing the P4 order.

## Stage 05 anchor sensitivity update

The completed sensitivity sweep retains the original corrected `.25 D` fixation-mean anchor and optical-prior width 1× fit as the reference. No sensitivity candidate replaces that fit or establishes a physiological accommodation scale. The exact shared physical `g`, corrected barrel-first ordering, and fresh Stage 01–04 calibration inputs are unchanged.

| Stage 05 case | P4 forward / inverse weighted RMS, full roster (px) | Held joint weighted RMS (px) | A lower-bound contacts |
|---|---:|---:|---:|
| Active baseline: fixation .25 D, optical 1× | 1.05974 / 1.26201 | 1.87097 | 5 |
| Fixation .25 D, optical 2× | 1.05334 / 1.26359 | 1.85164 | 3 |
| Fixation .50 D, optical 1× | 0.83149 / 0.99032 | 1.66299 | 8 |
| **Fixation .50 D, optical 2×** | **0.81874 / 0.98447** | **1.64055** | **3** |
| Capture .25 D, optical 1× | 0.75451 / 0.85430 | 1.60111 | 5,139 |
| Capture .50 D, optical 1× | 0.72602 / 0.82919 | 1.58353 | 2,917 |

The `.50 D`, optical 2× fixation-scope fit has the best held geometry among the bounded fixation-scope variants. Its held aggregate still has cell regressions: relative to the active baseline it is worse by 0.202 px at C1 +10° and 0.221 px at C3 0°. The capture-scope fits lower the aggregate held RMS further, but their 5,139 and 2,917 A lower-bound contacts are all in C1; the `.50 D` capture case is worse than baseline by 0.691 px at C1 +10° and 0.469 px at C3 0°. These remain conditional state-model results rather than physical accommodation measurements. All six cases passed held-state gates with 18,139 test rows each and zero invalid rows; all twelve production/training fits had two certified stationary starts.

The [sensitivity record](experiments/reverse_transform/p1_scale_barrel_first_sensitivity/RECORD.md), [comparison report](experiments/reverse_transform/p1_scale_barrel_first_sensitivity/stage_05_joint_states/results/sensitivity_comparison/REPORT.md), and [independent audit](experiments/reverse_transform/p1_scale_barrel_first_sensitivity/stage_05_joint_states/results/sensitivity_comparison/AUDIT.md) give full per-cell metrics and provenance. The pre-sensitivity status and plan are preserved in [`docs/legacy/CURRENT_STATUS_PRE_SENSITIVITY.md`](docs/legacy/CURRENT_STATUS_PRE_SENSITIVITY.md) and [`docs/legacy/CURRENT_PLAN_PRE_SENSITIVITY.md`](docs/legacy/CURRENT_PLAN_PRE_SENSITIVITY.md).

The data adapter uses 88,978 reviewed, matched P1/P4 frames from four captures and five fixation intervals per capture. Nominal gaze labels are −10°, −5°, 0°, +5°, +10°; capture demands are 0.36036036, 4, 3, and 2 D. Raw detections are [Capture 1](data/detections/capture_1_detections.pkl), [Capture 2](data/detections/capture_2_detections.pkl), [Capture 3](data/detections/capture_3_detections.pkl), and [Capture 4](data/detections/capture_4_detections.pkl). The frozen roster is [baseline_inputs.npz](experiments/reverse_transform/p4_gaze_accommodation_decoupling/stage_00_baseline_sensitivity/p4_p1_scale_effect/baseline_inputs.npz) with its [JSON hash manifest](experiments/reverse_transform/p4_gaze_accommodation_decoupling/stage_00_baseline_sensitivity/p4_p1_scale_effect/baseline_inputs.json). Exact `(capture, source_row)` identities and raw source hashes are preserved in Stage 01 arrays/provenance.

## Model and calibration

For frame `t` and vertex `v`, Stage 04/05 use

```text
phi_t = (theta_t - offset4) / 10
K4_c(p, theta_t) = diag(exp(b_c*phi_t^2), exp(-b_c*phi_t^2)) * p
                   / (1 + phi_t*(cx_c*p_x + cy_c*p_y)/R_c)
Z_tv = D_kappa(K4_c(B_v, theta_t))
mu_t = mean_v(Z_tv)
Xhat4_tv = g_t * (Z_tv - mu_t)
```

The forward radial map is `D_kappa(Y)=(1+kappa*||Y||^2)Y`. Inversion restores the trial centroid, applies `D_kappa^-1(X4/g + mu_t)`, then applies `K4_c^-1`. The shared offset is a nuisance coordinate in degrees, not a physical gaze origin; the fitted model uses horizontal gaze. Vertical gaze uses the horizontal initializer's first-order coefficient only as a diagnostic. There is no separate P4 scale or output normalization.

Stage 01 initializes horizontal gaze from capturewise fixation-mean P4-minus-P1 centroid displacement. Stage 02 supplies frozen P1 globals and the exact shared frame `g` from its P1-only profile with bounds [0.98, 1.02]; its evaluated values range from 0.98 to 1.00525 with mean 0.99158. At each fixation gaze, Stage 03 fits four captures' scale-corrected triangle means to a latent reference with a finite 1 px native-pixel reference width. The five C1 coefficients are fixed to zero as relative gauges, one at each gaze. The resulting equal-cell regression gives fresh `s0 = -5.12170816e-7 px^-2/D` and 20 frozen fixation-mean A targets. Stage 05 uses the existing 0.25 D fixation-mean A anchor and 0.5° frame-gaze anchor; its optical soft prior uses only fresh Stage 04 K/offset and fresh Stage 03 slope with widths 0.05 per K coefficient, 10° for the offset and `|s0|` for slope. There is no additional κ prior. These targets and priors are empirical within-recording regularizers, not physiological measurements.

Stage 04 initializes a fresh shared P4 template from the corrected C1 zero-gaze mean and fits capturewise K plus a shared offset to native pixels. Stage 05 jointly refines that template, capturewise K, offset, radial slope and frame states. The Stage 04 and Stage 05 template centroid and centered energy are fixed gauges; the three template chart coordinates have numerical bounds ±0.75. K bounds are `b∈[-0.25,0.25]`, `cx,cy∈[-0.2,0.2]`, the shared offset is bounded to ±120°, and `s/s0∈[0.05,3]`. The Stage 05 offset is −24.7785° and the fitted slope is −5.723895e-7 px^-2/D; the offset is a nuisance parameter, not a physical angle.

## Results

All three starts certified stationary solutions for Stages 03, 04, and 05. Stage 03 data-only rank is 7/7 at each gaze and Stage 04 rank is 16/16. The selected Stage 05 solution has zero invalid rows, five lower-A contacts, no global or gaze bound contacts, and a 16/17 data-only profiled optical rank. The missing rank direction is the slope/accommodation scale ambiguity: anchors select its units, while the observations alone do not identify it. Stage 05 frame states are bounded by the initialized gaze ±5° and A∈[0,6] D.

| Fit | P1 forward RMS (px) | P1 inverse RMS (px) | P4 forward RMS (px) | P4 inverse RMS (px) |
|---|---:|---:|---:|---:|
| Stage 04 fresh keystone | — | — | 1.640334 | 1.889480 |
| Stage 05 joint fit | 0.934843 | 0.859512 | 1.059744 | 1.262006 |

RMS is in native pixels: the weighted vertex RMS takes the square root after equally averaging squared Euclidean vertex error within each of the 20 capture/fixation groups; pooled RMS uses all reviewed rows. These are conditional model residuals, not independent accuracy estimates. The final capturewise fixation-mean A values (D) retain gaze-associated variation:

| Capture | −10° | −5° | 0° | +5° | +10° | Span (D) |
|---|---:|---:|---:|---:|---:|---:|
| C1 | 0.553454 | 0.568266 | 0.425286 | 0.336917 | 0.363668 | 0.231350 |
| C2 | 3.619975 | 3.872797 | 3.992377 | 4.399512 | 3.655206 | 0.779537 |
| C3 | 3.039607 | 3.500490 | 3.194995 | 3.256291 | 2.719788 | 0.780703 |
| C4 | 1.877697 | 2.061752 | 2.110608 | 2.076911 | 1.736668 | 0.373940 |

The large C2/C3 spans show that gaze-associated A structure remains. The per-gaze targets are derived from the same recording and can inherit relative keystone or template errors; they do not independently establish physical accommodation.

## Temporal held-vertex validation

The temporal split uses the first 75% of original source rows in each fixation for training, the next 5% as a gap, and the final 20% for test. The model fits all calibration and P4 parameters on 66,268 training frames, leaves a 4,571-row gap, and scores 18,139 test frames (54,417 held-vertex scores). At inference, the P4 state uses only the two retained raw vertices and their centroid; each omitted raw vertex is scored against that centroid. The training-learned optics, targets, and template are frozen. All six held-state fits passed their stationarity gate; there were zero invalid rows.

| Model | Weighted held RMS (px) | Pooled held RMS (px) |
|---|---:|---:|
| Initial | 2.487944 | 2.447092 |
| Fixed optics | 2.108794 | 2.083004 |
| Joint optics and state | 1.870972 | 1.839400 |

An independent audit recomputed the held scores and source identities. The score arrays reproduce to floating-point precision (maximum reconstruction difference 1.71e-13 px). This is same-recording conditional geometry validation; it is not an independent recording, physical gaze measurement, or physiological accommodation validation.

## Artifacts and reproduction

The active source and results are under [`experiments/reverse_transform/p1_scale_barrel_first_p4/`](experiments/reverse_transform/p1_scale_barrel_first_p4/). The [`comparison report`](experiments/reverse_transform/p1_scale_barrel_first_p4/stage_05_joint_states/results/comparison_report/REPORT.md) includes the full 20-cell target table, per-fixation residuals, A values and plots; the [`independent audit`](experiments/reverse_transform/p1_scale_barrel_first_p4/stage_05_joint_states/results/comparison_report/AUDIT.md) records source-boundary, identity, certificate, and held-score checks. The gaze/A plot is [`stage05_fixation_mean_A_vs_actual_gaze.png`](experiments/reverse_transform/p1_scale_barrel_first_p4/stage_05_joint_states/results/comparison_report/stage05_fixation_mean_A_vs_actual_gaze.png), and the source-frame accommodation trace is [`stage05_accommodation_by_original_frame.png`](experiments/reverse_transform/p1_scale_barrel_first_p4/stage_05_joint_states/results/comparison_report/stage05_accommodation_by_original_frame.png). Temporal scores and certificates are in [`validation/temporal/primary/summary.json`](experiments/reverse_transform/p1_scale_barrel_first_p4/stage_05_joint_states/results/validation/temporal/primary/summary.json). Stage 03–05 also contain the shared [`IMPLEMENTATION_AUDIT.json`](experiments/reverse_transform/p1_scale_barrel_first_p4/IMPLEMENTATION_AUDIT.json); exact hashes for the archived previous family are in its [`ARCHIVE_MANIFEST.json`](experiments/archive/reverse_transform/superseded_p1_scale_capturewise_p4/ARCHIVE_MANIFEST.json).

Production was run with three starts using CuPy float64 and one BLAS thread:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/aplab/.pyenv/versions/venv/bin/python \
  experiments/reverse_transform/p1_scale_barrel_first_p4/run.py \
  --backend cupy --starts 3 --calibration-maxiter 700 --maxiter 150 --state-maxiter 400
```

The matching training fit uses the same command with `--family experiments/reverse_transform/p1_scale_barrel_first_p4/validation_temporal`; held scoring uses `validation.py --backend cupy --maxiter 400`. Completed outputs are immutable; use a fresh family/output copy with the read-only Stage 01/02 inputs for any rerun. Stage arrays, source snapshots, protocols, checksums, row identities, and imported Stage 01/02 calibration hashes are saved with the results. [`PLAN_AT_RUN.md`](experiments/reverse_transform/p1_scale_barrel_first_p4/PLAN_AT_RUN.md) preserves the exact run-time plan specification (SHA-256 `1ceee17fb16fdca618b8aaf907c673f40e13b78a1e00c23ca62808b2a89aa36d`). The full protocol is in [`CURRENT_PLAN.md`](CURRENT_PLAN.md); earlier status and plans remain in [`docs/legacy/`](docs/legacy/).
