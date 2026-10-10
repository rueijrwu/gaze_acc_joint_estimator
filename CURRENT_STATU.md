# Current status and handoff

Updated 2026-10-10. Repository `/home/aplab/ACC`, branch `exp5_distortion_model`.

## Latest status

Raw Stage 06 is COMPLETE with a PASS CPU audit. The current workflow derives centers per frame, recalibrates gaze from five arithmetic fixation means, recomputes P1 magnification, and then fits accommodation. The user reports a clear visual improvement in the four-capture traces. The measured incremental improvement over raw Stage 04 is smaller: pooled P4 forward median/P95 is 1.255/3.079 → 1.221/2.996 px; fixation-mean A changes by at most .00613 D. Use Stage 06 as the current baseline for subsequent work; preserve its frozen reference, one shared P1 scale, raw keystone, and .25 D fixation-mean anchors. Remaining work is to understand residual fixation dependence and Capture 2 inverse-error tails. See [Stage 06 results](experiments/reverse_transform/raw_keystone/stage_06_framewise_centers/RESULTS.md) and the [four-capture accommodation plot](experiments/reverse_transform/raw_keystone/stage_06_framewise_centers/results/run/accommodation_by_source_order.png). No new implementation is scheduled.

## Active work

The active empirical estimator is the new [raw-keystone chain](experiments/reverse_transform/raw_keystone/README.md), following [REVERSE_TRANSFORM_NEXT_STEP.md](docs/REVERSE_TRANSFORM_NEXT_STEP.md). Historical normalized reverse-transform stages 01–05 remain preserved as comparison controls and supply no raw calibration states or optical coefficients.

The goal is to predict measured P1/P4 triangles and inverse-recover them to one fixed empirical reference. Corresponding vertices are compared in native camera coordinates. No equilateral replacement, rotation alignment, measured-radius normalization, or separate P4 magnification is fitted.

## Data, reference, and gaze

[data.py](distortion_model/data.py) admits captures 1–4 with five nominal horizontal fixations each: −10°, −5°, 0°, +5°, +10°. It applies detector P4 correspondence permutation `[2,1,0]` once; saved populations already have this correspondence. Source rows/frames identify observations; row order is not elapsed time because recordings have clock jumps.

There are 100,090 scheduled frames, 89,175 complete frames, and 10,915 unavailable frames. Complete frames by capture are 24,280 / 22,026 / 19,795 / 23,074.

The shared reference is Capture 1's nominal zero-gaze fixation, labeled **0.36036036036036034 D**. Its 4,900 complete frames define centered arithmetic-mean P1/P4 triangles, with reference RMS radii approximately 250.989832 / 191.254146 px. Baseline distortion may remain in this empirical reference; Capture 1 κ=0 defines a relative radial origin rather than zero physical barrel distortion.

Horizontal gaze is freshly calibrated from `mean(P4) − mean(P1)` with a quadratic fit to fixation labels. Vertical gaze uses the same first-order slope as horizontal gaze; no independent vertical calibration targets exist. Stages 01–05 enforce zero mean gaze at the reference fixation. Stage 06 instead maps the arithmetic mean corrected center separation at that fixation to zero; averaging framewise quadratic outputs can differ slightly because of within-fixation variance. Nominal labels are not framewise gaze/accommodation ground truth.

## Correct model contract

The active implementation is [raw_keystone.py](distortion_model/raw_keystone.py):

```text
H1 = center(K1(gaze, B1))
g_i = <X1_i,H1_i> / <H1_i,H1_i> > 0
prediction_P1 = g_i H1_i
prediction_P4 = g_i center(K4(gaze, radial_A(B4)))
```

There is **no division by a keystone RMS or area factor**. Saved size-factor arrays are diagnostics only. One positive P1-derived scale is shared with P4; joint fitting recomputes it at every trial gaze. The inverse restores the model optical centroid and undoes actual scale/projective/radial transforms, introducing no RMS factor.

The four global K coefficients control reciprocal exponential x/y stretches and a generalized 2D projective denominator. Identity at zero gaze and exclusion of an arbitrary isotropic gaze-scale function define the P1 scale convention. P1-only residuals cannot distinguish scalar-normalized from raw K because profiling g cancels that normalization exactly. Unchanged P1 SSE is therefore expected.

Primary fitting uses equal-fixation mean squared corresponding-vertex distance in **original centered camera pixels**. Reports include median/P95/RMS, signed vertex errors, radius diagnostics, and inverse-reference errors. Inverse errors are secondary and can amplify measurement errors near the radial turning point.

## Stage map

Canonical fresh outputs are each stage's `results/run/`. Scripts, frame arrays, summaries, plots, protocol, source snapshots, provenance, and CPU audits are stage-organized.

| Raw stage | Work | Evidence |
|---|---|---|
| 01 | Fresh Capture 1 P1/gaze/reference fit, profile common g | [Stage](experiments/reverse_transform/raw_keystone/stage_01_capture1_p1/README.md), [audit](experiments/reverse_transform/raw_keystone/stage_01_capture1_p1/results/run/audit.json) |
| 02 | Capture 1 P4 K, fixed new P1 g, κ=0 | [Stage](experiments/reverse_transform/raw_keystone/stage_02_capture1_p4/README.md), [audit](experiments/reverse_transform/raw_keystone/stage_02_capture1_p4/results/run/audit.json) |
| 03 | Independent capture P1/P4 K and relative κ; post-fit κ/demand law | [Stage](experiments/reverse_transform/raw_keystone/stage_03_independent_captures/README.md), [audit](experiments/reverse_transform/raw_keystone/stage_03_independent_captures/results/run/audit.json) |
| 04 | All-capture forward framewise A, raw optics frozen, .25 D fixation-mean anchors | [Stage](experiments/reverse_transform/raw_keystone/stage_04_framewise_accommodation/README.md), [audit](experiments/reverse_transform/raw_keystone/stage_04_framewise_accommodation/results/run/audit.json) |
| 05 | Conditional Capture 1 A-only / gaze-only / joint gaze+A comparison | [Runner](experiments/reverse_transform/raw_keystone/stage_05_capture1_state_ablation/scripts/run.py), [audit](experiments/reverse_transform/raw_keystone/stage_05_capture1_state_ablation/results/run/audit.json) |
| 06 | Framewise centers, corrected gaze recalibration, shared P1 scale, and accommodation update | [Stage](experiments/reverse_transform/raw_keystone/stage_06_framewise_centers/README.md), [results](experiments/reverse_transform/raw_keystone/stage_06_framewise_centers/RESULTS.md), [CPU audit](experiments/reverse_transform/raw_keystone/stage_06_framewise_centers/results/run/audit.json) |

Stages 01–05 are COMPLETE with PASS audits. Fits use CuPy 14.2.0 float64 on Tesla P100; runtimes approximately 1.63 / 1.79 / 8.17 / 7.54 / 36.71 seconds. NumPy audits check fresh population/reference/gaze, hashes, independent raw forward equations, analytic gradients, stationarity/free curvature, positive shared scale, exact synthetic inverse closure, and retained raw size response. No automated test suite was added or run.

## Current results

P1 equal-fixation SSE is 1.013176 px² per vertex, unchanged as predicted by the profiled-scale gauge. Capture 1 recovered P4/reference radius ratios after raw K are 0.991446 / 0.990559 / 0.999984 / 1.006584 / 1.004880. Their range is 0.01602 versus historical 0.01672; substantial gaze-conditioned size variation remains.

Fresh relative κ by capture is [0, −1.8542103e−6, −1.4746662e−6, −8.4791740e−7] px⁻². The newly fitted post-fit law is:

```text
κ(A) = −5.253231191989352e−7 (A − 0.36036036036036034) px⁻².
```

Demand enters only after independent image fitting. κ is shared within each capture at Stage 03, with different κ across captures. Demand occurs in one capture per level, so its association remains capture-confounded. Per-capture K coefficients are retained and inspected; no shared physiological K(gaze,A) law is promoted solely from coefficient differences. Native units and active bounds matter when examining their trends. Capture 2's P4 vertical projective coefficient hits its lower bound; its radial coefficient is not at a bound.

Stage 04 varies only A_i. Its objective is equal-fixation forward P4 vertex MSE plus `16 px²/D² × mean_fixation[(mean(A_i) − nominal_demand)²]`. The .25 D width is a soft mean penalty with 1 px residual scale, not a hard ±.25 D bound, per-frame prior, or uncertainty estimate. A is bounded to [0,6] D and the forward optical branch. Measured inverse validity cannot drop observations or tighten fit bounds.

All 89,175 fitted P4 inverses are valid; the constant-κ parent has 46 failures in Capture 2. P4 forward RMS, parent→framewise A, is 1.639→1.162 / 4.321→3.620 / 2.233→1.836 / 3.297→2.984 px. Full fitted inverse RMS is 1.163 / 5.696 / 2.070 / 3.147 px. Capture 2 inverse amplification must not be concealed by reporting only a common-valid subset.

Capture 1 fixation A means are **0.566258 / 0.587119 / 0.362072 / 0.211490 / 0.243050 D**. Their range is .375630 D versus historical .395497 D. Raw keystone by itself does not remove this trend. Historical Stage 04 fitted inverse errors; a separate [normalized-forward control](experiments/reverse_transform/raw_keystone/stage_04_framewise_accommodation/results/normalized_forward_control/summary.json) uses the same forward objective and mean-anchor policy. Its Capture 1 range is .392959 D, so raw K reduces the matched range about 4.4%. It is comparison evidence only and never supplies a raw-chain parent.

## Interpretation and next decisions

Read the [scientific review](experiments/reverse_transform/raw_keystone/SCIENTIFIC_REVIEW.md) and stage reports. Reduced errors establish a conditional fit to these recordings, not physiological framewise accommodation, absolute optical distortion, or held-out accuracy. Raw size change is retained, but the gaze-conditioned P4 size/A trend persists.

That remaining trend triggered the completed Stage 05: preserved raw Stage 04 A-only states (A0), horizontal-gaze-only fit (G, A at reference), and joint horizontal gaze+A (GA). Optics/reference/law and vertical gaze stay fixed; P1 scale is reprofiled at every trial gaze. G/GA use equal P1/P4 forward weights and .5° gaze / .25 D A fixation-mean anchors. All 24,280 inverses are valid. GA A means are .566973 / .587151 / .362161 / .214347 / .243099 D, changing A0 by at most .002857 D. Thus freeing horizontal gaze also fails to remove the A pattern. GA gaze SD reaches 11.52° at +5°; it has 1,155 lower/two upper gaze-bound frames and 878 lower A-bound frames. Selected G/GA objectives are 3.392117 / 2.490495 px², with multiple stationary basins. See the review for certificates and conditioning; these states are not validated physiological estimates.

Stage 05 pooled P1/P4 forward RMS (px) is A0 1.017/1.162, G .804/1.646, GA .834/1.170. Thus joint fitting improves P1 but does not improve P4 over A0. GA P4 median/P95 is .928/1.937 px, versus A0 .950/1.908; P4 inverse RMS is 1.173 versus 1.163. GA joint Jacobian condition median/P95 is 33.62/85.22 and acute column angle median 86.32° (condition uses degree/diopter units). Local columns are not near-collinear, but weak gaze sensitivity, broad state spread, bounds, and stationary basins prevent global truth claims.

Any further shared A-dependent K or differential gaze-size term needs a separately declared coordinate-prediction and identifiability comparison. Native horizontal quadratic K increases with demand (−1.4327e−5 / 1.0763e−5 / 2.7607e−5 / 6.2526e−5 deg⁻² at .36 / 2 / 3 / 4 D), making it a concrete next coupling candidate. Other K terms are not consistently monotonic and vertical calibration is weak. A shared constant-versus-linear horizontal K comparison is not implemented yet; the capture-confounded coefficient trend alone does not establish a valid framewise A coupling. Do not reintroduce normalization. Held-P4 validation remains later work.

## Latest intermediate step: framewise centers and gaze recalibration

[Raw Stage 06](experiments/reverse_transform/raw_keystone/stage_06_framewise_centers/README.md) is **COMPLETE / audit PASS**. The user requested this center/gaze intermediate step before further optical coefficient changes. The deferred accommodation-dependent K comparison has not been implemented.

Each frame has its own derived origins: `C1=c1−g*mu1`, `C4=c4−g*mu4`, where mu is the mean of the uncentered transformed empirical reference. Centers are not shared across frames and are not independently released fit variables. Compute `(C4−C1)/g`, take arithmetic means within each of five fixations separately for each capture, fit horizontal quadratic gaze to these five means (zero intercept at the zero-fixation input), and use the same first-order slope for vertical gaze. Apply this mapping to every frame, reprofile one P1 magnification, share it with P4, and fit framewise A with gaze fixed and the existing .25 D fixation-mean anchors. Recompute centers and repeat to self-consistency. Reference, raw K, gaze-coordinate units, and the post-fit kappa(A) law stay frozen. No extra keystone normalization is introduced.

CuPy float64 fit converged in six cycles, approximately 52.6 seconds. All 89,175 complete observations remain, with no P1/P4 inverse failures. The next center-to-gaze update is at most 1.48e−6 degrees. The [CPU audit](experiments/reverse_transform/raw_keystone/stage_06_framewise_centers/results/run/audit.json) verifies raw equations, per-frame origins, fixation-mean calibration, sequential scale and A updates, all-frame A gradients/stationarity/bounds, translation invariance, and synthetic origin/inverse closure. This is a conditional fixed-point procedure, not a jointly optimized gaze/A/D model or a proof of unique/physical centers.

| Capture | P4 forward RMS before → after (px) | P4 inverse RMS before → after (px) |
|---|---:|---:|
| 1 | 1.162 → 1.175 | 1.163 → 1.177 |
| 2 | 3.620 → 3.594 | 5.696 → 5.965 |
| 3 | 1.836 → 1.753 | 2.070 → 1.980 |
| 4 | 2.984 → 2.985 | 3.147 → 3.141 |

Capture 1 refined A means at −10/−5/0/+5/+10 degrees are .56664/.58709/.36226/.21374/.24386 D, close to the Stage 04 values. The recalibration gives a modest forward improvement for Captures 2 and 3, but does not remove the remaining fixation-dependent accommodation pattern. Capture 2's inverse RMS worsens despite better forward median/P95/RMS; report that tail amplification explicitly. All comparisons use corresponding measured vertices, not state ground truth. Final A hits 0 D for 943 frames and 6 D for 13 frames.

Scripts, snapshots, per-cycle inputs/calibration outputs, initial/final frame states and centers, summaries, plots and reports are under Stage 06. No automated test suite was added or run. Preserve Stages 01–05 and historical normalized controls. Before further model changes, inspect the Stage 06 plots and residuals; improvement is mixed and does not justify claiming that center correction solved gaze-related accommodation bias.

## Reproduction and standing preferences

Use `/home/aplab/.pyenv/versions/venv/bin/python`. Stage 01–04 wrappers accept `--output` for fits and `--results` for audit/plot; Stage 05 has its own runner. Stage READMEs contain commands. Use fresh outputs; later stages currently read canonical `results/run` parent paths. Preserve parent/source snapshots and historical controls. Source changes require refitting/re-auditing affected stages.

Root owns implementation/scientific review; mechanical work uses GPT-6-luna at low reasoning effort. Shell commands use `sandbox_permissions=require_escalated`. Prefer vectorized CuPy and stage-organized results/scripts. User authorized committing and pushing the completed raw chain and current handoff.

## Preserved history

Normalized reverse-transform stages under `experiments/reverse_transform/stage_*` are historical controls. Their previous recommendation for differential gaze scale was superseded by raw correction first. The separate archived estimator under `experiments/distortion_model` has historical Stage 6 decision **STOP_UNCERTIFIED_ADJACENT**; its covariance, area normalization, one-degree constraint, and gate plans do not define this raw empirical chain. See its [Stage 6 scientific review](experiments/distortion_model/stage_06_reference_sensitivity/results/attempt_03/SCIENTIFIC_REVIEW.md). Preserve unrelated changes and historical artifacts.
