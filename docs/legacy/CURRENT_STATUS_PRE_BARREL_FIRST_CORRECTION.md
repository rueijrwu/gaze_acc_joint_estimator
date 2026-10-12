# Current status: shared P1 magnification and capturewise P4 inversion

Updated 2026-10-11. The active specification is [CURRENT_PLAN.md](../../CURRENT_PLAN.md). The five-stage implementation and its controls have been executed on captures 1–4. This handoff supersedes the earlier status for the archived reverse-transform experiments.

## Protocol correction in progress

The results below document the superseded P4 stage ordering and are retained as historical evidence. They do **not** implement the current requested composition, which estimates relative per-gaze barrel first, fits capturewise P4 keystone after barrel inversion, then jointly refines the states and optics. Do not treat the former Stage 03/04/05 fits or their learned-reference/barrel-first controls as results for the corrected protocol. The corrected implementation is being developed in [`experiments/reverse_transform/p1_scale_barrel_first_p4/`](../../experiments/reverse_transform/p1_scale_barrel_first_p4/); final status will be rewritten after its independent audits and results.

## Main conclusion

**Ray tracing supplies one shared physical magnification for P1 and P4, and all current fits use the existing shared frame `g`.** Learning one shared P4 reference jointly with optics and states improves P4 forward RMS from 1.42049 to 1.33365 px and temporal held-vertex RMS from 2.53661 to 2.38157 px. The gaze-associated accommodation pattern changes only modestly. The subsequent per-gaze barrel-anchor/barrel-first variants do not improve on that result and are not promoted: anchor-off is slightly worse in production and temporal held scoring, while anchor-on is worse in both. P4 distortion decoupling remains unresolved. All numerical and independent held-score audits pass. The learned reference is a model sensitivity, not a measured distortion-free template or physical accommodation estimate.

## Data and implemented protocol

The cohort contains **88,978 matched reviewed fixation frames**, with three corresponding vertices for each pattern. Raw inputs are `data/detections/capture_{1,2,3,4}_detections.pkl` and `data/fixations/fixation_intervals.json`. The loader validates detection and frozen-roster checksums and preserves vertex labels, raw positions, original source rows/frames and timestamps. The existing frozen roster comes from `p4_p1_scale_effect/baseline_inputs.npz`; archival mask files are optional cross-checks. No rows are removed by these new fits' residuals.

Captures have nominal demand labels `[0.36036036, 4, 3, 2] D` and gaze fixations `[-10,-5,0,5,10]°`. Capture and demand, and gaze and fixation order, remain confounded. The model refines **horizontal gaze only**; vertical gaze is the polynomial's first-order diagnostic.

1. **Gaze:** independently calibrate each capture from fixation means of raw `mean(P4)-mean(P1)`. Set `u=(delta_x-x0)/h`, with nominal-zero origin and fixation displacement half-range. Fit `theta_initial=a1*u+a2*u²` with equal fixation weights. Vertical diagnostic is `(a1/h)*(delta_y-y0)`. All primary polynomials are monotone over their observed input range.
2. **P1:** fit shared keystone coefficients, a separate shared P1 nuisance offset, a learned triangle and bounded frame scale across all captures. The sole scale gauge fixes template energy to the Capture 1 nominal-zero mean P1 initializer. `g` is the exact P1-only scalar least-squares profile in **[0.98,1.02]**. Save the fixed-template control as well. P4 does not select or optimize `g`.
3. **P4 keystone:** fit independent coefficient triples per capture with one shared P4 nuisance offset, using each capture's scale-corrected nominal-zero initializer and exactly the P1 frame scale. Primary loss is native inverse vertex error; forward-fit and fixed-energy learned-shape controls are saved.
4. **Barrel:** form the common native P4 reference by equally averaging Capture 1's five means of recovered frame triangles. Preserve its recovered centroid/origin. Freeze K and fit one relative barrel coefficient per capture, setting Capture 1's coefficient to zero. Recompute the trial-dependent model centroid for every barrel candidate.
5. **Joint states/optics:** freeze P1 globals and its per-frame `g`; first fit frame horizontal gaze/A with fixed optics, then refine capturewise P4 K, the shared P4 offset and shared radial slope. Initial A-mean targets come from the capture barrel coefficients and a nominal-demand regression; the targets remain frozen. Compare stronger/weaker mean anchors, nominal targets and shared P4 K.

### Exact geometry and anchors

For a template point `B`, `phi=(theta-offset)/10`, barrel is `Q=(1+kappa*||B||²)*B`, and raw keystone is `Z=(exp(b*phi²)*Qx,exp(-b*phi²)*Qy)/(1+phi*(cx*Qx+cy*Qy)/R)`. Forward measured shape is `g*(Z-mean(Z))`. Inversion divides by `g`, restores the **current model centroid**, inverts keystone, then inverts barrel on its increasing branch. Frozen native `R` fixes coefficient units. No output RMS/area/determinant normalization or additional P4 scale is introduced. The fitted P1/P4 offsets are distinct nuisance parameters, without physical-angle meaning. A distortion center is not independently measured by triangle centering.

`kappa(A)=s*(A-0.36036036)`. Gaze soft-anchor width is **0.5° per frame**. A soft-anchor width is **0.25 D on each dynamically computed fixation mean**, without an A penalty on individual frames. State limits are `theta_initial±5°`, `A∈[0,6] D`. There is no temporal smoothing. The dimensionless objective combines equally weighted P1/P4 native data error using a fixed 1-px residual scale, gaze penalty, fixation-mean A penalty, and optical penalty. Optical penalty is the mean of 14 standardized coefficient residual squares, with coefficient units `0.05`, `10°`, and `|s0|`; thus individual effective Gaussian widths are `sqrt(14)` times those units. P4 offset search expands from ±30° in initialization to ±120° in refinement; slope stays within `[0.05,3]×s0`, preserving its sign. These boxes are numerical policies.

## Fixed-reference primary calibration and results

P1 parameters `(b,cx,cy,offset)` are `(0.00135536,-0.000141701,0.0277210,-66.7712°)`; native initializer radius is **250.9898 px**. The large nuisance offset is not an inferred physical gaze origin. Frame `g` spans **0.9800–1.00525**, mean **0.99158**; **1,775 frames (1.995%)** contact the lower bound, none the upper bound. Its data-only profiled Jacobian has rank 7/7, but offset profiles show scale/template sensitivity; formal rank does not establish physical magnification accuracy.

Relative Stage 04 `kappa` values are `[0,-1.820231e-6,-1.403100e-6,-7.948167e-7] px⁻²`. Initial slope `s0=-5.078736e-7 px⁻²/D`; joint slope `s=-5.157044e-7 px⁻²/D`. P4 joint nuisance offset is **−0.843433°**. Its final coefficient triples are:

| Capture | b | cx | cy |
|---|---:|---:|---:|
| 1 | −0.00129161 | −0.03176697 | −0.00761681 |
| 2 | 0.00606387 | −0.02553424 | −0.00171340 |
| 3 | 0.00368706 | −0.02670956 | −0.00507139 |
| 4 | 0.00134026 | −0.02520095 | −0.00447240 |

The native per-capture radii and templates are retained in the arrays; compare predicted maps as well as coefficients. Independent capture K may absorb accommodation-related effects.

### Geometry metrics

RMS here is the square root of squared **Euclidean vertex distance**, averaged over vertices and frames within fixation, then equally over fixations and captures. It is separate from RMS per coordinate and pooled-frame RMS. Forward residual compares centered native measured pixels against predictions; inverse residual compares recovered native triangles against the template. Omitted-vertex scores use the vertex's position **relative to the retained two-vertex centroid**, after inference excluding that vertex. Metrics across these coordinate spaces are not interchangeable.

| Protocol/result | P1 forward RMS (px) | P4 forward RMS (px) | P4 inverse RMS (px) |
|---|---:|---:|---:|
| P1-scale Stage 04 barrel initializer | — | 1.98875 | 2.26254 |
| Learned-template P1 scale, joint primary | 0.93516 | 1.42049 | 1.60414 |
| Fixed-template P1 scale, joint control | 1.03308 | 1.19583 | 1.33442 |
| Fully recalibrated g=1, joint diagnostic | 1.4195 | 0.9614 | 1.0510 |
| Learned shared P4 reference, same frozen P1 g | 0.93513 | 1.33365 | 1.51288 |

The primary reduces its P4 forward RMS by **0.568 px (28.6%)** relative to Stage 04. The fixed-template scale improves P4 relative to the learned-template scale even while P1 agreement is poorer. The `g=1` branch recalibrates P1/K/template and all downstream stages; it is not a simple substitution into an old inverse. These comparisons support investigating scale identification, without rejecting the shared physical magnification constraint.

### Learned shared P4 reference sensitivity

A separate Stage 05 branch learned one shared P4 reference shape using a three-coordinate chart while preserving the initializer's nonzero centroid and centered energy. Native capture radii and the P1-derived shared `g` stayed frozen; the chart had no prior and no output normalization was added. The fit had 17 global coordinates, passed stationarity with zero invalid rows, found consistent minima in all 3 starts, and made no chart-bound contacts. Its data-only profiled optical rank was 16/17; the slope/accommodation scale mode remains unidentifiable without anchors. All input arrays for the frozen Stage 01–05 artifacts matched their saved SHA-256 hashes.

| Metric | Fixed P4 reference | Learned P4 reference |
|---|---:|---:|
| P1 forward RMS (px) | 0.935165 | 0.935129 |
| P1 inverse RMS (px) | 0.859866 | 0.859851 |
| P4 forward RMS (px) | 1.420488 | 1.333654 |
| P4 inverse RMS (px) | 1.604137 | 1.512885 |
| Lower-bound A contacts | 596 | 140 |

P4 geometry improves under this refit while P1 metrics are effectively unchanged. The fitted P4 offset moves from −0.843° to −8.425° as a nuisance parameter; this does not measure a physical gaze origin. Learned chart coordinates are `(0.003629, 0.002084, 0.003898)` within numerical bounds ±0.75, with no bound contacts. The largest individual template coordinate adjustment is 2.372 px; centered energy and the native centroid are preserved. The learned reference is not established as physically undistorted.

| Capture / nominal demand | Fixed-reference A means at −10, −5, 0, +5, +10° (D) | Learned-reference A means at −10, −5, 0, +5, +10° (D) | Fixation-mean span, fixed → learned (D) | +10° minus −10°, fixed → learned (D) |
|---|---|---|---:|---:|
| C1 / 0.360 D | 0.527, 0.517, 0.345, 0.219, 0.203 | 0.530, 0.526, 0.358, 0.234, 0.222 | 0.324 → 0.308 | −0.324 → −0.308 |
| C2 / 4 D | 3.955, 4.022, 3.952, 3.997, 3.720 | 3.953, 4.014, 3.944, 3.993, 3.725 | 0.303 → 0.289 | −0.236 → −0.228 |
| C3 / 3 D | 3.206, 3.337, 3.132, 3.059, 2.854 | 3.206, 3.335, 3.131, 3.064, 2.867 | 0.483 → 0.468 | −0.352 → −0.338 |
| C4 / 2 D | 2.021, 2.064, 1.960, 1.854, 1.713 | 2.021, 2.065, 1.964, 1.864, 1.731 | 0.352 → 0.335 | −0.308 → −0.290 |

The fixation-associated A patterns persist and change only modestly. In particular, C3's fixation-mean span shifts from 0.4831 to 0.4676 D. This sensitivity does not resolve geometry/accommodation leakage or support a physiological interpretation of the learned template.

### Per-gaze barrel anchors and barrel-first inversion

The completed follow-up holds P1 calibration and its frame magnification `g` fixed. Stage 04 independently estimates five C1-relative barrel coefficients per capture at the five fixation gazes, with C1 fixed to zero at each gaze as a relative gauge. The empirical fit uses a 1 px latent-reference prior; the anchor-on Stage 05 branch adds a frozen coefficient-mean penalty with width `|s0| × 0.25 D`. The full twenty-target table is in the [comparison report](../../experiments/archive/reverse_transform/superseded_p1_scale_capturewise_p4/stage_05_joint_states/results/controls/keystone_first_anchor_on/comparison_report/REPORT.md). C1's zero coefficients are a gauge convention, not evidence of zero intrinsic barrel distortion. The targets imply A spans of 0, 0.974, 0.692 and 0.401 D for C1–C4; these are target implications using baseline `s0`, not final inferred state means.

| Stage 05 branch | P4 forward RMS (px) | P4 inverse RMS (px) | Temporal held RMS (px) | A fixation-mean spans C1–C4 (D) |
|---|---:|---:|---:|---|
| Learned P4 reference (retained comparator) | 1.33365 | 1.51288 | 2.38157 | 0.308, 0.289, 0.468, 0.335 |
| Barrel-first, anchor off | 1.34085 | 1.51940 | 2.39292 | 0.309, 0.294, 0.471, 0.336 |
| Barrel-first, anchor on | 1.44402 | 1.62360 | 2.52031 | 0.178, 0.502, 0.557, 0.282 |

All geometry figures are equal-weighted over the 20 capture/fixation groups and use all 88,978 reviewed rows without residual filtering. The held score uses 18,139 test frames and 54,417 omitted-vertex scores after the original source-row 75%/5%/20% split. Both branches pass the independent held-state audit with all six certificates stationary, no invalid rows, and exact score recomputation. Temporal held RMS worsens 0.48% for anchor-off and 5.83% for anchor-on relative to the learned-reference comparator; all four capture-average scores are also worse. Anchor-on's largest cell regressions include C2 at −10° (+68.55%), C2 at −5° (+48.06%) and C3 at −10° (+39.32%) relative to the learned-reference branch. Neither candidate is promoted.

### Primary fixation-mean accommodation (D)

| Nominal gaze | Capture 1 | Capture 2 | Capture 3 | Capture 4 |
|---|---:|---:|---:|---:|
| −10° | 0.527 | 3.955 | 3.206 | 2.021 |
| −5° | 0.517 | 4.022 | 3.337 | 2.064 |
| 0° | 0.345 | 3.952 | 3.132 | 1.960 |
| +5° | 0.219 | 3.997 | 3.059 | 1.854 |
| +10° | 0.203 | 3.720 | 2.854 | 1.713 |

Frozen capture A-mean targets are `[0.36036,3.94438,3.12306,1.92535] D`, repeated across fixations. Primary A has **596 lower-bound contacts**, none at the upper bound; gaze has no state-bound contacts. The zero-A segment near Capture 1 frame 20,000 remains a constrained model output. Its physiology is not established. Capture 3's fixation-mean span is **0.483 D**. Mean-anchor widths 0.125/0.25/0.5 D produce P4 forward RMS **1.711/1.420/1.010 px** and C3 mean spans **0.155/0.483/1.020 D**: stronger anchors flatten A while worsening geometry. Nominal-target anchoring gives P4 RMS 1.436 px; shared P4 K gives 1.478 px.

## Validation and accommodation sensitivity

Temporal calibration uses the original source-row first 75% of each fixation, followed by a 5% gap and final 20% test segment: **66,268 train, 4,571 gap, 18,139 test rows**. Each P4 vertex is omitted separately, giving **54,417 held-vertex scores**. Held-vertex measurements cannot enter its inference centering, gaze anchor or P1 scale. Test gaze anchors use training fixation means; test scale comes from P1 alone. All held state fits pass their state certificates. Test optical gradients need not vanish because optical calibration is frozen.

| Temporal held-relative-vertex result | Weighted RMS (px) |
|---|---:|
| P1 scale, initialized optical prediction | 3.13417 |
| P1 scale, fixed-optics state inference | 2.64214 |
| P1 scale, refined-optics state inference | 2.53661 |
| P1 scale, shared-K refined model | 2.61994 |
| P1 scale, learned shared P4 reference, refined optics | 2.38157 |
| Fully recalibrated g=1 refined model | 1.78776 |

The fixed-reference primary's temporal held score improves **19.1%** over its initialized optical prediction. With the learned P4 reference trained using only the temporal training rows and frozen during test inference, the initial and fixed-optics scores match the fixed-reference run (3.13417 and 2.64214 px), while the refined score falls from **2.53661 to 2.38157 px (6.11%)**. All 54,417 held-vertex scores are valid, and all six state certificates pass. The held score improves overall, but 5 of 20 capture/fixation cells and 24 of 60 individual held-vertex cells regress versus the fixed-reference refined model. Examples include C1 at −10° (2.36465→2.61875 px, +10.75%), C1 at 0° (1.68077→1.74582 px, +3.87%), C3 at +5° (1.74759→1.85074 px, +5.90%), and C4 at −10° (1.41764→1.58625 px, +11.89%). The independent five-fold leave-one-fixation-out refit was not repeated with the learned reference; its previously reported scores remain the fixed-reference comparison. This conditional same-recording result does not identify a physical template or accommodation.

### Leave-one-gaze-fixation-out validation

The five-fold validation refits the gaze initializer, P1 calibration, P4 initialization, barrel reference and joint optics using the four retained gaze fixations, then scores the held fixation on temporal test rows with each P4 vertex omitted in turn. The four ordinary folds (−10°, −5°, +5°, +10°) are combined; held nominal zero is reported separately because its empirical reference/intercept convention changes. Every fold passed its state-inference gate, with zero invalid rows.

| Held fixation | Initial RMS (px) | Fixed-optics RMS (px) | Joint RMS (px) |
|---:|---:|---:|---:|
| -10° | 2.856631 | 2.651796 | 2.545820 |
| -5° | 2.189074 | 2.026582 | 1.951616 |
| +0° | 1.928061 | 1.966850 | 1.903733 |
| +5° | 3.100864 | 2.595634 | 2.557633 |
| +10° | 6.188096 | 5.021525 | 4.976408 |
| Four ordinary folds combined | 3.900650 | 3.282228 | 3.224753 |

Across the four ordinary folds, joint refinement improves the weighted score by 1.75% relative to fixed optics, and aggregate joint scores improve in all five folds. Some capture/fixation cells regress against fixed-optics inference: C4 at −10° rises from 2.657451 to 3.076025 px (+15.75%), and C2 at +10° rises from 4.015582 to 4.748579 px (+18.25%). In the separately reported held nominal-zero reference-change fold, joint improves the aggregate over initialization, while C1 and C2 worsen by 26.99% and 9.26%, respectively. The +10° fold remains the most difficult (joint RMS 4.976408 px). These are conditional same-recording scores, not independent gaze or accommodation accuracy.

Local sensitivity includes centroid derivatives. `DeltaA_equiv=<JA,X4-pred4>/<JA,JA>`, `dA/dtheta=-<JA,Jtheta>/<JA,JA>`, and `dA/dg=-<JA,H>/<JA,JA>`. These are conditional P4 sensitivities, not physiological estimation error. A nonlinear compensation check on 200 reviewed frames finds a **+0.001 scale perturbation (~0.1%) → about +0.0501 D** median A adjustment. A +0.1° gaze perturbation gives about +0.00049 D median adjustment; individual fixation sensitivities vary. This small local gaze derivative does not rule out global calibration/model error or fixation-associated A bias. No frames have weak local A support under the declared threshold.

After frame gaze refinement, P1-only reprofiling changes `g` by only **5.73e-6 RMS**, maximum **3.53e-5**; freezing the primary scale is locally consistent. Data-only optical rank after eliminating frame-state directions is **13/14**: the slope/A reciprocal scale mode is exactly unidentifiable from geometry alone. The anchors select D units. P1 offset-profile sensitivity and that gauge remain distinct uncertainty sources.

A 500-replicate bootstrap resamples 200-original-source-row blocks within reviewed fixations to assess mean calibration/reference uncertainty. P1 initializer-radius SD is **0.02594 px (0.0103%)**. Calibration and local P4-reference-to-A propagation are saved. Optical globals and g stay frozen in this bootstrap; this is not a full refitted-estimator confidence interval. Complete leave-capture/demand-out validation and full optical/bootstrap uncertainty remain future checks. No independent accommodation truth is available.

## Audit, artifacts and next decision

Float64 NumPy/CuPy forward/inverse agreement, nonzero-reference-centroid round trips, analytic derivative checks, scalar KKT checks, active-bound checks and omitted-input isolation passed. Primary joint multistarts agree at the same minimum and pass both state/global stationarity checks; primary forward and inverse outputs have zero invalid rows. Two numerical issues were repaired: scalar barrel minima were analytically gradient-polished, and optical multistarts retry smaller optimizer coordinates after rejected domain steps. Failed pre-polish temporal Stage 04 outputs are archived, and metadata/JSON compaction changes preserve numeric hashes.

**Next decision:** retain the ray-tracing supplied shared physical magnification constraint and the learned-reference branch as the best current conditional comparator. Test P4 keystone and optical-anchor compatibility with a better identified reference model. The per-gaze barrel-first variants do not improve the fit or held prediction, and the stronger per-gaze penalty changes A spans unevenly. None of these same-recording results resolves the slope/A scale ambiguity or establishes a physically distortion-free template. Keep `g=1` as a diagnostic only; do not interpret it as physical magnification or separate P1/P4 scales.

Key artifacts, all in the current experiment family:

- [Learned P4-reference comparison report](../../experiments/archive/reverse_transform/superseded_p1_scale_capturewise_p4/stage_05_joint_states/results/controls/learned_p4_reference/reference_refit_report/REPORT.md), [learned-reference held-vertex results](../../experiments/archive/reverse_transform/superseded_p1_scale_capturewise_p4/stage_05_joint_states/results/validation/temporal/learned_p4_reference/RESULTS.md), [Five-stage primary report](../../experiments/archive/reverse_transform/superseded_p1_scale_capturewise_p4/stage_05_joint_states/results/REPORT.md), [primary numerical results](../../experiments/archive/reverse_transform/superseded_p1_scale_capturewise_p4/stage_05_joint_states/results/RESULTS.md), and [fixation states/sensitivities](../../experiments/archive/reverse_transform/superseded_p1_scale_capturewise_p4/stage_05_joint_states/results/fixation_states_sensitivity.csv).
- [Per-gaze barrel-anchor/barrel-first comparison](../../experiments/archive/reverse_transform/superseded_p1_scale_capturewise_p4/stage_05_joint_states/results/controls/keystone_first_anchor_on/comparison_report/REPORT.md), [production implementation audit](../../experiments/archive/reverse_transform/superseded_p1_scale_capturewise_p4/stage_05_joint_states/results/controls/keystone_first_anchor_on/IMPLEMENTATION_AUDIT.json), [anchor-off temporal results](../../experiments/archive/reverse_transform/superseded_p1_scale_capturewise_p4/stage_05_joint_states/results/validation/temporal/keystone_first_anchor_off/RESULTS.md), and [anchor-on temporal results](../../experiments/archive/reverse_transform/superseded_p1_scale_capturewise_p4/stage_05_joint_states/results/validation/temporal/keystone_first_anchor_on/RESULTS.md).
- [Accommodation over original source frames](../../experiments/archive/reverse_transform/superseded_p1_scale_capturewise_p4/stage_05_joint_states/results/accommodation_by_source_row.png) and [corrected gaze](../../experiments/archive/reverse_transform/superseded_p1_scale_capturewise_p4/stage_05_joint_states/results/corrected_gaze_by_source_row.png), restricted to the reviewed fixation cohort, with gaps preserved.
- [Recalibrated g=1 report](../../experiments/archive/reverse_transform/superseded_p1_scale_capturewise_p4/stage_05_joint_states/results/controls/g1/REPORT.md) and [fixed-template P1 scale control](../../experiments/archive/reverse_transform/superseded_p1_scale_capturewise_p4/stage_05_joint_states/results/controls/fixed_p1/RESULTS.md).
- [Temporal omitted-vertex validation](../../experiments/archive/reverse_transform/superseded_p1_scale_capturewise_p4/stage_05_joint_states/results/validation/temporal/primary/RESULTS.md), [its g=1 comparison](../../experiments/archive/reverse_transform/superseded_p1_scale_capturewise_p4/stage_05_joint_states/results/validation/temporal/g1/RESULTS.md), and [complete fixation validation](../../experiments/archive/reverse_transform/superseded_p1_scale_capturewise_p4/stage_05_joint_states/results/validation/fixation/RESULTS.md).
- [Post-fit sensitivity/rank audit](../../experiments/archive/reverse_transform/superseded_p1_scale_capturewise_p4/stage_05_joint_states/results/POST_FIT_AUDIT.json), [conditional bootstrap](../../experiments/archive/reverse_transform/superseded_p1_scale_capturewise_p4/audits/conditional_block_bootstrap.json), [compact handoff metrics](../../experiments/archive/reverse_transform/superseded_p1_scale_capturewise_p4/HANDOFF_METRICS.json), and [historical experiment record](../../experiments/RECORD.md).

Shared libraries are [capturewise_reverse.py](../../distortion_model/capturewise_reverse.py), [capturewise_states.py](../../distortion_model/capturewise_states.py), [p1_joint.py](../../distortion_model/p1_joint.py) and [p1_fixed_scale.py](../../distortion_model/p1_fixed_scale.py). The latter supports exact fixed-scale and GPU bounded-profile calibration. Large frame calculations use CuPy float64; small global optimization runs on CPU with BLAS threads limited to one. Root planned/audited; GPT-6.1-sol high implemented the scientific pipeline and GPT-6-Luna low handled mechanics.

To reproduce, use the environment's `/home/aplab/.pyenv/versions/venv/bin/python` and the family [run.py](../../experiments/archive/reverse_transform/superseded_p1_scale_capturewise_p4/run.py) with a **new output family**, `--include-g1 --validation --backend cupy --maxiter 1000 --joint-maxiter 80 --state-maxiter 200`. Run [validation.py](../../experiments/archive/reverse_transform/superseded_p1_scale_capturewise_p4/validation.py) for omitted-vertex scoring and [validation_fixation.py](../../experiments/archive/reverse_transform/superseded_p1_scale_capturewise_p4/validation_fixation.py) for complete fixation holdouts. Completed outputs are never silently overwritten. Per-stage summaries/protocols include numeric arrays, source/frame masks, checksums, solver seeds/settings, native templates/radii and source snapshots. Current results have not been committed or pushed in this task.
