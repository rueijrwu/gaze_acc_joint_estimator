# Corrected barrel-first P4 pipeline

**Protocol status:** The corrected five-stage pipeline is implemented and audited in [`experiments/reverse_transform/p1_scale_barrel_first_p4/`](experiments/reverse_transform/p1_scale_barrel_first_p4/). The previous plan is preserved at [`docs/legacy/CURRENT_PLAN_PRE_BARREL_FIRST_CORRECTION.md`](docs/legacy/CURRENT_PLAN_PRE_BARREL_FIRST_CORRECTION.md), and the superseded experiment family is archived at [`experiments/archive/reverse_transform/superseded_p1_scale_capturewise_p4/`](experiments/archive/reverse_transform/superseded_p1_scale_capturewise_p4/). The user-facing findings are summarized in [`CURRENT_STATUS.md`](CURRENT_STATUS.md). The exact run-time specification is preserved at [`experiments/reverse_transform/p1_scale_barrel_first_p4/PLAN_AT_RUN.md`](experiments/reverse_transform/p1_scale_barrel_first_p4/PLAN_AT_RUN.md); its SHA-256 is recorded by the stage summaries.

Ray tracing establishes one physical magnification `g`, shared by P1 and P4. The P1 calibration in Stage 02 supplies that exact fixed per-frame `g` to every later P4 stage. Do not fit a separate P4 scale or treat a `g=1` control as the physical constraint.

The active estimator follows the composition

```text
Y_tv = K4_c(B_v, theta_t)
Z_tv = D_kappa_cf(Y_tv)
mu_t = mean_v(Z_tv)
Xhat4_tv = g_t * (Z_tv - mu_t)
```

where the barrel map is `D_kappa(Y)=(1+kappa*||Y||^2)Y`, P4 keystone is capture-specific, and the radial coefficient is indexed by capture and fixation in Stage 03, then by frame state through `kappa=s*(A-A_ref)` in Stage 05. For every frame `t`, compute the raw transformed centroid `mu_t=mean_v(Z_tv)` from the current template, gaze, K and barrel coefficient. Inversion is

The horizontal keystone map used here is

```text
phi_t = (theta_t - offset4) / 10
K4_c(p, theta_t) = diag(exp(b_c*phi_t^2), exp(-b_c*phi_t^2)) * p
                   / (1 + phi_t*(cx_c*p_x + cy_c*p_y)/R_c)
```

`offset4` is a shared nuisance coordinate in degrees, not a physical gaze origin; `b_c`, `cx_c` and `cy_c` are dimensionless under the native capture radius `R_c`. The fitted bounds are `b_c∈[-0.25,0.25]`, `cx_c,cy_c∈[-0.2,0.2]`, shared `offset4∈[-120°,120°]`, and Stage 05 `s/s0∈[0.05,3]`. The shared template chart coordinates are bounded to `±0.75` with centered energy fixed as a gauge. The vertical gaze initializer remains a diagnostic; the fitted horizontal model uses `theta_t`.

```text
Y_tv = D_kappa^-1(X4_tv/g_t + mu_t)
B_v = K4_c^-1(Y_tv, theta_t)
```

Thus barrel inversion precedes keystone inversion. Never restore a stale centroid, reverse only the order of post-fit corrections, normalize output size, or introduce another P4 scale. Keep native pixel coordinates and report invalid rows and boundary contacts explicitly.

## 1. Data, identities and provenance

Use the frozen 88,978-row reviewed P1/P4 cohort and preserve the exact `(capture, source_row)` identity, original frame index and timestamps. `data.py` reads the raw detections and frozen identity bundle; the bundle fixes the reviewed roster, but does not supply fitted P4 parameters. Record raw-data and bundle hashes, source row conventions, target gaze/demand labels, and the reviewed-mask provenance. Do not import P4 Stage 03/04/05 arrays, template/reference, slopes, optical priors, states, or fitted parameters from the superseded family.

The family contains production and training-only temporal inputs. The temporal split is based on original source rows within each fixation: first 75% train, next 5% gap, final 20% test. All calibration, targets, templates, radii and optics for temporal validation are learned from training rows only.

## 2. Stage 01 — gaze initializer

Estimate horizontal gaze separately for each capture from fixation means of raw P4-minus-P1 centroid displacement. Use the nominal zero fixation as displacement origin and fit the quadratic `theta=a1*u+a2*u^2` on five fixation means. Evaluate the fitted gaze on every reviewed row; keep vertical gaze as a diagnostic only. For temporal validation, train the polynomial on training rows only and apply the frozen polynomial to the temporal test rows. Save the coefficients, observed ranges, residuals, monotonicity, source identities and masks.

## 3. Stage 02 — shared P1 calibration

Using Stage 01 gaze, fit one shared P1 template `B1`, shared P1 keystone and nuisance offset, and bounded per-frame `g` in `[0.98, 1.02]`. Preserve the existing P1 fit and scale conventions. Freeze the exact Stage 02 `g` array and use it for both patterns in Stages 03–05. Fit globals and each per-frame profile using only the declared fit rows; retain P1-only bounds, domain checks, stationary multistarts and CPU/GPU agreement evidence. No P4 result may tune Stage 02.

## 4. Stage 03 — per-gaze relative barrel calibration

At each of the five fixation gazes `f`, compare the four captures' centered P4 triangles after division by the shared P1 frame scale. Form the observed scale-corrected mean sufficient statistic for each capture/fixation cell and fit one relative radial coefficient `kappa_cf` for each capture and a latent centered reference triangle `B_f` for that gaze:

```text
M_cf = mean_rows(X4_t / g_t)
Z_cf = D_kappa_cf(B_f)
Mhat_cf = Z_cf - mean_vertices(Z_cf)
```

Set `kappa_1f=0` separately at each fixation to define the C1-relative gauge. This does not claim that Capture 1 has zero intrinsic barrel distortion. Fit the latent triangle with a finite 1 px native-pixel reference prior to the same-fixation C1 mean of `X4/g`; do not fix its energy, blend gaze conditions, or claim the targets are pure physical barrel coefficients. Use equal capture/fixation/vertex weighting and preserve within-cell variance and counts. Save all 20 coefficients, latent references, target uncertainty/width, domain checks, parameter bounds, multistart certificates and exact training identities.

Fit a fresh radial slope from all 20 relative coefficients by the equal-cell least-squares regression

```text
s0 = sum_cf [(demand_c - A_ref) * kappa_cf]
     / sum_cf [(demand_c - A_ref)^2]
Abar_cf = A_ref + kappa_cf/s0
```

using `A_ref=0.36036036 D`. Require an informative finite `s0`; do not clip target means. Freeze the resulting 20 `Abar_cf` targets for Stage 05 with the existing 0.25 D fixation-mean width. Report the target range and all capture/fixation cells. These are regularizers derived from the same recording, not independent physiological observations.

For every measured frame `t`, restore the Stage 03 latent raw centroid `mu_D(B_f,kappa_cf)` and invert barrel first:

```text
Y_tv = D_kappa_cf^-1(X4_tv/g_t + mu_D(B_f,kappa_cf))
```

This yields the corrected native triangles used to initialize Stage 04. Recompute centroids from the declared latent template, not from a held-out or mismatched condition.

## 5. Stage 04 — capturewise keystone after barrel inversion

Initialize one shared P4 template `B` from the C1 nominal-zero mean of the Stage 03 corrected `Y`. Preserve its native centroid and centered energy as explicit gauges. Compute each capture's native radius `R_c` afresh from its corrected nominal-zero mean triangle. Do not import a radius, template, `K`, offset, slope or optimum from the superseded P4 stages.

First fit capture-specific K maps to the centered Stage 03 corrected triangles, using the frozen Stage 03 per-gaze barrel coefficients. This is a seed only. Then evaluate the final initialization against original P4 pixels using the full forward composition `g*(D_kappa(K(B,theta))-mean(D_kappa(K(B,theta))))`, with trial-consistent centroid and no added output normalization. Jointly fit one shared P4 template `B`, capture-specific keystone coefficients and one shared nuisance offset. Keep the original P4 observations and Stage 03 barrel targets fixed. Use native-pixel inverse and forward diagnostics, full-domain checks, analytic gradients, multiple starts and stationarity gates. The Stage 04 initializer must be new; do not seed it from old P4 K or learned-reference fits.

## 6. Stage 05 — joint gaze, accommodation and optical refinement

Initialize solely from the new Stage 03/04 outputs. Keep Stage 02 P1 globals and exact shared frame `g` fixed. Jointly refine the shared P4 template chart, capture-specific K, shared P4 nuisance offset, radial slope `s`, horizontal gaze and frame accommodation using all frames and the composition in this plan. Let frame barrel vary only through `kappa=s*(A-A_ref)`; do not introduce an independent per-frame coefficient.

Preserve the existing frame gaze anchor width of 0.5° and the 0.25 D soft anchor on each dynamically computed fixation mean A against the frozen Stage 03 `Abar_cf`. State bounds are `theta0±5°` and `A∈[0,6] D`. There is no temporal smoothing or per-frame A penalty. Use fresh Stage 04 optics as raw-coordinate soft priors: mean of 14 standardized residual squares, with widths 0.05 for each capture K coefficient, 10° for shared offset and `|s0|` for slope. Do not use an extra Stage 03 `kappa` penalty; the frozen Stage 03 targets set the fixation-mean A regularizer already. Preserve the Stage 03/04 native template centroid and centered-energy gauges; no template prior is added.

Use equal P1/P4 pattern and capture/fixation weighting, native pixel residual scale 1 px, analytic derivatives and trial-consistent centroids. Save training/all-row losses by component, forward and inverse P1/P4 RMS, accommodation-equivalent residuals, all 20 state groups, radial slopes, contacts, domains, start spread and both global/state stationarity certificates. Select only a valid stationary start; do not promote an uncertified checkpoint.

## 7. Audits and held-out validation

Before production fitting, audit the new forward/inverse equations, nonzero-centroid inverse restoration, derivatives, synthetic coefficient recovery, mean-statistic equivalence, and CPU/GPU float64 agreement. Verify that Stage 03/04/05 contain no runtime imports or artifact reads from superseded P4 Stage 03/04/05. Compare new Stage 03 and Stage 04 references/priors only within the corrected family.

For temporal validation, fit Stages 01–05 from training rows only. For each test frame, omit one P4 vertex, infer gaze/state using the remaining two vertices and P1, and score the omitted raw vertex from the observed two-vertex centroid. No held vertex may affect gaze calibration, P1 scale, barrel targets, reference fitting, K fitting, trial centroid or state anchor. Infer test `g` from test P1 alone using frozen P1 globals. Preserve test source identities and report all invalid rows; do not silently drop failures. Include a 5% source-row gap and report results by capture/fixation as well as aggregate.

## 8. Artifact boundaries, implementation and handoff

The active family is `experiments/reverse_transform/p1_scale_barrel_first_p4/` with stages named `stage_01_gaze_initialization`, `stage_02_shared_p1_scale`, `stage_03_per_gaze_barrel`, `stage_04_capturewise_keystone`, and `stage_05_joint_states`. Stage 01/02 production and temporal arrays are read-only copies of the existing P1 calibration with verified hashes. The new stages do not read the superseded family's P4 stage arrays or priors. After the independent source-boundary audit passed, the previous family was archived intact as one unit at `experiments/archive/reverse_transform/superseded_p1_scale_capturewise_p4/`.

Use CuPy float64 for large frame workloads where available and CPU small-global optimization with one BLAS thread. Record backend, precision, seeds, settings, source hashes, exact array hashes, training masks and source IDs. Snapshot all source modules used. Do not commit or publish results as part of this analysis.

The final report and status distinguish measured held geometry from inferred accommodation, disclose reference and anchor conventions, retain prior results as historical context without attributing changes to ordering alone, state whether the corrected model is promoted, and identify remaining model/identifiability limitations. The root status is the user-facing conclusion; stage artifacts and a compact handoff carry the detailed audit trail.
