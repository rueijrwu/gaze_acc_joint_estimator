# Stage 05: Capture 1 gaze/accommodation state ablation

## Setup and scope

This compares three states on the same 24,280 complete Capture 1 frames and the fixed reviewed correspondence/reference:

- **A0:** exact saved Stage 04 gaze, accommodation, and P1 magnification. It was not reoptimized under the new objective.
- **G:** free horizontal gaze only; accommodation is fixed at the reference value and the relative barrel increment is zero.
- **GA:** free horizontal gaze and accommodation, using the frozen law \(\kappa=\beta(A-A_{ref})\). The law and slope are not refit.

Vertical gaze, references/origins, P1/P4 keystone coefficients, units, population, and correspondence are frozen. For G and GA, the positive common magnification is reprofiled from P1 at every trial gaze and then shared with P4. The forward objective is an equal-fixation mean of P1 plus P4 mean-three-vertex squared errors in centered camera coordinates, with equal weights. The only priors are arithmetic fixation-mean anchors: 0.5° for horizontal gaze and 0.25 D for GA accommodation. Operational bounds are θx ∈ [−20°, +20°] and A ∈ [0, 6] D; they are not framewise priors or calibration truth. Measured inverse failures are reported rather than trimmed.

## Selected fits and audit

| State model | Selected objective (px²) | Multistart objective range (px²) | Projected-gradient infinity norm | Minimum free-frame curvature | Selected certificate |
|---|---:|---:|---:|---:|---|
| A0 | — | — | — | — | Preserved Stage 04 baseline; no Stage 05 refit |
| G | 3.554731 | 0.005721 | 9.66e-11 | 1.21e-4 | Stationary, positive curvature |
| GA | 2.522570 | 0.005408 | 2.19e-8 | 1.25e-4 | Stationary, positive curvature |

The CuPy float64 canonical run on a Tesla P100 completed in 40.75 s. The independent CPU audit passed. It reproduced the forward operator to 1.99e-13 px, the all-frame Jacobians to relative error below 1.11e-9, and synthetic inverse closure below 1.31e-12 px. All 42 parent hashes and 6 source hashes match. The selected L-BFGS-B starts reached their iteration cap; block-Newton polishing then reached the stationary certificates reported above. The multistart ranges remain visible, so these are not claims of a unique global optimum.

Every model has 0 P4 inverse failures; all 24,280 frames are common-valid. The G model has 1,304 frames at the lower θ bound and 2 at the upper bound. GA has 1,311 lower and 2 upper θ-bound frames, plus 909 frames at A=0. A0 retains the Stage 04 A bounds (918 lower-bound frames). No model reaches the upper A bound.

## Pooled corresponding-vertex distances

Each cell gives median / P95 / RMS in px. Forward values use all frames; inverse values use the common-valid set of 24,280 frames (72,840 corresponding vertex distances).

| Model | P1 forward | P4 forward | P1 inverse, common-valid | P4 inverse, common-valid | P4 inverse failures |
|---|---:|---:|---:|---:|---:|
| A0 | 0.675 / 2.009 / 1.017 | 0.995 / 1.920 / 1.186 | 0.674 / 2.021 / 1.022 | 0.998 / 1.932 / 1.184 | 0 |
| G | 0.548 / 1.376 / 0.800 | 1.499 / 2.780 / 1.699 | 0.546 / 1.382 / 0.802 | 1.514 / 2.779 / 1.697 | 0 |
| GA | 0.548 / 1.378 / 0.801 | 0.958 / 1.940 / 1.185 | 0.546 / 1.385 / 0.803 | 0.970 / 1.959 / 1.187 | 0 |

G improves P1 but does not recover the P4 improvement from A0. GA reduces P4 errors relative to G while leaving P4 inverse RMS/P95 slightly above A0. The fixation-dependent A pattern persists after horizontal gaze is freed; the result is consistent with Case 4 in the next-step protocol, with no cross-agreement yet.

## Fitted states by fixation

Each state entry is mean ± within-fixation SD. Δθ is the mean fitted horizontal gaze minus the saved Stage 03 gaze. Bounds give counts as θ lower/upper, A lower/upper.

| Model | Nominal gaze | θx mean ± SD (deg) | Mean Δθ (deg) | A mean ± SD (D) | Bound counts θlo/hi, Alo/hi |
|---|---:|---:|---:|---:|---:|
| A0 | −10° | −9.993 ± 0.512 | 0.000 | 0.566 ± 0.078 | 0/0, 2/0 |
| A0 | −5° | −5.004 ± 0.252 | 0.000 | 0.592 ± 0.102 | 0/0, 0/0 |
| A0 | 0° | 0.000 ± 0.175 | 0.000 | 0.359 ± 0.060 | 0/0, 0/0 |
| A0 | +5° | 5.049 ± 0.570 | 0.000 | 0.196 ± 0.147 | 0/0, 891/0 |
| A0 | +10° | 9.981 ± 0.272 | 0.000 | 0.207 ± 0.082 | 0/0, 25/0 |
| G | −10° | −9.999 ± 5.174 | −0.007 | 0.360 ± 0.000 | 265/0, 0/0 |
| G | −5° | −5.002 ± 1.108 | 0.002 | 0.360 ± 0.000 | 10/0, 0/0 |
| G | 0° | 0.000 ± 2.462 | 0.000 | 0.360 ± 0.000 | 0/0, 0/0 |
| G | +5° | 5.020 ± 12.967 | −0.029 | 0.360 ± 0.000 | 1,028/0, 0/0 |
| G | +10° | 9.974 ± 1.482 | −0.006 | 0.360 ± 0.000 | 1/2, 0/0 |
| GA | −10° | −9.999 ± 5.309 | −0.007 | 0.569 ± 0.072 | 267/0, 2/0 |
| GA | −5° | −5.002 ± 1.243 | 0.002 | 0.591 ± 0.099 | 10/0, 0/0 |
| GA | 0° | 0.000 ± 2.416 | 0.000 | 0.361 ± 0.058 | 0/0, 0/0 |
| GA | +5° | 5.019 ± 12.968 | −0.030 | 0.201 ± 0.150 | 1,029/0, 882/0 |
| GA | +10° | 9.973 ± 1.659 | −0.007 | 0.205 ± 0.082 | 5/2, 25/0 |

## Forward and inverse distances by fixation

Cells use median / P95 / RMS (px); inverse metrics use the common-valid population. The complete per-fixation JSON also contains means, standard deviations, correlations, signed residual vectors, radius ratios, and all conditioning summaries.

| Model | Gaze | P1 forward | P4 forward | P1 inverse | P4 inverse |
|---|---:|---:|---:|---:|---:|
| A0 | −10° | 0.673 / 1.137 / 0.729 | 1.085 / 2.294 / 1.534 | 0.673 / 1.136 / 0.727 | 1.110 / 2.288 / 1.532 |
| A0 | −5° | 0.407 / 0.553 / 0.397 | 1.206 / 1.786 / 1.236 | 0.407 / 0.553 / 0.397 | 1.224 / 1.802 / 1.248 |
| A0 | 0° | 0.367 / 1.244 / 0.678 | 0.529 / 1.650 / 0.801 | 0.367 / 1.244 / 0.677 | 0.528 / 1.654 / 0.800 |
| A0 | +5° | 1.685 / 2.198 / 1.658 | 1.107 / 2.238 / 1.232 | 1.704 / 2.216 / 1.672 | 1.099 / 2.270 / 1.228 |
| A0 | +10° | 1.027 / 1.581 / 1.081 | 0.907 / 1.704 / 1.047 | 1.032 / 1.587 / 1.085 | 0.907 / 1.686 / 1.033 |
| G | −10° | 0.659 / 0.833 / 0.663 | 1.751 / 3.384 / 2.089 | 0.658 / 0.830 / 0.662 | 1.734 / 3.413 / 2.074 |
| G | −5° | 0.406 / 0.550 / 0.390 | 1.909 / 2.865 / 2.026 | 0.406 / 0.550 / 0.389 | 1.915 / 2.867 / 2.023 |
| G | 0° | 0.372 / 1.186 / 0.658 | 0.505 / 1.648 / 0.817 | 0.371 / 1.186 / 0.658 | 0.504 / 1.644 / 0.820 |
| G | +5° | 1.010 / 1.337 / 1.026 | 1.547 / 3.099 / 1.789 | 1.020 / 1.349 / 1.031 | 1.564 / 3.119 / 1.806 |
| G | +10° | 1.081 / 1.504 / 1.031 | 1.375 / 2.350 / 1.515 | 1.085 / 1.512 / 1.035 | 1.393 / 2.253 / 1.504 |
| GA | −10° | 0.662 / 0.823 / 0.662 | 1.089 / 2.286 / 1.517 | 0.660 / 0.820 / 0.661 | 1.109 / 2.287 / 1.518 |
| GA | −5° | 0.406 / 0.550 / 0.389 | 1.206 / 1.789 / 1.238 | 0.406 / 0.550 / 0.389 | 1.226 / 1.804 / 1.250 |
| GA | 0° | 0.372 / 1.187 / 0.658 | 0.509 / 1.644 / 0.789 | 0.371 / 1.186 / 0.658 | 0.510 / 1.651 / 0.791 |
| GA | +5° | 1.008 / 1.337 / 1.026 | 0.998 / 2.317 / 1.253 | 1.019 / 1.348 / 1.031 | 1.004 / 2.377 / 1.258 |
| GA | +10° | 1.084 / 1.509 / 1.034 | 0.918 / 1.646 / 1.046 | 1.089 / 1.516 / 1.038 | 0.935 / 1.621 / 1.032 |

## Radius, signed residual, and gaze/A association

Radius ratio is RMS radius about the triangle centroid divided by reference radius; it is diagnostic and not used to normalize fitted points.

| Model | Observed P4 / P1-scaled radius ratio: mean ± SD; P95 | Recovered P4 radius ratio: mean ± SD; P95 | Corr(Δθ, ΔA) |
|---|---:|---:|---:|
| A0 | 0.999370 ± 0.007434; 1.009739 | 0.999653 ± 0.003891; 1.003652 | n/a |
| G | 0.999367 ± 0.007431; 1.009725 | 0.999367 ± 0.007419; 1.009717 | 0.192 |
| GA | 0.999367 ± 0.007431; 1.009725 | 0.999678 ± 0.003898; 1.003982 | −0.068 |

The reported correlation uses Δθ = fitted θx minus the saved Stage 03 gaze and ΔA = model A minus the saved Stage 04 A0 on the same frame; for G, A is fixed, so this correlation describes change from A0 rather than fitted G accommodation variability.

Mean signed P4 vertex residuals (x,y) in px, vertices 1–3:

| Model | Vertex 1 | Vertex 2 | Vertex 3 |
|---|---:|---:|---:|
| A0 | (−0.133, −0.151) | (+0.111, −0.032) | (+0.022, +0.183) |
| G | (−0.076, −0.068) | (+0.100, −0.213) | (−0.024, +0.280) |
| GA | (−0.019, −0.096) | (+0.104, −0.142) | (−0.084, +0.238) |

The [canonical summary JSON](results/run/summary.json) contains exact per-fixation correlations, signed P4 vertex residuals, observed/recovered radius ratios, and per-fixation conditioning fields. [frames.npz](results/run/frames.npz) contains signed P1/P4 residual arrays and the corresponding per-frame state and conditioning arrays.

## Conditioning

P4-only and joint P1+P4 columns are from each saved state's Jacobian. Raw singular values and condition numbers use degree and diopter units; acute angles are unit-invariant. Values below are median / P95, except acute angle which is median / P05.

| Model | Jacobian | σmax | σmin | Condition number | Acute angle (deg) |
|---|---|---:|---:|---:|---:|
| A0 | P4-only | 6.404 / 6.425 | 0.109 / 0.210 | 58.537 / 495.896 | 87.798 / 87.460 |
| A0 | Joint P1+P4 | 6.404 / 6.425 | 0.147 / 0.334 | 43.449 / 102.207 | 88.355 / 87.920 |
| G | P4-only | 6.404 / 6.425 | 0.096 / 0.227 | 66.465 / 534.533 | 87.805 / 87.485 |
| G | Joint P1+P4 | 6.404 / 6.425 | 0.179 / 0.367 | 35.704 / 116.114 | 88.447 / 87.867 |
| GA | P4-only | 6.404 / 6.425 | 0.096 / 0.228 | 66.415 / 531.953 | 87.797 / 87.495 |
| GA | Joint P1+P4 | 6.404 / 6.425 | 0.179 / 0.368 | 35.821 / 120.291 | 88.447 / 87.856 |

The independent audit's near-bound subset includes frames within 0.5° or 0.05 D of an operational boundary:

| Model | Near-bound frames | P4 condition median / P95 | P4 acute-angle median / P05 | Joint condition median / P95 | Joint acute-angle median / P05 |
|---|---:|---:|---:|---:|---:|
| A0 | 1,240 | 39.484 / 39.562 | 87.405 / 87.347° | 26.368 / 26.422 | 88.267 / 88.228° |
| G | 1,323 | 71.052 / 71.065 | 88.122 / 87.952° | 26.825 / 26.825 | 89.291 / 89.227° |
| GA | 2,533 | 70.901 / 71.116 | 87.953 / 87.398° | 26.814 / 26.828 | 89.227 / 88.390° |

There were no P4 inverse failures in any near-bound or full population. These local calculations do not establish global or physiological identifiability. The A trend persists after gaze is freed, and separate cross-agreement has not been demonstrated; see the [scientific review](SCIENTIFIC_REVIEW.md) for interpretation and the recommended next experiment.

## Figures

- [State means by fixation](results/run/state_means_by_fixation.png)
- [Forward and inverse metrics by fixation](results/run/forward_inverse_metrics_by_fixation.png)
- [Corrected triangle means and clouds](results/run/corrected_triangle_means_clouds.png)
- [Conditioning by fixation](results/run/conditioning_by_fixation.png)
- [States by source row](results/run/states_by_source_row.png)
- [Joint gaze/A changes](results/run/delta_gaze_vs_delta_A.png)
