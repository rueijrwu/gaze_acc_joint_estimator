# Stage 1 results: capture1 p1

**Status:** `COMPLETE`; independent saved-results audit: `PASS`.

This report records the canonical saved run. The raw projective keystone output is not RMS- or area-normalized after transformation. Historical normalized-model artifacts are read-only comparisons; they did not supply raw fit coefficients or states.

## Coverage and run

| Scheduled | Complete | Unavailable | Runtime | Backend / device |
|---:|---:|---:|---:|---|
| 24,700 | 24,280 | 420 | 1.63 s | CuPy 14.2.0 float64 / Tesla P100-PCIE-16GB |

References are Capture 1 nominal zero-gaze centered arithmetic means (demand label 0.36036036036036034 D). P1 provides one positive profiled magnification per frame, shared with P4. The forward objective uses original centered camera coordinates and equal fixation weights; inverse metrics are secondary.

## Fitted coefficients and bounds

For each pattern the saved scaled coefficient order is `[a_x, a_y, b_x, b_y, lambda]`. The first four are dimensionless in scaled gaze coordinates `t_x=theta_x/u_x`, `t_y=theta_y/u_y`; their native equivalents are `[a_x/u_x², a_y/u_y², b_x/u_x, b_y/u_y]` in `[deg⁻², deg⁻², deg⁻¹, deg⁻¹]`. The optional radial `lambda` is dimensionless in the reference-radius basis; equivalent relative `κ` is px⁻². Active bound names are from the saved optimizer record.

| Capture | Pattern | Scaled coefficients (dimensionless) | Native gaze coefficients with units | κ (px⁻²) | Active bounds | Objective (px²) |
|---:|---|---|---|---:|---|---:|
| 1 | P1 | `0.001842711, 0.005201442, -0.004655544, 0.006156141` | `1.842711e-05, 0.01533394, -0.0004655544, 0.01056997` | — | none | 1.013 |

## Per-fixation point distances

Each metric cell gives median / P95 / RMS corresponding-vertex distance in pixels. Inverse cells use each pattern’s valid saved inverse frames. Coverage is complete frames per fixation; exact counts and signed vertex residuals are in the summary JSON.

| Capture | Nominal gaze | Complete | P1 forward | P1 inverse | P4 forward | P4 inverse |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | -10° | 4,433 | 0.673 / 1.137 / 0.729 | 0.673 / 1.136 / 0.727 | — | — |
| 1 | -5° | 4,800 | 0.407 / 0.553 / 0.397 | 0.407 / 0.553 / 0.397 | — | — |
| 1 | +0° | 4,900 | 0.367 / 1.244 / 0.678 | 0.367 / 1.244 / 0.677 | — | — |
| 1 | +5° | 4,993 | 1.685 / 2.198 / 1.658 | 1.704 / 2.216 / 1.672 | — | — |
| 1 | +10° | 5,154 | 1.027 / 1.581 / 1.081 | 1.032 / 1.587 / 1.085 | — | — |

## Figures

- [point_errors_by_fixation.png](results/run/point_errors_by_fixation.png)
- [scale_and_raw_size_by_fixation.png](results/run/scale_and_raw_size_by_fixation.png)
- [p1_triangles_raw_vs_historical.png](results/run/p1_triangles_raw_vs_historical.png)
- [vertex_residual_vectors_native_px.png](results/run/vertex_residual_vectors_native_px.png)

Scale-factor and inverse-radius plots are diagnostics, not fit normalization steps. Triangle plots use matched raw/historical valid frames in native coordinates without rotation alignment or radius normalization. See the [read-only historical comparison record](results/run/historical_comparison.json) for those fields.
