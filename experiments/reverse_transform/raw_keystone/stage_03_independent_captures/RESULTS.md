# Stage 3 results: independent captures

**Status:** `COMPLETE`; independent saved-results audit: `PASS`.

This report records the canonical saved run. The raw projective keystone output is not RMS- or area-normalized after transformation. Historical normalized-model artifacts are read-only comparisons; they did not supply raw fit coefficients or states.

## Coverage and run

| Scheduled | Complete | Unavailable | Runtime | Backend / device |
|---:|---:|---:|---:|---|
| 100,090 | 89,175 | 10,915 | 8.17 s | CuPy 14.2.0 float64 / Tesla P100-PCIE-16GB |

References are Capture 1 nominal zero-gaze centered arithmetic means (demand label 0.36036036036036034 D). P1 provides one positive profiled magnification per frame, shared with P4. The forward objective uses original centered camera coordinates and equal fixation weights; inverse metrics are secondary.

## Fitted coefficients and bounds

For each pattern the saved scaled coefficient order is `[a_x, a_y, b_x, b_y, lambda]`. The first four are dimensionless in scaled gaze coordinates `t_x=theta_x/u_x`, `t_y=theta_y/u_y`; their native equivalents are `[a_x/u_x², a_y/u_y², b_x/u_x, b_y/u_y]` in `[deg⁻², deg⁻², deg⁻¹, deg⁻¹]`. The optional radial `lambda` is dimensionless in the reference-radius basis; equivalent relative `κ` is px⁻². Active bound names are from the saved optimizer record.

| Capture | Pattern | Scaled coefficients (dimensionless) | Native gaze coefficients with units | κ (px⁻²) | Active bounds | Objective (px²) |
|---:|---|---|---|---:|---|---:|
| 1 | P1 | `0.001842711, 0.005201442, -0.004655544, 0.006156141` | `1.842711e-05, 0.01533394, -0.0004655544, 0.01056997` | — | none | 1.013 |
| 1 | P4 | `-0.001432657, 0.00163173, -0.007727538, -0.05763449` | `-1.432657e-05, 0.004810368, -0.0007727538, -0.09895721` | 0.00000000 | none | 2.735 |
| 2 | P1 | `0.001475461, 0.0002166153, -0.005716528, 0.008632921` | `1.475461e-05, 1.152139e-06, -0.0005716528, 0.0006296012` | — | none | 1.700 |
| 2 | P4 | `0.006252566, 0.1436564, 0.0005123313, -0.3738277, -0.06782358` | `6.252566e-05, 0.0007640827, 5.123313e-05, -0.02726335` | -0.00000185 | b_y | 20.118 |
| 3 | P1 | `0.0001831086, 0.003863189, -0.005585022, 0.006055925` | `1.831086e-06, 0.01799423, -0.0005585022, 0.01306996` | — | none | 1.212 |
| 3 | P4 | `0.002760664, -0.001461687, -0.004512243, -0.0591603, -0.05394056` | `2.760664e-05, -0.006808351, -0.0004512243, -0.1276804` | -0.00000147 | none | 5.386 |
| 4 | P1 | `-0.000278168, 0.0006068155, -0.002853156, 0.03107827` | `-2.78168e-06, 1.522654e-05, -0.0002853156, 0.004922992` | — | none | 0.442 |
| 4 | P4 | `0.001076269, -0.3137582, -0.003682072, 0.1804652, -0.03101525` | `1.076269e-05, -0.007872988, -0.0003682072, 0.02858682` | -0.00000085 | none | 10.669 |

## Per-fixation point distances

Each metric cell gives median / P95 / RMS corresponding-vertex distance in pixels. Inverse cells use each pattern’s valid saved inverse frames. Coverage is complete frames per fixation; exact counts and signed vertex residuals are in the summary JSON.

| Capture | Nominal gaze | Complete | P1 forward | P1 inverse | P4 forward | P4 inverse |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | -10° | 4,433 | 0.673 / 1.137 / 0.729 | 0.673 / 1.136 / 0.727 | 1.710 / 3.385 / 2.106 | 1.724 / 3.417 / 2.089 |
| 1 | -5° | 4,800 | 0.407 / 0.553 / 0.397 | 0.407 / 0.553 / 0.397 | 1.859 / 2.813 / 1.991 | 1.865 / 2.810 / 1.988 |
| 1 | +0° | 4,900 | 0.367 / 1.244 / 0.678 | 0.367 / 1.244 / 0.677 | 0.545 / 1.715 / 0.843 | 0.543 / 1.707 / 0.844 |
| 1 | +5° | 4,993 | 1.685 / 2.198 / 1.658 | 1.704 / 2.216 / 1.672 | 1.392 / 3.012 / 1.716 | 1.400 / 3.019 / 1.724 |
| 1 | +10° | 5,154 | 1.027 / 1.581 / 1.081 | 1.032 / 1.587 / 1.085 | 1.146 / 1.997 / 1.273 | 1.175 / 1.922 / 1.265 |
| 2 | -10° | 3,912 | 0.483 / 0.730 / 0.500 | 0.478 / 0.727 / 0.497 | 1.670 / 4.339 / 8.114 | 1.933 / 4.904 / 2.543 |
| 2 | -5° | 4,345 | 0.610 / 1.219 / 0.743 | 0.607 / 1.206 / 0.737 | 1.513 / 1.972 / 4.757 | 1.858 / 2.439 / 1.704 |
| 2 | +0° | 4,553 | 0.790 / 1.177 / 0.810 | 0.789 / 1.176 / 0.808 | 0.989 / 1.711 / 1.087 | 1.113 / 2.177 / 1.297 |
| 2 | +5° | 4,781 | 2.018 / 2.586 / 1.925 | 2.008 / 2.571 / 1.914 | 2.102 / 3.144 / 2.004 | 2.508 / 3.814 / 2.411 |
| 2 | +10° | 4,435 | 1.760 / 2.560 / 1.826 | 1.758 / 2.580 / 1.834 | 2.264 / 4.123 / 2.631 | 2.764 / 5.094 / 3.219 |
| 3 | -10° | 4,505 | 0.501 / 0.753 / 0.506 | 0.496 / 0.749 / 0.503 | 1.725 / 3.220 / 1.932 | 1.888 / 3.643 / 2.165 |
| 3 | -5° | 3,902 | 1.225 / 2.508 / 1.461 | 1.209 / 2.470 / 1.442 | 2.489 / 4.516 / 2.805 | 2.911 / 5.221 / 3.264 |
| 3 | +0° | 4,811 | 1.417 / 2.360 / 1.587 | 1.415 / 2.356 / 1.585 | 1.660 / 3.398 / 1.999 | 1.910 / 3.692 / 2.217 |
| 3 | +5° | 4,251 | 0.551 / 0.886 / 0.628 | 0.551 / 0.886 / 0.628 | 1.377 / 3.011 / 1.708 | 1.534 / 3.441 / 1.934 |
| 3 | +10° | 2,326 | 0.807 / 1.346 / 0.870 | 0.810 / 1.358 / 0.875 | 2.599 / 4.676 / 2.900 | 3.117 / 5.572 / 3.458 |
| 4 | -10° | 3,914 | 0.452 / 0.657 / 0.463 | 0.451 / 0.656 / 0.462 | 1.748 / 3.158 / 2.158 | 1.847 / 3.403 / 2.316 |
| 4 | -5° | 4,899 | 0.240 / 0.325 / 0.240 | 0.240 / 0.324 / 0.240 | 1.598 / 2.637 / 1.716 | 1.733 / 2.899 / 1.868 |
| 4 | +0° | 4,747 | 0.624 / 1.103 / 0.712 | 0.624 / 1.102 / 0.711 | 0.844 / 1.846 / 1.065 | 0.902 / 2.008 / 1.147 |
| 4 | +5° | 4,738 | 0.924 / 1.593 / 1.021 | 0.925 / 1.599 / 1.023 | 2.487 / 3.279 / 2.465 | 2.617 / 3.627 / 2.654 |
| 4 | +10° | 4,776 | 0.544 / 0.926 / 0.623 | 0.545 / 0.930 / 0.625 | 2.992 / 5.448 / 6.208 | 3.156 / 6.041 / 6.354 |

## Figures

- [point_errors_by_fixation.png](results/run/point_errors_by_fixation.png)
- [scale_and_raw_size_by_fixation.png](results/run/scale_and_raw_size_by_fixation.png)
- [p4_inverse_radius_ratio_by_fixation.png](results/run/p4_inverse_radius_ratio_by_fixation.png)
- [p1_triangles_raw_vs_historical.png](results/run/p1_triangles_raw_vs_historical.png)
- [p4_triangles_raw_vs_historical.png](results/run/p4_triangles_raw_vs_historical.png)
- [vertex_residual_vectors_native_px.png](results/run/vertex_residual_vectors_native_px.png)

Scale-factor and inverse-radius plots are diagnostics, not fit normalization steps. Triangle plots use matched raw/historical valid frames in native coordinates without rotation alignment or radius normalization. See the [read-only historical comparison record](results/run/historical_comparison.json) for those fields.
