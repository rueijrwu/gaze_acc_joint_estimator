# Stage 4 results: framewise accommodation

**Status:** `COMPLETE`; independent saved-results audit: `PASS`.

This report records the canonical saved run. The raw projective keystone output is not RMS- or area-normalized after transformation. Historical normalized-model artifacts are read-only comparisons; they did not supply raw fit coefficients or states.

## Coverage and run

| Scheduled | Complete | Unavailable | Runtime | Backend / device |
|---:|---:|---:|---:|---|
| 100,090 | 89,175 | 10,915 | 7.54 s | CuPy 14.2.0 float64 / Tesla P100-PCIE-16GB |

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

Stage 4 framewise A selected equal-fixation objective: `7.340934` px². It has `949` frames at A=0, `13` at A=6, and `0` upper bounds restricted by the forward branch. The saved selected certificate reports projected-gradient norm `1.24e-12` and minimum free-frame curvature `26.7389`.

## Per-fixation point distances

Each metric cell gives median / P95 / RMS corresponding-vertex distance in pixels. Inverse cells use each pattern’s valid saved inverse frames. Coverage is complete frames per fixation; exact counts and signed vertex residuals are in the summary JSON.

| Capture | Nominal gaze | Complete | P1 forward | P1 inverse | P4 forward | P4 inverse |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | -10° | 4,433 | 0.673 / 1.137 / 0.729 | 0.673 / 1.136 / 0.727 | 1.171 / 2.332 / 1.557 | 1.201 / 2.336 / 1.559 |
| 1 | -5° | 4,800 | 0.407 / 0.553 / 0.397 | 0.407 / 0.553 / 0.397 | 1.140 / 1.764 / 1.217 | 1.159 / 1.777 / 1.228 |
| 1 | +0° | 4,900 | 0.367 / 1.244 / 0.678 | 0.367 / 1.244 / 0.677 | 0.538 / 1.683 / 0.816 | 0.537 / 1.681 / 0.817 |
| 1 | +5° | 4,993 | 1.685 / 2.198 / 1.658 | 1.704 / 2.216 / 1.672 | 1.014 / 2.281 / 1.188 | 1.003 / 2.344 / 1.186 |
| 1 | +10° | 5,154 | 1.027 / 1.581 / 1.081 | 1.032 / 1.587 / 1.085 | 0.830 / 1.657 / 0.946 | 0.831 / 1.643 / 0.939 |
| 2 | -10° | 3,912 | 0.483 / 0.730 / 0.500 | 0.478 / 0.727 / 0.497 | 1.672 / 3.632 / 6.899 | 1.941 / 4.206 / 11.286 |
| 2 | -5° | 4,345 | 0.610 / 1.219 / 0.743 | 0.607 / 1.206 / 0.737 | 0.810 / 1.412 / 3.919 | 1.039 / 1.568 / 6.187 |
| 2 | +0° | 4,553 | 0.790 / 1.177 / 0.810 | 0.789 / 1.176 / 0.808 | 0.901 / 1.213 / 0.924 | 1.098 / 1.325 / 1.081 |
| 2 | +5° | 4,781 | 2.018 / 2.586 / 1.925 | 2.008 / 2.571 / 1.914 | 1.587 / 1.983 / 1.499 | 1.875 / 2.291 / 1.770 |
| 2 | +10° | 4,435 | 1.760 / 2.560 / 1.826 | 1.758 / 2.580 / 1.834 | 1.825 / 3.187 / 2.178 | 2.150 / 3.862 / 2.589 |
| 3 | -10° | 4,505 | 0.501 / 0.753 / 0.506 | 0.496 / 0.749 / 0.503 | 1.708 / 3.099 / 1.924 | 1.868 / 3.502 / 2.155 |
| 3 | -5° | 3,902 | 1.225 / 2.508 / 1.461 | 1.209 / 2.470 / 1.442 | 1.908 / 3.096 / 1.948 | 2.308 / 3.528 / 2.275 |
| 3 | +0° | 4,811 | 1.417 / 2.360 / 1.587 | 1.415 / 2.356 / 1.585 | 1.603 / 3.391 / 1.986 | 1.838 / 3.627 / 2.196 |
| 3 | +5° | 4,251 | 0.551 / 0.886 / 0.628 | 0.551 / 0.886 / 0.628 | 1.332 / 3.018 / 1.625 | 1.495 / 3.346 / 1.815 |
| 3 | +10° | 2,326 | 0.807 / 1.346 / 0.870 | 0.810 / 1.358 / 0.875 | 1.208 / 2.653 / 1.469 | 1.365 / 3.020 / 1.667 |
| 4 | -10° | 3,914 | 0.452 / 0.657 / 0.463 | 0.451 / 0.656 / 0.462 | 1.820 / 2.233 / 1.985 | 1.903 / 2.431 / 2.105 |
| 4 | -5° | 4,899 | 0.240 / 0.325 / 0.240 | 0.240 / 0.324 / 0.240 | 1.025 / 1.594 / 1.079 | 1.145 / 1.694 / 1.165 |
| 4 | +0° | 4,747 | 0.624 / 1.103 / 0.712 | 0.624 / 1.102 / 0.711 | 0.779 / 1.286 / 0.874 | 0.839 / 1.358 / 0.933 |
| 4 | +5° | 4,738 | 0.924 / 1.593 / 1.021 | 0.925 / 1.599 / 1.023 | 2.401 / 2.723 / 2.355 | 2.520 / 2.949 / 2.517 |
| 4 | +10° | 4,776 | 0.544 / 0.926 / 0.623 | 0.545 / 0.930 / 0.625 | 2.969 / 4.667 / 5.687 | 3.096 / 5.140 / 5.972 |

## Framewise A and matched normalized-forward control

State spreads are within-fixation standard deviations, not uncertainty of the means. The matched normalized control uses the same 89,175 complete rows, forward P4 residual objective, and fixation-mean anchors; it is a historical read-only comparison and not an input to the raw calibration.

| Fit | Objective (px²) | P4 forward median / P95 / RMS (px) | P4 inverse failures | A=0 / A=6 frames |
|---|---:|---:|---:|---:|
| Raw keystone | 7.341 | 1.255 / 3.079 / 2.580 | 0 | 949 / 13 |
| Normalized forward control | 8.009 | 1.299 / 3.304 / 2.688 | 0 | 989 / 13 |

| Capture | Gaze | Expected A (D) | Raw A mean ± SD (D) | Normalized-control A mean ± SD (D) | Raw P4 forward M / P95 / RMS (px) | Normalized-control P4 forward M / P95 / RMS (px) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | -10° | 0.360 | 0.566 ± 0.073 | 0.569 ± 0.073 | 1.171 / 2.332 / 1.557 | 1.091 / 2.282 / 1.526 |
| 1 | -5° | 0.360 | 0.587 ± 0.099 | 0.591 ± 0.099 | 1.140 / 1.764 / 1.217 | 1.208 / 1.786 / 1.238 |
| 1 | +0° | 0.360 | 0.362 ± 0.057 | 0.361 ± 0.059 | 0.538 / 1.683 / 0.816 | 0.529 / 1.646 / 0.800 |
| 1 | +5° | 0.360 | 0.211 ± 0.160 | 0.198 ± 0.151 | 1.014 / 2.281 / 1.188 | 1.101 / 2.285 / 1.235 |
| 1 | +10° | 0.360 | 0.243 ± 0.079 | 0.205 ± 0.084 | 0.830 / 1.657 / 0.946 | 0.907 / 1.698 / 1.041 |
| 2 | -10° | 4.000 | 3.926 ± 0.432 | 3.934 ± 0.436 | 1.672 / 3.632 / 6.899 | 1.963 / 3.203 / 7.613 |
| 2 | -5° | 4.000 | 4.074 ± 0.231 | 4.076 ± 0.232 | 0.810 / 1.412 / 3.919 | 0.819 / 1.461 / 4.392 |
| 2 | +0° | 4.000 | 4.008 ± 0.087 | 4.008 ± 0.088 | 0.901 / 1.213 / 0.924 | 0.904 / 1.193 / 0.920 |
| 2 | +5° | 4.000 | 4.114 ± 0.136 | 4.116 ± 0.135 | 1.587 / 1.983 / 1.499 | 1.879 / 2.200 / 1.731 |
| 2 | +10° | 4.000 | 3.720 ± 0.136 | 3.729 ± 0.135 | 1.825 / 3.187 / 2.178 | 2.098 / 3.382 / 2.462 |
| 3 | -10° | 3.000 | 3.138 ± 0.104 | 3.143 ± 0.103 | 1.708 / 3.099 / 1.924 | 1.720 / 3.105 / 1.932 |
| 3 | -5° | 3.000 | 3.377 ± 0.260 | 3.376 ± 0.259 | 1.908 / 3.096 / 1.948 | 1.940 / 3.025 / 1.933 |
| 3 | +0° | 3.000 | 3.094 ± 0.121 | 3.093 ± 0.121 | 1.603 / 3.391 / 1.986 | 1.596 / 3.384 / 1.979 |
| 3 | +5° | 3.000 | 3.021 ± 0.076 | 3.019 ± 0.077 | 1.332 / 3.018 / 1.625 | 1.338 / 2.974 / 1.612 |
| 3 | +10° | 3.000 | 2.754 ± 0.221 | 2.741 ± 0.211 | 1.208 / 2.653 / 1.469 | 1.218 / 2.682 / 1.505 |
| 4 | -10° | 2.000 | 2.065 ± 0.170 | 2.068 ± 0.170 | 1.820 / 2.233 / 1.985 | 1.862 / 2.489 / 2.056 |
| 4 | -5° | 2.000 | 2.165 ± 0.141 | 2.167 ± 0.141 | 1.025 / 1.594 / 1.079 | 1.082 / 1.609 / 1.073 |
| 4 | +0° | 2.000 | 2.043 ± 0.125 | 2.043 ± 0.125 | 0.779 / 1.286 / 0.874 | 0.790 / 1.817 / 0.975 |
| 4 | +5° | 2.000 | 1.915 ± 0.154 | 1.916 ± 0.153 | 2.401 / 2.723 / 2.355 | 2.577 / 2.995 / 2.548 |
| 4 | +10° | 2.000 | 1.740 ± 0.259 | 1.742 ± 0.261 | 2.969 / 4.667 / 5.687 | 3.299 / 5.150 / 5.234 |

The saved control audit is [here](results/normalized_forward_control/audit.json). The state overlay is [accommodation raw vs normalized control](results/run/accommodation_raw_vs_normalized_forward_control.png).

## Figures

- [point_errors_by_fixation.png](results/run/point_errors_by_fixation.png)
- [scale_and_raw_size_by_fixation.png](results/run/scale_and_raw_size_by_fixation.png)
- [p4_inverse_radius_ratio_by_fixation.png](results/run/p4_inverse_radius_ratio_by_fixation.png)
- [p4_triangles_raw_vs_historical.png](results/run/p4_triangles_raw_vs_historical.png)
- [vertex_residual_vectors_native_px.png](results/run/vertex_residual_vectors_native_px.png)
- [accommodation_by_fixation.png](results/run/accommodation_by_fixation.png)
- [accommodation_by_source_row.png](results/run/accommodation_by_source_row.png)
- [accommodation_by_source_order.png](results/run/accommodation_by_source_order.png) (all complete rows, four capture panels; source-row order, not elapsed time)
- [accommodation_raw_vs_normalized_forward_control.png](results/run/accommodation_raw_vs_normalized_forward_control.png)

Scale-factor and inverse-radius plots are diagnostics, not fit normalization steps. Triangle plots use matched raw/historical valid frames in native coordinates without rotation alignment or radius normalization. See the [read-only historical comparison record](results/run/historical_comparison.json) for those fields.
