# G7 saved compact-check aggregation

This report aggregates already-saved compact JSON records only. It does not run inference, refit a model, or turn the compact diagnostic into G8. All scheduled slots, missing outcomes, and endpoints are retained.

Attempt: `/home/aplab/ACC/experiments/distortion_model/stage_04_center_and_dm0_calibration/results/g7_attempt_02`

## Input files

| File | SHA-256 | Slots | Scored | Unavailable | Unresolved |
|---|---|---:|---:|---:|---:|
| `common_final.json` | `1cbc9640a3fc65128380b7522fb4c83daed1226116efea5fbb4e55b209a8e125` | 300 | 270 | 30 | 0 |
| `common_outer_08.json` | `d27f1286ffc3c6a11f030e88a20106102c7023f6ae6fdd01c6298f5818b6a7ca` | 300 | 270 | 30 | 0 |
| `common_outer_16.json` | `cc4762e1b73e3a85c9855cbe111511998f601fe7f42ae8163bde2d7c1daa0df4` | 300 | 270 | 30 | 0 |
| `common_outer_24.json` | `3a5d66b2189a4bc3a41d627d8e4a076d9796473732fea92dec8e635ceccad8d7` | 300 | 270 | 30 | 0 |
| `common_outer_32.json` | `cde964a70dc1ac7645d9b369cf8b229e31b9b67260d05ab6cc30249568dd2b74` | 300 | 270 | 30 | 0 |
| `perturbed_final.json` | `ecb50a5a6c71b6733155e36fa86464ecc2f92bfd2405482a645c4e7639112998` | 300 | 270 | 30 | 0 |
| `perturbed_outer_08.json` | `bf98f18eddccf491ac16b9ed3b148baf964825071c936e90f94c98f4014ae53b` | 300 | 270 | 30 | 0 |
| `perturbed_outer_16.json` | `84280dc4675001da04c0ede2cd348e39f4a0b7450cf250c72bda8eb15989a90a` | 300 | 270 | 30 | 0 |
| `perturbed_outer_24.json` | `643551cf0064542a6954bf035194018f799e6df8957743c8c6acde74e1f3c78d` | 300 | 270 | 30 | 0 |
| `perturbed_outer_32.json` | `f351c67cc15821d1af6d61f8c3583d97648ca55c507c34e591c783a8da926c94` | 300 | 270 | 30 | 0 |

## Selected perturbed final: pooled descriptive summaries

These are pooled slot-level descriptions, not equal-exposure scorecards. They retain the saved five schedule positions, including both endpoints.

### By held point

| Held P4 | Scored | Vector median (px) | p95 | max | x bias (px) | y bias (px) | x RMS (px) | y RMS (px) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 90 | 5.16485 | 11.3502 | 263.254 | 2.99602 | 4.90081 | 30.966 | 24.8612 |
| 1 | 90 | 5.64721 | 39.7874 | 333.292 | -4.7927 | 6.86668 | 21.8016 | 46.9606 |
| 2 | 90 | 4.0456 | 11.6872 | 199.025 | -1.39825 | 4.25899 | 11.094 | 28.1775 |

### By saved schedule position

Position 1 and position 5 are the original per-exposure endpoints; no endpoint slots are dropped.

| Held P4 | Position | Endpoint | Scored | Vector median (px) | p95 | max | x bias (px) | y bias (px) | x RMS (px) | y RMS (px) |
|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | yes | 19 | 4.56329 | 38.958 | 263.254 | 8.82654 | 10.3385 | 47.3612 | 38.0062 |
| 0 | 2 | no | 18 | 5.5727 | 10.2671 | 10.9718 | -1.34472 | 1.52631 | 4.57141 | 4.33933 |
| 0 | 3 | no | 19 | 5.00637 | 9.94594 | 10.6867 | -1.25036 | 1.03651 | 4.77939 | 3.75019 |
| 0 | 4 | no | 17 | 4.89015 | 11.4683 | 12.6449 | -1.62889 | 1.23003 | 4.93485 | 4.44856 |
| 0 | 5 | yes | 17 | 5.33348 | 59.4754 | 263.209 | 10.4465 | 10.3861 | 49.975 | 40.029 |
| 1 | 1 | yes | 19 | 4.27842 | 68.3621 | 333.292 | -6.94281 | 17.0242 | 28.0285 | 72.1592 |
| 1 | 2 | no | 18 | 6.63488 | 31.0069 | 37.172 | -4.1347 | 0.112375 | 13.5335 | 3.92809 |
| 1 | 3 | no | 19 | 5.23667 | 35.3161 | 45.779 | -2.36529 | -0.419965 | 15.5537 | 3.68183 |
| 1 | 4 | no | 17 | 5.77561 | 35.991 | 48.2588 | -2.78262 | -0.432755 | 17.0945 | 3.88311 |
| 1 | 5 | yes | 17 | 5.61172 | 99.0403 | 333.231 | -7.8094 | 18.1091 | 29.6949 | 76.2168 |
| 2 | 1 | yes | 19 | 3.76464 | 35.0455 | 199.025 | 0.713201 | 9.99283 | 16.0409 | 43.1457 |
| 2 | 2 | no | 18 | 4.3137 | 10.3713 | 11.5613 | -3.05932 | 0.180505 | 4.72033 | 4.08199 |
| 2 | 3 | no | 19 | 4.64188 | 10.6055 | 11.2855 | -2.69904 | 0.397063 | 4.82402 | 4.07788 |
| 2 | 4 | no | 17 | 4.27764 | 11.1463 | 11.7903 | -3.1452 | -0.22656 | 5.46043 | 3.61922 |
| 2 | 5 | yes | 17 | 2.93827 | 53.7438 | 198.997 | 1.20147 | 10.9708 | 16.8695 | 45.5355 |

### Bound versus non-bound slots

Pooled descriptive strata; these are not equal-exposure scorecards.

| State at bound | Scored | Vector median (px) | p95 | max | x bias (px) | y bias (px) | x RMS (px) | y RMS (px) |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| yes | 48 | 7.3143 | 263.238 | 333.292 | 1.50524 | 29.6333 | 51.1953 | 81.8968 |
| no | 222 | 4.8145 | 17.4003 | 48.2588 | -1.6207 | 0.090023 | 8.03955 | 4.03861 |

## Complete held-point triples

For each complete eligible frame, `E_i²` is the mean of the three squared 2D native-pixel errors. `Gtheta_i²` and `GA_i²` are the mean of the three pairwise squared state differences. Exposure means are computed on squared quantities and then equally weighted; an absent exposure makes the all-exposure summary incomplete.

| Compact file | Eligible triples | Exposures present | Missing exposures | Equal-exposure E (px) | Gtheta (deg) | GA (D) |
|---|---:|---:|---:|---:|---:|---:|
| `common_final.json` | 90 | 20/20 | none | 41.8142 | 1.50182 | 1.23825 |
| `common_outer_08.json` | 90 | 20/20 | none | 41.8308 | 1.50549 | 1.20338 |
| `common_outer_16.json` | 90 | 20/20 | none | 41.827 | 1.50485 | 1.20708 |
| `common_outer_24.json` | 90 | 20/20 | none | 41.8198 | 1.50305 | 1.22342 |
| `common_outer_32.json` | 90 | 20/20 | none | 41.8145 | 1.50184 | 1.23792 |
| `perturbed_final.json` | 90 | 20/20 | none | 41.8138 | 1.50113 | 1.31866 |
| `perturbed_outer_08.json` | 90 | 20/20 | none | 41.8269 | 1.50455 | 1.21094 |
| `perturbed_outer_16.json` | 90 | 20/20 | none | 41.8215 | 1.50379 | 1.21449 |
| `perturbed_outer_24.json` | 90 | 20/20 | none | 41.8189 | 1.50224 | 1.23134 |
| `perturbed_outer_32.json` | 90 | 20/20 | none | 41.8138 | 1.50113 | 1.31866 |

## Matched checkpoint coverage

Each checkpoint is compared with its same-start final snapshot on common eligible slot/frame IDs. Coverage gains/losses are reported separately from matched residual changes.

| Start | Checkpoint | Common slots | Lost | Gained | Common frames | Mean ΔE² (px²) |
|---|---|---:|---:|---:|---:|---:|
| common | `common_outer_08.json` | 270 | 0 | 0 | 90 | -1.3420070603409635 |
| common | `common_outer_16.json` | 270 | 0 | 0 | 90 | -1.0280990944878678 |
| common | `common_outer_24.json` | 270 | 0 | 0 | 90 | -0.452479061277026 |
| common | `common_outer_32.json` | 270 | 0 | 0 | 90 | -0.02384611155039251 |
| perturbed | `perturbed_outer_08.json` | 270 | 0 | 0 | 90 | -1.097844678174232 |
| perturbed | `perturbed_outer_16.json` | 270 | 0 | 0 | 90 | -0.6615474197282828 |
| perturbed | `perturbed_outer_24.json` | 270 | 0 | 0 | 90 | -0.46300121266351035 |
| perturbed | `perturbed_outer_32.json` | 270 | 0 | 0 | 90 | 2.09653071230578e-06 |

## Interpretation limits

The subset state spreads `Gtheta` and `GA` measure internal agreement among overlapping retained-point inversions, not accuracy. Errors use native relative pixels and are not divided by inferred scale. This compact schedule is a progress diagnostic, not the full G8 evaluation. The slot and frame records in `summary.json` preserve the identities and outcomes behind these tables.
