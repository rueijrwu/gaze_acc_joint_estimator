# G7 saved compact-check aggregation

This report aggregates already-saved compact JSON records only. It does not run inference, refit a model, or turn the compact diagnostic into G8. All scheduled slots, missing outcomes, and endpoints are retained.

Attempt: `/home/aplab/ACC/experiments/distortion_model/stage_04_center_and_dm0_calibration/results/g7_attempt_03`

## Input files

| File | SHA-256 | Slots | Scored | Unavailable | Unresolved |
|---|---|---:|---:|---:|---:|
| `continued_final.json` | `da9047b2d6252e49795a52c4c04fc17cb7b7d5e9eed7712030d02bf6f84d5a7b` | 300 | 270 | 30 | 0 |
| `continued_outer_04.json` | `4052bd60d45cf55c791c16920414e17fb54b1705a6db9b5ed5f1e0cfc9b6047b` | 300 | 270 | 30 | 0 |
| `continued_outer_08.json` | `da9047b2d6252e49795a52c4c04fc17cb7b7d5e9eed7712030d02bf6f84d5a7b` | 300 | 270 | 30 | 0 |

## Selected final (continued_final.json): pooled descriptive summaries

These are pooled slot-level descriptions, not equal-exposure scorecards. They retain the saved five schedule positions, including both endpoints.

### By held point

| Held P4 | Scored | Vector median (px) | p95 | max | x bias (px) | y bias (px) | x RMS (px) | y RMS (px) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 90 | 5.16494 | 11.3503 | 263.254 | 2.99602 | 4.90082 | 30.966 | 24.8612 |
| 1 | 90 | 5.64734 | 39.7873 | 333.292 | -4.79269 | 6.8667 | 21.8015 | 46.9606 |
| 2 | 90 | 4.04546 | 11.6874 | 199.025 | -1.39822 | 4.25899 | 11.0939 | 28.1775 |

### By saved schedule position

Position 1 and position 5 are the original per-exposure endpoints; no endpoint slots are dropped.

| Held P4 | Position | Endpoint | Scored | Vector median (px) | p95 | max | x bias (px) | y bias (px) | x RMS (px) | y RMS (px) |
|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | yes | 19 | 4.56337 | 38.958 | 263.254 | 8.82652 | 10.3385 | 47.3612 | 38.0063 |
| 0 | 2 | no | 18 | 5.57277 | 10.2673 | 10.972 | -1.34472 | 1.52633 | 4.57146 | 4.33941 |
| 0 | 3 | no | 19 | 5.0065 | 9.94606 | 10.6869 | -1.25035 | 1.03652 | 4.77944 | 3.75026 |
| 0 | 4 | no | 17 | 4.89028 | 11.4684 | 12.645 | -1.62888 | 1.23003 | 4.93488 | 4.44862 |
| 0 | 5 | yes | 17 | 5.33351 | 59.4755 | 263.209 | 10.4465 | 10.3861 | 49.975 | 40.029 |
| 1 | 1 | yes | 19 | 4.27845 | 68.362 | 333.292 | -6.94279 | 17.0243 | 28.0285 | 72.1593 |
| 1 | 2 | no | 18 | 6.6348 | 31.0068 | 37.172 | -4.13471 | 0.11238 | 13.5334 | 3.92806 |
| 1 | 3 | no | 19 | 5.23656 | 35.3161 | 45.779 | -2.36529 | -0.419953 | 15.5537 | 3.68181 |
| 1 | 4 | no | 17 | 5.77558 | 35.9909 | 48.2588 | -2.78262 | -0.432734 | 17.0945 | 3.88309 |
| 1 | 5 | yes | 17 | 5.61189 | 99.0403 | 333.231 | -7.80939 | 18.1091 | 29.6949 | 76.2169 |
| 2 | 1 | yes | 19 | 3.76443 | 35.046 | 199.025 | 0.713229 | 9.99284 | 16.0408 | 43.1458 |
| 2 | 2 | no | 18 | 4.31416 | 10.3712 | 11.5612 | -3.05932 | 0.180483 | 4.72035 | 4.082 |
| 2 | 3 | no | 19 | 4.64179 | 10.6055 | 11.2859 | -2.699 | 0.397054 | 4.82404 | 4.0779 |
| 2 | 4 | no | 17 | 4.2775 | 11.1463 | 11.7907 | -3.14515 | -0.226554 | 5.46047 | 3.61927 |
| 2 | 5 | yes | 17 | 2.93821 | 53.7443 | 198.997 | 1.20149 | 10.9708 | 16.8694 | 45.5356 |

### Bound versus non-bound slots

Pooled descriptive strata; these are not equal-exposure scorecards.

| State at bound | Scored | Vector median (px) | p95 | max | x bias (px) | y bias (px) | x RMS (px) | y RMS (px) |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| yes | 48 | 7.31436 | 263.239 | 333.292 | 1.50522 | 29.6334 | 51.1953 | 81.8969 |
| no | 222 | 4.81455 | 17.4009 | 48.2588 | -1.62068 | 0.0900236 | 8.03954 | 4.03865 |

## Complete held-point triples

For each complete eligible frame, `E_i²` is the mean of the three squared 2D native-pixel errors. `Gtheta_i²` and `GA_i²` are the mean of the three pairwise squared state differences. Exposure means are computed on squared quantities and then equally weighted; an absent exposure makes the all-exposure summary incomplete.

| Compact file | Eligible triples | Exposures present | Missing exposures | Equal-exposure E (px) | Gtheta (deg) | GA (D) |
|---|---:|---:|---:|---:|---:|---:|
| `continued_final.json` | 90 | 20/20 | none | 41.8138 | 1.50113 | 1.31867 |
| `continued_outer_04.json` | 90 | 20/20 | none | 41.8138 | 1.50113 | 1.31867 |
| `continued_outer_08.json` | 90 | 20/20 | none | 41.8138 | 1.50113 | 1.31867 |

### Largest frame contributions to the equal-exposure scorecard

Each contribution is the frame's squared metric divided by 20 times the number of eligible frames in that exposure. The table is ranked by E²; `Gtheta²` and `GA²` are that same frame's contributions, not their independent rank order. `Share` is fraction of the corresponding equal-exposure total. These are descriptive attributions and do not establish why a frame differs.

| Rank | Capture / exposure / row / source frame | Endpoint slots | Bound slots | E² (px²) | E² contribution | E² share | Gtheta² (deg²) | Gtheta² contribution | Gtheta² share | GA² (D²) | GA² contribution | GA² share |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | capture_2_detections.pkl / 6 / 6800 / 6800 | 3 | 3 | 73332.5 | 916.656 | 52.4285% | 92.0437 | 1.15055 | 51.0581% | 24 | 0.3 | 17.2523% |
| 2 | capture_2_detections.pkl / 5 / 6799 / 6799 | 3 | 3 | 73307.3 | 733.073 | 41.9284% | 91.8657 | 0.918657 | 40.7675% | 24 | 0.24 | 13.8018% |
| 3 | capture_2_detections.pkl / 8 / 20549 / 20549 | 0 | 1 | 816.515 | 10.2064 | 0.583761% | 1.70778 | 0.0213473 | 0.947336% | 5.22559 | 0.0653199 | 3.75639% |
| 4 | capture_2_detections.pkl / 8 / 19299 / 19299 | 0 | 1 | 738.735 | 9.23419 | 0.528153% | 1.5284 | 0.019105 | 0.847826% | 4.745 | 0.0593125 | 3.41092% |
| 5 | capture_1_detections.pkl / 4 / 20500 / 20500 | 3 | 2 | 601.864 | 7.5233 | 0.430298% | 1.3589 | 0.0169862 | 0.753804% | 10.3119 | 0.128899 | 7.41268% |
| 6 | capture_1_detections.pkl / 3 / 20499 / 20499 | 3 | 2 | 650.819 | 6.50819 | 0.372238% | 1.45747 | 0.0145747 | 0.646786% | 11.0332 | 0.110332 | 6.34492% |
| 7 | capture_2_detections.pkl / 8 / 18049 / 18049 | 0 | 1 | 501.572 | 6.26965 | 0.358595% | 0.92872 | 0.011609 | 0.515176% | 4.00467 | 0.0500583 | 2.87874% |
| 8 | capture_1_detections.pkl / 4 / 24399 / 24399 | 0 | 1 | 441.934 | 5.52418 | 0.315957% | 0.594606 | 0.00743258 | 0.329838% | 1.84211 | 0.0230264 | 1.3242% |
| 9 | capture_1_detections.pkl / 3 / 17999 / 17999 | 0 | 1 | 431.945 | 4.31945 | 0.247052% | 1.0221 | 0.010221 | 0.453581% | 6.60725 | 0.0660725 | 3.79968% |
| 10 | capture_1_detections.pkl / 4 / 23099 / 23099 | 0 | 1 | 337.951 | 4.22438 | 0.241615% | 0.402144 | 0.0050268 | 0.223076% | 1.35141 | 0.0168927 | 0.971458% |

The two largest E² rows contribute **94.3568%** of equal-exposure E². The two largest Gtheta² rows contribute **91.8257%** of equal-exposure Gtheta².

#### Independent top contributors by metric

| Metric | Rank | Capture / exposure / row / source frame | Frame squared metric | Weighted contribution | Share | Cumulative top-k share |
|---|---:|---|---:|---:|---:|---:|
| E² (px²) | 1 | capture_2_detections.pkl / 6 / 6800 / 6800 | 73332.5 | 916.656 | 52.4285% | 52.4285% |
| E² (px²) | 2 | capture_2_detections.pkl / 5 / 6799 / 6799 | 73307.3 | 733.073 | 41.9284% | 94.3568% |
| E² (px²) | 3 | capture_2_detections.pkl / 8 / 20549 / 20549 | 816.515 | 10.2064 | 0.583761% | 94.9406% |
| E² (px²) | 4 | capture_2_detections.pkl / 8 / 19299 / 19299 | 738.735 | 9.23419 | 0.528153% | 95.4688% |
| E² (px²) | 5 | capture_1_detections.pkl / 4 / 20500 / 20500 | 601.864 | 7.5233 | 0.430298% | 95.8991% |
| Gtheta² (deg²) | 1 | capture_2_detections.pkl / 6 / 6800 / 6800 | 92.0437 | 1.15055 | 51.0581% | 51.0581% |
| Gtheta² (deg²) | 2 | capture_2_detections.pkl / 5 / 6799 / 6799 | 91.8657 | 0.918657 | 40.7675% | 91.8257% |
| Gtheta² (deg²) | 3 | capture_2_detections.pkl / 8 / 20549 / 20549 | 1.70778 | 0.0213473 | 0.947336% | 92.773% |
| Gtheta² (deg²) | 4 | capture_2_detections.pkl / 8 / 19299 / 19299 | 1.5284 | 0.019105 | 0.847826% | 93.6208% |
| Gtheta² (deg²) | 5 | capture_1_detections.pkl / 4 / 20500 / 20500 | 1.3589 | 0.0169862 | 0.753804% | 94.3746% |
| GA² (D²) | 1 | capture_2_detections.pkl / 6 / 6800 / 6800 | 24 | 0.3 | 17.2523% | 17.2523% |
| GA² (D²) | 2 | capture_2_detections.pkl / 5 / 6799 / 6799 | 24 | 0.24 | 13.8018% | 31.0541% |
| GA² (D²) | 3 | capture_1_detections.pkl / 4 / 20500 / 20500 | 10.3119 | 0.128899 | 7.41268% | 38.4668% |
| GA² (D²) | 4 | capture_1_detections.pkl / 3 / 20499 / 20499 | 11.0332 | 0.110332 | 6.34492% | 44.8117% |
| GA² (D²) | 5 | capture_1_detections.pkl / 3 / 17999 / 17999 | 6.60725 | 0.0660725 | 3.79968% | 48.6114% |

#### Matched triple partitions

Complete triples are partitioned descriptively by whether any of the three scheduled held slots is an original endpoint (position 1 or 5), and whether any state is at a bound. RMS columns summarize squared frame metrics; median and p95 show the frame-level tails. These partitions are not causal comparisons.

| Partition | Frames | E RMS (px) | E median / p95 (px) | Gtheta RMS (deg) | Gtheta median / p95 (deg) | GA RMS (D) | GA median / p95 (D) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Endpoint: any endpoint | 36 | 64.3852 | 4.94411 / 86.8217 | 2.29885 | 0.13979 / 3.30161 | 1.49531 | 0.525194 / 3.71596 |
| Endpoint: interior only | 54 | 10.5433 | 5.75849 / 21.503 | 0.431867 | 0.176674 / 0.993585 | 1.14573 | 0.662286 / 2.29024 |
| Bound: any state at bound | 29 | 72.7074 | 13.7277 / 173.882 | 2.61642 | 0.53896 / 6.27352 | 2.02696 | 1.1625 / 4.26804 |
| Bound: no state at bound | 61 | 5.6313 | 4.93103 / 8.932 | 0.171593 | 0.138791 / 0.321244 | 0.726907 | 0.612804 / 1.22183 |

## Matched checkpoint coverage

Each checkpoint is compared with its same-start final snapshot on common eligible slot/frame IDs. Coverage gains/losses are reported separately from matched residual changes.

| Start | Checkpoint | Common slots | Lost | Gained | Common frames | Mean ΔE² (px²) |
|---|---|---:|---:|---:|---:|---:|
| continued | `continued_outer_04.json` | 270 | 0 | 0 | 90 | 7.336850768751507e-07 |
| continued | `continued_outer_08.json` | 270 | 0 | 0 | 90 | 0.0 |

## Interpretation limits

The subset state spreads `Gtheta` and `GA` measure internal agreement among overlapping retained-point inversions, not accuracy. Errors use native relative pixels and are not divided by inferred scale. This compact schedule is a progress diagnostic, not the full G8 evaluation. The slot and frame records in `summary.json` preserve the identities and outcomes behind these tables.
