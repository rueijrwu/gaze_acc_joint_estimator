# Phase 8.3: retained x versus retained x/y

The all-nine-fold run completed 18 response-by-fold tasks in 57.05 seconds. Each response/family/method cell scheduled 160 frames (480 point slots). The retained-x inverse used 49 scalar starts per valid retained subset and reused the frozen conditional response; retained-xy reuses the certified frozen holdouts.

The primary table uses equal-fixation weighting over 20 exposure groups within each family. For each reported RMS, squared values are averaged within each exposure and then equally across exposures before taking the square root. Axis metrics use scored point slots; E, Gθ, G_A, and worst-point metrics use complete three-point frames. E is the pixel RMS across the three heldout point errors; Gθ and G_A are the RMS pairwise state disagreements.

| Response | Family | Retained | Scored / 480 | Complete / 160 | Boundary available / scored | x-axis RMS (px) | y-axis RMS (px) | E RMS (px) | Gθ RMS (°) | G_A RMS (D) | Worst point RMS (px) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline27 | gaze | x | 429 | 143 | 39 / 39 | 2.904 | 3.038 | 4.203 | 0.930 | 1.010 | 5.417 |
| baseline27 | gaze | xy | 429 | 143 | 23 / 23 | 2.149 | 2.662 | 3.421 | 0.269 | 0.311 | 4.385 |
| baseline27 | capture | x | 429 | 143 | 40 / 40 | 2.363 | 3.411 | 4.149 | 0.721 | 0.906 | 5.130 |
| baseline27 | capture | xy | 429 | 143 | 8 / 8 | 1.846 | 3.565 | 4.014 | 0.281 | 0.319 | 5.222 |
| strong_anchor37 | gaze | x | 423 | 137 | 82 / 80 | 4.023 | 3.814 | 5.550 | 1.141 | 1.379 | 7.706 |
| strong_anchor37 | gaze | xy | 429 | 143 | 29 / 29 | 3.804 | 2.868 | 4.764 | 0.440 | 0.530 | 6.199 |
| strong_anchor37 | capture | x | 418 | 132 | 41 / 41 | 2.680 | 3.823 | 4.692 | 0.745 | 1.135 | 6.137 |
| strong_anchor37 | capture | xy | 429 | 143 | 30 / 30 | 2.056 | 3.769 | 4.293 | 0.276 | 0.366 | 5.593 |

The branch table includes every available slot, including unscored ambiguous or weak-rank cases. `plausible 1 / >1` counts slots with one versus multiple cost-plausible branches. Rank and branch counts are retained in full in [verification.json](verification.json).

| Response | Family | Retained | Available / scored | Rank 1 / rank 2 | Plausible 1 / >1 |
|---|---|---:|---:|---:|---:|
| baseline27 | gaze | x | 429 / 429 | 0 / 429 | 429 / 0 |
| baseline27 | gaze | xy | 429 / 429 | 0 / 429 | 429 / 0 |
| baseline27 | capture | x | 429 / 429 | 0 / 429 | 429 / 0 |
| baseline27 | capture | xy | 429 / 429 | 0 / 429 | 429 / 0 |
| strong_anchor37 | gaze | x | 429 / 423 | 2 / 427 | 424 / 5 |
| strong_anchor37 | gaze | xy | 429 / 429 | 0 / 429 | 429 / 0 |
| strong_anchor37 | capture | x | 429 / 418 | 1 / 428 | 419 / 10 |
| strong_anchor37 | capture | xy | 429 / 429 | 0 / 429 | 429 / 0 |

Paired squared-error changes are `xy − x`, so negative values favor xy. Values shown are equal-fixation means: compute each exposure's mean change over eligible paired points or frames, then average over exposures. “All” uses every shared-scored point or complete frame; “interior pair” requires the point or complete frame to be interior under both methods. Counts give the number of paired points and complete frames. Pooled-weighted counterparts remain in `summary.json`.

| Response | Family | Pair stratum | Point n | Δ‖e‖² (px²) | Frame n | ΔE² (px²) | ΔGθ² (°²) | ΔG_A² (D²) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| baseline27 | gaze | all | 429 | -5.962 | 143 | -5.962 | -0.792 | -0.924 |
| baseline27 | gaze | interior pair | 386 | -5.748 | 121 | -4.911 | -0.630 | -0.751 |
| baseline27 | capture | all | 429 | -1.100 | 143 | -1.100 | -0.442 | -0.718 |
| baseline27 | capture | interior pair | 383 | -2.197 | 112 | -0.259 | -0.359 | -0.482 |
| strong_anchor37 | gaze | all | 423 | -8.589 | 137 | -8.762 | -1.107 | -1.633 |
| strong_anchor37 | gaze | interior pair | 334 | -0.233 | 84 | -8.386 | -1.016 | -1.370 |
| strong_anchor37 | capture | all | 418 | -3.211 | 132 | -3.532 | -0.468 | -1.140 |
| strong_anchor37 | capture | interior pair | 374 | -3.482 | 110 | -1.519 | -0.147 | -0.558 |

See [phase83_metrics.png](phase83_metrics.png) for the equal-fixation-weighted E/G comparison. [plot_results.py](plot_results.py) regenerates it from `summary.json`. [verification.json](verification.json) includes frozen-input and implementation snapshot hashes, raw-data/model provenance, row coverage, independent E/G and x prediction/cost/certificate checks, xy reuse, branch/rank distributions, paired membership checks, and regression test evidence. All checks pass: 1,280 joined frames, 1,127 independent complete-frame E/G checks, 1,699 x holdout recomputations, 1,716 selected-branch gradient/curvature/polish checks, 640 frozen xy reuse checks, and four paired-membership checks.
