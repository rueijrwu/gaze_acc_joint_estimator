# Phase 8.3 retained-channel audit follow-up

Differential-y gives better overall, axis, and worst-point prediction errors than x and xy in baseline27, but does not uniformly improve state agreement or empirical support. Capture-family empirical-A slots are true/false 315/114 under differential-y versus 343/86 under xy, and complete empirical-state support is 99 versus 111 frames (empirical-state plus measured-P1: 73 versus 85). Gaze-family state disagreement is higher than xy, and common-y-only is worse, especially for capture. No mask is promoted because selection guards were not predeclared.

The frozen-response follow-up completed 54 jobs in 172.5335 seconds using 12 workers with OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1. Historical sources remained unchanged. The saved full suite records 73 passed in 11.65s.

Each score row below uses equal-exposure aggregation over the saved family frame set. E is excluded-point vector RMS (px); Gθ and G_A are pairwise latent-state disagreement; worst is the worst excluded point. Axis RMS uses all scored held-out slots. Coverage retains scheduled frames/slots and complete-triple counts; support counts are T/F/U over all scheduled slots. The complete-frame IDs and detailed cohort memberships are preserved in `support_cohorts.json`.

| Response | Family | Mask | Scheduled frames / slots | Input-eligible / scored slots | Complete triples | E px | Gθ deg | G_A D | Worst px | x px | y px | Empirical θ T/F/U | Empirical A T/F/U | P1 valid T/F/U | Context inside/outside/U | Parity seen/not/U | Absent full exposures |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline27 | gaze | x | 160 / 480 | 429 / 429 | 143 | 4.203 | 0.9299 | 1.0101 | 5.417 | 2.904 | 3.038 | 276/153/51 | 382/47/51 | 456/24/0 | 348/108/24 | 456/0/24 | 0 |
| baseline27 | gaze | common-y | 160 / 480 | 429 / 429 | 143 | 5.087 | 0.9863 | 1.0958 | 6.820 | 3.635 | 3.558 | 276/153/51 | 369/60/51 | 456/24/0 | 348/108/24 | 456/0/24 | 0 |
| baseline27 | gaze | differential-y | 160 / 480 | 429 / 429 | 143 | 3.326 | 0.3708 | 0.3703 | 3.954 | 2.147 | 2.540 | 276/153/51 | 394/35/51 | 456/24/0 | 348/108/24 | 456/0/24 | 0 |
| baseline27 | gaze | common+differential-y | 160 / 480 | 429 / 429 | 143 | 3.421 | 0.2694 | 0.3111 | 4.385 | 2.149 | 2.662 | 276/153/51 | 397/32/51 | 456/24/0 | 348/108/24 | 456/0/24 | 0 |
| baseline27 | gaze | xy | 160 / 480 | 429 / 429 | 143 | 3.421 | 0.2694 | 0.3111 | 4.385 | 2.149 | 2.662 | 276/153/51 | 397/32/51 | 456/24/0 | 348/108/24 | 456/0/24 | 0 |
| baseline27 | capture | x | 160 / 480 | 429 / 429 | 143 | 4.149 | 0.7213 | 0.9057 | 5.130 | 2.363 | 3.411 | 428/1/51 | 325/104/51 | 456/24/0 | 354/102/24 | 456/0/24 | 0 |
| baseline27 | capture | common-y | 160 / 480 | 429 / 429 | 143 | 7.593 | 1.4361 | 1.4378 | 9.668 | 5.669 | 5.052 | 395/34/51 | 375/54/51 | 456/24/0 | 354/102/24 | 456/0/24 | 0 |
| baseline27 | capture | differential-y | 160 / 480 | 429 / 429 | 143 | 3.586 | 0.2782 | 0.3019 | 4.146 | 1.642 | 3.188 | 429/0/51 | 315/114/51 | 456/24/0 | 354/102/24 | 456/0/24 | 0 |
| baseline27 | capture | common+differential-y | 160 / 480 | 429 / 429 | 143 | 4.014 | 0.2806 | 0.3194 | 5.222 | 1.846 | 3.565 | 429/0/51 | 343/86/51 | 456/24/0 | 354/102/24 | 456/0/24 | 0 |
| baseline27 | capture | xy | 160 / 480 | 429 / 429 | 143 | 4.014 | 0.2806 | 0.3194 | 5.222 | 1.846 | 3.565 | 429/0/51 | 343/86/51 | 456/24/0 | 354/102/24 | 456/0/24 | 0 |
| strong_anchor37 | gaze | x | 160 / 480 | 429 / 423 | 137 | 5.550 | 1.1406 | 1.3792 | 7.706 | 4.023 | 3.814 | 276/153/51 | 392/37/51 | 456/24/0 | 348/108/24 | 456/0/24 | 0 |
| strong_anchor37 | gaze | common-y | 160 / 480 | 429 / 429 | 143 | 6.298 | 1.0839 | 1.4859 | 8.524 | 5.058 | 3.752 | 276/153/51 | 387/42/51 | 456/24/0 | 348/108/24 | 456/0/24 | 0 |
| strong_anchor37 | gaze | differential-y | 160 / 480 | 429 / 429 | 143 | 4.762 | 0.5831 | 0.7434 | 5.817 | 3.729 | 2.961 | 276/153/51 | 404/25/51 | 456/24/0 | 348/108/24 | 456/0/24 | 0 |
| strong_anchor37 | gaze | common+differential-y | 160 / 480 | 429 / 429 | 143 | 4.764 | 0.4397 | 0.5297 | 6.199 | 3.804 | 2.868 | 276/153/51 | 401/28/51 | 456/24/0 | 348/108/24 | 456/0/24 | 0 |
| strong_anchor37 | gaze | xy | 160 / 480 | 429 / 429 | 143 | 4.764 | 0.4397 | 0.5297 | 6.199 | 3.804 | 2.868 | 276/153/51 | 401/28/51 | 456/24/0 | 348/108/24 | 456/0/24 | 0 |
| strong_anchor37 | capture | x | 160 / 480 | 429 / 418 | 132 | 4.692 | 0.7445 | 1.1352 | 6.137 | 2.680 | 3.823 | 429/0/51 | 304/125/51 | 456/24/0 | 354/102/24 | 456/0/24 | 0 |
| strong_anchor37 | capture | common-y | 160 / 480 | 429 / 429 | 143 | 7.410 | 1.0455 | 1.4537 | 9.232 | 5.413 | 5.061 | 402/27/51 | 338/91/51 | 456/24/0 | 354/102/24 | 456/0/24 | 0 |
| strong_anchor37 | capture | differential-y | 160 / 480 | 429 / 429 | 143 | 3.773 | 0.1820 | 0.2356 | 4.290 | 1.686 | 3.375 | 429/0/51 | 301/128/51 | 456/24/0 | 354/102/24 | 456/0/24 | 0 |
| strong_anchor37 | capture | common+differential-y | 160 / 480 | 429 / 429 | 143 | 4.293 | 0.2764 | 0.3665 | 5.593 | 2.056 | 3.769 | 419/10/51 | 320/109/51 | 456/24/0 | 354/102/24 | 456/0/24 | 0 |
| strong_anchor37 | capture | xy | 160 / 480 | 429 / 429 | 143 | 4.293 | 0.2764 | 0.3665 | 5.593 | 2.056 | 3.769 | 419/10/51 | 320/109/51 | 456/24/0 | 354/102/24 | 456/0/24 | 0 |

The paired table reports equal-exposure means of squared changes (candidate mask minus x) among exact shared interior point/frame memberships. Negative values indicate smaller squared errors/disagreement under the candidate mask. Each 3-mask × response × family comparison retains its paired membership hash in `summary.json`.

| Response | Family | Mask vs x | Shared interior points | Shared interior complete frames | Δvector error² px² | Δx error² px² | Δy error² px² | ΔE² px² | ΔGθ² deg² | ΔG_A² D² | Δworst² px² |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline27 | capture | common-y | 374 | 104 | 46.7535 | 32.1122 | 14.6413 | 25.8370 | 1.1705 | 0.6627 | 46.2576 |
| baseline27 | capture | common+differential-y | 383 | 112 | -2.1974 | -2.4252 | 0.2278 | -0.2586 | -0.3594 | -0.4823 | 1.9279 |
| baseline27 | capture | differential-y | 384 | 117 | -9.5468 | -6.3069 | -3.2399 | -3.5024 | -0.3503 | -0.4711 | -7.7134 |
| baseline27 | gaze | common-y | 372 | 111 | 9.1477 | 6.1951 | 2.9527 | 5.0386 | -0.0564 | -0.0003 | 11.1042 |
| baseline27 | gaze | common+differential-y | 386 | 121 | -5.7484 | -3.9500 | -1.7984 | -4.9115 | -0.6299 | -0.7509 | -8.8013 |
| baseline27 | gaze | differential-y | 387 | 121 | -11.2612 | -7.0816 | -4.1797 | -5.9515 | -0.5983 | -0.7507 | -13.0096 |
| strong_anchor37 | capture | common-y | 373 | 107 | 33.3570 | 22.2766 | 11.0803 | 41.0211 | 0.4086 | 0.9024 | 58.4408 |
| strong_anchor37 | capture | common+differential-y | 374 | 110 | -3.4824 | -3.5881 | 0.1058 | -1.5187 | -0.1472 | -0.5575 | -2.5834 |
| strong_anchor37 | capture | differential-y | 371 | 110 | -6.6818 | -4.0167 | -2.6651 | -6.5474 | -0.1956 | -0.6347 | -18.1065 |
| strong_anchor37 | gaze | common-y | 335 | 81 | 8.4488 | 8.3652 | 0.0836 | 3.2388 | 0.0211 | 0.0398 | 4.3363 |
| strong_anchor37 | gaze | common+differential-y | 334 | 84 | -0.2333 | 3.4951 | -3.7284 | -8.3865 | -1.0156 | -1.3695 | -22.9498 |
| strong_anchor37 | gaze | differential-y | 336 | 84 | -7.8546 | -4.6450 | -3.2096 | -8.8424 | -0.9255 | -1.2840 | -26.5096 |

## Support cohorts

Support cohorts require all three subset slots in a complete frame to meet the named support rule. Unknown support values exclude the frame and are counted; they are not imputed. `interior` means the inverse solutions are interior to the declared numerical bounds, while `empirical_state` means each state component lies inside its corresponding training-state extrema. These categories are independent: numerical interior status is not evidence of training support. Cohort rows use candidate-specific frame identities and are descriptive; do not rank candidates from those rows. Candidate ranking requires comparison on shared identities, as in the paired table above.

| Response / family / mask | Cohort | Complete frames | Scored slots | Contributing / expected exposures | Absent exposures | Unknown support frames / slots (all scheduled) | Unknown complete frames / slots excluded | E px | Gθ deg | G_A D | Worst px | x px | y px |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline27/capture/x_y_difference | full | 143 | 429 | 20 / 20 | 0 | 0 / 0 | 0 / 0 | 3.586 | 0.278 | 0.302 | 4.146 | 1.642 | 3.188 |
| baseline27/capture/x_y_difference | interior | 130 | 390 | 19 / 20 | 1 | 0 / 0 | 0 / 0 | 3.425 | 0.286 | 0.306 | 3.945 | 1.571 | 3.043 |
| baseline27/capture/x_y_difference | empirical_state | 99 | 297 | 15 / 20 | 5 | 17 / 51 | 0 / 0 | 3.021 | 0.305 | 0.335 | 3.471 | 1.483 | 2.632 |
| baseline27/capture/x_y_difference | measured_P1 | 111 | 333 | 17 / 20 | 3 | 8 / 24 | 0 / 0 | 3.706 | 0.156 | 0.176 | 4.266 | 1.365 | 3.445 |
| baseline27/capture/x_y_difference | empirical_state_and_P1 | 73 | 219 | 12 / 20 | 8 | 17 / 51 | 0 / 0 | 2.933 | 0.157 | 0.183 | 3.332 | 1.078 | 2.727 |
| baseline27/capture/xy | full | 143 | 429 | 20 / 20 | 0 | 0 / 0 | 0 / 0 | 4.014 | 0.281 | 0.319 | 5.222 | 1.846 | 3.565 |
| baseline27/capture/xy | interior | 137 | 411 | 20 / 20 | 0 | 0 / 0 | 0 / 0 | 3.909 | 0.281 | 0.321 | 5.135 | 1.894 | 3.420 |
| baseline27/capture/xy | empirical_state | 111 | 333 | 16 / 20 | 4 | 17 / 51 | 0 / 0 | 3.546 | 0.297 | 0.343 | 4.691 | 1.856 | 3.021 |
| baseline27/capture/xy | measured_P1 | 111 | 333 | 17 / 20 | 3 | 8 / 24 | 0 / 0 | 4.212 | 0.270 | 0.311 | 5.525 | 1.620 | 3.888 |
| baseline27/capture/xy | empirical_state_and_P1 | 85 | 255 | 13 / 20 | 7 | 17 / 51 | 0 / 0 | 3.564 | 0.268 | 0.328 | 4.750 | 1.653 | 3.157 |
| baseline27/gaze/x_y_difference | full | 143 | 429 | 20 / 20 | 0 | 0 / 0 | 0 / 0 | 3.326 | 0.371 | 0.370 | 3.954 | 2.147 | 2.540 |
| baseline27/gaze/x_y_difference | interior | 130 | 390 | 19 / 20 | 1 | 0 / 0 | 0 / 0 | 3.248 | 0.370 | 0.360 | 3.872 | 2.043 | 2.525 |
| baseline27/gaze/x_y_difference | empirical_state | 86 | 258 | 12 / 20 | 8 | 17 / 51 | 0 / 0 | 3.231 | 0.369 | 0.356 | 3.828 | 1.934 | 2.589 |
| baseline27/gaze/x_y_difference | measured_P1 | 109 | 327 | 17 / 20 | 3 | 8 / 24 | 0 / 0 | 3.106 | 0.232 | 0.272 | 3.669 | 1.682 | 2.610 |
| baseline27/gaze/x_y_difference | empirical_state_and_P1 | 63 | 189 | 10 / 20 | 10 | 17 / 51 | 0 / 0 | 2.717 | 0.164 | 0.187 | 3.168 | 1.177 | 2.449 |
| baseline27/gaze/xy | full | 143 | 429 | 20 / 20 | 0 | 0 / 0 | 0 / 0 | 3.421 | 0.269 | 0.311 | 4.385 | 2.149 | 2.662 |
| baseline27/gaze/xy | interior | 131 | 393 | 19 / 20 | 1 | 0 / 0 | 0 / 0 | 3.372 | 0.273 | 0.324 | 4.332 | 2.110 | 2.630 |
| baseline27/gaze/xy | empirical_state | 87 | 261 | 12 / 20 | 8 | 17 / 51 | 0 / 0 | 3.368 | 0.285 | 0.289 | 4.414 | 1.994 | 2.715 |
| baseline27/gaze/xy | measured_P1 | 109 | 327 | 17 / 20 | 3 | 8 / 24 | 0 / 0 | 3.265 | 0.220 | 0.310 | 4.226 | 1.723 | 2.773 |
| baseline27/gaze/xy | empirical_state_and_P1 | 63 | 189 | 10 / 20 | 10 | 17 / 51 | 0 / 0 | 3.029 | 0.214 | 0.276 | 4.014 | 1.360 | 2.707 |

## Information and diagnostic checks

The combined-y retained transform matched full xy over 1920 held-out slots: 1920 branch-cluster comparisons were equivalent; rank/availability/ambiguity disagreements: 0. Maximum selected-state differences were 4.39e-08° gaze and 5.3e-08 D accommodation; maximum objective-cost difference was 2.76e-10. These finite multistart checks compare discovered clusters and do not establish global branch completeness.

| Response | Mask | Diagnostic profile cases | Tasks |
|---|---|---:|---:|
| baseline27 | common-y | 0 | 9 |
| baseline27 | common+differential-y | 0 | 9 |
| baseline27 | differential-y | 0 | 9 |
| strong_anchor37 | common-y | 0 | 9 |
| strong_anchor37 | common+differential-y | 0 | 9 |
| strong_anchor37 | differential-y | 0 | 9 |

For these saved cases no new 65-grid profile diagnostic was triggered by the three transformed-y masks. The combined-y-versus-xy numerical equivalence is a finite multistart consistency result; it does not upgrade the heuristic inverse to a global completeness guarantee.

The supplemental retained-x profile audit covered 17 earlier cases (3 rank-weak, 15 ambiguous, with overlap); profiles were available for all, with 0 newly discovered branches and 0 lower-cost cases. Primary scores were unchanged. The finite grid is not a global completeness guarantee.

The y-mask jobs reuse frozen calibration responses and do not include a nested calibration study. In the four response/family cells, differential-y minus x had negative shared-interior changes in E², Gθ², G_A², and worst-point²; common-y minus x had positive E² changes in all four. For baseline27 capture, combined-y minus x reduced E² by 0.2586 px² while increasing worst-point² by 1.9279 px². A possible shared-bias clue is the training capture y-mean sequence: captures 2–4 were −0.848, +0.529, and +0.324 versus strong-anchor means −0.860, +0.586, and +0.351. This is a correlated secondary observation motivating train-only capture/context residual checks, not a completed nested calibration study. The xy baseline remains the reference. Captures 5 and 6 were untouched. A future calibration comparison should predeclare guards and use denser grouped conditions; these development results do not decide that choice.

## Independent verification

The saved [verification record](verification.json) reports all checks passed with 0 errors. It checked 54 transformed-response tasks, 5,148 retained covariance transforms, 5,750 objective/branch cases, 5,733 gradient/curvature/polish cases, 252,252 archived start records, 1,716 joined frame E/G/axis metrics, and 30,720 support records. Historical checks covered 234 joins and preserved 8,320 primary scores; 12 paired-membership checks and 3 representative full-grid inversions also passed. All 2,309 archived historical file hashes matched.

Reproduce from the repository root with the same single-threaded BLAS environment:

```sh
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python full_position/audit83.py --output experiments/full_position/phase83_audit_followup_v1 --workers 12 --mode all
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/full_position/phase83_audit_followup_v1/profile_existing_x.py
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/full_position/phase83_audit_followup_v1/verify_results.py
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/full_position/phase83_audit_followup_v1/summarize_results.py
```

The run entry point is [`full_position/audit83.py`](../../../full_position/audit83.py); the remaining commands profile the retained-x cases, verify saved artifacts, and regenerate this report and the support cohorts.
