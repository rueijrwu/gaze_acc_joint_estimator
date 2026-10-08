# Three certified models: full-period common-frame comparison

This comparison uses the saved log, square-root, and quadratic model results. It excludes the uncertified linear law, as requested.

Common complete frames: 89,172 / 100,090.
Expected exposures present: 20 / 20.

## Equal-exposure RMS on the shared cohort

| Law | P4 cross-prediction (px) | Gaze agreement (deg) | Accommodation agreement (D) | Worst-point error (px) |
|---|---:|---:|---:|---:|
| ar27_log | 4.79293 | 0.251929 | 0.0358291 | 6.29833 |
| ar27_sqrt | 7.31904 | 0.889524 | 0.105375 | 9.99848 |
| ar27_quadratic | 7.61178 | 0.483289 | 0.220705 | 10.5096 |

Rank by metric (lower is better):
- P4 cross-prediction (px): ar27_log < ar27_sqrt < ar27_quadratic
- Gaze agreement (deg): ar27_log < ar27_quadratic < ar27_sqrt
- Accommodation agreement (D): ar27_log < ar27_sqrt < ar27_quadratic
- Worst-point error (px): ar27_log < ar27_sqrt < ar27_quadratic

This is a common-cohort internal consistency comparison. It does not establish independent accuracy, choose a deployment model, or rank the excluded linear law.
