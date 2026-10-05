# m2 vs piecewise: corrected-label runs

Targets for fixations 10-14 come from `fixation_intervals.json` + `target_overrides_v1.json` (capture 3 is not overridden); held-out demand is 3 D. Errors are model state minus nominal target (retrospective model-state deviations, not measured physiological accuracy). Gaze in degrees, A in diopters.

## M0: piecewise, theta anchor 1.0 deg

### Training convergence

| fold | stage | status | accepted iter | nfev | solver s | driver wall s |
|---|---|---|---|---|---|---|
| holdout3 | quadratic | converged | 31 | 43 | 118.9 |  |
| holdout3 | robust | converged | 7 | 11 | 14.5 | 144.0 |
| full | quadratic | converged | 39 | 49 | 265.1 |  |
| full | robust | converged | 8 | 13 | 26.3 | 303.0 |

### Coefficients (robust stage)

| coefficient | holdout3 | full |
|---|---|---|
| b@0.36 | 0.142992 | 0.14286 |
| b@2 | 0.121693 | 0.122109 |
| b@3 | 0.107037 | 0.107571 |
| b@4 | 0.0923804 | 0.0931098 |
| s@0.36 | 0.0480242 | 0.0479252 |
| s@2 | 0.0413422 | 0.0417302 |
| s@3 | 0.0399238 | 0.0393703 |
| s@4 | 0.0385055 | 0.0387508 |
| r0 | 0.726263 | 0.727064 |
| r1 | 0.00992357 | 0.0090961 |
| r2 | 0.00514459 | 0.00551615 |
| r3(a) | -0.0215196 | -0.0222181 |
| r4(a t) | -0.0018385 | -0.00141706 |
| r5(a t2) | 0.00224667 | 0.00208019 |
| p (frozen exponent) | 0.61885 | 0.608861 |

### Held-out fixations 10-14, holdout3-model anchor-free inverse vs targets

| fixation | target deg | n | gaze bias | gaze RMSE | A bias | A RMSE |
|---|---|---|---|---|---|---|
| 10 | -15.000 | 3616 | 0.139 | 0.151 | -0.121 | 0.339 |
| 11 | -7.500 | 3913 | -0.005 | 0.058 | 0.417 | 0.458 |
| 12 | 0.000 | 3998 | -0.148 | 0.151 | -0.205 | 0.227 |
| 13 | 7.500 | 3758 | 0.142 | 0.195 | -0.070 | 0.264 |
| 14 | 15.000 | 4160 | 0.262 | 0.458 | 0.150 | 0.386 |
| all | n/a | 19445 | 0.078 | 0.248 | 0.038 | 0.346 |

### Held-out fixations 10-14, full-model (same anchor-free inverse) vs targets

| fixation | target deg | n | gaze bias | gaze RMSE | A bias | A RMSE |
|---|---|---|---|---|---|---|
| 10 | -15.000 | 3616 | 0.017 | 0.088 | -0.120 | 0.327 |
| 11 | -7.500 | 3913 | -0.068 | 0.091 | 0.396 | 0.437 |
| 12 | 0.000 | 3998 | -0.166 | 0.168 | -0.216 | 0.236 |
| 13 | 7.500 | 3758 | 0.196 | 0.248 | -0.073 | 0.268 |
| 14 | 15.000 | 4160 | 0.379 | 0.520 | 0.167 | 0.387 |
| all | n/a | 19445 | 0.074 | 0.280 | 0.035 | 0.340 |

### Agreement: holdout3 inverse vs full inverse (holdout3 minus full)

| fixation | n | mean gaze diff | gaze RMSE | mean A diff | A RMSE |
|---|---|---|---|---|---|
| 10 | 3616 | 0.121 | 0.130 | -0.000 | 0.015 |
| 11 | 3913 | 0.063 | 0.068 | 0.021 | 0.021 |
| 12 | 3998 | 0.018 | 0.018 | 0.011 | 0.011 |
| 13 | 3758 | -0.054 | 0.061 | 0.003 | 0.005 |
| 14 | 4160 | -0.117 | 0.148 | -0.017 | 0.024 |
| all | 19445 | 0.003 | 0.098 | 0.003 | 0.017 |

### Monotonicity of d in theta, theta in [-20,20], A in [0,6]

| fold | min dd/dtheta | at theta | at A | strictly monotonic |
|---|---|---|---|---|
| holdout3 | 0.03567 | -20.0 | 6.00 | True |
| full | 0.03751 | -20.0 | 6.00 | True |

### Inverse diagnostics (frames)

| inverse | frames | ambiguous | stationarity unverified | theta bound | A bound | max cost |
|---|---|---|---|---|---|---|
| holdout3_inverse | 19445 | 0 | 0 | 0 | 0 | 3.83e-24 |
| full_inverse | 19445 | 0 | 0 | 0 | 0 | 4.73e-24 |

## M2a1: m2, theta anchor 1.0 deg

### Training convergence

| fold | stage | status | accepted iter | nfev | solver s | driver wall s |
|---|---|---|---|---|---|---|
| holdout3 | quadratic | converged | 46 | 51 | 197.8 |  |
| holdout3 | robust | converged | 10 | 16 | 24.2 | 233.0 |
| full | quadratic | not_converged | 48 | 54 | 290.0 |  |
| full | robust | converged | 11 | 18 | 33.9 | 337.0 |

### Coefficients (robust stage)

| coefficient | holdout3 | full |
|---|---|---|
| b0 | 0.157485 | 0.158559 |
| b1 | -0.0250966 | -0.0245345 |
| s0 | 0.748849 | 0.754301 |
| s1 | -0.129474 | -0.126091 |
| c20 | -0.0555728 | -0.0641545 |
| c21 | 0.078202 | 0.0804261 |
| c3 | 0.0366415 | 0.024417 |
| r0 | 0.725098 | 0.725967 |
| r1 | 0.010697 | 0.00991299 |
| r2 | 0.0130393 | 0.0120308 |
| r3 | -0.0310338 | -0.0314085 |
| r4 | -0.00255521 | -0.00208531 |
| r5 | -0.00107997 | -0.000699572 |

### Held-out fixations 10-14, holdout3-model anchor-free inverse vs targets

| fixation | target deg | n | gaze bias | gaze RMSE | A bias | A RMSE |
|---|---|---|---|---|---|---|
| 10 | -15.000 | 3616 | -0.311 | 0.394 | -0.128 | 0.363 |
| 11 | -7.500 | 3913 | 0.021 | 0.067 | 0.229 | 0.308 |
| 12 | 0.000 | 3998 | 0.302 | 0.307 | -0.405 | 0.418 |
| 13 | 7.500 | 3758 | 0.589 | 0.625 | -0.093 | 0.303 |
| 14 | 15.000 | 4160 | -0.505 | 0.586 | 0.115 | 0.369 |
| all | n/a | 19445 | 0.014 | 0.445 | -0.054 | 0.355 |

### Held-out fixations 10-14, full-model (same anchor-free inverse) vs targets

| fixation | target deg | n | gaze bias | gaze RMSE | A bias | A RMSE |
|---|---|---|---|---|---|---|
| 10 | -15.000 | 3616 | -0.302 | 0.390 | -0.130 | 0.357 |
| 11 | -7.500 | 3913 | 0.084 | 0.105 | 0.264 | 0.333 |
| 12 | 0.000 | 3998 | 0.260 | 0.265 | -0.363 | 0.377 |
| 13 | 7.500 | 3758 | 0.484 | 0.525 | -0.092 | 0.300 |
| 14 | 15.000 | 4160 | -0.444 | 0.536 | 0.103 | 0.367 |
| all | n/a | 19445 | 0.013 | 0.400 | -0.041 | 0.349 |

### Agreement: holdout3 inverse vs full inverse (holdout3 minus full)

| fixation | n | mean gaze diff | gaze RMSE | mean A diff | A RMSE |
|---|---|---|---|---|---|
| 10 | 3616 | -0.010 | 0.011 | 0.002 | 0.007 |
| 11 | 3913 | -0.063 | 0.063 | -0.035 | 0.035 |
| 12 | 3998 | 0.042 | 0.042 | -0.042 | 0.042 |
| 13 | 3758 | 0.105 | 0.105 | -0.001 | 0.002 |
| 14 | 4160 | -0.060 | 0.061 | 0.012 | 0.012 |
| all | 19445 | 0.002 | 0.064 | -0.013 | 0.025 |

### Monotonicity of d in theta, theta in [-20,20], A in [0,6]

| fold | min dd/dtheta | at theta | at A | strictly monotonic |
|---|---|---|---|---|
| holdout3 | 0.02747 | -13.2 | 6.00 | True |
| full | 0.02617 | -18.9 | 6.00 | True |

### Inverse diagnostics (frames)

| inverse | frames | ambiguous | stationarity unverified | theta bound | A bound | max cost |
|---|---|---|---|---|---|---|
| holdout3_inverse | 19445 | 0 | 0 | 0 | 0 | 7.62e-24 |
| full_inverse | 19445 | 0 | 0 | 0 | 0 | 9.74e-24 |

## M2a01: m2, theta anchor 0.1 deg

### Training convergence

| fold | stage | status | accepted iter | nfev | solver s | driver wall s |
|---|---|---|---|---|---|---|
| holdout3 | quadratic | converged | 18 | 19 | 82.9 |  |
| holdout3 | robust | not_converged | 19 | 32 | 64.4 | 160.0 |
| full | quadratic | converged | 18 | 19 | 111.0 |  |
| full | robust | converged | 21 | 38 | 103.0 | 228.0 |

### Coefficients (robust stage)

| coefficient | holdout3 | full |
|---|---|---|
| b0 | 0.147961 | 0.14832 |
| b1 | -0.0164562 | -0.0160365 |
| s0 | 0.753223 | 0.754248 |
| s1 | -0.116488 | -0.113446 |
| c20 | 0.0116732 | 0.0036897 |
| c21 | 0.00436493 | 0.00857118 |
| c3 | 0.00825915 | 0.00234919 |
| r0 | 0.723159 | 0.724113 |
| r1 | 0.0104532 | 0.0081085 |
| r2 | 0.0199517 | 0.0183215 |
| r3 | -0.0294044 | -0.0297743 |
| r4 | -0.00250593 | -0.000912074 |
| r5 | -0.00637159 | -0.00575379 |

### Held-out fixations 10-14, holdout3-model anchor-free inverse vs targets

| fixation | target deg | n | gaze bias | gaze RMSE | A bias | A RMSE |
|---|---|---|---|---|---|---|
| 10 | -15.000 | 3616 | -0.027 | 0.102 | -0.145 | 0.312 |
| 11 | -7.500 | 3913 | 0.007 | 0.060 | 0.275 | 0.346 |
| 12 | 0.000 | 3998 | -0.030 | 0.046 | -0.407 | 0.421 |
| 13 | 7.500 | 3758 | 0.234 | 0.298 | -0.136 | 0.313 |
| 14 | 15.000 | 4160 | 0.062 | 0.387 | 0.193 | 0.388 |
| all | n/a | 19445 | 0.049 | 0.229 | -0.040 | 0.360 |

### Held-out fixations 10-14, full-model (same anchor-free inverse) vs targets

| fixation | target deg | n | gaze bias | gaze RMSE | A bias | A RMSE |
|---|---|---|---|---|---|---|
| 10 | -15.000 | 3616 | -0.021 | 0.103 | -0.163 | 0.310 |
| 11 | -7.500 | 3913 | 0.027 | 0.066 | 0.309 | 0.371 |
| 12 | 0.000 | 3998 | -0.042 | 0.054 | -0.349 | 0.366 |
| 13 | 7.500 | 3758 | 0.173 | 0.249 | -0.127 | 0.313 |
| 14 | 15.000 | 4160 | 0.034 | 0.387 | 0.144 | 0.376 |
| all | n/a | 19445 | 0.034 | 0.218 | -0.034 | 0.349 |

### Agreement: holdout3 inverse vs full inverse (holdout3 minus full)

| fixation | n | mean gaze diff | gaze RMSE | mean A diff | A RMSE |
|---|---|---|---|---|---|
| 10 | 3616 | -0.005 | 0.006 | 0.018 | 0.022 |
| 11 | 3913 | -0.021 | 0.021 | -0.034 | 0.034 |
| 12 | 3998 | 0.013 | 0.013 | -0.058 | 0.058 |
| 13 | 3758 | 0.061 | 0.061 | -0.009 | 0.010 |
| 14 | 4160 | 0.028 | 0.028 | 0.049 | 0.050 |
| all | 19445 | 0.015 | 0.032 | -0.007 | 0.040 |

### Monotonicity of d in theta, theta in [-20,20], A in [0,6]

| fold | min dd/dtheta | at theta | at A | strictly monotonic |
|---|---|---|---|---|
| holdout3 | 0.03401 | -12.2 | 6.00 | True |
| full | 0.03278 | -20.0 | 6.00 | True |

### Inverse diagnostics (frames)

| inverse | frames | ambiguous | stationarity unverified | theta bound | A bound | max cost |
|---|---|---|---|---|---|---|
| holdout3_inverse | 19445 | 0 | 0 | 0 | 0 | 7.59e-24 |
| full_inverse | 19445 | 0 | 0 | 0 | 0 | 7.51e-24 |

