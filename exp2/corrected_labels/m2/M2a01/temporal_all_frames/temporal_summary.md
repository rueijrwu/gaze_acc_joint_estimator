# Temporal correlation of gaze and accommodation: M2a01

Held-out fixations 10-14 (3 D). Anchor-free inverse; 1 ms frames. Bands: slow > 0.5 s, mid 50-500 ms, fast < 50 ms.
Statistics use real (non-interpolated) samples only.

## Holdout-3 model

| fixation | frames (filled; glitch-flagged) | SD gaze (deg) | SD A (D) | r raw | r slow | r mid | r fast | slope slow | slope mid | slope fast | predicted slope via rho4 | predicted slope via d |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 | 3616 (384; 0) | 0.099 | 0.277 | -0.87 | -0.95 | -0.60 | -0.69 | -0.30 | -0.39 | -0.60 | -0.40 | -8.35 |
| 11 | 3913 (7; 0) | 0.060 | 0.210 | -0.19 | -0.29 | +0.14 | -0.03 | -0.07 | +0.10 | -0.04 | +0.07 | -27.07 |
| 12 | 3998 (2; 0) | 0.035 | 0.108 | +0.68 | +0.68 | +0.55 | +0.73 | +0.19 | +0.24 | +0.48 | +0.41 | +17.04 |
| 13 | 3758 (2; 0) | 0.184 | 0.282 | +0.93 | +0.95 | +0.82 | +0.92 | +0.60 | +0.59 | +0.80 | +0.77 | +6.66 |
| 14 | 4160 (0; 0) | 0.382 | 0.337 | +0.88 | +0.98 | +0.20 | +0.89 | +0.98 | +0.42 | +1.83 | +1.01 | +4.84 |

Slopes are gaze change per accommodation change (deg / D). Predicted slopes are -(dd/dA)/(dd/dtheta) (error enters via rho4) and -(drho/dA)/(drho/dtheta) (error enters via d), medians over the fixation.

| fixation | share of gaze variance slow / mid / fast | share of A variance slow / mid / fast | peak xcorr slow (lag ms) | peak xcorr mid+fast (lag ms) | r(d, rho4) slow / mid / fast |
|---|---|---|---|---|---|
| 10 | 0.74 / 0.08 / 0.18 | 0.95 / 0.02 / 0.03 | -0.96 (-87) | -0.66 (0) | -0.59 / +0.02 / +0.36 |
| 11 | 0.60 / 0.22 / 0.18 | 0.95 / 0.04 / 0.01 | -0.29 (65) | -0.17 (-186) | +0.49 / -0.07 / +0.04 |
| 12 | 0.57 / 0.32 / 0.11 | 0.79 / 0.19 / 0.03 | +0.69 (-29) | +0.58 (0) | +0.74 / +0.46 / -0.14 |
| 13 | 0.88 / 0.10 / 0.02 | 0.91 / 0.08 / 0.01 | +0.96 (72) | +0.83 (0) | +0.65 / +0.49 / -0.06 |
| 14 | 0.73 / 0.14 / 0.13 | 0.92 / 0.04 / 0.04 | +0.99 (81) | +0.50 (0) | +0.12 / +0.67 / -0.47 |

Raw-measurement attribution (relative changes; rho4 = S4/S1, d = m/S1). Share of the rho4 band variance from the S4 term, share of the d band variance from the S1 term, SD of the relative S1 and S4 changes, and the correlation of the accommodation band with them.

| fixation | band | SD dS1/S1 (%) | SD dS4/S4 (%) | rho4 share from S4 | d share from S1 | r(A, dS4/S4) | r(A, dS1/S1) | r(gaze, dS1/S1) |
|---|---|---|---|---|---|---|---|---|
| 10 | slow | 0.104 | 0.203 | 0.67 | 0.24 | -0.99 | +0.96 | -0.96 |
| 10 | mid | 0.011 | 0.046 | 0.94 | -0.03 | -0.97 | +0.28 | -0.51 |
| 10 | fast | 0.007 | 0.048 | 0.98 | -0.00 | -0.99 | +0.12 | -0.19 |
| 11 | slow | 0.027 | 0.230 | 1.10 | 0.01 | -1.00 | -0.77 | +0.30 |
| 11 | mid | 0.006 | 0.041 | 1.01 | 0.00 | -0.99 | -0.03 | +0.00 |
| 11 | fast | 0.008 | 0.020 | 0.86 | 0.00 | -0.92 | +0.33 | +0.18 |
| 12 | slow | 0.022 | 0.093 | 0.88 | 0.02 | -0.98 | +0.56 | +0.01 |
| 12 | mid | 0.007 | 0.050 | 1.01 | -0.00 | -0.99 | -0.05 | +0.03 |
| 12 | fast | 0.007 | 0.018 | 0.87 | 0.00 | -0.93 | +0.36 | +0.18 |
| 13 | slow | 0.039 | 0.306 | 1.08 | -0.02 | -0.99 | -0.56 | -0.55 |
| 13 | mid | 0.019 | 0.083 | 0.99 | 0.01 | -0.97 | -0.01 | -0.09 |
| 13 | fast | 0.009 | 0.030 | 0.93 | 0.01 | -0.96 | +0.21 | +0.12 |
| 14 | slow | 0.028 | 0.307 | 0.94 | 0.02 | -1.00 | +0.66 | +0.60 |
| 14 | mid | 0.047 | 0.074 | 0.71 | 0.01 | -0.71 | +0.54 | -0.00 |
| 14 | fast | 0.011 | 0.057 | 0.97 | 0.00 | -0.96 | +0.09 | -0.04 |

Linearised attribution: share of the gaze (theta) and accommodation (A) band variance that comes from the rho4 observable (the rest comes from d). Shares sum to 1 with the d share. R2 = how well the linearisation reproduces the band.

| fixation | band | theta share from rho4 | A share from rho4 | linearisation R2 theta | R2 A |
|---|---|---|---|---|---|
| 10 | slow | 1.17 | 0.95 | 0.995 | 0.996 |
| 10 | mid | 0.32 | 0.93 | 0.998 | 0.991 |
| 10 | fast | 0.26 | 0.87 | 0.998 | 0.986 |
| 11 | slow | -0.09 | 1.02 | 0.995 | 0.999 |
| 11 | mid | 0.02 | 1.00 | 0.999 | 0.998 |
| 11 | fast | 0.00 | 1.01 | 1.000 | 0.997 |
| 12 | slow | 1.00 | 1.02 | 0.999 | 1.000 |
| 12 | mid | 0.51 | 0.99 | 0.999 | 0.999 |
| 12 | fast | 0.44 | 1.00 | 0.999 | 0.999 |
| 13 | slow | 1.19 | 1.05 | 0.999 | 0.999 |
| 13 | mid | 0.81 | 1.02 | 0.996 | 0.993 |
| 13 | fast | 0.76 | 0.96 | 0.997 | 0.995 |
| 14 | slow | 0.96 | 0.98 | 0.999 | 0.999 |
| 14 | mid | -0.16 | 1.14 | 1.000 | 0.996 |
| 14 | fast | 0.32 | 0.83 | 0.998 | 0.990 |

## Full model

| fixation | frames (filled; glitch-flagged) | SD gaze (deg) | SD A (D) | r raw | r slow | r mid | r fast | slope slow | slope mid | slope fast | predicted slope via rho4 | predicted slope via d |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 | 3616 (384; 0) | 0.101 | 0.264 | -0.87 | -0.95 | -0.61 | -0.69 | -0.32 | -0.42 | -0.65 | -0.43 | -9.54 |
| 11 | 3913 (7; 0) | 0.060 | 0.205 | -0.20 | -0.31 | +0.14 | -0.03 | -0.07 | +0.10 | -0.03 | +0.07 | -34.49 |
| 12 | 3998 (2; 0) | 0.034 | 0.108 | +0.66 | +0.66 | +0.54 | +0.72 | +0.18 | +0.22 | +0.47 | +0.39 | +17.78 |
| 13 | 3758 (2; 0) | 0.179 | 0.286 | +0.92 | +0.94 | +0.81 | +0.91 | +0.58 | +0.57 | +0.77 | +0.74 | +6.92 |
| 14 | 4160 (0; 0) | 0.386 | 0.347 | +0.87 | +0.97 | +0.18 | +0.89 | +0.96 | +0.38 | +1.81 | +0.99 | +4.92 |

Slopes are gaze change per accommodation change (deg / D). Predicted slopes are -(dd/dA)/(dd/dtheta) (error enters via rho4) and -(drho/dA)/(drho/dtheta) (error enters via d), medians over the fixation.

| fixation | share of gaze variance slow / mid / fast | share of A variance slow / mid / fast | peak xcorr slow (lag ms) | peak xcorr mid+fast (lag ms) | r(d, rho4) slow / mid / fast |
|---|---|---|---|---|---|
| 10 | 0.74 / 0.08 / 0.17 | 0.95 / 0.02 / 0.03 | -0.97 (-83) | -0.67 (0) | -0.59 / +0.02 / +0.36 |
| 11 | 0.60 / 0.22 / 0.18 | 0.95 / 0.04 / 0.01 | -0.31 (61) | -0.17 (-186) | +0.49 / -0.07 / +0.04 |
| 12 | 0.56 / 0.32 / 0.12 | 0.79 / 0.19 / 0.03 | +0.67 (-31) | +0.57 (0) | +0.74 / +0.46 / -0.14 |
| 13 | 0.88 / 0.10 / 0.02 | 0.91 / 0.08 / 0.01 | +0.96 (75) | +0.82 (0) | +0.65 / +0.49 / -0.06 |
| 14 | 0.73 / 0.14 / 0.13 | 0.92 / 0.04 / 0.04 | +0.98 (82) | +0.48 (0) | +0.12 / +0.67 / -0.47 |

Raw-measurement attribution (relative changes; rho4 = S4/S1, d = m/S1). Share of the rho4 band variance from the S4 term, share of the d band variance from the S1 term, SD of the relative S1 and S4 changes, and the correlation of the accommodation band with them.

| fixation | band | SD dS1/S1 (%) | SD dS4/S4 (%) | rho4 share from S4 | d share from S1 | r(A, dS4/S4) | r(A, dS1/S1) | r(gaze, dS1/S1) |
|---|---|---|---|---|---|---|---|---|
| 10 | slow | 0.104 | 0.203 | 0.67 | 0.24 | -0.99 | +0.96 | -0.97 |
| 10 | mid | 0.011 | 0.046 | 0.94 | -0.03 | -0.97 | +0.28 | -0.51 |
| 10 | fast | 0.007 | 0.048 | 0.98 | -0.00 | -0.99 | +0.12 | -0.19 |
| 11 | slow | 0.027 | 0.230 | 1.10 | 0.01 | -1.00 | -0.77 | +0.32 |
| 11 | mid | 0.006 | 0.041 | 1.01 | 0.00 | -0.99 | -0.03 | +0.00 |
| 11 | fast | 0.008 | 0.020 | 0.86 | 0.00 | -0.92 | +0.33 | +0.18 |
| 12 | slow | 0.022 | 0.093 | 0.88 | 0.02 | -0.98 | +0.56 | -0.01 |
| 12 | mid | 0.007 | 0.050 | 1.01 | -0.00 | -0.99 | -0.05 | +0.03 |
| 12 | fast | 0.007 | 0.018 | 0.87 | 0.00 | -0.93 | +0.36 | +0.18 |
| 13 | slow | 0.039 | 0.306 | 1.08 | -0.02 | -0.99 | -0.56 | -0.55 |
| 13 | mid | 0.019 | 0.083 | 0.99 | 0.01 | -0.97 | -0.01 | -0.09 |
| 13 | fast | 0.009 | 0.030 | 0.93 | 0.01 | -0.96 | +0.21 | +0.11 |
| 14 | slow | 0.028 | 0.307 | 0.94 | 0.02 | -1.00 | +0.66 | +0.60 |
| 14 | mid | 0.047 | 0.074 | 0.71 | 0.01 | -0.71 | +0.55 | -0.00 |
| 14 | fast | 0.011 | 0.057 | 0.97 | 0.00 | -0.96 | +0.09 | -0.04 |

Linearised attribution: share of the gaze (theta) and accommodation (A) band variance that comes from the rho4 observable (the rest comes from d). Shares sum to 1 with the d share. R2 = how well the linearisation reproduces the band.

| fixation | band | theta share from rho4 | A share from rho4 | linearisation R2 theta | R2 A |
|---|---|---|---|---|---|
| 10 | slow | 1.16 | 0.95 | 0.996 | 0.997 |
| 10 | mid | 0.33 | 0.94 | 0.998 | 0.992 |
| 10 | fast | 0.26 | 0.88 | 0.998 | 0.987 |
| 11 | slow | -0.09 | 1.02 | 0.996 | 0.999 |
| 11 | mid | 0.01 | 1.00 | 0.999 | 0.998 |
| 11 | fast | 0.00 | 1.01 | 1.000 | 0.997 |
| 12 | slow | 0.97 | 1.02 | 0.999 | 1.000 |
| 12 | mid | 0.49 | 0.99 | 0.999 | 0.999 |
| 12 | fast | 0.43 | 1.00 | 0.999 | 0.999 |
| 13 | slow | 1.19 | 1.05 | 0.999 | 0.999 |
| 13 | mid | 0.80 | 1.02 | 0.996 | 0.993 |
| 13 | fast | 0.76 | 0.96 | 0.997 | 0.995 |
| 14 | slow | 0.96 | 0.98 | 0.999 | 0.998 |
| 14 | mid | -0.16 | 1.14 | 0.999 | 0.995 |
| 14 | fast | 0.32 | 0.84 | 0.998 | 0.987 |

## Holdout-3 vs Full: temporal agreement

| fixation | r(A) slow band | r(gaze) slow band | SD of gaze difference slow / mid+fast (deg) | SD of A difference slow / mid+fast (D) | mean gaze diff (deg) | mean A diff (D) | r(gaze diff, A diff) |
|---|---|---|---|---|---|---|---|
| 10 | +1.000 | +1.000 | 0.002 / 0.001 | 0.012 / 0.003 | -0.005 | +0.018 | +0.99 |
| 11 | +1.000 | +1.000 | 0.001 / 0.000 | 0.005 / 0.001 | -0.021 | -0.034 | +0.97 |
| 12 | +1.000 | +1.000 | 0.001 / 0.001 | 0.000 / 0.000 | +0.013 | -0.058 | -0.89 |
| 13 | +1.000 | +1.000 | 0.004 / 0.001 | 0.004 / 0.001 | +0.061 | -0.009 | -0.98 |
| 14 | +1.000 | +1.000 | 0.003 / 0.002 | 0.010 / 0.003 | +0.028 | +0.049 | +0.63 |
