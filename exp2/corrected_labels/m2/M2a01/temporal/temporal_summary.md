# Temporal correlation of gaze and accommodation: M2a01

Held-out fixations 10-14 (3 D). Anchor-free inverse; 1 ms frames. Bands: slow > 0.5 s, mid 50-500 ms, fast < 50 ms.
Statistics use real (non-interpolated) samples only.

## Holdout-3 model

| fixation | frames (filled; glitch-flagged) | SD gaze (deg) | SD A (D) | r raw | r slow | r mid | r fast | slope slow | slope mid | slope fast | predicted slope via rho4 | predicted slope via d |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 | 3616 (990; 606) | 0.078 | 0.177 | -0.91 | -0.96 | -0.62 | -0.74 | -0.39 | -0.47 | -0.52 | -0.40 | -8.35 |
| 11 | 3913 (129; 122) | 0.040 | 0.209 | -0.11 | -0.24 | +0.26 | +0.07 | -0.03 | +0.16 | +0.03 | +0.07 | -27.07 |
| 12 | 3998 (2; 0) | 0.035 | 0.108 | +0.68 | +0.68 | +0.55 | +0.73 | +0.19 | +0.24 | +0.48 | +0.41 | +17.04 |
| 13 | 3758 (426; 424) | 0.171 | 0.267 | +0.92 | +0.93 | +0.78 | +0.95 | +0.58 | +0.54 | +0.80 | +0.77 | +6.66 |
| 14 | 4160 (607; 607) | 0.289 | 0.320 | +0.98 | +0.99 | +0.88 | +0.98 | +0.88 | +0.85 | +1.03 | +1.01 | +4.84 |

Slopes are gaze change per accommodation change (deg / D). Predicted slopes are -(dd/dA)/(dd/dtheta) (error enters via rho4) and -(drho/dA)/(drho/dtheta) (error enters via d), medians over the fixation.

| fixation | share of gaze variance slow / mid / fast | share of A variance slow / mid / fast | peak xcorr slow (lag ms) | peak xcorr mid+fast (lag ms) | r(d, rho4) slow / mid / fast |
|---|---|---|---|---|---|
| 10 | 0.84 / 0.10 / 0.05 | 0.95 / 0.03 / 0.02 | -0.99 (-83) | -0.67 (0) | +0.13 / +0.06 / +0.23 |
| 11 | 0.46 / 0.46 / 0.08 | 0.95 / 0.04 / 0.01 | -0.30 (126) | -0.26 (-186) | +0.61 / -0.16 / +0.06 |
| 12 | 0.57 / 0.32 / 0.11 | 0.79 / 0.19 / 0.03 | +0.69 (-29) | +0.58 (0) | +0.74 / +0.46 / -0.14 |
| 13 | 0.91 / 0.07 / 0.02 | 0.93 / 0.06 / 0.01 | +0.96 (68) | +0.79 (0) | +0.67 / +0.56 / -0.10 |
| 14 | 0.94 / 0.04 / 0.02 | 0.95 / 0.03 / 0.01 | +1.00 (86) | +0.90 (0) | +0.66 / +0.42 / -0.17 |

Raw-measurement attribution (relative changes; rho4 = S4/S1, d = m/S1). Share of the rho4 band variance from the S4 term, share of the d band variance from the S1 term, SD of the relative S1 and S4 changes, and the correlation of the accommodation band with them.

| fixation | band | SD dS1/S1 (%) | SD dS4/S4 (%) | rho4 share from S4 | d share from S1 | r(A, dS4/S4) | r(A, dS1/S1) | r(gaze, dS1/S1) |
|---|---|---|---|---|---|---|---|---|
| 10 | slow | 0.085 | 0.117 | 0.58 | -0.11 | -0.99 | +0.97 | -0.95 |
| 10 | mid | 0.012 | 0.036 | 0.89 | -0.03 | -0.94 | +0.38 | -0.55 |
| 10 | fast | 0.006 | 0.028 | 0.95 | 0.01 | -0.98 | +0.20 | +0.00 |
| 11 | slow | 0.027 | 0.229 | 1.10 | 0.02 | -1.00 | -0.76 | +0.21 |
| 11 | mid | 0.006 | 0.039 | 1.00 | 0.00 | -0.99 | +0.04 | +0.03 |
| 11 | fast | 0.008 | 0.019 | 0.86 | -0.00 | -0.92 | +0.35 | +0.03 |
| 12 | slow | 0.022 | 0.093 | 0.88 | 0.02 | -0.98 | +0.56 | +0.01 |
| 12 | mid | 0.007 | 0.050 | 1.01 | -0.00 | -0.99 | -0.05 | +0.03 |
| 12 | fast | 0.007 | 0.018 | 0.87 | 0.00 | -0.93 | +0.36 | +0.18 |
| 13 | slow | 0.037 | 0.288 | 1.07 | -0.02 | -0.99 | -0.49 | -0.46 |
| 13 | mid | 0.019 | 0.068 | 0.97 | 0.01 | -0.96 | +0.08 | -0.06 |
| 13 | fast | 0.009 | 0.027 | 0.92 | 0.02 | -0.95 | +0.23 | +0.14 |
| 14 | slow | 0.026 | 0.296 | 0.94 | 0.05 | -1.00 | +0.75 | +0.72 |
| 14 | mid | 0.021 | 0.058 | 0.89 | 0.04 | -0.94 | +0.31 | +0.17 |
| 14 | fast | 0.011 | 0.034 | 0.92 | 0.08 | -0.96 | +0.22 | +0.14 |

Linearised attribution: share of the gaze (theta) and accommodation (A) band variance that comes from the rho4 observable (the rest comes from d). Shares sum to 1 with the d share. R2 = how well the linearisation reproduces the band.

| fixation | band | theta share from rho4 | A share from rho4 | linearisation R2 theta | R2 A |
|---|---|---|---|---|---|
| 10 | slow | 0.92 | 0.97 | 0.999 | 0.999 |
| 10 | mid | 0.30 | 0.99 | 0.999 | 0.997 |
| 10 | fast | 0.39 | 0.96 | 0.999 | 0.997 |
| 11 | slow | -0.14 | 1.03 | 0.982 | 0.998 |
| 11 | mid | 0.03 | 1.00 | 0.999 | 0.998 |
| 11 | fast | 0.01 | 1.01 | 0.998 | 0.997 |
| 12 | slow | 1.00 | 1.02 | 0.999 | 1.000 |
| 12 | mid | 0.51 | 0.99 | 0.999 | 0.999 |
| 12 | fast | 0.44 | 1.00 | 0.999 | 0.999 |
| 13 | slow | 1.20 | 1.05 | 0.999 | 0.999 |
| 13 | mid | 0.82 | 1.06 | 0.995 | 0.992 |
| 13 | fast | 0.82 | 0.97 | 0.997 | 0.996 |
| 14 | slow | 1.11 | 1.00 | 0.999 | 0.998 |
| 14 | mid | 0.88 | 1.02 | 0.998 | 0.996 |
| 14 | fast | 0.86 | 0.92 | 0.997 | 0.992 |

## Full model

| fixation | frames (filled; glitch-flagged) | SD gaze (deg) | SD A (D) | r raw | r slow | r mid | r fast | slope slow | slope mid | slope fast | predicted slope via rho4 | predicted slope via d |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 | 3616 (990; 606) | 0.079 | 0.169 | -0.91 | -0.96 | -0.62 | -0.74 | -0.42 | -0.50 | -0.55 | -0.43 | -9.54 |
| 11 | 3913 (129; 122) | 0.040 | 0.204 | -0.14 | -0.28 | +0.25 | +0.06 | -0.03 | +0.16 | +0.03 | +0.07 | -34.49 |
| 12 | 3998 (2; 0) | 0.034 | 0.108 | +0.66 | +0.66 | +0.54 | +0.72 | +0.18 | +0.22 | +0.47 | +0.39 | +17.78 |
| 13 | 3758 (426; 424) | 0.167 | 0.271 | +0.91 | +0.93 | +0.77 | +0.94 | +0.55 | +0.51 | +0.77 | +0.74 | +6.92 |
| 14 | 4160 (607; 607) | 0.291 | 0.330 | +0.98 | +0.99 | +0.87 | +0.98 | +0.86 | +0.83 | +1.01 | +0.99 | +4.92 |

Slopes are gaze change per accommodation change (deg / D). Predicted slopes are -(dd/dA)/(dd/dtheta) (error enters via rho4) and -(drho/dA)/(drho/dtheta) (error enters via d), medians over the fixation.

| fixation | share of gaze variance slow / mid / fast | share of A variance slow / mid / fast | peak xcorr slow (lag ms) | peak xcorr mid+fast (lag ms) | r(d, rho4) slow / mid / fast |
|---|---|---|---|---|---|
| 10 | 0.84 / 0.10 / 0.05 | 0.95 / 0.03 / 0.02 | -0.99 (-82) | -0.67 (0) | +0.13 / +0.06 / +0.23 |
| 11 | 0.47 / 0.45 / 0.08 | 0.95 / 0.04 / 0.01 | -0.32 (119) | -0.26 (-186) | +0.61 / -0.16 / +0.06 |
| 12 | 0.56 / 0.32 / 0.12 | 0.79 / 0.19 / 0.03 | +0.67 (-31) | +0.57 (0) | +0.74 / +0.46 / -0.14 |
| 13 | 0.91 / 0.07 / 0.02 | 0.93 / 0.06 / 0.01 | +0.96 (71) | +0.78 (0) | +0.67 / +0.56 / -0.10 |
| 14 | 0.94 / 0.04 / 0.02 | 0.95 / 0.03 / 0.01 | +1.00 (87) | +0.90 (0) | +0.66 / +0.42 / -0.17 |

Raw-measurement attribution (relative changes; rho4 = S4/S1, d = m/S1). Share of the rho4 band variance from the S4 term, share of the d band variance from the S1 term, SD of the relative S1 and S4 changes, and the correlation of the accommodation band with them.

| fixation | band | SD dS1/S1 (%) | SD dS4/S4 (%) | rho4 share from S4 | d share from S1 | r(A, dS4/S4) | r(A, dS1/S1) | r(gaze, dS1/S1) |
|---|---|---|---|---|---|---|---|---|
| 10 | slow | 0.085 | 0.117 | 0.58 | -0.11 | -0.99 | +0.97 | -0.95 |
| 10 | mid | 0.012 | 0.036 | 0.89 | -0.03 | -0.94 | +0.38 | -0.55 |
| 10 | fast | 0.006 | 0.028 | 0.95 | 0.01 | -0.98 | +0.20 | -0.00 |
| 11 | slow | 0.027 | 0.229 | 1.10 | 0.02 | -1.00 | -0.76 | +0.24 |
| 11 | mid | 0.006 | 0.039 | 1.00 | 0.00 | -0.99 | +0.04 | +0.03 |
| 11 | fast | 0.008 | 0.019 | 0.86 | -0.00 | -0.92 | +0.35 | +0.03 |
| 12 | slow | 0.022 | 0.093 | 0.88 | 0.02 | -0.98 | +0.56 | -0.01 |
| 12 | mid | 0.007 | 0.050 | 1.01 | -0.00 | -0.99 | -0.05 | +0.03 |
| 12 | fast | 0.007 | 0.018 | 0.87 | 0.00 | -0.93 | +0.36 | +0.18 |
| 13 | slow | 0.037 | 0.288 | 1.07 | -0.02 | -0.99 | -0.49 | -0.46 |
| 13 | mid | 0.019 | 0.068 | 0.97 | 0.01 | -0.96 | +0.08 | -0.06 |
| 13 | fast | 0.009 | 0.027 | 0.92 | 0.02 | -0.95 | +0.23 | +0.14 |
| 14 | slow | 0.026 | 0.296 | 0.94 | 0.05 | -1.00 | +0.75 | +0.72 |
| 14 | mid | 0.021 | 0.058 | 0.89 | 0.04 | -0.94 | +0.31 | +0.17 |
| 14 | fast | 0.011 | 0.034 | 0.92 | 0.08 | -0.96 | +0.22 | +0.14 |

Linearised attribution: share of the gaze (theta) and accommodation (A) band variance that comes from the rho4 observable (the rest comes from d). Shares sum to 1 with the d share. R2 = how well the linearisation reproduces the band.

| fixation | band | theta share from rho4 | A share from rho4 | linearisation R2 theta | R2 A |
|---|---|---|---|---|---|
| 10 | slow | 0.92 | 0.97 | 0.999 | 0.999 |
| 10 | mid | 0.30 | 0.99 | 1.000 | 0.997 |
| 10 | fast | 0.39 | 0.96 | 1.000 | 0.997 |
| 11 | slow | -0.15 | 1.03 | 0.986 | 0.999 |
| 11 | mid | 0.03 | 1.00 | 0.999 | 0.998 |
| 11 | fast | 0.01 | 1.01 | 0.998 | 0.997 |
| 12 | slow | 0.97 | 1.02 | 0.999 | 1.000 |
| 12 | mid | 0.49 | 0.99 | 0.999 | 0.999 |
| 12 | fast | 0.43 | 1.00 | 0.999 | 0.999 |
| 13 | slow | 1.20 | 1.05 | 0.999 | 0.999 |
| 13 | mid | 0.81 | 1.06 | 0.995 | 0.992 |
| 13 | fast | 0.81 | 0.97 | 0.997 | 0.995 |
| 14 | slow | 1.11 | 1.00 | 0.999 | 0.998 |
| 14 | mid | 0.87 | 1.01 | 0.998 | 0.995 |
| 14 | fast | 0.85 | 0.91 | 0.996 | 0.990 |

## Holdout-3 vs Full: temporal agreement

| fixation | r(A) slow band | r(gaze) slow band | SD of gaze difference slow / mid+fast (deg) | SD of A difference slow / mid+fast (D) | mean gaze diff (deg) | mean A diff (D) | r(gaze diff, A diff) |
|---|---|---|---|---|---|---|---|
| 10 | +1.000 | +1.000 | 0.001 / 0.000 | 0.008 / 0.002 | -0.006 | +0.014 | +0.99 |
| 11 | +1.000 | +0.999 | 0.001 / 0.000 | 0.005 / 0.001 | -0.021 | -0.034 | +0.98 |
| 12 | +1.000 | +1.000 | 0.001 / 0.001 | 0.000 / 0.000 | +0.013 | -0.058 | -0.89 |
| 13 | +1.000 | +1.000 | 0.004 / 0.001 | 0.004 / 0.001 | +0.061 | -0.009 | -0.97 |
| 14 | +1.000 | +1.000 | 0.002 / 0.001 | 0.010 / 0.002 | +0.028 | +0.048 | +0.93 |
