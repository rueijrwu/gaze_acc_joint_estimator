# Axis anchor sensitivity results

The run recorded 36/36 certified calibrations and 36/36 complete evaluations. The independent verifier reconstructed 36 accepted fits; all passed coefficient, objective, physical projected-gradient, and inner-stationarity checks. It counted 126 final starts and 0 continuation checkpoint records.

The first table reports equal-fixation RMS over complete frames. E is the physical pixel RMS across three held-point errors; G is the RMS pairwise latent-state disagreement. Coverage counts retain all 160 scheduled frames and 480 slots, including invalid inputs. The comparison table uses exact shared complete-frame IDs for each candidate and comparator, with equal-fixation RMS calculated over that same shared membership.

| Family | Capacity | Variant | Frames / slots | Invalid frames / slots | Scored slots | Complete frames | E RMS px (equal fixation) | Gθ RMS deg (equal fixation) | G_A RMS D (equal fixation) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| gaze | conditional27 | anchor_gaze_strong | 160 / 480 | 17 / 51 | 429 | 143 | 3.4212 | 0.28715 | 0.33148 |
| gaze | conditional37 | anchor_gaze_strong | 160 / 480 | 17 / 51 | 429 | 143 | 4.7005 | 0.39814 | 0.43571 |
| capture | conditional27 | anchor_gaze_strong | 160 / 480 | 17 / 51 | 429 | 143 | 4.1695 | 0.29249 | 0.32446 |
| capture | conditional37 | anchor_gaze_strong | 160 / 480 | 17 / 51 | 429 | 143 | 4.3227 | 0.27787 | 0.35112 |
| gaze | conditional27 | anchor_accommodation_strong | 160 / 480 | 17 / 51 | 429 | 143 | 3.4256 | 0.26688 | 0.31545 |
| gaze | conditional37 | anchor_accommodation_strong | 160 / 480 | 17 / 51 | 429 | 143 | 4.7602 | 0.41953 | 0.45012 |
| capture | conditional27 | anchor_accommodation_strong | 160 / 480 | 17 / 51 | 429 | 143 | 4.1721 | 0.27797 | 0.31457 |
| capture | conditional37 | anchor_accommodation_strong | 160 / 480 | 17 / 51 | 429 | 143 | 4.3121 | 0.33772 | 0.40826 |

| Family | Capacity | Variant | Comparator | Shared frames | Candidate E / Gθ / G_A (equal fixation) | Comparator E / Gθ / G_A (equal fixation) |
|---|---|---|---|---:|---:|---:|
| gaze | conditional27 | anchor_gaze_strong | baseline | 143 | 3.4212 / 0.28715 / 0.33148 | 3.4207 / 0.26937 / 0.31111 |
| gaze | conditional27 | anchor_gaze_strong | anchor_strong | 143 | 3.4212 / 0.28715 / 0.33148 | 3.4385 / 0.28672 / 0.32290 |
| gaze | conditional37 | anchor_gaze_strong | baseline | 143 | 4.7005 / 0.39814 / 0.43571 | 6.7005 / 0.50927 / 0.92903 |
| gaze | conditional37 | anchor_gaze_strong | anchor_strong | 143 | 4.7005 / 0.39814 / 0.43571 | 4.7637 / 0.43969 / 0.52973 |
| capture | conditional27 | anchor_gaze_strong | baseline | 143 | 4.1695 / 0.29249 / 0.32446 | 4.0144 / 0.28057 / 0.31941 |
| capture | conditional27 | anchor_gaze_strong | anchor_strong | 143 | 4.1695 / 0.29249 / 0.32446 | 4.2458 / 0.29318 / 0.32897 |
| capture | conditional37 | anchor_gaze_strong | baseline | 143 | 4.3227 / 0.27787 / 0.35112 | 4.3066 / 0.32499 / 0.39171 |
| capture | conditional37 | anchor_gaze_strong | anchor_strong | 143 | 4.3227 / 0.27787 / 0.35112 | 4.2928 / 0.27636 / 0.36650 |
| gaze | conditional27 | anchor_accommodation_strong | baseline | 143 | 3.4256 / 0.26688 / 0.31545 | 3.4207 / 0.26937 / 0.31111 |
| gaze | conditional27 | anchor_accommodation_strong | anchor_strong | 143 | 3.4256 / 0.26688 / 0.31545 | 3.4385 / 0.28672 / 0.32290 |
| gaze | conditional37 | anchor_accommodation_strong | baseline | 143 | 4.7602 / 0.41953 / 0.45012 | 6.7005 / 0.50927 / 0.92903 |
| gaze | conditional37 | anchor_accommodation_strong | anchor_strong | 143 | 4.7602 / 0.41953 / 0.45012 | 4.7637 / 0.43969 / 0.52973 |
| capture | conditional27 | anchor_accommodation_strong | baseline | 143 | 4.1721 / 0.27797 / 0.31457 | 4.0144 / 0.28057 / 0.31941 |
| capture | conditional27 | anchor_accommodation_strong | anchor_strong | 143 | 4.1721 / 0.27797 / 0.31457 | 4.2458 / 0.29318 / 0.32897 |
| capture | conditional37 | anchor_accommodation_strong | baseline | 143 | 4.3121 / 0.33772 / 0.40826 | 4.3066 / 0.32499 / 0.39171 |
| capture | conditional37 | anchor_accommodation_strong | anchor_strong | 143 | 4.3121 / 0.33772 / 0.40826 | 4.2928 / 0.27636 / 0.36650 |

Axis-specific anchor tightening left held-point error essentially unchanged for conditional27. For conditional37 on the capture split, gaze-only tightening reduced the equal-fixation state disagreement to Gθ=0.277872° and G_A=0.351118 D from baseline 0.324988° and 0.391707 D. Accommodation-only tightening increased those metrics to 0.337719° and 0.408265 D. On the exact 126 shared interior frames, gaze-only changed G² by −0.036390 (gaze) and −0.040618 (accommodation); accommodation-only changed G² by +0.004942 and +0.003329. These are development sensitivity results, not a basis for selecting hyperparameters.

The sensitivity run took 634.1905 seconds with eight workers and OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1. The repository suite recorded 59 passed in 10.87 seconds. The transition follow-up completed and independently verified 8 comparisons across 1939 source hashes and 36 summary rows; see `../axis_anchor_transitions_v1/verification.json` and `summary.json`.
