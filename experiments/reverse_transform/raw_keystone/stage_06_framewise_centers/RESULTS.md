# Stage 06 results: framewise centers and sequential gaze recalibration

**Status:** `COMPLETE`; independent saved-results CPU audit: `PASS`.

This run derives two model-corrected centers per frame, fits a quadratic horizontal gaze map to five arithmetic fixation-mean center separations, maps vertical gaze with the same first-order slope at the zero-gaze reference, then reprofiles the P1 scale and fits A with gaze held fixed. It iterates this sequence to a fixed point. This is not a joint framewise gaze/A optimization. Raw keystone output is used without RMS/area rescaling; the P1 magnification is shared with P4.

## Coverage, protocol, and convergence

| Complete / scheduled | Unavailable | Runtime | Backend / device |
|---:|---:|---:|---|
| 89,175 / 100,090 | 10,915 | 52.57 s | CUPY 14.2.0 float64 / Tesla P100-PCIE-16GB |

All complete frames remain in the outer iteration. Center coordinates are frame-specific: `C1 = c1 − g·μ1` and `C4 = c4 − g·μ4`; corrected separation is `(C4−C1)/g`. Fixation means are calculated after division by the shared magnification. A uses the existing 0.25 D soft fixation-mean anchor (strength 16, residual scale 1 px), operational `[0,6]` D bounds, and original camera-coordinate P4 vertex residuals.

Horizontal gaze is quadratic in corrected horizontal separation. Each capture has a polynomial that is zero at that capture’s zero-gaze fixation-mean input and is least-squares fit to its five arithmetic fixation-mean inputs and nominal gaze labels. The reported polynomial value at a mean input can differ from the arithmetic mean of the framewise polynomial output because the quadratic acts before averaging. Vertical gaze uses the same first-order slope at that capture’s zero reference; there are no independent vertical targets.

| Outer cycle | Max absolute Δθx (deg) | Max absolute Δθy (deg) | Max absolute ΔA (D) | Max center change P1/P4 (px) | A-only objective (px²) |
|---:|---:|---:|---:|---:|---:|
| 1 | 2.13 | 0.593 | 0.0311 | 0.148 / 1.77 | 7.229720690 |
| 2 | 0.092 | 0.066 | 0.00316 | 0.0085 / 0.199 | 7.228469977 |
| 3 | 0.0099 | 0.00732 | 0.000287 | 0.000838 / 0.0222 | 7.228512104 |
| 4 | 0.0011 | 0.000809 | 2.38e-05 | 9.27e-05 / 0.00245 | 7.228520583 |
| 5 | 0.000121 | 8.94e-05 | 1.88e-06 | 1.02e-05 / 0.000271 | 7.228521539 |
| 6 | 1.34e-05 | 9.88e-06 | 1.43e-07 | 1.13e-06 / 2.99e-05 | 7.228521637 |

The outer loop converged in 6 cycles. The next undamped gaze recalibration differed from the final gaze by at most `1.48e-06°` across components (`1.48e-06°`, `1.09e-06°`). The final fixed-gaze A-only objective was `7.228521637 px²`; its selected start certificate has projected-gradient infinity norm `1.06e-12`, minimum free-frame curvature `26.7436`, 943 frames at A=0, 13 at A=6, and 0 forward-domain-limited frames.

## Corrected-center gaze calibration

Coefficients below are native polynomial coefficients for `θx = c0 + c1·Δx + c2·Δx²`, where Δx is the corrected center-separation coordinate in pixels. `c0` is not itself the zero-reference output: the polynomial is exactly zero at the listed `Δx` origin. The vertical map shares the horizontal first-order slope at its own zero-reference Δy.

| Capture | Zero-reference Δx (px) | Zero-reference Δy (px) | c0 (deg) | c1 (deg/px) | c2 (deg/px²) | Shared θy slope (deg/px) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 52.107 | 44.908 | -2.5534 | 0.0489285 | 1.425084e-06 | 0.0490771 |
| 2 | 9.699 | 47.464 | -0.59162 | 0.0610800 | -8.223607e-06 | 0.0609204 |
| 3 | 20.991 | 48.470 | -1.21558 | 0.0580542 | -6.866782e-06 | 0.0577659 |
| 4 | 39.092 | 46.600 | -2.10231 | 0.0539453 | -4.259078e-06 | 0.0536123 |

The calibration inputs, polynomial values at the arithmetic means, arithmetic means of framewise mapped gaze, derivative ranges, and zero-reference policy are saved in [`summary.json`](results/run/summary.json): per-cycle calibrations and the final `next_calibration`. The plot distinguishes nominal labels, the polynomial at each mean input, and the arithmetic mean of framewise mapped gaze.

## Forward and inverse point errors

Cells give median / P95 / RMS corresponding-vertex distance in original camera pixels, using the same complete population for initial and refined models. Inverse columns use the common valid population.

| Pattern and metric | Initial | Refined |
|---|---:|---:|
| P1 forward | 0.660 / 2.285 / 1.055 | 0.662 / 2.285 / 1.056 |
| P1 inverse, common | 0.660 / 2.280 / 1.055 | 0.662 / 2.280 / 1.056 |
| P4 forward | 1.255 / 3.079 / 2.580 | 1.221 / 2.996 / 2.560 |
| P4 inverse, common | 1.353 / 3.480 / 3.449 | 1.318 / 3.356 / 3.548 |

Both states have 89,175 complete frames and all P1/P4 inverses are valid. These are in-sample metrics from the calibration recording; inverse metrics remain secondary to the forward camera-coordinate fit.

### Per-capture errors

| Capture | Frames | P4 forward initial → refined (median / P95 / RMS px) | P4 inverse common initial → refined (median / P95 / RMS px) |
|---:|---:|---:|---:|
| 1 | 24,280 | 0.950 / 1.908 / 1.162 → 0.938 / 1.928 / 1.176 | 0.942 / 1.921 / 1.163 → 0.931 / 1.946 / 1.177 |
| 2 | 22,026 | 1.284 / 3.006 / 3.620 → 1.219 / 2.877 / 3.594 | 1.432 / 3.587 / 5.696 → 1.357 / 3.426 / 5.965 |
| 3 | 19,795 | 1.536 / 3.139 / 1.836 → 1.467 / 3.022 / 1.753 | 1.731 / 3.499 / 2.070 → 1.652 / 3.353 / 1.980 |
| 4 | 23,074 | 1.770 / 4.102 / 2.984 → 1.786 / 4.104 / 2.985 | 1.856 / 4.452 / 3.147 → 1.873 / 4.455 / 3.141 |

## Fixation-mean accommodation

The table reports mean ± within-fixation SD, not uncertainty of the mean. Each row corresponds to a distinct capture and nominal horizontal fixation; values remain conditioned on the assumed gaze calibration and fitted optical model.

| Capture | Nominal gaze | Frames | Expected A (D) | Initial A mean ± SD (D) | Refined A mean ± SD (D) |
|---:|---:|---:|---:|---:|---:|
| 1 | -10° | 4,433 | 0.360 | 0.566 ± 0.073 | 0.567 ± 0.073 |
| 1 | -5° | 4,800 | 0.360 | 0.587 ± 0.099 | 0.587 ± 0.099 |
| 1 | +0° | 4,900 | 0.360 | 0.362 ± 0.057 | 0.362 ± 0.057 |
| 1 | +5° | 4,993 | 0.360 | 0.211 ± 0.160 | 0.214 ± 0.161 |
| 1 | +10° | 5,154 | 0.360 | 0.243 ± 0.079 | 0.244 ± 0.078 |
| 2 | -10° | 3,912 | 4.000 | 3.926 ± 0.432 | 3.925 ± 0.432 |
| 2 | -5° | 4,345 | 4.000 | 4.074 ± 0.231 | 4.074 ± 0.232 |
| 2 | +0° | 4,553 | 4.000 | 4.008 ± 0.087 | 4.008 ± 0.087 |
| 2 | +5° | 4,781 | 4.000 | 4.114 ± 0.136 | 4.115 ± 0.135 |
| 2 | +10° | 4,435 | 4.000 | 3.720 ± 0.136 | 3.720 ± 0.136 |
| 3 | -10° | 4,505 | 3.000 | 3.138 ± 0.104 | 3.136 ± 0.104 |
| 3 | -5° | 3,902 | 3.000 | 3.377 ± 0.260 | 3.375 ± 0.258 |
| 3 | +0° | 4,811 | 3.000 | 3.094 ± 0.121 | 3.094 ± 0.121 |
| 3 | +5° | 4,251 | 3.000 | 3.021 ± 0.076 | 3.022 ± 0.076 |
| 3 | +10° | 2,326 | 3.000 | 2.754 ± 0.221 | 2.760 ± 0.225 |
| 4 | -10° | 3,914 | 2.000 | 2.065 ± 0.170 | 2.065 ± 0.170 |
| 4 | -5° | 4,899 | 2.000 | 2.165 ± 0.141 | 2.165 ± 0.140 |
| 4 | +0° | 4,747 | 2.000 | 2.043 ± 0.125 | 2.043 ± 0.125 |
| 4 | +5° | 4,738 | 2.000 | 1.915 ± 0.154 | 1.915 ± 0.154 |
| 4 | +10° | 4,776 | 2.000 | 1.740 ± 0.259 | 1.740 ± 0.259 |

Fixation-mean A ranges (max−min across the five gaze conditions) changed from 0.375630→0.373350 D in Capture 1, 0.394739→0.394144 D in Capture 2, 0.622785→0.614884 D in Capture 3, and 0.425318→0.425478 D in Capture 4. The largest absolute fixation-mean A change was 0.006127 D. These are fit-state summaries, not an independent accommodation measurement.

## Framewise center offsets

Offset is Euclidean distance from each derived optical center to that pattern’s measured image centroid. Values summarize framewise offsets (median / RMS px).

| Capture | P1 initial → refined | P4 initial → refined |
|---:|---:|---:|
| 1 | 0.378 / 0.531 → 0.398 / 0.537 | 1.406 / 1.965 → 1.535 / 2.019 |
| 2 | 0.351 / 0.485 → 0.350 / 0.485 | 0.640 / 1.699 → 0.661 / 1.789 |
| 3 | 0.363 / 0.495 → 0.371 / 0.508 | 0.833 / 1.100 → 0.852 / 1.241 |
| 4 | 0.198 / 0.279 → 0.200 / 0.280 | 0.486 / 0.677 → 0.489 / 0.684 |

## Audit and limitations

The independent CPU audit passed all six A-fit cycles, finite-difference gradients, stationary/free-curvature certificates, all-frame inverse validity, translation invariance, and synthetic center/inverse closure. Full checks and tolerances are in [`audit.json`](results/run/audit.json). The saved [protocol](results/run/protocol.json), [frame arrays](results/run/frames.npz), and [provenance](results/run/provenance.json) contain the full inputs and outputs.

This is a conditional, same-recording model-derived center and gaze calibration, not a physical-center measurement or an independent validation set. Fixation labels define the gaze map; capture and demand remain confounded. The vertical map is assumed from the horizontal first-order slope and has no independent vertical-target calibration. The outer loop shows numerical self-consistency, not physiological truth or a globally optimal joint objective.

## Figures

- [Accommodation by source order, four captures](results/run/accommodation_by_source_order.png)
- [Centers by source order](results/run/centers_by_source_order.png)
- [Measured and corrected center separation](results/run/corrected_center_separation_by_capture.png)
- [Corrected-center fixation means and gaze calibration](results/run/corrected_center_gaze_calibration.png)
- [Initial versus refined gaze](results/run/gaze_initial_vs_refined.png)
- [Forward point metrics](results/run/forward_point_metrics_by_capture.png)
- [Outer-cycle changes and A-only starts](results/run/outer_cycle_convergence.png)
- Plot hash record: [`plot_metadata.json`](results/run/plot_metadata.json).
