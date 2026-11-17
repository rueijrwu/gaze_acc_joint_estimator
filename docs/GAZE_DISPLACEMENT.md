# P1/P4 horizontal point separation over gaze rotation

This standalone analysis uses reviewed full fixation periods from captures 1 through 4. It plots P1 and P4 horizontal point separation in pixels against the five target eye-rotation positions. The P4 panel overlays all four accommodation conditions. The P4 y-range is set by the filtered data and standard-deviation bars, rounded outward to 5-pixel boundaries. The P1 y-range has the same max-minus-min width, centered on the overall filtered P1 mean. Both use 5-pixel tick intervals.

For each frame, the horizontal span is `max(x) - min(x)` across the three P1 points or the three P4 points. This is the absolute x distance between the rightmost and leftmost point, so the plotted value is nonnegative. The plotted point is the mean span during that fixation. Before calculating means and error bars, each reflection is despiked separately within each fixation. A 101-valid-frame rolling Hampel filter removes values more than `max(2 px, 6 × 1.4826 × local MAD)` from the local median. The CSV reports valid, retained, and removed frame counts, plus raw and filtered standard deviations. Error bars show the sample standard deviation after filtering. Each accommodation series has a quadratic fit, shown as a dashed line. The visible P1 outlier at capture 2 (4 D), 5° is marked with a red X and excluded from that fit. Fit coefficients and fit-point counts are saved separately.

Each fixation uses rows with all three P1 points valid, all three P4 points found, and finite coordinates. The same rows contribute to P1 and P4. Accommodation labels come from the reviewed fixation file: capture 1 = 0.36036 D, capture 2 = 4 D, capture 3 = 3 D, and capture 4 = 2 D. These are protocol labels, not independent accommodation measurements.

Run from any working directory with:

```sh
python experiments/gaze_displacement/analyze.py
```

The x-axis label is `Eye Rotation (deg)`. The y-axis labels use `$|P_{1a}-P_{1b}|$` and `$|P_{4a}-P_{4b}|$`. The plot uses a 1×2 layout with larger axis and tick labels and no title. It exports SVG with text preserved as editable text for Inkscape, plus vector PDF and PNG. The script reads the frozen detections, verifies each detection file hash, and writes these files beside itself:

- [`p1_p4_displacement_vs_gaze.svg`](../experiments/gaze_displacement/p1_p4_displacement_vs_gaze.svg) (editable in Inkscape)
- [`p1_p4_displacement_vs_gaze.pdf`](../experiments/gaze_displacement/p1_p4_displacement_vs_gaze.pdf)
- [`p1_p4_displacement_vs_gaze.png`](../experiments/gaze_displacement/p1_p4_displacement_vs_gaze.png)
- [`fixation_displacements.csv`](../experiments/gaze_displacement/fixation_displacements.csv)
- [`polynomial_fits.csv`](../experiments/gaze_displacement/polynomial_fits.csv)
- [`summary.json`](../experiments/gaze_displacement/summary.json)

Pass `--output PATH` to write results to another directory. The CSV has one row per fixation, with mean and standard deviation for both spans.
