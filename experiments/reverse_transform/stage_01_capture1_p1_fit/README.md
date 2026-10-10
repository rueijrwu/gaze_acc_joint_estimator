# Capture 1 P1 shape fit

This stage fits a fresh centroid-to-gaze map and a four-coefficient 2D P1 keystone shape to the five reviewed intervals in `capture_1_detections.pkl`. It uses capture1 detections only. No previous optical model, calibration, saved state, reference, or covariance is loaded. All 24,700 scheduled capture1 rows are retained in the population archive; 24,280 complete P1/P4 rows support the fit and 420 unavailable rows remain recorded.

The horizontal gaze coordinate is a quadratic function of the P4-minus-P1 mean x centroid, fitted to the five nominal fixation labels with equal interval weight. Vertical gaze uses the same first-order slope as x, applied to the P4-minus-P1 mean y offset from the zero-fixation mean. This is a stated calibration assumption: the session has no independently measured vertical targets. Both gaze coordinates are estimated from the same data used to fit the P1 shape.

The empirical reference triangle is the average centered P1 triangle in the nominal zero-gaze fixation. The four shared keystone coefficients describe shape. A centered, model-size-normalized triangle `K(theta; B)` is multiplied by one positive profiled magnification `M` per frame; `M` absorbs all overall scale, including the size change that the unnormalized keystone would otherwise introduce. There is no second camera-scale parameter. The fit minimizes original-coordinate centered P1 SSE with equal total weight per fixation.

For reproducibility, the run command is:

```bash
python experiments/reverse_transform/stage_01_capture1_p1_fit/scripts/run.py \
  --output experiments/reverse_transform/stage_01_capture1_p1_fit/results/attempt_01
```

The independent artifact audit is:

```bash
python experiments/reverse_transform/stage_01_capture1_p1_fit/scripts/audit.py \
  --results experiments/reverse_transform/stage_01_capture1_p1_fit/results/attempt_01
```

It passed; details are in the [independent audit](results/attempt_01/independent_audit.json) and [attempt 01 results](results/attempt_01/RESULTS.md). The fit took 6.84 s with Python 3.13.15, NumPy 2.5.3, and SciPy 1.18.1 using CPU SciPy L-BFGS-B (no GPU); Matplotlib 3.11.2 generated [fit distributions](results/attempt_01/fit_distributions.png), [mean triangles](results/attempt_01/mean_triangles.png), and [magnification and gaze by source row](results/attempt_01/magnification_gaze_by_row.png). Regenerate figures with `scripts/plot_results.py --results .../results/attempt_01`.

The estimated vertical gaze is not measured gaze, accommodation is not estimated, and pointwise agreement with the same-data empirical reference is not an accuracy estimate. In particular, the large native vertical shape coefficients need an independent sensitivity check before interpretation. The estimated vertical gaze spans -0.5824 to +0.5813° framewise, with interval means -0.2213, -0.0735, 0, +0.1571, +0.3526°. At the largest |theta_y|, the fitted vertical quadratic log-stretch term is 0.00520 (about 0.52%); the maximum vertical denominator contribution is 0.00564. For comparison the corresponding observed horizontal terms reach 0.00225 and 0.00503. The native `k_ay` is about 832 times `k_ax`, largely because vertical gaze has a much smaller span; these are reasons to test sensitivity to the assumed shared slope and reference. This stage makes no radial distortion or absolute optical-origin claim. A fuller scientific discussion is in [scientific review](SCIENTIFIC_REVIEW.md).
