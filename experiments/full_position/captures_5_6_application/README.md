> Superseded: this application used an exploratory fit with collapsed training accommodation. See `accommodation_diagnosis.json` and the corrected results in `../captures_5_6_quadratic_application`.

# Direct application to captures 5 and 6

Run from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python apply_full_position_captures_5_6.py
```

The script loads the certified `prior_reference_v2/full_development/conditional27/model.json`, fitted on all twenty calibration fixations in captures 1–4. It does not refit the joint model. Use `--model PATH` for another saved conditional27/37 fit and `--output PATH` for a separate result directory. The default GPU backend is CuPy; `--backend numpy` provides a CPU alternative.

The separate gaze-only calibration fits an affine mapping from the five capture-1 fixation means of horizontal `mean(P4_x) - mean(P1_x)` in pixels to nominal horizontal gaze in degrees. Each fixation has equal weight; no accommodation term enters that mapping.

- `joint_gaze_accommodation.png`: 2 × 2; columns are captures 5 and 6, rows are joint gaze (degrees) and accommodation (diopters).
- `linear_gaze_comparison.png`: 2 × 2; rows are captures 5 and 6, columns are linear gaze (degrees) and signed linear-minus-joint gaze difference (arcminutes).
- `capture_5_estimates.csv` and `capture_6_estimates.csv`: all original rows, elapsed time, both gaze estimates, accommodation, differences, solver status, bound flags, and weighted residual costs.
- `capture_1_linear_calibration.csv`: source fixation means and nominal labels.
- `summary.json`: source/model hashes, linear coefficients and coverage.

Elapsed time is `(timestamp_ms - first_timestamp_ms) / 1000`. Figures show every available estimate without smoothing or outlier exclusion. Linear gaze requires all three valid P4 points; the joint estimator can use two. Missing estimates remain NaN. Differences measure disagreement between estimators, not independently measured gaze error.

The accelerated application uses all 49 declared inverse starts, exact-polishes the cheapest batched seed, and falls back to scalar multistart inversion if its certificate fails. It does not enumerate/certify every alternate branch. Three sampled capture-5 full-frame estimates matched exhaustive scalar results within 1e-5 degrees/diopters; this is a numerical spot check.

To redraw the figures from the exported CSVs without estimating again:

```bash
python apply_full_position_captures_5_6.py --plots-only
```
