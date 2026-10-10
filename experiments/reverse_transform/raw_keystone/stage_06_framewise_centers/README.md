# Stage 06: framewise centers

This stage derives a corrected center for each frame and averages corrected center separations within each fixation. For horizontal gaze, each capture gets a quadratic with zero output at its own zero-fixation mean input, least-squares fit to the five arithmetic fixation-mean inputs and nominal gaze labels. Because gaze is then mapped per frame, the arithmetic mean of the framewise quadratic outputs can differ from the polynomial evaluated at the fixation-mean input. Vertical gaze uses the same first-order slope at each capture’s zero-gaze reference; there are no independent vertical targets. The stage updates gaze, reprofiles the P1 magnification, fits accommodation with gaze held fixed, and recomputes centers in an outer fixed-point loop. This is an alternating procedure, not a joint framewise gaze/A fit. Centers are frame-specific outputs. No RMS or area normalization is applied, and plots do not refit or alter the saved model.

The canonical run is complete and its independent CPU audit passed. See [results](RESULTS.md) for saved metrics, calibration coefficients, audit details, and plots.

## Replay

```bash
/home/aplab/.pyenv/versions/venv/bin/python experiments/reverse_transform/raw_keystone/stage_06_framewise_centers/scripts/run.py \
  --output experiments/reverse_transform/raw_keystone/stage_06_framewise_centers/results/run
/home/aplab/.pyenv/versions/venv/bin/python experiments/reverse_transform/raw_keystone/stage_06_framewise_centers/scripts/audit.py \
  --results experiments/reverse_transform/raw_keystone/stage_06_framewise_centers/results/run
/home/aplab/.pyenv/versions/venv/bin/python experiments/reverse_transform/raw_keystone/stage_06_framewise_centers/scripts/plot_results.py \
  --results experiments/reverse_transform/raw_keystone/stage_06_framewise_centers/results/run
```

The plot script reads only `summary.json` and `frames.npz`. It writes plots and a separate `plot_metadata.json` with input, source, and output hashes. Source row means acquisition order, not elapsed time. See [Stage 05](../stage_05_capture1_state_ablation/README.md) for the preceding state comparison and [raw-keystone chain review](../SCIENTIFIC_REVIEW.md) for the chain's scientific limits.

## Saved plots

- `accommodation_by_source_order.png`
- `centers_by_source_order.png`
- `corrected_center_separation_by_capture.png`
- `corrected_center_gaze_calibration.png`
- `gaze_initial_vs_refined.png`
- `forward_point_metrics_by_capture.png`
- `outer_cycle_convergence.png`
