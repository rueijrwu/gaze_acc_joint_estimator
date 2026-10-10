# Stage 05 — Capture 1 joint gaze/A state ablation

This saved experiment compares the raw-keystone A0 baseline, a horizontal-gaze-only fit (G), and a joint gaze/accommodation fit (GA) on all 24,280 complete Capture 1 frames. The raw centered projective transform is used without RMS or area rescaling. P1 and P4 have equal forward weight; P1 positive magnification is reprofiled at every trial gaze and shared with P4. Only fixation-mean anchors are used (0.5° gaze and 0.25 D accommodation); frame states have operational domains of −20° to +20° and 0–6 D.

The run completed on a Tesla P100 using CuPy 14.2.0 in about 36.7 seconds. Its saved independent CPU audit passed. This certifies the saved implementation and local stationary solutions under the stated model, not global uniqueness or physiological accuracy. See [RESULTS.md](RESULTS.md) for measurements and [the raw-chain scientific review](../SCIENTIFIC_REVIEW.md) for interpretation and limits.

## Reproduction

Use the repository environment and a fresh output directory for a new fit:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/aplab/.pyenv/versions/venv/bin/python \
  experiments/reverse_transform/raw_keystone/stage_05_capture1_state_ablation/scripts/run.py \
  --output experiments/reverse_transform/raw_keystone/stage_05_capture1_state_ablation/results/replay_run
```

Audit and plot a saved run without fitting again:

```sh
/home/aplab/.pyenv/versions/venv/bin/python \
  experiments/reverse_transform/raw_keystone/stage_05_capture1_state_ablation/scripts/audit.py \
  --results experiments/reverse_transform/raw_keystone/stage_05_capture1_state_ablation/results/run
/home/aplab/.pyenv/versions/venv/bin/python \
  experiments/reverse_transform/raw_keystone/stage_05_capture1_state_ablation/scripts/plot_results.py \
  --results experiments/reverse_transform/raw_keystone/stage_05_capture1_state_ablation/results/run
```

## Saved artifacts

- [summary.json](results/run/summary.json), [audit.json](results/run/audit.json), [protocol.json](results/run/protocol.json), and [provenance.json](results/run/provenance.json) contain the numerical record and integrity metadata.
- [frames.npz](results/run/frames.npz) contains saved per-frame states, forward predictions/residuals, inverse reconstructions, and conditioning arrays; it is loaded with `allow_pickle=False` by the audit.
- [state means by fixation](results/run/state_means_by_fixation.png), [forward/inverse metrics](results/run/forward_inverse_metrics_by_fixation.png), [corrected triangle clouds](results/run/corrected_triangle_means_clouds.png), [conditioning](results/run/conditioning_by_fixation.png), [states by source row](results/run/states_by_source_row.png), and [gaze/A changes](results/run/delta_gaze_vs_delta_A.png) are the saved plots.
- [run log](logs/run.log), [audit log](logs/audit.log), and [plot log](logs/plot.log) record the original execution.
