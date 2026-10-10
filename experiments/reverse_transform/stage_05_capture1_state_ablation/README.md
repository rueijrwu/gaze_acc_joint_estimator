# Stage 05: Capture 1 state ablation

Stage 05 compares the preserved Stage 04 A-only state (**A0**), a horizontal-gaze-only fit (**G**), and a joint horizontal-gaze plus accommodation fit (**GA**) on Capture 1. P1 and P4 use equal objective weights. The fitting keeps the reviewed references, units, vertical gaze, keystone coefficients, and empirical relative barrel law fixed; P1 magnification is reprofiled at every trial gaze and shared with P4. Fixation-mean anchors are 0.5° for gaze and 0.25 D for A; operational bounds are ±20° and 0–6 D.

The canonical run and independent audit are under `results/run/`. See [RESULTS.md](RESULTS.md) for pooled and fixation-level metrics, state estimates, conditioning, and figures. [SCIENTIFIC_REVIEW.md](SCIENTIFIC_REVIEW.md) records the scientific interpretation and limits. The exact numerical-limit development attempts are retained separately in `results/attempt_01_numerical_limit/` and `results/attempt_02_numerical_limit/`; neither is the canonical result.

## Reproduce

The runner refuses to overwrite an output directory. To create a fresh replay from the repository root, use the configured environment and a new output path:

```bash
/home/aplab/.pyenv/versions/venv/bin/python \
  experiments/reverse_transform/stage_05_capture1_state_ablation/scripts/run.py \
  --output experiments/reverse_transform/stage_05_capture1_state_ablation/results/replay

/home/aplab/.pyenv/versions/venv/bin/python \
  experiments/reverse_transform/stage_05_capture1_state_ablation/scripts/audit.py \
  --results experiments/reverse_transform/stage_05_capture1_state_ablation/results/replay

/home/aplab/.pyenv/versions/venv/bin/python \
  experiments/reverse_transform/stage_05_capture1_state_ablation/scripts/plot_results.py \
  --results experiments/reverse_transform/stage_05_capture1_state_ablation/results/replay
```

The run log is `logs/run.log` and the canonical audit log is `logs/audit.log`. Results store the fit and audit JSON, full per-frame arrays, provenance hashes, and plot metadata.
