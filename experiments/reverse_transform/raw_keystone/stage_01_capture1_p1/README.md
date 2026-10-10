# Stage 1: Capture 1 P1

Fresh raw-keystone calibration for Capture 1. Run from the repository root:

```bash
/home/aplab/.pyenv/versions/venv/bin/python experiments/reverse_transform/raw_keystone/stage_01_capture1_p1/scripts/run.py \
  --output experiments/reverse_transform/raw_keystone/stage_01_capture1_p1/results/run
/home/aplab/.pyenv/versions/venv/bin/python experiments/reverse_transform/raw_keystone/stage_01_capture1_p1/scripts/audit.py \
  --results experiments/reverse_transform/raw_keystone/stage_01_capture1_p1/results/run
/home/aplab/.pyenv/versions/venv/bin/python experiments/reverse_transform/raw_keystone/stage_01_capture1_p1/scripts/plot_results.py \
  --results experiments/reverse_transform/raw_keystone/stage_01_capture1_p1/results/run
```

The wrapper selects stage 1. Use a new output path for a replay because the runner does not overwrite results.

See [RESULTS.md](RESULTS.md) for saved per-fixation metrics, fitted coefficients, coverage, and plots. The report contains no additional fit or refit.

See [RESULTS.md](RESULTS.md) for saved per-fixation metrics, fitted coefficients, coverage, and figures.
