# Stage 2: Capture 1 P4

Fits the Capture 1 P4 raw-keystone shape using the Stage 1 common P1 magnification. Run from the repository root after Stage 1:

```bash
/home/aplab/.pyenv/versions/venv/bin/python experiments/reverse_transform/raw_keystone/stage_02_capture1_p4/scripts/run.py \
  --output experiments/reverse_transform/raw_keystone/stage_02_capture1_p4/results/run
/home/aplab/.pyenv/versions/venv/bin/python experiments/reverse_transform/raw_keystone/stage_02_capture1_p4/scripts/audit.py \
  --results experiments/reverse_transform/raw_keystone/stage_02_capture1_p4/results/run
/home/aplab/.pyenv/versions/venv/bin/python experiments/reverse_transform/raw_keystone/stage_02_capture1_p4/scripts/plot_results.py \
  --results experiments/reverse_transform/raw_keystone/stage_02_capture1_p4/results/run
```

The wrapper selects stage 2. Use a new output path for a replay because the runner does not overwrite results.

See [RESULTS.md](RESULTS.md) for saved per-fixation metrics, fitted coefficients, coverage, and plots. The report contains no additional fit or refit.

See [RESULTS.md](RESULTS.md) for saved per-fixation metrics, fitted coefficients, coverage, and figures.
