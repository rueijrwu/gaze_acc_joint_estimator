# Stage 3: Independent captures

Extends the raw P1/P4 fitting to captures 1–4, retaining separate capture coefficients and the shared P1 magnification within each frame. Run from the repository root after Stages 1–2:

```bash
/home/aplab/.pyenv/versions/venv/bin/python experiments/reverse_transform/raw_keystone/stage_03_independent_captures/scripts/run.py \
  --output experiments/reverse_transform/raw_keystone/stage_03_independent_captures/results/run
/home/aplab/.pyenv/versions/venv/bin/python experiments/reverse_transform/raw_keystone/stage_03_independent_captures/scripts/audit.py \
  --results experiments/reverse_transform/raw_keystone/stage_03_independent_captures/results/run
/home/aplab/.pyenv/versions/venv/bin/python experiments/reverse_transform/raw_keystone/stage_03_independent_captures/scripts/plot_results.py \
  --results experiments/reverse_transform/raw_keystone/stage_03_independent_captures/results/run
```

The wrapper selects stage 3. Use a new output path for a replay because the runner does not overwrite results.

See [RESULTS.md](RESULTS.md) for saved per-fixation metrics, fitted coefficients, coverage, and plots. The report contains no additional fit or refit.

See [RESULTS.md](RESULTS.md) for saved per-fixation metrics, fitted coefficients, coverage, and figures.
