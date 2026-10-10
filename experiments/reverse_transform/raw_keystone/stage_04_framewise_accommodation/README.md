# Stage 4: Framewise accommodation-like state

Fits one A value per complete frame in the forward raw model, holding the Stage 3 gaze, keystone coefficients, P1 magnification, and post-fit radial law fixed. Run from the repository root after Stage 3:

```bash
/home/aplab/.pyenv/versions/venv/bin/python experiments/reverse_transform/raw_keystone/stage_04_framewise_accommodation/scripts/run.py \
  --output experiments/reverse_transform/raw_keystone/stage_04_framewise_accommodation/results/run
/home/aplab/.pyenv/versions/venv/bin/python experiments/reverse_transform/raw_keystone/stage_04_framewise_accommodation/scripts/audit.py \
  --results experiments/reverse_transform/raw_keystone/stage_04_framewise_accommodation/results/run
/home/aplab/.pyenv/versions/venv/bin/python experiments/reverse_transform/raw_keystone/stage_04_framewise_accommodation/scripts/plot_results.py \
  --results experiments/reverse_transform/raw_keystone/stage_04_framewise_accommodation/results/run
```

The wrapper selects stage 4. Use a new output path for a replay because the runner does not overwrite results.

See [RESULTS.md](RESULTS.md) for saved per-fixation metrics, fitted coefficients, coverage, and plots. The report contains no additional fit or refit.

The matched normalized-forward control can be recreated with a fresh output path, then its comparison plot can be regenerated from saved data:

```bash
/home/aplab/.pyenv/versions/venv/bin/python experiments/reverse_transform/raw_keystone/scripts/normalized_forward_control.py \
  --output experiments/reverse_transform/raw_keystone/stage_04_framewise_accommodation/results/control_replay
/home/aplab/.pyenv/versions/venv/bin/python experiments/reverse_transform/raw_keystone/scripts/plot_normalized_control_comparison.py \
  --results experiments/reverse_transform/raw_keystone/stage_04_framewise_accommodation/results/run \
  --control experiments/reverse_transform/raw_keystone/stage_04_framewise_accommodation/results/control_replay
```
