# Stage 04: framewise accommodation

This stage replaces the prior constant capture barrel coefficient with one framewise accommodation value (A_i), using the frozen post-fit linear barrel law. It keeps the Stage 03 gaze, keystone, P1 magnification, reference, and law fixed. Fixation means have soft 0.25 D anchors; this run adds no framewise anchors and does not refit the law. A is bounded by 0–6 D and the per-frame monotone-inverse domain. All complete frames are fitted without temporal smoothing or P4 magnification fitting.

The canonical run already exists at `results/run/`; the runner refuses to overwrite an existing output folder. To make a fresh run, choose a new output directory, for example:

```bash
python experiments/reverse_transform/stage_04_framewise_accommodation/scripts/run.py \
  --output experiments/reverse_transform/stage_04_framewise_accommodation/results/replay \
  --anchor-width 0.25
```

Then audit and plot that same fresh directory:

```bash
python experiments/reverse_transform/stage_04_framewise_accommodation/scripts/audit.py \
  --results experiments/reverse_transform/stage_04_framewise_accommodation/results/replay
python experiments/reverse_transform/stage_04_framewise_accommodation/scripts/plot_results.py \
  --results experiments/reverse_transform/stage_04_framewise_accommodation/results/replay
```

The canonical run and figures are saved under `results/run/`; its audit summary is `results/run/audit.json`, with the full audit log at `audit_run.log` in this stage directory and the runner log at `console_run.log`. A fresh replay has its own output folder. See [RESULTS.md](RESULTS.md) for the fit, common-valid and all-complete metrics, fixation means, and plots. [SCIENTIFIC_REVIEW.md](SCIENTIFIC_REVIEW.md) records the independent scientific review. The plots use recovered coordinates directly; the radius figure is a diagnostic and does not normalize the fit.
