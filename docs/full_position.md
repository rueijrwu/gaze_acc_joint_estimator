# Full-position estimator

This package implements the conditional shared-state estimator described in [`Theory.md`](Theory.md) and [`ESTIMATOR_PLAN.md`](ESTIMATOR_PLAN.md). It predicts six P4 coordinates from the measured P1 triangle and estimates horizontal gaze and accommodation independently for each frame. It is an exploratory geometric model, not a physiological ground-truth estimator.

## Core implementation

- `accommodation.py`, `literal_power.py`: fixed accommodation response families and their basis derivatives.
- `calibrate.py`, `profile.py`, `gpu_profile.py`, `device_lsmr.py`: profiled coefficient fitting and CPU/CuPy numerical backends.
- `invert.py`, `batched_inverse.py`, `batched_holdout.py`: bounded 49-start state inversion, certification, and held-point evaluation.
- `population.py`, `crosscheck.py`, `scorecard.py`, `selection.py`: frozen schedules, common cohorts, scorecards and nested comparison support.
- `accommodation_study.py`, `accommodation_full_accelerated.py`, `literal_power_study.py`: response-law studies.

## Reproduction

From the repository root, install `requirements.txt`. Use a new output directory for each study to preserve saved evidence. The full-position CLI is:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m full_position \
  --output experiments/full_position/new_run \
  --folds gaze capture --train-per-fixation 48 --eval-per-fixation 8 \
  --calibration-starts 2 --max-nfev 500
```

Use `python -m full_position.report <run-directory>` to summarize a run. The response-law reports and current run state are linked from [`CURRENT_STATUS.md`](CURRENT_STATUS.md).
