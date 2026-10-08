# Project guide

This repository contains the frozen estimator baseline, the conditional full-position research package, and the retained accommodation-response studies.

## Documents

- [Theory](Theory.md) and [estimator plan](ESTIMATOR_PLAN.md) describe the measurement model and estimator.
- [Current status](CURRENT_STATUS.md) summarizes the retained results and limits.
- [Accommodation response theory](ACCOMMODATION_RESPONSE_THEORY.md), [response plan](ACCOMMODATION_RESPONSE_PLAN.md), [full calibration plan](ACCOMMODATION_FULL_CALIBRATION_PLAN.md), and [literal power plan](LITERAL_POWER_PLAN.md) define the research workstreams.
- [Experiment index](EXPERIMENTS.md) describes the retained result sets.
- [Cleanup manifest](CLEANUP_MANIFEST.md) records the repository pruning.
- [Full-position package guide](full_position.md) and [accommodation response guide](ACCOMMODATION_RESPONSE.md) describe the Python package.

## Code layout

- `scripts/` contains repository-level analysis and application scripts.
- `full_position/` contains the research estimator package.
- `lib/` and `joint_m2.py` contain the frozen baseline implementation.
- `experiments/` contains experiment-specific scripts and their retained results.
- `tests/` contains permanent tests. `tests/scratch/` is reserved for temporary scripts and local test outputs.
- `data/` and `models/` contain reviewed inputs and frozen model files.

Run repository-level scripts from the repository root, for example `python scripts/apply_quadratic_captures_5_6.py`.
