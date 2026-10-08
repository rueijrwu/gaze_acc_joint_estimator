# Workspace cleanup manifest

This file records the minimum reproducible starting point after the 2026-10-08
cleanup. The cleanup removed stale review documents, duplicate study snapshots,
failed run copies, and intermediate solver output. Source data was not changed.

## Keep for a clean start

- Inputs and frozen baseline: `data/`, `models/quadratic_model.json`,
  `joint_m2.py`, and `lib/`.
- Current geometric model and method: `Theory.md`, `ESTIMATOR_PLAN.md`,
  `full_position/`, and `tests/`.
- Accommodation response definitions and protocols:
  `ACCOMMODATION_RESPONSE_THEORY.md`, `ACCOMMODATION_RESPONSE_PLAN.md`,
  `ACCOMMODATION_FULL_CALIBRATION_PLAN.md`, and `LITERAL_POWER_PLAN.md`.
- Current result summaries and their compact model/input evidence under
  `experiments/full_position/`; see the experiment index and current status.
- The fitted log/square-root/quadratic models and training inputs used by the
  saved-model CPU/GPU solver parity tests.

## Removed

- Stale root audit documents and the obsolete `HANDOFF.md`.
- Repeated `source_snapshot`, `implementation_snapshot`, and `design_snapshot`
  copies. Run manifests retain source and input hashes where available.
- Failed GPU launch copies, the superseded CPU full-calibration run, and the
  interrupted v3 full-calibration run.
- Per-frame, per-start, and inverse-candidate archives from v2/v4 and the
  literal-power screen. Retained full comparisons use the common-frame result
  summaries and certified model artifacts.
- The incomplete literal-power confirmation work files. The completed coarse
  comparison and its frame-level scored results remain. Confirmation and full
  evaluation were not completed.
- Older grouped development, sensitivity, audit-follow-up, transition, and
  application experiment trees. Only current response-law reports and required
  test fixtures remain under `experiments/full_position/`.

The experiment tree was about 15 GB before cleanup and is about 151 MB now.
Source data remains about 60 MB. Previously committed artifacts remain in Git
history where they were committed. Historical phase runners whose saved run inputs
were removed cannot reproduce those old runs from this clean tree.
