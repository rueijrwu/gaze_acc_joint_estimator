# Distortion estimator experiments by stage

Follow the [stage gates](../../docs/STAGE_GATES.md). Stages 1–4 have executed; G7 attempt 02 completed uncertified and paused; G8 has not run.
Each implemented stage has its own runner, readable documentation, live resume
pointer, attempt ledger and preserved results.

| Stage | Scope | Plan | Experiment / evidence |
|---|---|---|---|
| 1 | S0–S1 / G0–G1: data census, geometry and centroid gaze bootstrap | [Plan](../../docs/stages/01_DATA_AND_BOOTSTRAP.md) | [Stage 1](stage_01_data_and_bootstrap/README.md): complete, GO_WITH_LIMIT |
| 2 | S2 / G2: P1 reference and common scale | [Plan](../../docs/stages/02_P1_REFERENCE_AND_SCALE.md) | [Stage 2](stage_02_p1_reference_and_scale/README.md): complete, GO_WITH_LIMIT |
| 3 | S3–S5 / G3–G5: independent P4 reference, accommodation baseline and gaze deformation | [Plan](../../docs/stages/03_P4_BASELINE_AND_DEFORMATION.md) | [Stage 3](stage_03_p4_baseline_and_deformation/README.md): G3–G5 complete, GO_WITH_LIMIT; G5 metric tradeoff |
| 4 | S6–S7 / G6–G7: corrected centers and full DM0 calibration | [Plan](../../docs/stages/04_CENTER_AND_DM0_CALIBRATION.md) | [Stage 4](stage_04_center_and_dm0_calibration/README.md): G6 COMPLETE, GO_WITH_LIMIT; G7 attempt 02 COMPLETE_UNCERTIFIED, PAUSE; [attempt report](stage_04_center_and_dm0_calibration/results/g7_attempt_02/STAGE_REPORT.md) |
| 5 | S8 / G8, optional GX: cross-agreement and model decision | [Plan](../../docs/stages/05_CROSSCHECK_AND_MODEL_DECISION.md) | Not run |

Layout within an implemented stage:

```text
stage_NN_<name>/
    README.md                 purpose, file map and run command
    scripts/                  experiment runner
    docs/
        RESULTS.md            readable result and limitations
        STAGE_REPORT.md       attempt ledger
        PROGRESS.md           current checkpoint and one next action
    results/
        attempt_NN/           arrays, plots, provenance and detailed audit
```

Shared mathematical implementation is in [distortion_model](../../distortion_model/),
and permanent checks are in [tests](../../tests/). Full source measurements remain in
[data](../../data/); experiment folders contain derived evidence. Future stage folders
will be created when their implementation begins.
