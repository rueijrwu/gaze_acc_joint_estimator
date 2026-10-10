# Project guide — exp5_distortion_model

## Active workstream

[Theory.md](Theory.md) defines the joint relative P1/P4 model: separate optical zeros, one P1-derived common scale, and corrected P4-minus-P1 center separation. [ESTIMATOR_PLAN.md](ESTIMATOR_PLAN.md) defines the S0–S8 implementation sequence and the DM0-first full-calibration target.

**Start execution with [STAGE_GATES.md](STAGE_GATES.md).** It supplies the empirical questions, go/repair/pause decisions, evidence records, and checkpoint recovery rules without changing the model or creating another workflow framework.

| Subplan | Existing stages |
|---|---|
| [Data and gaze bootstrap](stages/01_DATA_AND_BOOTSTRAP.md) | S0–S1 |
| [P1 reference and common scale](stages/02_P1_REFERENCE_AND_SCALE.md) | S2 |
| [P4 reference, accommodation, and gaze deformation](stages/03_P4_BASELINE_AND_DEFORMATION.md) | S3–S5 |
| [Corrected centers and full DM0 calibration](stages/04_CENTER_AND_DM0_CALIBRATION.md) | S6–S7 |
| [Cross-agreement and model decision](stages/05_CROSSCHECK_AND_MODEL_DECISION.md) | S8 and conditional extension |

All gates are proposed and NOT_RUN until implementation supplies evidence. The inspected branch at `e4d4ca933a65c2121fbca1a51928d9d308bf6938` contains documentation and reviewed data, not a runnable distortion estimator or its test suite. The stage documents do not start a calibration or claim that the optical assumptions have passed empirical tests.

## Historical context, not current execution evidence

[CURRENT_STATUS.md](CURRENT_STATUS.md), [EXPERIMENTS.md](EXPERIMENTS.md), and [CLEANUP_MANIFEST.md](CLEANUP_MANIFEST.md) describe prior work and pruning. Some inherited paths are absent from this cleaned branch; inspect the live tree instead of treating those paths as installed code or available result artifacts.

The [accommodation-response theory](ACCOMMODATION_RESPONSE_THEORY.md), [response plan](ACCOMMODATION_RESPONSE_PLAN.md), [full-calibration research plan](ACCOMMODATION_FULL_CALIBRATION_PLAN.md), and [literal-power plan](LITERAL_POWER_PLAN.md) retain historical design rationale. [full_position.md](full_position.md) and [ACCOMMODATION_RESPONSE.md](ACCOMMODATION_RESPONSE.md) describe earlier packages. They do not override the active theory, estimator plan, or stage gates.

Source detections and reviewed intervals remain under `data/`. Do not restore old code/result trees, change source observations, or infer implementation completion from an earlier chat. Record the current source commit and compatible run checkpoint when switching machines or resuming work.
