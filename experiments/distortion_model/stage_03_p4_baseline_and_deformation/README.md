# Stage 3: P4 reference, accommodation baseline and gaze deformation

Follows [the S3–S5 subplan](../../../docs/stages/03_P4_BASELINE_AND_DEFORMATION.md).
**S3/G3, S4/G4 and S5/G5 complete, GO_WITH_LIMIT.** G5's matched weighted/native results show a tradeoff; full optical calibration remains outstanding.

[Results](docs/RESULTS.md) · [Gate ledger](docs/STAGE_REPORT.md) · [Resume pointer](docs/PROGRESS.md)

## Files

- [scripts/run.py](scripts/run.py): G3 independent P4 reference and dual angles.
- [scripts/run_accommodation.py](scripts/run_accommodation.py): G4 near-reference DM0-M1, dynamic states, soft full means and declared sensitivities.
- [scripts/run_deformation.py](scripts/run_deformation.py): G5 native K4, full-frame A/M update and matched identity comparison.
- [scripts/g45_common.py](scripts/g45_common.py): checkpoint verification and evidence output.
- [distortion_model/accommodation.py](../../../distortion_model/accommodation.py): composed derivatives, native shape marginal, bounded state profile and conditional fit certificate.
- [distortion_model/p4.py](../../../distortion_model/p4.py): independent reference and native-angle diagnostics.
- [tests/test_stage4.py](../../../tests/test_stage4.py) and [test_stage5.py](../../../tests/test_stage5.py): permanent S4/S5 numerical contracts; test numbers follow S-step/gate numbers.
- `results/attempt_01/`: G3 evidence; `results/g4_attempt_01/`: G4 evidence; `results/g5_attempt_01/`: G5 evidence. Each contains exact source snapshots, provenance, arrays, plots and audit/checkpoint.

## Reproduce

Execute from repository root outside the sandbox as requested. Use [requirements-stage4.txt](../../../requirements-stage4.txt) / [requirements-stage5.txt](../../../requirements-stage5.txt). Choose fresh names; prior evidence is preserved. Default parents are the reviewed checkpoint above each gate. A successful execution still awaits scientific gate review.

```bash
python tests/run_stage4.py --output /tmp/acc-g4-tests-reproduction.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/distortion_model/stage_03_p4_baseline_and_deformation/scripts/run_accommodation.py --test-report /tmp/acc-g4-tests-reproduction.json --output experiments/distortion_model/stage_03_p4_baseline_and_deformation/results/g4_attempt_02
python tests/run_stage5.py --output /tmp/acc-g5-tests-reproduction.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/distortion_model/stage_03_p4_baseline_and_deformation/scripts/run_deformation.py --test-report /tmp/acc-g5-tests-reproduction.json --output experiments/distortion_model/stage_03_p4_baseline_and_deformation/results/g5_attempt_02
```

The current stage4 test runner discovers later permanent contracts too; archived G4 execution evidence has 50 passes and G5 has 55. These check mathematics/software; runners measure the recordings. CPU array operations keep native measurement/scale consistency; CuPy was available but unnecessary here. G3 reproduction remains `run.py` with `tests/run_stage3.py`, a fresh `results/attempt_NN` path and its reviewed G2 parent.

Next: stage 4 S6/G6 centers and S7/G7 full DM0 calibration.
