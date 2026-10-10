# Stage 2: P1 reference and scale

Implements S2/G2 from [the subplan](../../../docs/stages/02_P1_REFERENCE_AND_SCALE.md).
Status: **COMPLETE, GO_WITH_LIMIT**, reviewed attempt_02.

[Results](docs/RESULTS.md) · [Audit ledger](docs/STAGE_REPORT.md) · [Resume pointer](docs/PROGRESS.md)

## Files

- [scripts/run.py](scripts/run.py): full reviewed-data experiment, snapshots, arrays, diagnostics and plots.
- [distortion_model/p1.py](../../../distortion_model/p1.py): shared P1 model, analytic profile derivatives, vectorized objective and conditional fitter.
- [distortion_model/alignment.py](../../../distortion_model/alignment.py): observed pattern balance and empirical reference.
- [tests/test_stage2.py](../../../tests/test_stage2.py): permanent mathematical and implementation contracts.
- `results/attempt_01/`: preserved uncertified attempt and repair record.
- `results/attempt_02/`: reviewed output with exact source snapshot, provenance and checkpoint.

The experiment runner measures the recordings; permanent tests check the implementation using controlled inputs. They serve separate purposes. The runner requires a passing report whose source hashes match the current code.

## Reproduce

From repository root, use Python with [requirements-stage2.txt](../../../requirements-stage2.txt). Execute outside the sandbox as requested. Choose fresh output names; existing evidence is never overwritten.

```bash
python tests/run_stage2.py --output /tmp/acc-stage2-tests-reproduction.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/distortion_model/stage_02_p1_reference_and_scale/scripts/run.py --test-report /tmp/acc-stage2-tests-reproduction.json --output experiments/distortion_model/stage_02_p1_reference_and_scale/results/attempt_03
```

Default parent is stage 1 attempt_02. Raw data hashes, population, covariance, measurements and parent starts are checked. Result exit success leaves a pending review; the execution owner records the gate decision. Runtime here was 12.75 s on CPU with four independent worker threads. CuPy/GPU was available; no GPU backend is required for this small fit.

Next: S3/G3 independent P4 reference and dual-zero conversion. The later S4/G4 and S5/G5 tasks have their own gates.
