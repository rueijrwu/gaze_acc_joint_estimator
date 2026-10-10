# Stage 1: verify the measurements and initialize gaze

This stage answers whether the reviewed measurements can supply a usable starting
point for the distortion estimator. It covers S0/G0 and S1/G1 in the
[stage plan](../../../docs/stages/01_DATA_AND_BOOTSTRAP.md).

It checks the capture hashes, source-point correspondence, frame identities and
measurement availability for all twenty full reviewed intervals. It constructs
translation-invariant P1/P4 coordinates and preserves every scheduled row and
P4 omission slot, including unavailable inputs. It also saves an approximate
covariance for later weighting.

Next it fits a linear inverse using the five low-demand fixation mean centroid
displacements, then applies that inverse to each valid frame's own displacement.
The gaze and demand-based accommodation values are provisional starts. No real
optical calibration, accommodation estimation or omitted-point inference has run.

## Files

| Location | Purpose |
|---|---|
| [scripts/run.py](scripts/run.py) | Reproducible experiment runner |
| [docs/RESULTS.md](docs/RESULTS.md) | Short explanation of the result and its limits |
| [docs/STAGE_REPORT.md](docs/STAGE_REPORT.md) | Attempt ledger and links to detailed gate audits |
| [docs/PROGRESS.md](docs/PROGRESS.md) | Live checkpoint and one next action |
| [results/attempt_01](results/attempt_01/) | Preserved failed loader attempt |
| [results/attempt_02](results/attempt_02/) | Reviewed arrays, plots, summaries and audit |

Reusable estimator code is in [distortion_model](../../../distortion_model/).
Permanent contracts are in [tests/test_stage1.py](../../../tests/test_stage1.py).

## Reproduce

From the repository root, using [requirements-stage1.txt](../../../requirements-stage1.txt):

```bash
python tests/run_stage1.py --output /tmp/acc-stage1-tests-new.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/distortion_model/stage_01_data_and_bootstrap/scripts/run.py \
  --test-report /tmp/acc-stage1-tests-new.json \
  --output experiments/distortion_model/stage_01_data_and_bootstrap/results/new_attempt
```

Choose fresh output paths. The script rejects an existing attempt directory or a
failing, skipped, or source-incompatible test report. Gate decisions require an
execution-owner review of the evidence; script success leaves the result pending review.

The calculations use NumPy float64 vectorization and four threads for independent
capture loading. The five-by-two linear fit needs no GPU optimizer. At the recorded
execution, CuPy 14.2.0 reported no CUDA device.

## Read the saved data

Open arrays with `numpy.load(path, allow_pickle=False)`. Frame arrays share the
ordering in `population.npz`. `slots.npz` has three records per scheduled frame.
`p4_corresponding` in `measurements.npz` is already permuted; do not permute it again.
`initial_states.npz` preserves unavailable inputs, extrapolation and proposed-bound
flags without clipping or replacing trajectories with target labels.

`covariance.npz` preserves native and relative cross-correlations and the P1-edge
marginal. Its contiguous-difference estimate excludes capture 1's unreliable timing
and may contain real eye motion. It is an approximate weighting metric.

`synthetic_fixture.json` and `synthetic_forward.npz` check the DM0 forward equations
with nonzero independent optical offsets. These are numerical fixtures, not fitted
real optics. Full inversion, optical derivatives, symmetry and calibration belong
to later stages.

The original attempt-02 runner is preserved in `results/attempt_02/source_snapshot/`.
Its hash matches that attempt's original provenance. The current runner detects the
repository root after the move; historical outputs and provenance remain intact.
