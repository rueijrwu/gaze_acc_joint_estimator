# Full-position experiment index

The primary initial results are in [grouped_v2/RESULTS.md](grouped_v2/RESULTS.md).
The frozen baseline, detection files, and prior experiment directories were not
modified.

| Run | Use |
|---|---|
| `grouped_v2/` | Completed five gaze folds and four capture folds, both full-position capacities and a fresh two-channel control |
| `prior_reference_v2/` | Full-development, 24 original rows per fixation, prior 0.001; converged models |
| `zero_prior_v2/` | Matched development sampling without a coefficient prior; all fits reached their evaluation budgets and remain failed checkpoints |
| `gpu_check27/` | Optional CuPy solver versus scalar reference, 32 real-data full/subset checks; agreement on these cases |
| `gpu_check37/` | Optional acceleration disagreements and CPU refinement audit demonstrating a missed lower-cost inverse branch |
| `blas_benchmark.json` | Calibration-kernel thread-count benchmark |
| `gpu_batch_benchmark.json` | Batch-size timing on representative repeated rows; not new validation observations |

Superseded debug/interrupted runs and duplicate worker outputs were removed.
All 27 primary fitted models, frame/point results, sensitivity checkpoints,
GPU audits, and reproduction scripts remain. Identical evaluation-population
CSVs were consolidated into one losslessly compressed `population.csv.gz` per
fold. Worker configuration/completion is retained in fold `execution.json`
files where available. [Cleanup manifest](cleanup_manifest.json) records the
removed runs and the original population SHA-256 hashes.

The initial study is sampled and exploratory. It does not select a final
deployment model or establish physiological accuracy. Captures 5/6 have not been
used for this study.
