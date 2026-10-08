# Literal-power exponent study

This implements the approved coarse-to-fine plan. The response is `(A / 1 D)^n`,
with one shared fixed exponent per calibration. The outer search estimates a
useful exponent or exploratory range. It is separate from the shifted-power
family in `ACCOMMODATION_RESPONSE_THEORY.md`.

The initial grid is 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2. A fresh log control uses
the same accommodation bounds. The declared lower bound is 0.01 D; the upper
bound remains 6 D. Powers are evaluated without hidden clipping. Soft mean
anchors remain unchanged; anchors at zero can incur finite penalty at the
positive computational bound. Initial feasible states and the seven zero-A
inverse starts move to the declared bound. All 49 inverse starts are retained.

The coarse stage uses 256 original training rows and 64 separate evaluation
rows per fixation. Training rows form 16 contiguous blocks spread across each
full fixation period, so noise estimation uses uninterrupted, training-only
second differences. Evaluation rows are spread across the remaining interval.
Schedules are fixed before validity filtering, and evaluation denominators
retain invalid rows. Captures 1–4 supply all 20 conditions. Captures 5/6 are
excluded from this study.

The same calibration algorithm fits 27 coefficients and two independent states
per valid frame, with two starts, 300 evaluations per stage, two continuation
stages, coefficient prior 0.001, and soft mean-anchor scales 0.1 degree/0.25 D.
Temporal penalties remain zero. Every candidate refits its own coefficients
and states. The log pilot and weighting covariance are training-only and shared
within a stage. Column-normalized priors use the existing policy; this does not
make regularization identical in function space across exponents.

Certified candidates are compared on identical complete frames with equal
exposure weighting. Every scheduled exposure must contribute. P4 prediction
error ranks the screen; companion state disagreement, tails, bounds, coverage,
and training-support diagnostics remain visible. A failed candidate keeps an
explicit unavailable result and cannot abort result aggregation.

Refinement tests adjacent midpoint exponents around the best power. A boundary
winner adds one extension (0.1 or 3). The two best certified powers proceed to
confirmation with 512 training/128 disjoint evaluation rows per fixation. A
range within 5% of the lowest squared P4 loss is an exploratory shortlist,
not a confidence interval or a physiological measurement of the exponent.
The best confirmed exponent and a matched log control are also evaluated at
lower bounds 0.001 and 0.05 D. Bound sensitivity is reported; it is not hidden
by rescaling states or coefficients.

The confirmed shortlist and log control are frozen before a new full stage.
All 89,175 valid full-period calibration rows and 100,090 scheduled agreement
frames participate, with three held-point inverses per frame. These full
results measure internal consistency. Exponent selection used development
data and does not establish independent validation or authorize deployment.

CuPy FP64 executes profile QR, inner LSMR and all-start inverse refinement and
certification. SciPy controls the outer trust-region loop. Both GPUs are used,
with four workers, four CPU threads each, and 8,192 held-point inverses per
batch. The runner requires a current source-hash-bound validation report.

Run outside the sandbox:

```sh
rtk proxy env PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -u -m full_position.literal_power_study \
  --output experiments/full_position/literal_power_search_v1 \
  --devices 0 1 --workers 4 --cpu-threads 4 --batch-size 8192
```

The runner writes stage manifests, calibration checkpoints, inverse archives,
frame records, scorecards, comparisons, selection records and live progress.
No old fitted models are reused. An uncertified or incomplete confirmation
stops automatic full evaluation with an explicit progress state. Existing
studies and the three-model v4 comparison remain historical artifacts.
