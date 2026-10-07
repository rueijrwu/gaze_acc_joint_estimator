# Full-position prototype

This implements the conditional shared-state estimator in `Theory.md` and
`ESTIMATOR_PLAN.md`. It fits six normalized P4 coordinates with one horizontal
gaze/accommodation state, given the measured P1 triangle. State-dependent shape
changes remain in the observations. The frozen baseline and detections are
unchanged.

The new models use `theta_deg/10`, `log1p(A_diopters)`, and square-root P1 triangle
area. `conditional27` and `conditional37` predict the two translation components
and four matrix entries with separate global coefficients. A fresh 13-coefficient
two-channel control uses the same summary-response family as the baseline, with
new coefficients at scale 10. The frozen coefficients are never loaded into it.

## Verification and experiments

From the repository root:

```bash
rtk proxy python -m pip install -r full_position/requirements.txt
rtk proxy env OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests -v
rtk proxy env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m full_position \
  --output experiments/full_position/my_run \
  --folds gaze capture --train-per-fixation 48 --eval-per-fixation 8 \
  --calibration-starts 2 --max-nfev 500
rtk proxy python -m full_position.report experiments/full_position/my_run
```

Independent folds can run concurrently while keeping small BLAS kernels single
threaded:

```bash
rtk proxy env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m full_position.parallel \
  --output experiments/full_position/my_parallel_run --workers 6 --blas-threads 1
```

The workspace benchmark of QR plus profiled-Jacobian products was faster with
one BLAS thread than with two or four. Recheck for a larger full-data problem:

```bash
rtk proxy python -m full_position.benchmark \
  experiments/full_position/my_run/gaze_-10/conditional27/model.json \
  --threads 1 2 4 --output experiments/full_position/my_thread_benchmark.json
```

Output directories must be new. `--train-per-fixation 0` and
`--eval-per-fixation 0` use all central-core rows. The default sampling selects
evenly spaced original rows **before** validity inspection, retaining invalid
evaluation rows. A core is the central 80% of the reviewed interval. No jump or
residual filter is applied. Training uses all-three-valid sampled observations;
noise estimation uses the complete contiguous valid training cores, never
differences between downsampled frames. Fixed evaluation populations and sampling
membership are recorded once per fold in `population.csv.gz` (read with
`gzip.open` or `pandas.read_csv`). Successful parallel runs retain worker
configuration/completion in each fold's `execution.json` and remove scratch
outputs; failed workers retain their logs for diagnosis.

Gaze folds withhold one complete nominal condition across all four captures.
Capture folds withhold all five fixations of a capture/demand. Both capacities
are evaluated with fixed settings; this first run is an **exploratory paired
comparison**, not nested hyperparameter selection or final model selection.
Endpoint gaze and demand holdouts can be extrapolation. Captures 5/6 remain
untouched until a design is selected. Nominal demand is a soft mean anchor,
not an independent physiological reference.

Calibration profiles the global coefficients by normalized QR, then uses bounded
trust-region least squares and an exact matrix-free profiled Jacobian. The
Jacobian includes changes in optimal coefficients and is frozen at its
linearization point. Optical weights are equal per fixation. The mean-anchor
scales are 0.1 degree and 0.25 D; temporal strength is zero for all models.
Non-intercept coefficients shrink toward a training-only nominal-state
initializer in whitened design-column coordinates. Default prior strength is
0.001; use `--prior 0` for sensitivity runs. Prior strengths are not automatically
comparable across different response designs.

Calibration acceptance requires a physical-state unit-step projected gradient mapping below
`1e-3` and a scaled inner-coefficient gradient below `1e-7`. A callback can certify
convergence when those conditions accompany relative cost change below `1e-9`
and physical step size below `1e-4`; SciPy then records status `-2` for the
intentional callback stop. Accepted standard solver stops also require the
explicit gradient checks. An evaluation limit is a failure regardless of its
cost. Every calibration start and its diagnostics are saved. Failed checkpoints
cannot be loaded as application models by default.

An accepted standard solver stop can pass sufficient stationarity without a
stable-step/cost callback certificate. `acceptance_reason` and
`stable_step_cost_certificate` distinguish the two paths. New calibration
records also separate optical, anchor, and coefficient-prior costs.

The larger candidate also receives a start from the converged smaller candidate's
training states, when available. Those states come from the same training fold;
evaluation observations and all-three evaluation inverses never enter calibration
or P4-subset initialization.

Noise is a provisional robust effective covariance from training-coordinate
second differences divided by six, with five-MAD clipping, 10% diagonal shrinkage,
and a 0.02-pixel floor. It includes motion and detector effects and is not an
independent localization-noise measurement. A common fold-local 27-coefficient
pilot and median training anchor define fixed reference covariances for both
position capacities. The complete residual is differentiated through shared P1
geometry. The summary control uses propagation through its summaries at the same
reference coordinates. Fixed weights are an approximation, not a state-dependent
Gaussian likelihood. Monte Carlo and finite-difference tests check the mathematics.

Each frame and each subset uses the scalar reference solver with all 49 declared
starts: gaze `[-20,-10,-5,0,5,10,20]` crossed with accommodation `[0,...,6]`.
Only retained coordinates and their covariance marginal enter a subset solve.
Every finite start is retained and receives safeguarded exact-Hessian polishing
before clustering. Acceptance requires encoded projected stationarity at most
`1e-4`, a physical Newton correction at most `1e-5` (degrees/diopters), stable
cost, and nonnegative critical-face curvature. Active bounds are respected;
indefinite Hessians receive a descent safeguard. A preceding solver budget stop
is retained as a diagnostic and requires independent polish certification.
Near high-residual minima, stable-cost corrections can improve stationarity
despite floating-point cost cancellation; the gradient threshold is unchanged.
Branch clustering tolerances are 0.01 degree and 0.01 D; a cost difference of 2
defines a heuristic plausible-branch set. This is not a confidence region.
Ambiguous/weak-rank subsets retain their predictions and are inconclusive; the
excluded measurement never selects a branch. Correlated prediction covariance
includes the shared-P1 cross terms. Bounds suppress the local Gaussian score.

`full_position.profile` independently enumerates real stationary gaze roots of
the degree-six fixed-accommodation cost and refines a finite accommodation
grid. It is an audit check, not a proof of complete/global branch coverage.

Reevaluate saved models without retraining, preserving original results:

```bash
rtk proxy env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m full_position.reevaluate \
  --source experiments/full_position/grouped_v2 \
  --output experiments/full_position/audit_polished_v1 --workers 6
rtk proxy env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m full_position.audit_checks \
  --source experiments/full_position/grouped_v2 \
  --output experiments/full_position/audit_checks_v1 \
  --revised experiments/full_position/audit_polished_v1
```

Use new output paths when reproducing. Reevaluation keeps the original eight
selected rows per fixation and writes every start to compressed
`inverse_candidates.jsonl.gz`. `comparison.json` records coverage, state/branch
transitions, errors on matched original/revised support, and common-support
fixation means across all three candidates. Metric names explicitly distinguish
per-frame deviations from equally weighted fixation-mean deviations. Reported
calibration cost comes from the selected start; rejected-start cost is separate.

The audit checks include two recorded high-residual rows, polynomial profiles,
same-response centroid/area diagnostics, retained-x versus retained-x/y P4
prediction, and training-only signed residual/objective decomposition. All-three
summaries never enter excluded-P4 validation. These targeted ablations do not
constitute population-wide information or accuracy comparisons.

## Outputs and application

Each fold/model saves coefficient/schema/provenance JSON, calibration states and
alternatives, a determinant audit, all selected frame results, all subset branches
and predictions, and summaries. The report exports metrics, fixation means,
matched-support errors, and PNG figures. Coverage and failures are reported
separately from errors of testable predictions. Frame estimation uses the
baseline numerical P1 degeneracy rule; P1 condition numbers remain available from
the geometry API. A stronger uncertainty-based P1 gate needs a separate matched
support experiment.

A converged model can be applied with:

```bash
rtk proxy env OPENBLAS_NUM_THREADS=1 python -m full_position.apply \
  experiments/full_position/my_run/gaze_+0/conditional27/model.json \
  data/detections/capture_1_detections.pkl \
  --output experiments/full_position/my_application --every 100 --holdouts
```

`--every 100` is a diagnostic sample; all original rows are still written with
explicit `not_sampled` reasons. Omit it to estimate every usable row. That is
expensive with the intentionally scalar multistart reference. Applying a
fold model to its training capture is a development diagnostic. Model loading
rejects missing/incompatible schema, gaze scale, state encoding, and coefficients.

The current implementation covers the six-residual conditional model, not the
optional nine-component P1/P4 likelihood or ray-traced model. It has no temporal
prior, learned artifact alarm threshold, or automatic point repair. The default
inverse remains the scalar reference. Grouped sensitivity/selection, denser evaluation, noise/reference-policy
sensitivity, independent detector review, and physiological references remain
necessary before replacing the baseline or claiming accuracy.

## GPU acceleration candidate

CuPy is optional. `accelerated.solve_batch` accepts only retained coordinates and
their covariance marginals, with NumPy and CuPy backends. It uses batched bounded
Gauss–Newton candidates, not the scalar trust-region algorithm. The primary
grouped results retain the scalar reference. The GPU candidate is faster for
large batches but requires numerical branch/stationarity checks.

```bash
rtk proxy env OPENBLAS_NUM_THREADS=1 python -m full_position.gpu_check \
  experiments/full_position/my_run/gaze_-10/conditional27/model.json \
  --output experiments/full_position/my_gpu_check --sample 8 --device 0
rtk proxy env OPENBLAS_NUM_THREADS=1 python -m full_position.gpu_benchmark \
  experiments/full_position/my_run/gaze_-10/conditional27/model.json \
  --output experiments/full_position/my_gpu_timing.json --sizes 64 256 1024
rtk proxy env OPENBLAS_NUM_THREADS=1 python -m full_position.diagnostics \
  experiments/full_position/my_run
```

The initial 27-coefficient acceleration audit matched 32 scalar cases. The
37-coefficient audit found differing availability and a lower-cost branch, which
CPU refinement confirmed. Neither finite multistart policy guarantees exhaustive
branches. This is evidence to strengthen inversion before relying on accelerated
application or declaring larger-capacity generalization conclusively worse.
