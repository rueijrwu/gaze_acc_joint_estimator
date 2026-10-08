# Fixed accommodation response study

The implementation follows [the theory](ACCOMMODATION_RESPONSE_THEORY.md)
and [the plan](ACCOMMODATION_RESPONSE_PLAN.md). It adds four response laws.
It keeps the legacy log models and their artifact schema unchanged.

`PowerResponseModel` uses 27 coefficients for each law. Its exponent is fixed
at 0, 0.5, 1 or 2. The response is shifted Box–Cox with a scale of 1 diopter.
The direct linear accommodation term in horizontal displacement remains.
Each valid training frame has its own horizontal-gaze and accommodation state.
The fit has no temporal penalty. The nominal target is not a measured state.

The new schema requires the response family, exponent, coefficient order,
units, state scales, normalization, log pilot and covariance policy. A legacy
artifact cannot be loaded as a power artifact. A failed calibration is a
checkpoint. It cannot be used for prediction.

The first screen fits all four laws on five gaze splits and four capture
splits. It samples 48 training rows and eight evaluation rows per fixation
from the central 80%, before validity filtering. Each split creates a fresh
training-only log27 pilot and covariance. All four candidates share them.
Each candidate creates its own coefficient initializer, column scales and
prior. The prior strength is 0.001. Equal numerical strength does not give
equal function-space regularization across the laws.

The fixation-mean anchor scales are 0.10 degree and 0.25 diopter. They are
finite penalty scales. They are not accuracy limits or allowed movement.
The report separates nominal mean offset, temporal spread and nominal RMS.
The physical optimizer bounds are distinct from empirical training support.

For each evaluation frame, three inverses each retain only two P4 points.
The solver uses all P1 points, retained P4 x/y and the covariance marginal.
It freezes the state and branches before the scorer reads the held point.
Prediction loss and same-frame state disagreement are reported separately.
Full, shared interior and shared support cohorts use exact frame and point
identities. The frame cohort and point cohort can have different memberships.

## Full-population calibration

`full_position.accommodation_full` runs four fresh calibrations, one for each
fixed response law, on the same 20 reviewed conditions and all 89,175 valid
rows in the full fixation periods. It uses the existing joint optimizer and
scalar CPU inverse; four fit workers use one BLAS, OpenMP and MKL thread each.
The accelerated runner keeps that optimizer and inverse unchanged and offloads
the verified float64 linear Jacobian products to CuPy. This product parity does
not verify response-inverse branches, fit certificates or full-run convergence.
The old sampled screen fits are not reused. The `core` training and agreement
windows remain the CLI defaults; each window can be set independently.

The core defaults use the central-80% training rows (71,784 rows) and all
80,072 core frames for agreement per law. This run explicitly sets both windows
to `fixation_period`: it fits all 89,175 valid rows per law and schedules all
100,090 raw frames per law for agreement. Invalid or unavailable measurements
remain accounted for in the schedule; the complete schedule contains
1,201,080 P4 slots across four laws. Positive `--agreement-per-fixation`
values sample only agreement checks and never reduce calibration rows.

The current optional acceleration uses GPU profile QR and inner LSMR products
for fitting, while SciPy keeps control of the outer trust-region loop on the
host. It also batches the raw agreement frames and all 49 inverse starts on GPU
through refinement and certification. Float64 CuPy and Torch are supported by
the inverse interface; CuPy is the preferred measured backend here. This path
must pass both source-hash-gated reports before it starts. The profile report
covers all 89,175 fitting rows for all four response laws. The inverse report
covers backend tests, 180 representative real frame-law checks across all 20
conditions and larger throughput samples; it does not establish full-run
convergence or agreement results.

For a future fresh full-population run, use execution outside the sandbox for
GPU access and choose a new output directory:

```sh
rtk proxy env PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m full_position.accommodation_full_accelerated \
  --output experiments/full_position/accommodation_full_v4_gpu \
  --workers 4 --cpu-threads 4 \
  --linear-backend cupy \
  --parity-report experiments/full_position/gpu_vectorization_validation/profile_gpu_parity_report.json \
  --inverse-backend cupy --inverse-batch-size 8192 \
  --inverse-parity-report experiments/full_position/gpu_vectorization_validation/inverse_parity.json \
  --training-window fixation_period --agreement-window fixation_period \
  --max-nfev 300 --seed 17 --source-commit <current-40-character-commit>
```

The runner refits all four laws and does not reuse previous models. It includes
all valid fixation-period training rows and all scheduled fixation-period
agreement frames. Four workers use both GPUs, with two workers per card. The
default batch has 8,192 frames and 401,408 start candidates per worker, as
requested. This was the fastest measured batch in the memory test. A concurrent
test reserved 3 GiB for fitting per worker; peak free memory remained
5,672 MiB on the RTX 5070 Ti and 6,322 MiB on the RTX PRO 2000. The 32,768-frame
batch exceeded the 6.75 GiB per-worker allocation cap and failed the memory
headroom check. Throughput at 16,384 was about 2% below 8,192 on the sampled
subsets; larger batches do not establish a speed gain. See the
[memory report](../experiments/full_position/gpu_vectorization_validation/batch_memory_tuning.json).
These defaults apply to this two-card, four-worker setup; retune for other
hardware or worker counts. The default assigns one GPU to each law; `--inverse-devices 0
1` opts into splitting each law's inverse frames across both devices. On the
group-stratified 4,096-row comparison, the two-GPU option was slower than GPU 0
alone, so it is not enabled by default. The small 180-check CPU/GPU timing
sample was also slower on GPU; these are scoped throughput measurements, not
whole-run speed estimates. The profile gate is
[here](../experiments/full_position/gpu_vectorization_validation/profile_gpu_parity_report.json),
the inverse gate is
[here](../experiments/full_position/gpu_vectorization_validation/inverse_parity.json),
and the paired stratified timing record is
[here](../experiments/full_position/gpu_vectorization_validation/groupstratified_benchmark.json).
Passing these gates does not imply that a full calibration or agreement run has
completed, nor does it determine a winning response law.

```sh
rtk proxy env PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m full_position.accommodation_full \
  --output experiments/full_position/accommodation_full_v2 \
  --workers 4 --agreement-per-fixation 0 \
  --training-window fixation_period --agreement-window fixation_period
```

Use a new output directory for every invocation; each run refits all four
laws. This comparison measures internal cross-agreement, not independent
validation or physiological accuracy. It has no absolute RMS acceptance gate
and does not select a deployment model. Results are pending until the run
completes.

## Selection policy

`accommodation_selection.choose` provides a versioned inner selection policy.
It requires an independent immutable frame manifest with three held slots.
The sampled policy requires 80% scored, complete and shared frame coverage.
It requires 50% shared frame coverage in each expected exposure. Every
exposure must contribute. These are comparability rules, not accuracy limits.
Numerical certification and finite scores remain required.

Eligible candidates are compared by equal-exposure paired squared prediction
loss. An exact tie favors the log reference. State disagreement, axes, tails,
bounds and support remain visible as diagnostics and tradeoff flags. There
are no absolute accuracy or nominal-target cutoffs. Selection allows an outer
evaluation. It does not allow deployment promotion.

`accommodation_selection.nested_grouped` reuses the existing sealed coordinator
with the new policy. Declare inner partitions that hold out horizontal-gaze
conditions across captures, or whole captures. Declare all evaluation schedules
before predictions. Fit each inner candidate from its inner training groups.
A denser nested experiment has not been run. The sampled screen cannot choose
a global winner from its outer development scores.

## Run

Set the source commit to the current 40-character commit ID. Use the recorded
working-source hashes to identify uncommitted implementation changes.

```sh
rtk proxy env PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m full_position.accommodation_study \
  --output experiments/full_position/accommodation_response_v1 \
  --source-commit <source-commit> --workers 12
```

The output directory must not exist. The runner keeps failed fits, starts,
checkpoints, branches, fixed schedules and source snapshots. It checks that
historical files and implementation sources did not change during the run.
The older shifted-response screen output was removed during repository cleanup. Its minimal fitted-model fixtures remain under `tests/fixtures/response_inverse/` for the inverse contract test; no response-screen result set is retained.

The previous interrupted run used the earlier product-only acceleration and
scalar CPU inverses. Its partial results remain an interrupted record and are
not resumed or reused by the command above.

Captures 5/6, detections and historical studies remain unchanged. These optical
development comparisons do not identify physiological accommodation accuracy
or a lens response exponent.
