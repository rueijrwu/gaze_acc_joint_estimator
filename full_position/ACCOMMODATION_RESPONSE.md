# Fixed accommodation response study

The implementation follows [the theory](../ACCOMMODATION_RESPONSE_THEORY.md)
and [the plan](../ACCOMMODATION_RESPONSE_PLAN.md). It adds four response laws.
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
The completed screen is in [the results directory](../experiments/full_position/accommodation_response_v1/RESULTS.md).

Use execution outside the sandbox for this machine's GPU access. The new
response family currently uses the certified scalar CPU solver with parallel
workers. Legacy acceleration rejects power models. GPU use requires separate
forward, derivative, branch and certificate parity checks first. This restriction
also applies to the batched NumPy helper, whose basis is the legacy log basis.

Captures 5/6, detections and historical studies remain unchanged. These optical
development comparisons do not identify physiological accommodation accuracy
or a lens response exponent.
