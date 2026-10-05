# Relative measurement and isolated curvature experiment

This folder owns maintained commands, frozen manifests, audit summaries, predictions
and comparisons for documents/EstimationImplementationPlan.md. Existing full and
reduced calibration artifacts are read-only references. H0, the frozen holdout3
model, is the primary reference requested for mean-gaze comparison; matched full
FF0 is secondary. This is a choice of model-derived reference, not a claim of
physiological truth.

## Completed results: holdout3 and full calibration

All six new calibration cases converged under the original physical gradient,
step, cost and robust-weight rules. H0 remains the primary model-derived reference.
The following comparison uses the same **19,445 original frames in five 3 D
fixations**, with anchor-free inversion for every model. Those fixations are held
out for the holdout3 fits and in sample for the full fits. Mean-gaze RMSE gives
each fixation equal weight; frame RMSE gives each retained frame equal weight.

| Case | Calibration | Mean-gaze RMSE vs H0 (°) | Frame gaze RMSE (°) | Frame A RMSE (D) |
|---|---|---:|---:|---:|
| H0 | Frozen holdout3 reference | 0 | 0 | 0 |
| H0P | Holdout3, previous-mean prior only | <1e-8 | <1e-8 | <1e-8 |
| Hq | Holdout3, curvature | 0.18423 | 0.18705 | 0.02428 |
| HqP | Holdout3, curvature + weak previous-mean prior | 0.17242 | 0.17521 | 0.02392 |
| FC0 | Full, constrained map, no curvature | 0.04662 | 0.04930 | 0.02215 |
| FCq | Full, constrained map, curvature | 0.20220 | 0.20254 | 0.02705 |
| FF0 | Existing full, flexible map | 0.28871 | 0.38343 | 0.07352 |
| FFq | Full, flexible map, curvature | 0.33908 | 0.40587 | 0.07167 |

Holdout3 and full mean-gaze comparison on identical 3 D support (image omitted from this repository)

On this shared support, FC0 is the closest full-data family to H0. Curvature and
additional displacement freedom change the inferred means more. The weak
previous-mean prior pulls HqP modestly toward H0; H0P is effectively a no-op.
These are deviations from the requested reference, not errors against measured
gaze/accommodation. Lower objective values or flatter trajectories do not establish
physiological accuracy. Full training on 3 D makes this comparison in sample for
full models and does not establish held-out performance.

Across all20 fixations (77,756 frames), the full candidate deviations from the
same H0_all20 inverse are reported separately:

| Full case | Mean-gaze RMSE vs H0 (°) | Frame gaze RMSE (°) | Frame A RMSE (D) |
|---|---:|---:|---:|
| FC0 | 0.03886 | 0.03984 | 0.02104 |
| FCq | 0.19291 | 0.19474 | 0.03154 |
| FFq | 0.26944 | 0.29662 | 0.06343 |

The trained heldout3 candidates have no observed ambiguity, unverified inverse
stationarity, active bounds or extrapolation on their five evaluation fixations.
Across all20, H0 has 17,935 nominal-knot extrapolation rows; FC0, FCq and FFq have
15,984, 15,129 and 14,443. These are inferred accommodation outside the trained
protocol knot range, separate from computational bounds; all these cases have
zero active bounds and zero observed ambiguity. FCq has one inverse stationarity
exception, frame 24852/fixation 9 on training support: trusted scalar inversion
changes the frozen state by only 6.08e-7° and 1.02e-6 D, within the existing
branch-equivalence tolerances. Its original flag and predictions remain unchanged.
FFq has no trained inverse stationarity exceptions. The independent inversions of
initial-function coefficients are retained separately; FFq's initializer has
27 ambiguous and 1,431 unverified rows plus 148 accommodation-bound rows, and is
not presented as a validated trained state estimate.

FFq's original 900-second segment did not pass all stopping rules. An unchanged
objective continuation with tighter LSMR accuracy converged in 50.131 seconds;
total recorded runtime is 950.278 seconds. Both segments remain archived. Final q
values in d units are Hq .01807586, HqP .01681727, FCq .01980481 and FFq .01765012.

The [common-support table](summary/common_holdout3_summary.csv),
[complete matrix](summary/matrix_summary.csv),
[controlled contrasts and provenance](summary/matrix_summary.json), and per-case
[holdout3](predictions/Hq/comparison_summary.json) /
[full constrained](predictions/FC0/comparison_summary.json) /
[full flexible](predictions/FFq/comparison_summary.json) reports retain signed
fixation-mean bias, within-fixation SD, conditioning, branch/bound/extrapolation
flags, objective components and source identities. H0 joint training states and
anchor-free inverse states remain separate. Existing FF0 historical runtime is
marked incomplete because its earlier source directory is unavailable.

The predeclared matrix is Hq (holdout3, curvature, previous prior zero), HqP
(holdout3, curvature, previous strength .1), H0P (holdout3, legacy displacement,
previous strength .1), FC0 (all20, holdout3 constrained map, legacy displacement),
FCq (all20, holdout3 constrained map, curvature), and FFq (all20, flexible map,
curvature). Existing holdout3 H0 and matched full FF0 supply zero-prior legacy
controls. All previous priors use 1 degree uncertainty scale; nominal mean anchors
remain at strength 1. No setting is selected using held-out reference trajectories.

The optional curvature coefficient q starts at zero, is appended to the legacy
14 slots and adds q*(theta/15)**2. Its separate penalty is .1/2*(q/range_d)**2,
where range_d is the retained training frame displacement range. q is already
in normalized displacement d units (and equals the curvature contribution at
|theta|=15 degrees); the range is solely a prior denominator, never a multiplier
converting q to an effective coefficient. The initial
function prior also includes the q=0 shape. Noise, exponent, observable ranges and
initial function priors are constructed using each cohort's training observations.
Full constrained initialization projects the initial all20 nominal-mean coefficient
fit into the same displacement map as holdout3. Previous states initialize physical
theta/A only, then coefficients are reprofiled under the newly declared objective.
Previous mean centers are frozen ordinary means on exact source training support.

The scale decision is final: Stage 5 is conditional and its condition is not met.
No independent scale reference or verified repeat is available. On 77756 retained
frames, S1 median is 446.303 pixels, robust CV .264%, and p05–p95 444.893–449.294
pixels, with zero ordering failures. Within-capture nominal gaze slopes range
from -.059 to -.114 pixels/degree; demand is confounded with capture. Small
variation does not establish known scale. z=[m,S1,S4] and crossed pairs remain
quality diagnostics; the state estimator retains normalized [d,rho4]. Internal optical
agreement and deviations from prior model states are not physiological accuracy.

Numerical stopping remains physical gradient 1e-5, reference step 1e-5, relative
cost 1e-10 and robust weight 1e-7. Temporary mathematical verification is run
outside the repository. Checkpoints enforce objective identity; explicit new-objective
state initialization is separate from strict same-objective resume.

The full constrained prior projects all20 means; it is not identical to the
flexible initial function. Under the balanced five-gaze grid this equals the
constrained equal-fixation-mean displacement least-squares fit.


Commands are run from the repository root. The train examples use the actual
predeclared production budgets; each fit also freezes solver_settings.json and its
provenance/manifest. Resumed segments retain their own exact invocation records.
 `--previous-dir` verifies exact training
cohort/frame/observation/source support, rather than requiring an identical model
family label: FF0 may initialize FC0 because both train all20; FF0 cannot initialize
a strict holdout. It decodes physical states and reconstructs the source profile
before reuse. `--resume-dir` also requires unchanged objective identity, including
q, frozen means, scales, strengths, map, source and initial function prior. Repeat
all original settings/source arguments and write to a new destination when resuming.

```bash
rtk proxy env OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 OPENBLAS_NUM_THREADS=1 python -B exp2/relative_calibration/experiment.py measurements --fold holdout3 --output-dir exp2/relative_calibration/measurements/holdout3
rtk proxy env OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 OPENBLAS_NUM_THREADS=1 python -B exp2/relative_calibration/experiment.py train --fold holdout3 --curvature --previous-dir exp2/reduced_calibration/training/holdout3/robust --output-dir exp2/relative_calibration/training/Hq --max-nfev 600 --wall-seconds 900 --lsmr-maxiter 300 --robust-outer 24 --robust-max-nfev 25
rtk proxy env OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 OPENBLAS_NUM_THREADS=1 python -B exp2/relative_calibration/experiment.py train --fold holdout3 --curvature --previous-mean-strength .1 --previous-dir exp2/reduced_calibration/training/holdout3/robust --output-dir exp2/relative_calibration/training/HqP --max-nfev 600 --wall-seconds 900 --lsmr-maxiter 300 --robust-outer 24 --robust-max-nfev 25
rtk proxy env OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 OPENBLAS_NUM_THREADS=1 python -B exp2/relative_calibration/experiment.py train --fold holdout3 --previous-mean-strength .1 --previous-dir exp2/reduced_calibration/training/holdout3/robust --output-dir exp2/relative_calibration/training/H0P --max-nfev 600 --wall-seconds 900 --lsmr-maxiter 300 --robust-outer 24 --robust-max-nfev 25
rtk proxy env OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 OPENBLAS_NUM_THREADS=1 python -B exp2/relative_calibration/experiment.py train --fold full_constrained --previous-dir exp2/full_calibration/matched/robust --output-dir exp2/relative_calibration/training/FC0 --max-nfev 600 --wall-seconds 900 --lsmr-maxiter 300 --robust-outer 24 --robust-max-nfev 25
rtk proxy env OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 OPENBLAS_NUM_THREADS=1 python -B exp2/relative_calibration/experiment.py train --fold full_constrained --curvature --previous-dir exp2/full_calibration/matched/robust --output-dir exp2/relative_calibration/training/FCq --max-nfev 600 --wall-seconds 900 --lsmr-maxiter 300 --robust-outer 24 --robust-max-nfev 25
rtk proxy env OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 OPENBLAS_NUM_THREADS=1 python -B exp2/relative_calibration/experiment.py train --fold full_flexible --curvature --previous-dir exp2/full_calibration/matched/robust --output-dir exp2/relative_calibration/training/FFq --max-nfev 600 --wall-seconds 900 --lsmr-maxiter 300 --robust-outer 24 --robust-max-nfev 25
rtk proxy env OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 OPENBLAS_NUM_THREADS=1 python -B exp2/relative_calibration/experiment.py estimate --fold holdout3 --model-dir exp2/relative_calibration/training/Hq/robust --scope heldout --output-dir exp2/relative_calibration/predictions/Hq
rtk proxy env OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 OPENBLAS_NUM_THREADS=1 python -B exp2/relative_calibration/experiment.py estimate-reference --output-dir exp2/relative_calibration/predictions/H0_all20 --batch-size 512 --inverse-iterations 100
rtk proxy env OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 OPENBLAS_NUM_THREADS=1 python -B exp2/relative_calibration/experiment.py compare --prediction-dir exp2/relative_calibration/predictions/Hq --holdout-prediction-dir exp2/relative_calibration/predictions/H0_all20
rtk proxy env OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 OPENBLAS_NUM_THREADS=1 python -B exp2/relative_calibration/experiment.py summarize --prediction H0=exp2/relative_calibration/predictions/H0_all20 --case Hq=exp2/relative_calibration/training/Hq/robust --prediction Hq=exp2/relative_calibration/predictions/Hq --output-dir exp2/relative_calibration/summary
```

`estimate --scope all` exports all20 on the same anchor-free inverse policy and
marks training support. `compare --holdout-prediction-dir .../H0_all20` uses the frozen H0
anchor-free inverse as primary on exact shared support and labels matched-full
joint calibration states as secondary. H0 joint training states remain separate
in the matrix; they are never mixed with heldout inverse states. `summarize` takes repeated `--case LABEL=directory` and
`--prediction LABEL=directory` arguments for the whole predeclared matrix, includes
H0/FF0 automatically, and reports planned contrasts separately for joint training
states and anchor-free inverses on exact common support. Every output root must be
new; frozen outputs are never overwritten.

The audit/numerical_consistency.json records focused mathematical checks, including
an independent 14-slot basis reference, actual immutable legacy checkpoint profile,
objective and physical checks, dense constrained15 QR, derivative/adjoint checks,
source/resume guards, correlated basis costs and curved inverse branches/bounds.
Pre-feature code snapshots were mechanically reconstructed by reversing this
session's edits; they are explicitly labeled and are supplementary regression
checks, not untouched source snapshots. These tests establish implementation
consistency, not physiological accuracy or global uniqueness.

A max-evaluation=1 smoke produced an expected nonconverged checkpoint; strict
same-objective resume succeeded, and full-cohort constrained source plus synthetic
zero/reversed-order guards passed. This smoke is not a convergence certificate.

The matrix summarizer also verifies and reuses the frozen matched-full anchor-free
inverse from the existing reduced holdout3 report. This FF0 inverse covers only
fixations 10–14; full-cohort candidates may export all20, but each contrast joins
exact common support. It provides a same-inference control separately from FF0
joint calibration states, without changing any candidate predictions or refitting.

H0_all20 is a new prediction artifact from the unchanged holdout3 model, with no
refitting and no state/nominal/temporal anchors. Its training_support flag identifies
the 15 original training fixations; the remaining five are heldout3. Full candidate
comparisons use H0_all20 with the same inference policy on all20, while Hq/HqP/H0P
heldout predictions use their exact common five fixations. The old H0 prediction
artifacts stay unchanged. Mean bias and within-fixation SD are descriptive; the
primary reference choice does not make lower disagreement measured accuracy.

Mean-gaze comparisons to H0 include equal-fixation mean-deviation RMSE,
mean absolute fixation bias and every fixation's signed mean delta, alongside
frame gaze/A RMSE and within-fixation SD. The common fixation count is explicit
(five for heldout candidates, twenty for full candidates against H0_all20).
Different counts represent different evaluation cohorts. Numerical continuations
preserve the same objective and stopping tests; the matrix records each segment's
status/settings and aggregates runtime rather than showing only the final segment.

The [H0 all20 agreement audit](audit/H0_all20_agreement.json) verifies that its
19,445 heldout3 rows reproduce the original frozen H0 inverse exactly for gaze,
accommodation and both observations, with unchanged source hashes. The
[mathematical audit](audit/numerical_consistency.json),
[design and smoke audit](audit/design_and_smoke.json), and
[separation diagnostic](audit/separation_scale_diagnostic.json) retain the evidence
for implementation consistency, strict resume and the scale decision.

The converged flexible-full continuation is reproducible with the same objective:

```bash
rtk proxy env OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 OPENBLAS_NUM_THREADS=1 python -B exp2/relative_calibration/experiment.py train --fold full_flexible --curvature --previous-dir exp2/full_calibration/matched/robust --resume-dir exp2/relative_calibration/training/FFq/robust --output-dir exp2/relative_calibration/training/FFq_continued --max-nfev 600 --wall-seconds 900 --lsmr-maxiter 1000 --lsmr-atol 1e-9 --lsmr-btol 1e-9 --robust-outer 24 --robust-max-nfev 25
```

Use the final FFq_continued/robust model for FFq prediction and summary arguments;
the original FFq/robust budget-limited model remains a retained numerical segment.

The [final run audit](audit/final_run_audit.json) records calibration/inference
status and preserved source hashes. The
[FCq inverse exception audit](audit/FCq_inverse_exception.json) retains the trusted
scalar comparison for the one flagged trained inverse row. Frozen predictions
were not edited to remove the exception.
