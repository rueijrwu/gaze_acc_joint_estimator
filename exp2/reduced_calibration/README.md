# Constrained reduced-demand calibration

Status: protocol documented first; implementation passed the external numerical
audit. Both final robust holdout fits and the continued matched full fit met
the physical convergence criteria. Both retrospective comparisons and all eight
state/deviation plots are complete.
The earlier unconstrained demand-3 holdout is superseded by the constrained
workflow documented here. Its historical measurements are retained as text in
this report. The legacy reference and matched full fits are organized in
[`../full_calibration/README.md`](../full_calibration/README.md). New experiment
artifacts belong in this directory.

The previous unconstrained demand-3 holdout gave substantial state deviations.
Its training initializer and posthoc 3D displacement interpolation improved
agreement with the stored full-fit trajectory, but the patch increased inverse
ambiguity to 850 of 19,445 frames. That patch is not a validated replacement.
The implemented experiment constrains the missing displacement knot **during fitting**,
and checks inference branches rather than assuming improved RMSE is sufficient.

| Fit | Training demand levels (D) | Withheld capture/fixations | Free coefficients |
|---|---|---|---:|
| Demand-3 holdout | 1000/2775, 2, 4 | Capture 3; fixations 10–14 | 12 |
| Demand-2 holdout | 1000/2775, 3, 4 | Capture 4; fixations 15–19 | 12 |
| Fresh matched full control | 1000/2775, 2, 3, 4 | None | 14 |

Capture 5 is excluded. Use the existing frozen intervals, central-80% cuts and
stored measurement/pupil gates; preserve original-frame support and invalid-gap
temporal breaks. All available comparisons remain retrospective.

Keep the legacy coefficient order `b[4],s[4],rho[1,t,t²,a,at,at²]`, with
`a=A**p`, but define a fixed full-column-rank map `beta=E gamma`. For each
holdout, only b/s values at training demand knots are free. Expand the missing
internal value by exact linear interpolation between its training neighbors.
The ratio block remains six free coefficients. Full-data control uses `E=I14`.
Every quadratic/robust coefficient profile uses `H E`, and the same map enters
the initial-function prior, coefficient response and variable-projection
Jacobian. This is a hard model constraint, not a posthoc edit.

Initialize using ordinary **training** fixation-mean displacement lines and
ratio means. Fit the exponent to training ratio means over `[0.15,1]`, freeze it
within each fit, and interpolate missing b/s from training lines. Estimate
noise covariance only from valid consecutive training-frame differences. Use
the unchanged equal-fixation weights, soft nominal-mean anchors, temporal
penalties, bounds, and block soft-L1 refinement with kappa 2.

All three fits use the same training-only prior policy:
`prior_W=diag(1/ptp(training_frame_observations,axis=0)**2)`, with strength 0.1 on
the same initial-function grid. This keeps the existing demand-3 holdout prior
policy and applies it freshly to the full-data control. Ignore
`calibration_seed.json` and all historical full-fit parameters/checkpoints for
initialization, exponent, covariance and prior estimation. The preserved legacy
full fit remains the primary trajectory reference; the fresh matched full fit
exposes the historical prior/pipeline confound.

Run noise-weighted continuous quadratic fitting, then robust warm start. Retain
the existing physical stationarity/step/relative-cost/weight tolerances and exact
one-sided knot/endpoint checks. Allow larger deterministic budgets and explicit
same-objective continuations; checkpoint every accepted state and report actual
termination. A fit remains `not_converged` if physical criteria are unmet. The
checkpoint identity includes E, training demands, coefficient dimensions,
sources, intervals, frame/group/observation hashes, exponent, prior and noise.

Freeze each trained model and prediction provenance before accessing reference
states. Invert every valid heldout core frame independently with deterministic
bounded batched multistart search, no nominal gaze/A labels, no temporal anchors,
and no reference-state initialization. Also freeze a training-mean-initializer
prediction control for each holdout. Compare original-frame/fixation keys
exactly against preserved and matched full trajectories, verifying observation
and cohort identity. State differences against a model-derived reference are
not physiological accuracy.

The existing pure forward/Jacobian, batched inverse and physical optimality
helpers are extracted unchanged into shared `exp2/estimate_profiled.py`; both
experiments import that module. The new workflow does not depend on code under
the ignored historical experiment directory. Audit the extraction against the
saved original helper source, without changing historical model/prediction
artifacts or maintaining a second inverter implementation.

The implementation is [experiment.py](experiment.py), using the optional fixed
coefficient map in the shared `ProfiledProblem`. Its default no-map path preserves
the legacy coefficient profile. Fresh training runs quadratic then robust:

```bash
python exp2/reduced_calibration/experiment.py train --fold holdout3 --max-nfev 400 --wall-seconds 600 --lsmr-maxiter 300 --lsmr-atol 1e-9 --lsmr-btol 1e-9 --robust-outer 16 --robust-max-nfev 25
python exp2/reduced_calibration/experiment.py train --fold holdout2 --max-nfev 400 --wall-seconds 600 --lsmr-maxiter 300 --lsmr-atol 1e-9 --lsmr-btol 1e-9 --robust-outer 16 --robust-max-nfev 25
python exp2/reduced_calibration/experiment.py train --fold full --max-nfev 400 --wall-seconds 600 --lsmr-maxiter 300 --lsmr-atol 1e-9 --lsmr-btol 1e-9 --robust-outer 16 --robust-max-nfev 25
```

Resume the same objective into a new directory when diagnostics justify it:

```bash
python exp2/reduced_calibration/experiment.py train --fold holdout3 --resume-dir exp2/reduced_calibration/training/holdout3/robust --loss robust --output-dir exp2/reduced_calibration/training/holdout3_next --max-nfev 400 --wall-seconds 600 --lsmr-maxiter 300 --lsmr-atol 1e-9 --lsmr-btol 1e-9 --robust-outer 16 --robust-max-nfev 25
```

An explicit `--warm-start-dir` permits only quadratic-to-robust refinement. Every
source checkpoint must match fold, map, data, exponent, precision and prior.
Choose the intended final checkpoint explicitly for prediction; continuations
do not silently replace earlier outputs.

```bash
python exp2/reduced_calibration/experiment.py estimate --fold holdout3 --model-dir exp2/reduced_calibration/training/holdout3/robust
python exp2/reduced_calibration/experiment.py estimate --fold holdout2 --model-dir exp2/reduced_calibration/training/holdout2/robust
python exp2/reduced_calibration/experiment.py compare --fold holdout3 --prediction-dir exp2/reduced_calibration/predictions/holdout3 --matched-dir exp2/full_calibration/matched/robust
python exp2/reduced_calibration/experiment.py compare --fold holdout2 --prediction-dir exp2/reduced_calibration/predictions/holdout2 --matched-dir exp2/full_calibration/matched/robust
```

Prediction writes trained and initializer controls with frozen hashes and full
original-row status CSVs. Comparison validates both prediction hashes/support
before reading baseline files, then writes per-reference state/delta plots and
all-frame/per-fixation metrics. It also inverts the same heldout observations with
fixed fresh-full coefficients under the same inversion protocol; this separates
inference/anchor differences from calibration differences. Historical joint
states remain the primary reference. Reliable-subset metrics always include
coverage, and an empty subset has count zero and null metrics.

Audit the reduced coefficient map, derivatives and profile stationarity, heldout
mutation invariance, strict frame joins, synthetic bounded/multiple-root
inversion, sampled agreement with the trusted inverse, and noise-scaled physical
Jacobian grids. Report one-sided fixed-displacement ratio slopes, near-equivalent
branches, local conditioning, bound/extrapolation rates, reliable-subset coverage,
and all-frame/per-fixation state RMSE and bias. Local rank or physical
stationarity alone does not certify global uniqueness.

Tests and harnesses ran outside the repository; their transient logs were
discarded after recording results here. No repository test directories are added. Reports must
include training convergence and actual budgets for every stage, distinguish
the preserved legacy reference from fresh matched control, and label all work
retrospective. A successful result would justify independent repeat/reference
validation, not a claim that three demands are universally sufficient.

The external audit passed exact regression of the legacy no-map coefficient
profile/residual/Jacobian, identity-map agreement, independent dense constrained
QR, reduced Jacobian finite differences and adjoint products, constraint
preservation, and derivative continuity at omitted knots. It also passed both
folds' heldout observation/label mutation checks, strict original-frame joins,
map/provenance mismatch rejection, shared inverse extraction regression,
synthetic bounded/ambiguous inversion, actual legacy checkpoint reconstruction,
and a two-evaluation actual-data checkpoint/resume smoke check. These tests
establish numerical consistency of the implementation; they do not establish
calibration convergence, global inverse uniqueness, or physiological accuracy.
The production runs use 400 evaluations and 600 seconds per stage, LSMR cap 300
with atol/btol 1e-9, and sixteen robust outer rounds of at most 25 evaluations.
The physical stopping criteria remain unchanged.

The completed holdout stages have the following actual stopping results. Solver
callbacks stop a run only after the physical, step, cost and robust-weight checks
pass; their `StopIteration` message is therefore consistent with convergence.

| Fold/stage | Physical status | Evaluations | Elapsed seconds | Detail |
|---|---|---:|---:|---|
| Hold 3D, quadratic | Not converged | 134 | 394.32 | `ftol` satisfied, but projected physical optimality 1.2987e-5 exceeds 1e-5 |
| Hold 3D, robust | Converged | 83 | 91.45 | Objective 0.6622429002; projected physical optimality 9.00216e-6 |
| Hold 2D, quadratic | Converged | 99 | 309.59 | All declared physical stopping criteria passed |
| Hold 2D, robust | Converged | 200 | 157.26 | Objective 0.4912696588; projected physical optimality 3.79565e-6 |
| Matched full, quadratic first pass | Not converged | 113 | 601.52 | Wall limit; projected physical optimality 0.0044895 |
| Matched full, robust first pass | Not converged | 166 | 601.90 | Wall limit; objective 0.5553566471; projected physical optimality 0.000775233 |
| Matched full, robust continuation | Converged | 181 | 768.69 | Objective 0.5553252317; projected physical optimality 2.69604e-6 |

The preceding hold-3D quadratic stage's failed stationarity check does not negate
the final robust stage's independently audited convergence. The retained legacy
reference and previous unconstrained holdout have their own stopping histories;
their known nonconvergence does not describe these new converged robust fits.

The matched full robust objective was continued unchanged into
`../full_calibration/matched/robust/`, with 600 evaluations, 900 seconds, LSMR cap 1,000
and 24 robust outer rounds of at most 25 evaluations. Linear-solve and physical
tolerances are unchanged. The first pass hit the LSMR iteration cap in 109 of
121 robust linear solves while states/objective were still changing; its wall
termination does not establish a permanent solver stall. The continuation passed
all physical checks: maximum reference-scaled step 2.56012e-8, relative cost
change 1.78331e-13, fixed-weight change 1.48206e-8, and no zero-A or active-bound
issues. One near-but-not-exact knot frame remained; its gradient passed the
unchanged stationarity test. Use `../full_calibration/matched/robust` as the final
matched control, preserving the earlier nonconverged checkpoints in
`../full_calibration/matched/robust_initial`.

Both holdout prediction stages have frozen trained and initializer CSVs with
matching unique original-frame cohorts: 19,445 demand-3 frames and 19,070 demand-2
frames. All states are finite and within computational bounds. No reference
states were read for these predictions.

The trained demand-3 inversion has zero flagged ambiguity, bound, extrapolation
or physical-stationarity failures. The trained demand-2 inversion has zero
ambiguity/extrapolation and four +20° bound frames whose physical stationarity
is unverified (fixation 19, frames 27,702–27,705); these remain included and
flagged. The demand-3 initializer has
15 ambiguous frames and 1,109 extrapolated frames; the demand-2 initializer has
no such flags. Initializer performance is a separate control, not evidence of
adequate trained inverse behavior.

Each trained model's sampled grid contains 425 calibrated-support Jacobians,
425 full-bound coordinates (408 have defined derivatives; 17 at A=0 are singular
for p<1), and 36 one-sided protocol-knot Jacobians. All sampled determinants are
negative. Calibrated-support maximum noise-scaled condition numbers are 32.16
(hold 3D) and 32.34 (hold 2D); full-bound maxima are 82.83 and 63.35. Sampled
fixed-displacement ratio slopes remain negative in both domains. Each trained
and initializer model also passed 135 calibrated-support forward/inverse probes,
with no selected-state mismatch, missing root, ambiguity or stationarity failure.
These are finite sampled checks, not a global certificate; the four observed
demand-2 boundary cases illustrate why per-frame flags are also necessary.

The targeted boundary audit confirmed genuine bounded optima
at theta=+20° for all four flagged observations. Trusted SciPy inversions agree
with an independent scalar boundary solve and meet physical stationarity
(SciPy optimality about 8e-8 to 5e-7). The frozen batched states differ in A by at
most 2.41e-5D, despite their larger numerical gradient residuals. Extending theta
only for diagnosis finds exact optical roots at 20.128–20.173°, outside the
declared bounds. Production predictions and bounds were not changed. The four
frames remain in all-frame metrics and are excluded from the reliable subset;
these flagged numerical residuals must not be presented as converged inversions.

## Frozen holdouts versus the preserved full fit

The retrospective legacy audit verified source,
cohort, observation and original-frame identity after both controls were frozen.
All frames are included below. These are deviations from model-derived legacy
full-fit states, not measured gaze/accommodation accuracy. The converged fresh
matched control below tests whether the conclusion persists under the new
training-only prior policy and initialization.

| Fold/model | Frames | Gaze RMSE (°) | A RMSE (D) | Gaze bias (°) | A bias (D) |
|---|---:|---:|---:|---:|---:|
| Hold 3D, trained constraint | 19,445 | 0.381232 | 0.078568 | -0.110657 | +0.000712 |
| Hold 3D, training initializer | 19,445 | 0.381839 | 0.183751 | +0.029414 | +0.068589 |
| Hold 2D, trained constraint | 19,070 | 2.133779 | 0.200814 | +0.472995 | +0.158284 |
| Hold 2D, training initializer | 19,070 | 2.064560 | 0.176336 | +0.317774 | +0.120634 |

The previous unconstrained demand-3 experiment gave 1.346193°/0.353339D RMSE;
its retrospective posthoc patch gave 0.669661°/0.124942D with 850 ambiguous
frames. The new constrained demand-3 fit has zero ambiguous frames and much
smaller state deviations. It also used larger optimization budgets and achieved
physical convergence, so the change cannot be attributed solely to the hard
constraint.

| Hold 3D fixation | Gaze RMSE (°) | A RMSE (D) | Gaze bias (°) | A bias (D) |
|---|---:|---:|---:|---:|
| 10 | 0.289069 | 0.016657 | +0.269739 | +0.011402 |
| 11 | 0.074242 | 0.045117 | +0.062875 | +0.045116 |
| 12 | 0.099335 | 0.043451 | -0.087143 | +0.043423 |
| 13 | 0.380460 | 0.031497 | -0.300118 | +0.005962 |
| 14 | 0.679181 | 0.154879 | -0.455982 | -0.096137 |

| Hold 2D fixation | Gaze RMSE (°) | A RMSE (D) | Gaze bias (°) | A bias (D) |
|---|---:|---:|---:|---:|
| 15 | 2.713241 | 0.249361 | -2.698230 | +0.248872 |
| 16 | 0.613487 | 0.059174 | -0.589254 | +0.059105 |
| 17 | 0.339811 | 0.026645 | +0.337999 | +0.026602 |
| 18 | 2.302475 | 0.130228 | +2.300588 | +0.128167 |
| 19 | 3.246322 | 0.362116 | +3.243965 | +0.359947 |

Both tables follow nominal gaze targets [-15,-7.5,0,7.5,15] as fixation-mean
references only. The four flagged demand-2 boundary observations remain in
fixation 19 and all-frame metrics.

The saved-initializer geometry audit helps explain the
asymmetry. Interpolating the training 2D/4D mean lines predicts the missing 3D
gain s=0.04009747, almost equal to its retrospective observed nominal-mean line
(0.04004933; ratio 1.0012). Interpolating 0.36D/3D predicts missing 2D gain
s=0.03347865, only 79.90% of its observed line gain 0.04190237; the trained
constrained gain is 0.03314103 (79.09%). A smaller displacement gain inflates
inferred gaze magnitude, consistent with the negative/positive extreme-gaze
biases and poor demand-2 initializer before joint fitting.

These observed nominal-mean lines are retrospective diagnostics, not independent
state truth or causal proof of optical curvature. Between-demand response shape,
capture/session changes, and nominal accommodation assumptions remain possible
contributors. The complementary holdout shows that three-demand interpolation
is not uniformly adequate for this dataset; good 3D holdout behavior alone does
not validate dropping 2D calibration. No holdout observation was used to alter
or tune the frozen candidate models.

## Completed matched-control comparisons and plots

Both [hold-3D summary](predictions/holdout3/comparison_summary.json) and
[hold-2D summary](predictions/holdout2/comparison_summary.json) validate the frozen
candidate hashes and exact original-frame/observation identity before reference
access. Reference model and state-CSV hashes are retained in those summaries.
The selected matched full fit is `../full_calibration/matched/robust`; all three
selected robust fits converged. The preserved legacy reference did not pass
its own physical convergence checks, so it is retained as a historical control.

| Fold/model versus fresh matched full joint states | Gaze RMSE (°) | A RMSE (D) | Gaze bias (°) | A bias (D) |
|---|---:|---:|---:|---:|
| Hold 3D, trained constraint | 0.382697 | 0.073182 | -0.111354 | -0.007792 |
| Hold 3D, training initializer | 0.373222 | 0.183356 | +0.028717 | +0.060085 |
| Hold 2D, trained constraint | 2.156439 | 0.191460 | +0.483957 | +0.141545 |
| Hold 2D, training initializer | 2.086813 | 0.169409 | +0.328737 | +0.103895 |

The trained reliable subsets cover 100% of hold-3D frames and 99.9790% of
hold-2D frames (19,066/19,070). All-frame metrics above retain the four flagged
boundary frames. Both trained inversions have zero ambiguous frames and zero
accommodation extrapolation. Per-fixation, absolute-error and reliable-subset
metrics are stored in the linked summaries and comparison CSVs.

The additional fresh-full fixed-coefficient inverse uses exactly the same
anchor-free inversion protocol as each holdout. Its difference from fresh-full
joint states is only 0.003795°/0.001650D RMSE for the 3D cohort and
0.001777°/0.000815D for the 2D cohort. Both controls have zero ambiguity,
stationarity failures, bounds or extrapolation. Thus the much larger holdout
differences are associated primarily with calibrated model differences, rather
than removal of state anchors during inference, within this dataset.

The conclusion is consistent against both full references: the constrained
0.36D/2D/4D model gives substantially better 3D holdout agreement than the old
unconstrained fit, without its inverse-branch problem. The complementary
0.36D/3D/4D model retains substantial gaze errors at 2D. These retrospective
results support testing the former reduced protocol with independent repeat
and state-reference measurements; they do not support uniformly dropping any
internal demand or claiming physiological accuracy.

Each plot has five fixation rows and four columns: gaze state, accommodation
state, gaze deviation, and accommodation deviation. The horizontal coordinate
is the original frame index; deviations are candidate minus reference.

| Candidate | Preserved legacy reference | Fresh matched full reference |
|---|---|---|
| Hold 3D, trained | State and deviation plot (plot omitted from this repository) | State and deviation plot (plot omitted from this repository) |
| Hold 3D, initializer | State and deviation plot (plot omitted from this repository) | State and deviation plot (plot omitted from this repository) |
| Hold 2D, trained | State and deviation plot (plot omitted from this repository) | State and deviation plot (plot omitted from this repository) |
| Hold 2D, initializer | State and deviation plot (plot omitted from this repository) | State and deviation plot (plot omitted from this repository) |
