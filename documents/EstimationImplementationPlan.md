# Relative measurements and nonlinear gaze: implementation plan

Status: optional normalized curvature and frozen previous-mean priors are
implemented and mathematically verified in
[relative_calibration](../exp2/relative_calibration/README.md). Predeclared
calibrations are complete in that new artifact root: all six new cases passed
the unchanged stopping rules, with the flexible full case requiring a retained
same-objective continuation. H0 is the primary comparison reference; the report
shows holdout3 and full results on identical 3 D support and separately on all20. The scale gate retains two normalized state
channels and the third relative coordinate as a quality diagnostic. Mathematical definitions are
in [Theory.md](Theory.md); current and proposed objectives are in
[Estimation.md](Estimation.md).

## Intended outcome

Use the available relative Purkinje measurements with a shared gaze/accommodation
state, allow modest nonlinear gaze response, and optionally stabilize subsequent
fitting with previous fitted fixation means. Determine whether preserving
separation magnitude supplies useful state information after accounting for
scale. Absolute image position is not an estimator input.

Keep three questions separate:

1. Does a small nonlinear mean-displacement term improve the forward model?
2. Does a third relative measurement help once its nuisance variables are counted?
3. Does reusing fitted means stabilize fitting without simply preserving bias?

## Stage 1: relative observation contract and mathematical checks

Extend [observations.py](../exp2/observations.py) with an explicit relative-output
mode or separate helper. Preserve the existing two-channel helper and its sign
convention. The independent pixel-valued vector is `z = [m, S1, S4]`, where
`m = mean(P4_x) - mean(P1_x)` and `S1`, `S4` are positive ordered separations.
Also expose crossed `e1 = P4_left - P1_right` and
`e2 = P4_right - P1_left` as diagnostics or an equivalent basis.

Required checks, before any new fit:

- Verify `e1 = m - (S1 + S4)/2`, `e2 = m + (S1 + S4)/2`, and
  `S1 = e2 - e1 - S4` on synthetic and existing retained measurements.
- Recover exactly the existing `d = m/S1` and `rho4 = S4/S1` conventions.
- Demonstrate invariance of all differences under a common additive shift,
  and invariance of normalized channels under a common positive scale.
- Demonstrate rank three for independent horizontal differences and rank two
  for the normalized crossed-pair representation. Do not append redundant
  differences and regularize their covariance to claim additional information.
- Transform a correlated covariance between relative bases and verify the same
  Mahalanobis cost, including the chosen robust block loss.
- Preserve original frame/fixation support, source hashes, signs, validity gates
  and gap boundaries. Record ordering failures; do not silently relabel physical
  illuminators through a crossing. A minimum P1 separation remains required.

Acceptance: identities and covariance transformations pass, existing normalized
observations remain unchanged, and the observation schema identifies units,
ordering, basis and covariance convention. This is an implementation-consistency
audit, not a state-accuracy validation.

## Stage 2: assess the separation/scale assumption

Inspect training measurements and any available independent repeat recordings
to characterize `S1` and the shared separation magnitude. Record dependence on
nominal gaze, demand, recording and detection quality, recognizing that nominal
labels are assumptions rather than measured frame states. Do not infer that P1
width is an exclusive head-depth signal or invariant to gaze/accommodation.

Declare one of these outcomes before designing a relative-channel fit:

| Scale treatment | Consequence and required evidence |
|---|---|
| Known or independently calibrated | Three channels can leave a residual consistency direction for two states, conditional on the fixed model and local rank |
| Session-constant or weakly varying | Adds a testable nuisance assumption/prior; quantify sensitivity to violations and separate it from optical information |
| Freely varying per frame | Three channels serve gaze, accommodation and scale; do not claim an extra validation equation |

For any proposed relative model, examine whitened state/nuisance Jacobian rank
and state conditioning after nuisance projection. Declare the global scale gauge
and test that fixing it does not conceal an unidentifiable framewise nuisance.
If evidence does not support constraining scale, retain the normalized estimator
and the third dimension as a quality diagnostic. Do not force a three-channel
state estimator solely because three measurements are available.

## Stage 3: isolated nonlinear-gaze candidate

First assess model capacity without changing the observation basis. Add the
single shared term `q*(theta/15)**2` to normalized displacement, retaining the
existing ratio model, trained-demand interpolation constraints and exponent
policy. The extra basis column is `[t**2, 0]` for the two outputs. Its physical
gaze derivative is `2*q*theta/225`; a globally constant `q` adds no direct
accommodation derivative.

Use a new coefficient schema: append `q` to the legacy 14 slots and extend the
coefficient map by an independent scalar block. The constrained three-demand
family then has 13 free coefficients; the flexible four-knot family has 15.
The four-demand fit with the same constrained map also has 13. At `q=0`, the
forward model must reproduce the existing model exactly.

Declare the curvature penalty in physical output units and select its strength
using training evidence or a prespecified sensitivity study. Avoid a separate
curvature knot at an omitted demand. A later `q0 + q1*A**p` dependence or an
alternative polynomial basis requires evidence and a separate comparison.

Update the shared forward model, analytic derivatives, coefficient profile,
initial-function prior, inverse Jacobians and exported coefficient metadata
consistently. Exact coefficient profiling remains available for this added
linear column at fixed states. Verify dense-profile agreement, derivative and
adjoint consistency, omitted-knot constraints, and physical inverse checks.
These are mathematical requirements for the numerical change, not a request
to recreate a permanent repository test directory.

## Stage 4: optional prior from previous fitted gaze

Support initialization separately from an explicit prior. For compatible
training frames, previous fitted states can initialize a new objective; this
is a declared warm start, not a same-objective resume.

For the prior, compute and freeze one ordinary fitted gaze mean per training
fixation on exact shared support. Implement the mean-only term specified in
Estimation.md, with explicit `previous_mean_strength` and positive
`previous_mean_scale_deg` values (per-fixation scales only when justified).
Strength zero must reproduce the no-prior objective. The prior must contribute
consistently to residuals, gradients, Jacobian products, physical optimality and
objective accounting.

Required safeguards and comparisons:

- Verify source model hash, training fold, frame/fixation correspondence, units
  and finite means. Record the source and frozen prior values in the manifest.
- Exclude full-data models and withheld-capture predictions from the prior or
  initialization of a strict holdout fold. A training-only source model may
  initialize the corresponding fold; another fold is not automatically safe.
- Declare whether nominal anchors are retained or replaced, and their separate
  strengths. Reused estimates already depend on observations and prior anchors;
  their reuse is regularization, not an independent Gaussian measurement.
- Compare initialization alone with zero/weak prior settings. Freeze candidate
  settings before final evaluation; do not choose strength by matching held-out
  full-fit trajectories or by producing flatter gaze traces.
- Use realistic regularization scales; correlated frame counts and small fit
  residuals do not justify arbitrarily strong confidence in previous means.
- Keep held-out frame inversion anchor-free. A later temporal or fixation prior
  for deployment would be a separate explicitly evaluated inference method.

Any changed prior center, scale or strength changes objective identity and must
prevent an ordinary checkpoint resume. Do not update centers inside a fit or
overwrite the retained reference model.

## Stage 5: conditional three-channel candidate

Proceed only after the scale decision in Stage 2. Choose and document a compact
shared forward basis for `[m, S1, S4]`, with one shared gaze/accommodation state
and the declared nuisance variables. Predict crossed pairs by the exact basis
transformation. Do not fit unrelated per-pair gaze functions and force them to
agree afterward.

Estimate a full training-only 3-by-3 relative covariance; assess contamination
by actual motion and possible variation in localization uncertainty. Use
detector quality as calibrated uncertainty only when that relationship has
evidence. Keep equal total optical weight per fixation and explicitly specify
the robust block scale and any nuisance penalties. Compare predictive and state
diagnostics rather than raw objective values from different residual dimensions.

A direct fixed-basis model linear in shared coefficients can retain profiling
conditional on states/scale. A factorized model with an unknown separation
function multiplying unknown normalized functions generally cannot use the
existing all-coefficient QR profile unchanged. Derive the chosen solver before
implementation and test its rank, gauge and derivatives. Generalize hardcoded
two-channel shapes only through an explicit new problem/schema; regression-check
the old path. Extend manifests with relative basis, nuisance treatment, parameter
counts, precision, gauge, priors and active derivative conventions.

## Stage 6: controlled comparisons and reporting

The primary comparison reference is now the frozen holdout3 H0 model, as
requested for mean-gaze comparison. Its new all20 anchor-free inverse is generated
without refitting and remains separate from joint training states. Full fits are
secondary capacity/coverage controls; no model-derived reference is ground truth.

Use the current frozen results in
[full_calibration](../exp2/full_calibration/README.md) and
[reduced_calibration](../exp2/reduced_calibration/README.md) as baseline artifacts.
Do not alter their JSON/checkpoints or claim that they contain the new model.

For each declared model-capacity comparison, distinguish:

| Calibration support | Displacement family | Main comparison |
|---|---|---|
| 0.36, 2, 4 D | Constrained map | Existing reduced protocol |
| All four demands | The same constrained map | Effect of additional training demand within that family |
| All four demands | Flexible four-knot map | Effect of additional coefficient freedom on the same data |

Assess curvature, previous-mean priors and relative-scale treatment in separate
controlled comparisons rather than changing all three at once. Keep declared
preprocessing, exponent/noise/prior construction policies and convergence rules
consistent; report actual fold-specific values. Do not initialize a held-out
candidate from full-fit states. Training on the fourth demand makes it in-sample
for that fit; use separate recordings or independent references for a fair
out-of-sample accuracy comparison.

Retain physical stationarity, step, cost and robust-weight stopping criteria;
extend them to nuisance states if introduced. Report nonconvergence honestly.
Before evaluation, freeze candidate coefficients, prior sources/settings and
prediction provenance. Use exact original-frame joins. Report branch ambiguity,
conditioning, bounds/extrapolation, prior sensitivity, and data/prior objective
components along with estimated states. A model-derived reference remains a
comparison trajectory, not ground truth.

Pair agreement is an independent consistency test only to the extent that
accommodation/nuisance information was obtained independently of that agreement.
Repeated recordings assess repeatability; target-mean checks assess nominal
assumptions. Neither alone establishes physiological accuracy. Extra relative
residuals are informative only after accounting for nuisance variables and
model uncertainty.

## Deliverables and cleanup

Implementation deliverables should retain the maintained code,
versioned model/checkpoint/provenance artifacts, concise audit summaries,
comparison CSVs and plots, and updated documentation. Run focused temporary
numerical harnesses outside the repository and remove them, transient logs and
bytecode caches after verification. Preserve raw detections, selected intervals,
central full-calibration references and completed baseline comparisons.

Implementation now lives in [relative_calibration](../exp2/relative_calibration/README.md).
Stages 1, 3 and 4 have optional code and focused mathematical verification; Stage 2
selects the normalized estimator with relative quality diagnostics, so the condition
for Stage 5 is not met. The Stage 6 runner writes each predeclared fit and its actual convergence to new
artifact roots, without altering the preserved baseline artifacts. Legacy defaults
remain unchanged; new objectives and artifact roots are explicit.
