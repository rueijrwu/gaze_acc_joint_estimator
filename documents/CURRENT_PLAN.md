# Can full measurement information improve gaze and accommodation estimation?

Status: proposed evaluation plan. The experiments below have not yet been run.
Existing calibration results are summarized in
[relative_calibration](../exp2/relative_calibration/README.md).

## Objective

First use the current results to identify what limits estimation, then test
whether the extra relative measurement removes that limitation. Useful progress
can be made with the existing data before collecting more.

The goal is improved gaze and accommodation estimation, not simply closer
agreement with the holdout-3 model or a smaller combined training objective.

## What the current results already suggest

- Adding the fourth demand within the constrained family changes mean gaze
  relatively little: **0.047 degrees RMS** relative to holdout-3 on the common
  3 D frames. This measures a change in estimates, not an improvement in accuracy.
- Changing model flexibility produces much larger changes. Model assumptions
  and parameter coupling therefore deserve attention beyond simply adding
  calibration data.
- Curvature coefficients are similar across fits, approximately
  **0.017–0.020 in normalized displacement units**. This is a repeatable feature
  within this dataset worth investigating, although it could reflect shared
  calibration bias rather than independent evidence of the correct response.

The common 3 D evaluation frames were withheld from the holdout-3 calibrations
but included in the full calibrations. The existing comparisons do not establish
superior out-of-sample accuracy for either family.

## 1. Diagnose the current solutions using separate metrics

For H0, FC0, and the curvature/flexible variants, report:

- Optical residuals separately from gaze anchors, accommodation anchors, and
  other penalties.
- Fixation-mean gaze relative to nominal targets, labeled **target consistency**.
- Within-fixation variation, sensitivity to measurement noise, and
  gaze–accommodation coupling.
- Sensitivity to anchor strength and coefficient initialization.

Also deliberately move a fitted gaze mean, then refit the other parameters.
If substantially different states explain the optical data almost equally well,
this identifies a weakly determined direction. Profile the optical fit and
penalty contributions separately so that regularization is not mistaken for
measurement information. Objective profiles are useful for diagnosing this
type of ambiguity; see the
[profile-likelihood methodology](https://pubmed.ncbi.nlm.nih.gov/19505944/).

**Question answered:** Are the existing differences supported by measurements,
or mainly selected by assumptions?

## 2. Test what useful signal exists in the discarded separation magnitude

Compare the current normalized measurements

$$
\mathbf y=[d,\rho_4]^\top
$$

with the independent relative measurements

$$
\mathbf z=[m,S_1,S_4]^\top,
\qquad d=m/S_1,\quad \rho_4=S_4/S_1.
$$

Here, $m=\operatorname{mean}(P4_x)-\operatorname{mean}(P1_x)$, and $S_1$ and
$S_4$ are the ordered horizontal separations. Absolute image position remains
excluded from the state measurements.

Examine whether $S_1$ contains reproducible structure associated with gaze or
accommodation after accounting for recording, time, and detector variation.

Test scale explicitly: known scale, constrained varying scale, and freely
varying scale. The free-scale case is an essential control. An apparent gain
that depends entirely on a strong scale prior must be identified as such.
Known or constrained scale is a hypothesis requiring evidence, not an assumption
that the current data have already established.

**Question answered:** Does retaining $S_1$ supply usable state information,
and under which assumptions?

## 3. Separate additional channels from additional calibration conditions

Run a controlled comparison:

| Measurement inputs | Three calibration demands | Four calibration demands |
|---|---|---|
| Normalized $d,\rho_4$ | Baseline | Additional calibration data |
| Relative $m,S_1,S_4$ | Additional measurement information | Both additions |

Start with the same constrained gaze/accommodation model, without curvature or
previous-gaze priors. Hold shared exponent, noise-construction, and
regularization choices fixed where possible. Use consistent correlated-noise
models so improved weighting is not mistaken for an extra state constraint.
Record any unavoidable differences in parameter counts and scale assumptions.

Test curvature and previous-mean priors afterward as separate additions.

**Question answered:** Is any improvement due to retaining separation magnitude,
adding calibration conditions, changing model flexibility, or adding prior
constraints?

## 4. Evaluate information recovery, not agreement with H0

With existing data, use whole-fixation holdouts and training-only preprocessing
to assess target consistency and stability. Keep these results explicitly
conditional on nominal fixation assumptions. Randomly withholding neighboring
frames does not provide the same separation from the calibration data.

For a decisive test, obtain separate repeated recordings with randomized
gaze/demand order, controlled head-distance changes, and independent
gaze/accommodation references where available. Every candidate must face the
same withheld recordings. Accommodation demand alone cannot serve as measured
accommodation.

Separate the evaluation questions:

| Question | Metric and interpretation |
|---|---|
| Does the model explain the optical measurements? | Optical residuals, separate from penalties; small residuals alone do not establish state accuracy |
| Are estimated fixation means consistent with the protocol? | Mean gaze relative to nominal gaze, with nominal values treated as assumptions |
| Are estimates stable across recordings and nuisance changes? | Repeatability and sensitivity, reported separately from accuracy |
| Are the estimated states more accurate? | Errors against independent gaze/accommodation references on common validation recordings |
| How much does the new method change the existing result? | H0-relative deviations, retained as sensitivity diagnostics |

**Question answered:** Does the extra information improve estimates on data
that did not determine the calibration?

## 5. Declare success before examining final validation results

Call the full-information estimator better only if it delivers:

- Lower independently referenced gaze/accommodation error.
- Improvements larger than reference uncertainty and recording variability.
- Acceptable ambiguity, failure rates, and sensitivity to head position.
- Benefits that persist without forcing estimates toward H0 through a prior.

Declare the practical improvement and acceptable tradeoffs before examining
the final validation results. Report gaze and accommodation separately; do not
hide deterioration in one behind improvement in the other.

If independent state references are unavailable, the conclusions should remain
limited to target consistency, stability, and information conditional on the
model assumptions. Those results can guide the next experiment but do not
establish physiological accuracy.

## First deliverable

Begin with the **current-solution diagnosis in step 1**. It should identify which
uncertainty the third measurement needs to resolve, giving the full-information
experiment a specific, testable purpose.
