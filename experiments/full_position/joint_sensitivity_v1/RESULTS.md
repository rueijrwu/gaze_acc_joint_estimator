# Phase 8.2 joint-training sensitivity results

The saved run is complete; this report records matched evaluation results and
calibration sensitivity. It does not select a deployment model or establish
physiological accuracy.

## Design and denominators

The study refit conditional 27- and 37-coefficient models on the nine frozen
Phase 8.1 outer folds, covering five gaze and four capture/demand holdouts. It
crossed these folds with eight predeclared joint-calibration settings: baseline,
weak/strong mean anchors, weak/strong coefficient prior, fourfold covariance,
and Q25/Q75 reference states. This produced 144 fold × setting × capacity
outcomes. The saved completion record contains 144 task outcomes, with 143
calibrations and evaluations accepted and no task exceptions. The one
uncertified task is `prior_weak/capture_1/conditional27`; that task contributes
no evaluation scores, and its full failure record remains in `summary.json`,
its checkpoint files, and the coverage tables below.

Each split-family/model population schedules 160 frames and 480 held-point
slots. The fresh baseline has 429 input-valid and scored slots and 143 complete
three-point frames. Seventeen input-invalid scheduled frames account for 51
unscored slots. All accepted sensitivity fits retain the same
evaluation population. In the aggregate capture/27 prior-weak results, the
failed fold leaves 117 otherwise-valid slots unavailable: 312 of 429 valid
slots score, in 104 complete frames. Summary metrics are equal-fixation RMS
unless stated otherwise. `E_px` is complete-frame P4 error in pixels; `G_theta_deg`
and `G_A_D` are within-frame state disagreement in degrees and diopters.

The plot at [paired_interior_vs_baseline.png](paired_interior_vs_baseline.png)
shows every nonbaseline setting against its freshly fit baseline. Each pair uses
its own exact shared complete-interior frame IDs; labels give the matched frame
count and usable calibration/evaluation folds. Unavailable values are left
blank. Full-population scores and boundary counts remain separate in
`summary.json` and are not inferred from this interior-only figure.

The per-task [calibration diagnostics table](calibration_diagnostics.csv) has
144 rows, including the unaccepted fit. It records selected-start status,
physical projected stationarity, inner stationarity, evaluation count, weighted
objective components, raw training point RMS, nominal-anchor discrepancy, and
training-state shifts. Objective component costs use setting-specific weights
and must not be compared across settings as if they shared a common scale.
Rebuild it with:

```bash
rtk proxy python experiments/full_position/joint_sensitivity_v1/export_diagnostics.py
```

## Full-population findings

| Split / capacity | Setting | Scored slots / valid slots | Complete frames | Bound slots | Equal-fixation RMS E / Gθ / G_A |
|---|---|---:|---:|---:|---:|
| Gaze / 27 | Baseline | 429 / 429 | 143 | 23 | 3.4207 px / 0.2694° / 0.3111 D |
| Gaze / 27 | Prior weak | 429 / 429 | 143 | 34 | 5.3139 px / 1.0551° / 0.2683 D |
| Gaze / 37 | Baseline | 429 / 429 | 143 | 79 | 6.7005 px / 0.5093° / 0.9290 D |
| Gaze / 37 | Anchor strong | 429 / 429 | 143 | 29 | 4.7637 px / 0.4397° / 0.5297 D |
| Gaze / 37 | Prior strong | 429 / 429 | 143 | 25 | 4.7012 px / 0.4285° / 0.4615 D |
| Capture / 27 | Baseline | 429 / 429 | 143 | 8 | 4.0144 px / 0.2806° / 0.3194 D |
| Capture / 27 | Anchor weak | 429 / 429 | 143 | 46 | 3.8232 px / 0.2723° / 0.3130 D |
| Capture / 27 | Prior weak | 312 / 429 | 104 | 3 | 3.4647 px / 0.3103° / 0.2975 D |
| Capture / 37 | Baseline | 429 / 429 | 143 | 38 | 4.3066 px / 0.3250° / 0.3917 D |
| Capture / 37 | Anchor strong | 429 / 429 | 143 | 30 | 4.2928 px / 0.2764° / 0.3665 D |
| Capture / 37 | Prior weak | 429 / 429 | 143 | 41 | 3.7595 px / 0.6978° / 0.5537 D |

These full-population comparisons preserve the actual 37-capacity prior-strong
and capture prior-weak error reductions. Gaze/37 prior strong has identical
full-population score coverage and fewer bound slots than baseline, alongside
lower E, Gθ, and G_A. Capture/37 prior weak also scores all 429 valid slots and
lowers E, while Gθ and G_A increase. These are different outcomes across split
families, not a single sensitivity direction.

Strong anchors are another positive full-population 37-capacity result: E, Gθ,
and G_A are all lower than baseline in both split families, with the same 429
scores and fewer bound slots (79 to 29 for gaze; 38 to 30 for capture). The
capture/37 matched-interior comparison also improves these metrics together.
This is a promising development finding, not a selected setting or a reason to
skip the remaining independent validation.

Capture/27 anchor weak has lower full-population E, Gθ, and G_A than its fresh
baseline, with bound slots increasing from 8 to 46. Its matched-interior gains
are smaller, and G_A is slightly higher there. Gaze/27 prior weak scores all
valid slots and lowers only G_A; E and gaze disagreement rise substantially.
The Q25/Q75 reference changes have small effects in these summaries. Covariance
4× and anchor strong are equivalent up to an overall objective factor four
under the current normalization and yield essentially the same saved results;
they are not independent evidence that covariance inflation helps.
Across all 18 outer-fold/capacity strong-anchor versus covariance-4× controls,
the maximum fitted-state difference was `0.00053691` degree and `0.00059436`
D; the maximum relative objective-cost ratio error against the predicted
factor four was `9.12e-10`.

## Exact shared-interior comparisons

The entries below compare the fresh baseline and sensitivity fit on identical
complete-interior frames. Values are equal-fixation RMS E / Gθ / G_A; `n` is the
number of shared frames. Baseline values vary by setting because each setting
has its own exact support intersection.

| Split / capacity | Setting | n | Fresh baseline | Sensitivity fit |
|---|---|---:|---:|---:|
| Gaze / 27 | Prior weak | 130 | 3.364 / 0.270 / 0.315 | 5.308 / 1.081 / 0.277 |
| Gaze / 37 | Prior strong | 106 | 4.475 / 0.419 / 0.388 | 4.420 / 0.441 / 0.421 |
| Gaze / 37 | Anchor strong | 107 | 4.664 / 0.412 / 0.375 | 4.571 / 0.409 / 0.375 |
| Gaze / 37 | Prior weak | 101 | 4.602 / 0.396 / 0.366 | 6.666 / 1.156 / 0.594 |
| Capture / 27 | Anchor weak | 118 | 3.611 / 0.294 / 0.329 | 3.577 / 0.288 / 0.333 |
| Capture / 27 | Prior weak | 103 | 3.500 / 0.282 / 0.348 | 3.498 / 0.311 / 0.300 |
| Capture / 37 | Anchor strong | 126 | 4.075 / 0.338 / 0.413 | 4.058 / 0.278 / 0.371 |
| Capture / 37 | Prior weak | 120 | 3.792 / 0.326 / 0.407 | 3.596 / 0.447 / 0.477 |

The 37-capacity settings show tradeoffs between point error and state
agreement. The gaze/37 prior-strong full-population reduction remains a small
E reduction on its 106 shared interior frames, with both disagreement metrics
slightly higher. Capture/37 prior weak retains lower E on 120 interior frames
but higher state disagreement. Gaze/37 prior weak is worse on all three
interior measures.

Do not compare capture/27 prior-weak's 3.4647 px full-population RMS as if it
had baseline coverage: 117 valid slots are unavailable. On its exact 103-frame
shared-interior support, E is virtually unchanged, while Gθ is higher and G_A
lower. The plot labels this setting `usable folds=3/4`.

## Training-state and grouped-refit diagnostics

Across accepted fits, maximum training-trajectory RMS shift from the frozen
baseline was at most `4.9824e-5` degree and `7.6093e-5` D for the joint
baseline refit. Mean-anchor weakening reached `1.3290` degree and `1.1319` D;
prior weakening reached `1.5310` degree and `2.1094` D. Q25/Q75 reference
changes shifted those trajectories by at most `0.013724` degree and `0.019295`
D. Thus the jointly fitted states respond much more to anchor/prior settings
than to this reference-state change, in contrast to the earlier fixed-state
coefficient sensitivity check. The maximum baseline training fixation
mean discrepancy from nominal anchors across accepted fits was `0.60109` degree
and `1.41959` D; nominal anchors are soft calibration terms, not measured
physiological truth.

The fixed-context grouped response-dispersion probe also changed with the
training refits. For prior weak, its descriptive RMS response dispersion was
7.2919 px versus 4.1680 px baseline for gaze/27 (5/5 fits), 21.1117 versus
5.8928 px for gaze/37 (5/5), and 11.2839 versus 4.1653 px for capture/37
(4/4). Capture/27 prior weak was 7.6690 px across only 3/4 accepted folds,
versus 7.2994 px across 4/4 for baseline; those denominators differ. This is
response variation across correlated grouped refits at a fixed state/context,
not total predictive uncertainty, a confidence interval, or held-out predictive
error. The local conditional covariance still excludes coefficient uncertainty
and model discrepancy.

## Interpretation and limits

No predeclared setting dominates across both split families, capacities, point
error, and state agreement. These results describe grouped development
sensitivity only; they do not choose priors, establish physiological accuracy,
or authorize a deployment model. The saved scalar inversion policy is retained
throughout this controlled study. Evaluation support, calibration status,
boundary behavior, and the conditional-uncertainty definition should accompany
any metric quoted from this run. Later P2/Phase 8.3 information ablations and
independent physiological/detector review remain separate work.

For the run configuration and hashes, see [`config.json`](config.json) and the
[verification record](verification.json). Current repository status and the
historical/current audit are in root [CURRENT_STATUS.md](../../../CURRENT_STATUS.md)
and [AUDIT_REPORT.md](../../../AUDIT_REPORT.md).
