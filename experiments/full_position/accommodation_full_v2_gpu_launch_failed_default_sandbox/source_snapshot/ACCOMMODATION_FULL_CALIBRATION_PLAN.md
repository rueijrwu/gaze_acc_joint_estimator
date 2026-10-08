# Full calibration: compare accommodation models with the same fitting algorithm

**Runner:** [full_position/accommodation_full.py](full_position/accommodation_full.py)  
**Branch:** `exp5_full`  
**Revision:** 2026-10-07, following the user's clarification.  
**Execution status:** this plan describes the full-calibration comparison; editing the runner is not evidence that the real-data fits have run.

## 1. Required experiment

**Keep the existing algorithm. Change the accommodation response model. Run a fresh full calibration for every model. Compare the resulting calibrations using cross-agreement.**

Reuse `accommodation_study._fit_task`, which constructs the requested `PowerResponseModel` and calls the existing `calibrate.fit`. Reuse its existing same-frame P4 cross-check, numerical certification, covariance handling and scorecards. Do not write another optimizer, change its objective, or replace this task with new numerical-method development.

**Reusing the algorithm does not mean reusing a previously fitted model.** All 27 coefficients and all framewise gaze/accommodation states must be re-estimated separately for each response law on the complete calibration data. Do not copy coefficients from the old log fit, reinterpret coefficients under a different exponent, freeze old latent trajectories, or merely apply historical fold models to more frames.

The requested experiment has four fresh candidate calibrations, not 36 grouped fits:

| Candidate | Fixed exponent | Accommodation basis, with a = A / (1 D) |
|---|---:|---|
| `ar27_log` | 0 | `log(1+a)` |
| `ar27_sqrt` | 0.5 | `2*(sqrt(1+a)-1)` |
| `ar27_linear` | 1 | `a` |
| `ar27_quadratic` | 2 | `a+a*a/2` |

The last three are the existing shifted-power alternatives, not newly fitted per-point exponents. Each candidate has 27 global coefficients and two unknown states per valid frame. The direct linear-accommodation term in `Dx` remains unchanged. See [the response theory](ACCOMMODATION_RESPONSE_THEORY.md) for the basis definitions.

## 2. Data and controls held identical

Use all 20 reviewed fixation conditions from captures 1-4 together: horizontal targets -10, -5, 0, 5 and 10 degrees in every capture. No capture, fixation or gaze condition is excluded from calibration. The nominal vertical target is zero; image-y measurements remain intact and no vertical-gaze state is added.

`training_data(..., count=0)` uses all valid frames in the existing central-80% reviewed windows. Retain the existing window/validity policy: "full" means every eligible row in those declared windows, not a return to the prior 48-row-per-fixation screen. The agreement schedule includes original rows before validity filtering, preserving invalid and unavailable outcomes.

Keep unchanged across candidates:

- All three P1 references and all three P4 x/y measurements in calibration, their correspondence, and the square-root P1 triangle-area normalization.
- The existing joint optimizer, two nominal/perturbed starts, numerical certification and continuation policy, seed, and iteration budget.
- Soft fixation-mean anchor scales 0.10 degree / 0.25 D, column-normalized prior strength 0.001, zero temporal penalty, and numerical bounds [-20,20] degrees / [0,6] D.
- One fresh log27 weighting pilot, reference state and coordinate/residual covariance constructed from the complete calibration population and shared by all four laws.

The weighting pilot is not the fitted log candidate. Even `ar27_log` must receive its own fresh full joint fit. Candidate-specific coefficient initializers and column scales are recomputed by the existing algorithm for that law's basis; equal numerical prior strength is not a guarantee of identical function-space regularization.

Each frame retains its own state `(theta_x_i, A_i)`. Gaze and accommodation may vary during fixation. Nominal demand/target labels are finite mean-anchor penalties, not instantaneous truth, allowed-motion limits, or required RMS values. Do not flatten trajectories or strengthen anchors simply to reduce nominal-target RMS.

## 3. Execution with the existing implementation

1. Read the current branch and record source/data hashes. Run the applicable existing tests plus the wrapper's population/comparison checks. These verify integration; they are not a requirement to invent a new fitting algorithm.
2. Prepare one full-data input and one immutable agreement schedule, shared by all candidates. The configuration records all calibration and internal-agreement group IDs, `fresh_calibration_per_law=true`, and `reused_previous_fitted_models=false`.
3. Call the unchanged `accommodation_study._fit_task` once per candidate in a new output directory. Each call constructs a fresh model and jointly refits coefficients and framewise states. The wrapper does not load any historical candidate model to initialize or replace these fits.
4. Once a candidate fit is certified, freeze that newly fitted coefficient set for the existing three-way cross-check. Retain failed calibrations/checkpoints and all their scheduled slots; do not manufacture scores or borrow another model's fit.
5. Generate the individual and exact-common-frame comparison tables below. No nested selector or condition-held-out study is needed for this task.

Use serial execution by default, or up to four concurrent model fits when memory permits. Reducing worker count does not change the calibration algorithm or data. Do not silently reduce calibration rows because a full fit is larger than an old fold fit. Any demonstrated runtime defect should be fixed narrowly; do not turn this plan into a solver redesign.

From the repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m full_position.accommodation_full \
  --output experiments/full_position/accommodation_full_v2 \
  --workers 1 \
  --agreement-per-fixation 0
```

The output directory must be new. Every invocation recalibrates all four candidates; there is no saved-fit reuse mode. `--agreement-per-fixation 0` checks all original core rows. A positive value samples only the internal checks for a separately labeled diagnostic run; it never reduces the calibration population. Do not compare two candidates using different diagnostic schedules.

## 4. Cross-agreement metrics

After each full calibration, keep its global coefficients fixed. Within frame i, temporarily exclude P4 point j, estimate the shared state from all three P1 points and the other two P4 x/y points, then reconstruct P4 j. Repeat for j=1,2,3. Use the existing retained-only covariance, 49-start inverse and scorer. No coefficient refit or condition exclusion happens during this check.

For the three reconstruction errors `e_ij` and three simultaneous subset states:

- `E_i^2 = (||e_i1||^2 + ||e_i2||^2 + ||e_i3||^2)/3` measures P4 cross-reconstruction disagreement.
- `G_theta,i^2` is the average of the three squared pairwise differences between subset gaze estimates.
- `G_A,i^2` is the analogous same-frame accommodation disagreement.

Compute these per frame first, then average their squares within each fixation/exposure and equally across the 20 exposures. RMS is simply the square root used to display that disagreement. The common scorecard retains signed axes, per-point errors, tails, worst point, coverage, bounds, ambiguity and support.

**There are no hard RMS accuracy thresholds, nominal-target error cutoffs, or fixation-flatness requirements.** Numerical certification, finite geometry and honest population accounting remain required. A state change between frames is not the same as disagreement among measurements of the same frame.

The calibrated coefficients already used these observations. Accordingly, the metrics describe **internal calibration consistency**, not independent validation or physiological accuracy. Full-fit residuals and nominal/temporal trajectory diagnostics are secondary; they cannot replace cross-agreement when comparing the response models.

## 5. Decide which response gives better agreement

The script writes three complementary views:

1. **Individual scorecards:** every law's coverage, E, G_theta and G_A on its available full-calibration rows. These show failures and missing measurements, not a ranking of differently filtered populations.
2. **Each law versus freshly fitted log:** use the existing `paired_summary` on exact common point/frame identities. Store candidate-minus-log squared changes, per-exposure contributions and exact memberships. A missing shared exposure makes the primary comparative delta unavailable rather than silently dropping that exposure.
3. **One shared cohort across all certified laws:** compare E, G_theta and G_A on exactly the same complete frames and the same expected exposures. The report orders laws by P4 cross-prediction loss and stores separate metric orderings. If the fresh log reference is uncertified, fewer than two candidates are certified, or an exposure disappears from this common cohort, report an incomplete comparison instead of a winner.

Use lower E/loss as the primary optical cross-check comparison, accompanied by the two state-agreement metrics. A law with lower E but higher G has a reported tradeoff, not an automatic rejection or an unqualified overall win. Do not add quantities in pixels, degrees and diopters into an arbitrary composite score. Log is a reference, not a presumed winner.

`summary.json` records `lowest_cross_prediction_model` only when that common comparison is available; it does not automatically select a deployment model. Companion flags identify increased G_theta, G_A or worst-point RMS versus log; inspect the complete axis/tail/bound/support scorecards as well. No law must improve every observation or satisfy an absolute error ceiling.

Keep the initial model comparison isolated: no y-mask change, shared-y covariance adjustment, new coefficient capacity, or retuning of anchors only for a favored exponent. Those would be different experiments, not a reason to postpone this full-calibration comparison.

## 6. Outputs and completion

The existing worker saves fresh model artifacts or failed checkpoints, calibration candidates, full framewise training states, trajectory diagnostics, three-way holdout records, inverse candidates and per-law scorecards under `fits/full_calibration/<candidate>/`. `splits/full_calibration/` holds the common data/covariance and schedule; its name is just the inherited directory structure, not a held-out split.

The wrapper saves `config.json`, `outcomes.json`, `summary.json`, `paired_memberships.jsonl.gz`, `RESULTS.md` and `completion.json`. Report four planned candidate outcomes, completed versus certified counts, actual calibration rows per condition, complete/scored versus scheduled counts, and runtime. Failed or interrupted runs are not successful experiments. Use a new directory for another run rather than overwriting old results.

Verify the lambda=0 adapter against the existing log evaluator, preserve dynamic-state and excluded-point noninterference tests, and check same-population aggregation, absent exposures, exponent/artifact identity and the four fresh worker calls. Run the repository's existing numerical tests without changing their tolerances. Do not claim real-data results based only on mocked orchestration tests.

**Scope precedence:** this plan replaces the previous full-calibration plan's emphasis on new staged numerical work. It also supersedes condition-held-out/nested-selection instructions in the older [accommodation-response plan](ACCOMMODATION_RESPONSE_PLAN.md) for this specific task. Those historical experiments remain unchanged. Existing algorithm modules, detectors, fitted historical models and captures 5/6 are not modified.

**Deliverable:** four fresh full calibrations using the same existing algorithm, followed by cross-agreement evidence showing which accommodation model fits the shared three-pair state more consistently and where the models trade off.
