# Results and interpretation

The executable workflow separates optical information, target regularization and scale assumptions. Current recordings have no independent gaze/accommodation references or verified randomized repeats, so independent accuracy is **not evaluable without independent references**.

## Frozen-solution diagnosis

[Current-solution JSON](diagnostics/current_solutions/current_solution_diagnostic.json) and [fixation diagnostics](diagnostics/current_solutions/fixation_diagnostics.csv) reconstruct the original fitted objectives and report nominal consistency, within-fixation spread, optical residuals and conditional fixed-coefficient geometry. All seven objective reconstructions differ by at most `1e-15`; all fourteen frozen model/checkpoint hashes remained unchanged.

Optical costs are approximately `0.0018–0.0024`, whereas accommodation-anchor contributions are approximately `0.47–0.55`. This shows how the declared objective allocates cost. It does not alone identify a weak optical direction or rank solutions with different training cohorts/capacities. Nominal demand is an anchor target rather than a measured accommodation reference.

## Scale structure

[Scale structure](diagnostics/scale_structure/scale_structure.json) preserves the retained pixel basis and recording/fixation/frame identities. The existing 77,756-frame cohort has small within-fixation S1 variation, but gaze/demand/time/capture are not independently randomized. Small S1 variation therefore does not establish known or state-invariant scale. Consecutive-frame covariance contains detector variation and motion.

[Scale controls](diagnostics/scale_structure/scale_controls.json) test correlated covariance and the free-scale Schur complement. Free framewise scale analytically reduces to the matched normalized two-channel Mahalanobis problem, including the monotone robust-block optimum. It supplies no additional residual equation. Fixed or finite scale uncertainty are conditional sensitivity hypotheses.

## Reoptimized local mean profiles

The predeclared probes use the demand-2 center fixation 17, supported by both H0 and FC0. Requested shifts are ±0.5 degrees from each fitted mean; plots and tables use achieved means. Other training states and coefficients are reoptimized, and original constrained KKT stationarity is checked after removing only the selected mean multiplier. Incomplete points and continuation phases remain visible.

For the completed H0 negative probe, the achieved shift was approximately `-0.393447 deg`. The original objective increased by `0.0211989`, comprising optical increase `0.000130153` and aggregate anchor increase `0.0210372`. Its original constrained projected optimality was `3.42e-6`. The small optical change relative to anchor change describes this local regularized response; it is not a global identifiability certificate or confidence interval. Maintained summaries further separate gaze/accommodation anchors and temporal penalties.

## Matched internal experiments

[Predeclared matrix](matrix_protocol.json) fixes the common training-only policy and whole-fixation holdout 17. N3/N4 use normalized observations; R3/R4 add logS1 under hypothetical fixed scale. The same normalized coefficient map has 12 free coefficients in both demand arms. Full-relative models add three log-shape coefficients and a separately accounted training-only function prior. Two predeclared finite-uncertainty sensitivities retain that prior while inflating only the third covariance diagonal.

Generated matrix reports retain convergence, coverage, ambiguity/stationarity/bound flags, nominal mean consistency, within-fixation variability and conditional geometry on identical withheld frame support. Cross-scale costs are not likelihood comparisons and never select a scale assumption. Whole-fixation internal tests are not independent physiological accuracy validation.

The normalized cells have converged under the unchanged stopping rules:

| Cell | Training demands | Observations | Runtime (s) | Function evaluations | Accepted steps |
|---|---:|---|---:|---:|---:|
| [N3](matched/N3/model.json) | 3 | normalized two-channel | 583.67 | 562 | 483 |
| [N4](matched/N4/model.json) | 4 | normalized two-channel | 293.22 | 131 | 74 |

These runtimes describe the recorded solver trajectories. They do not rank calibration accuracy; held-out inference and the remaining conditional-scale cells are reported separately after completion.

## External validation

The maintained validator requires independently acquired state references, reference uncertainties/alignment, randomized repeated recordings, immutable source/cohort identities and predeclared success thresholds. Nominal labels, old fitted states and same-image inverses are excluded as independent references. Actual physical acquisition remains outstanding; missing references produce an explicit not-evaluable result.
