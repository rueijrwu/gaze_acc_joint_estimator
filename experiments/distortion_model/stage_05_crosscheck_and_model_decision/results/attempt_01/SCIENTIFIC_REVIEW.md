# G8 scientific review — frozen G7 attempt04

**Decision: COMPLETE_WITH_LIMIT / GO_WITH_LIMIT.** Scientific disposition: `alignment_or_identifiability_unresolved`. The frozen DM0 model is numerically usable for this internal measurement analysis. Shared accommodation estimates remain dependent on the retained measurement subset. Physical gaze/accommodation accuracy has not been established.

## Completed work and integrity

The experiment preserves all 100,090 scheduled native frames and all 300,270 omission slots. CuPy float64 runs the same 50-start inverse on each retained subset, with vectorized derivatives, compacted active candidates and chunks of 8,192 rows. The recorded full experiment took 191.64 seconds, including its small all-three diagnostic. This timing applies to this dataset and device.

There are 267,565 certified inversions, including 41,016 at a state-domain bound; 267,525 have an available held point and are scored. The remaining 32,705 slots have unavailable retained input. Forty certified inversions have an unavailable held measurement and remain unscored. No attempted inversion is unresolved or ambiguous under the declared multistart/numerical policy. This policy does not prove exhaustive branch enumeration.

All prior G7 artifacts are unchanged. The [preflight](../preflight_01/summary.json) compares the accelerated solver with the preserved legacy solver on actual records, checks CPU/GPU agreement, and perturbs omitted coordinates and availability through the whole inverse/prediction pipeline. The [independent saved-result audit](independent_audit.json) verifies the full schedule, covariance marginals, ancestry, public optical predictions and independent metric aggregation. Held-pixel reconstruction differs by zero; retained point costs agree within 1.68e-10. No full repository automated test suite was run.

## What the metrics measure

For omitted point j, the measured relative coordinate is `q_j - mean(P1)`. Its expected coordinate is `g_subset * (D + F4_j - mu1)`. Their difference is the held-point pixel error. Thus comparison uses the model origin correction without requiring an absolute camera position or independently observed distortion center.

For each eligible frame, E² averages the three squared 2D held-point errors. Gtheta² and GA² average the three squared pairwise differences among the same frame's subset state estimates. Each primary metric averages these squared quantities within each exposure, gives the twenty exposures equal weight, and then takes a square root. They measure internal agreement, not error against nominal targets or physiological ground truth.

| Full-period primary metric | Result |
|---|---:|
| Held-point E | 12.408526 px |
| Same-frame Gtheta | 0.477866 deg |
| Same-frame GA | 1.101718 D |

The primary population has 89,175 complete eligible triples across all twenty exposures. The pooled scalar-coordinate RMS is 8.860733 px and has a different definition. The earlier attempt03 compact score used a different population and is not an empirical improvement baseline.

Pooled quantiles of the per-frame agreement magnitudes use frame weighting, separately from the equal-exposure RMS above:

| Per-frame magnitude | Median | p95 | p99 |
|---|---:|---:|---:|
| E (px) | 5.79513 | 23.2926 | 29.0583 |
| Gtheta (deg) | 0.167588 | 1.05171 | 1.33335 |
| GA (D) | 0.671900 | 2.31717 | 2.75792 |

Full definitions/counts are in [native_frame_distributions.json](native_frame_distributions.json).

## Evidence about accommodation and structured residuals

1. **Direct state-anchor dependence is small on the diagnostic under frozen globals.** Label-free all-three states differ from anchored calibration states by 0.040203 deg / 0.071319 D equal-exposure RMS on 100 matched frames. Subset omissions give much larger A differences: 0.706610, 1.04813 and 1.23879 D. This weakens the hypothesis that the calibration mean anchors directly account for subset disagreement on those timestamps. It does not establish anchor independence over all frames or evaluate how anchors influenced the fitted global coefficients. The 110-row schedule deliberately includes twelve known capture2 neighbors; its state-difference score is not a representative full-period estimate. [State comparison and schedule partitions](state_comparison.json).

   The descriptive partition outside those twelve neighbors has 88 matched timestamps across all twenty exposures: all-three versus calibration differs by 0.039926 deg / 0.071253 D; omission A differences are 0.375236, 0.520937 and 1.058980 D. Omission-1 gaze disagreement drops from 2.92557 to 0.549161 deg in this partition. This distinguishes the neighborhood's influence without trimming the primary full-period population.

2. **One retained pair loses substantial accommodation information.** At identical all-three states, omitting P4 index 2 leaves median 0.079187 of full conditional A information after allowing gaze. Its median absolute derivative-column cosine is 0.816371. Omissions 0 and 1 retain about 0.448686 and 0.498404. These are fractions in the declared covariance metric, not independently calibrated physiological precision or proof that information loss explains every residual. The marginal-information fractions are checked to remain in [0,1]. [Saved information arrays](diagnostic_information.npz).

3. **A disagreement persists away from endpoints and clipping.** Central 90% frames have equal-exposure GA=1.101624 D. No-bound frames have pooled GA=0.752604 D, E=5.84935 px and Gtheta=0.185967 deg. Some exposures are absent from that stratum, so an all-twenty equal-exposure score is unavailable. The main denominator is unchanged. The largest individual full-period GA² contributor accounts for approximately 0.0322%; two endpoint outliers cannot explain this distributed disagreement. [Strata and independently ranked contributors](metrics.json).

4. **Omission 1 has a separate position-error signature.** Its vector-error p95 is 37.2896 px, versus 11.2517 and 11.18 px for omissions 0 and 2. Its x/y RMS is 15.7093/8.30462 px. Signed errors differ across recorded conditions and contiguous time blocks. It retains about half of full conditional A information in the diagnostic, so its large residuals cannot simply be assigned the same conditioning explanation as omission 2. [Joint point/exposure/block diagnostics](point_exposure_diagnostics.json).

5. **The reviewed capture2 anomaly persists in neighboring raw geometry.** Rows 6794–6805 are all available and maintain the same P4 triangle orientation. Their measured signed areas are about 94,000 px² versus approximately 42,000–43,000 px² in the anchored model prediction. This discrepancy and large held-point errors extend across the original exposure boundary. There are zero orientation conflicts among all 89,175 complete measurements. These findings provide no evidence of an isolated orientation flip; they do not exclude detector bias, correspondence errors preserving orientation, or shared optical/model error. Actual image-level correspondence was not independently established. [Native neighbor coordinates](neighbor_case.json), [geometry plot](neighbor_geometry.png).

## One proposed follow-up

**Operational-reference sensitivity**, using the previously documented adjacent P4 reference alternative: nominal -5 deg, `omega4=-5.0148989655383795 deg`, versus the current `omega4=-10.016974132845107 deg`. The P1 reference remains the existing convention. G3 selected an endpoint operational reference; its relation to physical symmetry and camera/source rotation is unresolved. Point-specific signed residuals and their condition dependence make this a discriminating next check. The current evidence does not select a new radial term or accommodation law.

Predeclare one matched comparison through the affected G4–G8 stages. Reinitialize/refit the baseline, keystone, center and frame states consistently for each reference convention, with the same full population, covariance, anchor/prior policies, domains, one-degree centroid-response bound and declared starts. Changing an angular zero changes the constrained optical model unless its entire baseline/operator is consistently re-expressed; editing only omega4 in the frozen checkpoint would be invalid.

Compare certified models on exact common frame/point identities and retain all-scheduled coverage. Report the signed omission-1 x/y pattern by exposure/block, E/Gtheta/GA and conditional A information separately. Improvement confined to the known neighborhood, new bound compression, loss of an exposure, or persistently weak A information leaves the mechanism unresolved. A broad, reproducible reduction in the structured residual with useful state information would support reference sensitivity within these conventions, not physiological correctness. No fixed RMS ceiling or combined mixed-unit score decides this comparison.

This follow-up is proposed; it was not executed as part of frozen G8.

## Remaining assumptions and recovery

Physical optical zeros/camera alignment, common P1/P4 axial scale transfer, independently calibrated localization covariance and absolute accommodation calibration remain unresolved. Demand and capture are confounded. Derived scale/state/error associations share measurements and are descriptive. The one-degree bound limits accommodation-induced centroid apparent gaze, not actual gaze error or Gtheta.

The last usable calibrated model remains G7 [attempt04](../../../stage_04_center_and_dm0_calibration/results/g7_attempt_04/checkpoint.json). G8 [checkpoint](checkpoint.json) sets `fit_complete=true`, `fit_certified=true`, `crosscheck_complete=true`, and `diagnostic_three_way_complete=true`. `comparison_complete=false` because a fresh cross-model comparison has not occurred. Saved full-schedule states/errors/statuses, raw inputs, diagnostic schedule, source snapshots and parent hashes support reconstruction without repeating inference. [slot_context.npz](slot_context.npz) explicitly records frozen-model calibration status and separate calibration-frame availability for every omission identity.
