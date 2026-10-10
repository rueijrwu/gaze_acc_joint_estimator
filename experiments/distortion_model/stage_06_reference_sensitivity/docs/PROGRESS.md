# Stage 6 progress

**Status: STOP_UNCERTIFIED_ADJACENT. Execution and audits are complete; paired comparison remains incomplete.** Control calibration and frozen G8 cross-check are reviewed `GO_WITH_LIMIT`. Adjacent calibration is `NO_GO_UNCERTIFIED`; its G8 roster is explicitly not run. There is no optical ranking and no next optical stage is justified.

| Candidate | G7 | G8 | Metrics / coverage |
|---|---|---|---|
| Control, `omega4=-10.016974132845107°` | Certified `GO_WITH_LIMIT`; `J=1925.6013386760935` | Complete and audited `GO_WITH_LIMIT` | 100,090 rows; 300,270 slots; 267,565 certified; 267,525 scored; 32,705 retained inputs unavailable; zero unresolved/ambiguous; 89,175 complete triples, 20/20 exposures |
| Adjacent, `omega4=-5.0148989655383795°` | `NO_GO_UNCERTIFIED`; `J=1925.8105580440379` | Not run; 300,270-slot roster preserved | No inversions or scores; paired metrics null |

Control G7 state/global/complementarity residuals are `3.92058e-12`, `6.36353e-8`, `6.88043e-11`. Adjacent residuals are `4.13310e-5`, `0.551904`, `1.80740e-6`, above the unchanged `1e-6` limits. Its feasibility, rank and curvature checks do not replace stationarity.

Control G8 equal-exposure results are `E=12.408526 px`, `Gtheta=0.477866 deg`, `GA=1.101718 D`; these describe internal consistency. The independent audit passes and confirms the frozen control parent and archived sources are unchanged. The adjacent omission roster remains unattempted by the certification rule, not because its score was poor. The paired comparison has not been completed.

## Evidence and next numerical question

See the [scientific review](../results/attempt_03/SCIENTIFIC_REVIEW.md), [calibration review](../results/attempt_03/CALIBRATION_REVIEW.md), [aggregate checkpoint](../results/attempt_03/checkpoint.json), [control G8 audit](../results/attempt_03/g8_control/independent_audit.json), [adjacent unattempted roster](../results/attempt_03/g8_adjacent/summary.json), and [comparison ledger](../results/attempt_03/comparison.json). The [boundary inspection](../results/attempt_03/boundary_inspection/summary.json) freezes all other globals and states, performs no optimization or inference, and identifies a gaze-floor derivative branch change around `alpha4=0`.

The next numerical investigation should evaluate one-sided/generalized stationarity and curvature at that interval kink, separating the broad NNLS near-active census from actual complementarity. This is the unresolved solver question; it is not proof of an adjacent-reference physical failure or a nonsmooth optimum. Do not launch another optical stage unless a matched comparison becomes numerically certifiable.
