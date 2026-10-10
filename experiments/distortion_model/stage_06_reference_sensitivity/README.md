# Stage 6 — Matched operational-reference sensitivity

**Status: STOP_UNCERTIFIED_ADJACENT.** Execution and audits are complete. The current-reference control passes G7 and G8 with limits. The adjacent-reference calibration is uncertified, so its full held-point inference was not run. `comparison_complete=false`; there is no optical ranking and no next optical stage is justified.

## Reviewed result

The predeclared protocol in [`PLAN.md`](PLAN.md) compares P4 `omega4=-10.016974132845107°` with `omega4=-5.0148989655383795°`, rebuilding each reference-specific baseline while preserving shared controls and the user-authorized one-degree horizontal centroid-response bound. This tests sensitivity to operational reference conventions, not a physical optical zero.

The root's [scientific review](results/attempt_03/SCIENTIFIC_REVIEW.md) and [calibration review](results/attempt_03/CALIBRATION_REVIEW.md) determine the gate. Control calibration is certified (`J=1925.6013386760935`; state/global/complementarity residuals `3.92058e-12`, `6.36353e-8`, `6.88043e-11`). Its frozen G8 cross-check completes on all 100,090 scheduled frames and 300,270 slots, with 267,565 certified retained-input inversions, 267,525 scored slots, 32,705 unavailable retained inputs, zero unresolved or ambiguous slots, and 89,175 complete triples across all 20 exposures. Equal-exposure `E=12.408526 px`, `Gtheta=0.477866 deg`, `GA=1.101718 D` are internal consistency measures, not ground-truth errors.

Adjacent calibration is `NO_GO_UNCERTIFIED` (`J=1925.8105580440379`; state/global/complementarity `4.13310e-5`, `0.551904`, `1.80740e-6`). Its [G8 roster](results/attempt_03/g8_adjacent/summary.json) records all 300,270 slots as unattempted; no inversion or score is claimed. The [comparison ledger](results/attempt_03/comparison.json) keeps paired metrics null and records no optical ranking. This is a numerical limitation of the adjacent fit, not evidence that the reference is optically worse.

The [boundary inspection](results/attempt_03/boundary_inspection/summary.json) froze saved parameters and performed no fitting or inference. It found a gaze-floor interval branch kink at `alpha4=0`; that observation does not certify a generalized optimum. The next numerical investigation is one-sided/generalized stationarity and curvature at this kink, with NNLS near-active rows separated from actual complementarity. Do not proceed to another optical mechanism on this evidence.

Original attempt 01 and separately declared attempts 02–03 remain preserved. No certification gate was relaxed or budget extended.

## Files

- [`docs/PROGRESS.md`](docs/PROGRESS.md): final gate and numerical follow-up.
- [`docs/RESULTS.md`](docs/RESULTS.md): reviewed calibration and control G8 evidence.
- [`docs/stages/06_REFERENCE_SENSITIVITY.md`](../../../docs/stages/06_REFERENCE_SENSITIVITY.md): cross-stage status.
- [`results/attempt_03/`](results/attempt_03/): aggregate checkpoint, control G8, adjacent unattempted roster, comparison ledger, and reviews.
- [`scripts/repair_rank.py`](scripts/repair_rank.py), [`scripts/continue_rank.py`](scripts/continue_rank.py), [`scripts/compare_saved.py`](scripts/compare_saved.py), and [`scripts/record_uninferred.py`](scripts/record_uninferred.py): preserved utilities used to prepare and report this campaign.
