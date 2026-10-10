# G7 attempt 04 — reviewed saved-point correction

**Status: COMPLETE_CERTIFIED_WITH_LIMIT / GO_WITH_LIMIT.** This record contains one approved, constraint-consistent local correction of attempt03's saved state. It is not a new fit campaign: no fit loop or compact inference ran. The attempt03 checkpoint remains unchanged.

## Scope and ancestry

Attempt04 starts from the selected attempt03 states and globals. The candidate is bitwise identical to the candidate accepted by the saved-point proposal probe in [g7_polish_repair_01](../g7_polish_repair_01/summary.json). The correction preserves the complete empirical objective, all 89,175 valid rows, covariance, anchors, priors, references, and user-authorized one-degree physical centroid bound. It uses the same smooth active-constraint Lagrangian derivative convention reviewed for the probe.

The proposal is a single constrained Newton correction, not a continuation loop. Maximum absolute scaled-global coordinate change is 6.34e-8 (coordinate p8); the maximum dimensionless normalized global step is 3.82e-7. Maximum frame-state changes are 1.70e-7 degrees and 6.28e-7 D. No new subset inference, model terms, or temporal penalty were introduced. The attempt04 `fit_certified` flag is true; `crosscheck_complete` and `comparison_complete` remain false. G8 requires separate authorization.

## Numerical result

| Quantity | Attempt04 |
|---|---:|
| Full population | 100,090 scheduled; 89,175 valid; 10,915 unavailable |
| Objective J | 1925.6013386759976 |
| Point / theta anchor / A anchor / prior | 1880.9906386064886 / 43.00444059194289 / 1.5818597253308582 / 0.02439975223521386 |
| State projected gradient | 1.32e-13 |
| Global scaled KKT residual | 4.24e-10 (threshold 1e-6) |
| Constraint complementarity | 3.15e-14 (threshold 1e-6) |
| Physical constraint | Feasible; minimum normalized slack 0; 3 active inequalities |
| State data rank | Rank 2 for all 89,175 valid rows |
| Free-state observed curvature | Positive; 0 negative among 169,935 free eigenvalues |
| Constrained observed profile curvature | Positive, rank 16; minimum normalized eigenvalue 4.94e-5 |
| Optical domain | P1 and P4 valid |
| Equal-exposure native relative-coordinate RMS | 4.678857454 px |

The unchanged attempt03 J was 1925.6013386760033; the candidate change is −5.9962e-12 by direct component evaluation and −5.6843e-12 by accumulated cost subtraction. Maximum predicted pixel movement is 6.19e-6 px. The equal-exposure native RMS changes by about 2.0e-8 px, so this correction claims no material empirical improvement.

## Independent verification

The adoption script recomputed the complete certificate and an independent NumPy objective/prediction reconstruction. A separate public-optics reconstruction in [independent_public_optics_audit.json](independent_public_optics_audit.json) evaluated all 89,175 valid rows: component differences from the saved objective are at most 2.28e-13, prediction difference is 3.41e-13 px, scale difference is 7.77e-16, and exposure-mean difference is 1.95e-14. All optical rows were valid, and all 82 files in attempt03 retained their original hashes.

The reviewed compact results are the existing attempt03 records, not a rerun for attempt04. Their 90 complete triples across all 20 exposures have E=41.8138 px, Gtheta=1.50113 degrees, and GA=1.31867 D. The two largest frame contributions account for 94.3568% of equal-exposure E squared and 91.8257% of Gtheta squared. These descriptive tails do not establish a cause and are not an empirical improvement claim.

Attempt04 provenance records the reporting-only hash change to the already-saved compact summarizer between probe and adoption, with both hashes; fitting, proposal, and certificate source hashes were pinned and matched. A later reporting-only guard change is listed in [reporting_source_history.json](reporting_source_history.json); no fit/proposal/certificate code changed. The original pending-review probe and attempts01–03 remain preserved.

## Files and execution

- [Summary and certificate](summary.json), [checkpoint](checkpoint.json), [model](model.json), [full arrays](fitted.npz), [verification](verification.json), and [independent public-optics audit](independent_public_optics_audit.json).
- [Saved-point proposal probe](../g7_polish_repair_01/STAGE_REPORT.md), its [original summary](../g7_polish_repair_01/summary.json), and [compact diagnostic report](../g7_polish_repair_01/empirical_diagnostics/REPORT.md).
- Reproduce the one-correction adoption from the preserved probe with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python experiments/distortion_model/stage_04_center_and_dm0_calibration/scripts/accept_g7_polish.py --output experiments/distortion_model/stage_04_center_and_dm0_calibration/results/g7_attempt_04` (use a fresh output directory).
- Reconstruct the independent public-optics audit with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python experiments/distortion_model/stage_04_center_and_dm0_calibration/scripts/audit_g7_polish.py --attempt experiments/distortion_model/stage_04_center_and_dm0_calibration/results/g7_attempt_04`.

Operational omega1/omega4 references remain fixed, omega1 continuous uncertainty remains unquantified, and G3 discrete alternatives/physical zeros remain unresolved. GO_WITH_LIMIT certifies this numerical fit under the declared model and bound; it does not establish optical adequacy, physical zeros, or physiological accuracy.
