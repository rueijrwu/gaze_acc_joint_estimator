# G7 repair01 — same-state derivative/certificate reassessment

**Status: REASSESSED_UNCERTIFIED. No optimization or refit was performed.** The reassessment loaded the selected perturbed states and globals from G7 attempt02, evaluated the repaired smooth active-constraint derivatives, and recomputed the certificate and curvature at that exact saved point.

## Findings

The states, globals, objective components and saved fit arrays are unchanged from attempt02. The objective reconstruction differences are exactly zero. The deterministic derivative regression passes. The physical one-degree centroid constraint is feasible with three active inequalities and minimum normalized slack 3.71e-14. The selected active branch is smooth; there are no nonsmooth active rows. The state projected gradient is 1.20e-9, and complementarity is 3.09e-12.

The repaired global KKT residual is 0.0035570, above the declared 1e-6 threshold. The local-state curvature census is positive: zero negative eigenvalues among 169,935 free eigenvalues, with minimum 124.709 under this reassessment's per-frame-Hessian normalization (observed Hessian divided by the positive exposure row weight). The constrained profile curvature is positive with rank 16. The local-eigenvalue magnitude is not numerically comparable with attempt02's weighted normalization; only the sign conclusion is directly comparable.

This resolves the earlier active-interval finite-difference branch instability at the same point, but it does not satisfy global stationarity. The residual justified one compatible continuation from the exact selected snapshot; it did not justify changing the objective, bound, references, template/prior origin, or population.

The repair differentiates the selected interval endpoint algebra automatically, retaining the original outward-rounded values for feasibility. The regression checks 22 directions on both signs of the saved near-zero center coefficient: maximum gradient error 5.36e-9 and Hessian error 1.72e-8. It reproduces the old multiplier-weighted H(15,15) artifacts of approximately -3.15e6 and -5.74e6; the smooth-branch value is exactly zero. At the sign tie, all three tested active rows are explicitly flagged as nonsmooth and cannot receive a smooth certificate. See [deterministic regression evidence](derivative_regression.json). Local-state and profile-curvature statuses are now independent; uncomputed profile curvature is `NOT_EVALUATED`.

## Reproduction and integrity

Command:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python experiments/distortion_model/stage_04_center_and_dm0_calibration/scripts/repair_g7.py --output experiments/distortion_model/stage_04_center_and_dm0_calibration/results/g7_repair_01
```

Runtime was 35.35 seconds. The derivative regression is an actual-data numerical audit, not an automated test suite. The repair provenance lists 144 files from attempt02, and all 144 SHA-256 values still match. Attempt02 remains unchanged. See [reassessment summary](summary.json), [provenance](provenance.json), and [console output](console.log).

## Gate

This same-state result is not a calibrated fit and does not authorize G8. The one bounded continuation is separately preserved in [G7 attempt03](../g7_attempt_03/STAGE_REPORT.md); it remains COMPLETE_UNCERTIFIED / PAUSE because its global KKT residual is still above tolerance.
