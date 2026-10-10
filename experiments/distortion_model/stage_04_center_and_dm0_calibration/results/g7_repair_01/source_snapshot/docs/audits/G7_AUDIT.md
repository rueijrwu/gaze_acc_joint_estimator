# G7 audit — constrained DM0 calibration

**Branch:** exp5_distortion_model  
**Inspected HEAD:** f48610d29df10521b199d3f8f13872875822ca9e  
**Status:** Independent read-only review; G7 remains COMPLETE_UNCERTIFIED / PAUSE. G8 is not authorized.

## Executive assessment

The selected G7 attempt 02 reproduces the objective and optics to numerical precision and satisfies the declared one-degree continuous-domain centroid constraint. The joint global calibration is **not certified**. Two failures are reported: scaled global KKT residual 2.8122e-4 versus 1e-6 and interval-constraint curvature finite-difference agreement 0.450766 versus 0.01. The source inspection and independent reconstruction identify a likely *branch-crossing finite-difference artifact* that must be resolved before either failure is interpreted as optical evidence. The compact held-P4 result (29.3673 px pooled coordinate RMS, 270 scored slots) is diagnostic, **not** the full G8 cross-agreement score.

## Findings

### A1 — Constraint derivative branch crossing (high confidence)

In `distortion_model/centroid_bound.py`, `CentroidBound.values` uses outward-rounded interval arithmetic and `min/max` endpoint selection. Its Jacobian uses centered differences with `step=1e-5`; the certificate then finite-differences this numerical Jacobian again with steps 1e-5 and 5e-6. The selected center polynomial has `sA_x=-0.0003131111846445346` reference px, corresponding to the scaled parameter `p[15]≈-1.64533e-6`. Both Hessian stencils cross its sign/endpoint-selection switch.

Independent constraint-only reconstruction of the multiplier-weighted Hessian (15,15) entry:
- step 1e-5: approximately -3.151e6
- step 5e-6: approximately -5.737e6
- step 1e-6: approximately -6.088e6
- step 5e-7: approximately -0.0061
- step 2.5e-7: approximately +0.0170

The 1e-5 versus 5e-6 matrix discrepancy reproduces 0.450766. Smaller same-branch steps give approximately 0.00421 **for the constraint-only calculation**, not a full constrained-Hessian certificate. Do not simply hard-code a smaller universal step. Implement and test active-branch analytical/automatic derivatives or a branch-aware generalized derivative. Retain outward interval bounds for feasibility.

### A2 — KKT must be recalculated after derivative repair

The constraint Jacobian is used in the QP, KKT multiplier solve and stationarity certificate. The near-switch derivative changes materially across step choices; the saved KKT residual is therefore not a robust stationarity assessment. Re-evaluate in order: constraint values, derivative validity, active set, multipliers/KKT, then constrained profile curvature. No unconditional extra outer iterations.

### A3 — Certificate conflates unavailable with failed local curvature

The exception handler in `BoundedJointDM0.certificate` sets `free_state_observed_curvature_positive=False` if a later constraint-Hessian stability check throws. The separate local-state census found zero negative eigenvalues among 169,935 free local eigenvalues (minimum 0.0012189). Record local curvature independently; represent uncomputed coupled/profile curvature as `NOT_EVALUATED`, not false evidence of negative curvature.

### A4 — Progress diagnostics are not final cross-agreement

The compact checker samples five equally spaced original scheduled rows per exposure, including both interval endpoints (40% of sampled rows). Transition contamination is known. The pooled coordinate RMS of 29.3673 px is not the full-population equal-exposure vector error and no aggregate `G_theta/G_A` is reported. Keep all slots, including unavailable rows, and decompose the existing compact records by omitted P4, x/y, exposure, interval position, state bounds and cross-subset agreement. Compare the same rows at saved G7 checkpoints. Do not trim endpoints to improve a headline score.

### A5 — Optical reference convention remains provisional

Both P1 and P4 operational symmetry zeros were selected at the sampled -10° endpoint: `omega1=-10.020904111586926°`, `omega4=-10.016974132845107°`. A minimum at the sampled endpoint does not locate a physical symmetry zero or prove parity assumptions of the keystone family. Their near-equality does not prove coincident axes. Retain independent offsets and evaluate fixed neighboring-reference sensitivity later, without introducing free framewise centers or changing multiple assumptions at once.

### A6 — Accommodation response remains provisional

The selected `M_slope=-0.0138514237 /D`; 8,415 frames have A at a bound. The slope differs materially from earlier conditional stages. Capture/demand confounding, reference dependence and unknown physical accommodation truth remain. The one-degree bound is a user-authorized constraint on accommodation-induced **horizontal centroid-equivalent shift**, not a one-degree gaze-accuracy guarantee.

### A7 — Implementation largely matches the relative optical model

The inspected `distortion_model/joint.py` uses two dynamic frame states, P1-only gaze-conditioned scale shared with P4, ten translation-free relative coordinate channels and their full raw covariance, two separate optical zeros, the center-separation D polynomial, full-fixation soft mean anchors, and no temporal penalty. The 19-global-parameter DM0 model is not the old empirical 27-parameter model. Saved independent NumPy reconstruction agrees with objective and predictions to numerical precision. This audit does not claim a full repository test-suite run.

## Next gate: targeted numerical repair, no optical expansion

1. Freeze the selected G7 attempt 02 and its provenance; do not overwrite or relabel it certified.
2. Build a small deterministic regression using the saved near-zero `p[15]`, actual active interval cells and multipliers. Compare active-branch derivatives with directional finite differences on each side and document ties.
3. Correct constraint derivatives and split certificate status fields; recompute KKT and constrained profile curvature at the **same saved state**.
4. Summarize existing compact held-P4 agreement, with matched point/axis/exposure/bound diagnostics and `G_theta,G_A`, explicitly labeling it a progress diagnostic.
5. Only if the repaired derivative certificate identifies an actionable optimization residual, authorize **one** compatible bounded continuation with unchanged objective, data, anchors, optical model and one-degree constraint.
6. Run full G8 only after a certified G7 checkpoint. Do not automatically promote DM1/DM2 or add accommodation exponents.

**Scientific disposition:** `numerically_unavailable` for the complete constrained calibration; `optical_model_unresolved` for physiological accommodation and independent gaze accuracy. A lower training objective is not proof of better three-pair agreement.

**Sources:** [G7 attempt 02 report](../../experiments/distortion_model/stage_04_center_and_dm0_calibration/results/g7_attempt_02/STAGE_REPORT.md), [constraint implementation](../../distortion_model/centroid_bound.py), [joint model](../../distortion_model/joint.py), [G7 saved fit](../../experiments/distortion_model/stage_04_center_and_dm0_calibration/results/g7_attempt_02/perturbed/fit.json), [stage resume pointer](../../experiments/distortion_model/stage_04_center_and_dm0_calibration/docs/PROGRESS.md).
