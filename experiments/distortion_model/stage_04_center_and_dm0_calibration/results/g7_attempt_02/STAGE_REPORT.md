# G7 attempt 02 — constrained full DM0 fit

**Status:** COMPLETE_UNCERTIFIED. **Decision:** PAUSE. **Reviewer:** root implementation/audit agent. **Reviewed:** 2026-10-10T03:55:39.015869Z. G8 is not authorized.

## Scope and declared physical bound

This was one fresh two-start campaign with 32 outer cycles per start, using all complete valid frames from the twenty reviewed intervals, the full ten-coordinate covariance, equal full-exposure weighting, the existing soft full-mean anchors, zero temporal penalty, and the same DM0 objective. It initialized from the reviewed G6 attempt 02 common and reproducible perturbed states. The unbounded G7 attempt 01 was preserved and not resumed.

The user explicitly authorized a **one-degree horizontal accommodation-induced apparent-gaze bound** without requiring a new optical export. At normalized (g=1), let (H_x=D_x+\mu_{4,x}-\mu_{1,x}). The sufficient continuous constraints are (H_{x,\theta}\ge s_{min}) and (|H_{x,A}|\le s_{min}/4), where (s_{min}=10.0270022259) reference px/degree, half the outward-interval lower slope from G6. Interval constraints cover visual theta [-20,20] degrees and A [0,6] D, extending the derivative bound across the current state domain. Over A=0 to 4 D, they imply at most a one-degree inverse-equivalent horizontal displacement when the monotone inverse lies within the declared theta domain.

This is a user-authorized model constraint, not a measured optical calibration, not a bound on individual theta or fixation-mean error, and not theta-label clamping or a new loss term. The bound is on the complete horizontal centroid law including optical means.

## Numerical results

The selected perturbed start reached J=1925.601338743 from the common G6-compatible initialization objective 2134.091665950 (9.77% lower). Components are point 1880.990476302, theta anchor 43.004634837, A anchor 1.581827941, regularization 0.024399663, temporal 0. The common start ended at J=1925.827634534. Despite the lower objective, neither start meets the declared numerical certificate.

| Start | J | State projected gradient∞ | Global KKT residual∞ | KKT complementarity∞ | Active bound rows | Rank-2 states | Profile curvature |
|---|---:|---:|---:|---:|---:|---:|---|
| Common | 1925.827635 | 3.6867e−3 | 255.1908 | 9.56e−10 | 3 | 89,175 / 89,175 | Not evaluated; two-step constraint-curvature agreement .4476 > .01 |
| Perturbed, selected | 1925.601339 | 1.2024e−9 | 2.8122e−4 | 3.12e−12 | 3 | 89,175 / 89,175 | Not evaluated; two-step constraint-curvature agreement .4508 > .01 |

The strict stationarity threshold is 1e−6. The selected KKT residual independently fails that threshold. The selected interval constraints are feasible: 5,760 inequalities, minimum normalized slack 3.71e−14, three active within 1e−7. P1/P4 optical domains are valid. An independent local-state Hessian census found zero negative eigenvalues among 169,935 free eigenvalues; minimum 0.0012189. This does not establish full coupled/profile curvature. The certificate’s curvature-stability check failed, so profile curvature was not evaluated; do not interpret that flag as evidence of negative local state curvature or global rank deficiency.

At 81 public-optics theta samples, the selected model's maximum absolute Hx(A=4)−Hx(A=0) was 10.02635 px, within the 10.02700 px envelope. This dense grid is a numerical cross-check; the continuous statement comes from the interval inequalities.

Both starts accepted 359/390 and 358/390 history events. Accepted joint-step median/max damping was 15/17 for common and 14/17 for perturbed. Maximum per-frame joint trust caps were 3.655 and 3.414. Rejections counted 4,725 and 5,804 infeasible one-degree-bound line-search proposals, plus 30 nonpositive shared proposal-curvature events per start. These history counts describe numerical proposals, not evidence about physiological law.

The population was 89,175 valid frames and 10,915 unavailable scheduled rows out of 100,090. Equal-exposure native relative-coordinate RMS is 4.678851 px, versus 4.560659 px at G6. The fixation-mean theta residual RMS against nominal labels is 0.9274° (versus 2.31665° for unbounded attempt 01); this is agreement with labels, not proof of gaze accuracy. The compact progress schedule has 300 slots: 270 retained-valid and individually certified, 30 unavailable, and zero unresolved; final progress RMS is 29.3673 px. These individual inversions do not certify the shared calibration fit and are not a G8 score.

## Reproduction commands and independent mechanical audit

The campaign command was `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python experiments/distortion_model/stage_04_center_and_dm0_calibration/scripts/run_joint_bounded.py --output experiments/distortion_model/stage_04_center_and_dm0_calibration/results/g7_attempt_02`. The independent saved-result audit was run with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python experiments/distortion_model/stage_04_center_and_dm0_calibration/scripts/audit_g7_bounded.py --attempt experiments/distortion_model/stage_04_center_and_dm0_calibration/results/g7_attempt_02`. No automated test suite ran.


Independent NumPy public-optics predictions and direct (R^{-1}) point-loss/anchor reconstruction reproduce the saved objective within 4.55e−13, predictions within 3.98e−13 px, (g) within 8.88e−16, and fixation means within 2.49e−14. The complete exposure schedule and compact missing slots were retained. G5/G6 ancestor hashes and all archived source-snapshot hashes match provenance. The two live fitting files were restored byte-for-byte from the source snapshot used by this run; live fitting code and attempt provenance now agree.

The runtime was 782.62 seconds, including compact inference and reporting. CPU maximum RSS was 974,220 KiB. GPU free memory and pool use are final snapshots, not peak measurements. The backend was CuPy 14.2.0 on a Tesla P100-PCIE-16GB; the same-equation CPU/GPU empirical preflight passed on 40 actual rows. No automated test suite was added or run.

The source snapshot records a launch-time version before two unrelated live numerical refinements were coordinated. The actual attempt's archived source files and provenance remain authoritative for this result; after completion, the live bounded runner and constraint module were restored to those exact archived bytes. No fit rerun occurred.

## Decision and next action

**PAUSE.** The physical bound and objective bookkeeping pass, but strict global KKT stationarity fails and the active-constraint curvature stability check prevents profile curvature evaluation. Preserve the selected lower-J snapshot as uncertified; keep fit_complete, fit_certified, crosscheck_complete, and comparison_complete false. Do not launch G8.

One next numerical question: why does the constrained interval formulation have unstable active-constraint Hessian near the boundary, and can a smooth equivalent certificate preserve the same physical bound and unchanged objective? No model expansion or fitting continuation is authorized before resolving it.

- [Full summary](summary.json), [checkpoint](checkpoint.json), [fit arrays](fitted.npz)
- [Independent mechanical reconstruction and census](mechanical_audit.json)
- [Source snapshot and provenance](provenance.json)
- [Console log](console.log)
- [One-degree preflight evidence](../preflight_1degree/preflight.json)
