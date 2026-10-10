# G7 attempt 03 — one compatible bounded continuation

**Status: COMPLETE_UNCERTIFIED. Decision: PAUSE. G8 is not authorized.** Reviewed by root implementation/audit agent on 2026-10-10. This was the one continuation allowed by the repaired same-state audit. It started from G7 attempt 02's selected perturbed solution and preserved the G6 reference/template/prior origin. It used the same full objective and the user-authorized continuous one-degree horizontal accommodation-to-apparent-gaze bound. No new start or budget extension was run.

## Execution and objective

The continuation command was:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python experiments/distortion_model/stage_04_center_and_dm0_calibration/scripts/continue_g7_repaired.py --output experiments/distortion_model/stage_04_center_and_dm0_calibration/results/g7_attempt_03
```

It used all 89,175 complete valid rows from the twenty reviewed intervals; all 10,915 unavailable rows remain among 100,090 scheduled rows. The selected attempt 02 states and 19 globals are bitwise identical to `continued/initial.npz`; the warm-start solution SHA-256 is `6ceca207d5b00154ce7b6d06a22e9fae239f3e0bc31d7d280adbc6b7bf7fd72b`. All 144 input-file hashes captured by repair01 still match attempt02.

Attempt03 ended at J=1925.601338676003, versus 1925.6013387428595 at the warm start, a reduction of about 6.69e-8. The point loss is 1880.990637222, gaze anchor 43.004442178, accommodation anchor 1.581859524, regularization 0.024399752, and temporal penalty zero. Equal-exposure native relative-coordinate RMS is 4.678857434 px, versus 4.678851016 px at attempt02. The small objective change does not certify convergence.

## Numerical evidence

The declared budget was at most 8 outer updates plus 6 observed-curvature polish proposals. All 8 outer updates completed; history records **one** polish proposal (iteration 0), which was unaccepted. Do not report six polish iterations as performed. History records 62 accepted and 35 rejected update events, including 27 accepted joint outer updates. Across joint proposals, the saved rejection counts are 810 same-objective line-search failures and 5 nonpositive shared-proposal-curvature failures. The unaccepted polish event records 36 line-search failures and 5 nonpositive-curvature failures; it has no accepted outcome. Accepted joint damping had median 0 and maximum 13; maximum per-frame trust cap was 1.0.

The repaired active constraint derivative regression passes, the selected active branch is smooth, the physical bound is feasible (1,920 interval cells / 5,760 inequalities; minimum normalized slack approximately -1.8e-14), and complementarity is 1.89e-13. State projected gradient is 1.23e-11. **Global scaled KKT residual is 2.5103e-5, above the 1e-6 threshold**, so the fit remains uncertified. Observed constrained profile curvature is positive with rank 16; the local state census has zero negative eigenvalues among 169,935 free eigenvalues. The minimum local eigenvalue reported by this audit is 124.707 per-frame Hessian divided by its positive exposure row weight; compare its sign, not its magnitude, with attempt02's differently normalized census.

The compact progress diagnostic retains 300 slots: 270 scored and individually certified, 30 unavailable, none unresolved, 48 inferred states at bounds. Its coordinate RMS is 29.367297 px. It is not G8 and does not certify the shared calibration. The saved aggregate also reports complete triples, matched checkpoints, endpoints, and per-axis residuals; see [compact diagnostics](compact_diagnostics/REPORT.md). Those subset-agreement and residual summaries are descriptive, not accuracy claims.

## Independent reconstruction and provenance

The saved-result audit independently reconstructs the objective, optical predictions, physical constraint and state/profile curvature. It verifies that attempt03's initial states/globals exactly match attempt02's selected solution; every attempt02 file listed in repair01 provenance still matches; the G6/G5 parent hashes match; and the archived source snapshot matches every runtime source hash. Fitting source hashes still match runtime provenance. Live source-hash differences are the post-fit audit-only label adaptation needed to read the single `continued` start and the live Stage 4 status text. The current auditor and compact aggregator hashes are recorded in `mechanical_audit.json`. These reporting changes did not change fitting code or saved fit arrays.

The same-state reassessment is recorded at [repair01](../g7_repair_01/STAGE_REPORT.md). It changed no states, globals, objective or fit arrays; its derivative regression and profile-curvature checks passed, while the global KKT residual remained 0.003557. The failed attempt02 interval-curvature result remains historical evidence; attempt03's repaired certificate evaluates positive profile curvature.

## Decision and next question

**PAUSE / COMPLETE_UNCERTIFIED.** Keep `fit_complete`, `fit_certified`, `crosscheck_complete`, and `comparison_complete` false. Do not launch G8 or another fit in this campaign. The next numerical question is why positive constrained profile curvature coexists with rejected full shared proposals and no acceptable same-objective polish step near a 2.51e-5 global residual. Diagnose QP accuracy/scaling and constraint-aware Newton/line-search behavior without relaxing the physical bound or changing the objective.

- [Saved summary](summary.json), [checkpoint](checkpoint.json), [full arrays](fitted.npz), [console log](console.log)
- [Independent audit](mechanical_audit.json), [runtime provenance](provenance.json), [source snapshot](source_snapshot/)
- [Compact aggregation](compact_diagnostics/REPORT.md) and [machine-readable records](compact_diagnostics/summary.json)
- [Repair reassessment](../g7_repair_01/STAGE_REPORT.md)
