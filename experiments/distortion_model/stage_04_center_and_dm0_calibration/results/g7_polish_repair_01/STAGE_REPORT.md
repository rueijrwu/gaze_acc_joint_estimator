# G7 saved-point proposal probe and compact-record interpretation

**Probe status: PROBE_PENDING_REVIEW; no calibration checkpoint was written.** The original [probe summary](summary.json) and [provenance](provenance.json) are preserved. After review, its bitwise-verified candidate was recorded separately as [attempt04](../g7_attempt_04/STAGE_REPORT.md). This report distinguishes the probe evidence from the later adopted result.

## N1 — Curvature conditions and constrained subproblem

The legacy full-shared-space proposal route rejected LM values 0 through 0.01 for nonpositive shared proposal curvature. At LM 0.1 and 1.0 it proposed zero shared steps and did not lower the objective. The constrained Lagrangian tangent-space route solved the same saved-point local subproblem with three active physical rows of rank 3 and tangent dimension 16. The two independent small-system routes' scaled steps differ by 6.92e-17. This demonstrates the route mismatch at this point; it does not show that curvature alone explains all prior rejections.

## N2 — Independent local KKT ledger

The null-space subproblem has scaled stationarity infinity norm 1.23e-11, active feasibility residual 6.25e-22, complementarity 2.06e-21, and minimum multiplier 3.2935. The minimum linearized physical slack is −6.25e-22 (roundoff); minimum box slack is 0.42188. The 19-coordinate shared step is about 1e-7 in scaled coordinates, with frame-state step maxima 1.70e-7 degrees and 6.28e-7 D.

## N3 — Feasibility and objective accounting

The full undamped proposal is accepted by the fixed saved-point rule. Minimum actual physical slack is 0, and maximum nonlinear-versus-linearized constraint discrepancy is 3.56e-15. Direct full-objective change from component evaluation is −5.9962076704e-12; accumulated subtraction is −5.6843418861e-12, a 3.12e-13 cancellation difference. The point component rises 1.38427e-6 while the gaze anchor falls 1.58610e-6; other changes are small. Maximum prediction movement is 6.19e-6 px. This is a local numerical correction, not material prediction improvement.

## N4 — Unchanged certificate threshold

The recomputed candidate has state/global projected gradients 1.32e-13 and 4.24e-10, against the unchanged 1e-6 threshold; complementarity is 3.15e-14. The one-degree continuous physical constraint is feasible. All 89,175 states have rank-two data support; free-state curvature is positive (zero negative eigenvalues among 169,935); constrained profile curvature is positive with rank 16; P1/P4 domains are valid. This was a saved-point proposal, not a complete checkpoint at probe time. Attempt04 later recorded this candidate after independent verification.

## E1 — Existing compact records, all identities retained

The [compact aggregation](empirical_diagnostics/REPORT.md) reads only the pre-existing attempt03 `compact/*.json`: 300 scheduled slots, 270 scored, 30 unavailable, 90 complete eligible triples, and all 20 exposures. Equal-exposure summaries are E=41.8138 px, Gtheta=1.50113 degrees, and GA=1.31867 D. The adjacent capture2 rows at exposure6/row6800 and exposure5/row6799 contribute 94.3568% of equal-exposure E² and 91.8257% of Gtheta² together. Endpoint and state-bound strata show larger upper tails in their descriptive RMS summaries. These are ranked empirical contributors; no transition cause is asserted, and no endpoint or high-error row was trimmed. Compact disagreement is internal subset behavior, not physical accuracy or G8 evaluation.

The aggregation script was enhanced after the original probe to display ranked contributions and strata. The adoption provenance records both the probe-time and current summarizer hashes and identifies the change as reporting-only; the probe's original provenance is untouched. No new inference, broad fit, automated test suite, or G8 evaluation ran.
