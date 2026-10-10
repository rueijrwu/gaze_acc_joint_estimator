# G2 / attempt 01 / P1 initialization

Status: COMPLETE (uncertified checkpoint). Decision: REPAIR.
Evidence kind: conditional empirical P1 fit and observed balance curves; no full optical fit.
Parent gate: stage 1 G0/G1 attempt 02, GO_WITH_LIMIT.
Source/config hashes and source snapshot: provenance.json and source_snapshot/.
Population: all 100,090 scheduled rows; 89,175 complete fit frames in all twenty exposures.
Fixed: empirical reference/length/origin/axes, operational omega1, trace-free scale convention,
upstream theta and R11. Fitted: two constrained P1 coefficients. Physical zero unknown.

The chosen observed-balance endpoint reference is nominal -10 degrees (operational
omega1 -10.0209041116 degrees). The coefficients are interior and conditional rank is two.
Both starts reduce the same objective from 2263.041338 to 1921.198495; local-domain checks pass.
The solver stopped on ftol after six evaluations, but scaled projected gradients were
8.78965e-5 and 1.81029e-3, above the previously declared 1e-6 certificate tolerance.
No certified P1 scale/bootstrap checkpoint was exported. This is a numerical stopping
issue; the tolerance is not weakened and no optical conclusion is inferred.

One next action: bounded Gauss–Newton polishing of the same full criterion, guarded by
floating-point cost-rounding and improving stationarity; add a loss-floor regression test
and execute one fresh attempt. Preserve these outcomes and source snapshot.
Reviewer: Codex implementation audit, 2026-10-09.
