# Stage 3 resume pointer

Current gates: S3/G3, S4/G4, S5/G5 and S6/G6 COMPLETE, GO_WITH_LIMIT; S7/G7 COMPLETE_UNCERTIFIED, PAUSE.
Latest preserved checkpoint: [G7 attempt 01, uncertified](../../stage_04_center_and_dm0_calibration/results/g7_attempt_01/checkpoint.json). Its `fit_complete` and `fit_certified` flags are false; G8 is not authorized.

Limits: endpoint optical reference/physical origin unknown, capture-demand confounding, approximate transferred visual theta, G4 reference-sensitive slope. G5 shows a weighted/native metric tradeoff, structured signed residuals, concentrated lower-bound A and mean-demand deviations. Both identity and composed snapshots are preserved. Conditional rank/stationarity does not certify full optical calibration or accommodation accuracy.

One next action: diagnose G7's nonpositive observed local curvature and large per-frame joint trust caps, starting with exposure 6 and outlier/transition rows. Consider one compatible continuation only if that diagnosis justifies it. G8 remains unrun. Verify source and pinned parent hashes before any continuation.
