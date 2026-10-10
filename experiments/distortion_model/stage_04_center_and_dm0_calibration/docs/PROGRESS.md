# Stage 4 resume pointer

S6/G6 is COMPLETE / GO_WITH_LIMIT. S7/G7 attempt03 is COMPLETE_UNCERTIFIED / PAUSE. G8 is not authorized.

Last usable parent checkpoint: [G6 attempt 02](../results/g6_attempt_02/checkpoint.json). Attempt03 is the one authorized bounded continuation from attempt02's selected perturbed state. Its [report](../results/g7_attempt_03/STAGE_REPORT.md), [summary](../results/g7_attempt_03/summary.json), [checkpoint](../results/g7_attempt_03/checkpoint.json), and [independent audit](../results/g7_attempt_03/mechanical_audit.json) preserve the full result. Attempt03's initial states/globals are bitwise identical to attempt02; all 144 attempt02 hashes recorded by repair01 still match. Attempts01/02 remain unchanged historical records.

The smooth derivative regression, physical bound, state curvature, and constrained profile curvature pass. Global scaled KKT residual remains 2.5103e-5 against a 1e-6 threshold. The 8-outer continuation reached J=1925.601338676003 from 1925.6013387428595, without satisfying stationarity. History records one unaccepted observed-polish proposal, 36 objective line-search rejections and 5 nonpositive shared proposal-curvature failures in that proposal. The compact diagnostic has 270/300 scored slots, 30 unavailable and zero unresolved; it remains progress-only.

**Next numerical question:** why does positive constrained profile curvature coexist with rejected full shared proposals and no acceptable same-objective polish step near a 2.51e-5 global residual? Diagnose QP accuracy/scaling and constraint-aware Newton/line-search behavior while preserving the objective and physical bound. No further fit in this campaign; keep G8 paused.

Reference limits remain: omega1/omega4 are fixed at operational empirical references; omega1 continuous uncertainty is unquantified; G3 discrete alternatives and physical zeros remain unresolved. The one-degree value is user-authorized, not independently measured optical calibration or an individual gaze-error bound. See [distortion-tracking source evidence](DISTORTION_TRACKING_EVIDENCE.md).
