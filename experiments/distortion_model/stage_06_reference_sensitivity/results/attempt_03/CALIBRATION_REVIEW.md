# Root calibration review — matched reference sensitivity

## Gate decision

**Control: GO_WITH_LIMIT to frozen G8. Adjacent: NO_GO_UNCERTIFIED. The reference comparison remains incomplete.**

The root reviewed both saved full certificates, original and continuation histories, the independent public-optics objective reconstruction, and the declared numerical controls. No inference or optical ranking is permitted for the adjacent calibration. The current convention remains the operational reference pending a complete certified comparison.

Both candidates used the same 100,090 scheduled frames, 89,175 complete calibration rows, twenty exposures, covariance, camera axes, omega1, state domains, full-mean anchors, model family and numeric one-degree physical bound. Their reference exposures and empirical template gauges differ as predeclared. This experiment estimates operational-reference sensitivity, not a physical symmetry zero.

## Numerical evidence

The original 48-round starts were preserved under attempt_01. The separately declared working-set continuation is under attempt_02. Its control starts both certified after four accepted steps. A near-active positive-slack row incorrectly initialized as a primal equality blocked the adjacent starts. Attempt_03 repairs that initialization, retaining every inequality and only using the remaining twelve-step continuation budget. Its control starts stop immediately; its adjacent starts each accept seven steps and then reject all twelve line-search fractions. There is no extra budget or weakened gate.

The selected control objective is **1925.6013386760935**, reproducing the earlier certified G7 objective within 9.6e-11. State stationarity is **3.92058e-12**, global KKT stationarity **6.36353e-8**, and complementarity **6.88043e-11**; each required residual is below 1e-6. All 89,175 local data ranks are two, free-state observed curvature is positive, the constrained profile is positive with rank/dimension sixteen, and both optics domains and the continuous physical bound pass. Accommodation is at a bound in 8,415 calibration rows; this remains a limitation.

The adjacent selected objective is **1925.8105580440379**. Its state residual **4.13310e-5**, global KKT residual **0.551904**, and complementarity **1.80740e-6** fail the unchanged 1e-6 gates. Positive curvature, rank and physical feasibility do not replace stationarity. Accommodation is at a bound in 9,628 calibration rows. Independent public-optics reconstruction passes for both candidates; adjacent prediction differences are at most 3.41e-13 px. These checks establish saved-result consistency, not calibration certification.

The adjacent alpha4 coefficient approaches **-4.62040e-11**. The final quadratic proposals switch their gaze-floor working row from interval index 0 to 23. Although the recorded final branch is smooth, this near-zero coefficient and the interval endpoint switching identify a numerical branch-handling question. They do not prove a physical failure of the adjacent reference or prove that a nonsmooth optimum is the answer. Every corrected final trial is physically feasible, but its stable original-objective change is positive (9.24e-8 even at fraction 1/2048), so rejection is justified. Accepting such a step or reporting a held-point score from this fit would bypass the declared gate.

## Scientific scope and next action

Run the identical full G8 pipeline for the certified control, including actual-record backend and omitted-input audits. Preserve an explicit unattempted omission roster for the adjacent candidate. Record an incomplete paired comparison, with no adjacent E, gaze disagreement or accommodation disagreement claimed.

The experiment establishes reproducibility of the original calibrated optimum and identifies a remaining constrained-solver problem for the alternative. It cannot decide whether the alternative reduces the structured omitted-point residual. A subsequent numerical investigation should examine one-sided interval endpoint derivatives around alpha4=0, distinguish the broad NNLS near-active census from actual complementarity, and verify any generalized stationarity or curvature certificate independently before resuming the optical comparison. Another optical mechanism is not authorized automatically by this outcome.

Absolute accommodation accuracy, physical gaze/accommodation zeros, calibrated localization-noise covariance, common camera scale transfer and capture/demand confounding remain unresolved. The prior full G8 findings remain valid and unchanged.
