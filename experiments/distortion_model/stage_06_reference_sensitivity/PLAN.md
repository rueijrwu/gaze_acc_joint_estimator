# Stage 6 — One matched operational P4 reference sensitivity

Authorized by the user's request to implement the next stage following completed G8. The protocol is declared before fitting, in `results/attempt_01/protocol.json`.

Compare current P4 omega4=-10.016974132845107 degrees against the documented adjacent omega4=-5.0148989655383795 degrees. Keep omega1 fixed. Reinitialize the empirical P4 template from each convention's low-demand reference exposure and near window, then perform fresh conditional G4/G5, exact G6 center initialization, bounded joint G7 and full frozen G8 for both conventions. This is a comparison of the same constrained model family under two operational references. It does not estimate a physical symmetry zero.

## Matched controls

Preserve all 100,090 scheduled frames and twenty exposures, source correspondence, native coordinates, covariance, state domains, .10 degree/.25 D full-mean anchors, zero temporal penalty, degree-two D and DM0-M1. Camera axes and rotation coefficients are not an additional intervention. Template tangent sigma .10 and normalized D curvature priors use the same formulas; empirical template radius and consequent native prior scale are reported for each reference. Changing the empirical reference changes this gauge, so the intervention is not a pure coordinate relabeling.

Use the **same numeric** physical gaze slope floor, 10.02700222585176 reference pixels/degree, and accommodation cap 2.50675055646294 reference pixels/D for both candidates. These are the certified existing G7 constants and retain the user-authorized one-degree centroid response bound. Recomputing the threshold from each candidate would change another control. If necessary, project only the fresh initializer's D coefficients into this common feasible domain and record that projection.

Each candidate gets two fresh starts: its own G6 initialization and the same PCG64 seed20261010 perturbation policy. Each start has at most 48 outer rounds, four joint steps per round and twelve constraint-consistent Newton polishing steps. Select the lowest full objective among certified starts. Preserve failed starts and numerical/branch differences. Budget exhaustion remains uncertified; it is not a negative optical-law result. No cross-error selects a calibration.

## Evidence and decision

Independently reconstruct the full objective and public optical predictions for each saved calibration before root review permits frozen inference. Use the existing identical 50-start G8 solver, full three-omission roster and fixed all-three diagnostic schedule. Verify masking, covariance marginals and CPU/GPU numerical agreement on actual recorded rows as experiment audits; no repository automated test suite is part of this stage.

Compare exact common frame and point identities with all-scheduled coverage. Report E, Gtheta and GA separately; signed point/axis/exposure/time-block residuals; native medians and tails; bounds; independent squared contribution changes; conditional A information; model scale and operational-reference limits. Show changes across all twenty exposures and separately outside the known capture2 neighborhood without changing the primary denominator. Missing common exposures or uncertified fits make a complete all-exposure comparison unavailable.

A broad reproducible reduction of the structured omitted-P4-index-1 residual with useful accommodation information supports reference sensitivity within these conventions. Gains concentrated in the known neighborhood, clipping/compression, or persistent weak A information leave the mechanism unresolved. No fixed RMS ceiling or mixed-unit ranking is used. Physical accuracy, localization-noise calibration, common axial scale transfer and capture/demand confounding remain unresolved.

Preserve previous G7/G8 artifacts. Place new scripts, calibrated candidates, crosschecks, comparison and handoff under this Stage 6 folder; do not overwrite previous attempts or start another mechanism automatically.
