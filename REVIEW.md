# Review of suggestions for resuming the full-information calibration

Scope: audit of a set of suggestions about measurement/model assumptions, scale
calibration and regularization-versus-optical-information, written after reading
`documents/FULL_INFORMATION_HANDOFF.md`. Read-only: no fits or new computations
were run. Evidence is from `documents/Theory.md`, `documents/CURRENT_PLAN.md`,
`exp2/full_information/*.py` and the existing JSON/CSV outputs.

## Main finding (missed by the original suggestions)

Each capture uses a single demand, and the five gaze targets (-15, -7.5, 0, 7.5, 15 deg)
are visited in consecutive frame blocks within it (`exp2/fixations/fixation_intervals.json`).
Demand is therefore confounded with recording, and gaze with time and drift.

The fixed-scale log-S1 shape `[1, theta/15, A**p]` is one global shape with no
per-capture offset (`exp2/full_information/joint_problem.py:21`). Its accommodation
term is effectively fit to 3-4 capture-level S1 medians
(445.89 / 446.12 / 445.96 / 447.02 px at 0.36 / 2 / 3 / 4 D). Any head-distance
change between captures is read as accommodation.

Consequence: even converged R3/R4 fits could not show that S1 carries gaze or
accommodation information in this archive.

## Verdict per suggestion

| Suggestion | Verdict | Evidence |
|---|---|---|
| Information bound from the N3 map | Wrong | N3 has 14 coefficients and no log-S1 model. S1 sensitivity exists only in the 3 shape coefficients of the unconverged 17-coefficient fits. The free-scale case is already proven to add nothing (`scale_controls.json`); the formula is in Theory.md. |
| Within-capture log-S1 plots vs gaze/demand | Already done; impossible for demand | `scale_structure.json` states demand is not identifiable with recording effects and already has a within-recording regression (R2 = 0.80). Gaze cannot be separated from time. |
| Physical framing (distance vs state) | Imprecise | Theory.md already treats S1 invariance as a hypothesis. S1 also tracks detector angle error (r = 0.76-0.89 per capture), which was not mentioned. |
| Scale proxy: glint spacing | Wrong (circular) | S1 is the P1 glint spacing. Rig features are fixed to the camera and say nothing about head distance. |
| Scale proxy: limbus/iris | Plausible, unverified | Needs the raw videos (repo has only detection pickles) and a new detector. Horizontal width changes with gaze, so only vertical extent is a candidate. |
| Break-even scale spec | Good idea, two errors | "Tens of microns" has no basis (no working distance in the repo; glint spacing may not scale as 1/D). The per-frame scale model (`joint_problem.py:253-286`) lets scale error average out over about 4,000 frames per fixation, whereas real head-distance error is per-capture or slow drift. |
| Held-out check with log-determinant terms | Withdraw | The loss is a robust radial loss, not a true likelihood; models predict different observables; held-out [d, rho] residuals are near zero for any two-state model. |
| Prior/data curvature split | Result fixed in advance | With 2 observations and 2 free states per frame, the optical term carries essentially no coefficient information, so every direction is anchor-determined by construction. |
| Anchor sweep over decades | Already answered | Anchor cost scales with strength (0.331 / 0.659 / 1.309 at 0.5x / 1x / 2x; 2x unconverged); means moved only 0.0009 deg / 0.0005 D at half strength. The claim that the 2x stall indicates worse conditioning is unverified. |
| More held-out fixations | Partly feasible | Holding out +/-15 deg means extrapolating in gaze; 3 D is untrained in the 3-demand arm. 9 (3-demand) or 12 (4-demand) fixations are usable. These are new production fits and belong after the handoff's items 1-4. |

## Knot / positive-profile stall (corrected)

- The two-channel map's `b(A)` and `s(A)` are piecewise linear with knots at 0.36, 2, 3, 4 D and
  right-hand derivatives at knots (`exp2/calibrate_profiled.py:34`).
- The worst coordinate in each failing positive probe is one frame in fixation 19
  (+15 deg, 2 D), not the probed fixation 17: H0 frame 24776, FC0 frame 24841, both at
  A = 2 + ~2e-10 (`validation/protocol/profile_actual_gradient_audit.json`).
- A kink minimum there would defeat any smooth-gradient test. This is a hypothesis; the
  pending one-sided finite-difference check on that frame is the right test and needs
  authorization.
- A smoother basis would contradict Theory.md and change every existing fit. A kink-aware
  optimality test (zero lies between left/right A-derivatives; other gradients vanish with
  A held at the knot) keeps the model, but still changes the acceptance rule: needs
  authorization and must be written down before inspecting results.

## Also missing

- The model assumes accommodation equals demand (no lag). The accommodation anchor term is
  83% of H0's objective (0.550 of 0.662). Only an independent accommodation measurement can
  test this assumption.
- Linear drift of `d` within a capture can be absorbed into the gain `s(A)` because gaze is
  ordered in time.

## Revised suggestions

1. No computation: record that this archive cannot attribute S1 changes to accommodation or
   gaze; decide whether to drop the full-relative (R/U) fits for this question.
2. Optional, labelled hypothetical: state how strongly log S1 would need to depend on gaze or
   accommodation, and how stable scale must be per capture, for S1 to matter; compare with
   observed variation (about 1% across fixations within a capture, 0.25% between captures).
3. If continuing the S1 question: write an acquisition spec with demand varied within one
   recording, randomized gaze order, a head-distance reference and ideally an independent
   accommodation measurement.
4. Knot: decide whether to authorize the one-sided check on the single frame.
5. Anchors/holdouts: vary gaze-anchor weight against accommodation-anchor weight separately;
   run holdouts only on the 9-12 usable fixations, after the earlier decisions.
