# Subplan 1 — Trust the measurements; initialize gaze

**Covers:** S0/G0 and S1/G1. **Status:** NOT_RUN.  
**Parent:** [Gate policy and recovery](../STAGE_GATES.md).  
**Equations:** [Theory](../Theory.md); [implementation stages S0–S1](../ESTIMATOR_PLAN.md).

## Scope and deliverable

Produce one input census and one empirical gaze-bootstrap report before developing the complete accommodation optimizer. Keep the outcome as an initializer, not a measured gaze trajectory or a completed optical calibration.

## S0 / G0 — Is the observation population trustworthy?

**Inputs fixed:** capture 1–4 source bytes, reviewed full fixation intervals, stored source correspondence, native camera units, sign P4-minus-P1. Captures 5/6 are not read for calibration decisions.

**Minimum work:** inspect payload metadata and point flags; reconstruct the ten-component relative vector and a candidate-independent frame/slot manifest. Report per-fixation counts, original frame/timestamp continuity, correspondence mapping, point availability, and interval-edge behavior. Inspect a few representative native point tracks from each condition. Preserve approximate interval-boundary uncertainty; do not assume every boundary row is a settled fixation or silently trim it.

Test the relative map's rank and translation invariance, covariance mapping, subset input masking, and an all-slot failure record on synthetic inputs. Establish the sign/units and numerical domain checks needed by the first forward adapter. Implement the minimal forward calculation from the authoritative equations with a synthetic parameter fixture; do not fit all optical parameters yet. Add only the parameter/derivative tests needed as later blocks become active.

**Save:** census and immutable manifest; the relative-map/correspondence test outcomes; one native-track view with interval boundaries and validity marks; config/data/code hashes. In `STAGE_REPORT.md`, state what was actually read and tested.

**GO:** all twenty intended exposures can be identified; no missing identity or inconsistent point mapping is hidden; enough valid data exist to initialize each needed reference condition; native relative coordinates and masks obey their contracts.

**GO_WITH_LIMIT:** known timestamp anomalies or approximate transition boundaries exist but row/frame IDs and declared full intervals remain usable. Name the affected diagnostics and revisit influence at G7/G8. Do not use unreliable timestamps for noise differences or claim a corrected timing signal.

**REPAIR:** incorrect permutation, unit conversion, duplicated identities, mismatched source hashes, unexplained missing records, or faulty covariance/mask code. Repair loading/bookkeeping only.

**PAUSE:** the actual source correspondence or required exposure cannot be established. A subset-only diagnostic may still be reported, but not a full twenty-condition calibration. Do not synthesize missing points or relabel demands.

**Next action:** G1 centroid bootstrap. Do not build a new detector, GUI, or data-cleaning framework at this gate.

## S1 / G1 — Does centroid displacement provide a usable gaze initializer?

**Free block:** one inverse bootstrap polynomial. **Frozen:** source data, population, other optical blocks. Before scale calibration, use the explicitly provisional reference-scale assumption.

For each frame in the lowest-demand recording, calculate spatial means c1 and c4, then its displacement c4x-c1x. Take the temporal mean within each of the five fixation intervals and fit the primarily linear displacement-to-visual-gaze initializer. Use the actual demand label for Aref, not a relabeled physiological zero. Apply the initializer to every frame's own displacement.

**Inspect:** the five mean displacements against nominal -10,-5,0,5,10 degrees; within-fixation distributions and several contiguous-block means; the sign and gain; conditioning and invertibility over the sampled displacement range; framewise initial gaze. These five target labels calibrate the initializer and are not independent accuracy observations. Do not infer a per-frame gaze error from their scatter.

Plot other captures on the same axes only to see provisional transfer and accommodation/capture offsets. Differences there do not reject the reference bootstrap: accommodation and scale corrections are not yet available.

**GO:** a finite, noncollapsed relationship provides usable starts across the five reference conditions without an unexplained ordering/sign contradiction. Nonzero residual and natural within-fixation spread are expected, not failures.

**GO_WITH_LIMIT:** initialization is approximate or has reproducible curvature. Keep the linear initializer or one declared degree-2/3 alternative within the existing plan; record ambiguous tails or candidate starts for G2/G7. Do not demand identical bootstrap behavior across accommodations.

**REPAIR:** wrong sign, interval pairing, source ordering or inversion convention. Fix it before fitting curvature.

**PAUSE:** no usable displacement–gaze relationship remains after the input checks, or multiple incompatible mappings would decide different symmetry references without evidence. Record the conflicting conditions and the smallest additional check. Do not force a fifth-degree interpolant through five summaries.

**Save:** bootstrap coefficients, degree/domain, five-condition table, actual framewise starts, and the gate decision. Revisit after P1 scale at G2 and after center correction at G6; the bootstrap is never the final center polynomial.

**Next action:** G2 P1 reference and scale, not direct accommodation-law selection.
