# Stage 05 scientific review: Capture 1 state separation

## Decision

**COMPLETE; independent numerical audit PASS. The remaining A/gaze trend is unresolved.**

This experiment most closely matches **Case 4** in
[the next-step protocol](../../../docs/REVERSE_TRANSFORM_NEXT_STEP.md): the fixation-dependent accommodation-like state persists after horizontal gaze is freed. Proceed to a controlled differential P4/P1 gaze-size ablation if requested. The present result does not warrant promoting the framewise gaze/A estimates to physiological measurements or starting held-P4 validation as though state separation had been established.

## What was fitted

All 24,280 complete Capture 1 frames were retained, with the same reviewed correspondence and capture-1 zero-gaze empirical references. No prior stages were refitted or overwritten.

- **A0:** the exact saved Stage 04 gaze, A, and P1 magnification. Its original inverse objective was preserved; it was not optimized again under the new objective.
- **G:** only horizontal gaze varies; A=A_ref and κ=0. Vertical gaze stays frozen.
- **GA:** horizontal gaze and A vary; κ=β(A−A_ref), with the existing β and A_ref frozen.

References/origins, both sets of keystone coefficients, units, population and correspondence are frozen. Every trial gaze recomputes the P1 shape and profiles the positive common magnification from P1. P4 uses that same scale; there is no free P4 magnification or differential gaze-size term.

The predeclared data objective is the equal-fixation mean of **P1 mean-three-vertex squared error plus P4 mean-three-vertex squared error** in original centered camera coordinates, with equal unit weights. Only arithmetic fixation-mean anchors are used: 0.5° for horizontal gaze and 0.25 D for GA accommodation, with residual scale 1 px. Operational ranges are horizontal gaze [−20,+20]° and A [0,6] D. These ranges are not framewise priors or ground truth. Forward fitting does not discard observations to satisfy the measured inverse domain; inverse failures are reported separately.

## Main comparison

The following are pooled point-distance RMS values; all three models have valid inverses for all 24,280 frames. See [RESULTS.md](RESULTS.md) for medians, P95 and fixation-level tables.

| Model | P1 forward RMS (px) | P4 forward RMS (px) | P1 inverse RMS (px) | P4 inverse RMS (px) |
|---|---:|---:|---:|---:|
| A0 | 1.017 | 1.186 | 1.022 | 1.184 |
| G | 0.800 | 1.699 | 0.802 | 1.697 |
| GA | 0.801 | 1.185 | 0.803 | 1.187 |

G does not recover the P4 improvement previously provided by A. GA substantially improves P1 relative to A0, but leaves P4 forward RMS essentially unchanged; P4 inverse RMS and P95 are slightly worse. GA outperforms G for P4 because the relative P4 size/radial state remains needed. That comparison alone does not identify physiological accommodation.

| Nominal gaze | A0 mean A (D) | GA mean A (D) | GA horizontal gaze SD (°) |
|---:|---:|---:|---:|
| −10° | 0.566 | 0.569 | 5.309 |
| −5° | 0.592 | 0.591 | 1.243 |
| 0° | 0.359 | 0.361 | 2.416 |
| +5° | 0.196 | 0.201 | 12.968 |
| +10° | 0.207 | 0.205 | 1.659 |

The A means change by only about 0.005 D at most. Their approximately 0.39-D peak-to-trough gaze association survives. Soft anchors keep the fixation gaze means near their nominal labels, while individual gaze states can move extensively: GA has 1,311 frames at −20° and two at +20°, and 909 frames at A=0. This diagnostic uses the centroid-derived frame gaze as initialization, without a framewise centroid-to-gaze penalty; its image objective uses centered triangles. The wide gaze distribution, especially at +5°, is a limitation of this unconstrained framewise state diagnostic. A fixation-mean constraint cannot validate individual frame gaze or prevent compensating state allocations across frames.

## Local conditioning and numerical evidence

The Jacobians include the derivative of the recomputed P1 magnification. Both P4-only and joint P1+P4 Jacobians were evaluated for every frame. For GA, the median acute angle between gaze and A response columns is about 87.8° for P4 and 88.4° jointly. The columns are therefore not locally near-collinear in this experiment. Raw joint condition number median/P95 is approximately 35.8/120.3 in degree/diopter units; this ratio also reflects unequal response amplitudes and depends on the units. The normalized angle and column correlation are invariant to state-unit rescaling.

Local column separation does not establish global or physiological identifiability. The weak gaze response, extensive state movement, operational bound hits and different stationary multistart solutions matter. G's multistart objective range is 0.00572 px² and GA's is 0.00541 px²; report the selected stationary solutions without claiming a unique global optimum. The radial law also remains capture/demand-confounded, and the reference retains baseline distortion.

The selected G/GA projected-gradient infinity norms independently replay to approximately 9.66e−11 and 2.19e−8. Minimum independent free-frame curvatures are positive, approximately 1.21e−4 and 1.25e−4. Mean-anchor curvature blocks are positive semidefinite, so positive independent frame curvature is a sufficient local free-subspace certificate. The initial L-BFGS-B iteration-limit flags were resolved by block Newton polishing; those flags are retained in the recorded optimizer history. Two earlier numerical-limit attempts are noncanonical and are retained only as development evidence.

The [independent CPU audit](results/run/audit.json) checks frozen hashes and population, exact A0 preservation, independent forward replay, all-frame Jacobian/gradient/Hessian finite differences, state bounds, projected stationarity, free curvature, inverse-array replay, synthetic closure and conditioning near bounds. This certifies the saved conditional solution numerically. It does not certify framewise optical truth. The vectorized fit used CuPy float64 on the P100; the canonical fit and result collection took about 40.7 seconds.

## What to test next

The current size-normalized keystone preserves the pre-keystone P4 RMS size. Gaze can affect P4 shape and the P1-profiled common magnification, but there is no independently represented differential P4/P1 gaze-size response. Allowing gaze states to vary has not removed the A trend.

The next controlled candidate is a reference-anchored differential gaze-size term such as `s41(θx)=1+c1 θx+c2 θx²`, with its fitting and validation policy declared beforehand. This term is **not implemented in Stage 05**. It must be compared against the present frozen model rather than assumed to explain the residual. Do not add response exponents, free framewise κ, refit β, free vertical gaze, or add framewise priors to this completed experiment. Held-P4 cross-agreement remains a later validation step after a credible state/model separation result.
