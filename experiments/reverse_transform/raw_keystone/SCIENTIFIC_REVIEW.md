# Raw-keystone correction: scientific review

Updated 2026-10-10. The active implementation is [raw_keystone.py](../../../distortion_model/raw_keystone.py). This chain implements the pipeline-wide correction in [the next-step plan](../../../docs/REVERSE_TRANSFORM_NEXT_STEP.md). Historical normalized stages and source snapshots remain preserved.

## What changed

The fitted output is now `H = center(K(reference))`, without dividing by its RMS radius, area, or a post-keystone size factor. For P4, the radial transform precedes K. One positive scale is profiled from P1 at each frame and applied unchanged to P4. In the joint-state experiment, that P1 scale is recomputed at every trial gaze. The exact inverse restores the model centroid, divides by the common P1 scale, and undoes the projective and radial transforms; it introduces no RMS factor.

Reference triangles are the centered arithmetic means of Capture 1's nominal zero-gaze fixation, labeled 0.360360 D. Capture 1 relative radial increment is zero by convention. The reference can contain baseline distortion, so relative κ is not absolute barrel calibration.

Primary fitting compares corresponding vertices in original centered camera pixels, with equal fixation weights. Inverse-reference errors are diagnostic. No measured triangle is size-normalized, rotated, or independently scaled for P4.

## Calibration findings

Fresh raw Stages 01–03 passed the saved-results CPU audits. The P1-only forward objective is unchanged at 1.013176 px² per vertex: profiling a free positive P1 scale cancels a scalar normalization of its model triangle exactly. Thus identical P1 residuals are expected and do not show that normalization remains in the implementation. The independent audit also exercises a deliberately nontrivial K that must retain its raw size variation.

For Capture 1, raw P1 keystone radius factors average 0.999872, 0.999944, 0.999996, 1.000116, and 1.000254 across −10°, −5°, 0°, +5°, +10°. The fitted raw P1 size response is small within this restricted four-coefficient parameterization; removal of normalization allows size change but does not require a large fitted effect.

After raw P4 inversion with κ=0, the Capture 1 recovered/reference radius ratios are 0.991446, 0.990559, 0.999984, 1.006584, and 1.004880. Their range is 0.01602, versus 0.01672 historically. The remaining gaze-conditioned size pattern is substantial; raw keystone removes about 4% of its range.

Independent raw capture fits give κ = [0, −1.8542103e−6, −1.4746662e−6, −8.4791740e−7] px⁻². The newly fitted post-fit association is

```text
κ(A) = −5.253231191989352e−7 · (A − 0.36036036036036034) px⁻².
```

This law was estimated after fresh image fits. Historical κ, β, gaze states, and accommodation states were not used as fixed raw-chain calibration. Its near numerical agreement with the previous law is an observed result, not inherited calibration. Demand occurs in one capture per level, so capture and demand remain confounded.

Per-capture K coefficients are retained. Native coefficient values and bound diagnostics must be considered before proposing any accommodation-dependent K term; different scaled coefficient values across captures are not directly comparable because their gaze units differ. Capture 2's P4 vertical projective coefficient is at its lower bound (scaled −0.3738277194); its radial coefficient is not at a bound. Independent capture differences alone do not identify a shared physiological K(θ,A) law. No additional differential P4 scale or automatic K coupling was introduced.

## Framewise A findings

Raw Stage 04 uses a 0.25 D soft anchor on each fixation's arithmetic mean A; it is neither a per-frame prior nor a hard ±0.25 D bound. A is operationally bounded to [0,6] D and the forward optical branch. Measured inverse failures cannot trim the fitting population or tighten its bounds.

All 89,175 complete frames remain in the forward fit, and all fitted P4 inverses are valid. The constant-κ parent has 46 measured inverse failures in Capture 2. These failures and the resulting unequal valid populations must remain explicit when comparing inverse statistics.

| Capture | Complete frames | Parent → fitted-A P4 forward RMS (px), all frames | Fitted-A P4 inverse RMS (px), all frames |
|---:|---:|---:|---:|
| 1 | 24,280 | 1.639 → 1.162 | 1.163 |
| 2 | 22,026 | 4.321 → 3.620 | 5.696 |
| 3 | 19,795 | 2.233 → 1.836 | 2.070 |
| 4 | 23,074 | 3.297 → 2.984 | 3.147 |

Capture 2 illustrates why forward and inverse errors are not interchangeable. Its forward error improves, while inversion near the radial turning point amplifies some errors. Reporting only a smaller common-valid inverse subset would conceal this behavior.

Capture 1 fixation-mean A is 0.566258, 0.587119, 0.362072, 0.211490, and 0.243050 D. Its range is 0.375630 D, compared with the historical inverse-fit range of 0.395497 D. The gaze trend persists. This historical comparison changes both keystone convention and fitting space.

The separate [normalized-forward control](stage_04_framewise_accommodation/results/normalized_forward_control/summary.json) passed its CPU audit and uses the same forward objective, 0.25 D mean anchors, and complete-frame population. Its Capture 1 means are 0.568601, 0.591254, 0.360897, 0.198295, and 0.204982 D: range 0.392959 D. Raw K reduces the matched-control range by about 4.4%. This comparison includes reoptimized raw optics, so it evaluates the full calibration convention rather than deleting a factor from old coefficients. It does not support attributing most of the remaining A pattern to normalization.

## Conditional joint gaze/A experiment

Raw Stage 05 ran after Stage 04 showed substantial residual bias. A0 preserves the raw Stage 04 states exactly. G varies horizontal gaze with A at the reference; GA varies horizontal gaze and A. Both reprofile the P1 scale at each trial gaze and use equal P1/P4 forward weights, frozen raw optical constants, frozen vertical gaze, and fixation-mean anchor widths 0.5° / 0.25 D. All 24,280 complete Capture 1 frames remain inverse-valid in all three models.

The selected G and GA objectives are 3.392117 and 2.490495 px². Three starts converge to distinct stationary solutions with objective ranges approximately 0.01308 / 0.01680 px². The independent CPU audit passed raw forward replay, finite-difference Jacobians/gradients/Hessians, stationarity, free frame curvature, and synthetic closure (maximum closure about 1.42e−13 px). These certificates do not establish a unique global optimum.

| Model | P1 forward median / P95 / RMS (px) | P4 forward median / P95 / RMS (px) | P4 inverse median / P95 / RMS (px) |
|---|---:|---:|---:|
| A0 | 0.675 / 2.009 / 1.017 | 0.950 / 1.908 / 1.162 | 0.942 / 1.921 / 1.163 |
| G | 0.549 / 1.377 / 0.804 | 1.429 / 2.707 / 1.646 | 1.438 / 2.700 / 1.644 |
| GA | 0.550 / 1.433 / 0.834 | 0.928 / 1.937 / 1.170 | 0.934 / 1.958 / 1.173 |

G improves P1 but does not recover A0's P4 accuracy. GA improves P4 relative to G, but its P4 RMS/P95 are slightly worse than A0. A0's objective includes only the A-dependent P4 fit; G/GA include P1 and different gaze anchoring. Compare corresponding data metrics rather than treating the raw optimizer objectives as identical tests.

GA A means are 0.566973, 0.587151, 0.362161, 0.214347, and 0.243099 D. The largest change from A0 is 0.002857 D, so freeing horizontal gaze does not remove the A pattern. GA gaze SD is 5.285°, 1.165°, 2.127°, 11.522°, and 1.612° across the five fixations. GA has 1,155 lower and two upper gaze-bound frames, plus 878 lower A-bound frames. Large within-fixation spread, bounds, and distinct stationary basins prevent a physiological interpretation of these framewise states.

GA's local two-column gaze/A Jacobian condition median/P95 is 52.53/228.75 for P4 and 33.62/85.22 jointly. Median acute response-column angles are 84.48° / 86.32°, respectively. Local columns are not near-collinear, but weak gaze sensitivity, operational bounds, and multiple basins still limit global/state identifiability. Condition numbers use degree and diopter units; column angles are unit invariant. State SD is within-fixation spread, not uncertainty of the mean.

No additional A-dependent K term is accepted here. The native horizontal quadratic coefficient does increase with demand: −1.4327e−5, 1.0763e−5, 2.7607e−5, and 6.2526e−5 deg⁻² at approximately 0.36, 2, 3, and 4 D. This is a concrete candidate for a later minimal K coupling comparison. Other K coefficients are not consistently monotonic, and vertical terms have weak calibration support. Each demand belongs to a different capture, so the coefficient association alone cannot establish accommodation causality or justify applying the same coupling to fluctuating framewise A. A shared constant-versus-linear horizontal coefficient comparison in original coordinates has not been performed here; retain independently fitted K until that comparison and identifiability checks are completed.

## Interpretation

The corrected forward and inverse transforms preserve raw keystone size change. That correction produces modest changes in the measured Capture 1 size and A trends; it does not establish that the remaining trend is true accommodation. A parameter fitted to reduce vertex error can absorb gaze-conditioned model mismatch.

The retained empirical reference, the assumed common first-order horizontal/vertical gaze slope, weak vertical excitation, capture/demand confounding, operational coefficient/state bounds, and use of the fitting recordings for evaluation limit optical interpretation. Numerical stationarity and synthetic closure certify implementation under the stated model; they do not certify physiology or held-out accuracy.

Further model work should start from this raw chain. Reintroducing size normalization would erase a permitted keystone effect. A new shared accommodation-dependent K term or differential gaze-size term requires its own coordinate-prediction and identifiability comparison before promotion. Held-P4 validation remains later work.


## Stage 06: conditional center-to-gaze intermediate step

The user's requested intermediate step derives origins separately for each frame, rather than releasing free center parameters: `C_r=c_r−g*mean(F_r)`. Five arithmetic fixation means of `(C4−C1)/g` calibrate a horizontal quadratic polynomial, with its origin at the zero-fixation mean; vertical gaze uses the same first-order slope. This fits the polynomial at the mean inputs, not the mean of quadratic outputs. Therefore framewise horizontal gaze can have a small nonzero fixation mean at the zero reference because of within-fixation variance. Each updated gaze profiles P1 magnification and then fits A conditional on that gaze, with the unchanged .25 D fixation-mean A penalty. The iteration does not add a separate centroid penalty or fit a Theory global D block. No RMS/area normalization occurs.

The six-cycle CuPy run is COMPLETE with an independent CPU PASS audit. It retains 89,175 frames, all inverses valid, and the next undamped gaze update is at most 1.48e−6 degrees. Every A-only subproblem has stationary projected gradients and positive free-frame curvature. This numerical certification does not certify physiological states, unique centers, or a joint optimum; the outer map need not decrease one common cost monotonically.

P4 forward RMS changes by capture from 1.162/3.620/1.836/2.984 to 1.175/3.594/1.753/2.985 px. Median improves for Captures 1–3, but Capture 1 RMS/P95 worsen slightly. Capture 2 inverse RMS increases from 5.696 to 5.965 px even though forward median/P95/RMS improve, emphasizing radial inverse amplification. Capture 3 improves both metrics. P1 forward RMS changes little. The remaining fixation-conditioned A pattern persists: Capture 1 means become .56664/.58709/.36226/.21374/.24386 D. Centers are conditional gauge-defined origins, not independently measured physical distortion centers. Retain the corrected center calculation and transparent diagnostics, without claiming it resolves accommodation bias. See [Stage 06](stage_06_framewise_centers/README.md) for replay and saved results.
