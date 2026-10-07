# exp5_single handoff — 2026-10-07

## Purpose and source precedence

This package provides the current exp5 baseline and a starting point for investigating whether all three P4 positions, including both x and y, improve joint gaze/accommodation estimation and geometric cross-checks.

This handoff consolidates the current `exp5/HANDOFF.md` with useful measurement, identifiability, and validation principles from `documents/Theory.md` and `documents/FULL_INFORMATION_HANDOFF.md` in the source workspace. The latter two describe older **two-P1/two-P4 horizontal-position experiments**. Their model equations, numerical outcomes, case names, and pause instructions belong to that historical work; they do not describe the current three-point exp5 estimator or establish the status of a future exp5 full-position estimator.

All runnable paths below are relative to this folder, which can become the root of a new repository. Consolidation did not resume the historical experiments or implement the proposed full-position estimator.

## Current package and completed baseline

- `data/detections/capture_{1..6}_detections.pkl`: complete stored detection payloads, including selected points and raw candidates.
- `data/fixations/fixation_intervals.json`: reviewed fixation intervals and nominal gaze/demand labels for captures 1–4.
- `models/quadratic_model.json`: frozen converged quadratic joint model.
- `joint_m2.py`: triangle measurements, model inversion, and training/estimation code; imports supporting modules from local `lib/`.
- `apply_quadratic_captures_5_6.py`: independent bounded inversion of every valid detection in captures 5 and 6, with state CSVs and trace plots.
- `compare_gaze_corrections.py`: capture-1 linear baseline and capture-5/6 comparison CSVs and plots.
- `experiments/captures_5_6_direct/` and `experiments/gaze_linear_vs_corrected/`: copied latest baseline outputs.
- `requirements.txt` and `README.md`: dependencies and basic run instructions.

The package contains detected data rather than source videos. Detector implementation, detector configuration, historical experiment trees, training checkpoints, and the original training report were not copied. They remain in the source workspace. Model provenance retains historical source references; these are provenance records, not necessarily runnable package paths. The local `lib/` modules support the copied code and do not change the exp5 triangle measurement definition.

The baseline scripts were adapted to package-local paths. Existing results were copied rather than regenerated during packaging; an isolated execution of the packaged workflow has not yet been performed.

From this folder, the existing workflow is:

```bash
python -m pip install -r requirements.txt
python apply_quadratic_captures_5_6.py
python compare_gaze_corrections.py
```

These commands regenerate the default output directories. In the original managed workspace, prefix shell commands with `rtk` as required by its instructions, for example `rtk proxy python compare_gaze_corrections.py`.

## Frozen three-point estimator

For each frame, the two observables are:

```text
d   = (mean(P4x) - mean(P1x)) / sqrt(area(P1))
rho = area(P4) / area(P1)
area = abs(2D cross product of two triangle edges) / 2
```

`measurements()` requires finite coordinates, all three `p4_found` flags, and a numerically nondegenerate P1 triangle. It does not impose a pupil gate. Although `p1_valid` is stored in the data, the joint measurement validity rule uses finite P1 geometry rather than explicitly gating on that flag. The separate linear calibration requires all P1 and P4 flags; preserve or report this support distinction in comparisons.

The model has 13 coefficients. With gaze `theta` in degrees, accommodation `A` in diopters, `t = theta/15`, and `L = log(1+A)`:

```text
d   = (b0+b1*A) + (s0+s1*L)*t + (c20+c21*L)*t^2 + c3*t^3
rho = (r0+r1*t+r2*t^2) + L*(r3+r4*t+r5*t^2)
```

This is an empirical forward model. Its logarithmic accommodation coordinate and gaze polynomial are modeling choices, not universal optical laws. The inverse uses bounded multistart Gauss–Newton independently per frame, with `theta ∈ [-20,20]` and `A ∈ [0,6]`. Equivalent minima are resolved by choosing the smallest accommodation, then gaze; diagnostics retain ambiguity information.

Training used the central 80% of 20 reviewed fixation intervals from captures 1–4. After rejecting 55 high-confidence jump observations, 71,729 observations remained. The quadratic stage converged in 17 accepted iterations. Its inverse on the training core had no ambiguous minima or bound hits; 7,463 estimates were flagged as extrapolations. The later robust stage did not converge under its weight tolerance. **Use the copied quadratic model as the baseline.** Earlier full-interval fits are superseded.

Nominal fixation gaze and accommodation demand are calibration anchors, not independent framewise ground truth. Two optical channels inverted into two states can have near-zero residuals without establishing physiological accuracy. Computational bounds, training support, and physiological limits are different concepts.

## Captures 5/6 and current comparison plots

Every valid detection was inverted without an application jump filter:

| Capture | Total rows | Valid estimates | Invalid rows |
|---|---:|---:|---:|
| 5 | 26,370 | 25,226 | 1,144 |
| 6 | 35,001 | 34,080 | 921 |

Invalid rows remain flagged in the state CSVs. Captures 5 and 6 have no reviewed gaze/accommodation target intervals. Existing optional cache lookup refers to an earlier `experiments/captures_5_6_quadratic/` directory that is not bundled; its absence causes the application script to compute the inverse directly.

The linear gaze baseline averages raw displacement `mean(P4x)-mean(P1x)` over each of the five full stored capture-1 intervals, then fits one equally weighted mean point per target `[-10,-5,0,5,10]` degrees:

```text
theta_deg = 0.04838772198701595 * displacement_px - 2.5061171953292125
R² = 0.9999773085483123
```

The figures use two rows sharing frame index: gaze above, signed difference below. Both angular axes use arcminutes (`1 degree = 60 arcminutes`). Difference means **linear gaze minus joint-model gaze**, an estimator discrepancy rather than reference error. The difference display is fixed to `[-40,+40]` arcminutes; larger values remain in the CSV. Current fitted gaze limits are `[-780,150]` for capture 5 and `[-750,210]` for capture 6.

A common plot mask excludes deviations from a 501-frame local median exceeding 1 degree in either gaze estimate or 0.5 degree in their difference. It excludes 172 capture-5 frames and 705 capture-6 frames. Missing values are interpolated only for median-based detection; retained plotted values are unsmoothed. Comparison CSV angles remain in degrees and include `plot_outlier` and `included_in_plot`. This display mask does not change state estimates and is distinct from the training jump filter.

## Full-position data and correspondence

Detection payloads contain `meta` and `arrays`:

- `p1_xy`, `p4_xy`: `(n,3,2)` pixel coordinates, x right and y down.
- `p1_valid`, `p4_found`: `(n,3)` flags.
- `frame_index`, `timestamp_ms`: frame identifiers and timestamps.
- Geometry diagnostics: `sim_scale`, `sim_angle_deg`, `sim_rms`, `triangle_angle_error_deg`, `triangle_edge_scale_error_fraction`.
- Raw candidates: `p4_candidate_xy`, `p4_candidate_offsets`; frame `i` uses `[offsets[i]:offsets[i+1]]`. Equivalent P1 candidate arrays are stored.

P1 points are sorted by x, while P4 points occupy detector-selected slots. **Read each payload's correspondence metadata before constructing point features. The recorded mapping is `pair_index = [2,1,0]`: P1 point `i` corresponds to P4 point `pair_index[i]`.** Centroid/area measurements are invariant to this ordering; individual-point measurements are not.

The source detector already uses geometric constraints and middle-point history to select P4. Triangle consistency computed from selected detections may therefore reuse detector assumptions. Raw candidates can help investigate alternatives and selection failures, but they do not provide an independent physiological reference. Source `exp5/detect.py` and `exp5/p1p4_geometry.py` are outside this package and may be consulted or copied if a later task requires detector changes.

## Principles retained from the older theory

### Extra coordinates versus extra information

In the older horizontal two-pair geometry, relative measurements `[m,S1,S4]` contain three scalar differences after removing common translation. Normalized crossed-pair displacements are an invertible rewriting of `[m/S1,S4/S1]`; they add no independent normalized channel. Retaining `S1` adds a magnitude channel, whose interpretation depends on scale assumptions.

Exp5 has three points per reflection in two dimensions. Its triangle area ratio is not the old pair-separation ratio, and the old crossed-pair identities must not be substituted for triangle measurements. Nevertheless, the same rank principle applies: adding every edge or pairwise displacement can produce redundant features without adding information. Declare the feature basis, its covariance, and the nuisance transformations it removes.

The current exp5 baseline discards vertical centroid displacement and most triangle orientation/shape information. Retaining those components is a candidate source of information and residual checks; it does not by itself prove that they independently constrain gaze/accommodation.

### Normalization and nuisance geometry

Under a shared positive isotropic image scale and common additive translation, centroid displacement scales linearly, triangle area scales quadratically, and the current `d` and `rho` are invariant. This does not establish invariance to perspective, differential magnification, physical eye motion, detector bias, or rotation of the image axes. Horizontal displacement remains tied to the x-axis.

Start full-position features in a P1-relative frame, retain the P1 geometry, and compare translation/size normalization with and without rotation normalization. Rotation may remove nuisance motion or useful state signal; evaluate that choice rather than assuming either outcome. Small within-fixation P1 size variation does not establish known or constant scale.

If additional scale or pose states are fitted, they consume information. For example, a free framewise scale can absorb a magnitude channel that would otherwise provide a residual constraint. A reference convention fixes a global scale gauge but does not identify framewise scale. Inspect whitened state information after allowing nuisance variation:

```text
Jx_w   = L Jx
Jeta_w = L Jeta
I_state_given_nuisance = Jx_wᵀ (I - Jeta_w Jeta_w⁺) Jx_w
```

Here `L` whitens residuals, `Jx` differentiates gaze/accommodation, `Jeta` differentiates nuisance states, and `+` denotes the pseudoinverse. Assess rank and conditioning using declared degree/diopter increments. Local rank does not establish global inverse uniqueness or physiological accuracy.

### Shared noise and honest residual checks

Coordinates and derived differences sharing P1/P4 points have correlated localization noise. For a linear difference operator `D`, propagate coordinate covariance as `Sigma_features = D Sigma_points Dᵀ`; for nonlinear normalized features, use an appropriate Jacobian approximation or another declared covariance model. Estimate preprocessing and noise using training data only. Adjacent-frame differences can include eye motion and detector variation as well as localization noise.

An invertible feature-basis change with correctly transformed covariance preserves the measurement residual. Treating redundant differences as independent incorrectly multiplies their weight; a covariance ridge does not create new information. Likewise, robust losses and temporal priors stabilize estimation through assumptions and do not remove systematic model bias.

Leave-one-point-out checks must truly exclude the held-out point from every feature used to estimate the state. A centroid or area involving that point leaks it into the estimate. A subset must still identify the states and any fitted nuisance variables; if it does not, report that limitation. Agreement forced by choosing accommodation or scale is not independent confirmation, and shared biases can survive agreement.

### Calibration, capacity, and validation

Changing the number of channels, coefficients, priors, training demands, or preprocessing can all change estimates. Compare added geometry under matched training support and declared model capacity. Freeze preprocessing within each fold and keep withheld rows out of covariance estimates, initialization policies derived from data, and prior construction. Reusing a previous estimate as a penalty introduces model-derived evidence and requires explicit sensitivity analysis.

Distinguish numerical convergence, statistical identification, and physiological validation. Preserve unsuccessful fits and report ambiguity, bounds, extrapolation, coverage, and failure rates. Nominal-label agreement and fixation stability are useful internal evidence; independent gaze/accommodation references are needed to claim physiological accuracy. Captures 5/6 support unlabeled trajectory and geometric consistency checks.

The historical full-information work illustrates these limits: accepted normalized cases and incomplete full-relative/scale-uncertainty cases did not establish superiority of extra channels. Capture, demand, time, and drift were confounded, and some constrained mean probes remained numerically unverified. These are historical cautionary examples, not exp5 results or pending tasks in this package. Its experimental solver changes and unresolved probes are not prerequisites for the exp5 geometry work.

## Proposed next work and decision criteria

The full-position estimator has **not been implemented or evaluated**. The immediate deliverable is an auditable feature/diagnostic prototype and a comparison with the frozen baseline, followed by a decision about which additional geometry supports estimation or error detection.

1. Audit correspondence and trajectories on reviewed fixations. Plot each P4 x/y coordinate, its corresponding P1-relative displacement, and detector diagnostics; investigate switches, missing points, and raw candidate alternatives.
2. Construct a compact full-position feature basis. Compare normalization choices, preserve P1 geometry for nuisance assessment, and separate common P4 displacement from triangle deformation. Candidate components include vertical displacement, edges, scale, rotation, and shape residuals.
3. Assess feature variability, redundancy, covariance, and state/nuisance sensitivity before selecting forward-model capacity. Add nuisance states only when evidence supports their identifiability.
4. Fit a forward model predicting the retained geometry from gaze/accommodation, then perform bounded inversion with branch, support, and conditioning diagnostics. Preserve the existing baseline and write new outputs separately.
5. Evaluate per-point prediction residuals and genuinely withheld-point predictions where identifiable. Determine whether disagreement localizes detection failures or supports useful uncertainty/rejection flags.
6. Compare against the frozen baseline using withheld fixations/captures and matched support. Report nominal agreement, fixation spread, held-out point residuals, inverse ambiguity, bound hits, extrapolation, failure rates, and sensitivity to nuisance/model assumptions. Use captures 5/6 for unlabeled consistency checks.

Reduced linear-versus-joint discrepancy, lower training residuals, or better nominal-anchor agreement alone must not select the new estimator. The decision should identify which added components provide reproducible predictive information or useful detection diagnostics, and which conclusions remain unsupported without new independently referenced acquisitions.
