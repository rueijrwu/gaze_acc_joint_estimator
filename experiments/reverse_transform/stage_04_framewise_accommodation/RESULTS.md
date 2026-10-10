# Stage 04: framewise accommodation fit

This run fits one accommodation value (A_i) per complete frame while holding the Stage 03 gaze, per-capture keystone coefficients, P1 magnification, P4 reference, and post-fit linear barrel law fixed. The objective averages inverse three-vertex squared distance equally by fixation and adds a soft penalty on each fixation's mean (A), with width 0.25 D. This changes only the strength of the fixation-mean anchor; there are no framewise anchors and no barrel-law refit. A is bounded by 0–6 D, additionally limited per frame when required to stay on the monotone inverse branch. All 89,175 complete frames were included. No P4 magnification, rotation, temporal smoothing, or radius normalization was fitted.

The shape-size convention internal to the frozen keystone operator is retained from Stage 03. The error figures below use the recovered vertex coordinates directly; they do not rescale recovered triangles to a common radius.

## Fit and audit

The CuPy float64 run on a Tesla P100 completed in 6.72 s. The objective was 8.805618 px²: 8.178932 data error plus 0.626686 anchor penalty. The independent CPU audit passed. Its projected-gradient infinity norm was 1.25e-11, minimum free-frame curvature was 25.212, and maximum all-frame gradient relative error was 6.62e-8. Forward/inverse synthetic closure was 1.71e-12 px; replay of the frozen parent inverse agreed to 5.97e-13 px.

There were 965 frames at the lower A bound, none at the upper bound, and 47 frames limited by the inverse-domain boundary. The saved certificate reports a stationary constrained solution. The largest fixation-mean shift from its expected anchor was +0.444 D at capture 3, gaze −5°. Capture 1 had 918 lower-bound frames. These are relevant constraints on interpreting the fitted values.

## Inverse point errors

The first table compares the parent capture-constant model, the frozen nominal A law, and framewise A on the same 89,129 frames where all three inverses are valid. Median and P95 are pooled over the three vertex distances per frame. RMS is computed from those same point distances.

| Capture | Common frames | Parent constant κ RMS (px) | Frozen nominal A RMS (px) | Framewise A RMS (px) | Framewise median / P95 (px) |
|---|---:|---:|---:|---:|---:|
| 1 | 24,280 | 1.698 | 1.698 | 1.184 | 0.998 / 1.932 |
| 2 | 21,980 | 2.469 | 2.481 | 1.984 | 1.661 / 3.664 |
| 3 | 19,795 | 2.577 | 2.714 | 1.994 | 1.698 / 3.387 |
| 4 | 23,074 | 3.311 | 3.318 | 3.010 | 1.926 / 4.532 |

The common-valid set omits 43 parent inverse failures and 46 frozen-law inverse failures. The fit still included all 89,175 complete frames, and it produced no invalid inverse. To show the effect of those outliers, framewise inverse RMS over **all** complete frames is 1.184, 4.149, 1.994, and 3.010 px for captures 1–4. Capture 2's all-frame RMS is much higher than its common-set RMS because it retains rows that failed the older inverses.

As a secondary check in the original camera-coordinate forward direction, parent residual RMS over all complete three-vertex points was 1.702, 4.739, 2.244, and 3.112 px (captures 1–4). Framewise-A forward residual RMS was 1.186, 3.999, 1.774, and 2.869 px. The improvement is also visible in the original camera coordinates.

## Fixation mean anchors

Each entry shows fitted mean (A) ± within-fixation SD, in D. The expected soft-anchor means are 0.360 D for capture 1, 4 D for capture 2, 3 D for capture 3, and 2 D for capture 4. The full table makes the sizeable fixation-dependent shifts visible despite the anchor penalty.

| Capture | −10° | −5° | 0° | +5° | +10° |
|---|---:|---:|---:|---:|---:|
| 1 | 0.566 ± 0.078 | 0.592 ± 0.102 | 0.359 ± 0.060 | 0.196 ± 0.147 | 0.207 ± 0.082 |
| 2 | 3.911 ± 0.435 | 4.095 ± 0.233 | 4.010 ± 0.088 | 4.139 ± 0.138 | 3.651 ± 0.131 |
| 3 | 3.164 ± 0.101 | 3.444 ± 0.271 | 3.103 ± 0.119 | 3.017 ± 0.077 | 2.695 ± 0.206 |
| 4 | 2.069 ± 0.168 | 2.186 ± 0.144 | 2.045 ± 0.127 | 1.893 ± 0.155 | 1.686 ± 0.128 |

In all four captures, the +10° fixation mean is below its expected anchor. The gaze-associated mean shifts show that framewise A can absorb remaining gaze-conditioned size effects. These values are not independent measurements of physiological accommodation.

## Figures

- [Inverse RMS, median, and P95 by fixation](results/run/inverse_error_by_fixation.png): common-valid frames; all three models use the same population.
- [Framewise inverse error over all complete frames](results/run/framewise_error_all_complete.png): preserves prior inverse-failure outliers.
- [Triangle means and vertex clouds](results/run/triangle_means_and_vertex_clouds.png): unnormalized recovered coordinates, common-valid frames.
- [A distribution by fixation](results/run/accommodation_by_fixation.png) and [A by source row](results/run/accommodation_by_source_row.png): row points are not joined across gaps or reversed source order.
- [Remaining radius variability](results/run/remaining_radius_variability.png): recovered-to-reference RMS radius diagnostic; no radius correction is applied during fitting.

See the [independent scientific review](SCIENTIFIC_REVIEW.md) for numerical interpretation and limitations. The main limitation remains that one fitted A per frame is an in-sample accommodation parameterization; it does not establish physiological accuracy or distinguish accommodation from other unmodeled size changes.
