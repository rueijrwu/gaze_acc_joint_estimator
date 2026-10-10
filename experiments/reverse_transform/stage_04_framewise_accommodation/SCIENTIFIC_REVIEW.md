# Scientific review: framewise accommodation with fixation-mean anchors

## Contract and numerical audit

The stage freezes stage 03's capture-specific keystone coefficients, estimated gaze, framewise P1 magnification, empirical P4 reference, and post-fit linear law:

`kappa(A) = -5.25521030136462e-7 * (A - 0.36036036036036034)` in px^-2.

Only accommodation A varies per frame. The objective is an equal-fixation average of the three recovered vertex squared distances to the reference, plus an equal-fixation mean-anchor penalty. Each of the 20 fixation means is softly anchored to its capture's nominal demand. The user selected a 0.25 D width; the declared 1 px residual scale gives penalty strength 16 px²/D². There is no framewise anchor and the barrel slope is not refitted. This width is a fitting choice, not a measured uncertainty or hard bound. A uses the operational range [0,6] D, with individual upper limits only where needed to stay on the valid inverse branch.

There is no fitted P4 magnification or temporal smoothing. Measured/recovered triangles are not radius-normalized in the objective. The deterministic keystone size convention inherited from stage 03 remains in the model, preserving barrel's size change. The trial-dependent model centroid and keystone normalization are recomputed for every trial A before inversion; a constant precomputed inverse-keystone triangle is not substituted for the full inverse.

All 89,175 complete frames are fitted, with no residual trimming. All final inverses are valid. The saved-results audit independently reproduces the frozen inputs, fitted inverses, fixation means, objective and constraints on NumPy, checks analytic gradients against finite differences for all frames, and verifies synthetic closure. Its projected gradient is below 1.3e-11, and the minimum free-frame observed curvature is 25.21 px²/D². The positive frame curvature plus positive semidefinite mean-anchor blocks certifies positive curvature on the free subspace. Three starts agree in the objective to numerical precision. This is a local numerical certificate, not a global or physiological claim.

## Error reduction

The common comparison set has 89,129 frames: the stage 03 constant-barrel baseline has 43 invalid inverses, and the frozen linear-law nominal-A baseline has 46. Each comparison uses precisely the same frames. RMS is the square root of the mean squared two-dimensional corresponding vertex distance, pooled over three vertices and frames; no uniform size or rotation is removed from this score.

| Capture | Constant-barrel inverse RMS | Framewise-A inverse RMS | Original-camera RMS, constant → framewise A |
|---:|---:|---:|---:|
| 1 | 1.698 px | 1.184 px | 1.702 → 1.186 px |
| 2 | 2.469 px | 1.984 px | 4.739 → 3.999 px |
| 3 | 2.577 px | 1.994 px | 2.244 → 1.774 px |
| 4 | 3.311 px | 3.010 px | 3.112 → 2.869 px |

The inverse columns use the common valid comparison set. The original-camera columns use all complete frames, including previously invalid inverse observations. Both coordinate views improve in all four captures. However, capture 2's fitted inverse RMS over *all* complete frames is 4.149 px; its common-set 1.984 px does not characterize its outlier-sensitive full population. The equal-fixation inverse vertex MSE over all fitted frames is 8.17893 px², with mean-anchor penalty 0.62669 px². The total objective is 8.80562 px².

## Accommodation interpretation and remaining limits

There are 965 fitted frames at A=0 D: 918 in capture 1, 46 in capture 2, none in capture 3, and one in capture 4. None reaches an upper bound. The inverse domain restricts 47 individual upper limits. About 3.8% of capture 1 therefore has a clipped accommodation estimate; its apparent fluctuation distribution cannot be interpreted without this qualification.

The largest fixation-mean shift is +0.44371 D at capture 3's −5° period. The mean anchors permit this deviation because they are soft. Mean A also retains a gaze-associated pattern: capture 1's −5° and +10° means are 0.592 and 0.207 D; capture 3's are 3.444 and 2.695 D. Every capture's +10° mean lies below its expected demand. These observations demonstrate that framewise A can absorb remaining gaze-conditioned size discrepancies; they do not establish that all recovered changes are physiological accommodation fluctuations.

The κ law remains empirical and demand/capture-confounded, the origin is an empirical convention, and capture 4 inherits its bounded vertical keystone coefficient. Adding one state per frame can reduce an in-sample residual. The original-camera improvements are useful geometric evidence, but there is no independent framewise accommodation ground truth or held-out accuracy certificate in this stage. No automated test suite was added or run; verification consists of the requested experiment and saved-results audit.
