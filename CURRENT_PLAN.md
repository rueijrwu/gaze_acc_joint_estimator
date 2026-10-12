# Current protocol and next diagnostic

The active, completed model is the [barrel-first P4 shared-template-size comparison](experiments/p1_scale_barrel_first_free_size/). Its run-time plan is preserved at [PLAN_AT_RUN.md](experiments/p1_scale_barrel_first_free_size/provenance/plans/PLAN_AT_RUN.md); the archived pre-cleanup plan is available in the [cleanup backup](experiments/archive/reverse_transform/cleanup_2026_10_12/documents/CURRENT_PLAN.md).

## Protocol

1. Use the reviewed 88,978 matched P1/P4 frames across four captures and five fixation intervals. Capture demand labels are 0.36036, 4, 3, and 2 D.
2. Estimate horizontal gaze per capture from the fixation-mean P4−P1 centroid displacement. Fit the shared P1-only frame scale `g∈[0.98,1.02]` and keep the exact `g` fixed for both patterns.
3. Calibrate relative capture/fixation barrel coefficients first. Then fit capturewise P4 keystone using the barrel-corrected triangles. These provide the frozen Stage 03–04 calibration and matched priors.
4. Compare paired Stage 05 models that differ by one shared log template-size coordinate `η`, where `λ=exp(η)∈[0.5,2]` and `Bλ = fixed_centroid + λ × centered_shape`. Centroid, capture radii and physical `g` are fixed. The size bound is numerical; no size prior or second scale is added.
5. Jointly fit shared shape/size, capturewise keystone and shared nuisance offset, radial slope, gaze and accommodation using `Xhat4 = g × center(Dκ(K4(Bλ,θ)))`, `κ=s(A−Aref)`, and `Aref=0.36036036 D`. Primary anchors are 0.50 D / optical 2×; robustness anchors are 0.25 D / optical 1×.
6. Validate on the fixed temporal split (66,268 train, 4,571 gap, 18,139 test rows). Infer each held state from P1 and the retained two P4 vertices; score the omitted raw vertex. Preserve all reviewed rows and report invalids rather than filtering residuals.

The completed result is a conditional geometric comparison, not a physical accommodation or magnification measurement. Current metrics and limitations are in [CURRENT_STATUS.md](CURRENT_STATUS.md) and the [canonical experiment record](experiments/RECORD.md). Previous root plans and status are preserved in [cleanup documents](experiments/archive/reverse_transform/cleanup_2026_10_12/documents/).

## Next diagnostic

A useful next diagnostic is to constrain each capture’s five fixation-mean A values to one common value while still allowing frame-level A fluctuations, then refit optics and compare held geometry and radial-residual costs, with particular attention to the +10° cells. This would test model coupling; it would not assume that physical accommodation is truly constant across fixation. A recording with fixation order varied independently of capture and demand, or independent gaze/accommodation measurements, may be needed to assess causality. No additional fit is run as part of this cleanup.
