# Independent scientific audit: shared template size

**PASS.** The paired experiment adds exactly one shared log-size coordinate to Stage05. Stage01–04 calibration, frozen physical P1/P4 scale `g`, capture radii, template centroid, accommodation reference, state constraints and optical prior targets are identical to the imported controls. Stage04 was deliberately kept fixed to isolate the added Stage05 degree of freedom. The composition remains forward `D(K(B))`, inverse barrel first then keystone. There is no size prior, second frame scale, or output normalization. Size bounds `[0.5, 2]` are numerical limits, not physical constraints.

## Artifact and numerical checks

All four production/training fits have two valid, stationary starts. Scientific Python hashes agree across fits and held/profile outputs; saved snapshots, arrays hashes, calibration input hashes, exact masks, `g`, and radii pass. Original inputs and completed sensitivity controls were preserved. Native predictions were independently reconstructed with separate NumPy pixel equations (maximum difference `1.14e-13 px`); inverse results reconstructed the observed native pixels within `3.98e-13 px`. The centroid and the relation `learned_energy = initial_energy * size_ratio²` pass. No final optical or size bound is active; all final forward/inverse rows are valid.

Production size ratios are `0.984661355` (.5 D/2x priors) and `0.987430605` (.25 D/1x). Training ratios are `0.983715550` and `0.986699509`. Across each pair of starts, loss spread is at most `8.48e-12`, and size spread at most `1.05e-7`. This establishes reproducibility of the tested anchored minima, not global uniqueness or physical parameter accuracy. Production accommodation lower-bound contacts increase from 3 to 28 for the primary comparison, and from 5 to 12 for robustness; training free-size contacts are 8 and 7.

The local data-only global rank is 17/18 after unconstrained frame gaze/accommodation projection. The slope/accommodation scale-null remains. Under the declared characteristic units, the weakest nonnull mode is predominantly log size (absolute scaled-coordinate loading about 0.999; singular value approximately 0.0017–0.0020). This audit excludes all priors and bounds, so it does not establish identifiability of physical accommodation or size.

## Conditional size profile

All four profile points meet the free nuisance-global and state certificates with zero invalid rows. Size is fixed at each point; its unconstrained gradient is reported separately and is excluded from the conditional certificate. Size one reproduces the exact immutable matched fixed-size optimum within `6.8e-15` objective units.

| Size ratio | Data loss | A-mean penalty | Optical penalty | Total loss |
|---:|---:|---:|---:|---:|
| 0.979738048 | 0.705153 | 0.024975 | 0.065477 | 0.797102 |
| 0.984661355 | 0.710117 | 0.028799 | 0.045622 | 0.785977 |
| 0.989584662 | 0.722494 | 0.044883 | 0.030204 | 0.798958 |
| 1.000000000 | 0.772210 | 0.146184 | 0.010447 | 0.930064 |

The small gaze penalty completes each total. The lower-size profile point fits the data better than the selected size, but incurs a larger optical penalty. Therefore the fitted size minimizes the **anchored objective**, and is not the data-only preferred reference size. This is a local nuisance-refit profile, not a confidence interval or a data-only likelihood profile.

## Matched temporal held-vertex validation

Training uses 66,268 rows; scoring uses exactly 18,139 test rows and 54,417 held-vertex scores per case. All 12 conditional state certificates were independently recomputed and pass (largest characteristic state gradient `1.63e-5`, zero invalid rows). Training artifact/source hashes and test source IDs match. Gaze anchors are training means only; scale is profiled from P1 alone at those anchors. State inference uses only two retained native P4 vertices, centered on their own centroid. Scores use the omitted raw vertex relative to the retained centroid. Native residual reconstruction differs by at most `1.71e-13 px`; CSV score reconstruction differs by at most `1.13e-13 px`. Initial and Stage04-fixed-state controls match the imported controls exactly to numerical tolerance.

The scientifically paired comparison is **each fitted joint fixed-size model versus its fitted joint free-size model**, not the internal Stage04-fixed-state diagnostic:

| Prior setting | Joint fixed-size held RMS | Joint free-size held RMS |
|---|---:|---:|
| .5 D / 2x optical | 1.640546 px | 1.501476 px |
| .25 D / 1x optical | 1.870972 px | 1.579695 px |

All four capture averages improve, but improvement is not uniform. Primary has 5/20 capture/gaze cells worse: C1 +5 (`+0.10838 px`), C1 +10 (`+0.23958`), C2 −10 (`+0.10407`), C4 zero (`+0.05234`), C4 +5 (`+0.02782`). Robustness has 4/20 worse: C1 +5 (`+0.49625`), C1 +10 (`+0.31082`), C4 zero (`+0.09360`), C4 +5 (`+0.09092`). These are equal-held-vertex cell RMS differences; individual omitted vertices can also regress.

## Accommodation and model separation

Across the two prior settings, the equal-20-cell RMS change in accommodation means falls from `0.09742 D` for fixed size to `0.04224 D` for free size. Removing each capture's constant shift gives `0.07633 → 0.04186 D`. This supports greater robustness to this joint accommodation/optical prior-width change; it does not isolate which prior caused the difference.

Within each fitted model, gaze dependence remains. Primary free-minus-fixed accommodation span changes for C1–C4 are `−0.1780, +0.1351, −0.1219, −0.1956 D`: C2 worsens. The robustness free model also has a wider C2 span. Lower held pixel error is not evidence that accommodation has become gaze independent or physically correct.

At fixed-control gaze and the same `g`/radii, an exact three-block native Shapley decomposition for the primary comparison gives pooled component RMS: keystone/shared offset `4.106 px`, template shape plus size `4.806 px`, radial coefficient `1.268 px`. Keystone/template cosine is `−0.961`; their combined geometry change is `1.422 px` and has cosine `−0.977` with the radial change. The net pixel change is only `0.328 px`; the remaining fitted-gaze change contributes `0.00235 px`. Every counterfactual map is domain valid. Thus substantial compensating reference/keystone/radial changes persist. This conditional decomposition does not prove physical keystone misspecification or justify separate P1/P4 physical magnification.

The test supports freeing a shared nuisance reference size for prediction and prior robustness. It does not resolve physical separation of template, capture keystone and accommodation. No immutable active result was replaced or promoted by this audit.
