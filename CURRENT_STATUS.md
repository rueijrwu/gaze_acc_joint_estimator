# Current status: shared P4 template geometry

The active free-size comparison is complete and independently audited in [the active family](experiments/p1_scale_barrel_first_free_size/). It improves conditional P4 geometry on the held split relative to its paired fixed-size controls. It does not identify physical magnification or accommodation, and the fixed-centered-energy fit remains the historical reference.

## Data and estimator

- The reviewed cohort contains 88,978 matched P1/P4 frames across four captures and five fixation intervals. Capture demand labels are 0.36036, 4, 3, and 2 D.
- Stage 01 estimates horizontal gaze per capture from a polynomial fit to fixation-mean P4−P1 centroid displacement. Stage 02 fits bounded P1-only frame scale `g∈[0.98,1.02]`; this exact physical `g` is fixed and shared by P1 and P4.
- Stage 03 calibrates barrel-first relative coefficients per capture/fixation; Stage 04 fits capturewise keystone after barrel inversion. Stage 05 jointly fits state and optical parameters using the same barrel-first composition.
- The final observation model is `Xhat4 = g × center(Dκ(K4(Bλ, θ)))`, with `κ=s(A−Aref)`, `Aref=0.36036036 D`, and `Bλ = fixed_centroid + λ × centered_shape`.
- The centroid, capture radii, Stage 01–04 calibration and matched priors remain fixed between each pair. The one added shared `λ∈[0.5,2]` is a numerical template-geometry coordinate, not physical magnification. There is no size prior, extra P4 scale, output normalization or post-fit row filter.

## Paired results

| Anchor / optical width | P4 forward / inverse all-row weighted RMS, fixed → free (px) | Held weighted RMS, fixed → free (px) | Free λ | A lower-bound contacts, fixed → free |
|---|---:|---:|---:|---:|
| Primary: 0.50 D / 2× | 0.818739 / 0.984475 → 0.739213 / 0.924861 | 1.640546 → 1.501476 | 0.984661 | 3 → 28 |
| Robustness: 0.25 D / 1× | 1.059744 / 1.262006 → 0.844091 / 1.047138 | 1.870972 → 1.579695 | 0.987431 | 5 → 12 |

Each pair used 66,268 training rows, 4,571 gap rows and 18,139 test rows, yielding 54,417 omitted-vertex scores. The 12 held-state certificates passed with zero invalid rows. All four capture-average held scores improve, but 5/20 primary cells and 4/20 robustness cells regress. The largest changes are primary C1 +10° (+0.240 px) and robustness C1 +5° (+0.496 px); see the [cell-level table](experiments/p1_scale_barrel_first_free_size/report/held_cell_changes.csv).

Fitted accommodation remains gaze-dependent. For the free primary fit, mean A near nominal +10° is below mean A near zero in C2–C4, but not C1:

| Capture | Mean fitted θ near 0° | A near 0° (D) | Mean fitted θ near +10° | A near +10° (D) | Difference +10° minus 0° (D) |
|---|---:|---:|---:|---:|---:|
| C1 | 0.0° | 0.30258 | 9.98° | 0.37653 | +0.07395 |
| C2 | 0.0° | 3.99433 | 10.00° | 3.71759 | −0.27675 |
| C3 | 0.03° | 3.17770 | 9.92° | 2.79286 | −0.38484 |
| C4 | 0.01° | 2.04561 | 9.95° | 1.77502 | −0.27059 |

These are fitted state means by nominal fixation, not a separate residual-bias measurement. Nominal gaze, capture/demand, and fixation order are confounded; the pattern is not causal evidence.

Across the two jointly changed anchor/optical settings, fixed/free RMS change across 20 fixation-mean A values falls from 0.09742 to 0.04224 D, and from 0.07633 to 0.04186 D after removing each capture offset. This is sensitivity to jointly changed settings, not an isolated anchor-width effect.

## Identification and interpretation

The primary conditional size profile chooses λ=0.984661 for the anchored objective. At λ=0.979738, data loss is lower (0.705153 vs 0.710117) but optical penalty is higher (0.065477 vs 0.045622); total loss is 0.797102 vs 0.785977. The selected size is therefore not a data-only estimate or confidence interval. The local data-only rank is 17/18 and the slope/accommodation scale-null remains. The results support a conditional geometry prediction sensitivity only; they do not establish physical separation of template size, keystone and accommodation.

See the [canonical experiment record](experiments/RECORD.md), [current plan](CURRENT_PLAN.md), [comparison artifact index and plots](experiments/p1_scale_barrel_first_free_size/report/REPORT.md), and [independent audit](experiments/p1_scale_barrel_first_free_size/audits/AUDIT.md). The complete pre-cleanup root documents and previous family narratives are preserved under [the cleanup backup manifest](experiments/archive/reverse_transform/cleanup_2026_10_12/PRE_CLEANUP_BACKUPS.json). Moved paths and byte-verified file hashes are recorded in the active family [relocation manifest](experiments/p1_scale_barrel_first_free_size/provenance/relocation_manifest.json). Historical experiment families are preserved under [the historical-family archive](experiments/archive/reverse_transform/historical_families/).
