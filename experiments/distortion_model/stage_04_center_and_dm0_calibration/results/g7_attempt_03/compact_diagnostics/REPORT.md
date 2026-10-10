# G7 saved compact-check aggregation

This report aggregates already-saved compact JSON records only. It does not run inference, refit a model, or turn the compact diagnostic into G8. All scheduled slots, missing outcomes, and endpoints are retained.

Attempt: `/home/aplab/ACC/experiments/distortion_model/stage_04_center_and_dm0_calibration/results/g7_attempt_03`

## Input files

| File | SHA-256 | Slots | Scored | Unavailable | Unresolved |
|---|---|---:|---:|---:|---:|
| `continued_final.json` | `da9047b2d6252e49795a52c4c04fc17cb7b7d5e9eed7712030d02bf6f84d5a7b` | 300 | 270 | 30 | 0 |
| `continued_outer_04.json` | `4052bd60d45cf55c791c16920414e17fb54b1705a6db9b5ed5f1e0cfc9b6047b` | 300 | 270 | 30 | 0 |
| `continued_outer_08.json` | `da9047b2d6252e49795a52c4c04fc17cb7b7d5e9eed7712030d02bf6f84d5a7b` | 300 | 270 | 30 | 0 |

## Complete held-point triples

For each complete eligible frame, `E_i²` is the mean of the three squared 2D native-pixel errors. `Gtheta_i²` and `GA_i²` are the mean of the three pairwise squared state differences. Exposure means are computed on squared quantities and then equally weighted; an absent exposure makes the all-exposure summary incomplete.

| Compact file | Eligible triples | Exposures present | Missing exposures | Equal-exposure E (px) | Gtheta (deg) | GA (D) |
|---|---:|---:|---:|---:|---:|---:|
| `continued_final.json` | 90 | 20/20 | none | 41.8138 | 1.50113 | 1.31867 |
| `continued_outer_04.json` | 90 | 20/20 | none | 41.8138 | 1.50113 | 1.31867 |
| `continued_outer_08.json` | 90 | 20/20 | none | 41.8138 | 1.50113 | 1.31867 |

## Matched checkpoint coverage

Each checkpoint is compared with its same-start final snapshot on common eligible slot/frame IDs. Coverage gains/losses are reported separately from matched residual changes.

| Start | Checkpoint | Common slots | Lost | Gained | Common frames | Mean ΔE² (px²) |
|---|---|---:|---:|---:|---:|---:|
| continued | `continued_outer_04.json` | 270 | 0 | 0 | 90 | 7.336850768751507e-07 |
| continued | `continued_outer_08.json` | 270 | 0 | 0 | 90 | 0.0 |

## Interpretation limits

The subset state spreads `Gtheta` and `GA` measure internal agreement among overlapping retained-point inversions, not accuracy. Errors use native relative pixels and are not divided by inferred scale. This compact schedule is a progress diagnostic, not the full G8 evaluation. The slot and frame records in `summary.json` preserve the identities and outcomes behind these tables.
