# Phase 8.2 audit follow-up: boundary transitions and state gains

The table compares `anchor_strong/conditional37` against baseline conditional37 on exact shared scored P4 slots. Each value is candidate minus baseline in squared vector error (px²); negative values indicate lower squared error. Counts are paired point slots. Paired complete-frame counts are shown separately.

| Evaluation family | Boundary transition | Paired slots | Mean Δ squared error (px²) | Median (px²) | Paired complete frames |
|---|---|---:|---:|---:|---:|
| Gaze | interior → interior | 339 | -2.26 | +0.06 | 107 |
| Gaze | interior → bound | 11 | -255.14 | -25.76 | 3 |
| Gaze | bound → interior | 61 | -65.49 | -33.12 | 17 |
| Gaze | bound → bound | 18 | -102.69 | -52.98 | 16 |
| Capture | interior → interior | 391 | +0.06 | +0.05 | 126 |
| Capture | interior → bound | 0 | — | — | 0 |
| Capture | bound → interior | 8 | +16.84 | +12.31 | 6 |
| Capture | bound → bound | 30 | -6.18 | -7.08 | 11 |

Gaze-family gains are concentrated in transitions involving a boundary. Capture-family interior-to-interior error is nearly unchanged, and the bound-to-interior stratum worsens. These results do not support a uniform improvement claim; they are descriptive comparisons on reused development folds.

Positive state-scale gains were fit on matched training trajectories only. Across five gaze folds, gaze gain ranged **0.9898–1.0258** and `1+A` gain **0.8871–1.0547**. Across four capture folds, gaze gain ranged **0.9788–0.9999** and `1+A` gain **0.9766–1.2170**. They are descriptive training-only gains; evaluation states were not rescaled.

`paired_transition_membership.jsonl.gz` preserves every scheduled point/frame identity and paired delta. `summary.json` contains fixation, signed-gaze, capture, transition, frame, and training-gain strata. `verify_results.py` independently checks all source hashes, membership totals, transition counts, and arithmetic summaries without rerunning fits.
