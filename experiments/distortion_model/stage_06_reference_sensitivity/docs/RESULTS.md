# Stage 6 results

**Status: STOP_UNCERTIFIED_ADJACENT.** The control candidate completed reviewed G7 calibration and full frozen G8 evaluation with `GO_WITH_LIMIT`. The adjacent candidate failed the unchanged calibration certificate and was not evaluated by G8. The comparison is incomplete; no optical ranking is available.

## Calibration gate

| Candidate | Full objective J | State stationarity | Global KKT | Complementarity | G8 eligibility |
|---|---:|---:|---:|---:|---|
| Control | 1925.6013386760935 | 3.92058e-12 | 6.36353e-8 | 6.88043e-11 | Certified; G8 complete with limits |
| Adjacent | 1925.8105580440379 | 4.13310e-5 | 0.551904 | 1.80740e-6 | Uncertified; G8 not run |

Stationarity and complementarity thresholds remain `1e-6`. Both fits pass the physical bound, optical domains, local rank and curvature checks. Control also passes the constrained profile certificate and has two certified starts. Adjacent has no certified start, so its held-point inversions are prohibited. Feasibility and positive curvature do not replace stationarity.

## Frozen control G8

The independently audited G8 result retains all 100,090 frames and all 300,270 scheduled omission slots. Of these, 267,565 retained-input inversions are certified, 267,525 have a held measurement and are scored, and 32,705 have unavailable retained inputs. There are zero unresolved or ambiguous inversions and 89,175 complete triples spanning all twenty exposures. The forty certified but unscored slots lack held measurements. The independent audit confirms parent and source integrity and reproduces the results.

| Equal-exposure complete-triple metric | Control |
|---|---:|
| Held-point vector consistency E | 12.408526 px |
| Same-frame gaze disagreement Gtheta | 0.477866 degrees |
| Same-frame accommodation disagreement GA | 1.101718 D |

These are internal consistency measures, not truth errors or accuracy claims. The adjacent roster preserves all 300,270 expected slots with zero inference attempted. The [comparison record](../results/attempt_03/comparison.json) sets paired metrics to null and `no_optical_ranking=true`.

## Boundary diagnosis and scope

The [saved-parameter boundary inspection](../results/attempt_03/boundary_inspection/summary.json) freezes the adjacent fit's states and other globals. It finds the minimum gaze-floor row at index 0 on both sides of `alpha4=0`; the selected derivative changes from approximately `+4.9064` on the negative side to `-6.0980` on the positive side. At zero, near-active rows 0/1 are nonsmooth. The inspection performs no optimization or inference and is not a generalized stationarity certificate.

The next numerical question is whether a one-sided/generalized stationarity and curvature certificate can resolve this interval branch while distinguishing NNLS near-active rows from true complementarity. Until then, the adjacent comparison cannot be completed. Do not infer optical inferiority, physical-law failure, or a nonsmooth optimum; do not proceed to a new optical mechanism from this result. Physical zeros, absolute accommodation accuracy, noise calibration, common scale transfer and capture/demand confounding remain unresolved.

See the [root scientific review](../results/attempt_03/SCIENTIFIC_REVIEW.md) and [live gate record](PROGRESS.md).
