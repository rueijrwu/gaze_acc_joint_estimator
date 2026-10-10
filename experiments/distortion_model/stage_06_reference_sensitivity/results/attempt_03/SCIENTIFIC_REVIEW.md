# Stage 6 final scientific review

## Decision

**STOP_UNCERTIFIED_ADJACENT. Execution and audits are finished; the paired reference comparison is incomplete. Keep the current operational reference. Do not proceed to another optical mechanism on this evidence.**

The control calibration and its full frozen G8 cross-check pass with the existing scientific limits. The adjacent calibration does not pass the unchanged numerical gate, so its held-point inversions were not run. This is a numerical limitation of this experiment, not evidence that the adjacent reference is optically worse.

## What was implemented and run

One matched intervention changed omega4 from -10.016974132845107 to -5.0148989655383795 degrees and rebuilt the reference-specific empirical P4 template. Both conventions received fresh conditional G4/G5 fits, exact G6 initialization and two joint G7 starts. All 100,090 scheduled rows, source identities, twenty exposures, covariance, camera axes, omega1, domains, full-mean anchors, zero temporal penalty and the degree-two DM0-M1 family were retained. Both used the same numeric one-degree bound: gaze-slope floor 10.02700222585176 px/degree and accommodation-response cap 2.50675055646294 px/D.

Original 48-round outcomes are preserved under attempt_01. A separately declared continuation solved new state/global working sets with independent bordered checks and original nonlinear feasibility and objective acceptance. Its twelve-step budget per original start was retained across attempts_02/03. A second repair distinguishes actually tight primal constraints from the broad NNLS near-active census. There was no relaxation of the 1e-6 stationarity/complementarity gates. Both full public-optics reconstructions and the recorded ancestry/source audits pass.

## Calibration results

| Selected result | Current reference | Adjacent reference |
| --- | ---: | ---: |
| Certified starts | 2/2 | 0/2 |
| Full objective J, numerical diagnostic | 1925.6013386760935 | 1925.8105580440379 |
| State stationarity | 3.92058e-12 | 4.13310e-5 |
| Global KKT stationarity | 6.36353e-8 | 0.551904 |
| Complementarity | 6.88043e-11 | 1.80740e-6 |
| Required residual thresholds | 1e-6 | 1e-6 |
| Optics domains / physical bound | Pass | Pass |
| Positive free-state/profile curvature | Pass | Pass |
| Accommodation-bound calibration rows | 8,415 | 9,628 |
| Frozen G8 eligibility | GO_WITH_LIMIT | NO_GO_UNCERTIFIED |

The control objective reproduces the earlier certified optimum within 9.6e-11. The adjacent objective and descriptive training residuals cannot establish optical superiority or infer physiological accuracy while its fit remains uncertified. Positive rank, curvature and feasibility do not replace stationarity.

## Full control cross-check

The independent backend and omitted-input preflight passes on forty actual rows per omission. Huge finite replacements and unavailable NaNs at the omitted point leave every checked inverse field and predicted held point invariant. Legacy, GPU and CPU selected states agree within 2.07e-10 degrees / 1.17e-10 D on the declared audit rows.

The full control run took 196.79 seconds on the GPU. It preserves all **300,270 scheduled omission slots**: 267,565 retained-input inversions are certified, 267,525 are scored, 32,705 have unavailable retained inputs, and there are zero unresolved or ambiguous inversions. The forty unscored certified inversions lack held measurements. There are 89,175 complete triples across all twenty exposures. Of the certified slots, 41,016 are at a state bound; 40,981 of those are scored.

| Equal-exposure complete-triple metric | Control result | Change from prior full G8 |
| --- | ---: | ---: |
| Held-point vector prediction E | 12.4085261743 px | -4.49e-11 px |
| Pairwise gaze disagreement Gtheta | 0.4778664312 degrees | -3.76e-12 degrees |
| Pairwise accommodation disagreement GA | 1.1017175503 D | +4.78e-11 D |

E compares each held P4 measurement with its predicted position relative to the measured P1 mean. Gtheta and GA measure disagreement among the three omission inversions on the same frame. These are consistency measures, not ground-truth errors. The result confirms reproducibility; it leaves the earlier structured residual and accommodation-identifiability findings unresolved.

The saved independent audit reconstructs held errors, scale and retained costs, validates whole-population identities and masks, and verifies frozen parent and archived sources. Maximum retained-cost reconstruction discrepancy is 2.37e-10. No repository automated test suite was run; the checks are actual-record experiment audits.

## Why the adjacent comparison stops

Both adjacent starts approach the same objective and fail the same original-objective line search. The selected alpha4 is -4.62040e-11. Final quadratic solves remain accurate and positive on their constrained tangent; all corrected trial points are physically feasible, but all original-objective changes are positive, including 9.24e-8 at step fraction 1/2048. Rejection is warranted.

The separately declared [boundary inspection](boundary_inspection/summary.json) freezes the saved states and every other global, performs no optimization/inference, and inspects the conservative interval bound around alpha4=0. For its minimum gaze-floor row, the native-alpha derivative is **+4.90641495** on the negative side and **-6.09804402** on the positive side. At zero, near-active rows 0/1 are flagged nonsmooth. This establishes an interval-bound branch kink. The selected smooth-branch derivative at the tie is not a generalized stationarity certificate. The inspection does not establish an optimal fit at alpha4=0 or identify a physiological mechanism.

The adjacent [unattempted roster](g8_adjacent/summary.json) preserves all 300,270 scheduled slots: retained measurements would be available for 89,175 / 89,215 / 89,175 rows for omissions 0 / 1 / 2, respectively. No inversion or score is claimed. [comparison.json](comparison.json) explicitly records missing paired metrics and no optical ranking.

## What can be learned and what comes next

The original calibrated optimum and full consistency scores reproduce from fresh initialization. The numerical working-set repairs resolve the control's original failure while preserving the same physical model and objective. The adjacent convention exposes a remaining interval-bound branch problem. We cannot yet decide whether changing reference reduces the structured omitted-point residual or improves useful accommodation information.

The next work is a bounded numerical investigation of one-sided/generalized stationarity and curvature at the interval kink, independently distinguishing near-active NNLS rows from actual complementarity and retaining every original inequality and threshold. Only a properly certified adjacent calibration can complete the matched G8 comparison. A new radial law, free camera rotation or additional optical mechanism does not follow automatically from this result.

The one-degree bound applies to horizontal mean-P4-minus-mean-P1 response at g=1 over accommodation 0–4 D within the declared monotone theta domain, with derivatives conservatively extended to 0–6 D. It is not a bound on individual gaze deviation. Physical zeros, absolute accommodation accuracy, calibrated localization-noise covariance, common scale transfer and capture/demand confounding remain unresolved.
