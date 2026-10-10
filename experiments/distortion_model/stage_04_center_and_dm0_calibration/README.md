# Stage 4 — corrected centers and full DM0 calibration

**S6/G6: complete, GO_WITH_LIMIT. S7/G7 attempt 04: COMPLETE_CERTIFIED_WITH_LIMIT, GO_WITH_LIMIT. G8 has not run and awaits separate authorization.**

G6 fits the declared forward visual-gaze center polynomial at the immutable G5 state/optical snapshot. It uses all complete frames in all twenty reviewed intervals. Corrected centers are derived diagnostics; the coefficient solve uses the complete ten-coordinate point criterion with all covariance cross-terms.

## Files

- [Estimator mathematics](../../../distortion_model/centers.py): center basis, derivatives, reference conversion, correction ledger, chunked weighted coefficient solve.
- [Experiment runner](scripts/run.py): compatible parent loading, immutable snapshot checks, empirical fit, full arrays, plots and provenance.
- [G7 unbounded runner](scripts/run_joint.py), [one-degree bounded runner](scripts/run_joint_bounded.py), [bounded-run audit](scripts/audit_g7_bounded.py), and [saved compact aggregator](scripts/summarize_g7_compact.py). Current result: [attempt04](results/g7_attempt_04/STAGE_REPORT.md), one reviewed constrained correction to the immutable [attempt03](results/g7_attempt_03/checkpoint.json) parent. Its [checkpoint](results/g7_attempt_04/checkpoint.json), [verification](results/g7_attempt_04/verification.json), and [independent public-optics audit](results/g7_attempt_04/independent_public_optics_audit.json) preserve the numerical evidence. The original pending-review [probe](results/g7_polish_repair_01/STAGE_REPORT.md) and historical attempts remain unchanged.
- [Results](docs/RESULTS.md), [attempt ledger](docs/STAGE_REPORT.md) and [live progress](docs/PROGRESS.md).
- [Attempt 01](results/g6_attempt_01/STAGE_REPORT.md): retained initial run; reporting repair required.
- [Attempt 02](results/g6_attempt_02/STAGE_REPORT.md): reviewed result after fixing capture-color legends and adding explicit model provenance fields. Numerical arrays/objectives are unchanged.
- [Requirements](../../../requirements-stage6.txt): same pinned numerical environment as G5. No GPU backend was required.
- [G7 requirements](../../../requirements-stage7.txt): pinned NumPy 2.5.3, SciPy 1.18.1, Matplotlib 3.11.2, and installed CuPy distribution 14.2.0.
- [One-degree bound preflight](results/preflight_1degree/preflight.json): continuous interval and dense optical derivative checks plus actual-row CPU/GPU proposal comparison.

Full campaign attempts preserve source/config/parent hashes, source copies, `model.json`, `fit.json`, `fitted.npz`, conditions, plots, checkpoints and audits. Saved-point repairs and adoption audits are intentionally narrower records: attempt04 contains the corrected solution/checkpoint and verification, while its empirical compact summary points to the pre-existing attempt03 records. The native data remain in [data](../../../data/). Full arrays preserve all 100,090 rows; the immutable upstream omission schedule still contains 300,270 slots.

## Reproduce

From the repository root, in the numerical environment:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python experiments/distortion_model/stage_04_center_and_dm0_calibration/scripts/run.py --output experiments/distortion_model/stage_04_center_and_dm0_calibration/results/g6_attempt_03
```

The default parent is the reviewed G5 `g5_attempt_01`. Override it with `--parent` only for a compatible reviewed G5 checkpoint. Existing output folders are rejected. This command runs the empirical experiment; it does not run automated tests.

The calculation forms chunked weighted moments and a ten-parameter linear system. Group weights and fixation means use full exposure counts, including across chunks. Curvature is weakly regularized in a fixed empirical reference-pixel scale; the policy is saved before fitting. Shape covariance, mean anchors, optical zeros, theta/A/g and template gauges remain fixed during G6.

The runner saves full point, theta-anchor, A-anchor and regularization components before and after the solve. It records numerical identities against the existing full forward adapter, with no extra center loss. Root reviews scientific findings; independent mechanical reconstruction does not assign a gate decision.

Attempt04 passes the declared numerical certificate, with global scaled KKT residual 4.24e-10 against 1e-6. The one-point correction changes J by −5.9962e-12 and maximum predictions by 6.19e-6 px; no material empirical improvement is claimed. `crosscheck_complete` and `comparison_complete` remain false. G8 has not run and awaits separate authorization. Carry forward the fixed operational omega1/omega4 references and unresolved continuous omega1 uncertainty, G3 alternatives, and physical zeros.

Reproduce the saved-point adoption from the preserved reviewed probe into a fresh output folder with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python experiments/distortion_model/stage_04_center_and_dm0_calibration/scripts/accept_g7_polish.py --output experiments/distortion_model/stage_04_center_and_dm0_calibration/results/g7_attempt_04
```

Independently reconstruct the accepted full objective/prediction with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python experiments/distortion_model/stage_04_center_and_dm0_calibration/scripts/audit_g7_polish.py --attempt experiments/distortion_model/stage_04_center_and_dm0_calibration/results/g7_attempt_04
```
