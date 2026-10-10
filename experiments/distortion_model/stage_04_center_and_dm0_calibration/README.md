# Stage 4 — corrected centers and full DM0 calibration

**S6/G6: complete, GO_WITH_LIMIT. S7/G7 attempt 03: complete, uncertified, PAUSE.**

G6 fits the declared forward visual-gaze center polynomial at the immutable G5 state/optical snapshot. It uses all complete frames in all twenty reviewed intervals. Corrected centers are derived diagnostics; the coefficient solve uses the complete ten-coordinate point criterion with all covariance cross-terms.

## Files

- [Estimator mathematics](../../../distortion_model/centers.py): center basis, derivatives, reference conversion, correction ledger, chunked weighted coefficient solve.
- [Experiment runner](scripts/run.py): compatible parent loading, immutable snapshot checks, empirical fit, full arrays, plots and provenance.
- [G7 unbounded runner](scripts/run_joint.py), [one-degree bounded runner](scripts/run_joint_bounded.py), [bounded-run audit](scripts/audit_g7_bounded.py), and [saved compact aggregator](scripts/summarize_g7_compact.py). The current result is [attempt 03](results/g7_attempt_03/STAGE_REPORT.md), one compatible continuation from [attempt 02](results/g7_attempt_02/STAGE_REPORT.md); its [checkpoint](results/g7_attempt_03/checkpoint.json) and [mechanical audit](results/g7_attempt_03/mechanical_audit.json) preserve the evidence. Same-state [repair01](results/g7_repair_01/STAGE_REPORT.md) changed no fit values. Attempts01/02 remain unchanged historical records.
- [Results](docs/RESULTS.md), [attempt ledger](docs/STAGE_REPORT.md) and [live progress](docs/PROGRESS.md).
- [Attempt 01](results/g6_attempt_01/STAGE_REPORT.md): retained initial run; reporting repair required.
- [Attempt 02](results/g6_attempt_02/STAGE_REPORT.md): reviewed result after fixing capture-color legends and adding explicit model provenance fields. Numerical arrays/objectives are unchanged.
- [Requirements](../../../requirements-stage6.txt): same pinned numerical environment as G5. No GPU backend was required.
- [G7 requirements](../../../requirements-stage7.txt): pinned NumPy 2.5.3, SciPy 1.18.1, Matplotlib 3.11.2, and installed CuPy distribution 14.2.0.
- [One-degree bound preflight](results/preflight_1degree/preflight.json): continuous interval and dense optical derivative checks plus actual-row CPU/GPU proposal comparison.

Every attempt preserves source/config/parent hashes, source copies, `model.json`, `fit.json`, `bookkeeping.json`, `fitted.npz`, `conditions.json`, plots, checkpoint and audit. The native data remain in [data](../../../data/). Full arrays preserve all 100,090 rows; the immutable upstream omission schedule still contains 300,270 slots.

## Reproduce

From the repository root, in the numerical environment:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python experiments/distortion_model/stage_04_center_and_dm0_calibration/scripts/run.py --output experiments/distortion_model/stage_04_center_and_dm0_calibration/results/g6_attempt_03
```

The default parent is the reviewed G5 `g5_attempt_01`. Override it with `--parent` only for a compatible reviewed G5 checkpoint. Existing output folders are rejected. This command runs the empirical experiment; it does not run automated tests.

The calculation forms chunked weighted moments and a ten-parameter linear system. Group weights and fixation means use full exposure counts, including across chunks. Curvature is weakly regularized in a fixed empirical reference-pixel scale; the policy is saved before fitting. Shape covariance, mean anchors, optical zeros, theta/A/g and template gauges remain fixed during G6.

The runner saves full point, theta-anchor, A-anchor and regularization components before and after the solve. It records numerical identities against the existing full forward adapter, with no extra center loss. Root reviews scientific findings; independent mechanical reconstruction does not assign a gate decision.

Current next numerical question: why does positive constrained profile curvature coexist with rejected full shared proposals and no acceptable same-objective polish step near a 2.51e-5 global residual? Diagnose QP accuracy/scaling and constraint-aware Newton/line-search behavior. The smooth derivative and profile-curvature checks now pass, but global stationarity still fails. No further fit in this campaign; keep G8 paused.
