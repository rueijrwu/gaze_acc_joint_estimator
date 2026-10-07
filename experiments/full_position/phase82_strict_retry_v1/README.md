# Phase 8.2 strict retry artifacts

`RESULTS.md` summarizes the training-only continuation retry. Strong-anchor conditional37 boundary-transition diagnostics are reported separately in `../phase82_audit_transitions_v1/RESULTS.md`. `verification.json` records independent reconstruction and source-hash checks. `calibration_candidates.jsonl.gz` retains every start/stage checkpoint. `design_snapshot/` and `implementation_snapshot/` preserve the exact design and Python sources represented by the retry completion hashes.

To reproduce the retry from the preserved failed training trajectory, run from the repository root:

```bash
rtk proxy env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m full_position.retry_calibration \
  --output experiments/full_position/phase82_strict_retry_reproduction
```

The output path must be new. This command uses only the `capture_1` training split and writes a separate artifact. The preserved original failed fit remains under `joint_sensitivity_v1`.

To independently verify a completed retry artifact in this directory:

```bash
rtk proxy env PYTHONPATH=.:tests OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python experiments/full_position/phase82_strict_retry_v1/verify_results.py
```

The verification reconstructs the training covariance, profiled coefficients, objective components, and unchanged convergence thresholds. It does not evaluate held-out rows.
