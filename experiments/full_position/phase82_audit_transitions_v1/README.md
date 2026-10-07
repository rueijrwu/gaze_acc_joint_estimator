# Phase 8.2 transition audit artifacts

`RESULTS.md` reports the strong-anchor conditional37 boundary transition decomposition and training-only state-scale gain ranges. The strict continuation result is documented separately in `../phase82_strict_retry_v1/RESULTS.md`. `paired_transition_membership.jsonl.gz` contains the saved point and frame membership/delta records. `summary.json` contains all model/variant/family strata and source hashes. `verification.json` records standalone integrity and arithmetic verification: 1,507 input/source hashes, 32 comparisons, 15,360 scheduled point memberships (13,611 paired scored), 4,537 paired frame memberships, and 124 independently recomputed transition summaries.

Run the standalone verifier from the repository root:

```bash
rtk proxy env PYTHONPATH=.:tests OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python experiments/full_position/phase82_audit_transitions_v1/verify_results.py
```

To regenerate the audit from the frozen Phase 8.2 sensitivity outputs, choose a new output directory:

```bash
rtk proxy env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m full_position.audit_followup \
  --source experiments/full_position/joint_sensitivity_v1 \
  --output experiments/full_position/phase82_audit_transitions_reproduction
```

This post-processes saved training trajectories and saved evaluation holdouts; it does not refit models. The original `joint_sensitivity_v1` artifacts remain unchanged.
