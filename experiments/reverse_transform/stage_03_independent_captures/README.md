# Independent P1/P4 fits across captures 1–4

This stage fits each reviewed capture independently, with capture 1’s nominal zero-gaze means at 0.360360 D as the shared P1 and P4 empirical references. Capture 1 reuses its audited P1 gaze and framewise magnification exactly. Captures 2–4 receive independent P1 keystone fits against the same reference. P4 uses fixed framewise P1 magnification and one independent keystone plus one constant relative radial coefficient per capture. Capture 1’s radial increment is fixed at zero as the reference convention.

Demand labels are nominal experimental labels, attached after fitting for descriptive association only. There is one capture per demand, no framewise accommodation estimate, and no P4 magnification fit. The empirical radial origin and reference may contain baseline distortion, so coefficients are relative and do not identify an absolute physical barrel or physiological accommodation.

The [current results](results/run/RESULTS.md) include coverage, P1 magnification summaries, per-capture coefficients and errors, audit status, and figures. See the [scientific review](SCIENTIFIC_REVIEW.md) for the interpretation and limits. The [protocol](results/run/protocol.json), [summary](results/run/summary.json), [audit](results/run/audit.json), and [provenance](results/run/provenance.json) preserve machine-readable details.

Reproduce the canonical run from the repository root into a new output directory:

```bash
python experiments/reverse_transform/stage_03_independent_captures/scripts/run.py \
  --output experiments/reverse_transform/stage_03_independent_captures/results/replay
python experiments/reverse_transform/stage_03_independent_captures/scripts/audit.py \
  --results experiments/reverse_transform/stage_03_independent_captures/results/replay
python experiments/reverse_transform/stage_03_independent_captures/scripts/plot_results.py \
  --results experiments/reverse_transform/stage_03_independent_captures/results/replay
```

The source uses only captures 1–4 and retains all complete frames in forward fits. Inverse metrics use the common valid set and show per-fixation counts. No test suite was run; the saved-result audit is the requested numerical verification.
