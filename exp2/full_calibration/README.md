# Full calibration references and matched controls

This directory groups the retained legacy full-fit reference and the fresh
matched full-data fits used for retrospective reduced-demand comparisons. The
artifacts were moved without modifying their contents; manifests, checkpoints,
predictions, and plots remain byte-for-byte intact.

Immutable provenance JSON may still name the artifacts' original pre-relocation
locations. Those are historical provenance paths; the directories listed below
are the live locations.

| Directory | Contents | Status |
|---|---|---|
| `legacy/` | Previous full-fit model, checkpoint, state CSVs and diagnostics | Historical reference; its own run history records a validation resume and it did not meet all physical convergence criteria |
| `matched/quadratic/` | Fresh training-only-prior full-data quadratic stage | Initial stage did not converge within its wall budget |
| `matched/robust_initial/` | First robust full-data pass and checkpoint | Initial pass did not converge within its wall budget |
| `matched/robust/` | Continued robust fit and final full-data state trajectories | Converged final matched control |
| `matched/metadata/` | Provenance and selected-interval records for both matched passes | Preserved inputs and run metadata |

The final matched fit used the same objective during continuation: 600 function
evaluations, 900 seconds, LSMR cap 1,000, and 24 robust outer rounds of at most
25 evaluations. It met the stated physical stopping criteria. The initial
quadratic and robust passes remain available to preserve the complete history.
The demand-holdout predictions and retrospective comparisons remain under
[`../reduced_calibration/predictions`](../reduced_calibration/predictions/).

To resume the legacy checkpoint into a new output directory:

```bash
python exp2/calibrate_continuation.py \
  --resume-dir exp2/full_calibration/legacy \
  --variant robust_noise --output-dir exp2/fit_next \
  --max-nfev 150 --robust-outer 6 --robust-max-nfev 25 --wall-seconds 240
```

To use the converged matched model for a retrospective comparison:

```bash
python exp2/reduced_calibration/experiment.py compare \
  --fold holdout3 \
  --prediction-dir exp2/reduced_calibration/predictions/holdout3 \
  --matched-dir exp2/full_calibration/matched/robust
```

The holdout metrics are differences from model-derived trajectories, not
independent measurements of physiological state accuracy. See
[`../reduced_calibration/README.md`](../reduced_calibration/README.md) for
protocol, budgets, limitations, and the complete results.
