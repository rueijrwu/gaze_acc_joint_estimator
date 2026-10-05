# Maintained commands

Run from the repository root. Outputs below must be new directories; existing fitted artifacts are never replaced. Commands use the declared single-thread environment and suppress Python bytecode caches.

```bash
rtk env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B exp2/full_information/experiment.py diagnose --output-dir exp2/full_information/diagnostics/current
rtk env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B exp2/full_information/experiment.py scale --output-dir exp2/full_information/diagnostics/scale
rtk env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B exp2/full_information/experiment.py mean-profile --model-dir exp2/reduced_calibration/training/holdout3/robust --fixation 17 --mean-offset-deg 0.5 --output-dir exp2/full_information/profiles/H0_plus
```

The mean-profile defaults are 600 function evaluations, 900 seconds, LSMR maximum 300 with absolute/relative tolerance `1e-7`, 24 robust outer iterations and 25 inner evaluations. Physical gradient/step, relative-cost and robust-weight stopping thresholds remain unchanged. Use offset `-0.5` for the negative probe, and `exp2/relative_calibration/training/FC0/robust` for the full-data constrained source. `--resume-dir` accepts only the same guarded probe objective, with a new output directory; original and continuation artifacts remain separate.

```bash
rtk env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B exp2/full_information/experiment.py sensitivity --model-dir exp2/reduced_calibration/training/holdout3/robust --anchor-multiplier 0.5 --output-dir exp2/full_information/sensitivity/H0_anchor_half
rtk env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B exp2/full_information/experiment.py sensitivity --model-dir exp2/reduced_calibration/training/holdout3/robust --anchor-multiplier 2 --output-dir exp2/full_information/sensitivity/H0_anchor_double
rtk env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B exp2/full_information/experiment.py sensitivity --model-dir exp2/reduced_calibration/training/holdout3/robust --perturb-initial-states --seed 20261004 --output-dir exp2/full_information/sensitivity/H0_perturbed
```

Both nominal anchors change together in anchor sensitivity; the function prior stays fixed. State perturbation uses standard deviations `0.2 deg` and `0.05 D`, clips physical bounds, and records clipping counts.

The four predeclared matched cells are recorded in [matrix_protocol.json](matrix_protocol.json). Replace the case-specific flags/output for each row:

| Case | `--demands` | `--scale-hypothesis` | Output directory |
|---|---:|---|---|
| N3 | 3 | free | `exp2/full_information/matched/N3` |
| N4 | 4 | free | `exp2/full_information/matched/N4` |
| R3 | 3 | known | `exp2/full_information/matched/R3` |
| R4 | 4 | known | `exp2/full_information/matched/R4` |

```bash
rtk env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B exp2/full_information/matched_experiment.py --demands 3 --scale-hypothesis free --heldout-fixation 17 --output-dir exp2/full_information/matched/N3
rtk env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B exp2/full_information/matched_predict.py --model-dir exp2/full_information/matched/N3 --output-dir exp2/full_information/matched/N3/predictions
rtk env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B exp2/full_information/matrix_summary.py --matrix-dir exp2/full_information/matched --output-dir exp2/full_information/matched_summary
```

Repeated `--heldout-fixation` arguments select complete arbitrary fixation holdouts. Preprocessing is rebuilt from the common training support for each fold. Prediction refuses an incomplete fit unless `--allow-incomplete` is explicit; that status remains in the report. Known scale is a hypothetical fixed-scale sensitivity, never a claim of calibrated scale. Optional `--scale-hypothesis uncertain --log-scale-sd 0.001` or `0.01` exposes predeclared finite-uncertainty sensitivities; these are outside the primary four-cell matrix. Do not choose assumptions by their total objective.

Known and finite-uncertainty models additionally expose `--coefficient-solver newton` as an optional numerical acceleration. Default IRLS remains unchanged, and the exact free-scale control requires IRLS. Solver choice is recorded per phase separately from the mathematical objective identity; coefficient/weight profile certification remains mandatory. Same-objective `--resume-dir` continuations restore verified saved states, coefficients and weights into a new output directory, preserving each phase's artifacts and cumulative solver runtime/evaluation counts. `matrix_summary.py --case NAME=CONTINUED_DIRECTORY` explicitly selects a reviewed continuation without overwriting or silently choosing among original results.

```bash
rtk env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B exp2/full_information/experiment.py profile-summary --profile-dir exp2/full_information/profiles/H0_minus --profile-dir exp2/full_information/profiles/H0_plus_refined --profile-dir exp2/full_information/profiles/FC0_minus --profile-dir exp2/full_information/profiles/FC0_plus_refined --output-dir exp2/full_information/profile_summary
rtk env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B exp2/full_information/experiment.py sensitivity-summary --sensitivity-dir exp2/full_information/sensitivity/H0_anchor_half --sensitivity-dir exp2/full_information/sensitivity/H0_anchor_double --sensitivity-dir exp2/full_information/sensitivity/H0_perturbed --output-dir exp2/full_information/sensitivity_summary
```

Summary inputs above are explicit artifact paths; a missing or incomplete probe is not silently replaced. Profile tables retain its actual convergence/KKT status and use achieved means rather than requested targets.

```bash
rtk env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B exp2/full_information/experiment.py validation-contract --output-dir exp2/full_information/validation/protocol_v2
rtk env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B exp2/full_information/experiment.py validation-evaluate --output-dir exp2/full_information/validation/current_status
```

Without independently acquired references and predeclared thresholds, evaluation reports `not_evaluable_without_independent_references`. Supply `--prediction-csv`, `--reference-csv` and `--protocol-json` only for an independently documented withheld validation cohort satisfying the generated contract.
