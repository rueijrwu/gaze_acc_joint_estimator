# Gaze and accommodation workflow

The preserved legacy estimator uses normalized horizontal P1/P4 displacement and pair
separation to fit framewise gaze and accommodation jointly. Read
[Estimation.md](../documents/Estimation.md) for the method and
[Theory.md](../documents/Theory.md) for the optical assumptions.
The legacy reference and matched full-calibration stages are organized in
[full_calibration/README.md](full_calibration/README.md); the current constrained
holdout workflow and results are in [reduced_calibration/README.md](reduced_calibration/README.md).
The new [reduced-calibration experiment](reduced_calibration/README.md)
enforces missing-demand displacement interpolation during every coefficient
profile, with complementary holdouts and a fresh matched full-data control.

## Files

| File or directory | Purpose |
|---|---|
| `detect.py` | Native pupil/P1/P4 detection and frame-aligned pickle export |
| `observations.py` | Shared normalized optical measurements |
| `extract_fixations.py` | Review and select five ordered fixations per capture |
| `calibrate_continuation.py` | Single CLI for fresh calibration, resume, and robust refinement |
| `calibrate_profiled.py` | Shared model, coefficient profiling, derivatives, inversion, and reporting |
| `estimate_profiled.py` | Shared pure batched inversion and physical-optimality helpers, extracted unchanged for both experiments |
| `analyze_pair_x.py`, `analyze_angles_scale.py`, `analyze_miss_frames.py` | Detection-quality inspection |
| `calibration_seed.json` | Legacy full-data exponent and prior-weight seed; excluded from the new experiment |
| `fixations/` | Selected intervals and review plots |
| `reduced_calibration/` | Constrained-demand runner, converged holdout fits, frozen predictions and audits |
| `full_calibration/` | Central legacy reference and matched full-fit stages |

Run commands from the repository root. Calibration needs Python, NumPy, SciPy,
and Matplotlib. Detection additionally needs OpenCV and `pkj_image_process_py`.
The numerical scripts default to one OpenMP/BLAS thread to avoid oversubscription.

## Detection and fixation selection

```bash
python exp2/detect.py exp2/capture_1.mkv \
  --output-pickle exp2/capture_1_detections.pkl \
  --overlay exp2/capture_1_overlay.png
```

The default native configuration is the repository's `config.json`; the input
frame dimensions must match it. Detection requires both native pupil-valid flags
before accepting P1/P4 pairs. Missing/invalid rows stay frame-aligned with NaN
selected coordinates. An established pair seed is retained across invalid-pupil
frames and reused when pupil validity returns. Candidate shape, containment,
angle, scale, and association checks remain relevant.

Use `--max-frames`, `--config`, and `--log-file` as needed. Pickles retain original
frame indices, timestamps, selected point pairs, candidate arrays, and validity
flags. Inspect detections before attributing abrupt traces to eye motion.

The selected calibration intervals are already in `exp2/fixations`. Use a new
output directory when reviewing a revised selection:

```bash
python exp2/extract_fixations.py --output-dir exp2/fixations_next
python exp2/analyze_pair_x.py exp2/capture_1_detections.pkl
python exp2/analyze_angles_scale.py exp2/capture_1_detections.pkl
python exp2/analyze_miss_frames.py --help
```

Calibration retains the central 80% of each frozen interval, then applies the
measurement and stored pupil gates. Captures 1–4 provide 20 fixations and 77,756
valid core frames. Capture 5 is outside this calibration workflow.

## Legacy fresh fit and robust refinement

Start with the noise-weighted quadratic model in a new directory:

```bash
python exp2/calibrate_continuation.py --fresh \
  --variant quadratic_noise --output-dir exp2/fit_quadratic \
  --max-nfev 150 --wall-seconds 240
```

Fresh full-data fits use `calibration_seed.json` when present, preserving the
working fixed exponent and initial-function prior weights. Optional `--p` and
`--seed-path` declare alternatives. `--holdout-fixation 0..19` excludes that entire
fixation and uses training-only initialization/noise; a full-data seed is ignored
and an explicit `--seed-path` is prohibited with a holdout.

Warm-start robust refinement from a matching quadratic checkpoint:

```bash
python exp2/calibrate_continuation.py \
  --warm-start-dir exp2/fit_quadratic \
  --variant robust_noise --output-dir exp2/fit_robust \
  --max-nfev 150 --robust-outer 6 --robust-max-nfev 25 --wall-seconds 240
```

Resume the retained working robust fit without changing its objective:

```bash
python exp2/calibrate_continuation.py \
  --resume-dir exp2/full_calibration/legacy \
  --variant robust_noise --output-dir exp2/fit_next \
  --max-nfev 150 --robust-outer 6 --robust-max-nfev 25 --wall-seconds 240
```

Always use an output directory outside the source checkpoint directory, without
an existing checkpoint. Resume restores the accepted fit state and starts fresh
trust-region machinery. `--intervals` explicitly selects another frozen interval
file; changing support or priors requires a fresh fit.

## Controls and outputs

`--scaling curvature` is the default coordinate preconditioner; `none` retains the
original coordinate scaling. `--lsmr-maxiter` (default 100), `--lsmr-atol`, and
`--lsmr-btol` control the inner linear solve. Inspect `lsmr_history.json` before
increasing its budget. `solver_only` retains range-based observation weighting;
`quadratic_noise` and `robust_noise` use the same empirical correlated covariance.

`--max-nfev` limits total trajectory evaluations. Robust subproblem and outer
budgets control weight updates; coefficient IRLS must converge. Wall/evaluation
limits save the last accepted point and report `not_converged` unless every
physical stopping criterion has passed. Tolerances are explicit
`--physical-gtol`, `--physical-step-tol`, and `--relative-cost-tol` options.

Outputs include the embedded-identity `checkpoint.npz`, `model.json`,
`diagnostics.json`, `status.json`, `history.json`, `lsmr_history.json`, per-capture
state CSVs, fixation summaries, and observation/state plots. Raw frames remain
represented in the CSVs; state estimates are written only for calibration cores.
The manifest prevents silent changes to support, order, precision, model, or priors.

The current small optical residuals are training fit errors. Independent gaze and
accommodation reference measurements are needed to establish state accuracy.

## Reduced-demand experiment

The new experiment uses a fixed linear map from free coefficients to the legacy
14-slot representation. Demand-3 holdout trains only 0.36D/2D/4D; demand-2 holdout
trains only 0.36D/3D/4D. Each has 12 free coefficients, with the missing internal
displacement intercept/gain constrained to interpolate its neighbors throughout
training. The matched full-data control has 14 free coefficients. All folds use
training-only exponent/noise/initialization and the same frame-range-based prior
rule at strength 0.1; historical seeds and reference parameters are excluded.

Keep existing outputs intact. New fits, continuations, frozen heldout inversions,
initial-model controls, exact-frame comparisons, and audit summaries belong under
`reduced_calibration/`. Larger evaluation/wall budgets retain the same physical
convergence tolerances. Report nonconvergence, branch ambiguity, conditioning,
bounds, extrapolation and state deviations honestly. The protocol, commands and
results are in [reduced_calibration/README.md](reduced_calibration/README.md).
Both robust holdout fits converged. Demand-3 holdout gives 0.3812° gaze/0.07857D
accommodation RMSE against the preserved full-fit trajectory; demand-2 holdout
gives 2.1338°/0.20081D. Both have zero ambiguous trained inversions; four demand-2
boundary frames retain unverified stationarity flags. The fresh matched full
control converged; its comparisons give 0.3827°/0.07318D for demand-3 holdout and
2.1564°/0.19146D for demand-2 holdout. All comparison plots are linked in the
report. These retrospective results do not establish
physiological accuracy or uniformly adequate three-demand calibration.


Optional relative-observation quality diagnostics, isolated displacement curvature
and frozen previous-training fixation-mean gaze priors are maintained in
[relative_calibration](relative_calibration/README.md). That experiment owns its
new artifacts; the full/reduced reference directories retain their original models.
The scale gate keeps the normalized state estimator and reports S1 as a quality
diagnostic. Nominal anchors remain separate from model-derived previous means.
