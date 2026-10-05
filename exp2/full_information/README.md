# Full optical information: staged workflow

This isolated workflow implements `documents/CURRENT_PLAN.md`. Existing full,
reduced and relative calibration artifacts are immutable inputs. Its purpose is
to distinguish optical evidence, soft nominal anchors, regularization and nuisance
assumptions. No existing fitted trajectory is ground truth or an accuracy ranking.

The maintained `experiment.py diagnose` command reconstructs and verifies each
original model objective, then separates optical, gaze/accommodation anchor,
temporal, initial-function, curvature and previous-mean costs. It reports nominal
fixation-mean consistency, within-fixation spread and conditional noise/cross-talk
from analytic physical Jacobians. Noise propagation holds coefficients fixed and
ignores coefficient uncertainty/model bias; it is not physiological accuracy or a
confidence interval. Calibration-state diagnostics remain separate from anchor-free
inference diagnostics.

```bash
rtk proxy env OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 OPENBLAS_NUM_THREADS=1 python -B exp2/full_information/experiment.py diagnose --output-dir exp2/full_information/diagnostics/current_solutions
rtk proxy env OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 OPENBLAS_NUM_THREADS=1 python -B exp2/full_information/experiment.py validation-contract --output-dir exp2/full_information/validation/protocol
```

Whole-fixation mean profiles and anchor/initialization sensitivity will explicitly
reoptimize other states and coefficients. A soft probe must report its achieved
mean, separate probe cost and constrained stationarity at that achieved mean; it
must never be called an exact target-mean profile or a formal confidence interval.

The proposed complete relative coordinate `[d,rho4,log(S1)]` is invertible from
ordered positive `[m,S1,S4]`. Its transformed noise is a declared approximation.
No independently known scale is available. Fixed or constrained scale prototypes
are conditional assumptions; a free framewise scale control must reduce exactly
to the matched two-channel marginal-noise estimator and cannot create an extra
validation residual. Implementation/run gates will be recorded before comparisons.

The validation contract specifies randomized repeat acquisition, independent gaze
and accommodation references, exact provenance/alignment and heldout recording
identity. Those physical observations are unavailable in the current archive and
cannot be fabricated. Nominal target labels or capture5 without protocol/reference
annotations do not satisfy external validation.
