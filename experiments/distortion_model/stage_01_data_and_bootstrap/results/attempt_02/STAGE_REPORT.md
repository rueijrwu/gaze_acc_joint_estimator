# Stage 1 reviewed evidence

Execution status: COMPLETE. Scientific disposition: protocol_ready for the stage-one
relative-coordinate pipeline and provisional initializer. Full optical calibration,
certification, cross-check and model comparison have not been performed.

The experiment completed in 6.678 seconds, including loading,
array compression and three plots, on NumPy float64 CPU. Four threads load the captures;
BLAS uses one thread. CuPy 14.2.0 reported `cudaErrorNoDevice` in this runtime.
The bootstrap has only two free coefficients and needs no GPU optimizer.

## G0 / attempt 02 / bootstrap_only

Status: COMPLETE. Decision: **GO_WITH_LIMIT**.

Source commit: `efd88bd3b326653aedc3b383143ef10856d22092`. Dirty-source content hash: `c43a344ae12e99a5a92f8cfa58162a4923563c014c8e78689a54c04fb86e549b`.
Theory, plan, config and individual implementation/test hashes are in
[provenance.json](provenance.json). Source payload hashes and exact reviewed identities
are in [population.json](population.json). The remote branch was subsequently verified
to match this local parent; [reproduction_checks.json](reproduction_checks.json) records
that observation. The provenance file preserves the earlier failed remote-read observation.

Parent gate/attempt: [attempt 01 repair](../attempt_01/STAGE_REPORT.md); no usable prior checkpoint.
Question: Do trusted native measurements, identities, correspondence and masks support
initialization on all twenty full reviewed exposures?

Fixed: original source bytes, stored `[2,1,0]` P4 permutation, native pixel axes,
full half-open reviewed intervals, P4-minus-P1 sign, presence/finite flags and complete-pair
initialization policy. Pupil, detector status, area, shape and residual thresholds do not
select rows. No trimming, subsampling or source detection changes were applied.
Fitted: no empirical optical coefficients. Derived: ten relative coordinates,
candidate-independent frame and slot identities, approximate frozen covariance.
The synthetic optical fixture is independent of real reference selection.

Population: **100,090 scheduled / 89,175 complete valid / 10,915 unavailable**.
Every row remains scheduled. All **300,270** P4 omission slots are present with raw,
retained and held availability fields and explicit `not_run_stage1` outcomes.
Optimized/calibrated states: zero; 89,175 provisional starts are available at G1.
All twenty exposure counts are explicit:

| Capture | Target (deg) | Demand (D) | Scheduled | Complete valid / starts | Unavailable | Slots |
|---|---:|---:|---:|---:|---:|---:|
| capture_1_detections.pkl | -10 | 0.36036036 | 4800 | 4433 | 367 | 14400 |
| capture_1_detections.pkl | -5 | 0.36036036 | 4800 | 4800 | 0 | 14400 |
| capture_1_detections.pkl | 0 | 0.36036036 | 4900 | 4900 | 0 | 14700 |
| capture_1_detections.pkl | 5 | 0.36036036 | 5000 | 4993 | 7 | 15000 |
| capture_1_detections.pkl | 10 | 0.36036036 | 5200 | 5154 | 46 | 15600 |
| capture_2_detections.pkl | -10 | 4 | 4600 | 3912 | 688 | 13800 |
| capture_2_detections.pkl | -5 | 4 | 5000 | 4345 | 655 | 15000 |
| capture_2_detections.pkl | 0 | 4 | 5000 | 4553 | 447 | 15000 |
| capture_2_detections.pkl | 5 | 4 | 5000 | 4781 | 219 | 15000 |
| capture_2_detections.pkl | 10 | 4 | 5290 | 4435 | 855 | 15870 |
| capture_3_detections.pkl | -10 | 3 | 4700 | 4505 | 195 | 14100 |
| capture_3_detections.pkl | -5 | 3 | 4900 | 3902 | 998 | 14700 |
| capture_3_detections.pkl | 0 | 3 | 5000 | 4811 | 189 | 15000 |
| capture_3_detections.pkl | 5 | 3 | 5000 | 4251 | 749 | 15000 |
| capture_3_detections.pkl | 10 | 3 | 5500 | 2326 | 3174 | 16500 |
| capture_4_detections.pkl | -10 | 2 | 4800 | 3914 | 886 | 14400 |
| capture_4_detections.pkl | -5 | 2 | 4900 | 4899 | 1 | 14700 |
| capture_4_detections.pkl | 0 | 2 | 5000 | 4747 | 253 | 15000 |
| capture_4_detections.pkl | 5 | 2 | 4900 | 4738 | 162 | 14700 |
| capture_4_detections.pkl | 10 | 2 | 5800 | 4776 | 1024 | 17400 |

Evidence: [census.json](census.json), [population.npz](population.npz),
[slots.npz](slots.npz), [measurements.npz](measurements.npz),
[source_metadata.json](source_metadata.json), [native tracks](native_tracks.png),
[full interval boundaries](full_interval_boundaries.png),
[tests.json](tests.json), [covariance.json](covariance.json),
[covariance.npz](covariance.npz), [synthetic fixture](synthetic_fixture.json),
[synthetic forward evaluations](synthetic_forward.npz), and
[reproduction checks](reproduction_checks.json).
Evidence kinds: synthetic numerical contracts and empirical input census; no optical fit.

All four source hashes match the reviewed metadata. There are no duplicate source-frame
identities, within-exposure frame gaps, finite/presence flag disagreements, or mismatches
between recalculated and reviewed valid/timestamp counts. L has rank 10 and common
translation is a null direction. All 17 permanent tests passed, zero failed/errors/skipped,
at rtol 1e-10 and atol 1e-9. They cover relative geometry and covariance, retained-input
noninterference, independent optical zeros, native-forward composition and scale,
center signs, bounds/denominators, immutable reference snapshots, dynamic bootstrap
states, trusted loading and complete failure-slot accounting. Full inversion, optical
parameter derivatives and GPU parity are deferred to the gates where those blocks exist.
All 533 synthetic forward proposals were admissible; maximum scale-recovery error was
4.44e-16. This is numerical evidence only.

The timing limitation is real: capture 1 has **601** backward timestamp jumps in the
source, **550** within its reviewed intervals. Its original rows/frames remain usable;
its timestamps are not repaired or used in the noise-difference policy. Covariance uses
**63,756** complete adjacent row/frame differences within reliable capture 2–4 intervals,
with no gap/interval bridging. The full native covariance is shrunk 5% toward its
coordinate variances, mapped by L, and marginalized before whitening. It contains
shared point errors and may contain eye motion. Its minimum native/relative eigenvalues
are positive (0.000318688 / 0.000500602 px²). It is an approximate weighting metric,
not hardware localization noise or the singular post-profile residual covariance.

Boundary plots show transition excursions and changing availability within the declared
100-row boundary uncertainty. They remain in the calculations. The weakest exposure,
capture 3 at +10 degrees, has 2,326 valid of 5,500 scheduled rows; its missing coverage
is visible in both frame and slot rosters. No boundary or residual-based cleaning was used.

What is supported: source integrity, usable complete measurements in every exposure,
the specified relative observation and retained-input convention.
Unknown: actual optical templates/zeros, physical centers, common axial scale validity,
physiological states and full-estimator branch/rank behavior.
Deferred check: G7/G8 owns the effect of approximate boundaries and unreliable capture 1
timing; later reference/full-fit gates own the frozen weighting approximation.
One next action: G1 centroid bootstrap using this immutable population.
Last usable checkpoint: frame manifest `840e580212e1be89af1dbff7b7a79d8e9630c4143cad956d8048d0cf29beede5`.
Reviewer: Codex implementation and empirical audit. Decision timestamp: 2026-10-10T00:55:41.653065+00:00.

## G1 / attempt 02 / bootstrap_only

Status: COMPLETE. Decision: **GO_WITH_LIMIT**.

Source/theory/plan/config: same immutable evaluation snapshot as G0 above.
Parent gate/checkpoint: G0 attempt 02 and its exact frame manifest.
Question: Does the low-demand recording provide an invertible, ordered centroid
relationship for approximate visual gaze initialization?

Fixed: reference capture 1, Aref = 0.36036036036036034 D (recorded demand;
not measured physiological zero), provisional common scale g=1, degree 1, equal
weight for the five fixation means. Fitted: two inverse-bootstrap coefficients.
Derived: an individual gaze start from each complete frame's own centroid displacement;
accommodation starts use that frame's recorded demand. Those starts are not optimized
state estimates, temporal constraints, or the final D polynomial.

The initializer is:

```text
theta_start_deg = 0.04838772198701595 * (P4_centroid_x - P1_centroid_x - 51.79241949025186)
```

Its standardized design has rank 2 and condition number 1.0000000000000002.
The displacement means are strictly increasing with the targets; the inverse gain is
positive and finite. The five reference summaries are:

| Target (deg) | Valid frames | Mean dx (px) | dx spread (px) | Mean prediction (deg) | Residual (deg) |
|---|---:|---:|---:|---:|---:|
| -10 | 4433 | -155.303589582 | 10.636749323 | -10.020904112 | -0.020904112 |
| -5 | 4800 | -51.527630869 | 5.234183025 | -4.999421872 | +0.000578128 |
| 0 | 4900 | 52.108006094 | 3.619617353 | 0.015270517 | +0.015270517 |
| 5 | 4993 | 156.208899440 | 11.741795438 | 5.052475603 | +0.052475603 |
| 10 | 5154 | 257.476412368 | 5.576915383 | 9.952579865 | -0.047420135 |

These residuals describe the same five means used to fit the initializer. They are
not per-frame errors, independent accuracy estimates, or an empirical RMS gate.
Spatial means are calculated per frame, then temporally averaged within each full
reviewed interval; the stored framewise states use the individual displacement.

Population: 24,280 valid reference frames out of 24,700 scheduled; starts are saved for
all 89,175 complete valid scheduled frames across the twenty exposures. Every unavailable
row has a reason. No finite start is outside the proposed [-20,20] degree search box.
**8,022** starts fall outside the five-mean reference displacement range; these are
explicit extrapolations, including normal endpoint variation, and are not clipped or dropped.
Dynamic within-fixation variation is preserved. See
[bootstrap.json](bootstrap.json), [bootstrap plots](bootstrap.png),
[initial_states.npz](initial_states.npz), and [reproduction checks](reproduction_checks.json).

Raw transfer to captures 2–4 changes offset and gain, especially at positive gaze.
The plot shows this before accommodation and P1-scale correction. It limits transfer
accuracy but does not contradict using the reference relationship as a provisional start.
The five contiguous-block summaries and first/last-100-row summaries preserve variation
and boundary excursions; blocks are not independent physiological observations.

What is supported: an ordered, noncollapsed linear initializer for the next P1 step.
Unknown: scale-corrected gaze, the physical accommodation trajectory, optical zeros,
and whether capture/demand differences arise from accommodation, nuisance scale or pose.
Deferred check: G2 revisits the bootstrap with framewise P1 scale; G6/G7 revisit it with
center/mean corrections and refined states. G7/G8 revisit boundaries and extrapolated tails.
One next action: **G2/S2 P1 reference, independent symmetry evidence and P1-only common scale**.
Last usable checkpoint: population.npz plus initial_states.npz, state content hash
`b171db81fe620b2a6f879df13775fa58cd7e8efb9d779ea1e1bd6c37ebf5d478`; bootstrap-only, not a calibrated optical model.
Reviewer: Codex implementation and empirical audit. Decision timestamp: 2026-10-10T00:55:41.653065+00:00.
