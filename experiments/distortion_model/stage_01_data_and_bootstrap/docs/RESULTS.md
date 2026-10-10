# Stage 1 result

The inputs and the linear centroid initializer are usable for the next P1 reference
step. G0 and G1 both received **GO_WITH_LIMIT** after implementation and empirical
audit. This supports initialization; no calibrated gaze or accommodation result is
claimed.

| Recorded evidence | Result |
|---|---:|
| Reviewed exposures | 20, captures 1–4 |
| Scheduled frames | 100,090 |
| Complete valid measurements / provisional starts | 89,175 |
| Unavailable rows retained | 10,915 |
| Explicit P4 omission slots | 300,270 |
| Permanent tests | 17 passed, 0 failures/errors/skips |
| Run time, including files and plots | 6.68 seconds |

The gaze initializer uses the horizontal difference between the P4 and P1 centroids:

```text
theta_start_deg = 0.04838772198701595 * (P4_centroid_x - P1_centroid_x - 51.79241949025186)
```

For example, a centroid difference of approximately 51.79 pixels gives an initial
visual gaze near zero degrees. The five reference fixation means increase consistently
with nominal gaze, so this line supplies a usable initial mapping. Each frame receives
its own value. Variation inside a fixation is preserved. Accommodation starts are
simply the recorded demands, not fitted accommodation values.

Capture 1 has 601 backward timestamp jumps in the source, 550 within the reviewed
intervals. Its rows and frame identities remain intact; its unreliable timing was
excluded from the covariance estimate. The reviewed interval edges contain transition
excursions and remain included. Applying the reference inverse to other captures is
approximate: scale, accommodation and pose corrections are still missing. Of the
finite starts, 8,022 extrapolate beyond the reference-mean displacement range; none
exceed the proposed [-20,20] degree search box. These are flags, not rejection rules.

The numerical tests verify relative coordinates, covariance, masks, source identities
and the synthetic forward calculation. They do not establish physiological accuracy.
The three omission slots per frame are a complete future test schedule; their state
inference is explicitly marked not run.

See the [detailed gate audit](../results/attempt_02/STAGE_REPORT.md),
[bootstrap plots](../results/attempt_02/bootstrap.png),
[native tracks](../results/attempt_02/native_tracks.png), and
[reproduction checks](../results/attempt_02/reproduction_checks.json).
The one next action is **stage 2 / S2 / G2: P1 reference and common scale**, as recorded
in [PROGRESS.md](PROGRESS.md).
