# Progress

Current gate: G1/S1 COMPLETE, GO_WITH_LIMIT; G0 COMPLETE, GO_WITH_LIMIT.
Last reviewed attempt: [results/attempt_02/STAGE_REPORT.md](../results/attempt_02/STAGE_REPORT.md).
Last usable checkpoint: [results/attempt_02/population.npz](../results/attempt_02/population.npz),
[bootstrap.json](../results/attempt_02/bootstrap.json), and
[initial_states.npz](../results/attempt_02/initial_states.npz).
Source parent: `efd88bd3b326653aedc3b383143ef10856d22092`; source content: `c43a344ae12e99a5a92f8cfa58162a4923563c014c8e78689a54c04fb86e549b`.
Population content: `840e580212e1be89af1dbff7b7a79d8e9630c4143cad956d8048d0cf29beede5`.

Limitations: capture 1 timestamps are unreliable; interval boundaries include transitions;
noise weights are an approximation from reliable captures; g=1 bootstrap transfer is
approximate and 8,022 starts extrapolate the reference-mean range. No optical model has
been calibrated. Preserve all scheduled rows and deferred G2/G6/G7/G8 checks.

One authorized next action: **G2/S2 P1 reference and scale**, beginning with source-based
symmetry evidence and fixed reference/length/origin conventions. This stage-one task ends
here; the checkpoint authorizes the next calculation without claiming physiological accuracy.
