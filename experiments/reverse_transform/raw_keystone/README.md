# Raw-keystone reverse-transform chain

This chain recalibrates the reviewed captures with the raw projective keystone operator. Its transformed triangles retain their RMS-size response: there is no post-keystone RMS or area normalization. One positive magnification is profiled per frame from P1 and shared with P4. Historical normalized-keystone results are read only for comparison; they do not supply current coefficients or states.

## Stage order

| Stage | Folder | Role | Parent |
|---|---|---|---|
| 1 | [Capture 1 P1](./stage_01_capture1_p1/README.md) | Fresh Capture 1 gaze/reference/P1 calibration | None |
| 2 | [Capture 1 P4](./stage_02_capture1_p4/README.md) | Fit Capture 1 P4 with the shared P1 scale | Stage 1 |
| 3 | [Independent captures](./stage_03_independent_captures/README.md) | Extend independent P1/P4 fits across captures | Stages 1–2 |
| 4 | [Framewise accommodation](./stage_04_framewise_accommodation/README.md) | Fit one forward-model A value per complete frame with Stage 3 optics frozen | Stage 3 |
| 5 | [Capture 1 state ablation](./stage_05_capture1_state_ablation/README.md) | Compare fixed Stage 4 states (A0), free horizontal gaze (G), and free gaze plus A (GA) with raw optics frozen | Stages 3–4 |
| 6 | [Framewise centers](./stage_06_framewise_centers/README.md) | Derive framewise centers, recalibrate gaze from fixation-mean corrected separations, reprofile scale, and update A-only with gaze fixed | Stages 3–4 |

Stages 1–4 use thin wrappers that select a stage number and forward arguments to the shared scripts in this directory. Stages 5 and 6 have their own run, audit, and plot scripts. Stage 5 compares three joint-state models; Stage 6 alternates center-derived gaze recalibration with a fixed-gaze A-only fit. Each stage writes a run under its own `results/run/`, with frame arrays, protocol, provenance, historical comparison, and audit output. Run and audit before plotting; plotting reads saved results only.

Scientific conclusions belong in a reviewed results report after the corresponding saved-results audit passes. No physical or physiological interpretation follows from in-sample residual reduction alone.
