# exp5 single-estimator package

This folder is a compact, standalone snapshot of the exp5 gaze/accommodation estimator and its capture 5/6 comparison plots. It contains the six stored detection outputs, capture 1 fixation intervals used for the gaze-only calibration, the frozen converged quadratic model, the estimator scripts, and the Python modules needed to import the estimator without the surrounding repository.

See [HANDOFF.md](HANDOFF.md) for the current baseline, consolidated measurement theory, validation limits, and proposed full-position estimator work.

The `.pkl` files contain detector outputs; the `.mkv` source videos are not needed to run this package. The estimator consumes the stored detections directly and does not rerun detection.

## Run

From this folder, install the listed Python dependencies and run:

```bash
python apply_quadratic_captures_5_6.py
python compare_gaze_corrections.py
```

The first command estimates gaze and accommodation for every valid detection in captures 5 and 6 and writes state CSVs, summaries, and trace plots to `experiments/captures_5_6_direct/`. The second command fits the capture 1 linear gaze baseline and writes comparison CSVs and figures to `experiments/gaze_linear_vs_corrected/`.

The committed `experiments/` outputs are the latest results from the source exp5 workspace. Running the commands regenerates them. The comparison figures show the estimator discrepancy between the linear baseline and joint-model gaze; captures 5 and 6 have no reviewed gaze or accommodation reference labels.

## Contents

- `data/detections/`: `capture_1` through `capture_6` stored detections.
- `data/fixations/fixation_intervals.json`: reviewed fixation intervals and nominal labels for captures 1–4.
- `models/quadratic_model.json`: converged 13-coefficient joint gaze/accommodation model.
- `apply_quadratic_captures_5_6.py`: bounded inverse applied independently to valid rows in captures 5 and 6.
- `compare_gaze_corrections.py`: capture 1 linear calibration and capture 5/6 comparison plots.
- `joint_m2.py` and `lib/`: measurement, inverse, and supporting model/calibration code.
- `experiments/`: copied state estimates, summaries, calibration/comparison CSVs, and plots.

The model was trained on reviewed fixation data from captures 1–4. Its nominal accommodation labels are calibration anchors, not measured accommodation ground truth. The per-frame optical fit and the capture 5/6 comparison do not establish physiological accuracy.
