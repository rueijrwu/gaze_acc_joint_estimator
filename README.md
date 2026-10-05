# XRecorder calibration

This repository contains the maintained calibration code, selected capture
detection data, fixation intervals, existing model artifacts, and research
notes. It is prepared to upload to GitHub and open in GitHub Codespaces.

## Open in a cloud environment

Create a GitHub repository from this folder and push its contents. On GitHub,
choose **Code → Codespaces → Create codespace on main**. The dev container
installs the pinned numerical dependencies and opens a shell at the repository
root. The same environment can be built locally with `docker build -t xrecorder .`.

To verify the environment after it opens:

```bash
python --version
python -c 'import numpy, scipy, matplotlib; print(numpy.__version__, scipy.__version__, matplotlib.__version__)'
```

The existing workflow documentation starts at [exp2/README.md](exp2/README.md).
The current scientific status, preserved results, and steps to revisit before
resuming work are in [documents/FULL_INFORMATION_HANDOFF.md](documents/FULL_INFORMATION_HANDOFF.md).
That handoff records that implementation and scientific runs are paused. The
container setup only installs dependencies; it does not run fits or diagnostics.

The four stored detection pickles are about 5.8–6.1 MB each. They contain
frame-aligned measurements and candidate detections from roughly 29,000 frames
per capture. Videos are not included. Files are kept as ordinary Git files; the
largest is below GitHub's 100 MiB per-file limit.

Redundant per-frame state CSV exports are omitted to keep the cloud checkout
small. Their canonical state arrays remain in the corresponding `checkpoint.npz`
files. Model metadata, histories, diagnostics, held-out predictions, and summary
tables remain available for resuming and reviewing the preserved runs.

## Local setup

Python 3.14.5 is pinned in the dev container. To create a matching environment
without Docker:

```bash
python3.14 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
```

Calibration uses the pinned NumPy, SciPy, and Matplotlib dependencies. Native
video detection additionally requires OpenCV and `pkj_image_process_py`, and
the source videos are not part of this repository.
