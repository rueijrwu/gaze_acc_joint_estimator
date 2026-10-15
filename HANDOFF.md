# Native library handoff

The user resumed work to move the reusable C++/CUDA implementation out of the experiment. The project now builds from the repository root: `include/acc_inverse/`, `src/`, `apps/`, `tests/` and `cmake/`. The experiment retains its Python integration, inputs and scientific results. See [the library guide](docs/NativeLibrary.md) and [relocation evidence](experiments/p4_joint_frame_gaze_accommodation_inverse/results/native/reorganization/). Historical benchmark/checkpoint path strings remain unchanged as provenance; use the current paths below for new builds.

Relocation checks passed the fresh P100 build, all three CTests, installed external C++ consumption, a core-only build with HDF5 disabled, and Python discovery of the relocated build. A fresh standalone fit again accepted 159 iterations and returned RMS `0.6848295432534102` px, within `1.49e-13` px of the frozen reference. [Relocation summary](experiments/p4_joint_frame_gaze_accommodation_inverse/results/native/reorganization/summary.json) records the current paths, hashes and remaining verification evidence.

The preceding native inverse solver implementation reached a verified safe checkpoint. The original checkpoint and measurements below describe that implementation before relocation. No optical equations or solver behavior were changed by the move. After completing the requested reorganization, await the user's next instruction before starting further implementation or experiments. The normal GNOME GPU contexts are unrelated desktop processes.

Checkpoint recorded at `2026-10-15T06:25:00.704127+00:00`. Repository: `/home/aplab/ACC`; shell: bash. This handoff was written after verification; it did not rerun builds or tests. No commit was requested. The worktree contains broader prior cleanup changes: preserve them and do not reset or revert unrelated changes.

## User workflow for the next session

- The root agent implements and audits. Delegate **all mechanical tasks** to `gpt-6-luna` with **MAX** reasoning, per the user's latest instruction. Build with **16 jobs**, outside the sandbox.
- The user specifically requested `gpt-6-astra` with **MEDIUM** reasoning for this checkpoint handoff. Fast mode for subagents is represented by supported `priority` service-tier metadata; do not invent unsupported tool arguments.
- Python: `/home/aplab/.pyenv/versions/venv/bin/python`. CUDA compiler: `/usr/local/cuda-12.9/bin/nvcc`. Tesla P100 hardware is available; Blackwell hardware is not.
- Read the checkpoint and audit before acting after resumption. Ask what to resume if the user supplies no new scope. Do not propose a new optical experiment or treat Blackwell runtime benchmarking as an automatically authorized next task.

## Authoritative files and evidence

The active experiment is `experiments/p4_joint_frame_gaze_accommodation_inverse` (called `EXP` below).

| File under the repository | Purpose |
|---|---|
| [NATIVE.md](experiments/p4_joint_frame_gaze_accommodation_inverse/NATIVE.md) | Interfaces, scientific conventions, schema and reproduction commands |
| [NATIVE_RESULTS.md](experiments/p4_joint_frame_gaze_accommodation_inverse/NATIVE_RESULTS.md) | Final validation, timings, parameter tables and plots |
| [NativeLibrary.md](docs/NativeLibrary.md) | Build, API ownership, model extension and standalone CLI |
| [NativeAudit.md](docs/NativeAudit.md) | Root implementation and correctness audit |
| [CUDA_CXX_INVERSE_PLAN.md](docs/CUDA_CXX_INVERSE_PLAN.md) | Original design and acceptance criteria; final reports govern achieved results |
| [safe_checkpoint.json](experiments/p4_joint_frame_gaze_accommodation_inverse/results/native/validation_native/safe_checkpoint.json) | Source/binary/reference hashes, completed checks, log paths and idle-process evidence |
| [final_build_verification.json](experiments/p4_joint_frame_gaze_accommodation_inverse/results/native/final_build_verification.json) | Three fresh fits and final-build measurements |
| [validation.json](experiments/p4_joint_frame_gaze_accommodation_inverse/results/native/validation.json) and [validation_cpu.json](experiments/p4_joint_frame_gaze_accommodation_inverse/results/native/validation_cpu.json) | Full-data GPU and CPU audits |
| [p100-profile-notes.md](experiments/p4_joint_frame_gaze_accommodation_inverse/results/native/profiling/p100-profile-notes.md) | Profiling evidence and interpretation limits |

All native outputs live under `EXP/results/native`. Original CuPy inputs and results are preserved. `results/native/benchmark.json` remains the preserved benchmark baseline; final-build measurements are in `final_build_verification.json`.

Checkpoint SHA256: `2ef6250ab6ed220c3b73fd6c9b1edd923d249a0fb877408d933258c9ad6dd018`.

Final library `/tmp/acc-native-p100/libacc_inverse.so.1.0.0` SHA256: `cc5b4302907f8f54b2761ae87b95090e181d7a4880f632a5dcfe8b3ce56312d4`.

Final CLI `/tmp/acc-native-p100/acc_inverse_cli` SHA256: `426b0bee963f8fb97920804710a8b1af4a2f07afe587db5b99bcf525b118c65e`.

## Implemented architecture

The repository root is an installable C++17/CUDA FP64 library with a CUDA-style C ABI (`include/acc_inverse/api.h`), move-only C++ RAII wrapper (`acc_inverse.hpp`), typed statuses and original backend error codes, and optional callback logging. It exports CMake target `acc::inverse`. HDF5 belongs to the optional standalone CLI in `apps/`; the core library works without HDF5. The experiment's `native_bridge.py` provides ctypes host integration and asynchronous CuPy device buffers, including nondefault producer streams and lifetime tracking.

Pure shared optical code is separated into `scalar.hpp`, `optical_stages.hpp` and `model.hpp`. Compiled model policies support runtime selection of barrel → keystone or keystone → barrel; inversion automatically reverses the selected composition and includes model-centroid derivatives. The default scientific order is barrel → horizontal keystone → vertical keystone → centering. The symmetric kappa-relative barrel inverse uses a safeguarded numerical inner branch and implicit derivatives. The policy extension point supports compatible compiled optical laws; changing the current dimensions requires a new ABI: 16 shared parameters (including origins and phi), six residuals and one accommodation state per frame.

`solver_options.hpp` separates model composition from fit policy. Parameter masks, frozen accommodation, staged fits, LM damping, projected line search and tolerances are configurable. cuBLAS forms normals and Schur reductions; cuSOLVER FP64 POTRF/POTRS is preferred, with checked Eigen fallback. The persistent FP64 device workspace is approximately 118 MiB, with no per-iteration allocation. One workspace requires externally serialized operations and completion before reuse/destruction; independent workspaces may run independently.

## Scientific protocol to preserve

- 88,978 frames, 20 fixations, four captures and gaze positions `[-10, -5, 0, 5, 10]`. Both refined gaze coordinates, refined reference triangle Q and saved P1 correction are fixed.
- All 16 optical parameters are shared across every frame. Accommodation initializes from the 20 prior **mean-triangle fixed-model estimates**, repeated over each fixation's frames. Bounds are anchor ±0.25 D intersected with 0–5 D.
- Equal-fixation residual weights are `1/(60*N_g)`. There are no demand, offset, gaze or temporal penalties. Desired demand/cosine correction is reporting only.
- The 20 fixation offsets and zero-mean frame variations are an exact reporting decomposition of fitted accommodation, not additional frame biases or fit variables.
- The authoritative fixed triangle is `selected_model.cfg.reference_px`; acquisition reference metadata describes an older triangle.
- The reusable core accepts any nonnegative group ID and caller-supplied finite nonnegative accommodation bounds. The scientific HDF5 CLI separately enforces the canonical 20 fixations and 0–5 D limits.
- Both transform orders passed independent NumPy synthetic composition/derivative checks. The preserved real-data fit and comparison use the default barrel-first order.

## Completed verification and measured result

Final verification passed 3/3 CTests and CUDA compute-sanitizer with zero errors. CPU/GPU system audits cover all 88,978 frames, all 17 derivative columns, active bounds, frozen inference, HDF5 frame identities and CuPy nondefault-stream integration. Both transform orders, masks, accepted staged fits, typed errors, logging and RAII passed. An installed external C++ consumer configured, built and ran; a core-only build succeeded with HDF5 discovery disabled.

Three fresh final-build native fits each accepted 159 iterations. Initial inverse RMS was `0.904977458521` px; final native RMS was `0.684829543253410` px versus CuPy `0.684829543253261` px (about `1.49e-13` px difference). Median native solve time was `1.11953935` s. Final standalone HDF5 frozen-model inference reused native output with `--state fit` and returned zero invalid or ambiguous frames.

The controlled CuPy baseline measured `85.7945` s wall time, including `17.7234` s measured I/O and host-copy costs. Subtraction gives `68.0712` s estimated math time and an approximately `60.8×` ratio against the native solve. Checkpoint packing/decomposition remains in that estimate, and checkpoint cadences differ. **Do not describe this as a matched total-I/O wall-time speedup.**

Objective convergence is true; **frame-state-step stability remains false**. Neither objective convergence nor finite-grid inference proves unique or physiologically stable accommodation.

Blackwell compile-only verification embedded ELF targets `sm_100`, `sm_103`, `sm_120` and `sm_121`. There was no Blackwell execution or performance measurement. Future runtime benchmarking requires actual hardware and explicit user resumption.

Profiling predates the final API refactor. It supports the P100 thread-per-frame, 256-thread choice (about 11% better than warp-per-frame 128 for the measured comparison). CUDA Graph opportunity is limited by the measured launch headroom. Blocking CUDA API waits overlap GPU work; do not add those waits to GPU time as independent overhead.

## Reproduction reference — run only after user resumes

The relocation uses the fresh build directory `/tmp/acc-native-reorganized-p100`. Earlier `/tmp/acc-native-p100`, `/tmp/acc-native-blackwell-audit` and `/tmp/acc-native-library-only` caches point to the removed experiment-local source directory: do not reuse them for rebuilding. Temporary directories may not survive a future session. Use CUDA 12.9 for Pascal/P100 compilation; `build/p100` is the documented repository-local build tree.

From `/home/aplab/ACC`, delegate mechanical execution as specified above and run builds outside the sandbox:

```bash
EXP=experiments/p4_joint_frame_gaze_accommodation_inverse
PY=/home/aplab/.pyenv/versions/venv/bin/python
cmake -S . -B build/p100 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc \
  -DCUDAToolkit_ROOT=/usr/local/cuda-12.9 \
  -DCMAKE_CUDA_ARCHITECTURES=60
cmake --build build/p100 --parallel 16
ctest --test-dir build/p100 --output-on-failure
```

Use distinct output paths to preserve checkpoint artifacts. For example, after an explicitly requested reproduction:

```bash
mkdir -p /tmp/acc-native-resume
build/p100/acc_inverse_cli \
  --input "$EXP/results/native/inputs.h5" \
  --output /tmp/acc-native-resume/joint.h5 \
  --mode joint --state initial --maxiter 200 --device 0
build/p100/acc_inverse_cli \
  --input /tmp/acc-native-resume/joint.h5 \
  --output /tmp/acc-native-resume/inference.h5 \
  --mode infer --state fit --grid 33 --refine 96 --device 0
```

`--state fit` is essential when reusing the fitted optics. A warm joint restart uses explicit damping and does not reproduce an uninterrupted damping history. The documented Python audit/report commands and full final log list are in the linked native documentation and checkpoint manifest; there is no need to repeat completed checks merely because a new session begins.

Current action: **paused; await the user's next instruction.**
