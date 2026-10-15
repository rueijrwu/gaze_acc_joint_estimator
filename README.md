# ACC inverse optics

ACC provides a C++17/CUDA FP64 library for inverse barrel–keystone calibration. It exposes a C API and C++ RAII interface, configurable transformation order and staged fitting, with cuBLAS/cuSOLVER computation and CPU validation through Eigen. The installable CMake target is `acc::inverse`.

```text
CMakeLists.txt               Library, applications, tests and installation
include/acc_inverse/         Public API and reusable optical model headers
src/                        Library implementation and CUDA kernels
apps/                       Optional HDF5 command-line application
tests/                      C++ unit and application integration tests
cmake/                      Package configuration and generated build metadata
docs/                       Theory, library guide and implementation audit
experiments/                Calibration datasets, Python clients and results
```

Build from the repository root (CUDA 12.9 for P100):

```bash
cmake -S . -B build/p100 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc \
  -DCUDAToolkit_ROOT=/usr/local/cuda-12.9 -DCMAKE_CUDA_ARCHITECTURES=60
cmake --build build/p100 --parallel 16
ctest --test-dir build/p100 --output-on-failure
```

[Library/API guide](docs/NativeLibrary.md) · [C API](include/acc_inverse/api.h) · [C++ interface](include/acc_inverse/acc_inverse.hpp) · [Implementation audit](docs/NativeAudit.md)

The optional `acc_inverse_cli` application reads and writes HDF5. Set `ACC_BUILD_CLI=OFF` to build the library without HDF5. GPU architecture is selected through `CMAKE_CUDA_ARCHITECTURES`; Blackwell compilation is verified, while hardware performance remains to be measured there.

The active calibration experiment is [joint inverse barrel–keystone calibration](experiments/p4_joint_frame_gaze_accommodation_inverse/README.md). It fixes refined per-frame gaze and jointly fits shared optics and accommodation bounded to ±0.25 D around each fixation's mean-triangle estimate, without a demand penalty.

[Native results and parameter tables](experiments/p4_joint_frame_gaze_accommodation_inverse/NATIVE_RESULTS.md) · [Experiment HDF5 interface](experiments/p4_joint_frame_gaze_accommodation_inverse/NATIVE.md) · [Accommodation over time](experiments/p4_joint_frame_gaze_accommodation_inverse/results/native/accommodation_over_time.png) · [Preserved CuPy reference](experiments/p4_joint_frame_gaze_accommodation_inverse/RESULTS.md)

```bash
pyenv activate venv
python experiments/p4_joint_frame_gaze_accommodation_inverse/native_run.py convert
python experiments/p4_joint_frame_gaze_accommodation_inverse/native_run.py run --mode joint --maxiter 200
python experiments/p4_joint_frame_gaze_accommodation_inverse/native_audit.py
```

The experiment's Python clients discover `build/p100` and `build` automatically. `ACC_INVERSE_LIBRARY` and `ACC_INVERSE_EXECUTABLE` select another build or installation. Complete optimization and frozen-model scalar inversion run natively; Python converts the frozen NPZ bundle once and validates or plots the HDF5 result. NumPy, h5py, Matplotlib and CuPy are installed in the existing `venv` environment. The original `run.py` remains available for controlled reference reproduction.

- [Theory V2](docs/Theory_V2.md) describes the mean-triangle calibration model; the experiment README defines its frame-fit extension.
- [Simulation model](docs/SimulationModel.md) records the ray-tracing evidence.
- [C++/CUDA plan and implementation status](docs/CUDA_CXX_INVERSE_PLAN.md) cover the math, setup and optimization protocol for P100 and Blackwell.
- [Data](data/) retains raw detections, fixation intervals and reusable P1/P4 scales.
- [Archive](experiments/archive/README.md) contains earlier experiments and the retired estimator library.
