# Native inverse optics

The repository root is the C++17/CUDA library project. Public headers live in [include/acc_inverse](../include/acc_inverse/), implementation and kernels in [src](../src/), the optional HDF5 application in [apps](../apps/), and native tests in [tests](../tests/). Package configuration belongs to [cmake](../cmake/). Calibration data, Python clients and scientific results remain in [experiments](../experiments/).

The standalone executable evaluates the FP64 optical model, runs the complete bounded joint fit or frozen-model scalar inversion, and reads/writes HDF5. Python is optional for conversion, integration checks and plots. The optical evaluator, derivatives and optimization are implemented in the reusable library.

## Build

Use CUDA 12.9 for Tesla P100; CUDA 13 removes Pascal compilation support. The core build needs Eigen 3.4. The optional standalone CLI additionally needs HDF5 development libraries. Use a separate build tree such as the ignored `build/p100` directory or a directory outside the repository.

```bash
cmake -S . -B build/p100 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc \
  -DCUDAToolkit_ROOT=/usr/local/cuda-12.9 -DCMAKE_CUDA_ARCHITECTURES=60
cmake --build build/p100 --parallel 16
ctest --test-dir build/p100 --output-on-failure
```

For a portable CUDA 12.9 build, set `-DCMAKE_CUDA_ARCHITECTURES='60-real;100-real;103-real;120-real;121-real;100-virtual'`. For Blackwell-only CUDA 13 builds select the actual compiler, toolkit and GPU architecture. Host binaries must be built separately for x86-64 and Arm64. No architecture-specific `a`/`f` instructions, tensor cores, FP32 or fast math are used.

Set `ACC_BUILD_CLI=OFF` to build the core library without finding or linking HDF5. The reusable CMake package exports `acc::inverse`:

```cmake
find_package(acc_inverse CONFIG REQUIRED)
target_link_libraries(my_calibrator PRIVATE acc::inverse)
```

Install a build with `cmake --install build/p100 --prefix /tmp/acc-native-install`; set `CMAKE_PREFIX_PATH` to that prefix for consumers. Public headers include the C ABI, move-only C++ workspace, typed errors, solver controls and optical stages. Existing build caches created from the old experiment-local source directory must be replaced with a fresh build tree.

`ACC_CUDA_FMAD=ON` allows FP64 fused multiply-add. `ACC_CUDA_RESOURCE_REPORT=ON` prints compiler register/spill reports. Runtime metadata records compiler/toolkit/device/architecture, actual launch settings and allocated device memory. The P100 launch default is the measured thread-per-frame derivative kernel with 256 threads per block; other devices use the generic warp-per-frame, 128-thread path until measured independently. Both paths are configurable.

## Run without Python

An existing version-1 HDF5 input can be reused directly:

```bash
build/p100/acc_inverse_cli --input inputs.h5 --output joint.h5 \
  --mode joint --state initial --maxiter 200
build/p100/acc_inverse_cli --input joint.h5 --output inference.h5 \
  --mode infer --state fit --grid 33 --refine 96
build/p100/acc_inverse_cli --input inputs.h5 --output replay.h5 \
  --mode evaluate --state reference --derivatives
```

`--device -1` uses the scalar CPU implementation for diagnostics. `--solver cusolver` prefers cuSOLVER FP64 POTRF/POTRS with symmetric diagonal equilibration and original-system residual checks; Eigen LDLT is its diagnostic fallback and the CPU reference. `--solver strict` requires cuSOLVER; `--solver eigen` forces Eigen. Metadata counts the selected factorization calls. `--threads N --derivative-mode thread|warp` override launch configuration. A fresh fit starts with damping `0.001`; `--damping` explicitly controls a warm restart from `--state fit`.

`--order barrel-keystone` preserves the reference optical model. `--order keystone-barrel` composes keystone before barrel and automatically reverses the inverse sequence. Both modes recompute the predicted centroid and its derivatives. These are different optical models; the saved-reference numerical benchmark applies to the default order. The actual order is saved in `/config/transform_order` and metadata; subsequent native-fit reuse reads it automatically unless `--order` explicitly overrides it.

The transform order and fitting sequence are independent. Shared-parameter masks and frozen accommodation implement staged fitting:

```bash
build/p100/acc_inverse_cli --input inputs.h5 --output barrel-stage.h5 \
  --mode joint --state initial --shared-mask barrel --freeze-accommodation
build/p100/acc_inverse_cli --input barrel-stage.h5 --output keystone-stage.h5 \
  --mode joint --state fit --shared-mask keystone --freeze-accommodation
build/p100/acc_inverse_cli --input keystone-stage.h5 --output joint-stage.h5 \
  --mode joint --state fit --shared-mask all
```

`--shared-mask` accepts `barrel`, `keystone`, `all`, `none`, or a bit mask such as `0x7000`. Barrel selects coefficient/origin slots 12–14; keystone selects slots 0–11 and orientation 15. A frozen parameter is retained exactly, including an orientation outside the optimization interval. Debug Jacobians retain all 17 columns after every staged fit. `--freeze-accommodation` retains every frame state exactly.

The versioned `AccSolveSettings` descriptor exposes masks, accommodation fitting, damping updates/attempts, line-search scales and stopping tolerances. Initialize it with `acc_solve_settings_default`, then customize fields. The CLI also exposes `--line-search`, `--damping-attempts`, `--damping-decrease`, `--damping-increase`, `--minimum-damping`, `--projected-gradient-tolerance`, `--objective-tolerance`, `--state-step-tolerance`, `--sustained-gradient-tolerance` and `--sustained-iterations`. Defaults reproduce the reviewed solver policy.

Joint fitting retains the reference active-bound rule, eight damping attempts, six projected line-search scales, orientation bound, accommodation clipping and stopping criteria. Final gradients are recomputed at the returned state and included in the reported solver time. Objective convergence and frame-state step stability remain separate diagnostics. The accommodation offset/variation decomposition is computed after solving; it adds no fit variables or penalty.

Frozen-model inference tests both endpoints and every valid coarse local minimum, then refines each valid bracket independently. It never walks through an observed invalid interval. Ambiguity and sensitivity are diagnostics at the specified coarse-grid resolution; a finite grid cannot certify global uniqueness or detect an arbitrarily narrow valid island. Bound flags identify selected endpoints exactly. A failed frame returns validity 0 and an infinite cost; an ambiguous frame returns validity 2. The inference CLI preserves these diagnostics even when some frames fail.

Output uses a temporary sibling file followed by an atomic rename. Input and output paths must differ. The executable validates schema version, types, shapes, all 20 groups/counts, equal-fixation weights, finite inputs and mean-anchor bounds before computation. Existing CuPy inputs and results are never modified.

## HDF5 schema

Root attribute `schema_version` is integer 1. All floating datasets are FP64.

| Dataset | Shape / meaning |
|---|---|
| `/config/R`, `/config/aref`, `/config/span` | Scalars |
| `/config/reference_px` | `[3,2]`, fixed reference in pixels |
| `/config/transform_order` | Optional scalar int32: 0 barrel→keystone, 1 keystone→barrel; absent means 0 |
| `/inputs/observed_normalized` | `[N,3,2]`, already centered and P1 corrected, divided by R once |
| `/inputs/gaze_deg` | `[N,2]`, both fixed gaze axes |
| `/inputs/lower_D`, `/inputs/upper_D` | `[N]`, mean-triangle anchor ±0.25 D intersected with 0–5 D |
| `/inputs/weights` | `[N]`, `1/(60*count[group])` |
| `/inputs/group` | `[N]` int32, labels 0–19 |
| `/inputs/demand_D` | `[N]`, reporting reference only |
| `/initial/params`, `/reference/params` | `[16]`, initial and saved-reference optics |
| `/initial/accommodation_D`, `/reference/accommodation_D` | `[N]`, corresponding states |
| `/groups/counts`, `/groups/mean_anchor_D` | Optional `[20]` consistency checks |
| `/provenance/manifest_json` | UTF-8 source manifest |

Additional frame identifiers in `/inputs`, plus `/config`, `/groups` and `/provenance`, are copied into output. `/fit` contains `params`, `accommodation_D`, `corrected_normalized`, `predicted_normalized`, `residual_px`, `valid`, `fixation_offsets_D`, and `accommodation_variation_D`. `--derivatives` also saves `jacobian_px` with shape `[N,3,2,17]`. Native outputs are valid inputs using `--state fit`.

`/iterations` contains one dataset per `AccIteration` field, `/summary` contains scalar/string attributes, and `/profile` contains cumulative CUDA event durations and counters. Frozen inference adds `/inference/{cost_px2,valid,bound,sensitivity_px_per_D,ambiguity_gap_px2,minima_count}`. Infinite cost/gap values are intentional diagnostics.

## C ABI and device buffers

[`include/acc_inverse/api.h`](../include/acc_inverse/api.h) specifies ABI version 1. C++ owns persistent GPU allocations. Public host arrays are contiguous row-major. Corrected/predicted coordinates are normalized; residuals and all 17 derivatives are in corrected pixels. Derivative slots are the 16 shared parameters followed by accommodation; gaze derivatives are deliberately omitted.

`acc_normals` and `acc_step` expose active-bound blocks and the actual Schur/linear-solve path for independent comparisons. Host evaluation can use the CPU or CUDA backend on the same workspace. C ABI exceptions become typed `AccStatus` codes, `acc_last_error()` messages and original CUDA/cuBLAS/cuSOLVER codes via `acc_last_backend_code()`. Error queries preserve the most recent diagnostic until the next operation. Logging is disabled by default; a native callback can receive operation and iteration events without library file I/O, and callback exceptions are contained. The default solver path has no Python callbacks.

`acc_inverse/acc_inverse.hpp` provides `acc_inverse::Workspace`, a move-only RAII wrapper over the same core, plus `acc_inverse::Error` carrying typed status/backend codes. C++ clients can evaluate, assemble normals, inspect a Schur step, fit a sequence of `AccSolveSettings`, or infer accommodation directly. The core accepts nonnegative group IDs and caller-supplied finite nonnegative accommodation bounds; the scientific HDF5 CLI separately enforces this experiment's 20 fixations and 0–5 D bounds.

`acc_evaluate_device` accepts CUDA pointers and a stream directly, allowing CuPy integration without host array transfers. Pointer ownership and stream device are checked. It is asynchronous; the caller owns buffer lifetimes and must synchronize before another operation on the same workspace. All operations and destruction on one workspace require external serialization; independent workspaces are independent.

GPU storage uses SoA observations, weighted Jacobians laid out for cuBLAS, scalar frame cross blocks and ping-pong accommodation/trial buffers. Optical/root/derivative/bound kernels are custom CUDA; dense Gram, Schur and gradient operations use cuBLAS, and the preferred dense factorization uses cuSOLVER. Radial roots are safeguarded on the device and differentiated implicitly, never through numerical iterations. Full-sized arrays remain allocated across all iterations.

Native CTest executables cover optical branches, nonzero vertical gaze, inverse closure, all 17 finite-difference columns, dense/Schur agreement, outward active bounds, scalar endpoint minima/flat ambiguity, C ABI input rejection, CPU/CUDA parity and HDF5 round trips including native-output reuse. The CUDA test skips when no GPU is accessible; it must be run with GPU access on a CUDA validation machine.

## Model extension points

`scalar.hpp` supplies the small directional dual and point types. `optical_stages.hpp` contains pure `BarrelStage`, `KeystoneStage` and `CenteringStage` operations. Barrel inversion owns the safeguarded branch and implicit derivative; keystone owns analytic forward/inverse projective transforms; centering owns centroid calculation and restoration.

`model.hpp` keeps parameter layout and accommodation law in `RelativeBarrelKeystonePolicy<Order>`. The `ModelPolicy<Order>` alias is the localized compiled model extension point. A compatible policy provides compile-time parameter/residual/state counts plus primal evaluation and directional derivatives. CPU and CUDA consumers share it, while the solver only consumes residual/normal blocks and `SolveControl`. The current ABI deliberately fixes dimensions to 16 shared parameters, six residuals and one scalar frame state; a model with different dimensions needs an explicitly versioned ABI.

Both transform orders are instantiated as CUDA kernels and selected on the host. Adding or changing an optical law or composition therefore does not require editing Schur elimination, the factorization backend or line-search logic. `solver_options.hpp` keeps fitting behavior independent of optical composition, and repeated warm calls with different controls implement a fit sequence.
