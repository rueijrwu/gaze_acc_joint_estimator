# C++/CUDA inverse-distortion solver plan

Status: native C++17/CUDA FP64 solver implemented and validated on Tesla P100, 2026-10-15. The reusable library is now the repository-root CMake project, with conventional `include/`, `src/`, `apps/`, `tests/` and `cmake/` directories. The standalone executable reads/writes HDF5 and performs the complete optimization without Python callbacks. [Native results](../experiments/p4_joint_frame_gaze_accommodation_inverse/NATIVE_RESULTS.md) record full saved-model replay, all 17 fitted derivatives, fresh-fit parity, measured P100 timings and plots; [the library guide](NativeLibrary.md) and [experiment interface](../experiments/p4_joint_frame_gaze_accommodation_inverse/NATIVE.md) give build, schema and runtime commands. Blackwell runtime performance requires a Blackwell machine; no local speed is inferred from cross-compilation.

## Implementation status

- Exact independent CPU model and CUDA evaluator, implicit barrel derivatives, both gaze axes and inverse branch checks are implemented.
- Persistent device workspaces, cuBLAS normal/Schur reductions, checked cuSOLVER POTRF/POTRS and an Eigen diagnostic/fallback are implemented. The original host-Eigen design below is retained as design context; the current default follows the requested official CUDA-library solver path.
- A reusable C ABI with typed status/backend errors, optional native logging and a C++17 RAII wrapper separates data ownership from compiled model policies and solver settings. Runtime transform order and staged barrel/keystone/joint masks are explicit. The installable `acc::inverse` CMake target does not require HDF5 when the optional CLI is disabled.
- The native damping/line-search hot loop and bounded frozen-model scan/refinement are implemented. Both matched fresh fits take 159 accepted iterations and agree to approximately 1.5×10⁻¹³ px RMS on P100.
- HDF5 input conversion preserves the five local source hashes, frame identifiers, exact fixed gaze, mean-triangle anchors, weights and bounds. Native output, plots, NPZ/pickle exports and benchmark checkpoints are isolated in `results/native/`.
- Python system validation checks 88,978-frame saved-model replay, initial/saved/random-safe states, all 17 derivative columns, real-data active sets, independent scalar inversion, checkpoint reload, HDF5 mapping and CuPy device buffers on a nondefault stream. C++ CTest owns fine-grained model and linear-algebra unit cases.
- Fixed-state warmed repetitions and controlled fresh solves separate native compute, reference checkpoint/log costs and HDF5 process timing. The unchanged reference input and scientific result files remain preserved.

The numbered sections below retain the numerical specification and engineering rationale used to implement the solver. The maintained native source and measured reports are authoritative for current implementation details.

## 1. Scope and fixed reference implementation

Port [the current experiment](../experiments/p4_joint_frame_gaze_accommodation_inverse/README.md), including numerical inversion and joint calibration, into a native C++17/CUDA library. The standalone runtime reads/writes HDF5. Python converts the existing frozen NPZ bundle once and produces validation, Markdown tables and 4×1 plots. The iterative solver and model evaluation run in C++/CUDA without Python callbacks.

The reference dataset has 88,978 frames, 20 capture–gaze fixations, three P4 vertices per frame, and 16 shared optical parameters. Both frame gaze coordinates and the reference triangle are fixed. Accommodation is bounded to ±0.25 D around each fixation's saved mean-triangle estimate, intersected with 0–5 D. Preserve the saved P1 scale correction, frame selection, weights and accepted calibration offsets.

Provide two modes using the same inverse evaluator:

1. **Joint calibration:** optimize the 16 shared optical parameters and bounded frame accommodation, matching the current experiment.
2. **Frozen-model accommodation estimation:** hold the fitted optics and input gaze fixed and solve one scalar accommodation problem per frame. This is numerical inversion of the calibrated model, with no newly trained estimator. Use the same bounds for direct comparison with joint calibration; any wider inference interval must be an explicit setting.

The current saved inverse RMS is **0.684829543253261 px**. The recorded calibration plus final numerical verification took **84.797 s for 159 iterations** on P100; that timing excludes the subsequent plot/export stage. This is a historical observation, not a controlled native/CuPy benchmark. Objective convergence was reported, while frame-state step stability was false; a native port must not turn objective convergence into a claim that every accommodation estimate is unique or stable.

## 2. Setup for P100 and Blackwell

The following environment was inspected locally:

| Component | Installed/observed |
|---|---|
| GPU | Tesla P100-PCIE-16GB, compute capability 6.0 |
| Driver | 580.178.04 |
| CUDA compiler | `/usr/local/cuda-12.9/bin/nvcc`, release 12.9.86 |
| Host compiler | GCC/G++ 13.3.0 |
| CMake | 3.28.3 |
| Eigen | 3.4.0, headers `/usr/include/eigen3` |

A temporary compiled probe ran FP64 CUDA arithmetic and an Eigen 16×16 solve successfully. This verifies the local toolchain, not the future solver. Blackwell execution has not been tested here.

Use two build profiles from one source tree:

| Profile | CUDA/toolchain policy | GPU targets |
|---|---|---|
| Common portable build | CUDA 12.9, including the installed toolchain | P100 plus supported Blackwell targets |
| Blackwell performance build | A pinned, tested CUDA 13.x toolchain on the Blackwell machine | The actual Blackwell device; no P100 target in this build |

P100 is `sm_60`. Blackwell includes multiple compute capabilities: B200/GB200 are 10.0, B300/GB300 are 10.3, RTX 5090 and RTX PRO 6000 Blackwell are 12.0, and GB10/DGX Spark is 12.1. Query the actual device rather than treating Blackwell as one target. These mappings follow NVIDIA's [current GPU table](https://developer.nvidia.com/cuda/gpus) and [legacy GPU table](https://developer.nvidia.com/cuda/gpus/legacy).

CUDA 12.9 supports the relevant Pascal and Blackwell compiler targets. CUDA 13.0 removed offline compilation/library support for Pascal, so retain CUDA 12.x for the P100 build. See the [CUDA 12.9 compiler targets](https://docs.nvidia.com/cuda/archive/12.9.2/cuda-compiler-driver-nvcc/index.html) and [CUDA 13.0 architecture changes](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html#deprecated-architectures). GCC 13 is inside the [CUDA 12.9 supported compiler range](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-installation-guide-linux/index.html#host-compiler-support-policy).

Record the device name, compute capability, CPU architecture, driver, CUDA runtime/compiler, Eigen version and compiler flags with every benchmark. Build separately for x86-64 hosts and Arm64 hosts such as Grace/GB10 systems; shared source does not imply a shared host binary.

Planned CMake setup:

```cmake
cmake_minimum_required(VERSION 3.24)
project(acc_inverse LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
find_package(CUDAToolkit 12.9 REQUIRED)
find_package(Eigen3 3.4 REQUIRED NO_MODULE)

add_library(acc_inverse SHARED
    src/model_cpu.cpp src/solver.cpp src/api.cpp
    src/model_cuda.cu)
target_include_directories(acc_inverse PUBLIC include)
target_link_libraries(acc_inverse PRIVATE Eigen3::Eigen CUDA::cudart CUDA::cublas CUDA::cusolver)
target_compile_options(acc_inverse PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
```

Example commands **after the planned source files exist**, from the repository root:

```bash
# P100 build on this machine, from the repository root.
cmake -S . -B build/p100 -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc \
      -DCUDAToolkit_ROOT=/usr/local/cuda-12.9 \
      -DCMAKE_CUDA_ARCHITECTURES=60
cmake --build build/p100 --parallel 16

# Common CUDA 12.9 build: native code for known devices plus generic Blackwell PTX.
cmake -S . -B build/portable -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc \
      -DCUDAToolkit_ROOT=/usr/local/cuda-12.9 \
      '-DCMAKE_CUDA_ARCHITECTURES=60-real;100-real;103-real;120-real;121-real;100-virtual'
cmake --build build/portable --parallel 16
```

For Blackwell-only builds, select the installed CUDA 13.x compiler and toolkit root explicitly and use the actual device architecture. Include generic PTX where forward compatibility is wanted; architecture-specific `a`/`f` features need separate dispatch and compatibility checks. The initial solver needs no such features. See [CMake architecture selection](https://cmake.org/cmake/help/v3.28/prop_tgt/CUDA_ARCHITECTURES.html) and NVIDIA's [Blackwell compatibility guide](https://docs.nvidia.com/cuda/archive/12.9.2/blackwell-compatibility-guide/index.html).

Use Eigen for host-side small dense linear algebra. Keep Eigen factorization in `.cpp` translation units; use explicit small-vector operations in CUDA kernels. Eigen also supports fixed-size device expressions, but they are unnecessary for the first implementation. See [Eigen CUDA guidance](https://libeigen.gitlab.io/eigen/docs-nightly/TopicCUDA.html).

## 3. Exact optical model to port

The executable specification is [optics.py](../experiments/p4_joint_frame_gaze_accommodation_inverse/optics.py). Use its full two-axis convention, including the small, fixed vertical gaze values.

### Coordinates and shared parameters

Let \(Q_j\) be the fixed reference vertices divided by \(R\), and let

\[
Y_{f,j}=\frac{P4_{f,j}-\operatorname{mean}_j P4_{f,j}}{m_{P1,f}R},
\qquad a_f=\frac{A_f-A_{ref}}{\Delta A},
\qquad t_x=\theta_{x,f}/10^\circ,\quad t_y=\theta_{y,f}/10^\circ.
\]

The saved `frames.npz` observations already include centering and P1 scale correction; divide them by \(R\) once, without repeating those corrections. Current constants are \(R=191.56618404715252\) px, \(A_{ref}=0.36036036036036034\) D, and \(\Delta A=3.6396396396396398\) D. Read these from the saved configuration rather than hard-coding them in the library.

| Parameter slots | Meaning |
|---|---|
| 0–1 | \(\ell_x(a)=p_0+p_1a\) |
| 2–3 | \(\alpha_x(a)=p_2+p_3a\) |
| 4–5 | \(\ell_y(a)=p_4+p_5a\) |
| 6–7 | \(\alpha_y(a)=p_6+p_7a\) |
| 8–9 | \(\gamma(a)=p_8+p_9a\) |
| 10–11 | Keystone origin \(O_K\), normalized coordinates |
| 12 | Relative barrel coefficient \(\beta=p_{12}\), with \(k(A)=\beta a\) |
| 13–14 | Barrel origin \(O_B\), normalized coordinates |
| 15 | Image-axis orientation \(\phi\), radians |

All 16 parameters are shared across frames, captures and gaze. The two origins can differ. The physical barrel slope is \(\beta/(R^2\Delta A)\) in px\(^{-2}\)/D. There is no independent frame barrel or keystone coefficient.

### Forward transform: barrel, keystone, centering

First apply relative symmetric barrel:

\[
B_A(Q)=Q+k(A)\|Q-O_B\|^2(Q-O_B).
\]

Rotate around the keystone origin: \(z=R_\phi^T[B_A(Q)-O_K]\). Define

\[
s_x=1+\ell_x(a)t_x+\alpha_x(a)t_x^2,\quad
s_y=1+\ell_y(a)t_x+\alpha_y(a)t_x^2,\quad q_x=\gamma(a)t_x,
\]
\[
h_x=\frac{s_xz_x}{1+q_xz_y},\qquad
h_y=\frac{s_yz_y}{1+q_xz_y}.
\]

The existing vertical convention reuses that law after a 90° rotation:

\[
v_x=1+\ell_x(a)t_y+\alpha_x(a)t_y^2,\quad
v_y=1+\ell_y(a)t_y+\alpha_y(a)t_y^2,\quad q_y=\gamma(a)t_y,
\]
\[
o_x=\frac{v_yh_x}{1-q_yh_x},\qquad
o_y=\frac{v_xh_y}{1-q_yh_x}.
\]

Finally,

\[
F_{f,j}=O_K+R_\phi o_{f,j},\qquad
\mu_f=\tfrac13\sum_jF_{f,j},\qquad
\widehat Y_{f,j}=F_{f,j}-\mu_f.
\]

The vertical extension is the current model convention, not an independently calibrated vertical-target law. Preserve it during the port.

### Inverse transform: restore centroid, keystone, barrel

For a trial \((p,A_f)\), restore the **predicted uncentered centroid**:

\[
o=R_\phi^T(Y_{f,j}+\mu_f-O_K).
\]

Invert vertical keystone, then horizontal:

\[
d_y=v_y+q_yo_x,\qquad
h_x=o_x/d_y,\qquad h_y=o_yv_y/(v_xd_y),
\]
\[
d_x=s_y-q_xh_y,\qquad
z_x=h_xs_y/(s_xd_x),\qquad z_y=h_y/d_x.
\]

The target barrel-space point is \(T=O_K+R_\phi z\). Set \(d=T-O_B\) and \(r_d=\|d\|\), solve

\[
r+kr^3=r_d,
\qquad
\widehat Q=O_B+(r/r_d)d.
\]

Handle \(r_d=0\) explicitly and use \(r=r_d\) at \(k=0\). Retain safeguarded Newton with a valid bracket. For \(k\ge0\), use \([0,r_d]\). For \(k<0\), use the inner monotonic branch ending at \(r_t=1/\sqrt{-3k}\), and reject \(r_d\ge2r_t/3\). Reject an invalid trial instead of substituting another cubic root.

Initially retain the reference's maximum 40 radial iterations, normalized root tolerance \(2\times10^{-13}\), positive scale/projective-denominator and radial-derivative thresholds \(10^{-7}\), and barrel-equation residual threshold \(10^{-9}\). These are matching criteria, not new model parameters.

Recompute \(\mu_f\) and its derivatives when accommodation or shared parameters change. Freezing it, using the observed centroid in its place, or reversing the transform order changes the fitted objective.

## 4. Accommodation, offsets and objective

For fixation \(g\), read the prior mean-triangle estimate \(\widehat A_g\) from the frozen input bundle. Every frame in it uses

\[
L_g=\max(0,\widehat A_g-0.25),\qquad
U_g=\min(5,\widehat A_g+0.25),\qquad A_f\in[L_g,U_g].
\]

Desired demand is a reported reference, not a penalty target. The current geometric demand is \(A_{dem,g}=A_{c,0}\cos(\theta_{target,g})\), using the nominal target angle, separately from the refined gaze used by the optics.

Report shared calibration offsets through

\[
b_g=\operatorname{mean}_{f\in g}A_f-A_{dem,g},\qquad
v_f=A_f-A_{dem,g}-b_g,\qquad \operatorname{mean}_{f\in g}v_f=0.
\]

This is an exact decomposition. The optimization has \(N\) effective accommodation variables, equivalently 20 offsets and \(N-20\) centered variations. Do not add 20 redundant fit variables or a demand/offset/temporal penalty.

With six residual components per frame, in P1-corrected pixels,

\[
r_f=\operatorname{vec}\{R(\widehat Q_{f,j}-Q_j)\}_{j=1}^3,
\qquad
w_f=\frac1{60N_{g(f)}},\qquad
E=\sqrt{\sum_fw_f\,r_f^Tr_f}.
\]

This gives equal weight to 20 fixations and three vertices. Preserve the inverse-vertex objective; forward vertex RMS and coordinate-wise RMS have different definitions and are not interchangeable benchmarks.

## 5. Derivatives and native optimization

### Use the implicit barrel derivative

Let \(u=\widehat Q-O_B\), \(r^2=u^Tu\), and

\[
H=r^2I+2uu^T,\qquad J_B=I+kH.
\]

For any fitted scalar \(\xi\),

\[
\frac{\partial\widehat Q}{\partial\xi}
=J_B^{-1}\left[
\frac{\partial T}{\partial\xi}
-r^2u\frac{\partial k}{\partial\xi}
+kH\frac{\partial O_B}{\partial\xi}
\right],
\]
\[
J_B^{-1}=\frac{I-\dfrac{2k}{1+3kr^2}uu^T}{1+kr^2}.
\]

Differentiate the forward centroid and both analytic keystone inverses. Use analytic derivatives or a small forward-mode dual scalar for those expressions, checked against the existing implicit complex-step implementation. Do not differentiate through numerical root iterations.

Compute only the 16 shared derivatives and the accommodation derivative. The current generic evaluator also computes two gaze columns that this experiment never optimizes. Removing that work and the expanded complex-valued temporaries is a concrete performance opportunity, to be measured.

### Eliminate the independent accommodation steps

For \(J_f=\partial r_f/\partial p\in\mathbb R^{6\times16}\) and \(j_f=\partial r_f/\partial A_f\in\mathbb R^6\), accumulate

\[
G=\sum_fw_fJ_f^TJ_f,\quad g=\sum_fw_fJ_f^Tr_f,
\]
\[
B_f=w_fJ_f^Tj_f,\quad C_f=w_fj_f^Tj_f,\quad c_f=w_fj_f^Tr_f.
\]

Match the current active-bound rule: set the accommodation derivative to zero at a bound when the local descent direction points outside the feasible interval. Retain the shared-parameter derivative.

For damping \(\lambda\), define

\[
G_\lambda=G+\lambda\operatorname{diag}(\max(\operatorname{diag}G,10^{-12})),
\qquad D_f=C_f+\lambda\max(C_f,10^{-12}).
\]

Solve the 16×16 Schur system

\[
\left(G_\lambda-\sum_fB_fB_f^T/D_f\right)\delta p
=-g+\sum_f B_fc_f/D_f,
\]
\[
\delta A_f=-(c_f+B_f^T\delta p)/D_f.
\]

Use Eigen's fixed-size LDLT on the host for this small solve, check factorization and linear residuals, and increase damping if necessary. A pivoted QR diagnostic can identify numerical trouble; do not form an explicit matrix inverse. Keep the large reductions and accommodation back-substitution on the GPU. Moving the tiny solve to cuSOLVER is a later benchmark option if host synchronization is material.

Initially copy the existing damping updates, eight damping attempts, projected line search at \(\alpha\in\{1,1/2,1/4,1/8,1/16,1/32\}\), accommodation clipping, orientation bound \([-\pi/2,\pi/2]\), validity rules and stopping criteria. Report objective convergence separately from state-step stability. Change solver policy only after numerical equivalence is established.

## 6. CUDA design and performance sequence

### Portable implementation first

1. Upload fixed observations, fixed gaze, bounds, group labels and weights once. Store coordinates in a structure-of-arrays layout. Keep parameters, accommodation and workspaces allocated across iterations.
2. Start with one CUDA thread per frame for residual-only evaluation, computing its three vertices together so the model centroid is local. Benchmark a warp-per-frame derivative kernel against a thread-per-frame version; 17 sensitivity directions can create substantial register pressure.
3. In the first correct version, materialize the compact real Jacobian and use CUDA/cuBLAS reductions. For this dataset, the 6×16 frame Jacobian is about 65.2 MiB and the per-frame 16-component cross block about 10.9 MiB. This fits the current GPU and is simpler to verify than immediate maximal fusion.
4. Profile, then fuse residual/Jacobian construction with block-level normal-equation reductions where it reduces total time. Avoid contended global atomics for every Hessian entry. Retain a repeatable reduction mode for parity checks.
5. Cache \(G,g,B,C,c\) within a damping attempt sequence. A changed damping value requires new Schur reductions, not new optical derivatives. Evaluate line-search trials with a residual-only kernel.
6. Evaluate the radial loop entirely on the device. Use per-frame/vertex convergence masks instead of a host synchronization after every Newton iteration. Never truncate at a lower fixed iteration count without an error check.
7. Transfer only the small Schur system, its right-hand side and convergence scalars to the host during optimization. Transfer full frame states for selected checkpoints and final export. Time checkpoint serialization separately.

This sequence follows the general priorities of reducing transfers, repeated allocation and unnecessary launches in NVIDIA's [CUDA best-practices guide](https://docs.nvidia.com/cuda/archive/12.9.0/cuda-c-best-practices-guide/index.html). The particular kernel choices above are design hypotheses, not measured speedups.

### Tune P100 and Blackwell independently

Maintain a generic SIMT path for both GPU families. Dispatch using runtime device properties, not a hard-coded GPU name. Tune block sizes, frames per block, derivative lane mapping and reduction tiles separately for P100, data-center Blackwell and RTX Blackwell. NVIDIA documents differing occupancy/shared-memory characteristics even between Blackwell 10.0 and 12.0 in its [Blackwell tuning guide](https://docs.nvidia.com/cuda/archive/12.8.1/blackwell-tuning-guide/index.html).

After profiling, test CUDA Graph capture for stable residual/reduction sequences and tiled asynchronous copies on suitable devices. Keep the existing portable path as the correctness reference. All warp collectives must use correct active masks and explicit synchronization; do not rely on Pascal-era implicit warp scheduling when running on Blackwell.

Start with FP64 for coordinates, radial roots, derivatives, normal equations and the shared solve on every GPU. Remote fitted origins and cancellation make a silent FP32 conversion inappropriate. Different Blackwell products must be measured individually; product generation alone does not establish this solver's FP64 speed. Test mixed precision or tensor-core-backed reductions only as a separate, documented numerical experiment after the FP64 port passes. Preserve FP64 residual verification and accepted error budgets. The 16×16 host solve itself is not a reason to use tensor cores.

Use CUDA events for GPU intervals and a host monotonic clock for end-to-end latency. Check which profiling tools support the actual GPU before use; kernel counters available on Blackwell should not be assumed available on P100. Compiler register/spill reports and CUDA-event timings are useful on both.

### Frozen-model per-frame accommodation

Reuse the same fused inverse-cost kernel. Solve each bounded scalar problem independently using a coarse scan to identify valid candidate minima followed by bracketed refinement, checking endpoints. A single unguarded Newton step is insufficient when sensitivity is weak or the cost has multiple minima. Batch candidates on the GPU; avoid constructing an \(N\times\text{grid}\times\text{derivative}\) temporary tensor. Return accommodation, cost, validity, bound status and ambiguity/sensitivity diagnostics. Compute the same shared-fixation offset decomposition for reporting, without adding a physiological constraint.

## 7. Source layout and data interface

The reusable library is independent of any experiment:

```text
ACC/
  CMakeLists.txt
  include/acc_inverse/       # C/C++ API, model policies and optical stages
  src/                      # CPU/CUDA implementation and shared solver
  apps/acc_inverse_cli.cpp  # Optional HDF5 application
  tests/                    # C++ unit and HDF5 integration tests
  cmake/                    # Package config and build-metadata template
  docs/NativeLibrary.md
  experiments/p4_joint_frame_gaze_accommodation_inverse/
    native_bridge.py        # Experiment's Python client
    native_run.py           # Input conversion, launch and plots
    inputs/
    results/
```

Use a small C ABI with typed contiguous input/output buffers, called by Python's standard `ctypes` initially. C++ owns the GPU workspace and the complete iteration loop. Python loads the existing frozen inputs, checks frame alignment/hashes, and exports native results in the current NPZ/pickle schema. This avoids adding pickle/NPZ readers to the compute library or depending on CuPy in its hot path. Validate scalar types, shapes and buffer lifetimes at the boundary.

Use the current local `inputs/` bundle and preserve its source hashes. Save native benchmarks under a separate results subdirectory; retain the reviewed CuPy fit as the reference until parity and performance checks pass. Keep the current offset tables, triangle overlays and 4×1 time plots.

## 8. Implementation milestones and acceptance criteria

| Milestone | Work | Required evidence |
|---|---|---|
| 1. Lock reference fixtures | Export selected frames, all 20 means, initial/final parameters, bounds and expected residuals | Hashes, frame IDs and exact anchor convention |
| 2. CPU model | Port forward/inverse transforms and implicit derivatives | NumPy parity, branch/roundtrip and derivative checks |
| 3. CUDA model | Add FP64 residual and derivative kernels | CPU/CUDA parity on the full frame set |
| 4. Joint solver | Add active bounds, Schur reduction and Eigen solve | Dense/Schur agreement on a reduced problem; full-data replay |
| 5. Native calibration/inference | Add C ABI, scalar accommodation mode and existing exports | Same objective, bounds, fixed gaze, offset convention and reports |
| 6. Optimize both GPU families | Profile and tune fusion, reductions and launch configuration | Separate P100 and actual-Blackwell benchmark reports |

Initial numerical targets are: full-data replay RMS agreement within \(10^{-8}\) px; maximum per-vertex corrected-coordinate difference below \(10^{-6}\) px on valid fixtures; derivative agreement no worse than the current \(10^{-3}\) relative check, using an absolute check for near-zero derivatives; and reduced-problem Schur/dense step agreement within \(10^{-9}\) relative error away from singular cases. Exercise zero radius, zero barrel coefficient, negative-barrel turning points, projective denominator margins, bound-active frames and nonzero vertical gaze.

For a fresh native fit, use the same initialization, bounds, tolerances and stopping policy. Require a final objective consistent with the CuPy fit, initially within \(10^{-6}\) px, before calling the optimization path equivalent. If numerical conditioning leads to different state vectors or stopping behavior, compare predictions, active sets, gradients and inverse branches; do not accept lower RMS produced by changed weights, omitted frames or a changed model. State uniqueness is a separate question.

Benchmark at least residual evaluation, residual plus derivatives, normal-equation assembly, Schur solve, line search, full calibration, and frozen-model accommodation throughput. Warm up both implementations, repeat fixed-state kernels, and report medians and spread, GPU memory, allocation count, host synchronizations and accepted/rejected steps. Run P100/CuPy and P100/native on the same machine, and Blackwell/CuPy and Blackwell/native on the same Blackwell machine. Cross-GPU timing alone cannot establish a benefit from the C++ port.

The first performance target is at least a 2× reduction in matched full-calibration wall time, with per-kernel results explaining the gain. This is an engineering target, not a forecast. Tune the measured bottleneck while preserving the accuracy tests; retain a portable FP64 path if an architecture-specific variant fails the numerical budget.

Implementation order: **exact CPU model → FP64 CUDA evaluator/derivatives → native Schur solver → full-data parity → measured P100 and Blackwell optimization**.
