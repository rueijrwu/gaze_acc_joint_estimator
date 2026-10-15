# Root implementation audit

The reusable library now lives in the repository-root CMake project. Public headers are in `include/acc_inverse/`, library code in `src/`, the HDF5 application in `apps/`, tests in `tests/`, and configuration templates in `cmake/`. The move preserved C++ and CUDA file contents; only build paths, client discovery and documentation were updated. Historical benchmark/checkpoint records retain their original paths as provenance. [Relocation manifest and verification](../experiments/p4_joint_frame_gaze_accommodation_inverse/results/native/reorganization/) record the new layout separately.

The fresh root-source build passes all three CTests, installed external C++ usage and a core-only build with HDF5 disabled. The relocated executable reproduces 159 accepted iterations and RMS `0.6848295432534102` px. Its fixed gaze, mean-triangle anchors and accommodation bounds match the frozen data exactly. The [relocation summary](../experiments/p4_joint_frame_gaze_accommodation_inverse/results/native/reorganization/summary.json) contains current build paths and hashes.

The native library is ready for a verified pause. The root agent planned and reviewed the implementation, optical equations, solver controls, public interfaces, tests and measured results. Mechanical verification ran outside the sandbox; final builds used 16 parallel jobs.

## Model and solver review

- `optical_stages.hpp` separates barrel, two-axis keystone and centering. Both forward orders are compiled; each inverse reverses its selected order and restores the model centroid. Centroid derivatives remain part of the residual derivative.
- `model.hpp` supplies the shared parameter law and compiled policy. Barrel derivatives use implicit differentiation on the safeguarded inner radial branch. The root reviewed the origin terms and the derivative composition for both orders.
- The solver consumes residual and normal blocks independently of optical composition. Shared parameters and frame accommodation can be frozen separately. Frozen values remain exact, including orientation outside its optimization interval; diagnostic Jacobians remain unmasked.
- cuBLAS constructs the normal and Schur systems. cuSOLVER is the preferred shared solve, with checked Eigen fallback. Diagonal equilibration changes the coordinates of the linear solve without changing the objective or adding a penalty.
- The default scientific fit retains fixed refined gaze, fixed reference, saved P1 correction, equal fixation weights and mean-triangle accommodation anchors ±0.25 D. Demand and fixation offsets have no penalty. The 20 offsets are a reporting decomposition of the frame states.

## API and ownership review

The versioned C descriptors preserve the original entry points and defaults. The C++ wrapper owns an opaque workspace through move-only RAII. Errors retain typed status, detail and original backend code. Optional logging contains callback exceptions and performs no formatting in the default solver path.

The caller serializes operations on one workspace and retains asynchronous device buffers until completion. The Python/CuPy binding enforces this lifetime convention. Independent workspaces can be used on separate host threads. Core bounds and group labels are generic; the HDF5 experiment runner separately enforces its 20-fixation protocol. The library builds without HDF5 and exports the installed `acc::inverse` target.

## Findings corrected during review

- Kept the refined configuration reference authoritative; acquisition reference metadata is an earlier triangle.
- Corrected near-endpoint scalar refinement, candidate counting, ambiguity deduplication and bound classification.
- Recomputed final gradients at the returned state; rejected zero-step iterations no longer imply frame-state stability.
- Enabled standalone inference and continuation from native `/fit` results, preserving transformation order in HDF5.
- Kept frozen orientation exact and public diagnostic derivatives independent of fitting masks.
- Validated device ordinals before CUDA selection and avoided CUDA cleanup on an unselected partial workspace. Captured runtime errors retain their original diagnostic while clearing stale CUDA error state.
- Distinguished blocking CUDA API durations from independent host overhead in the profiling conclusions.

## Verification evidence

| Check | Result |
|---|---|
| Native CPU, CUDA and HDF5 CTests | 3/3 passed |
| CUDA compute-sanitizer memcheck | 0 errors |
| Both model orders, all 17 finite-difference columns | Passed |
| Dense/Schur, active bounds, masks, staged fits, errors, logging, RAII | Passed |
| CPU/GPU system replay of 88,978 frames and CuPy stream integration | Passed |
| Installed external C++ package consumer | Passed |
| Core build with HDF5 discovery disabled | Passed |
| Blackwell sm_100, sm_103, sm_120, sm_121 compilation | Passed; execution untested |

The final default fit retains 159 accepted iterations and inverse RMS approximately **0.684829543253410 px**, agreeing with the frozen CuPy result to approximately **1.5e-13 px**. Three native fresh fits have median solve time approximately **1.12 s**. The approximately 61× estimated math speedup uses the separately recorded CuPy baseline after measured I/O/host-copy costs are subtracted; the report explains the differing checkpoint cadences.

Objective convergence does not establish unique or physiologically stable frame accommodation. The accepted fit still reports frame-state step stability as false. Alternative transformation order is validated with independent synthetic composition and derivatives; the preserved real-data benchmark uses barrel then keystone.

[Final test logs](../experiments/p4_joint_frame_gaze_accommodation_inverse/results/native/validation_native/) · [Final build verification](../experiments/p4_joint_frame_gaze_accommodation_inverse/results/native/final_build_verification.json) · [System validation](../experiments/p4_joint_frame_gaze_accommodation_inverse/results/native/validation.json) · [CUDA profile](../experiments/p4_joint_frame_gaze_accommodation_inverse/results/native/profiling/p100-profile-notes.md) · [Results and parameter tables](../experiments/p4_joint_frame_gaze_accommodation_inverse/NATIVE_RESULTS.md)

The safe-checkpoint manifest records final source and binary hashes. Blackwell runtime benchmarking remains a future hardware task. No further implementation or experiments are scheduled before user resumption.
