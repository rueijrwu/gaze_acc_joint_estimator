#include "acc_inverse/error.hpp"
#include "acc_inverse/inference.hpp"
#include "internal.hpp"
#include <algorithm>
#include <chrono>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <math_constants.h>
#include <sstream>
#include <stdexcept>
#include <utility>
namespace acc {
namespace {
void check(cudaError_t code, const char *operation) {
    if (code != cudaSuccess) {
        const char *description = cudaGetErrorString(code);
        (void)cudaGetLastError();
        throw CudaError(int32_t(code), std::string(operation) + ": " + description);
    }
}
void check_blas(cublasStatus_t code, const char *operation) {
    if (code != CUBLAS_STATUS_SUCCESS)
        throw NativeError(ACC_ERROR_CUBLAS, int32_t(code),
                          std::string(operation) + ": cuBLAS status " + std::to_string(int(code)));
}
#define CU(call) check((call), #call)
#define BLAS(call) check_blas((call), #call)
void check_solver(cusolverStatus_t code, const char *operation) {
    if (code != CUSOLVER_STATUS_SUCCESS)
        throw NativeError(ACC_ERROR_CUSOLVER, int32_t(code),
                          std::string(operation) + ": cuSOLVER status " +
                              std::to_string(int(code)));
}
#define SOLVER(call) check_solver((call), #call)
struct DeviceView {
    int64_t n;
    const double *y, *gaze, *low, *high, *weights;
    double *r, *corrected, *predicted, *J, *JA, *B, *C, *c;
    unsigned char *valid, *active;
    int *invalid;
};
__device__ void load_frame(const DeviceView &v, int64_t f, double *y) {
    for (int c = 0; c < 6; ++c)
        y[c] = v.y[c * v.n + f];
}
__device__ void save_frame(const DeviceView &v, int64_t f, const Frame &out) {
    double sw = ::sqrt(v.weights[f]);
    for (int c = 0; c < 6; ++c) {
        v.r[c * v.n + f] = out.residual[c] * sw;
        v.corrected[c * v.n + f] = out.corrected[c];
        v.predicted[c * v.n + f] = out.predicted[c];
    }
    v.valid[f] = out.valid ? 1 : 0;
    if (!out.valid)
        atomicExch(v.invalid, 1);
}
template <TransformOrder Order>
__global__ void residual_kernel(Config cfg, DeviceView v, const double *p, const double *A) {
    int64_t f = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (f >= v.n)
        return;
    double y[6];
    load_frame(v, f, y);
    auto out = ModelPolicy<Order>::evaluate(cfg, p, A[f], v.gaze[f], v.gaze[v.n + f], y);
    save_frame(v, f, out);
}
/* One warp owns a frame. Lanes 0..NP differentiate p and A respectively.
 * Only lane0 solves radial roots; primal inverse-barrel state is broadcast
 * using explicit full-warp shuffles on Pascal and independent-thread GPUs. */
template <TransformOrder Order>
__global__ void derivatives_warp(Config cfg, DeviceView v, const double *p, const double *A,
                                 uint32_t parameter_mask, bool optimize_A) {
    int lane = threadIdx.x & 31;
    int64_t f = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
    if (f >= v.n)
        return;
    double y[6];
    load_frame(v, f, y);
    Frame base;
    if (lane == 0) {
        base = ModelPolicy<Order>::evaluate(cfg, p, A[f], v.gaze[f], v.gaze[v.n + f], y);
        save_frame(v, f, base);
    }
    int valid = __shfl_sync(0xffffffff, lane == 0 ? int(base.valid) : 0, 0);
    base.valid = valid;
    base.k = __shfl_sync(0xffffffff, lane == 0 ? base.k : 0., 0);
    for (int j = 0; j < 3; ++j)
        base.r2[j] = __shfl_sync(0xffffffff, lane == 0 ? base.r2[j] : 0., 0);
    for (int c = 0; c < 6; ++c) {
        base.u[c] = __shfl_sync(0xffffffff, lane == 0 ? base.u[c] : 0., 0);
        base.residual[c] = __shfl_sync(0xffffffff, lane == 0 ? base.residual[c] : 0., 0);
    }
    double dr[6] = {0., 0., 0., 0., 0., 0.};
    if (lane < ND && valid && (lane == NP ? optimize_A : ((parameter_mask >> lane) & 1)))
        ModelPolicy<Order>::derivative(cfg, p, A[f], v.gaze[f], v.gaze[v.n + f], y, base, lane, dr);
    double ja[6], rawc = 0., norm = 0.;
    for (int c = 0; c < 6; ++c) {
        ja[c] = __shfl_sync(0xffffffff, dr[c], NP);
        rawc += ja[c] * base.residual[c];
        norm += ja[c] * ja[c];
    }
    double wt = v.weights[f];
    rawc *= wt;
    bool active = !optimize_A || (A[f] <= v.low[f] + 1e-9 && rawc > 0.) ||
                  (A[f] >= v.high[f] - 1e-9 && rawc < 0.);
    if (lane < NP) {
        double b = 0., sw = ::sqrt(wt);
        for (int c = 0; c < 6; ++c) {
            v.J[(c * v.n + f) * NP + lane] = dr[c] * sw;
            b += dr[c] * ja[c];
        }
        v.B[f * NP + lane] = active ? 0. : wt * b;
    }
    if (lane == NP) {
        for (int c = 0; c < 6; ++c)
            v.JA[c * v.n + f] = dr[c];
        v.C[f] = active ? 0. : wt * norm;
        v.c[f] = active ? 0. : rawc;
        v.active[f] = active ? 1 : 0;
    }
}
template <TransformOrder Order>
__global__ void derivatives_thread(Config cfg, DeviceView v, const double *p, const double *A,
                                   uint32_t parameter_mask, bool optimize_A) {
    int64_t f = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (f >= v.n)
        return;
    double y[6];
    load_frame(v, f, y);
    auto base = ModelPolicy<Order>::evaluate(cfg, p, A[f], v.gaze[f], v.gaze[v.n + f], y);
    save_frame(v, f, base);
    double ja[6] = {0., 0., 0., 0., 0., 0.};
    if (base.valid && optimize_A)
        ModelPolicy<Order>::derivative(cfg, p, A[f], v.gaze[f], v.gaze[v.n + f], y, base, NP, ja);
    double rawc = 0., norm = 0., wt = v.weights[f], sw = ::sqrt(wt);
    for (int c = 0; c < 6; ++c) {
        rawc += ja[c] * base.residual[c];
        norm += ja[c] * ja[c];
        v.JA[c * v.n + f] = ja[c];
    }
    rawc *= wt;
    bool active = !optimize_A || (A[f] <= v.low[f] + 1e-9 && rawc > 0.) ||
                  (A[f] >= v.high[f] - 1e-9 && rawc < 0.);
    v.C[f] = active ? 0. : wt * norm;
    v.c[f] = active ? 0. : rawc;
    v.active[f] = active ? 1 : 0;
    for (int d = 0; d < NP; ++d) {
        double dr[6] = {0., 0., 0., 0., 0., 0.};
        if (base.valid && ((parameter_mask >> d) & 1))
            ModelPolicy<Order>::derivative(cfg, p, A[f], v.gaze[f], v.gaze[v.n + f], y, base, d,
                                           dr);
        double b = 0.;
        for (int c = 0; c < 6; ++c) {
            v.J[(c * v.n + f) * NP + d] = sw * dr[c];
            b += dr[c] * ja[c];
        }
        v.B[f * NP + d] = active ? 0. : wt * b;
    }
}
__global__ void max_reduction(const double *data, int64_t n, double *result) {
    __shared__ double values[256];
    double local = 0.;
    for (int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; index < n;
         index += int64_t(gridDim.x) * blockDim.x)
        local = ::fmax(local, data[index]);
    values[threadIdx.x] = local;
    __syncthreads();
    for (int step = blockDim.x / 2; step > 0; step >>= 1) {
        if (threadIdx.x < step)
            values[threadIdx.x] = ::fmax(values[threadIdx.x], values[threadIdx.x + step]);
        __syncthreads();
    }
    if (threadIdx.x == 0)
        result[blockIdx.x] = values[0];
}
__global__ void gradient_values(const double *C, const double *c, int64_t n, double *work) {
    int64_t f = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (f < n)
        work[f] = ::fabs(c[f]) / ::sqrt(::fmax(C[f], 1e-20));
}
__global__ void scale_schur(const double *B, const double *C, const double *c, int64_t n,
                            double damping, double *scaledB, double *scaledc) {
    int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= n * NP)
        return;
    int64_t f = index / NP;
    double D = C[f] + damping * ::fmax(C[f], 1e-12), scale = 1. / ::sqrt(D);
    scaledB[index] = B[index] * scale;
    if (index % NP == 0)
        scaledc[f] = c[f] * scale;
}
__global__ void damp_global(const double *G, const double *g, double *S, double *rhs,
                            double damping) {
    int i = threadIdx.x;
    if (i < NP * NP) {
        S[i] = G[i];
        if (i % ND == 0)
            S[i] += damping * ::fmax(G[i], 1e-12);
    }
    if (i < NP)
        rhs[i] = -g[i];
}
__global__ void backsub_kernel(const double *B, const double *C, const double *c, const double *dp,
                               int64_t n, double damping, double *dA) {
    int64_t f = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (f >= n)
        return;
    double dot = 0.;
    for (int d = 0; d < NP; ++d)
        dot += B[f * NP + d] * dp[d];
    double D = C[f] + damping * ::fmax(C[f], 1e-12);
    dA[f] = -(c[f] + dot) / D;
}
__global__ void trial_kernel(const double *A, const double *dA, const double *lo, const double *hi,
                             int64_t n, double alpha, double *trialA, double *work,
                             bool optimize_A) {
    int64_t f = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (f >= n)
        return;
    double value = optimize_A ? ::fmax(lo[f], ::fmin(hi[f], A[f] + alpha * dA[f])) : A[f];
    trialA[f] = value;
    work[f] = ::fabs(value - A[f]);
}
template <TransformOrder Order>
__global__ void infer_kernel(Config cfg, DeviceView v, const double *p, AccInferOptions options,
                             double *A, double *cost, unsigned char *valid, int *bound,
                             double *sensitivity, double *gap, int *count) {
    int64_t f = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (f >= v.n)
        return;
    double y[6];
    load_frame(v, f, y);
    auto result =
        infer_frame<Order>(cfg, p, v.gaze[f], v.gaze[v.n + f], y, v.low[f], v.high[f], options);
    A[f] = result.A;
    cost[f] = result.cost;
    valid[f] = result.valid;
    bound[f] = result.bound;
    sensitivity[f] = result.sensitivity;
    gap[f] = result.gap;
    count[f] = result.count;
}
__global__ void pack_evaluation(DeviceView v, bool derivatives, double *residual, double *corrected,
                                double *predicted, double *J, unsigned char *valid) {
    int64_t f = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (f >= v.n)
        return;
    double sw = ::sqrt(v.weights[f]);
    for (int c = 0; c < 6; ++c) {
        int64_t source = c * v.n + f, target = f * 6 + c;
        if (residual)
            residual[target] = v.r[source] / sw;
        if (corrected)
            corrected[target] = v.corrected[source];
        if (predicted)
            predicted[target] = v.predicted[source];
        if (derivatives && J) {
            for (int d = 0; d < NP; ++d)
                J[target * ND + d] = v.valid[f] ? v.J[source * NP + d] / sw : CUDART_NAN;
            J[target * ND + NP] = v.valid[f] ? v.JA[source] : CUDART_NAN;
        }
    }
    if (valid)
        valid[f] = v.valid[f];
}
__global__ void equilibrate_system(const double *S, const double *rhs, double *factor,
                                   double *solution, double *scale) {
    int index = threadIdx.x;
    if (index < NP)
        scale[index] = 1. / ::sqrt(S[index * ND]);
    __syncthreads();
    if (index < NP * NP) {
        int row = index % NP, col = index / NP;
        factor[index] = .5 * (S[index] + S[row * NP + col]) * scale[row] * scale[col];
    }
    if (index < NP)
        solution[index] = rhs[index] * scale[index];
}
__global__ void unscale_solution(double *solution, const double *scale) {
    int index = threadIdx.x;
    if (index < NP)
        solution[index] *= scale[index];
}
__global__ void finish_loss(const int *invalid, double *loss) {
    if (*invalid)
        *loss = CUDART_INF;
}
} // namespace
struct Gpu::Impl {
    Workspace &w;
    int device;
    size_t bytes = 0;
    cusolverDnHandle_t solver = nullptr;
    int solver_mode = 0;
    uint64_t cusolver_calls = 0, eigen_calls = 0;
    double *factor = nullptr, *solution = nullptr, *scale = nullptr, *solver_work = nullptr;
    int *solver_info = nullptr;
    int solver_workspace_size = 0;
    cublasHandle_t blas = nullptr;
    cudaEvent_t start = nullptr, stop = nullptr;
    DeviceView v{};
    double *p = nullptr, *trialp = nullptr, *A = nullptr, *trialA = nullptr, *G = nullptr,
           *g = nullptr, *S = nullptr, *rhs = nullptr, *dp = nullptr;
    double *scaledB = nullptr, *scaledc = nullptr, *dA = nullptr, *work = nullptr,
           *partial = nullptr, *maximum = nullptr, *loss = nullptr;
    int *bound = nullptr, *count = nullptr;
    double *cost = nullptr, *sensitivity = nullptr, *gap = nullptr;
    std::vector<void *> allocations;
    int threads = 128, mode = 0;
    bool device_selected = false;
    Impl(Workspace &workspace, int id) : w(workspace), device(id) {
        allocations.reserve(64);
    }
    ~Impl() {
        if (!device_selected)
            return;
        cudaSetDevice(device);
        cudaDeviceSynchronize();
        for (void *p : allocations)
            cudaFree(p);
        if (blas)
            cublasDestroy(blas);
        if (solver)
            cusolverDnDestroy(solver);
        if (start)
            cudaEventDestroy(start);
        if (stop)
            cudaEventDestroy(stop);
    }
    template <class T> void allocate(T *&pointer, size_t count) {
        CU(cudaMalloc(reinterpret_cast<void **>(&pointer), count * sizeof(T)));
        allocations.push_back(pointer);
        bytes += count * sizeof(T);
        ++w.profile.allocation_count;
    }
    template <class T> void copy(T *destination, const T *source, size_t count) {
        CU(cudaMemcpy(destination, source, count * sizeof(T), cudaMemcpyHostToDevice));
    }
    template <TransformOrder Order>
    void launch_order(bool derivatives, const double *parameters, const double *state,
                      cudaStream_t stream, uint32_t mask, bool optimize_A) {
        if (derivatives) {
            if (mode == 0)
                derivatives_warp<Order>
                    <<<unsigned((w.n * 32 + threads - 1) / threads), threads, 0, stream>>>(
                        w.config, v, parameters, state, mask, optimize_A);
            else
                derivatives_thread<Order>
                    <<<unsigned((w.n + threads - 1) / threads), threads, 0, stream>>>(
                        w.config, v, parameters, state, mask, optimize_A);
        } else
            residual_kernel<Order><<<unsigned((w.n + threads - 1) / threads), threads, 0, stream>>>(
                w.config, v, parameters, state);
    }
    void launch(bool derivatives, const double *parameters, const double *state,
                cudaStream_t stream = 0, const SolveControl *control = nullptr) {
        CU(cudaMemsetAsync(v.invalid, 0, sizeof(int), stream));
        uint32_t mask = control ? control->shared_parameter_mask : ((uint32_t(1) << NP) - 1);
        bool optimize_A = !control || control->optimize_accommodation;
        if (w.order == TransformOrder::KeystoneThenBarrel)
            launch_order<TransformOrder::KeystoneThenBarrel>(derivatives, parameters, state, stream,
                                                             mask, optimize_A);
        else
            launch_order<TransformOrder::BarrelThenKeystone>(derivatives, parameters, state, stream,
                                                             mask, optimize_A);
        CU(cudaGetLastError());
    }
    void begin() {
        CU(cudaEventRecord(start));
    }
    double end() {
        CU(cudaEventRecord(stop));
        CU(cudaEventSynchronize(stop));
        float ms = 0;
        CU(cudaEventElapsedTime(&ms, start, stop));
        return ms;
    }
    double max_work() {
        int blocks = int(std::min<int64_t>(1024, (w.n + 255) / 256));
        max_reduction<<<blocks, 256>>>(work, w.n, partial);
        max_reduction<<<1, 256>>>(partial, blocks, maximum);
        CU(cudaGetLastError());
        double value;
        CU(cudaMemcpy(&value, maximum, sizeof(double), cudaMemcpyDeviceToHost));
        return value;
    }
    void select() {
        CU(cudaSetDevice(device));
        device_selected = true;
    }
};
Gpu::Gpu(Workspace &w, int device) : impl(new Impl(w, device)) {
    auto &a = *impl;
    int count = 0;
    CU(cudaGetDeviceCount(&count));
    if (device < 0 || device >= count)
        throw std::invalid_argument("CUDA device index out of range");
    a.select();
    CU(cudaFree(nullptr));
    BLAS(cublasCreate(&a.blas));
    BLAS(cublasSetMathMode(a.blas, CUBLAS_DEFAULT_MATH));
    SOLVER(cusolverDnCreate(&a.solver));
    CU(cudaEventCreate(&a.start));
    CU(cudaEventCreate(&a.stop));
    int64_t n = w.n;
    a.v.n = n;
    cudaDeviceProp properties{};
    CU(cudaGetDeviceProperties(&properties, device));
    if (properties.major == 6 && properties.minor == 0) {
        a.threads = 256;
        a.mode = 1;
        w.residual_threads = 256;
        w.derivative_mode = 1;
    }
    double *y = nullptr, *gaze = nullptr, *low = nullptr, *high = nullptr, *weights = nullptr;
    a.allocate(y, n * 6);
    a.allocate(gaze, n * 2);
    a.allocate(low, n);
    a.allocate(high, n);
    a.allocate(weights, n);
    a.v.y = y;
    a.v.gaze = gaze;
    a.v.low = low;
    a.v.high = high;
    a.v.weights = weights;
    std::vector<double> soa(n * 6);
    for (int c = 0; c < 6; ++c)
        for (int64_t f = 0; f < n; ++f)
            soa[c * n + f] = w.y[6 * f + c];
    a.copy(y, soa.data(), n * 6);
    soa.resize(n * 2);
    for (int c = 0; c < 2; ++c)
        for (int64_t f = 0; f < n; ++f)
            soa[c * n + f] = w.gaze[2 * f + c];
    a.copy(gaze, soa.data(), n * 2);
    a.copy(low, w.low.data(), n);
    a.copy(high, w.high.data(), n);
    a.copy(weights, w.weight.data(), n);
    a.allocate(a.p, NP);
    a.allocate(a.trialp, NP);
    a.allocate(a.A, n);
    a.allocate(a.trialA, n);
    a.allocate(a.v.r, n * 6);
    a.allocate(a.v.corrected, n * 6);
    a.allocate(a.v.predicted, n * 6);
    a.allocate(a.v.J, n * 6 * NP);
    a.allocate(a.v.JA, n * 6);
    a.allocate(a.v.B, n * NP);
    a.allocate(a.v.C, n);
    a.allocate(a.v.c, n);
    a.allocate(a.v.valid, n);
    a.allocate(a.v.active, n);
    a.allocate(a.v.invalid, 1);
    a.allocate(a.G, NP * NP);
    a.allocate(a.g, NP);
    a.allocate(a.S, NP * NP);
    a.allocate(a.rhs, NP);
    a.allocate(a.dp, NP);
    a.allocate(a.scaledB, n * NP);
    a.allocate(a.scaledc, n);
    a.allocate(a.dA, n);
    a.allocate(a.work, n);
    a.allocate(a.partial, 1024);
    a.allocate(a.maximum, 1);
    a.allocate(a.loss, 1);
    a.allocate(a.bound, n);
    a.allocate(a.count, n);
    a.allocate(a.cost, n);
    a.allocate(a.sensitivity, n);
    a.allocate(a.gap, n);
    a.allocate(a.factor, NP * NP);
    a.allocate(a.solution, NP);
    a.allocate(a.scale, NP);
    a.allocate(a.solver_info, 1);
    SOLVER(cusolverDnDpotrf_bufferSize(a.solver, CUBLAS_FILL_MODE_LOWER, NP, a.factor, NP,
                                       &a.solver_workspace_size));
    a.allocate(a.solver_work, a.solver_workspace_size);
}
Gpu::~Gpu() = default;
std::string Gpu::metadata() const {
    auto &a = *impl;
    cudaDeviceProp prop{};
    CU(cudaGetDeviceProperties(&prop, a.device));
    int driver = 0, runtime = 0;
    CU(cudaDriverGetVersion(&driver));
    CU(cudaRuntimeGetVersion(&runtime));
    std::ostringstream out;
    out << "\"device\":\"" << prop.name << "\",\"device_index\":" << a.device
        << ",\"compute_capability\":\"" << prop.major << '.' << prop.minor
        << "\",\"cuda_driver\":" << driver << ",\"cuda_runtime\":" << runtime
        << ",\"cuda_compiler\":\"" << __CUDACC_VER_MAJOR__ << '.' << __CUDACC_VER_MINOR__ << '.'
        << __CUDACC_VER_BUILD__ << "\",\"allocated_device_bytes\":" << a.bytes
        << ",\"shared_solver\":\"" << (a.solver_mode == 1 ? "Eigen LDLT" : "cuSOLVER POTRF/POTRS")
        << "\",\"solver_mode\":" << a.solver_mode << ",\"cusolver_calls\":" << a.cusolver_calls
        << ",\"eigen_calls\":" << a.eigen_calls;
    return out.str();
}
void Gpu::set_launch(int threads, int mode) {
    impl->threads = threads;
    impl->mode = mode;
}
void Gpu::upload(const double *p, const double *A) {
    auto &a = *impl;
    a.select();
    a.copy(a.p, p, NP);
    a.copy(a.A, A, a.w.n);
}
double Gpu::evaluate(bool derivatives, bool trial, const SolveControl *control) {
    auto &a = *impl;
    a.select();
    a.begin();
    a.launch(derivatives, trial ? a.trialp : a.p, trial ? a.trialA : a.A, 0, control);
    CU(cudaEventRecord(a.stop));
    BLAS(cublasSetPointerMode(a.blas, CUBLAS_POINTER_MODE_DEVICE));
    BLAS(cublasDdot(a.blas, int(a.w.n * 6), a.v.r, 1, a.v.r, 1, a.loss));
    finish_loss<<<1, 1>>>(a.v.invalid, a.loss);
    CU(cudaGetLastError());
    BLAS(cublasSetPointerMode(a.blas, CUBLAS_POINTER_MODE_HOST));
    double result = 0.;
    CU(cudaMemcpy(&result, a.loss, sizeof(double), cudaMemcpyDeviceToHost));
    float ms = 0.;
    CU(cudaEventElapsedTime(&ms, a.start, a.stop));
    if (derivatives) {
        a.w.profile.derivative_kernel_ms += ms;
        ++a.w.profile.derivative_calls;
    } else {
        a.w.profile.residual_kernel_ms += ms;
    }
    ++a.w.profile.evaluation_calls;
    return result;
}
void Gpu::download_evaluation(bool derivatives, double *residual, double *corrected,
                              double *predicted, double *J, uint8_t *valid) {
    auto &a = *impl;
    a.select();
    int64_t n = a.w.n;
    auto transpose = [&](const double *source, double *output, bool unweight) {
        if (!output)
            return;
        std::vector<double> buffer(n * 6);
        CU(cudaMemcpy(buffer.data(), source, buffer.size() * sizeof(double),
                      cudaMemcpyDeviceToHost));
        for (int64_t f = 0; f < n; ++f)
            for (int c = 0; c < 6; ++c)
                output[6 * f + c] = buffer[c * n + f] / (unweight ? std::sqrt(a.w.weight[f]) : 1.);
    };
    transpose(a.v.r, residual, true);
    transpose(a.v.corrected, corrected, false);
    transpose(a.v.predicted, predicted, false);
    std::vector<uint8_t> validity(n);
    CU(cudaMemcpy(validity.data(), a.v.valid, n * sizeof(uint8_t), cudaMemcpyDeviceToHost));
    if (valid)
        std::copy(validity.begin(), validity.end(), valid);
    if (derivatives && J) {
        std::vector<double> shared(n * 6 * NP), state(n * 6);
        CU(cudaMemcpy(shared.data(), a.v.J, shared.size() * sizeof(double),
                      cudaMemcpyDeviceToHost));
        CU(cudaMemcpy(state.data(), a.v.JA, state.size() * sizeof(double), cudaMemcpyDeviceToHost));
        for (int64_t f = 0; f < n; ++f)
            for (int c = 0; c < 6; ++c) {
                for (int d = 0; d < NP; ++d)
                    J[(f * 6 + c) * ND + d] =
                        validity[f] ? shared[(c * n + f) * NP + d] / std::sqrt(a.w.weight[f])
                                    : std::numeric_limits<double>::quiet_NaN();
                J[(f * 6 + c) * ND + NP] =
                    validity[f] ? state[c * n + f] : std::numeric_limits<double>::quiet_NaN();
            }
    }
}
Normal Gpu::normals(bool download_blocks, const SolveControl *control) {
    auto &a = *impl;
    Normal out;
    out.loss = evaluate(true, false, control);
    if (!std::isfinite(out.loss))
        return out;
    a.begin();
    double one = 1., zero = 0.;
    int M = int(a.w.n * 6);
    BLAS(cublasDgemm(a.blas, CUBLAS_OP_N, CUBLAS_OP_T, NP, NP, M, &one, a.v.J, NP, a.v.J, NP, &zero,
                     a.G, NP));
    BLAS(cublasDgemv(a.blas, CUBLAS_OP_N, NP, M, &one, a.v.J, NP, a.v.r, 1, &zero, a.g, 1));
    gradient_values<<<unsigned((a.w.n + 255) / 256), 256>>>(a.v.C, a.v.c, a.w.n, a.work);
    CU(cudaGetLastError());
    a.w.profile.normal_assembly_ms += a.end();
    out.sg = a.max_work();
    CU(cudaMemcpy(out.G.data(), a.G, NP * NP * sizeof(double), cudaMemcpyDeviceToHost));
    CU(cudaMemcpy(out.g.data(), a.g, NP * sizeof(double), cudaMemcpyDeviceToHost));
    for (int d = 0; d < NP; ++d)
        out.pg = std::max(out.pg, std::abs(out.g(d)) / std::sqrt(std::max(out.G(d, d), 1e-20)));
    if (download_blocks) {
        out.B.resize(a.w.n * NP);
        out.C.resize(a.w.n);
        out.c.resize(a.w.n);
        out.active.resize(a.w.n);
        download_normals(out.B.data(), out.C.data(), out.c.data(), out.active.data());
    }
    return out;
}
void Gpu::download_normals(double *B, double *C, double *c, uint8_t *active) {
    auto &a = *impl;
    a.select();
    int64_t n = a.w.n;
    if (B)
        CU(cudaMemcpy(B, a.v.B, n * NP * sizeof(double), cudaMemcpyDeviceToHost));
    if (C)
        CU(cudaMemcpy(C, a.v.C, n * sizeof(double), cudaMemcpyDeviceToHost));
    if (c)
        CU(cudaMemcpy(c, a.v.c, n * sizeof(double), cudaMemcpyDeviceToHost));
    if (active)
        CU(cudaMemcpy(active, a.v.active, n * sizeof(uint8_t), cudaMemcpyDeviceToHost));
}
void Gpu::schur(double damping, Matrix &S, Vector &rhs) {
    auto &a = *impl;
    a.begin();
    scale_schur<<<unsigned((a.w.n * NP + 255) / 256), 256>>>(a.v.B, a.v.C, a.v.c, a.w.n, damping,
                                                             a.scaledB, a.scaledc);
    damp_global<<<1, 256>>>(a.G, a.g, a.S, a.rhs, damping);
    CU(cudaGetLastError());
    double minus = -1., plus = 1.;
    BLAS(cublasDgemm(a.blas, CUBLAS_OP_N, CUBLAS_OP_T, NP, NP, int(a.w.n), &minus, a.scaledB, NP,
                     a.scaledB, NP, &plus, a.S, NP));
    BLAS(cublasDgemv(a.blas, CUBLAS_OP_N, NP, int(a.w.n), &plus, a.scaledB, NP, a.scaledc, 1, &plus,
                     a.rhs, 1));
    a.w.profile.schur_assembly_ms += a.end();
    CU(cudaMemcpy(S.data(), a.S, NP * NP * sizeof(double), cudaMemcpyDeviceToHost));
    CU(cudaMemcpy(rhs.data(), a.rhs, NP * sizeof(double), cudaMemcpyDeviceToHost));
}
void Gpu::backsubstitute(double damping, const Vector &dp) {
    auto &a = *impl;
    a.copy(a.dp, dp.data(), NP);
    a.begin();
    backsub_kernel<<<unsigned((a.w.n + 255) / 256), 256>>>(a.v.B, a.v.C, a.v.c, a.dp, a.w.n,
                                                           damping, a.dA);
    CU(cudaGetLastError());
    a.w.profile.backsubstitution_ms += a.end();
}
double Gpu::trial(const double *parameters, double alpha, double &maximum_step, double current_loss,
                  bool optimize_A) {
    auto &a = *impl;
    a.copy(a.trialp, parameters, NP);
    trial_kernel<<<unsigned((a.w.n + 255) / 256), 256>>>(a.A, a.dA, a.v.low, a.v.high, a.w.n, alpha,
                                                         a.trialA, a.work, optimize_A);
    CU(cudaGetLastError());
    double objective = evaluate(false, true);
    maximum_step = objective < current_loss ? a.max_work() : 0.;
    return objective;
}
void Gpu::accept() {
    std::swap(impl->p, impl->trialp);
    std::swap(impl->A, impl->trialA);
}
void Gpu::download_step(double *output) {
    auto &a = *impl;
    a.select();
    CU(cudaMemcpy(output, a.dA, a.w.n * sizeof(double), cudaMemcpyDeviceToHost));
}
void Gpu::set_solver_mode(int mode) {
    impl->solver_mode = mode;
}
bool Gpu::linear_solve(const Matrix &S, const Vector &rhs, Vector &dp, double &relative) {
    auto &a = *impl;
    a.select();
    bool valid = false;
    if (a.solver_mode != 1) {
        ++a.cusolver_calls;
        equilibrate_system<<<1, 256>>>(a.S, a.rhs, a.factor, a.solution, a.scale);
        CU(cudaGetLastError());
        SOLVER(cusolverDnDpotrf(a.solver, CUBLAS_FILL_MODE_LOWER, NP, a.factor, NP, a.solver_work,
                                a.solver_workspace_size, a.solver_info));
        int info = 0;
        CU(cudaMemcpy(&info, a.solver_info, sizeof(int), cudaMemcpyDeviceToHost));
        if (info == 0) {
            SOLVER(cusolverDnDpotrs(a.solver, CUBLAS_FILL_MODE_LOWER, NP, 1, a.factor, NP,
                                    a.solution, NP, a.solver_info));
            CU(cudaMemcpy(&info, a.solver_info, sizeof(int), cudaMemcpyDeviceToHost));
            if (info == 0) {
                unscale_solution<<<1, 32>>>(a.solution, a.scale);
                CU(cudaGetLastError());
                CU(cudaMemcpy(dp.data(), a.solution, NP * sizeof(double), cudaMemcpyDeviceToHost));
                double error = (S * dp - rhs).norm();
                relative = error / std::max(rhs.norm(), 1e-300);
                double backward = error / std::max(S.norm() * dp.norm() + rhs.norm(), 1e-300);
                valid = dp.allFinite() && std::isfinite(relative) && relative < 1e-5 &&
                        backward < 5e-14;
            }
        }
    }
    if (!valid && a.solver_mode != 2) {
        ++a.eigen_calls;
        valid = acc::solve_linear(S, rhs, dp, relative);
    }
    return valid;
}
void Gpu::evaluate_device(const double *p, const double *A, bool derivatives, double *residual,
                          double *corrected, double *predicted, double *J, uint8_t *valid,
                          double *loss, uintptr_t stream_handle) {
    auto &a = *impl;
    a.select();
    if (!p || !A || !loss)
        throw std::invalid_argument("Device p, A and loss pointers are required");
    auto verify = [&](const void *pointer) {
        if (!pointer)
            return;
        cudaPointerAttributes attributes{};
        CU(cudaPointerGetAttributes(&attributes, pointer));
        if (attributes.type != cudaMemoryTypeDevice || attributes.device != a.device)
            throw std::invalid_argument(
                "Device buffer belongs to another GPU or is not CUDA device memory");
    };
    verify(p);
    verify(A);
    verify(residual);
    verify(corrected);
    verify(predicted);
    verify(J);
    verify(valid);
    verify(loss);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);
    if (stream) {
        unsigned int flags = 0;
        CU(cudaStreamGetFlags(stream, &flags));
        int stream_device = -1;
        CU(cudaStreamGetDevice(stream, &stream_device));
        if (stream_device != a.device)
            throw std::invalid_argument("CUDA stream belongs to another GPU");
    }
    BLAS(cublasSetStream(a.blas, stream));
    BLAS(cublasSetPointerMode(a.blas, CUBLAS_POINTER_MODE_DEVICE));
    try {
        a.launch(derivatives, p, A, stream);
        BLAS(cublasDdot(a.blas, int(a.w.n * 6), a.v.r, 1, a.v.r, 1, loss));
        finish_loss<<<1, 1, 0, stream>>>(a.v.invalid, loss);
        pack_evaluation<<<unsigned((a.w.n + 127) / 128), 128, 0, stream>>>(
            a.v, derivatives, residual, corrected, predicted, J, valid);
        CU(cudaGetLastError());
    } catch (...) {
        cublasSetPointerMode(a.blas, CUBLAS_POINTER_MODE_HOST);
        cublasSetStream(a.blas, 0);
        throw;
    }
    BLAS(cublasSetPointerMode(a.blas, CUBLAS_POINTER_MODE_HOST));
    BLAS(cublasSetStream(a.blas, 0));
    ++a.w.profile.evaluation_calls;
    if (derivatives)
        ++a.w.profile.derivative_calls;
}
void Gpu::download_A(double *output) {
    auto &a = *impl;
    a.select();
    CU(cudaMemcpy(output, a.A, a.w.n * sizeof(double), cudaMemcpyDeviceToHost));
}
void Gpu::infer(const double *p, const AccInferOptions &options, double *A, double *cost,
                uint8_t *valid, int32_t *bound, double *sensitivity, double *gap, int32_t *count) {
    auto &a = *impl;
    a.select();
    a.copy(a.p, p, NP);
    a.begin();
    if (a.w.order == TransformOrder::KeystoneThenBarrel)
        infer_kernel<TransformOrder::KeystoneThenBarrel><<<unsigned((a.w.n + 127) / 128), 128>>>(
            a.w.config, a.v, a.p, options, a.A, a.cost, a.v.valid, a.bound, a.sensitivity, a.gap,
            a.count);
    else
        infer_kernel<TransformOrder::BarrelThenKeystone><<<unsigned((a.w.n + 127) / 128), 128>>>(
            a.w.config, a.v, a.p, options, a.A, a.cost, a.v.valid, a.bound, a.sensitivity, a.gap,
            a.count);
    CU(cudaGetLastError());
    a.w.profile.inference_ms += a.end();
    auto download = [&](void *out, const void *in, size_t size) {
        if (out)
            CU(cudaMemcpy(out, in, size, cudaMemcpyDeviceToHost));
    };
    download(A, a.A, a.w.n * sizeof(double));
    download(cost, a.cost, a.w.n * sizeof(double));
    download(valid, a.v.valid, a.w.n * sizeof(uint8_t));
    download(bound, a.bound, a.w.n * sizeof(int));
    download(sensitivity, a.sensitivity, a.w.n * sizeof(double));
    download(gap, a.gap, a.w.n * sizeof(double));
    download(count, a.count, a.w.n * sizeof(int));
}
} // namespace acc
