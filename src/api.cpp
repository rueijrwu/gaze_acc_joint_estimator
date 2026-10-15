#include "acc_inverse/error.hpp"
#include "build_info.hpp"
#include "internal.hpp"
#include <algorithm>
#include <climits>
#include <cmath>
#include <exception>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#ifndef ACC_BUILD_FLAGS
#define ACC_BUILD_FLAGS "unspecified"
#endif
#ifndef ACC_CUDA_ARCHITECTURES
#define ACC_CUDA_ARCHITECTURES "unspecified"
#endif
namespace {
thread_local std::string error_message;
thread_local int32_t error_status = ACC_SUCCESS;
thread_local int32_t error_backend_code = 0;
std::mutex handle_mutex;
std::unordered_set<AccWorkspace> handles;
struct Logger {
    AccLogCallback callback = nullptr;
    void *user = nullptr;
    int32_t minimum_level = ACC_LOG_OFF;
};
std::unordered_map<AccWorkspace, Logger> loggers;
bool logging_enabled(AccWorkspace handle, int32_t level) {
    std::lock_guard<std::mutex> lock(handle_mutex);
    const auto found = loggers.find(handle);
    return found != loggers.end() && found->second.callback && level >= found->second.minimum_level;
}
void set_error(int32_t status, int32_t backend_code, const char *message) noexcept {
    error_status = status;
    error_backend_code = backend_code;
    try {
        error_message = message ? message : "";
    } catch (...) {
        error_message.clear();
    }
}
void emit_log(AccWorkspace handle, int32_t level, int32_t status, const char *operation) noexcept {
    try {
        Logger logger;
        {
            std::lock_guard<std::mutex> lock(handle_mutex);
            auto found = loggers.find(handle);
            if (found == loggers.end() || !found->second.callback ||
                level < found->second.minimum_level)
                return;
            logger = found->second;
        }
        // Format only after confirming logging is enabled. A callback may use
        // another workspace; preserve this operation's thread-local diagnostics.
        auto saved_message = error_message;
        const auto saved_status = error_status, saved_backend = error_backend_code;
        std::string message(operation);
        message += status == ACC_SUCCESS ? " completed" : ": ";
        if (status != ACC_SUCCESS)
            message += error_message.empty() ? acc_status_string(status) : error_message;
        try {
            logger.callback(logger.user, level, status, message.c_str());
        } catch (...) {
            // No callback exception may cross an extern-C entry point.
        }
        error_message.swap(saved_message);
        error_status = saved_status;
        error_backend_code = saved_backend;
    } catch (...) {
        // Logging is optional and must never change numerical control flow.
    }
}
acc::Workspace &checked(AccWorkspace handle) {
    std::lock_guard<std::mutex> lock(handle_mutex);
    if (!handle || !handles.count(handle))
        throw std::invalid_argument("Invalid workspace handle");
    return *static_cast<acc::Workspace *>(handle);
}
void finite_input(const double *data, int64_t size, const char *name) {
    if (!data)
        throw std::invalid_argument(std::string("Null ") + name);
    for (int64_t i = 0; i < size; ++i)
        if (!std::isfinite(data[i]))
            throw std::invalid_argument(std::string("Non-finite ") + name);
}
void state_input(acc::Workspace &w, const double *p, const double *A) {
    finite_input(p, 16, "parameters");
    finite_input(A, w.n, "accommodation");
}
void backend_input(acc::Workspace &w, int backend) {
    if (backend != 0 && backend != 1)
        throw std::invalid_argument("Backend must be 0 (CPU) or 1 (CUDA)");
    if (backend == 1 && !w.gpu)
        throw std::invalid_argument("CUDA requested from a CPU-only workspace");
}
AccModelOptions default_model() {
    AccModelOptions options{};
    options.struct_size = sizeof(options);
    options.version = ACC_OPTIONS_VERSION;
    options.transform_order = ACC_BARREL_THEN_KEYSTONE;
    return options;
}
AccSolveSettings default_settings() {
    AccSolveSettings settings{};
    settings.struct_size = sizeof(settings);
    settings.version = ACC_OPTIONS_VERSION;
    settings.shared_parameter_mask = 0xffffu;
    settings.optimize_accommodation = 1;
    settings.max_iterations = 200;
    settings.max_damping_attempts = 8;
    settings.line_search_count = 6;
    settings.sustained_iterations = 5;
    settings.initial_damping = 1e-3;
    settings.damping_decrease = .3;
    settings.damping_increase = 10.;
    settings.minimum_damping = 1e-9;
    settings.projected_gradient_tolerance = 1e-6;
    settings.objective_tolerance_px2 = 1e-9;
    settings.state_step_tolerance = 1e-5;
    settings.sustained_gradient_tolerance = 1e-5;
    for (int i = 0; i < settings.line_search_count; ++i)
        settings.line_search_scales[i] = std::ldexp(1., -i);
    return settings;
}
acc::TransformOrder model_order(const AccModelOptions *options) {
    const auto defaults = default_model();
    const auto &value = options ? *options : defaults;
    if (value.struct_size < sizeof(AccModelOptions) || value.version != ACC_OPTIONS_VERSION ||
        value.reserved != 0 ||
        (value.transform_order != ACC_BARREL_THEN_KEYSTONE &&
         value.transform_order != ACC_KEYSTONE_THEN_BARREL))
        throw std::invalid_argument("Invalid model options version, size, reserved field or order");
    return static_cast<acc::TransformOrder>(value.transform_order);
}
acc::SolveControl solve_control(const AccSolveSettings *settings) {
    const auto defaults = default_settings();
    const auto &value = settings ? *settings : defaults;
    if (value.struct_size < sizeof(AccSolveSettings) || value.version != ACC_OPTIONS_VERSION ||
        value.reserved != 0 || (value.shared_parameter_mask & ~0xffffu) != 0 ||
        (value.optimize_accommodation != 0 && value.optimize_accommodation != 1))
        throw std::invalid_argument(
            "Invalid solve settings version, size, mask, boolean or reserved field");
    if (value.max_iterations < 0 || value.iteration_offset < 0 ||
        value.max_iterations > INT_MAX - value.iteration_offset || value.max_damping_attempts < 1 ||
        value.line_search_count < 1 || value.line_search_count > ACC_MAX_LINE_SEARCH_STEPS ||
        value.sustained_iterations < 1)
        throw std::invalid_argument(
            "Invalid solve iteration, damping-attempt or line-search count");
    if (!(std::isfinite(value.initial_damping) && value.initial_damping > 0. &&
          std::isfinite(value.minimum_damping) && value.minimum_damping > 0. &&
          std::isfinite(value.damping_decrease) && value.damping_decrease > 0. &&
          value.damping_decrease < 1. && std::isfinite(value.damping_increase) &&
          value.damping_increase > 1.))
        throw std::invalid_argument("Invalid damping values");
    for (double tolerance : {value.projected_gradient_tolerance, value.objective_tolerance_px2,
                             value.state_step_tolerance, value.sustained_gradient_tolerance})
        if (!(std::isfinite(tolerance) && tolerance >= 0.))
            throw std::invalid_argument("Solver tolerances must be finite and nonnegative");
    for (int i = 0; i < value.line_search_count; ++i)
        if (!(std::isfinite(value.line_search_scales[i]) && value.line_search_scales[i] > 0. &&
              value.line_search_scales[i] <= 1.))
            throw std::invalid_argument("Line-search scales must be finite and in (0,1]");
    acc::SolveControl control;
    control.shared_parameter_mask = value.shared_parameter_mask;
    control.optimize_accommodation = value.optimize_accommodation != 0;
    control.max_iterations = value.max_iterations;
    control.iteration_offset = value.iteration_offset;
    control.max_damping_attempts = value.max_damping_attempts;
    control.line_search_count = value.line_search_count;
    control.sustained_iterations = value.sustained_iterations;
    control.initial_damping = value.initial_damping;
    control.damping_decrease = value.damping_decrease;
    control.damping_increase = value.damping_increase;
    control.minimum_damping = value.minimum_damping;
    control.projected_gradient_tolerance = value.projected_gradient_tolerance;
    control.objective_tolerance_px2 = value.objective_tolerance_px2;
    control.state_step_tolerance = value.state_step_tolerance;
    control.sustained_gradient_tolerance = value.sustained_gradient_tolerance;
    std::copy(std::begin(value.line_search_scales), std::end(value.line_search_scales),
              control.line_search_scales.begin());
    return control;
}
void attach_iteration_logging(acc::SolveControl &control, AccWorkspace handle) {
    if (!logging_enabled(handle, ACC_LOG_INFO))
        return;
    control.observer = [handle](const AccIteration &iteration) {
        // Default execution never constructs this observer or formats rows.
        try {
            if (!logging_enabled(handle, ACC_LOG_INFO))
                return;
            std::ostringstream message;
            message << "iteration " << iteration.iteration << " inverse RMS "
                    << iteration.inverse_RMS_px << " accepted " << iteration.accepted;
            emit_log(handle, ACC_LOG_INFO, ACC_SUCCESS, message.str().c_str());
        } catch (...) {
            // Optional instrumentation cannot terminate a numerical iteration.
        }
    };
}
AccWorkspace create_checked(const AccConfig *cfg, int64_t n, const double *y, const double *gaze,
                            const double *low, const double *high, const double *weights,
                            const int32_t *groups, int32_t device, acc::TransformOrder order) {
    if (!cfg)
        throw std::invalid_argument("Null configuration");
    if (!(std::isfinite(cfg->R) && cfg->R > 0. && std::isfinite(cfg->aref) &&
          std::isfinite(cfg->span) && cfg->span > 0.))
        throw std::invalid_argument("Invalid R, aref or span");
    finite_input(cfg->reference_px, 6, "reference triangle");
    if (n <= 0 || n > INT_MAX / 6)
        throw std::invalid_argument("Frame count must be in [1,INT_MAX/6]");
    finite_input(y, n * 6, "observations");
    finite_input(gaze, n * 2, "gaze");
    finite_input(low, n, "lower bounds");
    finite_input(high, n, "upper bounds");
    finite_input(weights, n, "weights");
    if (!groups)
        throw std::invalid_argument("Null group labels");
    for (int64_t f = 0; f < n; ++f) {
        if (low[f] > high[f] || low[f] < 0.)
            throw std::invalid_argument("Accommodation bounds must satisfy 0<=low<=high");
        if (!(weights[f] > 0.))
            throw std::invalid_argument("Weights must be positive");
        if (groups[f] < 0)
            throw std::invalid_argument("Group labels must be nonnegative");
    }
    auto workspace = std::make_unique<acc::Workspace>(*cfg, n, y, gaze, low, high, weights, groups,
                                                      device, order);
    const auto result = workspace.get();
    {
        std::lock_guard<std::mutex> lock(handle_mutex);
        handles.insert(result);
    }
    workspace.release();
    return result;
}
template <class F>
int guarded(F &&operation, AccWorkspace handle = nullptr,
            const char *name = "native API") noexcept {
    try {
        set_error(ACC_SUCCESS, 0, "");
        operation();
    } catch (const acc::BackendError &e) {
        set_error(e.status == ACC_SUCCESS ? ACC_ERROR_INTERNAL : e.status, e.backend_code,
                  e.what());
    } catch (const std::bad_alloc &e) {
        set_error(ACC_ERROR_OUT_OF_MEMORY, 0, e.what());
    } catch (const std::invalid_argument &e) {
        set_error(ACC_ERROR_INVALID_ARGUMENT, 0, e.what());
    } catch (const std::out_of_range &e) {
        set_error(ACC_ERROR_INVALID_ARGUMENT, 0, e.what());
    } catch (const std::exception &e) {
        set_error(ACC_ERROR_INTERNAL, 0, e.what());
    } catch (...) {
        set_error(ACC_ERROR_INTERNAL, 0, "Unknown native exception");
    }
    const auto status = error_status;
    emit_log(handle, status == ACC_SUCCESS ? ACC_LOG_INFO : ACC_LOG_ERROR, status, name);
    return status;
}
void matrix_output(const acc::Matrix &matrix, double *output) {
    if (output)
        for (int i = 0; i < 16; ++i)
            for (int j = 0; j < 16; ++j)
                output[i * 16 + j] = matrix(i, j);
}
void vector_output(const acc::Vector &vector, double *output) {
    if (output)
        std::copy(vector.data(), vector.data() + 16, output);
}
} // namespace
namespace acc {
Workspace::Workspace(const AccConfig &cfg, int64_t N, const double *Y, const double *G,
                     const double *L, const double *U, const double *W, const int32_t *groups,
                     int device, TransformOrder model_order)
    : config(cfg), order(model_order), n(N), y(Y, Y + N * 6), gaze(G, G + N * 2), low(L, L + N),
      high(U, U + N), weight(W, W + N), group(groups, groups + N) {
    if (device >= 0)
        gpu = std::make_unique<Gpu>(*this, device);
}
Workspace::~Workspace() = default;
} // namespace acc
extern "C" {
int32_t acc_abi_version(void) {
    return 1;
}
const char *acc_last_error(void) {
    return error_message.empty() && error_status != ACC_SUCCESS ? acc_status_string(error_status)
                                                                : error_message.c_str();
}
int32_t acc_last_status(void) {
    return error_status;
}
int32_t acc_last_backend_code(void) {
    return error_backend_code;
}
const char *acc_status_string(int32_t status) {
    switch (status) {
    case ACC_SUCCESS:
        return "Success";
    case ACC_ERROR_INVALID_ARGUMENT:
        return "Invalid argument";
    case ACC_ERROR_OUT_OF_MEMORY:
        return "Out of memory";
    case ACC_ERROR_CUDA:
        return "CUDA runtime failure";
    case ACC_ERROR_NUMERICAL:
        return "Numerical failure";
    case ACC_ERROR_CUBLAS:
        return "cuBLAS failure";
    case ACC_ERROR_CUSOLVER:
        return "cuSOLVER failure";
    case ACC_ERROR_INTERNAL:
        return "Internal failure";
    default:
        return "Unknown status";
    }
}
int32_t acc_model_options_default(AccModelOptions *output) {
    return guarded([&] {
        if (!output)
            throw std::invalid_argument("Null model options output");
        *output = default_model();
    });
}
int32_t acc_solve_settings_default(AccSolveSettings *output) {
    return guarded([&] {
        if (!output)
            throw std::invalid_argument("Null solve settings output");
        *output = default_settings();
    });
}
AccWorkspace acc_create(const AccConfig *cfg, int64_t n, const double *y, const double *gaze,
                        const double *low, const double *high, const double *weights,
                        const int32_t *groups, int32_t device) {
    AccWorkspace result = nullptr;
    acc_create_ex(cfg, nullptr, n, y, gaze, low, high, weights, groups, device, &result);
    return result;
}
int32_t acc_create_ex(const AccConfig *cfg, const AccModelOptions *options, int64_t n,
                      const double *y, const double *gaze, const double *low, const double *high,
                      const double *weights, const int32_t *groups, int32_t device,
                      AccWorkspace *output) {
    if (output)
        *output = nullptr;
    return guarded([&] {
        if (!output)
            throw std::invalid_argument("Null workspace output");
        *output = create_checked(cfg, n, y, gaze, low, high, weights, groups, device,
                                 model_order(options));
    });
}
int32_t acc_destroy(AccWorkspace handle) {
    return guarded(
        [&] {
            auto &w = checked(handle);
            {
                std::lock_guard<std::mutex> lock(handle_mutex);
                handles.erase(handle);
                loggers.erase(handle);
            }
            delete &w;
        },
        handle, "acc_destroy");
}
int32_t acc_set_log_callback(AccWorkspace handle, AccLogCallback callback, void *user,
                             int32_t minimum_level) {
    return guarded(
        [&] {
            checked(handle);
            if (minimum_level != ACC_LOG_DEBUG && minimum_level != ACC_LOG_INFO &&
                minimum_level != ACC_LOG_WARNING && minimum_level != ACC_LOG_ERROR &&
                minimum_level != ACC_LOG_OFF)
                throw std::invalid_argument("Invalid minimum log level");
            std::lock_guard<std::mutex> lock(handle_mutex);
            if (callback && minimum_level != ACC_LOG_OFF)
                loggers[handle] = Logger{callback, user, minimum_level};
            else
                loggers.erase(handle);
        },
        handle, "acc_set_log_callback");
}
const char *acc_metadata(AccWorkspace handle) {
    const char *result = nullptr;
    guarded(
        [&] {
            auto &w = checked(handle);
            std::ostringstream out;
            out << "{\"abi_version\":1,\"backend\":\"" << (w.gpu ? "CUDA" : "CPU")
                << "\",\"precision\":\"FP64\",\"frames\":" << w.n << ",\"host_compiler\":\""
                << __VERSION__ << "\",\"host_architecture\":\""
#if defined(__aarch64__)
                << "aarch64"
#elif defined(__x86_64__)
                << "x86_64"
#else
                << "unknown"
#endif
                << "\",\"eigen\":\"" << EIGEN_WORLD_VERSION << '.' << EIGEN_MAJOR_VERSION << '.'
                << EIGEN_MINOR_VERSION << "\",\"build_flags\":\"" << ACC_BUILD_FLAGS
                << "\",\"cuda_architectures\":\"" << ACC_CUDA_ARCHITECTURES
                << "\",\"residual_threads\":" << w.residual_threads
                << ",\"derivative_mode\":" << w.derivative_mode
                << ",\"transform_order\":" << static_cast<int>(w.order)
                << ",\"allocation_count\":" << w.profile.allocation_count;
            if (w.gpu)
                out << ',' << w.gpu->metadata();
            out << '}';
            w.metadata = out.str();
            result = w.metadata.c_str();
        },
        handle, "acc_metadata");
    return result;
}
int32_t acc_get_profile(AccWorkspace handle, AccProfile *output) {
    return guarded(
        [&] {
            auto &w = checked(handle);
            if (!output)
                throw std::invalid_argument("Null profile output");
            *output = w.profile;
        },
        handle, "acc_get_profile");
}
int32_t acc_set_launch_config(AccWorkspace handle, int32_t threads, int32_t mode) {
    return guarded(
        [&] {
            auto &w = checked(handle);
            if (threads == 0)
                threads = w.residual_threads;
            if (mode == -1)
                mode = w.derivative_mode;
            if (threads < 32 || threads > 1024 || threads % 32 || (mode != 0 && mode != 1))
                throw std::invalid_argument(
                    "Threads must be a multiple of32 in[32,1024]; derivative mode0or1");
            w.residual_threads = threads;
            w.derivative_mode = mode;
            if (w.gpu)
                w.gpu->set_launch(threads, mode);
        },
        handle, "acc_set_launch_config");
}
int32_t acc_set_solver_mode(AccWorkspace handle, int32_t mode) {
    return guarded(
        [&] {
            auto &w = checked(handle);
            if (mode < 0 || mode > 2)
                throw std::invalid_argument("Solver mode must be0,1or2");
            if (w.gpu)
                w.gpu->set_solver_mode(mode);
            else if (mode == 2)
                throw std::invalid_argument("cuSOLVER requested from CPU-only workspace");
        },
        handle, "acc_set_solver_mode");
}
int32_t acc_evaluate_device(AccWorkspace handle, const double *p, const double *A,
                            int32_t derivatives, double *r, double *corrected, double *predicted,
                            double *J, uint8_t *valid, double *loss, uintptr_t stream) {
    return guarded(
        [&] {
            auto &w = checked(handle);
            if (!w.gpu)
                throw std::invalid_argument("Device API requires CUDA workspace");
            w.gpu->evaluate_device(p, A, derivatives != 0, r, corrected, predicted, J, valid, loss,
                                   stream);
        },
        handle, "acc_evaluate_device");
}
int32_t acc_evaluate(AccWorkspace handle, const double *p, const double *A, int32_t derivatives,
                     int32_t backend, double *residual, double *corrected, double *predicted,
                     double *J, uint8_t *valid, double *loss) {
    return guarded(
        [&] {
            auto &w = checked(handle);
            state_input(w, p, A);
            backend_input(w, backend);
            double objective;
            if (backend) {
                w.gpu->upload(p, A);
                objective = w.gpu->evaluate(derivatives != 0);
                w.gpu->download_evaluation(derivatives != 0, residual, corrected, predicted, J,
                                           valid);
            } else
                acc::evaluate_cpu(w, p, A, derivatives != 0, residual, corrected, predicted, J,
                                  valid, objective);
            if (loss)
                *loss = objective;
        },
        handle, "acc_evaluate");
}
int32_t acc_normals(AccWorkspace handle, const double *p, const double *A, int32_t backend,
                    double *G, double *g, double *B, double *C, double *c, uint8_t *active,
                    double *loss, double *pg, double *sg) {
    return guarded(
        [&] {
            auto &w = checked(handle);
            state_input(w, p, A);
            backend_input(w, backend);
            acc::Normal normal;
            if (backend) {
                w.gpu->upload(p, A);
                normal = w.gpu->normals();
                if (std::isfinite(normal.loss))
                    w.gpu->download_normals(B, C, c, active);
            } else {
                normal = acc::normals_cpu(w, p, A);
                if (std::isfinite(normal.loss)) {
                    if (B)
                        std::copy(normal.B.begin(), normal.B.end(), B);
                    if (C)
                        std::copy(normal.C.begin(), normal.C.end(), C);
                    if (c)
                        std::copy(normal.c.begin(), normal.c.end(), c);
                    if (active)
                        std::copy(normal.active.begin(), normal.active.end(), active);
                }
            }
            matrix_output(normal.G, G);
            vector_output(normal.g, g);
            if (loss)
                *loss = normal.loss;
            if (pg)
                *pg = normal.pg;
            if (sg)
                *sg = normal.sg;
        },
        handle, "acc_normals");
}
int32_t acc_step(AccWorkspace handle, const double *p, const double *A, double damping,
                 int32_t backend, double *S, double *rhs, double *dp, double *dA,
                 double *relative) {
    return guarded(
        [&] {
            auto &w = checked(handle);
            state_input(w, p, A);
            backend_input(w, backend);
            if (!(std::isfinite(damping) && damping > 0.))
                throw std::invalid_argument("Damping must be finite and positive");
            acc::Normal normal;
            if (backend) {
                w.gpu->upload(p, A);
                normal = w.gpu->normals();
            } else
                normal = acc::normals_cpu(w, p, A);
            if (!std::isfinite(normal.loss))
                throw acc::NativeError(ACC_ERROR_NUMERICAL,
                                       "Invalid inverse model in step evaluation");
            acc::Matrix schur;
            acc::Vector vector, step;
            double error;
            if (backend)
                w.gpu->schur(damping, schur, vector);
            else
                acc::schur_cpu(normal, damping, schur, vector);
            if (!(backend ? w.gpu->linear_solve(schur, vector, step, error)
                          : acc::solve_linear(schur, vector, step, error)))
                throw acc::NativeError(ACC_ERROR_NUMERICAL,
                                       "Shared solve failed the original-system residual check");
            matrix_output(schur, S);
            vector_output(vector, rhs);
            vector_output(step, dp);
            if (relative)
                *relative = error;
            if (backend) {
                w.gpu->backsubstitute(damping, step);
                if (dA) { /* debug-only download method below */
                    w.gpu->download_step(dA);
                }
            } else if (dA)
                for (int64_t f = 0; f < w.n; ++f) {
                    double dot = 0.;
                    for (int d = 0; d < 16; ++d)
                        dot += normal.B[f * 16 + d] * step(d);
                    dA[f] = -(normal.c[f] + dot) /
                            (normal.C[f] + damping * std::max(normal.C[f], 1e-12));
                }
        },
        handle, "acc_step");
}
int32_t acc_solve(AccWorkspace handle, double *p, double *A, const AccSolveOptions *options,
                  AccIteration *records, int32_t capacity, AccSolveSummary *summary) {
    return guarded(
        [&] {
            auto &w = checked(handle);
            state_input(w, p, A);
            if (!options || !summary || (options->max_iterations > 0 && !records))
                throw std::invalid_argument("Null solve options/records/summary");
            if (options->max_iterations < 0 || options->iteration_offset < 0 ||
                options->max_iterations > INT_MAX - options->iteration_offset)
                throw std::invalid_argument("Invalid iteration count or offset");
            auto control = acc::SolveControl::defaults(*options);
            attach_iteration_logging(control, handle);
            acc::solve(w, p, A, control, records, capacity, *summary);
        },
        handle, "acc_solve");
}
int32_t acc_solve_ex(AccWorkspace handle, double *p, double *A, const AccSolveSettings *settings,
                     AccIteration *records, int32_t capacity, AccSolveSummary *summary) {
    return guarded(
        [&] {
            auto &w = checked(handle);
            state_input(w, p, A);
            auto control = solve_control(settings);
            if (!summary || (control.max_iterations > 0 && !records) ||
                capacity < control.max_iterations)
                throw std::invalid_argument("Null solve outputs or insufficient record capacity");
            attach_iteration_logging(control, handle);
            acc::solve(w, p, A, control, records, capacity, *summary);
        },
        handle, "acc_solve_ex");
}
int32_t acc_infer(AccWorkspace handle, const double *p, const AccInferOptions *options, double *A,
                  double *cost, uint8_t *valid, int32_t *bound, double *sensitivity, double *gap,
                  int32_t *count) {
    return guarded(
        [&] {
            auto &w = checked(handle);
            finite_input(p, 16, "parameters");
            if (!options || options->grid_points < 3 || options->grid_points > 4097 ||
                options->refine_iterations < 1 || options->refine_iterations > 1000 ||
                !(std::isfinite(options->accommodation_tolerance_D) &&
                  options->accommodation_tolerance_D > 0.) ||
                !(std::isfinite(options->ambiguity_cost_tolerance_px2) &&
                  options->ambiguity_cost_tolerance_px2 >= 0.))
                throw std::invalid_argument("Invalid scalar minimization options");
            if (w.gpu)
                w.gpu->infer(p, *options, A, cost, valid, bound, sensitivity, gap, count);
            else
                acc::infer_cpu(w, p, *options, A, cost, valid, bound, sensitivity, gap, count);
        },
        handle, "acc_infer");
}
} // extern C
