#pragma once

#include "acc_inverse/api.h"
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace acc_inverse {

enum class TransformOrder : std::int32_t {
    BarrelThenKeystone = ACC_BARREL_THEN_KEYSTONE,
    KeystoneThenBarrel = ACC_KEYSTONE_THEN_BARREL
};

class Error : public std::runtime_error {
  public:
    Error(std::int32_t status, std::int32_t backend_code, std::string message)
        : std::runtime_error(std::move(message)), status_(status), backend_code_(backend_code) {}
    std::int32_t status() const noexcept {
        return status_;
    }
    std::int32_t backend_code() const noexcept {
        return backend_code_;
    }

  private:
    std::int32_t status_, backend_code_;
};

inline void check(std::int32_t status, const char *operation = "acc_inverse") {
    if (status != ACC_SUCCESS) {
        const auto backend = acc_last_backend_code();
        const std::string message = acc_last_error();
        throw Error(status, backend, std::string(operation) + ": " + message);
    }
}

inline AccModelOptions model_options(TransformOrder order = TransformOrder::BarrelThenKeystone) {
    AccModelOptions options{};
    check(acc_model_options_default(&options), "model options defaults");
    options.transform_order = static_cast<std::int32_t>(order);
    return options;
}

inline AccSolveSettings solve_settings() {
    AccSolveSettings settings{};
    check(acc_solve_settings_default(&settings), "solve settings defaults");
    return settings;
}

struct EvaluationBuffers {
    double *residual = nullptr, *corrected = nullptr, *predicted = nullptr, *jacobian = nullptr;
    std::uint8_t *valid = nullptr;
};
struct NormalBuffers {
    double *G = nullptr, *g = nullptr, *B = nullptr, *C = nullptr, *c = nullptr;
    std::uint8_t *active = nullptr;
};
struct NormalDiagnostics {
    double loss, shared_gradient, state_gradient;
};
struct StepBuffers {
    double *schur = nullptr, *rhs = nullptr, *dp = nullptr, *dA = nullptr;
};
struct InferenceBuffers {
    double *accommodation = nullptr, *cost = nullptr;
    std::uint8_t *valid = nullptr;
    std::int32_t *bound = nullptr;
    double *sensitivity = nullptr, *ambiguity_gap = nullptr;
    std::int32_t *minima_count = nullptr;
};
struct SolveResult {
    AccSolveSummary summary{};
    std::vector<AccIteration> records;
};

// Move-only ownership of one workspace. All operations, including destruction
// and completion of async device calls, require external serialization on that
// workspace. Input/output array ownership stays with the caller. The opaque
// native_handle() is borrowed and must not be independently destroyed.
class Workspace {
  public:
    Workspace() noexcept = default;
    Workspace(const AccConfig &config, std::int64_t n, const double *observed, const double *gaze,
              const double *low, const double *high, const double *weights,
              const std::int32_t *groups, std::int32_t device = 0,
              TransformOrder order = TransformOrder::BarrelThenKeystone)
        : Workspace(config, model_options(order), n, observed, gaze, low, high, weights, groups,
                    device) {}
    Workspace(const AccConfig &config, const AccModelOptions &options, std::int64_t n,
              const double *observed, const double *gaze, const double *low, const double *high,
              const double *weights, const std::int32_t *groups, std::int32_t device = 0)
        : device_(device) {
        check(acc_create_ex(&config, &options, n, observed, gaze, low, high, weights, groups,
                            device, &handle_),
              "create workspace");
    }
    ~Workspace() noexcept {
        reset();
    }
    Workspace(const Workspace &) = delete;
    Workspace &operator=(const Workspace &) = delete;
    Workspace(Workspace &&other) noexcept
        : handle_(std::exchange(other.handle_, nullptr)), device_(other.device_) {}
    Workspace &operator=(Workspace &&other) noexcept {
        if (this != &other) {
            reset();
            handle_ = std::exchange(other.handle_, nullptr);
            device_ = other.device_;
        }
        return *this;
    }
    explicit operator bool() const noexcept {
        return handle_ != nullptr;
    }
    AccWorkspace native_handle() const noexcept {
        return handle_;
    }
    std::int32_t backend() const noexcept {
        return device_ < 0 ? 0 : 1;
    }
    void close() {
        if (handle_) {
            auto handle = std::exchange(handle_, nullptr);
            check(acc_destroy(handle), "close workspace");
        }
    }
    std::string metadata() const {
        const char *value = acc_metadata(handle_);
        if (!value)
            check(acc_last_status(), "workspace metadata");
        return value ? value : "";
    }
    AccProfile profile() const {
        AccProfile result{};
        check(acc_get_profile(handle_, &result), "workspace profile");
        return result;
    }
    void set_launch_config(std::int32_t threads = 0, std::int32_t derivative_mode = -1) {
        check(acc_set_launch_config(handle_, threads, derivative_mode), "launch configuration");
    }
    void set_solver_mode(std::int32_t mode = 0) {
        check(acc_set_solver_mode(handle_, mode), "dense solver configuration");
    }
    void set_log_callback(AccLogCallback callback, void *user_data = nullptr,
                          std::int32_t minimum_level = ACC_LOG_INFO) {
        check(acc_set_log_callback(handle_, callback, user_data, minimum_level),
              "logging configuration");
    }
    double evaluate(const double *p, const double *A, bool derivatives = false,
                    EvaluationBuffers output = {}, std::int32_t selected_backend = -1) {
        double loss = 0.;
        check(acc_evaluate(handle_, p, A, derivatives ? 1 : 0,
                           selected_backend < 0 ? backend() : selected_backend, output.residual,
                           output.corrected, output.predicted, output.jacobian, output.valid,
                           &loss),
              "evaluate");
        return loss;
    }
    NormalDiagnostics normals(const double *p, const double *A, NormalBuffers output = {},
                              std::int32_t selected_backend = -1) {
        NormalDiagnostics result{};
        check(acc_normals(handle_, p, A, selected_backend < 0 ? backend() : selected_backend,
                          output.G, output.g, output.B, output.C, output.c, output.active,
                          &result.loss, &result.shared_gradient, &result.state_gradient),
              "normals");
        return result;
    }
    double step(const double *p, const double *A, double damping, StepBuffers output = {},
                std::int32_t selected_backend = -1) {
        double relative = 0.;
        check(acc_step(handle_, p, A, damping, selected_backend < 0 ? backend() : selected_backend,
                       output.schur, output.rhs, output.dp, output.dA, &relative),
              "Schur step");
        return relative;
    }
    SolveResult solve(double *p, double *A, const AccSolveSettings &settings = solve_settings()) {
        if (settings.max_iterations < 0)
            throw Error(ACC_ERROR_INVALID_ARGUMENT, 0, "Negative iteration count");
        SolveResult result;
        result.records.resize(static_cast<std::size_t>(settings.max_iterations));
        check(acc_solve_ex(handle_, p, A, &settings, result.records.data(), settings.max_iterations,
                           &result.summary),
              "solve");
        if (result.summary.iterations < 0 || result.summary.iterations > settings.max_iterations)
            throw Error(ACC_ERROR_INTERNAL, 0, "Invalid native iteration record count");
        result.records.resize(static_cast<std::size_t>(result.summary.iterations));
        return result;
    }
    void infer(const double *p, InferenceBuffers output,
               const AccInferOptions &options = AccInferOptions{33, 96, 1e-10, 1e-8}) {
        check(acc_infer(handle_, p, &options, output.accommodation, output.cost, output.valid,
                        output.bound, output.sensitivity, output.ambiguity_gap,
                        output.minima_count),
              "infer accommodation");
    }
    // Asynchronous device buffers: the caller retains arrays and stream and
    // must synchronize that stream before reusing/closing this workspace.
    void evaluate_device(const double *p, const double *A, bool derivatives,
                         EvaluationBuffers output, double *loss, std::uintptr_t stream = 0) {
        check(acc_evaluate_device(handle_, p, A, derivatives ? 1 : 0, output.residual,
                                  output.corrected, output.predicted, output.jacobian, output.valid,
                                  loss, stream),
              "enqueue device evaluation");
    }

  private:
    void reset() noexcept {
        if (handle_)
            acc_destroy(std::exchange(handle_, nullptr));
    }
    AccWorkspace handle_ = nullptr;
    std::int32_t device_ = 0;
};

} // namespace acc_inverse
