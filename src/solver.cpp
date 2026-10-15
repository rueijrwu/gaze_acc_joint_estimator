#include "acc_inverse/error.hpp"
#include "acc_inverse/inference.hpp"
#include "internal.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
namespace acc {
bool solve_linear(const Matrix &input, const Vector &rhs, Vector &step, double &residual) {
    residual = std::numeric_limits<double>::infinity();
    if (!input.allFinite() || !rhs.allFinite())
        return false;
    /* Symmetric diagonal equilibration changes only coordinates of the linear
     * solve. It does not change damping, objective or the accepted step. */
    Matrix S = (input + input.transpose()).eval() * .5;
    Vector scale;
    for (int i = 0; i < NP; ++i) {
        if (!(S(i, i) > 0.))
            return false;
        scale(i) = 1. / std::sqrt(S(i, i));
    }
    Matrix equilibrated = scale.asDiagonal() * S * scale.asDiagonal();
    Eigen::LDLT<Matrix> factor(equilibrated);
    if (factor.info() != Eigen::Success || !factor.isPositive())
        return false;
    step = scale.asDiagonal() * factor.solve(scale.asDiagonal() * rhs);
    if (factor.info() != Eigen::Success || !step.allFinite())
        return false;
    for (int pass = 0; pass < 2; ++pass) {
        Vector error = rhs - S * step;
        if (error.norm() <= 5e-13 * std::max(rhs.norm(), 1e-300))
            break;
        Vector correction = scale.asDiagonal() * factor.solve(scale.asDiagonal() * error);
        if (!correction.allFinite())
            return false;
        step += correction;
    }
    double error = (input * step - rhs).norm();
    residual = error / std::max(rhs.norm(), 1e-300);
    double backward = error / std::max(input.norm() * step.norm() + rhs.norm(), 1e-300);
    return step.allFinite() && std::isfinite(residual) && residual < 1e-5 && backward < 5e-14;
}
void solve(Workspace &w, double *p, double *A, const AccSolveOptions &options,
           AccIteration *records, int capacity, AccSolveSummary &summary) {
    solve(w, p, A, SolveControl::defaults(options), records, capacity, summary);
}
void solve(Workspace &w, double *p, double *A, const SolveControl &options, AccIteration *records,
           int capacity, AccSolveSummary &summary) {
    if (options.max_iterations < 0 || options.max_iterations > capacity ||
        options.iteration_offset < 0 || (options.max_iterations > 0 && !records) ||
        options.max_damping_attempts < 1 || options.max_damping_attempts > 64 ||
        options.line_search_count < 1 || options.line_search_count > 16 ||
        options.sustained_iterations < 1 || (options.shared_parameter_mask >> NP) != 0)
        throw std::invalid_argument("Invalid solve iteration, mask or search settings");
    auto positive = [](double x) { return std::isfinite(x) && x > 0.; };
    auto nonnegative = [](double x) { return std::isfinite(x) && x >= 0.; };
    if (!positive(options.initial_damping) || !positive(options.minimum_damping) ||
        !positive(options.damping_decrease) || options.damping_decrease > 1. ||
        !std::isfinite(options.damping_increase) || options.damping_increase <= 1. ||
        !nonnegative(options.projected_gradient_tolerance) ||
        !nonnegative(options.objective_tolerance_px2) ||
        !nonnegative(options.state_step_tolerance) ||
        !nonnegative(options.sustained_gradient_tolerance))
        throw std::invalid_argument("Invalid damping or stopping tolerance");
    for (int index = 0; index < options.line_search_count; ++index)
        if (!positive(options.line_search_scales[index]) ||
            options.line_search_scales[index] > 1. ||
            (index > 0 &&
             options.line_search_scales[index] >= options.line_search_scales[index - 1]))
            throw std::invalid_argument("Line search scales must decrease in (0,1]");
    for (int64_t f = 0; f < w.n; ++f)
        if (A[f] < w.low[f] - 1e-10 || A[f] > w.high[f] + 1e-10)
            throw std::invalid_argument("Initial accommodation outside bounds");
    auto start = std::chrono::steady_clock::now();
    summary = {};
    double loss;
    if (w.gpu) {
        w.gpu->upload(p, A);
        loss = w.gpu->evaluate(false);
    } else
        evaluate_cpu(w, p, A, false, nullptr, nullptr, nullptr, nullptr, nullptr, loss);
    if (!std::isfinite(loss))
        throw NativeError(ACC_ERROR_NUMERICAL, "Initial model has an invalid inverse branch");
    summary.initial_inverse_RMS_px = std::sqrt(loss);
    double damping = options.initial_damping;
    std::vector<double> cpu_step(w.gpu ? 0 : w.n), trialA(w.gpu ? 0 : w.n);
    bool converged = false;
    int termination = 0;
    double pg = 0., sg = 0.;
    for (int iteration = 0; iteration < options.max_iterations; ++iteration) {
        Normal normal = w.gpu ? w.gpu->normals(false, &options) : normals_cpu(w, p, A, &options);
        if (!std::isfinite(normal.loss))
            throw NativeError(ACC_ERROR_NUMERICAL,
                              "Accepted model became invalid during normal evaluation");
        loss = normal.loss;
        pg = normal.pg;
        sg = normal.sg;
        if (std::max(pg, sg) < options.projected_gradient_tolerance) {
            converged = true;
            termination = 1;
            break;
        }
        bool accepted = false;
        double improvement = 0., maximum_step = 0.,
               linear_residual = std::numeric_limits<double>::infinity();
        int attempts = 0, lines = 0;
        for (int attempt = 0; attempt < options.max_damping_attempts; ++attempt) {
            ++attempts;
            Matrix S;
            Vector rhs, dp;
            if (w.gpu)
                w.gpu->schur(damping, S, rhs);
            else
                schur_cpu(normal, damping, S, rhs);
            if (!(w.gpu ? w.gpu->linear_solve(S, rhs, dp, linear_residual)
                        : solve_linear(S, rhs, dp, linear_residual))) {
                damping *= options.damping_increase;
                continue;
            }
            if (w.gpu)
                w.gpu->backsubstitute(damping, dp);
            else
                for (int64_t f = 0; f < w.n; ++f) {
                    double dot = 0.;
                    for (int d = 0; d < NP; ++d)
                        dot += normal.B[f * NP + d] * dp(d);
                    double D = normal.C[f] + damping * std::max(normal.C[f], 1e-12);
                    cpu_step[f] = -(normal.c[f] + dot) / D;
                }
            for (int line = 0; line < options.line_search_count; ++line) {
                double alpha = options.line_search_scales[line];
                ++lines;
                std::array<double, NP> trialp;
                for (int d = 0; d < NP; ++d)
                    trialp[d] = options.parameter_enabled(d) ? p[d] + alpha * dp(d) : p[d];
                if (options.parameter_enabled(15))
                    trialp[15] = std::max(-1.57079632679489661923,
                                          std::min(1.57079632679489661923, trialp[15]));
                double trial = 0., step = 0.;
                if (w.gpu)
                    trial = w.gpu->trial(trialp.data(), alpha, step, loss,
                                         options.optimize_accommodation);
                else {
                    for (int64_t f = 0; f < w.n; ++f) {
                        trialA[f] = options.optimize_accommodation
                                        ? std::max(w.low[f],
                                                   std::min(w.high[f], A[f] + alpha * cpu_step[f]))
                                        : A[f];
                        step = std::max(step, std::abs(trialA[f] - A[f]));
                    }
                    evaluate_cpu(w, trialp.data(), trialA.data(), false, nullptr, nullptr, nullptr,
                                 nullptr, nullptr, trial);
                }
                if (trial < loss) {
                    improvement = loss - trial;
                    maximum_step = step;
                    loss = trial;
                    std::copy(trialp.begin(), trialp.end(), p);
                    if (w.gpu)
                        w.gpu->accept();
                    else
                        std::copy(trialA.begin(), trialA.end(), A);
                    damping = std::max(damping * options.damping_decrease, options.minimum_damping);
                    accepted = true;
                    break;
                }
            }
            if (accepted)
                break;
            damping *= options.damping_increase;
        }
        AccIteration row{};
        row.iteration = options.iteration_offset + iteration + 1;
        row.accepted = accepted ? 1 : 0;
        row.damping_attempts = attempts;
        row.line_search_evaluations = lines;
        row.inverse_RMS_px = std::sqrt(loss);
        row.improvement_px2 = improvement;
        row.damping = damping;
        row.normalized_shared_gradient = pg;
        row.normalized_state_gradient = sg;
        row.maximum_state_step = maximum_step;
        row.linear_relative_residual = linear_residual;
        records[summary.iterations++] = row;
        if (options.observer)
            options.observer(row);
        if (!accepted) {
            termination = 2;
            break;
        }
        if (improvement < options.objective_tolerance_px2 * std::max(1., loss) &&
            maximum_step < options.state_step_tolerance) {
            converged = true;
            termination = 3;
            break;
        }
        if (summary.iterations >= options.sustained_iterations &&
            std::max(pg, sg) < options.sustained_gradient_tolerance) {
            bool small = true;
            for (int r = summary.iterations - options.sustained_iterations; r < summary.iterations;
                 ++r)
                small = small && records[r].accepted &&
                        records[r].improvement_px2 <
                            options.objective_tolerance_px2 * std::max(1., loss);
            if (small) {
                converged = true;
                termination = 4;
                break;
            }
        }
    }
    Normal final_normal = w.gpu ? w.gpu->normals(false, &options) : normals_cpu(w, p, A, &options);
    if (!std::isfinite(final_normal.loss))
        throw NativeError(ACC_ERROR_NUMERICAL, "Final inverse verification failed");
    loss = final_normal.loss;
    pg = final_normal.pg;
    sg = final_normal.sg;
    if (w.gpu)
        w.gpu->download_A(A);
    summary.converged = converged ? 1 : 0;
    summary.termination_code = termination;
    summary.final_inverse_RMS_px = std::sqrt(loss);
    summary.final_normalized_shared_gradient = pg;
    summary.final_normalized_state_gradient = sg;
    summary.frame_state_step_stable =
        summary.iterations > 0 && records[summary.iterations - 1].accepted &&
        records[summary.iterations - 1].maximum_state_step < options.state_step_tolerance;
    summary.seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}
void infer_cpu(Workspace &w, const double *p, const AccInferOptions &options, double *A,
               double *cost, uint8_t *valid, int32_t *bound, double *sensitivity, double *gap,
               int32_t *count) {
    for (int64_t f = 0; f < w.n; ++f) {
        auto result =
            w.order == TransformOrder::KeystoneThenBarrel
                ? infer_frame<TransformOrder::KeystoneThenBarrel>(w.config, p, w.gaze[2 * f],
                                                                  w.gaze[2 * f + 1], &w.y[6 * f],
                                                                  w.low[f], w.high[f], options)
                : infer_frame<TransformOrder::BarrelThenKeystone>(w.config, p, w.gaze[2 * f],
                                                                  w.gaze[2 * f + 1], &w.y[6 * f],
                                                                  w.low[f], w.high[f], options);
        if (A)
            A[f] = result.A;
        if (cost)
            cost[f] = result.cost;
        if (valid)
            valid[f] = result.valid;
        if (bound)
            bound[f] = result.bound;
        if (sensitivity)
            sensitivity[f] = result.sensitivity;
        if (gap)
            gap[f] = result.gap;
        if (count)
            count[f] = result.count;
    }
}
} // namespace acc
