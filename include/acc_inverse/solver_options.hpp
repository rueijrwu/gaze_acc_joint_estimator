#pragma once
#include "acc_inverse/model.hpp"
#include <array>
#include <functional>
namespace acc {
/* Numerical policy is separate from optical composition and data ownership.
 * A stage is one SolveControl plus a warm (p,A) state. Calling solve for a
 * sequence of controls implements staged fitting without changing the model. */
struct SolveControl {
    int max_iterations = 200;
    int iteration_offset = 0;
    uint32_t shared_parameter_mask = (uint32_t(1) << NP) - 1;
    bool optimize_accommodation = true;
    int max_damping_attempts = 8;
    int line_search_count = 6;
    int sustained_iterations = 5;
    double initial_damping = 1e-3;
    double damping_decrease = .3;
    double damping_increase = 10.;
    double minimum_damping = 1e-9;
    double projected_gradient_tolerance = 1e-6;
    double objective_tolerance_px2 = 1e-9;
    double state_step_tolerance = 1e-5;
    double sustained_gradient_tolerance = 1e-5;
    std::array<double, 16> line_search_scales{1., .5, .25, .125, .0625, .03125};
    std::function<void(const AccIteration &)> observer;
    bool parameter_enabled(int direction) const {
        return (shared_parameter_mask & (uint32_t(1) << direction)) != 0;
    }
    static SolveControl defaults(const AccSolveOptions &options) {
        SolveControl control;
        control.max_iterations = options.max_iterations;
        control.iteration_offset = options.iteration_offset;
        control.initial_damping = options.initial_damping;
        return control;
    }
};
} // namespace acc
