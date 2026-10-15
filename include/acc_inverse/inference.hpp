#pragma once
#include "acc_inverse/model.hpp"
namespace acc {
struct ScalarMinimum {
    double A = 0., cost = HUGE_VAL;
    double sensitivity = 0., gap = HUGE_VAL;
    int bound = 0, count = 0;
    unsigned char valid = 0;
};
template <TransformOrder Order = TransformOrder::BarrelThenKeystone>
ACC_HD inline double scalar_cost(const Config &cfg, const double *p, double A, double gx, double gy,
                                 const double *y) {
    return frame_cost(ModelPolicy<Order>::evaluate(cfg, p, A, gx, gy, y));
}
ACC_HD inline void add_minimum(double A, double cost, double separation, double &bestA,
                               double &bestCost, double &secondA, double &secondCost, int &count) {
    if (!finite(cost))
        return;
    if (!finite(bestCost)) {
        bestA = A;
        bestCost = cost;
        ++count;
        return;
    }
    if (::fabs(A - bestA) <= separation) {
        if (cost < bestCost) {
            bestA = A;
            bestCost = cost;
        }
        return;
    }
    if (cost < bestCost) {
        if (finite(bestCost)) {
            secondA = bestA;
            secondCost = bestCost;
        }
        bestA = A;
        bestCost = cost;
        ++count;
    } else if (::fabs(A - secondA) > separation) {
        if (cost < secondCost) {
            secondA = A;
            secondCost = cost;
        }
        ++count;
    } else if (cost < secondCost) {
        secondA = A;
        secondCost = cost;
    }
}
/* Golden refinement is confined to a coarse bracket whose three samples are
 * valid. An invalid newly sampled point closes that side of the bracket,
 * preventing a refinement step from walking through an invalid interval. */
template <TransformOrder Order = TransformOrder::BarrelThenKeystone>
ACC_HD inline void refine_minimum(const Config &cfg, const double *p, double gx, double gy,
                                  const double *y, double lo, double hi, double seed,
                                  double seed_cost, const AccInferOptions &options, double &bestA,
                                  double &bestCost) {
    bestA = seed;
    bestCost = seed_cost;
    constexpr double golden = .3819660112501051518;
    double x = seed;
    for (int iteration = 0;
         iteration < options.refine_iterations && hi - lo > options.accommodation_tolerance_D;
         ++iteration) {
        bool left = x - lo > hi - x;
        double trial = x + (left ? -golden * (x - lo) : golden * (hi - x));
        if (trial == x)
            break;
        double value = scalar_cost<Order>(cfg, p, trial, gx, gy, y);
        if (!finite(value)) {
            if (trial < x)
                lo = trial;
            else
                hi = trial;
            continue;
        }
        if (value < bestCost) {
            if (trial < x)
                hi = x;
            else
                lo = x;
            x = trial;
            bestA = trial;
            bestCost = value;
        } else {
            if (trial < x)
                lo = trial;
            else
                hi = trial;
        }
    }
}
template <TransformOrder Order = TransformOrder::BarrelThenKeystone>
ACC_HD inline ScalarMinimum infer_frame(const Config &cfg, const double *p, double gx, double gy,
                                        const double *y, double low, double high,
                                        const AccInferOptions &options) {
    ScalarMinimum result;
    result.A = low;
    if (low == high) {
        auto base = ModelPolicy<Order>::evaluate(cfg, p, low, gx, gy, y);
        result.cost = frame_cost(base);
        if (base.valid) {
            double dr[6];
            ModelPolicy<Order>::derivative(cfg, p, low, gx, gy, y, base, NP, dr);
            double sum = 0.;
            for (double z : dr)
                sum += z * z;
            result.sensitivity = ::sqrt(sum);
            result.valid = 1;
            result.bound = -1;
            result.count = 1;
        }
        return result;
    }
    double step = (high - low) / (options.grid_points - 1),
           separation = ::fmax(1e-10, (high - low) * 1e-8);
    double bestA = low, best = HUGE_VAL, secondA = low, second = best;
    int candidates = 0;
    double leftA = low, left = scalar_cost<Order>(cfg, p, low, gx, gy, y), before = HUGE_VAL;
    /* Every true endpoint is checked. It becomes a minimum candidate only
     * when its one-sided coarse neighbourhood is nondecreasing or invalid.
     * Refining that basin replaces its endpoint candidate, avoiding a false
     * ambiguity between a minimum and a nearby point in the same basin. */
    for (int index = 1; index < options.grid_points; ++index) {
        double rightA = index == options.grid_points - 1 ? high : low + index * step;
        double right = scalar_cost<Order>(cfg, p, rightA, gx, gy, y);
        bool local = finite(left) && left <= before && left <= right;
        if (local) {
            double candidateA = leftA, candidateCost = left;
            if (index == 1 && finite(right)) {
                double midpoint = (leftA + rightA) * .5,
                       middle = scalar_cost<Order>(cfg, p, midpoint, gx, gy, y);
                if (finite(middle)) {
                    double refinedA, refinedCost;
                    refine_minimum<Order>(cfg, p, gx, gy, y, leftA, rightA, midpoint, middle,
                                          options, refinedA, refinedCost);
                    if (refinedCost < candidateCost) {
                        candidateA = refinedA;
                        candidateCost = refinedCost;
                    }
                }
            } else if (index >= 2 && finite(before) && finite(right))
                refine_minimum<Order>(cfg, p, gx, gy, y, leftA - step, rightA, leftA, left, options,
                                      candidateA, candidateCost);
            add_minimum(candidateA, candidateCost, separation, bestA, best, secondA, second,
                        candidates);
        }
        if (index == options.grid_points - 1 && finite(right) && right <= left) {
            double candidateA = rightA, candidateCost = right;
            if (finite(left)) {
                double midpoint = (leftA + rightA) * .5,
                       middle = scalar_cost<Order>(cfg, p, midpoint, gx, gy, y);
                if (finite(middle)) {
                    double refinedA, refinedCost;
                    refine_minimum<Order>(cfg, p, gx, gy, y, leftA, rightA, midpoint, middle,
                                          options, refinedA, refinedCost);
                    if (refinedCost < candidateCost) {
                        candidateA = refinedA;
                        candidateCost = refinedCost;
                    }
                }
            }
            add_minimum(candidateA, candidateCost, separation, bestA, best, secondA, second,
                        candidates);
        }
        before = left;
        left = right;
        leftA = rightA;
    }
    if (!finite(best))
        return result;
    result.A = bestA;
    result.cost = best;
    result.gap = finite(second) ? ::fmax(0., second - best) : second;
    result.count = candidates > 0 ? candidates : 1;
    result.valid = result.gap <= options.ambiguity_cost_tolerance_px2 ? 2 : 1;
    result.bound = bestA == low ? -1 : (bestA == high ? 1 : 0);
    auto base = ModelPolicy<Order>::evaluate(cfg, p, bestA, gx, gy, y);
    double dr[6], sum = 0.;
    ModelPolicy<Order>::derivative(cfg, p, bestA, gx, gy, y, base, NP, dr);
    for (double z : dr)
        sum += z * z;
    result.sensitivity = ::sqrt(sum);
    return result;
}
} // namespace acc
