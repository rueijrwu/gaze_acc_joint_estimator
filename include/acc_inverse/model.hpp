#pragma once
#include "acc_inverse/api.h"
#include "acc_inverse/optical_stages.hpp"
namespace acc {
enum class TransformOrder : int32_t { BarrelThenKeystone = 0, KeystoneThenBarrel = 1 };
struct Config {
    double R, aref, span, Q[6];
    explicit Config(const AccConfig &config) : R(config.R), aref(config.aref), span(config.span) {
        for (int i = 0; i < 6; ++i)
            Q[i] = config.reference_px[i] / R;
    }
};
template <class T> ACC_HD inline T parameter(const double *p, int i, int direction) {
    return T(p[i]);
}
template <> ACC_HD inline Dual parameter<Dual>(const double *p, int i, int direction) {
    return {p[i], direction == i ? 1. : 0.};
}
template <class T> ACC_HD inline T accommodation(double A, int direction) {
    return T(A);
}
template <> ACC_HD inline Dual accommodation<Dual>(double A, int direction) {
    return {A, direction == 16 ? 1. : 0.};
}
template <class T> struct Transform : BarrelCoefficients<T>, KeystoneCoefficients<T> {
    T mx, my, predicted[6];
    bool valid;
};
/* Parameter layout and the accommodation-dependent optical law live in this
 * model policy, independently of the reusable transform stages and solver. */
template <class T>
ACC_HD inline Transform<T> coefficients(const Config &cfg, const double *p, double A, double gx,
                                        double gy, int direction = -1) {
    Transform<T> e;
    T a = (accommodation<T>(A, direction) - T(cfg.aref)) / T(cfg.span);
    T lx = parameter<T>(p, 0, direction) + parameter<T>(p, 1, direction) * a;
    T ax = parameter<T>(p, 2, direction) + parameter<T>(p, 3, direction) * a;
    T ly = parameter<T>(p, 4, direction) + parameter<T>(p, 5, direction) * a;
    T ay = parameter<T>(p, 6, direction) + parameter<T>(p, 7, direction) * a;
    T gamma = parameter<T>(p, 8, direction) + parameter<T>(p, 9, direction) * a;
    double tx = gx / 10., ty = gy / 10.;
    e.sx = T(1) + lx * T(tx) + ax * T(tx * tx);
    e.sy = T(1) + ly * T(tx) + ay * T(tx * tx);
    e.qx = gamma * T(tx);
    e.vx = T(1) + lx * T(ty) + ax * T(ty * ty);
    e.vy = T(1) + ly * T(ty) + ay * T(ty * ty);
    e.qy = gamma * T(ty);
    e.okx = parameter<T>(p, 10, direction);
    e.oky = parameter<T>(p, 11, direction);
    e.k = parameter<T>(p, 12, direction) * a;
    e.obx = parameter<T>(p, 13, direction);
    e.oby = parameter<T>(p, 14, direction);
    T phi = parameter<T>(p, 15, direction);
    e.c = cosine(phi);
    e.s = sine(phi);
    e.valid = KeystoneStage::valid_scales(e);
    return e;
}
template <class T, TransformOrder Order = TransformOrder::BarrelThenKeystone>
ACC_HD inline Transform<T> forward(const Config &cfg, const double *p, double A, double gx,
                                   double gy, int direction = -1) {
    auto e = coefficients<T>(cfg, p, A, gx, gy, direction);
    for (int j = 0; j < 3; ++j) {
        Point2<T> q{T(cfg.Q[2 * j]), T(cfg.Q[2 * j + 1])}, raw;
        if constexpr (Order == TransformOrder::BarrelThenKeystone)
            raw = KeystoneStage::forward(BarrelStage::forward(q, e), e, e.valid);
        else
            raw = BarrelStage::forward(KeystoneStage::forward(q, e, e.valid), e);
        e.predicted[2 * j] = raw.x;
        e.predicted[2 * j + 1] = raw.y;
        e.valid = e.valid && finite(value(raw.x)) && finite(value(raw.y));
    }
    auto centroid = CenteringStage::center_triangle(e.predicted);
    e.mx = centroid.x;
    e.my = centroid.y;
    return e;
}
/* ABI1 explicitly describes this model's 16 shared parameters and six pixel
 * residuals. Other scientific models can reuse the stages and provide the
 * same policy interface without changing the calibration algorithm. */
struct Frame {
    double residual[6], corrected[6], predicted[6], u[6], r2[3], k;
    bool valid;
};
template <TransformOrder Order> struct RelativeBarrelKeystonePolicy {
    static constexpr int parameter_count = 16, residual_count = 6, state_count = 1;
    static constexpr TransformOrder order = Order;
    ACC_HD static Frame evaluate(const Config &cfg, const double *p, double A, double gx, double gy,
                                 const double *y) {
        Frame out;
        auto e = forward<double, Order>(cfg, p, A, gx, gy);
        out.valid = e.valid;
        out.k = e.k;
        for (int i = 0; i < residual_count; ++i)
            out.predicted[i] = e.predicted[i];
        for (int j = 0; j < 3; ++j) {
            auto restored = CenteringStage::restore(Point2<double>{y[2 * j], y[2 * j + 1]},
                                                    Point2<double>{e.mx, e.my});
            Point2<double> target = restored;
            if constexpr (Order == TransformOrder::BarrelThenKeystone)
                target = KeystoneStage::inverse(restored, e, out.valid);
            auto radial = BarrelStage::inverse(target, e);
            out.valid = out.valid && radial.valid;
            Point2<double> corrected = radial.point;
            if constexpr (Order == TransformOrder::KeystoneThenBarrel)
                corrected = KeystoneStage::inverse(corrected, e, out.valid);
            out.u[2 * j] = radial.u.x;
            out.u[2 * j + 1] = radial.u.y;
            out.r2[j] = radial.r2;
            out.corrected[2 * j] = corrected.x;
            out.corrected[2 * j + 1] = corrected.y;
            out.residual[2 * j] = (corrected.x - cfg.Q[2 * j]) * cfg.R;
            out.residual[2 * j + 1] = (corrected.y - cfg.Q[2 * j + 1]) * cfg.R;
            out.valid = out.valid && finite(out.residual[2 * j]) && finite(out.residual[2 * j + 1]);
        }
        return out;
    }
    ACC_HD static void derivative(const Config &cfg, const double *p, double A, double gx,
                                  double gy, const double *y, const Frame &base, int direction,
                                  double *dr) {
        auto e = forward<Dual, Order>(cfg, p, A, gx, gy, direction);
        bool valid = e.valid;
        for (int j = 0; j < 3; ++j) {
            auto target = CenteringStage::restore(Point2<double>{y[2 * j], y[2 * j + 1]},
                                                  Point2<Dual>{e.mx, e.my});
            if constexpr (Order == TransformOrder::BarrelThenKeystone)
                target = KeystoneStage::inverse(target, e, valid);
            Point2<double> u{base.u[2 * j], base.u[2 * j + 1]};
            Point2<double> dtarget{target.x.d, target.y.d};
            Point2<double> dorigin{e.obx.d, e.oby.d};
            Point2<double> derivative =
                BarrelStage::inverse_jvp(dtarget, u, base.r2[j], base.k, e.k.d, dorigin);
            if constexpr (Order == TransformOrder::KeystoneThenBarrel) {
                Point2<Dual> intermediate{Dual(p[13] + u.x, derivative.x),
                                          Dual(p[14] + u.y, derivative.y)};
                auto corrected = KeystoneStage::inverse(intermediate, e, valid);
                derivative = Point2<double>{corrected.x.d, corrected.y.d};
            }
            dr[2 * j] = derivative.x * cfg.R;
            dr[2 * j + 1] = derivative.y * cfg.R;
        }
    }
};
/* Compiled model extension point: select a policy family here. A compatible
 * policy provides parameter_count/residual_count/state_count, evaluate and
 * derivative (JVP). The existing public ABI fixes these counts to 16/6/1;
 * supporting other counts also requires an explicitly versioned public ABI. */
template <TransformOrder Order> using ModelPolicy = RelativeBarrelKeystonePolicy<Order>;
using DefaultModelPolicy = ModelPolicy<TransformOrder::BarrelThenKeystone>;
constexpr int NP = DefaultModelPolicy::parameter_count, NR = DefaultModelPolicy::residual_count,
              ND = NP + 1;
static_assert(DefaultModelPolicy::state_count == 1, "Scalar Schur solver needs one frame state");
ACC_HD inline Frame evaluate_frame(const Config &cfg, const double *p, double A, double gx,
                                   double gy, const double *y,
                                   TransformOrder order = TransformOrder::BarrelThenKeystone) {
    if (order == TransformOrder::KeystoneThenBarrel)
        return ModelPolicy<TransformOrder::KeystoneThenBarrel>::evaluate(cfg, p, A, gx, gy, y);
    return DefaultModelPolicy::evaluate(cfg, p, A, gx, gy, y);
}
ACC_HD inline void derivative_frame(const Config &cfg, const double *p, double A, double gx,
                                    double gy, const double *y, const Frame &base, int direction,
                                    double *dr,
                                    TransformOrder order = TransformOrder::BarrelThenKeystone) {
    if (order == TransformOrder::KeystoneThenBarrel)
        ModelPolicy<TransformOrder::KeystoneThenBarrel>::derivative(cfg, p, A, gx, gy, y, base,
                                                                    direction, dr);
    else
        DefaultModelPolicy::derivative(cfg, p, A, gx, gy, y, base, direction, dr);
}
ACC_HD inline double frame_cost(const Frame &frame) {
    if (!frame.valid)
        return HUGE_VAL;
    double cost = 0.;
    for (int c = 0; c < NR; ++c)
        cost += frame.residual[c] * frame.residual[c];
    return cost;
}
} // namespace acc
