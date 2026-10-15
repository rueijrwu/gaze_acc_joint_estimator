#pragma once
#include "acc_inverse/scalar.hpp"
namespace acc {
struct ModelNumerics {
    static constexpr double root_tolerance = 2e-13;
    static constexpr double validity_margin = 1e-7;
    static constexpr double equation_tolerance = 1e-9;
    static constexpr int radial_iterations = 40;
};
template <class T> struct BarrelCoefficients {
    T k, obx, oby;
};
template <class T> struct KeystoneCoefficients {
    T sx, sy, vx, vy, qx, qy, okx, oky, c, s;
};
struct BarrelInverse {
    Point2<double> point, u;
    double r2;
    bool valid;
};
/* Pure radial stage. Its inverse is restricted to the inner monotonic branch
 * for a negative coefficient. It never differentiates numerical iterations. */
struct BarrelStage {
    template <class T>
    ACC_HD static Point2<T> forward(Point2<T> q, const BarrelCoefficients<T> &b) {
        T ux = q.x - b.obx, uy = q.y - b.oby, r2 = ux * ux + uy * uy;
        return {q.x + b.k * r2 * ux, q.y + b.k * r2 * uy};
    }
    ACC_HD static BarrelInverse inverse(Point2<double> target,
                                        const BarrelCoefficients<double> &b) {
        double vx = target.x - b.obx, vy = target.y - b.oby, k = b.k;
        double rd = ::sqrt(vx * vx + vy * vy), r = rd;
        bool valid = finite(rd) && finite(k);
        if (k < 0.) {
            double turn = 1. / ::sqrt(-3. * k);
            valid = valid && rd < turn * (1. + k * turn * turn);
        }
        if (!valid)
            r = 0.;
        else if (rd == 0. || k == 0.)
            r = rd;
        else {
            double low = 0., high = k < 0. ? 1. / ::sqrt(-3. * k) : rd;
            r = ::fmin(rd, .95 * high);
            for (int iteration = 0; iteration < ModelNumerics::radial_iterations; ++iteration) {
                double f = r * (1. + k * r * r) - rd;
                if (::fabs(f) < ModelNumerics::root_tolerance)
                    break;
                if (f < 0.)
                    low = r;
                else
                    high = r;
                double next = r - f / ::fmax(1. + 3. * k * r * r, 1e-12);
                r = next >= low && next <= high ? next : (low + high) * .5;
            }
        }
        double scale = rd > 0. ? r / rd : 0., ux = vx * scale, uy = vy * scale,
               r2 = ux * ux + uy * uy;
        double tangential = 1. + k * r2, radial = 1. + 3. * k * r2;
        valid = valid && radial > ModelNumerics::validity_margin &&
                ::fabs(tangential * ux - vx) < ModelNumerics::equation_tolerance &&
                ::fabs(tangential * uy - vy) < ModelNumerics::equation_tolerance;
        return {{b.obx + ux, b.oby + uy}, {ux, uy}, r2, valid};
    }
    /* J_B^-1[dT-r^2*u*dk+k*(r^2I+2uu^T)*dOB]. */
    ACC_HD static Point2<double> inverse_jvp(Point2<double> dtarget, Point2<double> u, double r2,
                                             double k, double dk, Point2<double> dOB) {
        double rhsx = dtarget.x - r2 * u.x * dk, rhsy = dtarget.y - r2 * u.y * dk;
        if (dOB.x != 0.) {
            rhsx += k * (r2 + 2 * u.x * u.x) * dOB.x;
            rhsy += k * (2 * u.x * u.y) * dOB.x;
        }
        if (dOB.y != 0.) {
            rhsx += k * (2 * u.x * u.y) * dOB.y;
            rhsy += k * (r2 + 2 * u.y * u.y) * dOB.y;
        }
        double factor = 2 * k / (1. + 3 * k * r2), tangential = 1. + k * r2,
               dot = u.x * rhsx + u.y * rhsy;
        return {(rhsx - factor * u.x * dot) / tangential, (rhsy - factor * u.y * dot) / tangential};
    }
};
/* The existing vertical convention reuses the horizontal law after a 90deg
 * rotation. This stage includes orientation and its separate optical origin. */
struct KeystoneStage {
    template <class T> ACC_HD static bool valid_scales(const KeystoneCoefficients<T> &e) {
        return value(e.sx) > ModelNumerics::validity_margin &&
               value(e.sy) > ModelNumerics::validity_margin &&
               value(e.vx) > ModelNumerics::validity_margin &&
               value(e.vy) > ModelNumerics::validity_margin;
    }
    template <class T>
    ACC_HD static Point2<T> forward(Point2<T> q, const KeystoneCoefficients<T> &e, bool &valid) {
        T wx = q.x - e.okx, wy = q.y - e.oky;
        T zx = e.c * wx + e.s * wy, zy = -e.s * wx + e.c * wy;
        T dx = T(1) + e.qx * zy, hx = e.sx * zx / dx, hy = e.sy * zy / dx;
        T dy = T(1) - e.qy * hx, ox = e.vy * hx / dy, oy = e.vx * hy / dy;
        Point2<T> raw{e.okx + e.c * ox - e.s * oy, e.oky + e.s * ox + e.c * oy};
        valid = valid && value(dx) > ModelNumerics::validity_margin &&
                value(dy) > ModelNumerics::validity_margin && finite(value(raw.x)) &&
                finite(value(raw.y));
        return raw;
    }
    template <class T>
    ACC_HD static Point2<T> inverse(Point2<T> target, const KeystoneCoefficients<T> &e,
                                    bool &valid) {
        T wx = target.x - e.okx, wy = target.y - e.oky;
        T ox = e.c * wx + e.s * wy, oy = -e.s * wx + e.c * wy;
        T dy = e.vy + e.qy * ox, hx = ox / dy, hy = oy * e.vy / (e.vx * dy);
        T dx = e.sy - e.qx * hy, zx = hx * e.sy / (e.sx * dx), zy = hy / dx;
        Point2<T> out{e.okx + e.c * zx - e.s * zy, e.oky + e.s * zx + e.c * zy};
        valid = valid && value(dx) > ModelNumerics::validity_margin &&
                value(dy) > ModelNumerics::validity_margin && finite(value(out.x)) &&
                finite(value(out.y));
        return out;
    }
};
struct CenteringStage {
    template <class T> ACC_HD static Point2<T> center_triangle(T *raw) {
        T mx = T(0), my = T(0);
        for (int j = 0; j < 3; ++j) {
            mx = mx + raw[2 * j];
            my = my + raw[2 * j + 1];
        }
        mx = mx / T(3);
        my = my / T(3);
        for (int j = 0; j < 3; ++j) {
            raw[2 * j] = raw[2 * j] - mx;
            raw[2 * j + 1] = raw[2 * j + 1] - my;
        }
        return {mx, my};
    }
    template <class T>
    ACC_HD static Point2<T> restore(Point2<double> centered, Point2<T> centroid) {
        return {T(centered.x) + centroid.x, T(centered.y) + centroid.y};
    }
};
} // namespace acc
