#pragma once
#include <cmath>
#include <cstdint>
#ifdef __CUDACC__
#define ACC_HD __host__ __device__
#else
#define ACC_HD
#endif
namespace acc {
/* One sensitivity direction per scalar keeps CUDA register pressure bounded.
 * Numerical roots only use primal doubles; BarrelStage supplies their JVP. */
struct Dual {
    double v, d;
    ACC_HD Dual(double value = 0., double derivative = 0.) : v(value), d(derivative) {}
};
ACC_HD inline Dual operator+(Dual a, Dual b) {
    return {a.v + b.v, a.d + b.d};
}
ACC_HD inline Dual operator-(Dual a, Dual b) {
    return {a.v - b.v, a.d - b.d};
}
ACC_HD inline Dual operator-(Dual a) {
    return {-a.v, -a.d};
}
ACC_HD inline Dual operator*(Dual a, Dual b) {
    return {a.v * b.v, a.d * b.v + a.v * b.d};
}
ACC_HD inline Dual operator/(Dual a, Dual b) {
    double t = a.v / b.v;
    return {t, (a.d - t * b.d) / b.v};
}
ACC_HD inline double value(double x) {
    return x;
}
ACC_HD inline double value(Dual x) {
    return x.v;
}
ACC_HD inline double sine(double x) {
    return ::sin(x);
}
ACC_HD inline double cosine(double x) {
    return ::cos(x);
}
ACC_HD inline Dual sine(Dual x) {
    return {::sin(x.v), ::cos(x.v) * x.d};
}
ACC_HD inline Dual cosine(Dual x) {
    return {::cos(x.v), -::sin(x.v) * x.d};
}
ACC_HD inline bool finite(double x) {
    return x == x && ::fabs(x) <= 1.7976931348623157e308;
}
template <class T> struct Point2 {
    T x, y;
};
} // namespace acc
