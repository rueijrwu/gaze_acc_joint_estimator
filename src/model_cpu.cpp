#include "internal.hpp"
#include <algorithm>
#include <stdexcept>
namespace acc {
void evaluate_cpu(Workspace &w, const double *p, const double *A, bool derivatives,
                  double *residual, double *corrected, double *predicted, double *J, uint8_t *valid,
                  double &loss) {
    loss = 0.;
    bool all_valid = true;
    for (int64_t f = 0; f < w.n; ++f) {
        auto ev = evaluate_frame(w.config, p, A[f], w.gaze[2 * f], w.gaze[2 * f + 1], &w.y[6 * f],
                                 w.order);
        if (valid)
            valid[f] = ev.valid ? 1 : 0;
        all_valid = all_valid && ev.valid;
        if (residual)
            std::copy(ev.residual, ev.residual + 6, residual + 6 * f);
        if (corrected)
            std::copy(ev.corrected, ev.corrected + 6, corrected + 6 * f);
        if (predicted)
            std::copy(ev.predicted, ev.predicted + 6, predicted + 6 * f);
        loss += w.weight[f] * frame_cost(ev);
        if (derivatives && J) {
            for (int d = 0; d < ND; ++d) {
                double dr[6];
                if (ev.valid)
                    derivative_frame(w.config, p, A[f], w.gaze[2 * f], w.gaze[2 * f + 1],
                                     &w.y[6 * f], ev, d, dr, w.order);
                else
                    for (double &v : dr)
                        v = std::numeric_limits<double>::quiet_NaN();
                for (int r = 0; r < 6; ++r)
                    J[(6 * f + r) * ND + d] = dr[r];
            }
        }
    }
    if (!all_valid)
        loss = std::numeric_limits<double>::infinity();
}
Normal normals_cpu(Workspace &w, const double *p, const double *A, const SolveControl *control) {
    Normal out;
    out.B.resize(w.n * NP);
    out.C.resize(w.n);
    out.c.resize(w.n);
    out.active.resize(w.n);
    bool valid = true;
    for (int64_t f = 0; f < w.n; ++f) {
        auto ev = evaluate_frame(w.config, p, A[f], w.gaze[2 * f], w.gaze[2 * f + 1], &w.y[6 * f],
                                 w.order);
        if (!ev.valid) {
            valid = false;
            break;
        }
        Eigen::Matrix<double, 6, NP> J;
        Eigen::Matrix<double, 6, 1> j, r;
        for (int v = 0; v < 6; ++v)
            r(v) = ev.residual[v];
        for (int d = 0; d < ND; ++d) {
            double dr[6];
            derivative_frame(w.config, p, A[f], w.gaze[2 * f], w.gaze[2 * f + 1], &w.y[6 * f], ev,
                             d, dr, w.order);
            for (int v = 0; v < 6; ++v) {
                if (d < NP)
                    J(v, d) = dr[v];
                else
                    j(v) = dr[v];
            }
        }
        double wt = w.weight[f], rawc = wt * j.dot(r);
        bool active = (control && !control->optimize_accommodation) ||
                      (A[f] <= w.low[f] + 1e-9 && rawc > 0.) ||
                      (A[f] >= w.high[f] - 1e-9 && rawc < 0.);
        out.active[f] = active ? 1 : 0;
        if (active)
            j.setZero();
        out.C[f] = wt * j.squaredNorm();
        out.c[f] = wt * j.dot(r);
        if (control)
            for (int d = 0; d < NP; ++d)
                if (!control->parameter_enabled(d))
                    J.col(d).setZero();
        Vector bMasked = wt * J.transpose() * j;
        for (int d = 0; d < NP; ++d)
            out.B[f * NP + d] = bMasked(d);
        out.G.noalias() += wt * J.transpose() * J;
        out.g.noalias() += wt * J.transpose() * r;
        out.loss += wt * r.squaredNorm();
        out.sg = std::max(out.sg, std::abs(out.c[f]) / std::sqrt(std::max(out.C[f], 1e-20)));
    }
    if (!valid) {
        out.loss = std::numeric_limits<double>::infinity();
        return out;
    }
    for (int d = 0; d < NP; ++d)
        out.pg = std::max(out.pg, std::abs(out.g(d)) / std::sqrt(std::max(out.G(d, d), 1e-20)));
    return out;
}
void schur_cpu(const Normal &normal, double damping, Matrix &S, Vector &rhs) {
    S = normal.G;
    rhs = -normal.g;
    for (int d = 0; d < NP; ++d)
        S(d, d) += damping * std::max(normal.G(d, d), 1e-12);
    for (size_t f = 0; f < normal.C.size(); ++f) {
        Eigen::Map<const Vector> b(&normal.B[f * NP]);
        double D = normal.C[f] + damping * std::max(normal.C[f], 1e-12);
        S.noalias() -= (b * b.transpose()) / D;
        rhs.noalias() += b * (normal.c[f] / D);
    }
}
} // namespace acc
