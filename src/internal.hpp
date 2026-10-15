#pragma once
#include "acc_inverse/model.hpp"
#include "acc_inverse/solver_options.hpp"
#include <Eigen/Dense>
#include <array>
#include <memory>
#include <string>
#include <vector>
namespace acc {
using Matrix = Eigen::Matrix<double, NP, NP>;
using Vector = Eigen::Matrix<double, NP, 1>;
struct Normal {
    Matrix G = Matrix::Zero();
    Vector g = Vector::Zero();
    std::vector<double> B, C, c;
    std::vector<uint8_t> active;
    double loss = 0., pg = 0., sg = 0.;
};
struct InferResult {
    double A, cost, sensitivity, gap;
    int32_t bound, count;
    uint8_t valid;
};
class Gpu;
struct Workspace {
    Config config;
    TransformOrder order = TransformOrder::BarrelThenKeystone;
    int64_t n;
    std::vector<double> y, gaze, low, high, weight;
    std::vector<int32_t> group;
    std::unique_ptr<Gpu> gpu;
    std::string metadata;
    AccProfile profile{};
    int residual_threads = 128, derivative_mode = 0;
    Workspace(const AccConfig &, int64_t, const double *, const double *, const double *,
              const double *, const double *, const int32_t *, int,
              TransformOrder = TransformOrder::BarrelThenKeystone);
    ~Workspace();
};
void evaluate_cpu(Workspace &, const double *, const double *, bool, double *, double *, double *,
                  double *, uint8_t *, double &);
Normal normals_cpu(Workspace &, const double *, const double *, const SolveControl * = nullptr);
void schur_cpu(const Normal &, double, Matrix &, Vector &);
bool solve_linear(const Matrix &, const Vector &, Vector &, double &);
void solve(Workspace &, double *, double *, const AccSolveOptions &, AccIteration *, int,
           AccSolveSummary &);
void solve(Workspace &, double *, double *, const SolveControl &, AccIteration *, int,
           AccSolveSummary &);
void infer_cpu(Workspace &, const double *, const AccInferOptions &, double *, double *, uint8_t *,
               int32_t *, double *, double *, int32_t *);
/* Device storage is opaque to host translation units. Every public method is
 * synchronous at its result boundary; allocations occur only at construction. */
class Gpu {
  public:
    explicit Gpu(Workspace &, int);
    ~Gpu();
    std::string metadata() const;
    void set_launch(int, int);
    void upload(const double *, const double *);
    double evaluate(bool derivatives, bool trial = false, const SolveControl *control = nullptr);
    void download_evaluation(bool, double *, double *, double *, double *, uint8_t *);
    Normal normals(bool download_blocks = false, const SolveControl *control = nullptr);
    void download_normals(double *, double *, double *, uint8_t *);
    void schur(double, Matrix &, Vector &);
    void backsubstitute(double, const Vector &);
    double trial(const double *, double, double &, double, bool);
    void accept();
    void download_A(double *);
    void download_step(double *);
    bool linear_solve(const Matrix &, const Vector &, Vector &, double &);
    void set_solver_mode(int);
    void evaluate_device(const double *, const double *, bool, double *, double *, double *,
                         double *, uint8_t *, double *, uintptr_t);
    void infer(const double *, const AccInferOptions &, double *, double *, uint8_t *, int32_t *,
               double *, double *, int32_t *);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
} // namespace acc
