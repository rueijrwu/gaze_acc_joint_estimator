#ifndef ACC_INVERSE_API_H
#define ACC_INVERSE_API_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Existing ABI1 callers may continue testing status != 0. Last-error state is
 * thread-local, replaced by the next API operation, and preserved by queries. */
typedef enum {
    ACC_SUCCESS = 0,
    ACC_ERROR_INTERNAL = -1,
    ACC_ERROR_INVALID_ARGUMENT = -2,
    ACC_ERROR_OUT_OF_MEMORY = -3,
    ACC_ERROR_CUDA = -4,
    ACC_ERROR_NUMERICAL = -5,
    ACC_ERROR_CUBLAS = -6,
    ACC_ERROR_CUSOLVER = -7
} AccStatus;
typedef enum {
    ACC_LOG_DEBUG = 10,
    ACC_LOG_INFO = 20,
    ACC_LOG_WARNING = 30,
    ACC_LOG_ERROR = 40,
    ACC_LOG_OFF = 50
} AccLogLevel;
/* Optional native logging, disabled by default with no library file I/O.
 * Messages are borrowed for the call. Callbacks must not throw or re-enter an
 * operation on the same workspace. C++ callback exceptions are contained.
 * user_data must outlive registration. Device events report enqueue status. */
typedef void (*AccLogCallback)(void *user_data, int32_t level, int32_t status, const char *message);
#define ACC_OPTIONS_VERSION 1u
#define ACC_MAX_LINE_SEARCH_STEPS 16
typedef enum {
    ACC_PARAMETER_ELL_X0 = 0,
    ACC_PARAMETER_ELL_XA = 1,
    ACC_PARAMETER_ALPHA_X0 = 2,
    ACC_PARAMETER_ALPHA_XA = 3,
    ACC_PARAMETER_ELL_Y0 = 4,
    ACC_PARAMETER_ELL_YA = 5,
    ACC_PARAMETER_ALPHA_Y0 = 6,
    ACC_PARAMETER_ALPHA_YA = 7,
    ACC_PARAMETER_GAMMA0 = 8,
    ACC_PARAMETER_GAMMAA = 9,
    ACC_PARAMETER_KEYSTONE_ORIGIN_X = 10,
    ACC_PARAMETER_KEYSTONE_ORIGIN_Y = 11,
    ACC_PARAMETER_BARREL_BETA = 12,
    ACC_PARAMETER_BARREL_ORIGIN_X = 13,
    ACC_PARAMETER_BARREL_ORIGIN_Y = 14,
    ACC_PARAMETER_ORIENTATION = 15
} AccParameter;
#define ACC_PARAMETERS_BARREL 0x7000u
#define ACC_PARAMETERS_KEYSTONE 0x8fffu
#define ACC_PARAMETERS_ALL 0xffffu
typedef enum { ACC_BARREL_THEN_KEYSTONE = 0, ACC_KEYSTONE_THEN_BARREL = 1 } AccTransformOrder;
/* Initialize versioned descriptors with acc_*_default before customization.
 * struct_size must cover this version, version must be1, reserved must be0.
 * Model order is fixed per workspace. Debug derivatives are always unmasked. */
typedef struct {
    uint32_t struct_size, version;
    int32_t transform_order, reserved;
} AccModelOptions;
typedef struct {
    uint32_t struct_size, version, shared_parameter_mask;
    int32_t optimize_accommodation, max_iterations, iteration_offset;
    int32_t max_damping_attempts, line_search_count, sustained_iterations, reserved;
    double initial_damping, damping_decrease, damping_increase, minimum_damping;
    double projected_gradient_tolerance, objective_tolerance_px2;
    double state_step_tolerance, sustained_gradient_tolerance;
    double line_search_scales[ACC_MAX_LINE_SEARCH_STEPS];
} AccSolveSettings;
/* ABI version 1. Every array is contiguous FP64 row-major unless documented.
 * Coordinates: observed/corrected/predicted are normalized by config.R;
 * residuals and Jacobians are in P1-corrected pixels. The Jacobian shape is
 * [N,3,2,17], columns p[0:16] then accommodation in diopters. Both gaze axes
 * are fixed and use degrees. Bounds, weights and group labels are immutable.
 * Optional output arrays may be NULL; non-optional input pointers may not.
 * The caller must externally serialize ALL operations and destruction on the
 * same workspace, including completion of asynchronous device operations.
 * Independent workspaces may be used by independent host threads. */
typedef void *AccWorkspace;
typedef struct {
    double R, aref, span;
    double reference_px[6];
} AccConfig;
typedef struct {
    int32_t max_iterations, iteration_offset;
    double initial_damping;
} AccSolveOptions;
typedef struct {
    int32_t iteration, accepted, damping_attempts, line_search_evaluations;
    double inverse_RMS_px, improvement_px2, damping;
    double normalized_shared_gradient, normalized_state_gradient;
    double maximum_state_step, linear_relative_residual;
} AccIteration;
typedef struct {
    int32_t iterations, converged, frame_state_step_stable, termination_code;
    double initial_inverse_RMS_px, final_inverse_RMS_px, seconds;
    double final_normalized_shared_gradient, final_normalized_state_gradient;
} AccSolveSummary;
/* termination: 0 iteration limit, 1 projected gradient, 2 no decreasing step,
 * 3 objective AND state step, 4 sustained objective AND projected gradient. */
typedef struct {
    int32_t grid_points, refine_iterations;
    double accommodation_tolerance_D, ambiguity_cost_tolerance_px2;
} AccInferOptions;
typedef struct {
    double residual_kernel_ms, derivative_kernel_ms, normal_assembly_ms;
    double schur_assembly_ms, backsubstitution_ms, inference_ms;
    uint64_t evaluation_calls, derivative_calls, allocation_count;
} AccProfile;
int32_t acc_abi_version(void);
const char *acc_last_error(void);
int32_t acc_last_status(void);
int32_t acc_last_backend_code(void);
const char *acc_status_string(int32_t status);
int32_t acc_model_options_default(AccModelOptions *output);
int32_t acc_solve_settings_default(AccSolveSettings *output);
/* device < 0 creates a CPU-only workspace; device >= 0 selects that CUDA GPU.
 * y:[N,3,2], gaze:[N,2], low/high/weights:[N], groups:[N] int32.
 * weights must be finite and positive. Groups need not be sorted. */
AccWorkspace acc_create(const AccConfig *config, int64_t n, const double *y, const double *gaze,
                        const double *low, const double *high, const double *weights,
                        const int32_t *groups, int32_t device);
/* Status-returning constructor; *output is NULL on every failure. NULL model
 * options select defaults. acc_create retains the original default order. */
int32_t acc_create_ex(const AccConfig *config, const AccModelOptions *options, int64_t n,
                      const double *y, const double *gaze, const double *low, const double *high,
                      const double *weights, const int32_t *groups, int32_t device,
                      AccWorkspace *output);
int32_t acc_destroy(AccWorkspace workspace);
int32_t acc_set_log_callback(AccWorkspace workspace, AccLogCallback callback, void *user_data,
                             int32_t minimum_level);
const char *acc_metadata(AccWorkspace workspace);
/* threads0 and derivative_mode-1 retain the runtime-selected defaults. */
int32_t acc_set_launch_config(AccWorkspace workspace, int32_t residual_threads,
                              int32_t derivative_mode);
/* derivative_mode: 0 warp/frame (17 lanes), 1 thread/frame (17 directions).
 * Both paths use the same scalar dual and implicit barrel derivative. */
/* Timing fields are cumulative event durations; counters are cumulative. */
int32_t acc_get_profile(AccWorkspace workspace, AccProfile *output);
/* mode0 prefers cuSOLVER POTRF/POTRS and uses Eigen on diagnostic failure;
 * mode1 forces Eigen; mode2 requires cuSOLVER. CPU-only always uses Eigen. */
int32_t acc_set_solver_mode(AccWorkspace workspace, int32_t mode);
/* Asynchronous CUDA-device buffer API. All pointers must belong to the
 * workspace GPU. Outputs have the same row-major shapes as acc_evaluate.
 * No host array transfers occur. loss_device is required; other outputs may
 * be NULL. stream is a cudaStream_t cast to uintptr_t (0 is default stream).
 * Caller owns buffer lifetimes and must synchronize before another operation
 * on this workspace. One outstanding operation per workspace is supported. */
int32_t acc_evaluate_device(AccWorkspace workspace, const double *p_device, const double *A_device,
                            int32_t derivatives, double *residual_device, double *corrected_device,
                            double *predicted_device, double *jacobian_device,
                            uint8_t *valid_device, double *loss_device, uintptr_t stream);
/* backend 0 CPU, 1 CUDA. valid is [N] uint8; loss is weighted sum of squared
 * pixel residuals and is +inf whenever any frame is invalid. */
int32_t acc_evaluate(AccWorkspace workspace, const double *p, const double *A, int32_t derivatives,
                     int32_t backend, double *residual, double *corrected, double *predicted,
                     double *jacobian, uint8_t *valid, double *loss);
/* Normal blocks use active-bound accommodation columns; debug Jacobians above
 * are unprojected. G:[16,16], g:[16], B:[N,16], C/c:[N], active:[N]. */
int32_t acc_normals(AccWorkspace workspace, const double *p, const double *A, int32_t backend,
                    double *G, double *g, double *B, double *C, double *c, uint8_t *active,
                    double *loss, double *shared_gradient, double *state_gradient);
/* Schur debug follows the exact cached-normal path used by the CUDA solver.
 * S:[16,16], rhs/dp:[16], dA:[N]. Inputs may be CPU-only (backend 0). */
int32_t acc_step(AccWorkspace workspace, const double *p, const double *A, double damping,
                 int32_t backend, double *schur, double *rhs, double *dp, double *dA,
                 double *linear_relative_residual);
/* Complete native optimization. CPU-only workspace runs the same Eigen solver
 * and a scalar CPU evaluator for reduced diagnostics. CUDA workspace retains
 * all full-sized state and intermediate arrays on its selected device. */
int32_t acc_solve(AccWorkspace workspace, double *p, double *A, const AccSolveOptions *options,
                  AccIteration *records, int32_t record_capacity, AccSolveSummary *summary);
/* Mask bits0..15 select shared variables; all other bits must be0.
 * optimize_accommodation=0 freezes frame A. Stages call this repeatedly with
 * changed masks and explicit initial damping. NULL settings select defaults. */
int32_t acc_solve_ex(AccWorkspace workspace, double *p, double *A, const AccSolveSettings *settings,
                     AccIteration *records, int32_t record_capacity, AccSolveSummary *summary);
/* Frozen-model scalar minimization. Uses all endpoints and valid coarse local
 * minima; each valid bracket is refined independently. Outputs [N]. cost is
 * unweighted sum of six squared pixel residuals; sensitivity is ||dr/dA||.
 * ambiguity_gap is the cost gap to the next distinct resolved minimum (+inf
 * when absent); minima_count counts distinct resolved local minima.
 * The coarse grid limits which minima/valid islands can be detected; ambiguity
 * is a diagnostic at that resolution, not a uniqueness certificate.
 * bound: -1 lower, +1 upper, 0 interior; valid: 0 invalid, 1 valid, 2 ambiguous. */
int32_t acc_infer(AccWorkspace workspace, const double *p, const AccInferOptions *options,
                  double *A, double *cost, uint8_t *valid, int32_t *bound, double *sensitivity,
                  double *ambiguity_gap, int32_t *minima_count);
#ifdef __cplusplus
}
#endif
#endif
