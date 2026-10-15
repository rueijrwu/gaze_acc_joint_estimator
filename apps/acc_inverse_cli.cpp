#include "acc_inverse/api.h"
#include <H5Cpp.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>
namespace fs = std::filesystem;
namespace {
void native_check(int code) {
    if (code)
        throw std::runtime_error(acc_last_error());
}
struct Handle {
    AccWorkspace value = nullptr;
    ~Handle() {
        if (value)
            acc_destroy(value);
    }
};
struct Temporary {
    fs::path path;
    bool committed = false;
    ~Temporary() {
        if (!committed) {
            std::error_code ignored;
            fs::remove(path, ignored);
        }
    }
};
bool exists(const H5::H5File &file, const std::string &name) {
    return H5Lexists(file.getId(), name.c_str(), H5P_DEFAULT) > 0;
}
std::vector<hsize_t> shape(const H5::DataSet &dataset) {
    auto space = dataset.getSpace();
    int rank = space.getSimpleExtentNdims();
    std::vector<hsize_t> result(rank);
    if (rank)
        space.getSimpleExtentDims(result.data());
    return result;
}
size_t elements(const std::vector<hsize_t> &dimensions) {
    size_t n = 1;
    for (auto dimension : dimensions)
        n *= size_t(dimension);
    return n;
}
std::vector<double> read_double(H5::H5File &file, const std::string &path,
                                const std::vector<hsize_t> &expected) {
    auto dataset = file.openDataSet(path);
    if (shape(dataset) != expected || dataset.getTypeClass() != H5T_FLOAT ||
        dataset.getDataType().getSize() != 8)
        throw std::runtime_error("Expected FP64 dataset with exact shape: " + path);
    std::vector<double> out(elements(expected));
    dataset.read(out.data(), H5::PredType::NATIVE_DOUBLE);
    return out;
}
double scalar(H5::H5File &file, const std::string &path) {
    auto value = read_double(file, path, {});
    return value[0];
}
std::vector<int32_t> read_group(H5::H5File &file, hsize_t n) {
    auto dataset = file.openDataSet("/inputs/group");
    if (shape(dataset) != std::vector<hsize_t>{n} || dataset.getTypeClass() != H5T_INTEGER ||
        dataset.getDataType().getSize() != 4)
        throw std::runtime_error("Expected /inputs/group int32[N]");
    std::vector<int32_t> out(n);
    dataset.read(out.data(), H5::PredType::NATIVE_INT32);
    return out;
}
void write(H5::H5File &file, const std::string &path, const void *data, const H5::DataType &type,
           const std::vector<hsize_t> &dimensions) {
    H5::DataSpace space(dimensions.empty() ? H5S_SCALAR : H5S_SIMPLE);
    if (!dimensions.empty())
        space = H5::DataSpace(int(dimensions.size()), dimensions.data());
    auto dataset = file.createDataSet(path, type, space);
    dataset.write(data, type);
}
void attr(H5::H5Object &object, const std::string &name, double value) {
    auto attribute =
        object.createAttribute(name, H5::PredType::NATIVE_DOUBLE, H5::DataSpace(H5S_SCALAR));
    attribute.write(H5::PredType::NATIVE_DOUBLE, &value);
}
void attr(H5::H5Object &object, const std::string &name, int32_t value) {
    auto attribute =
        object.createAttribute(name, H5::PredType::NATIVE_INT32, H5::DataSpace(H5S_SCALAR));
    attribute.write(H5::PredType::NATIVE_INT32, &value);
}
void attr(H5::H5Object &object, const std::string &name, const std::string &value) {
    H5::StrType type(H5::PredType::C_S1, H5T_VARIABLE);
    type.setCset(H5T_CSET_UTF8);
    auto attribute = object.createAttribute(name, type, H5::DataSpace(H5S_SCALAR));
    attribute.write(type, value);
}
std::string termination(int code) {
    switch (code) {
    case 1:
        return "Projected gradient";
    case 2:
        return "No decreasing projected step";
    case 3:
        return "Objective and frame-state step tolerance";
    case 4:
        return "Sustained objective tolerance and projected gradient";
    default:
        return "Iteration limit";
    }
}
struct Arguments {
    std::string input, output, mode = "joint", state = "initial", order;
    int device = 0, maxiter = 200, grid = 33, refine = 96, threads = 0, derivative_mode = -1,
        solver_mode = 0;
    double tolerance = 1e-10, ambiguity = 1e-6, damping = 1e-3;
    bool derivatives = false, freeze_accommodation = false;
    uint32_t shared_mask = 0xffff;
    AccSolveSettings settings{};
    Arguments() {
        native_check(acc_solve_settings_default(&settings));
    }
};
void usage() {
    std::cout
        << "acc_inverse_cli --input INPUT.h5 --output OUTPUT.h5 [--mode joint|infer|evaluate]\n"
           "  [--state initial|reference|fit] [--maxiter 200] [--device 0|-1]\n"
           "  [--order barrel-keystone|keystone-barrel] [--shared-mask 0xffff]\n"
           "  [--freeze-accommodation] [--line-search 1,.5,.25,.125,.0625,.03125]\n"
           "  [--threads N] [--derivative-mode warp|thread] [--solver cusolver|eigen|strict]\n"
           "  [--damping .001] [--grid 33] [--refine 96] [--tolerance 1e-10]\n"
           "  [--ambiguity-cost 1e-6] [--derivatives]\n"
           "Input schema_version=1; all arithmetic FP64. Output is written atomically.\n";
}
Arguments parse(int argc, char **argv) {
    Arguments a;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if (key == "--help" || key == "-h") {
            usage();
            std::exit(0);
        }
        if (key == "--freeze-accommodation") {
            a.freeze_accommodation = true;
            continue;
        }
        if (key == "--derivatives") {
            a.derivatives = true;
            continue;
        }
        if (i + 1 == argc)
            throw std::invalid_argument("Missing argument for " + key);
        std::string value = argv[++i];
        if (key == "--input")
            a.input = value;
        else if (key == "--output")
            a.output = value;
        else if (key == "--mode")
            a.mode = value;
        else if (key == "--state")
            a.state = value;
        else if (key == "--order") {
            if (value != "barrel-keystone" && value != "keystone-barrel")
                throw std::invalid_argument("Order must be barrel-keystone or keystone-barrel");
            a.order = value;
        } else if (key == "--shared-mask") {
            if (value == "barrel")
                a.shared_mask = ACC_PARAMETERS_BARREL;
            else if (value == "keystone")
                a.shared_mask = ACC_PARAMETERS_KEYSTONE;
            else if (value == "all")
                a.shared_mask = ACC_PARAMETERS_ALL;
            else if (value == "none")
                a.shared_mask = 0;
            else {
                size_t consumed = 0;
                auto mask = std::stoul(value, &consumed, 0);
                if (consumed != value.size() || mask > 0xffff)
                    throw std::invalid_argument("Shared mask must use only bits0..15");
                a.shared_mask = uint32_t(mask);
            }
        } else if (key == "--line-search") {
            size_t begin = 0;
            int count = 0;
            while (begin < value.size()) {
                if (count == ACC_MAX_LINE_SEARCH_STEPS)
                    throw std::invalid_argument("Too many line-search scales");
                size_t end = value.find(',', begin);
                std::string token =
                    value.substr(begin, end == std::string::npos ? end : end - begin);
                size_t consumed = 0;
                a.settings.line_search_scales[count++] = std::stod(token, &consumed);
                if (consumed != token.size())
                    throw std::invalid_argument("Invalid line-search scale");
                if (end == std::string::npos)
                    break;
                begin = end + 1;
            }
            a.settings.line_search_count = count;
        } else if (key == "--projected-gradient-tolerance")
            a.settings.projected_gradient_tolerance = std::stod(value);
        else if (key == "--objective-tolerance")
            a.settings.objective_tolerance_px2 = std::stod(value);
        else if (key == "--state-step-tolerance")
            a.settings.state_step_tolerance = std::stod(value);
        else if (key == "--sustained-gradient-tolerance")
            a.settings.sustained_gradient_tolerance = std::stod(value);
        else if (key == "--sustained-iterations")
            a.settings.sustained_iterations = std::stoi(value);
        else if (key == "--damping-attempts")
            a.settings.max_damping_attempts = std::stoi(value);
        else if (key == "--damping-decrease")
            a.settings.damping_decrease = std::stod(value);
        else if (key == "--damping-increase")
            a.settings.damping_increase = std::stod(value);
        else if (key == "--minimum-damping")
            a.settings.minimum_damping = std::stod(value);
        else if (key == "--device")
            a.device = std::stoi(value);
        else if (key == "--maxiter")
            a.maxiter = std::stoi(value);
        else if (key == "--grid")
            a.grid = std::stoi(value);
        else if (key == "--refine")
            a.refine = std::stoi(value);
        else if (key == "--threads")
            a.threads = std::stoi(value);
        else if (key == "--tolerance")
            a.tolerance = std::stod(value);
        else if (key == "--ambiguity-cost")
            a.ambiguity = std::stod(value);
        else if (key == "--damping")
            a.damping = std::stod(value);
        else if (key == "--derivative-mode") {
            if (value != "warp" && value != "thread")
                throw std::invalid_argument("Derivative mode must be warp or thread");
            a.derivative_mode = value == "thread" ? 1 : 0;
        } else if (key == "--solver") {
            if (value != "cusolver" && value != "eigen" && value != "strict")
                throw std::invalid_argument("Solver must be cusolver,eigen or strict");
            a.solver_mode = value == "eigen" ? 1 : (value == "strict" ? 2 : 0);
        } else
            throw std::invalid_argument("Unknown option " + key);
    }
    if (a.input.empty() || a.output.empty())
        throw std::invalid_argument("--input and --output are required");
    if (a.mode != "joint" && a.mode != "infer" && a.mode != "evaluate")
        throw std::invalid_argument("Unknown mode " + a.mode);
    if (a.state != "initial" && a.state != "reference" && a.state != "fit")
        throw std::invalid_argument("State must be initial,reference or fit");
    if (a.maxiter < 0 || a.maxiter > 1000000)
        throw std::invalid_argument("maxiter out of range");
    return a;
}
void save_records(H5::H5File &file, const std::vector<AccIteration> &rows) {
    file.createGroup("/iterations");
    hsize_t n = rows.size();
    std::vector<double> values(n);
    std::vector<int32_t> integers(n);
#define RECORD_INT(field)                                                                          \
    for (size_t i = 0; i < n; ++i)                                                                 \
        integers[i] = rows[i].field;                                                               \
    write(file, "/iterations/" #field, integers.data(), H5::PredType::NATIVE_INT32, {n})
#define RECORD_DOUBLE(field)                                                                       \
    for (size_t i = 0; i < n; ++i)                                                                 \
        values[i] = rows[i].field;                                                                 \
    write(file, "/iterations/" #field, values.data(), H5::PredType::NATIVE_DOUBLE, {n})
    RECORD_INT(iteration);
    RECORD_INT(accepted);
    RECORD_INT(damping_attempts);
    RECORD_INT(line_search_evaluations);
    RECORD_DOUBLE(inverse_RMS_px);
    RECORD_DOUBLE(improvement_px2);
    RECORD_DOUBLE(damping);
    RECORD_DOUBLE(normalized_shared_gradient);
    RECORD_DOUBLE(normalized_state_gradient);
    RECORD_DOUBLE(maximum_state_step);
    RECORD_DOUBLE(linear_relative_residual);
#undef RECORD_INT
#undef RECORD_DOUBLE
}
int run(const Arguments &args) {
    if (fs::weakly_canonical(args.input) == fs::weakly_canonical(args.output))
        throw std::invalid_argument("Input and output paths must differ");
    H5::H5File input(args.input, H5F_ACC_RDONLY);
    if (!input.attrExists("schema_version"))
        throw std::runtime_error("Missing HDF5 schema_version");
    int32_t version = 0;
    input.openAttribute("schema_version").read(H5::PredType::NATIVE_INT32, &version);
    if (version != 1)
        throw std::runtime_error("Only schema_version=1 is supported");
    auto observation_shape = shape(input.openDataSet("/inputs/observed_normalized"));
    if (observation_shape.size() != 3 || observation_shape[1] != 3 || observation_shape[2] != 2 ||
        observation_shape[0] == 0)
        throw std::runtime_error("Expected observations [N,3,2]");
    hsize_t n = observation_shape[0];
    if (n > hsize_t(std::numeric_limits<int>::max() / 6))
        throw std::runtime_error("Too many frames");
    AccConfig cfg{};
    cfg.R = scalar(input, "/config/R");
    cfg.aref = scalar(input, "/config/aref");
    cfg.span = scalar(input, "/config/span");
    auto Q = read_double(input, "/config/reference_px", {3, 2});
    std::copy(Q.begin(), Q.end(), cfg.reference_px);
    auto y = read_double(input, "/inputs/observed_normalized", {n, 3, 2}),
         gaze = read_double(input, "/inputs/gaze_deg", {n, 2});
    auto low = read_double(input, "/inputs/lower_D", {n}),
         high = read_double(input, "/inputs/upper_D", {n}),
         weights = read_double(input, "/inputs/weights", {n}),
         demand = read_double(input, "/inputs/demand_D", {n});
    auto group = read_group(input, n);
    std::array<int64_t, 20> counts{};
    for (auto g : group) {
        if (g < 0 || g >= 20)
            throw std::runtime_error("Group outside[0,19]");
        ++counts[g];
    }
    for (int g = 0; g < 20; ++g)
        if (counts[g] == 0)
            throw std::runtime_error("Every one of20fixations must have frames");
    if (exists(input, "/groups/counts")) {
        auto dataset = input.openDataSet("/groups/counts");
        if (shape(dataset) != std::vector<hsize_t>{20})
            throw std::runtime_error("Expected group counts[20]");
        std::array<int64_t, 20> expected{};
        dataset.read(expected.data(), H5::PredType::NATIVE_INT64);
        if (expected != counts)
            throw std::runtime_error("Group count mismatch");
    }
    for (size_t f = 0; f < n; ++f) {
        if (!(std::isfinite(low[f]) && std::isfinite(high[f]) && low[f] >= 0. &&
              low[f] <= high[f] && high[f] <= 5.))
            throw std::runtime_error("Scientific CLI requires 0<=low<=high<=5 D");
        if (!std::isfinite(demand[f]))
            throw std::runtime_error("Nonfinite demand");
        double expected = 1. / (60. * counts[group[f]]);
        if (std::abs(weights[f] - expected) > 1e-12 * expected)
            throw std::runtime_error("Weights violate equal-fixation inverse objective");
    }
    if (exists(input, "/groups/mean_anchor_D")) {
        auto anchor = read_double(input, "/groups/mean_anchor_D", {20});
        for (size_t f = 0; f < n; ++f) {
            double L = std::max(0., anchor[group[f]] - .25),
                   U = std::min(5., anchor[group[f]] + .25);
            if (std::abs(low[f] - L) > 1e-12 || std::abs(high[f] - U) > 1e-12)
                throw std::runtime_error("Bounds differ from mean-triangle anchor+/-0.25D");
        }
    }
    auto p = read_double(input, "/" + args.state + "/params", {16}),
         A = read_double(input, "/" + args.state + "/accommodation_D", {n});
    int32_t transform_order = 0;
    if (exists(input, "/config/transform_order")) {
        auto dataset = input.openDataSet("/config/transform_order");
        if (!shape(dataset).empty() || dataset.getTypeClass() != H5T_INTEGER ||
            dataset.getDataType().getSize() != 4)
            throw std::runtime_error("Expected scalar int32 /config/transform_order");
        dataset.read(&transform_order, H5::PredType::NATIVE_INT32);
    }
    if (!args.order.empty())
        transform_order = args.order == "keystone-barrel" ? 1 : 0;
    AccModelOptions model_options{};
    native_check(acc_model_options_default(&model_options));
    model_options.transform_order = transform_order;
    Handle handle;
    native_check(acc_create_ex(&cfg, &model_options, int64_t(n), y.data(), gaze.data(), low.data(),
                               high.data(), weights.data(), group.data(), args.device,
                               &handle.value));
    native_check(acc_set_launch_config(handle.value, args.threads, args.derivative_mode));
    native_check(acc_set_solver_mode(handle.value, args.solver_mode));
    AccSolveSummary summary{};
    std::vector<AccIteration> records(args.maxiter);
    std::vector<double> inference_cost, sensitivity, gap;
    std::vector<uint8_t> inference_valid;
    std::vector<int32_t> bound, minima;
    auto compute_start = std::chrono::steady_clock::now();
    if (args.mode == "joint") {
        AccSolveSettings options = args.settings;
        options.max_iterations = args.maxiter;
        options.initial_damping = args.damping;
        options.shared_parameter_mask = args.shared_mask;
        options.optimize_accommodation = args.freeze_accommodation ? 0 : 1;
        native_check(acc_solve_ex(handle.value, p.data(), A.data(), &options, records.data(),
                                  int(records.size()), &summary));
        records.resize(summary.iterations);
    } else if (args.mode == "infer") {
        records.clear();
        inference_cost.resize(n);
        sensitivity.resize(n);
        gap.resize(n);
        inference_valid.resize(n);
        bound.resize(n);
        minima.resize(n);
        AccInferOptions options{args.grid, args.refine, args.tolerance, args.ambiguity};
        native_check(acc_infer(handle.value, p.data(), &options, A.data(), inference_cost.data(),
                               inference_valid.data(), bound.data(), sensitivity.data(), gap.data(),
                               minima.data()));
    } else
        records.clear();
    double compute_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - compute_start).count();
    std::vector<double> residual(n * 6), corrected(n * 6), predicted(n * 6),
        J(args.derivatives ? n * 6 * 17 : 0);
    std::vector<uint8_t> valid(n);
    double loss = 0.;
    native_check(acc_evaluate(handle.value, p.data(), A.data(), args.derivatives ? 1 : 0,
                              args.device >= 0 ? 1 : 0, residual.data(), corrected.data(),
                              predicted.data(), args.derivatives ? J.data() : nullptr, valid.data(),
                              &loss));
    if (args.mode == "evaluate")
        compute_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - compute_start).count();
    if (args.mode != "joint") {
        summary.initial_inverse_RMS_px = summary.final_inverse_RMS_px = std::sqrt(loss);
        summary.seconds = compute_seconds;
    }
    if (!std::isfinite(loss) && args.mode != "infer")
        throw std::runtime_error("Final state includes invalid inverse frames");
    std::array<double, 20> offsets{};
    for (size_t f = 0; f < n; ++f)
        offsets[group[f]] += A[f] - demand[f];
    for (int g = 0; g < 20; ++g)
        offsets[g] /= counts[g];
    std::vector<double> variation(n);
    std::array<double, 20> mean{};
    for (size_t f = 0; f < n; ++f) {
        variation[f] = A[f] - demand[f] - offsets[group[f]];
        mean[group[f]] += variation[f];
        if (A[f] < low[f] - 1e-10 || A[f] > high[f] + 1e-10)
            throw std::runtime_error("Output accommodation outside bounds");
    }
    for (int g = 0; g < 20; ++g)
        if (std::abs(mean[g] / counts[g]) > 1e-10)
            throw std::runtime_error("Offset decomposition failed");
    fs::path output_path(args.output);
    if (!output_path.parent_path().empty())
        fs::create_directories(output_path.parent_path());
    Temporary temporary{output_path.string() + ".tmp." + std::to_string(getpid())};
    {
        H5::H5File output(temporary.path, H5F_ACC_TRUNC);
        attr(output, "schema_version", int32_t(1));
        for (const char *path : {"/config", "/inputs", "/groups", "/provenance"})
            if (exists(input, path))
                if (H5Ocopy(input.getId(), path, output.getId(), path, H5P_DEFAULT, H5P_DEFAULT) <
                    0)
                    throw std::runtime_error(std::string("Failed to copy ") + path);
        if (exists(output, "/config/transform_order"))
            H5Ldelete(output.getId(), "/config/transform_order", H5P_DEFAULT);
        write(output, "/config/transform_order", &transform_order, H5::PredType::NATIVE_INT32, {});
        output.createGroup("/fit");
        write(output, "/fit/params", p.data(), H5::PredType::NATIVE_DOUBLE, {16});
        write(output, "/fit/accommodation_D", A.data(), H5::PredType::NATIVE_DOUBLE, {n});
        write(output, "/fit/corrected_normalized", corrected.data(), H5::PredType::NATIVE_DOUBLE,
              {n, 3, 2});
        write(output, "/fit/predicted_normalized", predicted.data(), H5::PredType::NATIVE_DOUBLE,
              {n, 3, 2});
        write(output, "/fit/residual_px", residual.data(), H5::PredType::NATIVE_DOUBLE, {n, 3, 2});
        write(output, "/fit/valid", valid.data(), H5::PredType::NATIVE_UINT8, {n});
        write(output, "/fit/fixation_offsets_D", offsets.data(), H5::PredType::NATIVE_DOUBLE, {20});
        write(output, "/fit/accommodation_variation_D", variation.data(),
              H5::PredType::NATIVE_DOUBLE, {n});
        if (args.derivatives)
            write(output, "/fit/jacobian_px", J.data(), H5::PredType::NATIVE_DOUBLE, {n, 3, 2, 17});
        if (args.mode == "infer") {
            output.createGroup("/inference");
            write(output, "/inference/cost_px2", inference_cost.data(), H5::PredType::NATIVE_DOUBLE,
                  {n});
            write(output, "/inference/valid", inference_valid.data(), H5::PredType::NATIVE_UINT8,
                  {n});
            write(output, "/inference/bound", bound.data(), H5::PredType::NATIVE_INT32, {n});
            write(output, "/inference/sensitivity_px_per_D", sensitivity.data(),
                  H5::PredType::NATIVE_DOUBLE, {n});
            write(output, "/inference/ambiguity_gap_px2", gap.data(), H5::PredType::NATIVE_DOUBLE,
                  {n});
            write(output, "/inference/minima_count", minima.data(), H5::PredType::NATIVE_INT32,
                  {n});
        }
        save_records(output, records);
        auto group_summary = output.createGroup("/summary");
        attr(group_summary, "mode", args.mode);
        attr(group_summary, "state", args.state);
        attr(group_summary, "transform_order", transform_order);
        attr(group_summary, "shared_parameter_mask", int32_t(args.shared_mask));
        attr(group_summary, "optimize_accommodation", int32_t(args.freeze_accommodation ? 0 : 1));
        attr(group_summary, "metadata_json", std::string(acc_metadata(handle.value)));
#define SUMMARY_INT(field) attr(group_summary, #field, summary.field)
#define SUMMARY_DOUBLE(field) attr(group_summary, #field, summary.field)
        SUMMARY_INT(iterations);
        SUMMARY_INT(converged);
        SUMMARY_INT(frame_state_step_stable);
        SUMMARY_INT(termination_code);
        SUMMARY_DOUBLE(initial_inverse_RMS_px);
        SUMMARY_DOUBLE(final_inverse_RMS_px);
        SUMMARY_DOUBLE(seconds);
        SUMMARY_DOUBLE(final_normalized_shared_gradient);
        SUMMARY_DOUBLE(final_normalized_state_gradient);
#undef SUMMARY_INT
#undef SUMMARY_DOUBLE
        attr(group_summary, "termination",
             args.mode == "joint" ? termination(summary.termination_code) : "Evaluation only");
        attr(group_summary, "compute_seconds", compute_seconds);
        attr(group_summary, "replay_inverse_RMS_px", std::sqrt(loss));
        attr(group_summary, "gaze_fitted", int32_t(0));
        attr(group_summary, "fixed_optical_parameters", int32_t(args.mode == "infer"));
        AccProfile profile{};
        native_check(acc_get_profile(handle.value, &profile));
        auto profile_group = output.createGroup("/profile");
#define PROFILE(field) attr(profile_group, #field, double(profile.field))
        PROFILE(residual_kernel_ms);
        PROFILE(derivative_kernel_ms);
        PROFILE(normal_assembly_ms);
        PROFILE(schur_assembly_ms);
        PROFILE(backsubstitution_ms);
        PROFILE(inference_ms);
        PROFILE(evaluation_calls);
        PROFILE(derivative_calls);
        PROFILE(allocation_count);
#undef PROFILE
        output.flush(H5F_SCOPE_GLOBAL);
    }
    fs::rename(temporary.path, output_path);
    temporary.committed = true;
    std::cout << std::setprecision(17) << "{\"mode\":\"" << args.mode << "\",\"frames\":" << n
              << ",\"inverse_RMS_px\":"
              << (std::isfinite(loss) ? std::to_string(std::sqrt(loss)) : "null")
              << ",\"iterations\":" << summary.iterations << ",\"converged\":" << summary.converged
              << ",\"frame_state_step_stable\":" << summary.frame_state_step_stable
              << ",\"compute_seconds\":" << compute_seconds << ",\"output\":\"" << args.output
              << "\"}\n";
    return 0;
}
} // namespace
int main(int argc, char **argv) {
    H5::Exception::dontPrint();
    try {
        return run(parse(argc, argv));
    } catch (const H5::Exception &error) {
        std::cerr << "HDF5: " << error.getDetailMsg() << '\n';
        return 1;
    } catch (const std::exception &error) {
        std::cerr << "acc_inverse: " << error.what() << '\n';
        return 1;
    }
}
