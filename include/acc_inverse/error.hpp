#pragma once

#include "acc_inverse/api.h"
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

namespace acc {

// backend_code is the original CUDA/cuBLAS/cuSOLVER status value. The public
// header deliberately does not require CUDA headers or link client code to it.
class BackendError : public std::runtime_error {
  public:
    BackendError(AccStatus category, std::int32_t code, std::string message)
        : std::runtime_error(std::move(message)), status(category), backend_code(code) {}
    BackendError(AccStatus category, std::string message, std::int32_t code = 0)
        : BackendError(category, code, std::move(message)) {}
    const AccStatus status;
    const std::int32_t backend_code;
};

using NativeError = BackendError;

class CudaError : public BackendError {
  public:
    CudaError(std::int32_t code, std::string message)
        : BackendError(ACC_ERROR_CUDA, code, std::move(message)) {}
};

} // namespace acc
