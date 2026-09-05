#ifndef SAFETENSORS_CUDA_EXCEPTION_HPP
#define SAFETENSORS_CUDA_EXCEPTION_HPP

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <format>
#include <spdlog/logger.h>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <string_view>

namespace toby::cuda {
class CudaException : public std::runtime_error {
public:
    CudaException(std::string_view op, cudaError_t error)
        : std::runtime_error(std::format("{}: cuda error: {} ({})", op, cudaGetErrorString(error),
                                         cudaGetErrorName(error))) {}
};

inline void throw_if_error(std::string_view op, cudaError_t error) {
    if (error != cudaSuccess) {
        throw CudaException(op, error);
    }
}

inline void warn_if_error(std::string_view op, cudaError_t error) {
    if (error == cudaSuccess) {
        return;
    }

    spdlog::warn("{}: cuda error: {} ({})", op, cudaGetErrorString(error), cudaGetErrorName(error));
}
} // namespace toby::cuda

#endif
