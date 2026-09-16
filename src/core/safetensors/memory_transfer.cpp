#include "toby/safetensors/memory_transfer.hpp"

#include "toby/safetensors/tensor_types.hpp"

#include <cstddef>
#include <cstring>
#include <span>
#include <stdexcept>

#if TOBY_HAVE_CUDA
#include "toby/cuda_utils/cuda_exception.hpp"

#include <cuda_runtime_api.h>
#include <driver_types.h>
#endif

namespace toby::tensors {
void copy_bytes(std::span<std::byte> dst, DeviceType dst_device, std::span<const std::byte> src,
                DeviceType src_device) {
    if (dst.size_bytes() < src.size_bytes()) {
        throw std::invalid_argument{"copy destination is smaller than source"};
    }
    const auto valid_device = [](DeviceType device) {
        return device == DeviceType::CPU || device == DeviceType::GPU;
    };
    if (!valid_device(dst_device) || !valid_device(src_device)) {
        throw std::invalid_argument{"unknown copy device"};
    }
    if (dst_device == DeviceType::CPU && src_device == DeviceType::CPU) {
        if (!src.empty()) {
            std::memcpy(dst.data(), src.data(), src.size_bytes());
        }
        return;
    }
#if TOBY_HAVE_CUDA
    if (src.empty()) {
        return;
    }
    auto kind = cudaMemcpyDeviceToDevice;
    if (dst_device == DeviceType::CPU) {
        kind = cudaMemcpyDeviceToHost;
    } else if (src_device == DeviceType::CPU) {
        kind = cudaMemcpyHostToDevice;
    }
    cuda::throw_if_error("cudaMemcpy", cudaMemcpy(dst.data(), src.data(), src.size_bytes(), kind));
    // cudaMemcpy alone need not wait for H2D or D2D completion on the host.
    cuda::throw_if_error("cudaStreamSynchronize", cudaStreamSynchronize(nullptr));
#else
    throw std::invalid_argument{"GPU not supported in this build"};
#endif
}
} // namespace toby::tensors
