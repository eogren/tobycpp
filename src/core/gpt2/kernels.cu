#include "kernels.h"
#include "toby/cuda_utils/cuda_alloc.hpp"
#include "toby/cuda_utils/cuda_exception.hpp"
#include "toby/safetensors/memory_transfer.hpp"
#include "toby/safetensors/tensor.hpp"
#include "toby/safetensors/tensor_types.hpp"

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <format>
#include <span>
#include <stdexcept>

using toby::cuda::CudaAlloc;
using toby::tensors::DeviceType;
using toby::tensors::Tensor;

// TODO: the cpu stuff should probably be in a .cpp file so it works on non CUDA builds technically.

namespace {
/*
 * Check two tensors for itemwise equality. Assumes they are already the same size.
 */
template <typename T>
__global__ void tensors_equal_tmpl(int* out, const T* t1, const T* t2, std::size_t size) {
    // do the dumbest possible thing for now. each thread = 1 cell of the array, and always
    // write results out if false
    size_t idx = (blockDim.x * blockIdx.x) + threadIdx.x;
    bool is_equal =
        (idx >= size)
            ? true
            : (t1[idx] == t2[idx]); // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)

    if (!is_equal) {
        *out = 0;
    }
}

template <typename T> bool tensors_equal_cpu_compare(const T* t1, const T* t2, std::size_t size) {
    // TOOD: could try to make this simd, but compiler probably does it
    for (std::size_t i = 0; i < size; i++) {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        if (t1[i] != t2[i]) {
            return false;
        }
    }

    return true;
}

/**
 * On CPU kernel for equality. Assumes the checks about size, shape, dtype, etc are a
 * already done.
 */
bool tensors_equal_cpu(const Tensor& t1, const Tensor& t2) {
    switch (t1.dtype()) {
    case toby::tensors::DataType::U16: {
        const auto* p1 = static_cast<const std::uint16_t*>(t1.data());
        const auto* p2 = static_cast<const std::uint16_t*>(t2.data());

        return tensors_equal_cpu_compare(p1, p2, t1.shape().numel());
    }
    default:
        throw std::invalid_argument{"unsupported type"};
    }
}

bool tensors_equal_gpu(const Tensor& t1, const Tensor& t2) {
    switch (t1.dtype()) {
    case toby::tensors::DataType::U16: {
        const auto* p1 = static_cast<const std::uint16_t*>(t1.data());
        const auto* p2 = static_cast<const std::uint16_t*>(t2.data());

        CudaAlloc alloc = CudaAlloc::anonymous(sizeof(int));
        int* out = static_cast<int*>(alloc.addr());
        int ret{1};

        toby::tensors::copy_bytes(std::as_writable_bytes(std::span{out, 1}), DeviceType::GPU,
                                  std::as_bytes(std::span{&ret, 1}), DeviceType::CPU);

        // todo - maybe could be smarter about this later
        auto num_blocks = (t1.shape().numel() / 256) + 1;

        /// launch kernel (p1, p2, t1.shape().numel());
        tensors_equal_tmpl<std::uint16_t><<<num_blocks, 256>>>(out, p1, p2, t1.shape().numel());
        toby::cuda::throw_if_error("kernel_submit", cudaGetLastError());

        // 3. Force the host CPU to wait for the GPU to finish execution
        toby::cuda::throw_if_error("synchronize", cudaDeviceSynchronize());

        toby::tensors::copy_bytes(std::as_writable_bytes(std::span{&ret, 1}), DeviceType::CPU,
                                  std::as_bytes(std::span{out, 1}), DeviceType::GPU);
        return (ret != 0);
    }
    default:
        throw std::invalid_argument{"unsupported type"};
    }
}
} // namespace

namespace toby::gpt2 {
bool tensors_equal(const tensors::Tensor& t1, const tensors::Tensor& t2) {
    if (t1.device() != t2.device()) {
        throw std::invalid_argument{
            std::format("Can't compare t1 on {} and t2 on {}", t1.device(), t2.device())};
    }

    if (t1.dtype() != t2.dtype()) {
        return false;
    }

    if (t1.shape() != t2.shape()) {
        return false;
    }

    if (t1.device() == tensors::DeviceType::CPU) {
        return tensors_equal_cpu(t1, t2);
    }
    if (t1.device() == tensors::DeviceType::GPU) {
        return tensors_equal_gpu(t1, t2);
    }

    throw std::invalid_argument{"Unknown device type"};
}
} // namespace toby::gpt2
