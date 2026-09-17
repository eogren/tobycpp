#include "toby/cuda_utils/cuda_alloc.hpp"
#include "toby/cuda_utils/cuda_exception.hpp"
#include "toby/gpt2/kernels.h"
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
} // namespace

namespace toby::gpt2::detail {
bool tensors_equal_gpu(const Tensor& t1, const Tensor& t2) {

    CudaAlloc alloc = CudaAlloc::anonymous(sizeof(int));
    int* out = static_cast<int*>(alloc.addr());
    int ret{1};

    toby::tensors::copy_bytes(std::as_writable_bytes(std::span{out, 1}), DeviceType::GPU,
                              std::as_bytes(std::span{&ret, 1}), DeviceType::CPU);

    // todo - maybe could be smarter about this later
    auto num_blocks = (t1.shape().numel() / 256) + 1;

    switch (t1.dtype()) {
    case toby::tensors::DataType::U16: {
        const auto* p1 = static_cast<const std::uint16_t*>(t1.data());
        const auto* p2 = static_cast<const std::uint16_t*>(t2.data());

        tensors_equal_tmpl<std::uint16_t><<<num_blocks, 256>>>(out, p1, p2, t1.shape().numel());
        break;
    }
    case toby::tensors::DataType::F32: {
        const auto* p1 = static_cast<const float*>(t1.data());
        const auto* p2 = static_cast<const float*>(t2.data());

        tensors_equal_tmpl<float><<<num_blocks, 256>>>(out, p1, p2, t1.shape().numel());
        break;
    }
    default:
        throw std::invalid_argument{"unsupported type"};
    }

    // 3. Force the host CPU to wait for the GPU to finish execution
    toby::cuda::throw_if_error("kernel_submit", cudaGetLastError());

    toby::cuda::throw_if_error("synchronize", cudaDeviceSynchronize());

    toby::tensors::copy_bytes(std::as_writable_bytes(std::span{&ret, 1}), DeviceType::CPU,
                              std::as_bytes(std::span{out, 1}), DeviceType::GPU);
    return (ret != 0);
}
} // namespace toby::gpt2::detail
