#include "toby/cuda_utils/cuda_alloc.hpp"
#include "toby/cuda_utils/cuda_exception.hpp"
#include "toby/safetensors/memory_transfer.hpp"
#include "toby/safetensors/tensor.hpp"
#include "toby/safetensors/tensor_types.hpp"

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <device_atomic_functions.h>
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
    __shared__ bool is_equal[8]; // NOLINT(cppcoreguidelines-avoid-c-arrays, modernize-avoid-c-arrays)

    // todo: Early exit somewhere?

    // warp-level: collect whether equality is false and use ballot_sync to coordinate
    // among all warp threads

    size_t idx = (blockDim.x * blockIdx.x) + threadIdx.x;
    bool local_is_equal =
        (idx >= size)
            ? true
            : (t1[idx] == t2[idx]); // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    unsigned int equality_mask = __ballot_sync(0xFFFFFFFF, local_is_equal);

    if (threadIdx.x % 32 == 0) {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
        is_equal[threadIdx.x / 32] =
            (equality_mask ==
             0xFFFFFFFF);
    }

    __syncthreads();

    // block level: write out if failed

    if (threadIdx.x == 0 && (!is_equal[0] || !is_equal[1] || !is_equal[2] || !is_equal[3] ||
                             !is_equal[4] || !is_equal[5] || !is_equal[6] || !is_equal[7])) {
        atomicExch(out, 0);
    }
}
} // namespace

namespace toby::gpt2::detail {
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
} // namespace toby::gpt2::detail
