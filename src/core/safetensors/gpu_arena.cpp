#include "gpu_arena.hpp"

#include "cuda_exception.hpp"
#include "toby/safetensors/arena.hpp"

#include <cstddef>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <vector>

using toby::cuda::throw_if_error;

namespace toby::tensors::detail {
CudaAlloc CudaAlloc::anonymous(std::size_t len) {
    void* ptr = nullptr;
    auto err = cudaMalloc(&ptr, len);
    throw_if_error("cudaMalloc", err);

    return CudaAlloc{ptr, len};
}

CudaAlloc::~CudaAlloc() {
    if (addr_ != nullptr) {
        auto err = cudaFree(addr_);
        cuda::warn_if_error("cudaFree", err);
    }
}

void GpuArena::bulk_memcpy(const std::vector<MemcpyInfo>& copies) {
    for (auto const& memcpy : copies) {
        // NOLINTNEXTLINE
        auto new_dst = static_cast<std::byte*>(mapping_.addr()) + memcpy.new_offset;
        const auto* new_src = static_cast<const std::byte*>(memcpy.src) + memcpy.src_offset;

        auto err = cudaMemcpy(new_dst, new_src, memcpy.size, ::cudaMemcpyDefault);
        throw_if_error("cudaMemcpy", err);
    }
}

} // namespace toby::tensors::detail
