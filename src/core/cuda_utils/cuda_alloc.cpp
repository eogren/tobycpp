#include "toby/cuda_utils/cuda_alloc.hpp"

#include "toby/cuda_utils/cuda_exception.hpp"

#include <cstddef>
#include <cstdlib>
#include <cuda_runtime_api.h>

namespace toby::cuda {
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

} // namespace toby::cuda
