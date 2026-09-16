#ifndef SAFETENSORS_GPU_ARENA_HPP
#define SAFETENSORS_GPU_ARENA_HPP

#include "toby/cuda_utils/cuda_alloc.hpp"
#include "toby/safetensors/arena.hpp"
#include "toby/safetensors/tensor_types.hpp"

#include <cstddef>
#include <optional>
#include <string_view>
#include <utility>

namespace toby::tensors::detail {
class GpuArena : public toby::tensors::Arena {
public:
    explicit GpuArena(std::optional<std::string_view> name, std::size_t len)
        : GpuArena(name, cuda::CudaAlloc::anonymous(len)) {}

    explicit GpuArena(std::optional<std::string_view> name, cuda::CudaAlloc&& mapping)
        : toby::tensors::Arena(name.value_or("gpu_arena"), DeviceType::GPU, mapping.addr(),
                               mapping.size()),
          mapping_(std::move(mapping)) {}

private:
    cuda::CudaAlloc mapping_;
};
} // namespace toby::tensors::detail
#endif
