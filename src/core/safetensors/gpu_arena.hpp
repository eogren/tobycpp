#ifndef SAFETENSORS_GPU_ARENA_HPP
#define SAFETENSORS_GPU_ARENA_HPP

#include "toby/safetensors/arena.hpp"
#include "toby/safetensors/tensor_types.hpp"

#include <cstddef>
#include <optional>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace toby::tensors::detail {
class CudaAlloc {
public:
    CudaAlloc(CudaAlloc&& other) noexcept
        : addr_(std::exchange(other.addr_, nullptr)), len_(std::exchange(other.len_, 0)) {}

    CudaAlloc& operator=(CudaAlloc&& other) noexcept;
    ~CudaAlloc();

    CudaAlloc(const CudaAlloc&) = delete;
    CudaAlloc& operator=(const CudaAlloc&) = delete;

    [[nodiscard]] void* addr() const { return addr_; }

    [[nodiscard]] std::size_t size() const { return len_; }

    static CudaAlloc anonymous(size_t len);

private:
    CudaAlloc(void* addr, std::size_t len) noexcept : addr_(addr), len_(len) {}

    void* addr_ = nullptr;
    std::size_t len_ = 0;
};

class GpuArena : public toby::tensors::Arena {
public:
    explicit GpuArena(std::optional<std::string_view> name, std::size_t len)
        : GpuArena(name, CudaAlloc::anonymous(len)) {}

    explicit GpuArena(std::optional<std::string_view> name, CudaAlloc&& mapping)
        : toby::tensors::Arena(name.value_or("gpu_arena"), DeviceType::GPU, mapping.addr(),
                               mapping.size()),
          mapping_(std::move(mapping)) {}

private:
    CudaAlloc mapping_;
};
} // namespace toby::tensors::detail
#endif
