#ifndef SAFETENSORS_CPU_ARENA_HPP
#define SAFETENSORS_CPU_ARENA_HPP

#include "toby/safetensors/arena.hpp"
#include "toby/safetensors/file.hpp"

#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

namespace toby::tensors::detail {
class ScopedMapping {
public:
    ScopedMapping(ScopedMapping&& other) noexcept
        : addr_(std::exchange(other.addr_, nullptr)), len_(std::exchange(other.len_, 0)) {}

    ScopedMapping& operator=(ScopedMapping&& other) noexcept;
    ~ScopedMapping(); // munmap(addr_, len_) if addr_

    ScopedMapping(const ScopedMapping&) = delete;
    ScopedMapping& operator=(const ScopedMapping&) = delete;

    [[nodiscard]] void* addr() const { return addr_; }

    [[nodiscard]] std::size_t size() const { return len_; }

    static ScopedMapping anonymous(size_t len);
    static ScopedMapping from_fd(int fd, FileMode mode);

private:
    ScopedMapping(int fd, int prot, std::size_t len);

    void* addr_ = nullptr;
    std::size_t len_ = 0;
};

class CpuArena : public toby::tensors::Arena {
public:
    explicit CpuArena(std::size_t len) : CpuArena(ScopedMapping::anonymous(len)) {}

    explicit CpuArena(ScopedMapping&& mapping)
        : toby::tensors::Arena("cpu_arena", mapping.addr(), mapping.size()),
          mapping_(std::move(mapping)) {}

    void bulk_memcpy(const std::vector<MemcpyInfo>& copies) override;

private:
    ScopedMapping mapping_;
};
} // namespace toby::tensors::detail
#endif
