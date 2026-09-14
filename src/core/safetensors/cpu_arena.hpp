#ifndef SAFETENSORS_CPU_ARENA_HPP
#define SAFETENSORS_CPU_ARENA_HPP

#include "toby/safetensors/arena.hpp"
#include "toby/safetensors/file.hpp"
#include "toby/safetensors/tensor_types.hpp"

#include <cstddef>
#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace toby::tensors::detail {
class ScopedMapping {
public:
    ScopedMapping(ScopedMapping&& other) noexcept
        : addr_(std::exchange(other.addr_, nullptr)), size_(std::exchange(other.size_, 0)),
          used_(std::exchange(other.used_, 0)) {}

    ScopedMapping& operator=(ScopedMapping&& other) noexcept;
    ~ScopedMapping(); // munmap(addr_, len_) if addr_

    ScopedMapping(const ScopedMapping&) = delete;
    ScopedMapping& operator=(const ScopedMapping&) = delete;

    [[nodiscard]] void* addr() const { return addr_; }

    /**
     * Various size/capacity methods. This abstraction isn't quite right,
     * but is intended to distinguish between an mmap regoin that comes in
     * with valid data vs a totally anonymous map.
     *
     * Once the Mapping is passed to an arena it is probably doing its own
     * size tracking so these methods are not necessarily accurate. But the
     * mapping should be consumed by the Arena so this becomes mostly an internal
     * implementation detail.
     */
    [[nodiscard]] std::size_t size() const { return used_; }

    [[nodiscard]] std::size_t capacity() const { return size_; }

    void resize(std::size_t new_size) {
        if (new_size > size_) {
            throw std::invalid_argument{"Can't resize to larger than initial size"};
        }

        used_ = new_size;
    }

    static ScopedMapping anonymous(size_t len);
    static ScopedMapping from_fd(int fd, FileMode mode);

private:
    ScopedMapping(int fd, int prot, std::size_t len);

    void* addr_ = nullptr;
    std::size_t size_ = 0;
    std::size_t used_ = 0;
};

class CpuArena : public toby::tensors::Arena {
public:
    explicit CpuArena(std::optional<std::string_view> name, std::size_t len)
        : CpuArena(name, ScopedMapping::anonymous(len)) {}

    explicit CpuArena(std::optional<std::string_view> name, ScopedMapping&& mapping)
        : toby::tensors::Arena(name.value_or("cpu_anon"), DeviceType::CPU, mapping.addr(),
                               mapping.capacity()),
          mapping_(std::move(mapping)) {
        if (mapping_.size() != 0) {
            (void)alloc(mapping_.size());
        }
    }

private:
    ScopedMapping mapping_;
};
} // namespace toby::tensors::detail
#endif
