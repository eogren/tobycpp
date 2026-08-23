#ifndef SAFETENSORS_ARENA_HPP
#define SAFETENSORS_ARENA_HPP

#include "toby/safetensors/tensor.hpp"

#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace toby::tensors {
struct MemcpyInfo {
    const void* src;
    std::size_t src_offset;
    std::size_t new_offset;
    std::size_t size;
};

class Arena {
public:
    static std::unique_ptr<Arena> alloc_anonymous(std::size_t size, DeviceType device_type);

    virtual ~Arena();

    Arena(const Arena&) = delete;
    Arena& operator=(const Arena&) = delete;

    [[nodiscard]] std::string_view name() const { return name_; }

    [[nodiscard]] std::span<const std::byte> byte_span() const;

    // using vector& instea dof something more generic to avoid
    // leaking cuda headers etc everywhere
    virtual void bulk_memcpy(const std::vector<MemcpyInfo>& copies) = 0;

protected:
    Arena(std::string_view name, void* base, std::size_t size)
        : name_(name), base_(base), size_(size) {}

    Arena(Arena&&) noexcept = default;
    Arena& operator=(Arena&&) noexcept = default;

    void* base() { return base_; }

private:
    std::string name_;
    void* base_;
    std::size_t size_;
};
} // namespace toby::tensors

#endif
